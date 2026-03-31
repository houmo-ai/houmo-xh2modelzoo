import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
import vllm
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig

from vllm.distributed import (get_pp_group, 
                              get_dp_group,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper, PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from vllm.attention.layer import Attention
from .moe_layer import GroupedMoE
from .layernorm import FP32RMSNorm
from vllm.logger import init_logger
import os
logger = init_logger(__name__)

class LayerNorm(nn.LayerNorm):
    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        # orig_dtype = x.dtype
        # x = x.to(torch.float32)
        # if residual is not None:
        #     x = x + residual.to(torch.float32)
        #     residual = x.to(orig_dtype)
        #     residual = x
        if residual is not None:
            x = x + residual
            residual = x
        x = super().forward(x)
        if residual is None:
            return x
        else:
            return x, residual

class ExEmbedding(nn.Module):
    def __init__(self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: Optional[torch.dtype] = None,
        org_num_embeddings: Optional[int] = None,
        padding_size: int = 64,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = ""
    ):
        super().__init__()
        self.word_embeddings = VocabParallelEmbedding(
                num_embeddings,
                embedding_dim,
                params_dtype,
                org_num_embeddings,
                padding_size,
                quant_config,
                f"{prefix}.word_embeddings",
            )

    def forward(self, input_):
        return self.word_embeddings(input_)

class IPTMLP(nn.Module):
    def __init__(
        self,
        config,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
        gate_gelu = False,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.clamp_input_value = 0
        if hasattr(config, "clamp_training"):
            clamp_training_cfg = config.clamp_training
            self.clamp_input_value = clamp_training_cfg["clamp_input_value"]

        self.bias = bias
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_gelu = gate_gelu
        self.quant_config = quant_config
        self.prefix = prefix

        self.fc1 = self.build_fc1(skip_bias_add=False)
        self.activation_fn = self.build_activation()
        self.fc2 = self.build_fc2(reduce_results)

    def build_fc1(
        self,
        skip_bias_add: Optional[bool]=False,
    ):
        if self.gate_gelu:
            return MergedColumnParallelLinear(
                self.hidden_size, [self.intermediate_size] * 2,
                bias=self.bias,
                skip_bias_add=skip_bias_add,
                quant_config=self.quant_config,
                prefix=f"{self.prefix}.fc1",)
        else:
            return ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=self.bias,
                skip_bias_add=skip_bias_add,
                quant_config=self.quant_config,
                prefix=f"{self.prefix}.fc1"
            )

    def build_fc2(
        self,
        reduce_results: Optional[bool]=True,
    ):
        return RowParallelLinear(self.intermediate_size,
                self.hidden_size,
                bias=self.bias,
                quant_config=self.quant_config,
                reduce_results=reduce_results,
                prefix=f"{self.prefix}.fc2",)

    def build_activation(
        self,
    ):
        if self.gate_gelu:
            return GeluAndMul()
        return get_act_fn("gelu", self.quant_config, self.intermediate_size)

    def forward(self, inputs):
        if self.clamp_input_value > 0:
            inputs = torch.clamp_(inputs, -self.clamp_input_value, self.clamp_input_value)
        intermediate_parallel, _ = self.fc1(inputs)
        intermediate_parallel = self.activation_fn(intermediate_parallel)
        if self.clamp_input_value > 0:
            intermediate_parallel = torch.clamp_(intermediate_parallel, -self.clamp_input_value, self.clamp_input_value)
        output, _ = self.fc2(intermediate_parallel)
        return output


class IPTRouter(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """Initialize grouped-moe Router."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_groups = config.num_groups
        self.num_routed_experts_per_group = config.num_routed_experts - config.num_shared_experts

        self.num_experts = self.num_routed_experts_per_group
        
        self.gating = ReplicatedLinear(self.hidden_size,
                                        self.num_groups * self.num_experts,
                                        bias=False,
                                        params_dtype=torch.float32,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.gating")

        self.enable_noaux_tc = True
        if self.enable_noaux_tc:
            self.e_score_correction_bias = nn.Parameter(torch.empty(self.num_groups * self.num_routed_experts_per_group, dtype=torch.float32))

    def forward(
        self, 
        hidden_states: torch.Tensor
    ):  
        hidden_states = hidden_states.float()
        logits, _ = self.gating(hidden_states)
        logits = logits.view(-1, self.num_groups, self.num_experts)
        return logits


class IPTMoE(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.prefix = prefix

        self.clamp_input_value = 0
        if hasattr(config, "clamp_training"):
            clamp_training_cfg = config.clamp_training
            self.clamp_input_value = clamp_training_cfg["clamp_input_value"]

        self.tp_size = get_tensor_model_parallel_world_size()

        self.router = IPTRouter(
            config=config,
            quant_config=None,
            prefix=f"{prefix}.router"
        )

        self.num_experts = (config.num_routed_experts - config.num_shared_experts) * config.num_groups
            
        num_pad_experts = 0
        if hasattr(config, "num_pad_experts"):
            num_pad_experts = config.num_pad_experts
        self.routed_experts = GroupedMoE(
            num_experts=self.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=self.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            clamp_input_value=self.clamp_input_value,
            routed_scaling_factor=config.routed_scaling_factor,
            calc_denominator_cross_groups=config.calc_denominator_cross_groups,
            reduce_results=False,
            quant_config=quant_config,
            num_shared_experts=config.num_shared_experts,
            num_pad_experts=num_pad_experts,
            prefix=f"{prefix}.routed_experts",
            e_score_correction_bias=self.router.e_score_correction_bias,
            activation="gelu"
        )

        self.shared_experts = None
        if config.num_shared_experts > 0:
            num_shared_experts = config.num_shared_experts * config.num_groups
            self.shared_experts = IPTMLP(
                config=config,
                hidden_size=self.hidden_size,
                intermediate_size=config.moe_intermediate_size * num_shared_experts,
                bias=config.bias,
                gate_gelu=config.gate_gelu,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts"
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        layer_idx =  self.prefix.split(sep='.')[1]
        local_dp_rank = os.getenv('VLLM_DP_RANK_LOCAL')
        if self.shared_experts:
            shared_output = self.shared_experts(hidden_states)
        
        logits = self.router(hidden_states)
        output = self.routed_experts(
            hidden_states=hidden_states,
            router_logits=logits) 
        if self.shared_experts:
            output = output + shared_output
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output.view(num_tokens, hidden_dim)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class IPTMultiheadLatentAttention(nn.Module):
    def __init__(
        self, 
        config,
        hidden_size: Optional[int]=None,
        num_heads: Optional[int]=None,
        num_kv_heads: Optional[int]=None,
        head_dim: Optional[int]=None,
        bias: Optional[bool]=None,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        partial_rotary_factor: Optional[float] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.clamp_input_value = 0
        if hasattr(config, "clamp_training"):
            clamp_training_cfg = config.clamp_training
            self.clamp_input_value = clamp_training_cfg["clamp_input_value"]

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.bias = bias
        self.qkv_bias = getattr(config, "qkv_bias", False)
        
        self.apply_q_lora = config.apply_q_lora
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        self.num_local_heads = self.num_heads // self.tensor_model_parallel_size
        assert (
            self.num_local_heads * self.tensor_model_parallel_size
            == self.num_heads
        ), "Number of heads must be divisible by model parallel size"

        if self.apply_q_lora:
            self.q_lora_rank = config.q_lora_rank
            self.q_down_proj = ReplicatedLinear(self.hidden_size,
                                                self.q_lora_rank,
                                                bias=self.bias,
                                                quant_config=quant_config,
                                                prefix=f"{prefix}.q_down_proj")
            self.q_down_layernorm = FP32RMSNorm(
                self.q_lora_rank,
                eps=config.rms_norm_eps,
                dtype=torch.float32
            )
        else:
            self.q_lora_rank = self.hidden_size

        self.q_up_proj = ColumnParallelLinear(self.q_lora_rank,
                                               self.num_heads * 
                                               self.q_head_dim,
                                               bias=self.qkv_bias,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_up_proj")

        self.kv_down_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias = self.bias,
            quant_config = quant_config,
            prefix = f"{prefix}.kv_down_proj_with_mqa")

        self.kv_down_layernorm = FP32RMSNorm(
            self.kv_lora_rank,
            eps=config.rms_norm_eps,
            dtype=torch.float32
        )
        self.kv_up_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=self.qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_up_proj"
        )
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        bias=self.bias,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")
        
        self.norm_factor = self.q_head_dim**-0.5
        if rope_scaling and "yarn" in rope_scaling.get("rope_type", "default"):
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            factor = rope_scaling["factor"]
            rope_scaling["rope_type"] = "deepseek_yarn"
            mscale = yarn_get_mscale(factor, float(mscale_all_dim))
            self.norm_factor = self.norm_factor * mscale * mscale

        self.rotary_emb = get_rope(self.qk_rope_head_dim,
                                   rotary_dim=self.qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   dtype=torch.float32,
                                   is_neox_style=True)
        if vllm.__version__ == "0.8.5":
            self.rotary_emb._forward_method = self.rotary_emb.forward_native
       
        self.core_attention = Attention(
            num_heads=self.num_local_heads,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.norm_factor,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.core_attention",
            use_mla=True,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.q_head_dim,
            v_head_dim=self.v_head_dim,
            kv_b_proj=self.kv_up_proj,
        )


    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.clamp_input_value > 0:
            hidden_states = torch.clamp_(hidden_states, -self.clamp_input_value, self.clamp_input_value)
        local_dp_rank = os.getenv('VLLM_DP_RANK_LOCAL')
        if self.apply_q_lora:
            q = self.q_down_proj(hidden_states)[0]
            q = self.q_down_layernorm(q)
            if self.clamp_input_value > 0:
                q = torch.clamp_(q, -self.clamp_input_value, self.clamp_input_value)
            q = self.q_up_proj(q)[0]
        else:
            q = self.q_up_proj(hidden_states)[0]
        
        kv_a, k_pe = self.kv_down_proj_with_mqa(hidden_states)[0].split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_a = self.kv_down_layernorm(kv_a.contiguous())
        if self.clamp_input_value > 0:
            kv_a = torch.clamp_(kv_a, -self.clamp_input_value, self.clamp_input_value)

        q = q.view(-1, self.num_local_heads, self.q_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        origin_dtype = q.dtype
        q = q.float()
        k_pe = k_pe.float()
        q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim:], k_pe)
        q = q.to(origin_dtype)
        k_pe = k_pe.to(origin_dtype)
        
        core_attn_out = self.core_attention(
            q,
            kv_a,
            k_pe.squeeze(0),
            output_shape=(hidden_states.shape[0], self.num_local_heads * self.v_head_dim)
        )
        
        if self.clamp_input_value > 0:
            core_attn_out = torch.clamp_(core_attn_out, -self.clamp_input_value, self.clamp_input_value)
        
        return self.o_proj(core_attn_out)[0]


class IPTAttention(nn.Module):
    def __init__(
        self,
        config,
        hidden_size: Optional[int]=None,
        num_heads: Optional[int]=None,
        num_kv_heads: Optional[int]=None,
        head_dim: Optional[int]=None,
        bias: Optional[bool]=None,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        partial_rotary_factor: Optional[float] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.clamp_input_value = 0
        if hasattr(config, "clamp_training"):
            clamp_training_cfg = config.clamp_training
            self.clamp_input_value = clamp_training_cfg["clamp_input_value"]

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or self.num_heads
        self.head_dim = head_dim

        self.bias = bias

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        
        self.norm_factor = 1.0 / math.sqrt(self.head_dim)

        tp_size = get_tensor_model_parallel_world_size()
        if self.num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.num_kv_heads == 0
        self.model_parallel_size = tp_size

        self.num_heads_per_partition = self.num_heads // self.model_parallel_size
        assert (
            self.num_heads_per_partition * self.model_parallel_size == self.num_heads
        ), "Number of heads must be divisible by model parallel size"

        self.num_kv_heads_per_partition = self.num_kv_heads // self.model_parallel_size
        assert (
            self.num_kv_heads_per_partition * self.model_parallel_size == self.num_kv_heads
        ), "Number of KV heads must be divisible by model parallel size"

        self.q_size = self.num_heads_per_partition * self.head_dim
        self.kv_size = self.num_kv_heads_per_partition * self.head_dim

        self.q_k_v_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=self.bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_k_v_proj",
        )

        self.core_attention = Attention(self.num_heads_per_partition,
                              self.head_dim,
                              scale=self.norm_factor,
                              num_kv_heads=self.num_kv_heads_per_partition,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.core_attention")
    
        self.out_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=self.bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj"
        )

        if partial_rotary_factor is None:
            partial_rotary_factor = 1.0
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.clamp_input_value > 0:
            hidden_states = torch.clamp_(hidden_states, -self.clamp_input_value, self.clamp_input_value)
        qkv, _ = self.q_k_v_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.core_attention(q, k, v)
        if self.clamp_input_value > 0:
            attn_output = torch.clamp_(attn_output, - self.clamp_input_value, self.clamp_input_value)
        output, _ = self.out_proj(attn_output)
        return output


class IPTDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.cache_config = cache_config
        self.quant_config = quant_config

        self.prefix = prefix
        self.hidden_size = config.hidden_size
        self.apply_mla = config.apply_mla

        self.normalize_before = config.normalize_before
        self.use_rmsnorm = getattr(config, "use_rmsnorm", False)
        
        if self.use_rmsnorm:
            assert self.normalize_before, "`model.normalize_before` should be True for using RMSNorm."

        if not self.use_rmsnorm:
            self.layer_norm = LayerNorm(self.hidden_size,
                                            eps=config.rms_norm_eps)
        else:
            self.layer_norm = FP32RMSNorm(self.hidden_size,
                                        eps=config.rms_norm_eps, dtype=torch.float32)

        self.attention = self.build_attention()

        if not self.use_rmsnorm:
            self.final_layer_norm = LayerNorm(self.hidden_size,
                                    eps=config.rms_norm_eps)
        else:
            self.final_layer_norm = FP32RMSNorm(self.hidden_size,
                                    eps=config.rms_norm_eps, dtype=torch.float32)

        # MoE configs
        self.use_moe = False
        layer_idx = int(prefix.split(sep='.')[-1])
        self.layer_idx = layer_idx 
        
        self.expert_interval = config.expert_interval
        self.skip_first_n_layers = config.skip_first_n_layers
        if layer_idx >= self.skip_first_n_layers:
            self.use_moe = True
        self.mlp = self.build_mlp()

    def build_attention(
        self,
    ):
        attn_kwargs = {
            "config": self.config,
            "hidden_size": self.hidden_size,
            "num_heads": self.config.num_attention_heads,
            "num_kv_heads": getattr(self.config, "num_kv_heads", None),
            "head_dim": getattr(self.config, "head_dim", None),
            "bias": self.config.bias,
            "rope_theta": self.config.rope_theta,
            "rope_scaling": self.config.rope_scaling,
            "max_position_embeddings": self.config.max_position_embeddings,
            "partial_rotary_factor": self.config.partial_rotary_factor,
            "cache_config": self.cache_config,
            "quant_config": self.quant_config,
            "prefix": f"{self.prefix}.attention"
        }
        if self.apply_mla:
            return IPTMultiheadLatentAttention(**attn_kwargs)
        else:
            return IPTAttention(**attn_kwargs)
    
    def build_mlp(
        self,
    ):
        apply_gmoe = getattr(self.config, "apply_gmoe", False)
        if apply_gmoe and self.use_moe:
            return IPTMoE(
                config=self.config,
                quant_config=self.quant_config,
                prefix=f"{self.prefix}.mlp"
            )
        else:
            return IPTMLP(
                config=self.config,
                hidden_size=self.hidden_size,
                intermediate_size=self.config.intermediate_size,
                bias=self.config.bias,
                gate_gelu=self.config.gate_gelu,
                quant_config=self.quant_config,
                prefix=f"{self.prefix}.mlp"
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.normalize_before:
            if residual is None:
                residual = hidden_states
                hidden_states = self.layer_norm(hidden_states)
            else:
                hidden_states, residual = self.layer_norm(
                    hidden_states, residual)
        else:
            residual = hidden_states

        hidden_states = self.attention(positions=positions,
                                hidden_states=hidden_states)
        
        if self.normalize_before:
            hidden_states, residual = self.final_layer_norm(hidden_states, residual)
        else:
            hidden_states, residual = self.layer_norm(hidden_states, residual)
        
        hidden_states = self.mlp(hidden_states)

        if not self.normalize_before:
            hidden_states, residual = self.final_layer_norm(hidden_states, residual)
            residual = None
        return hidden_states, residual


class IPTTransformer(nn.Module):
    """Transformer class."""
    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.use_rmsnorm = getattr(config, "use_rmsnorm", False)
        self.mtp_version = getattr(config, "mtp_version", 1)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: IPTDecoderLayer(config=config,
                                        cache_config=cache_config,
                                        quant_config=quant_config,
                                        prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            if not self.use_rmsnorm:
                self.layernorm = LayerNorm(
                    config.hidden_size,
                    eps=config.rms_norm_eps,
                )
            else:
                self.layernorm = FP32RMSNorm(
                    config.hidden_size,
                    eps=config.rms_norm_eps,
                    dtype=torch.float32
            )
        else:
            self.layernorm = PPMissingLayer()


    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.layernorm(hidden_states, residual)
        return hidden_states

@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class IPTModel(nn.Module):
    def __init__(self, 
                *, vllm_config: VllmConfig,
                prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config= vllm_config.cache_config

        if get_pp_group().is_first_rank:
            self.embedding = ExEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix='embedding'
            )
        else:
            self.embedding = PPMissingLayer()

        # Transformer
        self.transformer = IPTTransformer(
            config,
            cache_config=cache_config,
            quant_config=quant_config,            
            prefix=maybe_prefix(prefix, "transformer")
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        # print("====== start forward: ", input_ids)
        hidden_states = self.transformer(
            hidden_states=hidden_states,
            positions=positions,
            intermediate_tensors=intermediate_tensors
        )
        return hidden_states
# tensor_np = tensor.numpy()
# np.savetxt('tensor_2d.txt', tensor_np, fmt='%.6f', delimiter=',')       

class IPTForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {}

    # LoRA specific attributes
    supported_lora_modules = [
        "q_k_v_proj",
        "out_proj",
        "fc1",
        "fc2",
    ]
    embedding_modules = {}
    embedding_padding_modules = []
    
    def __init__(
        self, *,
        vllm_config: VllmConfig,
        prefix: str = ""
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        
        self.config = config
        self.quant_config = quant_config
        self.lora_config = lora_config

        self.model = IPTModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        self.clamp_input_value = 0
        if hasattr(config, "clamp_training"):       
            clamp_training_cfg = config.clamp_training
            self.clamp_input_value = clamp_training_cfg["embed_state"]["clamp_input_value"]

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                        config.hidden_size,
                                        quant_config=quant_config)
        else:
            self.lm_head = PPMissingLayer()
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embedding.word_embeddings.weight

        self.logits_processor = LogitsProcessor(config.padded_vocab_size)
    
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.clamp_input_value > 0:
            hidden_states = torch.clamp_(hidden_states, -self.clamp_input_value, self.clamp_input_value)
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], split_size: int=1000000, split_index: int=0):
        skip_prefixes = ["_extra_state"]

        num_experts = self.config.num_groups * (self.config.num_routed_experts - self.config.num_shared_experts)
        expert_params_mapping = GroupedMoE.make_expert_params_mapping(num_experts=num_experts, grouped_gemm=self.config.grouped_gemm)

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if any(name.startswith(p) for p in skip_prefixes):
                logger.debug("Skipping weight %s", name)
                continue
            
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param,
                            loaded_weight,
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            split_size=split_size, 
                            split_index=split_index)
                break   
            else:
                if is_pp_missing_parameter(name, self):
                    continue

                if name not in params_dict:
                    logger.debug("Skipping weight %s", name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params