from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union, get_args

import torch
import torch.nn.functional as F
import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoEMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils import cdiv, direct_register_custom_op

from .fused_moe import fused_experts


logger = init_logger(__name__)

try:
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer
except:
    logger.warning("current vllm not support router replay")


@CustomOp.register("unquantized_grouped_moe")
class UnquantizedGroupedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        fc1_output_size: int,
        fc2_input_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        fc1_weights = torch.nn.Parameter(
            torch.empty(num_experts, fc1_output_size, hidden_size, dtype=params_dtype), requires_grad=False
        )
        layer.register_parameter("fc1_weights", fc1_weights)
        set_weight_attrs(fc1_weights, extra_weight_attrs)

        fc2_weights = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, fc2_input_size, dtype=params_dtype), requires_grad=False
        )
        layer.register_parameter("fc2_weights", fc2_weights)
        set_weight_attrs(fc2_weights, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        top_k: int,
        router_logits: torch.Tensor,
        num_shared_experts: int = 0,
        global_num_experts: int = -1,
        routed_scaling_factor: float = 1.0,
        calc_denominator_cross_groups: bool = False,
        expert_map: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        activation: str = "gelu",
        clamp_input_value: int = 0,
        num_pad_experts: int = 0,
        ep_rank: int = 0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = GroupedMoE.select_experts(
            logits=router_logits,
            top_k=top_k,
            routed_scaling_factor=routed_scaling_factor,
            calc_denominator_cross_groups=calc_denominator_cross_groups,
            e_score_correction_bias=e_score_correction_bias,
            num_pad_experts=num_pad_experts,
            layer_id=layer_id,
        )

        if self.fused_experts is not None:
            result = self.fused_experts(
                hidden_states=x,
                w1=layer.fc1_weights,
                w2=layer.fc2_weights,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )
        else:
            assert fused_experts is not None
            result = fused_experts(
                hidden_states=x,
                w1=layer.fc1_weights,
                w2=layer.fc2_weights,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                quant_config=self.moe_quant_config,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                clamp_input_value=clamp_input_value,
            )
        return result

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        return FUSED_MOE_UNQUANTIZED_CONFIG


def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    global_num_experts: int,
    expert_placement_strategy: ExpertPlacementStrategy = "linear",
    num_fused_shared_experts: int = 0,
    return_expert_mask: bool = False,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    """
    Calculates how many experts should be assigned to each rank for EP and
    creates a mapping from global to local expert index. Experts are
    distributed evenly across ranks. Any remaining are assigned to the
    last rank.

    Args:
        ep_size: The size of the expert parallel group
        ep_rank: The rank of the current process in the expert parallel
            group
        global_num_experts: The total number of experts in the model.
        expert_placement_strategy: The expert placement strategy.

    Returns:
        tuple[int, Optional[torch.Tensor]]: A tuple containing:
            - local_num_experts (int): The number of experts assigned
                to the current rank.
            - expert_map (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts,) mapping from global to local index.
                Contains -1 for experts not assigned to the current rank.
                Returns None if ep_size is 1.
            - expert_mask (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts + num_fused_shared_experts + 1,)
                containing 1 for experts assigned to the current rank
                and 0 for sentinel.
                Returns None if ep_size is 1.
                Used only when AITER MOE is enabled.
    """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None, None)

    # Distribute experts as evenly as possible to each rank.
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    # Create an expert map for the local experts
    if expert_placement_strategy == "linear":
        start_idx = ep_rank * base_experts + min(ep_rank, remainder)
        expert_map[start_idx : start_idx + local_num_experts] = torch.arange(0, local_num_experts, dtype=torch.int32)
    elif expert_placement_strategy == "round_robin":
        local_log_experts = torch.arange(ep_rank, global_num_experts, ep_size, dtype=torch.int32)

        expert_map[local_log_experts] = torch.arange(0, local_num_experts, dtype=torch.int32)
    else:
        raise ValueError(
            "Unsupported expert placement strategy "
            f"'{expert_placement_strategy}', expected one of "
            f"{get_args(ExpertPlacementStrategy)}"
        )

    expert_mask = None
    if return_expert_mask:
        expert_mask = torch.ones((global_num_experts + num_fused_shared_experts + 1,), dtype=torch.int32)
        expert_mask[-1] = 0
        expert_mask[:global_num_experts] = expert_map > -1
        expert_map = torch.cat(
            (
                expert_map,
                torch.tensor(
                    [local_num_experts + i for i in range(num_fused_shared_experts)],
                    dtype=torch.int32,
                ),
            ),
            dim=0,
        )

    return (local_num_experts, expert_map, expert_mask)


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    """
    Compresses the expert map by removing any -1 entries.

    Args:
        expert_map (torch.Tensor): A tensor of shape (global_num_experts,)
            mapping from global to local index. Contains -1 for experts not
            assigned to the current rank.

    Returns:
        str: A string mapping from local to global index.
            Using str to support hashing for logging once only.
    """
    global_indices = torch.where(expert_map != -1)[0]
    local_indices = expert_map[global_indices]
    return ", ".join(
        f"{local_index.item()}->{global_index.item()}"
        for local_index, global_index in zip(local_indices, global_indices)
    )


@CustomOp.register("grouped_moe")
class GroupedMoE(CustomOp):
    """GroupedMoE layer for MoE models"""

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        num_shared_experts: int,
        hidden_size: int,
        intermediate_size: int,
        clamp_input_value: int,
        routed_scaling_factor: float = None,
        calc_denominator_cross_groups: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        num_expert_group: Optional[int] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        prefix: str = "",
        apply_router_weight_on_input: bool = False,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        activation: str = "gelu",
        is_sequence_parallel=False,
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        num_pad_experts: int = 0,
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        vllm_config = get_current_vllm_config()
        tp_size_ = tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        dp_size_ = dp_size if dp_size is not None else get_dp_group().world_size

        self.is_sequence_parallel = is_sequence_parallel
        self.sp_size = tp_size_ if is_sequence_parallel else 1

        self.moe_parallel_config: FusedMoEParallelConfig = FusedMoEParallelConfig.make(
            tp_size_=tp_size_,
            dp_size_=dp_size_,
            vllm_parallel_config=vllm_config.parallel_config,
        )

        self.global_num_experts = num_experts

        # For smuggling this layer into the fused moe custom op
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer name: {}".format(prefix))
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix
        from vllm.model_executor.models.utils import extract_layer_index

        self.layer_id = extract_layer_index(self.layer_name)

        self.enable_eplb = enable_eplb

        if self.use_ep:
            # if self.enable_eplb:
            #     assert self.global_num_experts % self.ep_size == 0, (
            #         "EPLB currently only supports even distribution of "
            #         "experts across ranks."
            #     )
            # else:
            #     assert num_redundant_experts == 0, (
            #         "Redundant experts are only supported with EPLB."
            #     )

            expert_placement_strategy = vllm_config.parallel_config.expert_placement_strategy
            if expert_placement_strategy == "round_robin":
                # TODO(Bruce): will support round robin expert placement with
                # EPLB enabled in the future.
                round_robin_supported = (
                    (num_expert_group is not None and num_expert_group > 1)
                    and num_redundant_experts == 0
                    and not self.enable_eplb
                )

                if not round_robin_supported:
                    logger.warning(
                        "Round-robin expert placement is only supported for "
                        "models with multiple expert groups and no redundant "
                        "experts. Falling back to linear expert placement."
                    )
                    expert_placement_strategy = "linear"

            self.expert_map: torch.Tensor | None
            local_num_experts, expert_map, expert_mask = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=expert_placement_strategy,
            )

            self.local_num_experts = local_num_experts
            self.register_buffer("expert_map", expert_map)
            self.register_buffer("expert_mask", expert_mask)
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Expert "
                "placement strategy: %s. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.",
                self.ep_rank,
                self.ep_size,
                expert_placement_strategy,
                self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self.expert_map),
            )
        else:
            self.local_num_experts, self.expert_map, self.expert_mask = (
                self.global_num_experts,
                None,
                None,
            )

        self.hidden_size = hidden_size
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.num_shared_experts = num_shared_experts
        self.calc_denominator_cross_groups = calc_denominator_cross_groups
        self.e_score_correction_bias = e_score_correction_bias
        self.reduce_results = reduce_results
        self.clamp_input_value = clamp_input_value
        self.num_pad_experts = num_pad_experts

        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation
        self.intermediate_size = intermediate_size
        assert intermediate_size % self.tp_size == 0

        self.fc2_input_size = self.intermediate_size // self.tp_size
        self.fc1_output_size = self.fc2_input_size
        if self.activation == "gelu":
            self.fc1_output_size = self.fc1_output_size * 2

        if vllm_config.model_config is not None:
            moe_in_dtype = vllm_config.model_config.dtype
        else:
            # TODO (bnell): This is a hack to get test_mixtral_moe to work
            # since model_config is not set in the pytest test.
            moe_in_dtype = params_dtype

        moe = FusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=moe_in_dtype,
            max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
            has_bias=False,
        )
        self.moe_config = moe
        self.quant_config = quant_config

        def _get_quant_method() -> FusedMoEMethodBase:
            """
            Helper method to ensure self.quant_method is never None and
            of the proper type.
            """
            quant_method = None
            if self.quant_config is not None:
                quant_method = self.quant_config.get_quant_method(self, prefix)
            if quant_method is None:
                quant_method = UnquantizedGroupedMoEMethod(self.moe_config)
            assert isinstance(quant_method, FusedMoEMethodBase)
            return quant_method

        self.quant_method: FusedMoEMethodBase = _get_quant_method()

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "fc1_output_size": self.fc1_output_size,
            "fc2_input_size": self.fc2_input_size,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if self.quant_method.__class__.__name__ in (
            "GPTQMarlinMoEMethod",
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            moe_quant_params["intermediate_size_full"] = self.intermediate_size

        self.quant_method.create_weights(layer=self, **moe_quant_params)

        # Chunked all2all staging tensor
        self.batched_hidden_states: Optional[torch.Tensor] = None
        self.batched_router_logits: Optional[torch.Tensor] = None

        # TODO(bnell): flashinfer uses non-batched format.
        # Does it really need a batched buffer?
        if self.moe_parallel_config.use_pplx_kernels or self.moe_parallel_config.use_deepep_ll_kernels:
            if vllm_config.parallel_config.enable_dbo:
                self.batched_hidden_states = torch.zeros(
                    (2, moe.max_num_tokens, self.hidden_size), dtype=moe.in_dtype, device=torch.cuda.current_device()
                )

                # Note here we use `num_experts` which is logical expert count
                self.batched_router_logits = torch.zeros(
                    (2, moe.max_num_tokens, num_experts), dtype=moe.in_dtype, device=torch.cuda.current_device()
                )
            else:
                self.batched_hidden_states = torch.zeros(
                    (moe.max_num_tokens, self.hidden_size), dtype=moe.in_dtype, device=torch.cuda.current_device()
                )

                # Note here we use `num_experts` which is logical expert count
                self.batched_router_logits = torch.zeros(
                    (moe.max_num_tokens, num_experts), dtype=moe.in_dtype, device=torch.cuda.current_device()
                )

    @staticmethod
    def select_experts(
        logits: torch.Tensor,
        top_k: int,
        routed_scaling_factor: float,
        calc_denominator_cross_groups: bool,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        num_pad_experts: int = 0,
        layer_id: Optional[int] = None,
    ):
        num_tokens, num_groups, num_experts_per_group = logits.shape
        scores = logits.sigmoid()  # [num_tokens, groups, num_routed_experts_per_group]
        if num_pad_experts > 0:
            scores = scores[:, :, :-num_pad_experts]
            scores = F.pad(scores, (0, num_pad_experts), value=0)

        scores_for_choice = scores.view(num_tokens, -1) + e_score_correction_bias.unsqueeze(
            0
        )  # [num_tokens, groups * num_routed_experts_per_group]
        scores_for_choice = scores_for_choice.view_as(scores)  # [num_tokens, groups, num_routed_experts_per_group]
        _, topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)  # [num_tokens, groups, topk]
        topk_probs = scores.gather(-1, topk_indices)
        if top_k > 1:
            if calc_denominator_cross_groups:
                denominator = topk_probs.view(topk_probs.size(0), -1)
                denominator = denominator.sum(dim=-1, keepdim=True) + 1e-20
                denominator = denominator.unsqueeze(-1)
            else:
                denominator = topk_probs.sum(dim=-1, keepdim=True) + 1e-20
            topk_probs = topk_probs / denominator
        topk_probs = topk_probs * routed_scaling_factor

        head_incre = (
            torch.arange(num_groups, dtype=topk_indices.dtype, device=topk_indices.device) * num_experts_per_group
        ).view(1, -1, 1)
        topk_indices = (topk_indices + head_incre).view(num_tokens, -1).to(torch.int32)
        topk_probs = topk_probs.view(num_tokens, -1).to(torch.float32)
        # print('topk_indices: ', topk_indices)
        try:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.capture(  # noqa
                    layer_id=layer_id,
                    topk_ids=topk_indices,
                )
        except:
            pass

        return topk_probs, topk_indices

    @classmethod
    def make_expert_params_mapping(cls, num_experts: int, grouped_gemm: bool) -> List[Tuple[str, str, int, str]]:
        if grouped_gemm:
            return [
                ("routed_experts.fc1_weights", "routed_experts.fc1_weights", 0, "fc1"),
                ("routed_experts.fc2_weights", "routed_experts.fc2_weights", 0, "fc2"),
            ]
        else:
            return [
                # (param_name, weight_name, expert_id, shard_id)
                (
                    "routed_experts.fc1_weights" if weight_name == "fc1" else "routed_experts.fc2_weights",
                    f"routed_experts.{expert_id}.{weight_name}.weight",
                    expert_id,
                    weight_name,
                )
                for expert_id in range(num_experts)
                for weight_name in ["fc1", "fc2"]
            ]

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        split_size: int = 1000000,
        split_index: int = 0,
    ) -> None:
        if not self.use_ep:
            SHARD_ID_TO_SHARDED_DIM = {"fc1": 0, "fc2": 1}
            shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
            full_load = len(loaded_weight.shape) == 3
            if full_load:
                shard_dim += 1

            expert_data = param.data if full_load else param.data[expert_id]
            if "weights" in weight_name:
                if shard_id == "fc1":
                    gate, up = loaded_weight.chunk(2, dim=shard_dim)
                    shard_size = self.intermediate_size // self.tp_size
                    gate = gate.narrow(shard_dim, shard_size * self.tp_rank, shard_size)
                    up = up.narrow(shard_dim, shard_size * self.tp_rank, shard_size)
                    loaded_weight = torch.cat((gate, up), dim=shard_dim)
                else:
                    shard_size = expert_data.shape[shard_dim]
                    loaded_weight = loaded_weight.narrow(shard_dim, shard_size * self.tp_rank, shard_size)
                # assert expert_data.shape == loaded_weight.shape
                # expert_data.copy_(loaded_weight)
                if "fc1_weight" in weight_name:
                    expert_data[:, :, split_size * split_index : split_size * (split_index + 1)].copy_(loaded_weight)
                else:
                    expert_data[:, split_size * split_index : split_size * (split_index + 1), :].copy_(loaded_weight)

        # ep weight loader
        else:
            full_load = len(loaded_weight.shape) == 3
            if full_load:
                indices = torch.where(self.expert_map != -1)[0]
                for local_expert_id, global_expert_id in enumerate(indices):
                    expert_data = param.data[local_expert_id]
                    local_loaded_weight = loaded_weight[global_expert_id]
                    # assert expert_data.shape == local_loaded_weight.shape
                    # expert_data.copy_(local_loaded_weight)
                    if "fc1_weight" in weight_name:
                        expert_data[:, split_size * split_index : split_size * (split_index + 1)].copy_(
                            local_loaded_weight
                        )
                    else:
                        expert_data[split_size * split_index : split_size * (split_index + 1), :].copy_(
                            local_loaded_weight
                        )
            else:
                global_expert_id = expert_id
                local_expert_id = self._map_global_expert_id_to_local_expert_id(global_expert_id)
                if local_expert_id == -1:
                    # Failed to load this param since it's not local to this rank
                    return None
                expert_data = param.data[local_expert_id]
                # print(f"[EP-Check] Rank {self.ep_rank}: Loading Global Expert {global_expert_id} "
                #   f"into Local Index {local_expert_id}. "
                #   f"Weight Shape: {loaded_weight.shape}")
                if "weights" in weight_name:
                    if expert_data.shape != loaded_weight.shape:
                        raise ValueError(
                            f"Shape mismatch for Expert {global_expert_id} ({weight_name}): "
                            f"Disk shape {loaded_weight.shape} vs GPU shape {expert_data.shape}"
                        )
                expert_data.copy_(loaded_weight)

    def ensure_moe_quant_config(self):
        if self.quant_method.moe_quant_config is None:
            self.quant_method.moe_quant_config = self.quant_method.get_fused_moe_quant_config(self)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        if current_platform.is_tpu():
            return self.forward_impl(hidden_states, router_logits)
        else:
            return torch.ops.vllm.grouped_moe_forward(hidden_states, router_logits, self.layer_name)

    def forward_impl_chunked(
        self,
        full_hidden_states: torch.Tensor,
        full_router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None
        assert self.batched_hidden_states.dtype == full_hidden_states.dtype
        assert self.batched_router_logits.dtype == full_router_logits.dtype
        # Check size compatibility.
        assert self.batched_hidden_states.size(-1) == full_hidden_states.size(-1)
        assert self.batched_router_logits.size(-1) == full_router_logits.size(-1)

        self.ensure_moe_quant_config()

        full_fused_final_hidden_states = torch.empty_like(full_hidden_states)

        def process_chunk(chunk_start, chunk_end, skip_result_store=False):
            chunk_size = chunk_end - chunk_start
            hidden_states = full_hidden_states[chunk_start:chunk_end, :]
            router_logits = full_router_logits[chunk_start:chunk_end, :]

            assert self.batched_hidden_states is not None
            assert self.batched_router_logits is not None

            batched_hidden_states = self.batched_hidden_states
            batched_router_logits = self.batched_router_logits

            assert (
                batched_hidden_states.size(0)  # type: ignore
                >= chunk_size
            )
            assert (
                batched_router_logits.size(0)  # type: ignore
                >= chunk_size
            )
            staged_hidden_states = batched_hidden_states[:chunk_size, :]  # type: ignore
            staged_router_logits = batched_router_logits[:chunk_size, :]  # type: ignore
            staged_hidden_states.copy_(hidden_states, non_blocking=True)
            staged_router_logits.copy_(router_logits, non_blocking=True)

            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=staged_hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map,
                num_shared_experts=self.num_shared_experts,
                routed_scaling_factor=self.routed_scaling_factor,
                calc_denominator_cross_groups=self.calc_denominator_cross_groups,
                e_score_correction_bias=self.e_score_correction_bias,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                activation=self.activation,
                clamp_input_value=self.clamp_input_value,
                num_pad_experts=self.num_pad_experts,
                layer_id=self.layer_id,
            )

            assert self.shared_experts is None or isinstance(final_hidden_states, tuple)

            if self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, tuple)
                assert self.shared_experts is None
                final_hidden_states, zero_expert_result = final_hidden_states
                if zero_expert_result is not None:
                    final_hidden_states += zero_expert_result

            if not skip_result_store:
                full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(final_hidden_states, non_blocking=True)

        ctx = get_forward_context()
        # flashinfer_cutlass_kernels can handle: optional DP + TP/EP
        max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu
        moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens

        # If the input to the MoE is sequence parallel then divide by sp_size
        # to find the maximum number of tokens for any individual dispatcher.
        if self.is_sequence_parallel:
            max_tokens_across_dispatchers = cdiv(max_tokens_across_dispatchers, self.sp_size)

        num_tokens = full_hidden_states.size(0)
        for chunk_idx, chunk_start_ in enumerate(range(0, max_tokens_across_dispatchers, moe_dp_chunk_size_per_rank)):
            chunk_start = chunk_start_
            chunk_end = min(chunk_start + moe_dp_chunk_size_per_rank, max_tokens_across_dispatchers)
            # clamp start and end
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            with ctx.dp_metadata.chunked_sizes(self.sp_size, moe_dp_chunk_size_per_rank, chunk_idx):
                process_chunk(chunk_start, chunk_end, skip_result_store=chunk_start_ >= num_tokens)

        return full_fused_final_hidden_states

    def forward_impl(self, hidden_states: torch.Tensor, router_logits):
        assert self.quant_method is not None

        self.ensure_moe_quant_config()

        if self.moe_parallel_config.use_pplx_kernels or self.moe_parallel_config.use_deepep_ll_kernels:
            return self.forward_impl_chunked(hidden_states, router_logits)

        do_naive_dispatch_combine: bool = self.dp_size > 1 and not self.moe_parallel_config.use_deepep_ht_kernels

        ctx = get_forward_context()
        sp_ctx = ctx.dp_metadata.sp_local_sizes(self.sp_size) if ctx.dp_metadata else nullcontext()

        with sp_ctx:
            if do_naive_dispatch_combine:
                hidden_states, router_logits = get_ep_group().dispatch(
                    hidden_states, router_logits, self.is_sequence_parallel
                )
            # Matrix multiply.
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map,
                num_shared_experts=self.num_shared_experts,
                routed_scaling_factor=self.routed_scaling_factor,
                calc_denominator_cross_groups=self.calc_denominator_cross_groups,
                e_score_correction_bias=self.e_score_correction_bias,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                activation=self.activation,
                clamp_input_value=self.clamp_input_value,
                num_pad_experts=self.num_pad_experts,
                layer_id=self.layer_id,
            )

            def reduce_output(states: torch.Tensor, do_combine: bool = True) -> torch.Tensor:
                if do_naive_dispatch_combine and do_combine:
                    states = get_ep_group().combine(states, self.is_sequence_parallel)

                if not self.is_sequence_parallel and self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
                    states = self.maybe_all_reduce_tensor_model_parallel(states)
                return states

            return reduce_output(final_hidden_states)

    def extra_repr(self) -> str:
        s = (
            f"global_num_experts={self.global_num_experts}, "
            f"local_num_experts={self.local_num_experts}, "
            f"num_shared_experts={self.num_shared_experts}, "
            f"top_k={self.top_k}, "
            f"intermediate_size={self.intermediate_size}, "  # noqa: E501
            f"tp_size={self.tp_size},\n"
            f"ep_size={self.ep_size}, "
            f"reduce_results={self.reduce_results}"
        )

        s += f", activation='{self.activation}'"  # noqa: E501
        return s

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

    @property
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep

    @property
    def use_pplx_kernels(self):
        return self.moe_parallel_config.use_pplx_kernels

    @property
    def use_deepep_ht_kernels(self):
        return self.moe_parallel_config.use_deepep_ht_kernels

    @property
    def use_deepep_ll_kernels(self):
        return self.moe_parallel_config.use_deepep_ll_kernels


def grouped_moe_forward(hidden_states: torch.Tensor, router_logits: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.quant_method is not None

    return self.forward_impl(hidden_states, router_logits)


def grouped_moe_forward_fake(hidden_states: torch.Tensor, router_logits: torch.Tensor, layer_name: str) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="grouped_moe_forward",
    op_func=grouped_moe_forward,
    mutates_args=[],
    fake_impl=grouped_moe_forward_fake,
    dispatch_key=current_platform.dispatch_key,
)
