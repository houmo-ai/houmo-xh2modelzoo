from types import SimpleNamespace

import torch
import torch.nn as nn


class LagunaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        raise NotImplementedError


class LagunaRotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, max_seq_len: int = 32):
        super().__init__()
        self.rope_type = "default"
        self.config = SimpleNamespace(
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
            head_dim=rotary_dim,
            hidden_size=rotary_dim,
            num_attention_heads=1,
            max_position_embeddings=max_seq_len,
        )
        self.register_buffer("inv_freq", torch.ones(rotary_dim // 2))
        self.register_buffer("original_inv_freq", torch.ones(rotary_dim // 2))
        self.attention_scaling = 1.0
        self.max_seq_len_cached = max_seq_len
        self.original_max_seq_len = max_seq_len

    def forward(self, hidden_states, position_ids):
        raise NotImplementedError


class LagunaMLP(nn.Module):
    def __init__(self, config, intermediate_size: int):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class LagunaTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.router_logit_softcapping = 0.0
        self.weight = nn.Parameter(torch.zeros(config.num_experts, config.hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(config.num_experts), requires_grad=False)

    def forward(self, hidden_states):
        raise NotImplementedError


class LagunaExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(config.num_experts, 2 * config.moe_intermediate_size, config.hidden_size)
        )
        self.down_proj = nn.Parameter(torch.empty(config.num_experts, config.hidden_size, config.moe_intermediate_size))
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj, mean=0.0, std=0.02)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, selected_experts, routing_weights):
        raise NotImplementedError


class LagunaSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = LagunaTopKRouter(config)
        self.experts = LagunaExperts(config)
        self.shared_expert = LagunaMLP(config, config.shared_expert_intermediate_size)
        self.routed_scaling_factor = config.moe_routed_scaling_factor

    def forward(self, hidden_states):
        raise NotImplementedError


class LagunaAttention(nn.Module):
    def __init__(self, config, layer_idx: int, num_heads: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = config.head_dim
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.gating = True
        self.gate_per_head = True
        self.q_proj = nn.Linear(config.hidden_size, num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * config.head_dim, config.hidden_size, bias=False)
        self.g_proj = nn.Linear(config.hidden_size, num_heads, bias=False)
        self.q_norm = LagunaRMSNorm(config.head_dim)
        self.k_norm = LagunaRMSNorm(config.head_dim)

    def forward(self, hidden_states, position_embeddings, **kwargs):
        raise NotImplementedError


class LagunaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        num_heads = config.num_attention_heads_per_layer[layer_idx]
        self.self_attn = LagunaAttention(config, layer_idx, num_heads)
        self.mlp = (
            LagunaMLP(config, config.intermediate_size)
            if layer_idx in config.mlp_only_layers
            else LagunaSparseMoeBlock(config)
        )
        self.input_layernorm = LagunaRMSNorm(config.hidden_size)
        self.post_attention_layernorm = LagunaRMSNorm(config.hidden_size)

    def forward(self, hidden_states, **kwargs):
        raise NotImplementedError


class LagunaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([LagunaDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.norm = LagunaRMSNorm(config.hidden_size)
        self.rotary_emb = LagunaRotaryEmbedding(config.head_dim // 2)
        self.swa_rotary_emb = LagunaRotaryEmbedding(config.head_dim)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, inputs_embeds, **kwargs):
        raise NotImplementedError


class LagunaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LagunaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, inputs_embeds, **kwargs):
        raise NotImplementedError


def build_tiny_laguna(num_hidden_layers: int = 2) -> LagunaForCausalLM:
    layer_types = ["full_attention" if index % 4 == 0 else "sliding_attention" for index in range(num_hidden_layers)]
    config = SimpleNamespace(
        vocab_size=1024,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_attention_heads_per_layer=[4 if layer_type == "full_attention" else 6 for layer_type in layer_types],
        num_key_value_heads=2,
        head_dim=4,
        layer_types=layer_types,
        sliding_window=8,
        gating="per-head",
        hidden_act="silu",
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        norm_topk_prob=True,
        moe_routed_scaling_factor=2.5,
        mlp_only_layers=[0],
        eos_token_id=[2, 24],
        pad_token_id=0,
    )
    return LagunaForCausalLM(config).eval().half()
