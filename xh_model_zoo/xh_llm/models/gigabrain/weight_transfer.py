import torch
import torch.nn as nn
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM


def create_and_inject_llm_skeleton(gigabrain_policy):
    """
    按 PaliGemma 配置中的 text_config 构建 Gemma2ForCausalLM skeleton，
    并注入 GigaBrain 中 expert_model 的 LLM 权重。
    """
    print("Creating local Gemma2 config from GigaBrain text_config...")

    expert_model = gigabrain_policy.paligemma_with_expert

    hidden_size = 2304
    num_attention_heads = 8
    num_key_value_heads = 4
    head_dim = 256

    config = Gemma2Config(
        vocab_size=257216,
        hidden_size=hidden_size,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
    )
    config.query_pre_attn_scalar = 256
    config.attn_logit_softcapping = 50.0
    config.final_logit_softcapping = 30.0
    config.sliding_window = 4096

    print("Initializing empty Gemma2ForCausalLM skeleton...")
    standard_llm = Gemma2ForCausalLM(config)
    standard_llm.eval()

    print("Extracting LLM weights (index=0) from GigaBrain policy and injecting...")

    with torch.no_grad():
        # --- Token Embeddings ---
        assert standard_llm.model.embed_tokens.weight.shape == expert_model.embed_tokens.weight.shape, (
            f"embed_tokens mismatch: "
            f"{standard_llm.model.embed_tokens.weight.shape} vs {expert_model.embed_tokens.weight.shape}"
        )
        standard_llm.model.embed_tokens.weight.copy_(expert_model.embed_tokens.weight)

        # --- Decoder Layers ---
        for i, layer in enumerate(expert_model.layers):
            target_layer = standard_llm.model.layers[i]

            # Attention
            assert target_layer.self_attn.q_proj.weight.shape == layer.self_attn.q_proj[0].weight.shape, (
                f"layer {i} q_proj mismatch: "
                f"{target_layer.self_attn.q_proj.weight.shape} vs {layer.self_attn.q_proj[0].weight.shape}"
            )
            assert target_layer.self_attn.k_proj.weight.shape == layer.self_attn.k_proj[0].weight.shape, (
                f"layer {i} k_proj mismatch: "
                f"{target_layer.self_attn.k_proj.weight.shape} vs {layer.self_attn.k_proj[0].weight.shape}"
            )
            assert target_layer.self_attn.v_proj.weight.shape == layer.self_attn.v_proj[0].weight.shape, (
                f"layer {i} v_proj mismatch: "
                f"{target_layer.self_attn.v_proj.weight.shape} vs {layer.self_attn.v_proj[0].weight.shape}"
            )
            assert target_layer.self_attn.o_proj.weight.shape == layer.self_attn.o_proj[0].weight.shape, (
                f"layer {i} o_proj mismatch: "
                f"{target_layer.self_attn.o_proj.weight.shape} vs {layer.self_attn.o_proj[0].weight.shape}"
            )

            target_layer.self_attn.q_proj.weight.copy_(layer.self_attn.q_proj[0].weight)
            target_layer.self_attn.k_proj.weight.copy_(layer.self_attn.k_proj[0].weight)
            target_layer.self_attn.v_proj.weight.copy_(layer.self_attn.v_proj[0].weight)
            target_layer.self_attn.o_proj.weight.copy_(layer.self_attn.o_proj[0].weight)

            # MLP
            assert target_layer.mlp.gate_proj.weight.shape == layer.mlps[0].gate_proj.weight.shape, (
                f"layer {i} gate_proj mismatch: "
                f"{target_layer.mlp.gate_proj.weight.shape} vs {layer.mlps[0].gate_proj.weight.shape}"
            )
            assert target_layer.mlp.up_proj.weight.shape == layer.mlps[0].up_proj.weight.shape, (
                f"layer {i} up_proj mismatch: "
                f"{target_layer.mlp.up_proj.weight.shape} vs {layer.mlps[0].up_proj.weight.shape}"
            )
            assert target_layer.mlp.down_proj.weight.shape == layer.mlps[0].down_proj.weight.shape, (
                f"layer {i} down_proj mismatch: "
                f"{target_layer.mlp.down_proj.weight.shape} vs {layer.mlps[0].down_proj.weight.shape}"
            )

            target_layer.mlp.gate_proj.weight.copy_(layer.mlps[0].gate_proj.weight)
            target_layer.mlp.up_proj.weight.copy_(layer.mlps[0].up_proj.weight)
            target_layer.mlp.down_proj.weight.copy_(layer.mlps[0].down_proj.weight)

            # LayerNorm
            assert target_layer.input_layernorm.weight.shape == layer.input_layernorms[0].weight.shape
            assert target_layer.post_attention_layernorm.weight.shape == layer.post_attention_layernorms[0].weight.shape
            assert target_layer.pre_feedforward_layernorm.weight.shape == layer.pre_feedforward_layernorms[0].weight.shape
            assert target_layer.post_feedforward_layernorm.weight.shape == layer.post_feedforward_layernorms[0].weight.shape

            target_layer.input_layernorm.weight.copy_(layer.input_layernorms[0].weight)
            target_layer.post_attention_layernorm.weight.copy_(layer.post_attention_layernorms[0].weight)
            target_layer.pre_feedforward_layernorm.weight.copy_(layer.pre_feedforward_layernorms[0].weight)
            target_layer.post_feedforward_layernorm.weight.copy_(layer.post_feedforward_layernorms[0].weight)

        # --- Final Norm ---
        assert standard_llm.model.norm.weight.shape == expert_model.norms[0].weight.shape, (
            f"final norm mismatch: "
            f"{standard_llm.model.norm.weight.shape} vs {expert_model.norms[0].weight.shape}"
        )
        standard_llm.model.norm.weight.copy_(expert_model.norms[0].weight)

        # --- LM Head ---
        assert standard_llm.lm_head.weight.shape == expert_model.lm_head.weight.shape, (
            f"lm_head mismatch: "
            f"{standard_llm.lm_head.weight.shape} vs {expert_model.lm_head.weight.shape}"
        )
        standard_llm.lm_head.weight.copy_(expert_model.lm_head.weight)

    print("Successfully injected GigaBrain LLM weights into standard Gemma2 skeleton!")
    return standard_llm
def create_and_inject_expert_skeleton(gigabrain_policy):
    """
    手工构建 Gemma2Config，创建无权重的 Gemma2ForCausalLM 骨架，
    并将 GigaBrain 中 Expert 部分 (index=1) 的权重注入其中。
    """
    print("Creating local Gemma2 config from GigaBrain parameters for Expert...")
    expert_model = gigabrain_policy.paligemma_with_expert

    hidden_size = getattr(expert_model, 'expert_hidden_size', 1024)
    intermediate_size = getattr(expert_model, 'expert_intermediate_size', 2048)
    num_attention_heads = getattr(expert_model, 'expert_num_attention_heads', 8)
    num_key_value_heads = getattr(expert_model, 'expert_num_key_value_heads', 4)
    head_dim = getattr(expert_model, 'expert_head_dim', 256)
    rms_norm_eps = getattr(expert_model, 'expert_rms_norm_eps', 1e-6)
    
    # 构造 Expert 专用的 Config
    config = Gemma2Config(
        vocab_size=getattr(expert_model, 'paligemma_vocab_size', 257216),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=len(expert_model.layers), # 26
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
        hidden_activation=getattr(expert_model, 'expert_hidden_act', 'gelu_pytorch_tanh'),
        attention_bias=getattr(expert_model, 'expert_attention_bias', False),
    )
    config.query_pre_attn_scalar = getattr(expert_model, 'expert_query_pre_attn_scalar', 256.0)

    print("Initializing empty Gemma2ForCausalLM skeleton for Expert...")
    standard_expert = Gemma2ForCausalLM(config)
    standard_expert.eval()

    print("Extracting Expert weights (index=1) from GigaBrain policy and injecting...")
    with torch.no_grad():
        for i, layer in enumerate(expert_model.layers):
            target_layer = standard_expert.model.layers[i]

            # --- Attention --- (取 [1])
            target_layer.self_attn.q_proj.weight.copy_(layer.self_attn.q_proj[1].weight)
            target_layer.self_attn.k_proj.weight.copy_(layer.self_attn.k_proj[1].weight)
            target_layer.self_attn.v_proj.weight.copy_(layer.self_attn.v_proj[1].weight)
            target_layer.self_attn.o_proj.weight.copy_(layer.self_attn.o_proj[1].weight)

            # --- MLP --- (取 [1])
            target_layer.mlp.gate_proj.weight.copy_(layer.mlps[1].gate_proj.weight)
            target_layer.mlp.up_proj.weight.copy_(layer.mlps[1].up_proj.weight)
            target_layer.mlp.down_proj.weight.copy_(layer.mlps[1].down_proj.weight)

            # --- LayerNorms --- (取 [1], 注意其中 input 和 pre_ffn 带有 dense 层)
            # target_layer.input_layernorm.weight.copy_(layer.input_layernorms[1].weight)
            if hasattr(layer.input_layernorms[1], 'dense'):
                target_layer.input_layernorm.dense = nn.Linear(*layer.input_layernorms[1].dense.weight.shape[::-1])
                target_layer.input_layernorm.dense.weight.copy_(layer.input_layernorms[1].dense.weight)
                target_layer.input_layernorm.dense.bias.copy_(layer.input_layernorms[1].dense.bias)

            target_layer.post_attention_layernorm.weight.copy_(layer.post_attention_layernorms[1].weight)

            # target_layer.pre_feedforward_layernorm.weight.copy_(layer.pre_feedforward_layernorms[1].weight)
            if hasattr(layer.pre_feedforward_layernorms[1], 'dense'):
                target_layer.pre_feedforward_layernorm.dense = nn.Linear(*layer.pre_feedforward_layernorms[1].dense.weight.shape[::-1])
                target_layer.pre_feedforward_layernorm.dense.weight.copy_(layer.pre_feedforward_layernorms[1].dense.weight)
                target_layer.pre_feedforward_layernorm.dense.bias.copy_(layer.pre_feedforward_layernorms[1].dense.bias)

            target_layer.post_feedforward_layernorm.weight.copy_(layer.post_feedforward_layernorms[1].weight)

        # --- Final Norm ---
        # standard_expert.model.norm.weight.copy_(expert_model.norms[1].weight)
        if hasattr(expert_model.norms[1], 'dense'):
            standard_expert.model.norm.dense = nn.Linear(*expert_model.norms[1].dense.weight.shape[::-1])
            standard_expert.model.norm.dense.weight.copy_(expert_model.norms[1].dense.weight)
            standard_expert.model.norm.dense.bias.copy_(expert_model.norms[1].dense.bias)

    print("Successfully injected GigaBrain Expert weights into standard Gemma2 skeleton!")
    return standard_expert
