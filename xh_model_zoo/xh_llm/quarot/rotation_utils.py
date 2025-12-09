import math
import typing

import torch
import tqdm

from . import model_utils, utils

# from quarot_quant import quant_utils
from .hadamard_utils import is_pow2, random_hadamard_matrix

# from fast_hadamard_transform import hadamard_transform

def bake_mean_into_conv(conv: torch.nn.Conv2d) -> None:
    """
    This function takes a convolutional layer and subtracts the means from the
    weights and biases. This will result in the convolutional layer performing
    the mean substitution which is usually done inside layernorm.
    """
    conv_dtype = conv.weight.dtype
    W_ = conv.weight.data.double()
    conv.weight.data = W_ - W_.mean(dim=0, keepdim=True)
    conv.weight.data = conv.weight.data.to(conv_dtype)
    if conv.bias is not None:
        b_ = conv.bias.data.double()
        conv.bias.data = b_ - b_.mean()
        conv.bias.data = conv.bias.data.to(conv_dtype)


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias") and layernorm.bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)


def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_merger_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        w_o, w_i = W_.shape
        size = layernorm.weight.shape[0]
        linear.weight.data = (W_.view(w_o, -1, size) * layernorm.weight.double()).to(linear_dtype).view(w_o, w_i)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64).to(W_))
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_.view(w_o, -1, size), layernorm.bias.double()
            ).sum(dim=-1)
            linear.bias.data = linear.bias.data.to(linear_dtype)

    layernorm.weight.data = torch.ones_like(layernorm.weight.data)
    if hasattr(layernorm, "bias"):
        layernorm.bias.data = torch.zeros_like(layernorm.bias.data)



def fuse_deepstack_linear(model):
    for layer in model.visual.deepstack_merger_list:
        fuse_merger_linear(layer.norm, [layer.linear_fc1])
        rms_weight_size = layer.norm.weight.shape[0]
        model_utils.replace_modules(
            layer, torch.nn.LayerNorm, lambda x: model_utils.Qwen3RMSNorm(rms_weight_size, eps=x.eps), False
        )


@torch.no_grad()
def fuse_layer_norms(model, device=None, llm_rotate=True):
    model_type = model_utils.get_model_type(model)

    kwargs = {"model": model, "model_type": model_type}

    # Embedding fusion
    for W in model_utils.get_embeddings(**kwargs):
        if model_type in [model_utils.LLAMA_MODEL, model_utils.QWEN_MODEL, model_utils.QWEN2_5_VL_MODEL,  model_utils.QWEN3_VL_MODEL]:
            continue
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    if model_type == model_utils.QWEN2_5_VL_MODEL:
        for layer in model.model.visual.blocks:
            fuse_ln_linear(layer.norm1, [layer.attn.qkv])
            layer.norm1.weight.fill_(1.0)
            fuse_ln_linear(layer.norm2, [layer.mlp.gate_proj, layer.mlp.up_proj])
            layer.norm2.weight.fill_(1.0)
        fuse_merger_linear(
            model.model.visual.merger.ln_q,
            [model.model.visual.merger.mlp[0]],
        )
        model.model.visual.merger.ln_q.weight.fill_(1.0)

    elif model_type == model_utils.QWEN3_VL_MODEL:
        bake_mean_into_conv(model.visual.patch_embed.proj)

        for layer in model.visual.blocks:
            fuse_ln_linear(layer.norm1, [layer.attn.qkv])
            fuse_ln_linear(layer.norm2, [layer.mlp.linear_fc1])

            bake_mean_into_linear(layer.attn.proj)
            bake_mean_into_linear(layer.mlp.linear_fc2)

        model_utils.replace_modules(
            model.visual.blocks,
            torch.nn.LayerNorm,
            lambda x: model_utils.Qwen3RMSNorm(x.weight.shape[0], eps=x.eps),
            False,
        )

        fuse_merger_linear(model.visual.merger.norm, [model.visual.merger.linear_fc1])

        model_utils.replace_modules(
            model.visual.merger,
            torch.nn.LayerNorm,
            lambda x: model_utils.Qwen3RMSNorm(x.weight.shape[0], eps=x.eps),
            False,
        )

        fuse_deepstack_linear(model)

    if not llm_rotate:
        print("Not fusing layernorms for llm part")
        return

    layers = model_utils.get_transformer_layers(**kwargs)
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    progress_bar = tqdm.tqdm(layers)
    progress_bar.set_description("Fusing layernorms")
    for layer in progress_bar:
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAMA_MODEL:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
            fuse_ln_linear(
                layer.input_layernorm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
        elif model_type in (model_utils.QWEN_MODEL, model_utils.QWEN3_MODEL, model_utils.QWEN2_5_VL_MODEL, model_utils.QWEN3_VL_MODEL):
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
            fuse_ln_linear(
                layer.input_layernorm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
        elif model_type == model_utils.QWEN3MOE_MODEL:
            linears = [layer.mlp.gate]
            for expert in layer.mlp.experts:
                linears.append(expert.gate_proj)
                linears.append(expert.up_proj)
            fuse_ln_linear(layer.post_attention_layernorm, linears)
            fuse_ln_linear(
                layer.input_layernorm,
                [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
            )
        elif model_type == model_utils.OPT_MODEL:
            fuse_ln_linear(
                layer.self_attn_layer_norm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f"Unknown model type {model_type}")

        layer.post_attention_layernorm.fuse_weight = True
        layer.post_attention_layernorm.weight.fill_(1.0)
        layer.input_layernorm.fuse_weight = True
        layer.input_layernorm.weight.fill_(1.0)

        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)

    fuse_ln_linear(
        model_utils.get_pre_head_layernorm(**kwargs),
        [model_utils.get_lm_head(**kwargs)],
    )
    if model_type in [model_utils.QWEN2_5_VL_MODEL, model_utils.QWEN3_VL_MODEL]:
       model.model.language_model.norm.weight.fill_(1.0)
       model.model.language_model.norm.fuse_weight = True
    else:
        model.model.norm.weight.fill_(1.0)
        model.model.norm.fuse_weight = True


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def rotate_conv(layer, Q_v, embed_dims):
    dtype = layer.weight.dtype
    weight_shape = layer.weight.data.shape
    layer.weight.data = (
        torch.matmul(Q_v.T, layer.weight.data.double().view(embed_dims, -1)).to(dtype).view(weight_shape)
    )
    if layer.bias is not None:
        layer.bias.data = torch.matmul(layer.bias.data.double(), Q_v).to(dtype)


def rotate_qwen2_5_vl_attention_inputs(layer, Q, is_visual=False) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    layer_list = [layer.self_attn.qkv] if not is_visual else [layer.attn.qkv]
    for W in layer_list:
        dtype = W.weight.dtype
        W_ = W.weight.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)


def rotate_qwen2_5_vl_attention_output(layer, Q, is_visual=False) -> None:
    # Rotate output matrix of the self-attention layer.
    if is_visual:
        W = layer.attn.proj
    else:
        W = layer.self_attn.o_proj

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)


def rotate_qwen2_5_vl_mlp_input(layer, Q) -> None:
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.gate_proj, layer.mlp.up_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)


def rotate_qwen2_5_vl_mlp_output(layer, Q, online_hadamard=False):
    out_layer = layer.mlp.down_proj
    # out_layer = layer.mlp.c_proj if hasattr(layer.mlp, "c_proj") else layer.mlp.fc2
    # Rotate the MLP output weights and bias.
    dtype = out_layer.weight.data.dtype
    W_ = out_layer.weight.data.to(dtype=torch.float64)
    out_layer.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)

    if online_hadamard:
        # input做hadamard变换, 它的输入做在线hadamard变换
        apply_exact_had_to_linear(
            out_layer, had_dim=-1, output=False
        )  # apply exact (inverse) hadamard on the weights of mlp output

    if out_layer.bias is not None:
        b = out_layer.bias.data.to(dtype=torch.float64)
        out_layer.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)


def rotate_qwen2_5_vl_ov_proj(layer, head_num, head_dim, is_visual=False):
    if is_visual:
        qkv = layer.attn.qkv
        o_proj = layer.attn.proj

        qkv_weight = qkv.weight.data
        split_qkv_weight = qkv_weight.split(qkv.out_features // 3, dim=0)
        v_weight = split_qkv_weight[2]

        Q = get_orthogonal_matrix(head_dim, mode="hadamard")
        dtype = v_weight.dtype
        W_ = v_weight.to(dtype=torch.float64).T.reshape(-1, head_num, head_dim)

        v_weight = torch.matmul(W_, Q).reshape(-1, head_num * head_dim).T.to(dtype=dtype)

        qkv.weight.data = torch.cat([split_qkv_weight[0], split_qkv_weight[1], v_weight], dim=0)
        if qkv.bias is not None:
            qkv_bias = qkv.bias.data
            split_qkv_bias = qkv_bias.split(qkv.out_features // 3, dim=0)
            v_bias = split_qkv_bias[2]
            v_bias = v_bias.to(dtype=torch.float64).reshape(head_num, head_dim)
            v_bias = torch.matmul(v_bias, Q).to(dtype=dtype).reshape(-1)
            qkv.bias.data = torch.cat([split_qkv_bias[0], split_qkv_bias[1], v_bias], dim=0)

        W_ = o_proj.weight.data.to(dtype=torch.float64).reshape(-1, head_num, head_dim)
        o_proj.weight.data = torch.matmul(W_, Q).reshape(-1, head_num * head_dim).to(dtype=dtype)
    else:
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
        apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)


def rotate_visual_merger(visual_model, Q: torch.Tensor) -> None:
    # Rotate the head.
    dtype = visual_model.merger.mlp[0].weight.dtype

    q_shape = Q.shape[0]
    o_shape, i_shape = visual_model.merger.mlp[0].weight.shape

    W_ = visual_model.merger.mlp[0].weight.to(dtype=torch.float64).reshape(o_shape, -1, q_shape)
    visual_model.merger.mlp[0].weight.data = torch.matmul(W_, Q).to(dtype=dtype).reshape(o_shape, i_shape).contiguous()


def get_orthogonal_matrix(size, mode, device=utils.DEV):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_qwen2_5_vl_visual_model(visual_model):
    raw_device = next(visual_model.parameters()).device
    visual_model.to(utils.DEV)

    num_heads = visual_model.blocks[0].attn.num_heads
    head_dim = visual_model.blocks[0].attn.qkv.in_features // num_heads
    Q_v = get_orthogonal_matrix(visual_model.blocks[0].attn.qkv.in_features, mode="hadamard")

    rotate_conv(
        visual_model.patch_embed.proj,
        Q_v,
        visual_model.blocks[0].attn.qkv.in_features,
    )

    for idx, layer in enumerate(
        tqdm.tqdm(
            visual_model.blocks,
            unit="layer",
            desc="Rotating Qwen2_5 Visual",
        )
    ):
        rotate_qwen2_5_vl_attention_inputs(layer, Q_v, is_visual=True)
        rotate_qwen2_5_vl_attention_output(layer, Q_v, is_visual=True)
        rotate_qwen2_5_vl_mlp_input(layer, Q_v)
        rotate_qwen2_5_vl_mlp_output(layer, Q_v, False)

        rotate_qwen2_5_vl_ov_proj(
            layer,
            num_heads,
            head_dim,
            is_visual=True,
        )
    rotate_visual_merger(visual_model, Q_v)
    visual_model.to(raw_device)
    utils.cleanup_memory()


def rotate_visual_patch_pos_embed(model, Q_v: torch.Tensor):
    model.visual.pos_embed.weight.data = torch.matmul(model.visual.pos_embed.weight.data.double(), Q_v).to(model.visual.pos_embed.weight.dtype)


def rotate_qwen3_vl_visual_merger(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    dtype = model.visual.merger.linear_fc1.weight.dtype

    q_shape = Q.shape[0]
    o_shape, i_shape = model.visual.merger.linear_fc1.weight.shape

    W_ = (
        model.visual.merger.linear_fc1
        .weight.to(dtype=torch.float64)
        .reshape(o_shape, -1, q_shape)
    )
    model.visual.merger.linear_fc1.weight.data = (
        torch.matmul(W_, Q).to(dtype=dtype).reshape(o_shape, i_shape).contiguous()
    )


def rotate_qwen3_vl_deepstack_merger(model, Q: torch.Tensor) -> None:
    for layer in model.visual.deepstack_merger_list:
        dtype = layer.linear_fc1.weight.dtype
        q_shape = Q.shape[0]
        o_shape, i_shape = layer.linear_fc1.weight.shape
        W_ = layer.linear_fc1.weight.to(dtype=torch.float64).reshape(o_shape, -1, q_shape)
        layer.linear_fc1.weight.data = torch.matmul(W_, Q).to(dtype=dtype).reshape(o_shape, i_shape).contiguous()



def rotate_qwen3_vl_visual_model(model, model_type):
    raw_device = next(model.visual.parameters()).device
    model.visual.to(utils.DEV)

    num_heads = model.visual.blocks[0].attn.num_heads
    head_dim = model.visual.blocks[0].attn.qkv.in_features // num_heads
    Q_v = get_orthogonal_matrix(model.visual.blocks[0].attn.qkv.in_features, "hadamard")
    
    rotate_conv(
        model.visual.patch_embed.proj,
        Q_v,
        model.visual.blocks[0].attn.qkv.in_features,
    )

    rotate_visual_patch_pos_embed(model, Q_v)

    for idx, layer in enumerate(
        tqdm.tqdm(
            model.visual.blocks,
            unit="layer",
            desc="Rotating Qwen3 VL Visual",
        )
    ):
        rotate_attention_inputs(layer, Q_v, model_type)
        rotate_attention_output(layer, Q_v, model_type)
        rotate_mlp_input(layer, Q_v, model_type)
        rotate_mlp_output(layer, Q_v, model_type)

    rotate_qwen3_vl_visual_merger(model, Q_v)
    rotate_qwen3_vl_deepstack_merger(model, Q_v)
    model.visual.to(raw_device)
    utils.cleanup_memory()

def rotate_qwen2_5_vl_embeddings(model, Q) -> None:
    Q = Q.to(model.language_model.embed_tokens.weight.device)
    dtype = model.language_model.embed_tokens.weight.data.dtype
    W_ = model.language_model.embed_tokens.weight.data.to(dtype=torch.float64)
    model.language_model.embed_tokens.weight.data = torch.matmul(W_, Q).to(dtype=dtype)

    Q = Q.to(model.visual.merger.mlp[2].weight.device)
    W_ = model.visual.merger.mlp[2].weight.data.to(dtype=torch.float64)
    model.visual.merger.mlp[2].weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
    if model.visual.merger.mlp[2].bias is not None:
        b = model.visual.merger.mlp[2].bias.data.to(dtype=torch.float64)
        model.visual.merger.mlp[2].bias.data = torch.matmul(b, Q).to(dtype=dtype)


def rotate_qwen3_vl_embeddings(model, Q) -> None:
    Q = Q.to(model.language_model.embed_tokens.weight.device)
    dtype = model.language_model.embed_tokens.weight.data.dtype
    W_ = model.language_model.embed_tokens.weight.data.to(dtype=torch.float64)
    model.language_model.embed_tokens.weight.data = torch.matmul(W_, Q).to(dtype=dtype)

    Q = Q.to(model.visual.merger.linear_fc2.weight.device)
    W_ = model.visual.merger.linear_fc2.weight.data.to(dtype=torch.float64)
    model.visual.merger.linear_fc2.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
    if model.visual.merger.linear_fc2.bias is not None:
        b = model.visual.merger.linear_fc2.bias.data.to(dtype=torch.float64)
        model.visual.merger.linear_fc2.bias.data = torch.matmul(b, Q).to(dtype=dtype)

    for layer in model.visual.deepstack_merger_list:
        Q = Q.to(layer.linear_fc2.weight.device)
        W_ = layer.linear_fc2.weight.data.to(dtype=torch.float64)
        layer.linear_fc2.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
        if layer.linear_fc2.bias is not None:
            b = layer.linear_fc2.bias.data.to(dtype=torch.float64)
            layer.linear_fc2.bias.data = torch.matmul(b, Q).to(dtype=dtype)
        
def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    for W in model_utils.get_embeddings(model, model_type):
        raw_device = W.weight.device
        W = W.to(Q.device)
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(dtype=torch.float64)
        if W_.shape[-1] != Q.shape[0]:
            origin_shape = W_.shape
            W_ = W_.reshape(-1, Q.shape[0])
            W.weight.data = torch.matmul(W_, Q).to(dtype=dtype).to(raw_device).reshape(origin_shape)
        else:
            W.weight.data = torch.matmul(W_, Q).to(dtype=dtype).to(raw_device)


def rotate_attention_inputs(layer, Q, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    try:
        layer_list = [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
    except:
        layer_list = [layer.attn.qkv]

    for W in layer_list:
        dtype = W.weight.dtype
        W_ = W.weight.to(dtype=torch.float64)
        if W_.shape[-1] != Q.shape[0]:
            origin_shape = W_.shape
            W_ = W_.reshape(-1, Q.shape[0])
            W.weight.data = torch.matmul(W_, Q).to(dtype=dtype).reshape(origin_shape)
        else:
            W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)


def rotate_attention_output(layer, Q, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.self_attn.out_proj
    elif model_type in (model_utils.QWEN_MODEL, model_utils.QWEN3_MODEL, model_utils.QWEN2_5_VL_MODEL, model_utils.QWEN3_VL_MODEL):
        try:
            W = layer.self_attn.o_proj
        except:
            W = layer.attn.proj
    elif model_type == model_utils.QWEN3MOE_MODEL:
        W = layer.self_attn.o_proj
    else:
        raise ValueError(f"Unknown model type {model_type}")

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    if Q.shape[0] != W_.shape[0]:
        origin_shape = W_.shape
        W_ = W_.reshape(Q.shape[0], -1)
        W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype).reshape(origin_shape)
    else:
        W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)

    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float64)
        if Q.shape[0] != b.shape[0]:
            origin_shape = b.shape
            b = b.reshape(Q.shape[0], -1)
            W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype).reshape(origin_shape)
        else:
            W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)


def rotate_mlp_input(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_type == model_utils.LLAMA_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1]
    elif model_type in (model_utils.QWEN_MODEL, model_utils.QWEN3_MODEL, model_utils.QWEN2_5_VL_MODEL):
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.QWEN3_VL_MODEL:
        try:
            mlp_inputs = [layer.mlp.linear_fc1]
        except:
            mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.QWEN3MOE_MODEL:
        mlp_inputs = [layer.mlp.gate]
        for expert in layer.mlp.experts:
            mlp_inputs.append(expert.gate_proj)
            mlp_inputs.append(expert.up_proj)
    else:
        raise ValueError(f"Unknown model type {model_type}")

    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(dtype=torch.float64)
        if W_.shape[-1] != Q.shape[0]:
            origin_shape = W_.shape
            W_ = W_.reshape(-1, Q.shape[0])
            W.weight.data = torch.matmul(W_, Q).to(dtype=dtype).reshape(origin_shape)
        else:
            W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)


def rotate_mlp_output(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.QWEN3MOE_MODEL:
        for expert in layer.mlp.experts:
            W = expert.down_proj
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(dtype=torch.float64)
            if Q.shape[0] != W_.shape[0]:
                origin_shape = W_.shape
                W_ = W_.reshape(Q.shape[0], -1)
                W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype).reshape(origin_shape)
            else:
                W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)

            if W.bias is not None:
                b = W.bias.data.to(dtype=torch.float64)
                if Q.shape[0] != b.shape[0]:
                    origin_shape = b.shape
                    b = b.reshape(Q.shape[0], -1)
                    W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype).reshape(origin_shape)
                else:
                    W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)
        return

    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    elif model_type in (model_utils.QWEN_MODEL, model_utils.QWEN3_MODEL, model_utils.QWEN2_5_VL_MODEL):
        W = layer.mlp.down_proj
    elif model_type == model_utils.QWEN3_VL_MODEL:
        try:
            W = layer.mlp.linear_fc2
        except:
            W = layer.mlp.down_proj
    else:
        raise ValueError(f"Unknown model type {model_type}")

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    if Q.shape[0] != W_.shape[0]:
        origin_shape = W_.shape
        W_ = W_.reshape(Q.shape[0], -1)
        W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype).reshape(origin_shape)
    else:
        W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)

    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float64)
        if Q.shape[0] != b.shape[0]:
            origin_shape = b.shape
            b = b.reshape(Q.shape[0], -1)
            W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype).reshape(origin_shape)
        else:
            W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)


def matmul_hadU_cuda_had(X, hadK, transpose=False):
    """
    Apply hadamard transformation.
    It reshapes X and applies Walsh-Hadamard transform to the last dimension.
    Then, it will multiply the retult by another hadamard matrix.
    """
    from fast_hadamard_transform import hadamard_transform
    from hadamard_utils import get_had172

    n = X.shape[-1]
    K = hadK.shape[-1]

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.float().cuda().view(-1, K, n // K)
    input = hadamard_transform(input.contiguous(), scale=1 / math.sqrt(n))
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.to(X.device).to(X.dtype).reshape(X.shape)


def rotate_faster_down_proj(layer, model_type, hardK):
    from fast_hadamard_transform import hadamard_transform

    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
        dtype = W.weight.data.dtype
        W.weight.data = matmul_hadU_cuda_had(W.weight.data.float().cuda(), hardK)
        W.weight.data = W.weight.data.to(device="cpu", dtype=dtype)
    elif model_type == model_utils.QWEN3MOE_MODEL:
        for expert in layer.mlp.experts:
            W = expert.down_proj
            dtype = W.weight.data.dtype
            W.weight.data = matmul_hadU_cuda_had(W.weight.data.float().cuda(), hardK)
            W.weight.data = W.weight.data.to(device="cpu", dtype=dtype)
    else:
        raise ValueError(f"Faster MLP is onlu supported for LLaMa models!")


def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model, model_type=model_utils.model_type_extractor(model))
    raw_device = W.weight.device
    W = W.to(Q.device)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    if W_.shape[-1] != Q.shape[0]:
        origin_shape = W_.shape
        W_ = W_.reshape(-1, Q.shape[0])
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype).to(raw_device).reshape(origin_shape)
    else:
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype).to(raw_device)


def rotate_ov_proj(layer, model_type, head_num, head_dim):
    v_proj = layer.self_attn.v_proj
    if model_type == model_utils.LLAMA_MODEL:
        o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        o_proj = layer.self_attn.out_proj
    elif model_type == model_utils.QWEN_MODEL or model_type == model_utils.QWEN3_MODEL:
        o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.QWEN3MOE_MODEL:
        o_proj = layer.self_attn.o_proj
    else:
        raise ValueError(f"Unknown model type {model_type}")


@torch.inference_mode()
def rotate_model(model, rotate_mode, device, quarot_matrix_size=None):
    model_type = model_utils.model_type_extractor(model)

    llm_hidden_size = model.config.hidden_size if (model_type != model_utils.QWEN3_VL_MODEL) else model.config.text_config.hidden_size

    if quarot_matrix_size is None:
        Q = get_orthogonal_matrix(llm_hidden_size, rotate_mode, device=device)
    else:
        Q = torch.zeros(
            [llm_hidden_size, llm_hidden_size],
            dtype=torch.float64,
            device=device,
            requires_grad=False,
        )
        assert (
            int(llm_hidden_size / quarot_matrix_size) * quarot_matrix_size == llm_hidden_size
        ), "not fully dividen"
        for i in range(0, llm_hidden_size, quarot_matrix_size):
            local_Q = get_orthogonal_matrix(quarot_matrix_size, rotate_mode, device=device)
            Q[i : i + quarot_matrix_size, i : i + quarot_matrix_size] = local_Q

    config = model.config
    num_heads = config.num_attention_heads if (model_type != model_utils.QWEN3_VL_MODEL) else config.text_config.num_attention_heads
    model_dim = config.hidden_size  if (model_type != model_utils.QWEN3_VL_MODEL) else config.text_config.hidden_size
    if model_type != model_utils.QWEN3_VL_MODEL:
        head_dim = model_dim // num_heads
    else:
        if not hasattr(model.config.text_config, "head_dim"):
            head_dim = model_dim // num_heads
        else:
            head_dim = model.config.text_config.head_dim



    if model_type == model_utils.QWEN2_5_VL_MODEL:
        rotate_qwen2_5_vl_visual_model(model.model.visual)
    elif model_type == model_utils.QWEN3_VL_MODEL:
        rotate_qwen3_vl_visual_model(model.model, model_type)

    if model_type == model_utils.QWEN2_5_VL_MODEL:
        rotate_qwen2_5_vl_embeddings(model.model, Q)
    elif model_type == model_utils.QWEN3_VL_MODEL:
        rotate_qwen3_vl_embeddings(model.model, Q)
    else:
        rotate_embeddings(model, Q)
    rotate_head(model, Q)
    utils.cleanup_memory()

    layers = model_utils.get_transformer_layers(model, model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        raw_device = next(layer.parameters()).device
        # Move layer to target device in-place
        layer.to(device)
        rotate_attention_inputs(layer, Q, model_type)
        rotate_attention_output(layer, Q, model_type)
        rotate_mlp_input(layer, Q, model_type)
        rotate_mlp_output(layer, Q, model_type)
        # Move layer back to original device in-place
        layer.to(raw_device)
        # rotate_ov_proj(layers[idx], model_type, num_heads, head_dim)


@torch.inference_mode
def online_rotate(module, inp):
    x = torch.nn.functional.linear(inp[0], module.Q)
    return (x,) + inp[1:]


def register_online_rotation(module, Q: torch.Tensor):
    assert not hasattr(module, "Q")
    module.register_buffer("Q", Q.T.to(module.weight.data))  # Note F.linear(x, A) performs x@A.T

    # We use forward_pre_hook because we capture the input using forward_hook, which could then capture the rotated input.
    # If we implement in the forward() the un-rotated original input will be captured.
    module.rotate_handle = module.register_forward_pre_hook(online_rotate)


class QKRotationWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(head_dim), f"Only power of 2 head_dim is supported for K-cache Quantization!"
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs["k_groupsize"] in [
                -1,
                head_dim,
            ], f"Only token-wise/{head_dim}g quantization is supported for K-cache"
            self.k_bits = kwargs["k_bits"]
            self.k_groupsize = kwargs["k_groupsize"]
            self.k_sym = kwargs["k_sym"]
            self.k_clip_ratio = kwargs["k_clip_ratio"]
            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
            )

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = hadamard_transform(q.float(), scale=1 / math.sqrt(q.shape[-1])).to(dtype)
        k = hadamard_transform(k.float(), scale=1 / math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, self.config.hidden_size)
            self.k_quantizer.find_params(token_wise_k)
            k = self.k_quantizer(token_wise_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
        else:  # head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)

        self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """
    import functools

    import monkeypatch

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)
