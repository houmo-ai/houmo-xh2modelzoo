import torch
import torch.nn as nn
from transformers import T5EncoderModel
from transformers.models.t5.modeling_t5 import T5Stack


def get_hadmard_matrix(hidden_size, quarot_matrix_size=None, rotate_mode="hadamard", device=None):
    from hadamard_utils import random_hadamard_matrix

    if quarot_matrix_size is None:
        Q = random_hadamard_matrix(hidden_size, device=device)
    else:
        Q = torch.zeros(
            [hidden_size, hidden_size],
            dtype=torch.float64,
            device=device,
            requires_grad=False,
        )
        assert int(hidden_size / quarot_matrix_size) * quarot_matrix_size == hidden_size, "not fully dividen"
        for i in range(0, hidden_size, quarot_matrix_size):
            local_Q = random_hadamard_matrix(quarot_matrix_size, device=device)
            Q[i : i + quarot_matrix_size, i : i + quarot_matrix_size] = local_Q

    return Q


def rotation_weight(layer, Q, with_bias=False, transpose=False):
    dtype = layer.weight.data.dtype
    W = layer.weight.data.to(dtype=torch.float64)
    if transpose:
        layer.weight.data = torch.matmul(Q.T, W).to(dtype=dtype)
    else:
        layer.weight.data = torch.matmul(W, Q).to(dtype=dtype)

    if with_bias and layer.bias is not None:
        b = layer.bias.data.to(dtype=torch.float64)
        if transpose:
            layer.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)
        else:
            layer.bias.data = torch.matmul(b, Q).to(dtype=dtype)


def fuse_layernorm_weight(linear, ln):
    assert hasattr(ln, "weight")
    assert hasattr(linear, "weight")
    linear_dtype = linear.weight.dtype
    W = linear.weight.data.double()
    linear.weight.data = (W * ln.weight.double()).to(linear_dtype)
    # ln.weight.fill_(1.)

    if hasattr(ln, "bias") and ln.bias is not None:
        if linear.bias is None:
            linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
        linear.bias.data = linear.bias.data.double() + torch.matmul(W, ln.bias.double())
        linear.bias.data = linear.bias.data.to(linear_dtype)
        # ln.bias.fill_(0.)


def hadmard_t5(t5: T5EncoderModel):
    if t5 is None:
        return
    assert isinstance(t5, T5EncoderModel), "t5 must be T5EncoderModel, but get {}".format(type(t5))

    # t5 = t5.to(torch.float32)
    for name, module in t5.named_modules():
        if isinstance(module, T5Stack):
            hidden_size = module.embed_tokens.weight.shape[-1]
            device = module.embed_tokens.weight.device
            Q = get_hadmard_matrix(hidden_size, device=device)

            # online_rotation = "pre_norm"
            online_rotation = "after_norm"
            if online_rotation == "pre_norm":
                module.add_module(
                    "online_rotation",
                    nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False, device=device),
                )
                module.online_rotation.weight.data = Q.to(torch.float32)
            else:
                module.add_module(
                    "online_rotation_after_norm",
                    nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False, device=device),
                )
                module.online_rotation_after_norm.weight.fill_(0.0)
                for i in range(hidden_size):
                    module.online_rotation_after_norm.weight[i][i].fill_(1.0)
                # module.online_rotation_after_norm.weight.data = Q.to(torch.float32)

                fuse_layernorm_weight(module.online_rotation_after_norm, module.final_layer_norm)
                if hasattr(module.final_layer_norm, "weight"):
                    module.final_layer_norm.weight.fill_(1.0)
                if hasattr(module.final_layer_norm, "bias") and module.final_layer_norm.bias is not None:
                    module.final_layer_norm.bias.fill_(0.0)

                rotation_weight(module.online_rotation_after_norm, Q)

            rotation_last_ffn_wo = False
            if rotation_last_ffn_wo:
                rotation_weight(module.block[-1].layer[1].DenseReluDense.wo, Q, with_bias=True, transpose=True)

                module.block[-1].layer[-1].add_module(
                    "online_rotation",
                    nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False, device=device),
                )
                module.block[-1].layer[-1].online_rotation.weight.data = Q.t().to(torch.float32)
                continue

            rotation_last_attn_wo = False
            if rotation_last_attn_wo:
                module.block[-1].layer[0].add_module(
                    "online_rotation",
                    nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False, device=device),
                )
                module.block[-1].layer[0].online_rotation.weight.data = Q.t().to(torch.float32)

                # module.block[-1].layer[0].add_module(
                #     "online_rotation_debug",
                #     nn.Linear(
                #         in_features=hidden_size,
                #         out_features=hidden_size,
                #         bias=False,
                #         device=device)
                # )
                # module.block[-1].layer[0].online_rotation_debug.weight.data = Q.to(torch.float32)

                # module.block[-1].layer[1].add_module(
                #     "online_rotation_debug",
                #     nn.Linear(
                #         in_features=hidden_size,
                #         out_features=hidden_size,
                #         bias=False,
                #         device=device)
                # )
                # module.block[-1].layer[1].online_rotation_debug.weight.data = Q.to(torch.float32)

                module.block[-1].layer[0].SelfAttention.o = module.block[-1].layer[0].SelfAttention.o.to(torch.float32)
                rotation_weight(module.block[-1].layer[0].SelfAttention.o, Q, with_bias=True, transpose=True)

                module.block[-1].layer[1].DenseReluDense.wi_0 = (
                    module.block[-1].layer[1].DenseReluDense.wi_0.to(torch.float32)
                )
                module.block[-1].layer[1].DenseReluDense.wi_1 = (
                    module.block[-1].layer[1].DenseReluDense.wi_1.to(torch.float32)
                )
                # fuse RMSnorm weight
                fuse_layernorm_weight(
                    module.block[-1].layer[1].DenseReluDense.wi_0, module.block[-1].layer[1].layer_norm
                )
                fuse_layernorm_weight(
                    module.block[-1].layer[1].DenseReluDense.wi_1, module.block[-1].layer[1].layer_norm
                )
                if hasattr(module.block[-1].layer[1].layer_norm, "weight"):
                    module.block[-1].layer[1].layer_norm.weight.fill_(1.0)
                if hasattr(module.block[-1].layer[1].layer_norm, "bias"):
                    module.block[-1].layer[1].layer_norm.bias.fill_(0.0)

                rotation_weight(module.block[-1].layer[1].DenseReluDense.wi_0, Q)
                rotation_weight(module.block[-1].layer[1].DenseReluDense.wi_1, Q)

                rotation_weight(module.block[-1].layer[1].DenseReluDense.wo, Q, with_bias=True, transpose=True)

            rotation_all = True
            if rotation_all:
                rotation_weight(module.embed_tokens, Q)
                for i, target in enumerate(module.block):
                    fuse_layernorm_weight(target.layer[0].SelfAttention.q, target.layer[0].layer_norm)
                    fuse_layernorm_weight(target.layer[0].SelfAttention.k, target.layer[0].layer_norm)
                    fuse_layernorm_weight(target.layer[0].SelfAttention.v, target.layer[0].layer_norm)
                    if hasattr(target.layer[0].layer_norm, "weight"):
                        target.layer[0].layer_norm.weight.fill_(1.0)
                    if hasattr(target.layer[0].layer_norm, "bias") and target.layer[0].layer_norm.bias is not None:
                        target.layer[0].layer_norm.bias.fill_(0.0)

                    rotation_weight(target.layer[0].SelfAttention.q, Q)
                    rotation_weight(target.layer[0].SelfAttention.k, Q)
                    rotation_weight(target.layer[0].SelfAttention.v, Q)
                    rotation_weight(target.layer[0].SelfAttention.o, Q, with_bias=True, transpose=True)

                    fuse_layernorm_weight(target.layer[1].DenseReluDense.wi_0, target.layer[1].layer_norm)
                    fuse_layernorm_weight(target.layer[1].DenseReluDense.wi_1, target.layer[1].layer_norm)
                    if hasattr(target.layer[1].layer_norm, "weight"):
                        target.layer[1].layer_norm.weight.fill_(1.0)
                    if hasattr(target.layer[1].layer_norm, "bias") and target.layer[1].layer_norm.bias is not None:
                        target.layer[1].layer_norm.bias.fill_(0.0)

                    rotation_weight(target.layer[1].DenseReluDense.wi_0, Q)
                    rotation_weight(target.layer[1].DenseReluDense.wi_1, Q)
                    rotation_weight(target.layer[1].DenseReluDense.wo, Q, with_bias=True, transpose=True)

            force_fp16 = True
            if force_fp16:
                pass
                # if hasattr(module, "online_rotation"):
                #     module.online_rotation = module.online_rotation.to(torch.float16)
                #     module.online_rotation = module.online_rotation.to(torch.float32)

                # target = module.block[-1]
                # target.layer[1].DenseReluDense.wi_0 = target.layer[1].DenseReluDense.wi_0.to(torch.float16)
                # target.layer[1].DenseReluDense.wi_1 = target.layer[1].DenseReluDense.wi_1.to(torch.float16)
                # target.layer[1].DenseReluDense.wo = target.layer[1].DenseReluDense.wo.to(torch.float16)

                # target.layer[1].DenseReluDense.wi_0 = target.layer[1].DenseReluDense.wi_0.to(torch.float32)
                # target.layer[1].DenseReluDense.wi_1 = target.layer[1].DenseReluDense.wi_1.to(torch.float32)
                # target.layer[1].DenseReluDense.wo = target.layer[1].DenseReluDense.wo.to(torch.float32)

                # target.layer[0].SelfAttention.q = target.layer[0].SelfAttention.q.to(torch.float16)
                # target.layer[0].SelfAttention.k = target.layer[0].SelfAttention.k.to(torch.float16)
                # target.layer[0].SelfAttention.v = target.layer[0].SelfAttention.v.to(torch.float16)
                # target.layer[0].SelfAttention.o = target.layer[0].SelfAttention.o.to(torch.float16)

                for i, target in enumerate(module.block):
                    # target = target.to(torch.float16)
                    target.layer[1].DenseReluDense.wo = target.layer[1].DenseReluDense.wo.to(torch.float16)
                    # target.layer[1].DenseReluDense.wo = target.layer[1].DenseReluDense.wo.to(torch.float32)
