import torch
import torch.nn as nn

from xhquant.nn import FX_LEAF_MODULES


def qlinear_cuda_old_converter(self: nn.Module):
    from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear as CudaOldQuantLinear

    assert isinstance(self, CudaOldQuantLinear)
    if self.bits in [2, 4, 8]:
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16 if self.bits == 8 else torch.int8)

        zeros = zeros + 1
        zeros = torch.bitwise_and(
            zeros, (2**self.bits) - 1
        )  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.

        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = self.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weight = torch.bitwise_right_shift(
            torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
            self.wf.unsqueeze(-1),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**self.bits) - 1)
        weight = weight.reshape(-1, self.group_size, weight.shape[2])
    elif self.bits == 3:
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = torch.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        )

        zeros = zeros + 1
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = self.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
        weight = weight.reshape(-1, self.group_size, weight.shape[2])
    else:
        raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    quant_weight = weight - zeros
    weight = scales * quant_weight
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    quant_weight = quant_weight.reshape(quant_weight.shape[0] * quant_weight.shape[1], quant_weight.shape[2])

    max_val = 2 ** (self.bits - 1)
    min_val = -max_val

    assert quant_weight.max() < max_val and quant_weight.min() >= min_val, f"{quant_weight.max()} {quant_weight}.min()"
    if hasattr(self, "qweight"):
        delattr(self, "qweight")
    if hasattr(self, "qzeros"):
        delattr(self, "qzeros")
    if hasattr(self, "scales"):
        delattr(self, "scales")
    if hasattr(self, "g_idx"):
        delattr(self, "g_idx")
    weight = weight.t()
    quant_weight = quant_weight.t()
    assert quant_weight.dtype in [torch.int8, torch.int16], (
        f"Expected quant_weight to be int8 or int16, but got {quant_weight.dtype}"
    )
    self.register_parameter("weight", nn.Parameter(weight))
    self.register_buffer("quant_weight", quant_weight)
    self.__class__ = nn.Linear
    self.out_features = self.outfeatures
    self.in_features = self.infeatures


def general_qlinear_converter(self: nn.Module):
    if self.bits in [2, 4, 8]:
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        zeros = torch.bitwise_and(zeros, (2**self.bits) - 1)

        zeros = zeros + 1
        zeros = zeros.reshape(self.scales.shape)

        weight = torch.bitwise_right_shift(
            torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
            self.wf.unsqueeze(-1),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**self.bits) - 1)
    elif self.bits == 3:
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = torch.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        )
        zeros = zeros + 1
        zeros = zeros.reshape(self.scales.shape)

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
    else:
        raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    # weights = self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()])

    quant_weight = weight - zeros[self.g_idx.long()]
    weight = self.scales[self.g_idx.long()] * quant_weight
    # weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    # quant_weight = quant_weight.reshape(quant_weight.shape[0] * quant_weight.shape[1], quant_weight.shape[2])

    maxq = (2**self.bits) / 2

    assert quant_weight.max() < maxq and quant_weight.min() >= -maxq, f"{quant_weight.max()} {quant_weight}.min()"
    if hasattr(self, "qweight"):
        delattr(self, "qweight")
    if hasattr(self, "qzeros"):
        delattr(self, "qzeros")
    if hasattr(self, "scales"):
        delattr(self, "scales")
    if hasattr(self, "g_idx"):
        delattr(self, "g_idx")
    weight = weight.t()
    quant_weight = quant_weight.t()
    self.register_parameter("weight", nn.Parameter(weight))
    self.register_buffer("quant_weight", quant_weight)
    self.__class__ = nn.Linear
    self.out_features = self.outfeatures
    self.in_features = self.infeatures


def ensure_gptqmodel_unpack_buffers(module: nn.Module) -> None:
    """Initialize unpack buffers that GPTQModel 5.8 creates lazily."""
    unpack_buffer_names = ("wf_unsqueeze_zero", "wf_unsqueeze_neg_one")
    if all(getattr(module, name, None) is not None for name in unpack_buffer_names):
        return

    init_unpack_buffers = getattr(module, "_init_wf_unsqueeze_buffers", None)
    if callable(init_unpack_buffers):
        init_unpack_buffers()

    missing = [name for name in unpack_buffer_names if getattr(module, name, None) is None]
    if missing:
        raise RuntimeError(f"{type(module).__name__} failed to initialize GPTQ unpack buffers: " + ", ".join(missing))


def gptqmodel_torch_qlinear_converter(self: nn.Module):
    import torch as t  # conflict with torch.py

    ensure_gptqmodel_unpack_buffers(self)
    if self.bits in [2, 4, 8]:
        zeros = t.bitwise_right_shift(
            t.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero,  # self.wf.unsqueeze(0),
        ).to(self.dequant_dtype)
        zeros = t.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)

        weight = t.bitwise_and(
            t.bitwise_right_shift(
                t.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one,  # self.wf.unsqueeze(-1)
            ).to(self.dequant_dtype),
            self.maxq,
        )
    elif self.bits == 3:
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf_unsqueeze_zero  # self.wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = t.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        ).reshape(self.scales.shape)

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf_unsqueeze_neg_one) & 0x7  # self.wf.unsqueeze(-1)
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = t.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    quant_weight = weight - zeros[self.g_idx.long()]
    weight = self.scales[self.g_idx.long()] * quant_weight
    maxq = 2 ** (self.bits - 1)
    # diff = quant_weight.to(torch.int32) - quant_weight
    # error = diff.abs().float()
    # assert torch.allclose(error, t.tensor(0.0), atol=1e-3), f"{error.max()}"
    assert quant_weight.max() < maxq and quant_weight.min() >= -maxq, (
        f"min={quant_weight.min()}, max={quant_weight.max()}, not in [{-maxq}, {maxq})"
    )
    if hasattr(self, "qweight"):
        delattr(self, "qweight")
    if hasattr(self, "qzeros"):
        delattr(self, "qzeros")
    if hasattr(self, "scales"):
        delattr(self, "scales")
    if hasattr(self, "g_idx"):
        delattr(self, "g_idx")
    weight = weight.t()
    quant_weight = quant_weight.t()
    self.register_parameter("weight", nn.Parameter(weight))
    self.register_buffer("quant_weight", quant_weight)
    self.__class__ = nn.Linear


def gptqmodel_torch_qlinear_packed_converter(module: nn.Module) -> nn.Module:
    """Adopt one canonical GPTQ QuantLinear without unpacking or copying it.

    The returned xhquant wrapper keeps GPTQ's int32 words as its sole weight
    representation.  Its XH2 QModule performs the W4/W8 -> HM signed-int8
    layout conversion later, when quantization reaches this module.
    """

    from xhquant.nn import GPTQPackedLinear, GPTQPackedWeight

    adapter = getattr(module, "adapter", None)
    if adapter is not None:
        raise NotImplementedError("GPTQ packed XH2 conversion does not support adapters")

    required = ("qweight", "qzeros", "scales", "g_idx")
    missing = [name for name in required if not torch.is_tensor(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f"GPTQ packed Linear is missing tensors: {missing}")

    qzero_format_getter = getattr(module, "qzero_format", None)
    qzero_format = int(qzero_format_getter()) if callable(qzero_format_getter) else 1
    if qzero_format != 2:
        raise ValueError(
            "GPTQ qzeros must be converted to canonical v2 before ownership transfer; "
            f"got v{qzero_format}"
        )

    devices = {getattr(module, name).device for name in required}
    bias = getattr(module, "bias", None)
    if torch.is_tensor(bias):
        devices.add(bias.device)
    if len(devices) != 1:
        raise ValueError(f"GPTQ packed tensors must be on one device, got {sorted(map(str, devices))}")

    packed_weight = GPTQPackedWeight(
        qweight=module.qweight,
        qzeros=module.qzeros,
        scales=module.scales,
        g_idx=module.g_idx,
        bits=int(module.bits),
        group_size=int(module.group_size),
        in_features=int(module.in_features),
        out_features=int(module.out_features),
        pack_dtype_bits=int(module.pack_dtype_bits),
        qzero_format=qzero_format,
        sym=bool(module.sym),
        desc_act=bool(module.desc_act),
    )
    packed_linear = GPTQPackedLinear(packed_weight, bias=bias)
    packed_linear.train(module.training)
    return packed_linear


def replace_gptqmodel_quant_linears_with_packed(model: nn.Module) -> tuple[str, ...]:
    """Replace every GPTQModel QuantLinear with an ownership-preserving wrapper."""

    from gptqmodel.nn_modules.qlinear import PackableQuantLinear

    module_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, PackableQuantLinear)
    ]
    for name in module_names:
        if not name:
            raise ValueError("the model root cannot itself be a GPTQ QuantLinear")
        module = model.get_submodule(name)
        model.set_submodule(name, gptqmodel_torch_qlinear_packed_converter(module))
    return tuple(module_names)


@FX_LEAF_MODULES.register_module()
class DequantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def __getattr__(self, name):
        if name == "weight":
            # Prefer an already materialized weight and skip lazy reconstruction.
            weight_param = self._parameters.get("weight") if hasattr(self, "_parameters") else None
            if weight_param is not None:
                return weight_param

            weight = self.quant_weight.to("meta") * self.scale_or_exp.to("meta")
            weight = torch.empty_like(weight, device=self.quant_weight.device)
            return weight

        return super().__getattr__(name)


def autoround_torch_qlinear_converter(self: nn.Module):
    quant_module_name = type(self).__module__
    add_one_to_zeros = quant_module_name.endswith("qlinear_torch_zp")
    old_device = None
    try:
        old_device = next(iter(self.parameters())).device  # make sure parameters are initialized
    except StopIteration:
        pass

    if old_device is None:
        try:
            old_device = next(iter(self.buffers())).device  # try buffers if no parameters
        except StopIteration:
            pass
    if old_device is None:
        old_device = torch.device("cpu")  # default to CPU if no parameters or buffers
    device = torch.device("cuda") if torch.cuda.is_available() else old_device
    self.to(device)
    if self.bits in [2, 4, 8]:
        if self.wf.device != self.qzeros.device:
            self.wf = torch.tensor(
                list(range(0, 32, self.bits)), dtype=torch.int32, device=self.qzeros.device
            ).unsqueeze(0)
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(self.dequant_dtype)
        zeros = torch.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)

        weight = torch.bitwise_and(
            torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
                self.wf.unsqueeze(-1),
            ).to(self.dequant_dtype),
            self.maxq,
        )
    elif self.bits == 3:
        if self.wf.device != self.qzeros.device:
            wf_3bits = [
                [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
            ]
            self.wf = torch.tensor(wf_3bits, dtype=torch.int32, device=self.qzeros.device).reshape(1, 3, 12)

        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = torch.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        ).reshape(self.scales.shape)

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
    else:
        raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    if add_one_to_zeros:
        zeros = zeros + 1

    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    if hasattr(self, "g_idx"):
        quant_weight = weight - zeros[self.g_idx.long()]
        dense_weight = self.scales[self.g_idx.long()] * quant_weight
        scale_or_exp = self.scales[self.g_idx.long()].contiguous()
    else:
        repeat_scales = self.scales.repeat_interleave(self.group_size, dim=0)
        repeat_zeros = zeros.repeat_interleave(self.group_size, dim=0)
        quant_weight = weight - repeat_zeros
        dense_weight = repeat_scales * quant_weight
        scale_or_exp = repeat_scales.contiguous()

    if hasattr(self, "qweight"):
        delattr(self, "qweight")
    if hasattr(self, "qzeros"):
        delattr(self, "qzeros")
    if hasattr(self, "scales"):
        delattr(self, "scales")
    if hasattr(self, "g_idx"):
        delattr(self, "g_idx")

    dense_weight = dense_weight.t().contiguous()
    quant_weight = quant_weight.t().contiguous()
    scale_or_exp = scale_or_exp.t().contiguous()

    self.register_parameter("weight", nn.Parameter(dense_weight, requires_grad=False))
    self.register_buffer("quant_weight", quant_weight)
    # self.register_buffer("scale_or_exp", scale_or_exp)
    self.__class__ = nn.Linear
    # self.__class__ = DequantLinear
    self.out_features = self.outfeatures
    self.in_features = self.infeatures
    self.to(old_device)


def restore_autoround_qwen3_5_moe_sparse_block(module: nn.Module) -> nn.Module:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeExperts,
        Qwen3_5MoeSparseMoeBlock,
    )

    restored = object.__new__(Qwen3_5MoeSparseMoeBlock)
    nn.Module.__init__(restored)

    restored.gate = module.gate
    if isinstance(module.experts, nn.ModuleList):
        if not module.experts:
            raise ValueError("LinearQwen3_5MoeSparseMoeBlock must contain at least one expert.")
        first_expert = module.experts[0]
        gate_rows = first_expert.gate_proj.weight.shape[0]
        up_rows = first_expert.up_proj.weight.shape[0]
        gate_up_proj = first_expert.gate_proj.weight.new_empty(
            (len(module.experts), gate_rows + up_rows, *first_expert.gate_proj.weight.shape[1:])
        )
        down_proj = first_expert.down_proj.weight.new_empty((len(module.experts), *first_expert.down_proj.weight.shape))
        with torch.no_grad():
            for expert_idx, expert in enumerate(module.experts):
                gate_up_proj[expert_idx, :gate_rows].copy_(expert.gate_proj.weight)
                gate_up_proj[expert_idx, gate_rows:].copy_(expert.up_proj.weight)
                down_proj[expert_idx].copy_(expert.down_proj.weight)
        restored_experts = object.__new__(Qwen3_5MoeExperts)
        nn.Module.__init__(restored_experts)
        restored_experts.num_experts = len(module.experts)
        restored_experts.hidden_dim = first_expert.gate_proj.weight.shape[1]
        restored_experts.intermediate_dim = gate_rows
        restored_experts.config = first_expert.config
        restored_experts.act_fn = first_expert.act_fn
        restored_experts.gate_up_proj = nn.Parameter(
            gate_up_proj,
            requires_grad=any(
                expert.gate_proj.weight.requires_grad or expert.up_proj.weight.requires_grad
                for expert in module.experts
            ),
        )
        restored_experts.down_proj = nn.Parameter(
            down_proj,
            requires_grad=any(expert.down_proj.weight.requires_grad for expert in module.experts),
        )
        restored.experts = restored_experts
    else:
        restored.experts = module.experts
    restored.shared_expert = module.shared_expert
    restored.shared_expert_gate = module.shared_expert_gate
    restored.train(module.training)
    return restored
