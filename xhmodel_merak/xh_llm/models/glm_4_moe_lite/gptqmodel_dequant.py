import importlib.util
import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from tqdm import tqdm

# 本文件只服务于 glm_4_moe_lite 的 GPTQModel checkpoint 导出链路。
# 这里放的是“本模型目录内可复用”的 qlinear -> nn.Linear 反量化工具，
# 不向 xhmodel_merak/xh_llm/ 公共层泄漏能力。
#
# =============================================================================
# OBSOLETE AFTER GPTQMODEL HOOK REFACTOR
# 该文件中的 GLM 本地反量化实现当前不再被导出链路调用。
# 当前路径统一使用 xhmodel_merak/xh_llm/base_model.py 中的公共 GPTQModel
# 反量化逻辑；这里暂时保留仅作回滚/对照，不要在新代码中继续调用。
# =============================================================================


def _is_gptqmodel_available() -> bool:
    try:
        from transformers.utils import is_gptqmodel_available

        return bool(is_gptqmodel_available())
    except Exception:
        return importlib.util.find_spec("gptqmodel") is not None


def _mark_dequantized_linear_origin(module: nn.Module, origin: str):
    module._xhquant_weight_origin = origin


def _get_linear_features(module: nn.Module, weight: torch.Tensor) -> tuple[int, int]:
    in_features = getattr(module, "in_features", getattr(module, "infeatures", weight.shape[1]))
    out_features = getattr(module, "out_features", getattr(module, "outfeatures", weight.shape[0]))
    return int(in_features), int(out_features)


def _finalize_module_as_linear(
    module: nn.Module,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    quant_weight: torch.Tensor | None = None,
    in_features: int | None = None,
    out_features: int | None = None,
    force_cpu: bool = False,
):
    # 通用收口逻辑：把各种 qlinear 最终恢复为标准 nn.Linear。
    if force_cpu:
        weight = weight.detach().to("cpu")
        if bias is not None:
            bias = bias.detach().to("cpu")
        if quant_weight is not None:
            quant_weight = quant_weight.detach().to("cpu")

    module.__class__ = nn.Linear
    module.in_features = int(in_features if in_features is not None else weight.shape[1])
    module.out_features = int(out_features if out_features is not None else weight.shape[0])

    module._parameters.pop("weight", None)
    module._parameters["weight"] = nn.Parameter(weight, requires_grad=False)

    if bias is None:
        module._parameters.pop("bias", None)
        module.bias = None
    else:
        module._parameters.pop("bias", None)
        module._parameters["bias"] = nn.Parameter(bias, requires_grad=False)

    if quant_weight is not None:
        module._buffers["quant_weight"] = quant_weight
    else:
        module._buffers.pop("quant_weight", None)


def _drop_quant_attrs(module: nn.Module):
    for attr_name in (
        "qweight",
        "qzeros",
        "scales",
        "g_idx",
        "wf",
        "wf_unsqueeze_zero",
        "wf_unsqueeze_neg_one",
        "quant_weight",
    ):
        if hasattr(module, attr_name):
            delattr(module, attr_name)


def _convert_float_backed_gptq_linear(
    module: nn.Module,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    # 某些 GPTQModel 槽位虽然模块类型还是 qlinear，但 checkpoint 里实际存的是浮点 weight。
    _drop_quant_attrs(module)
    _finalize_module_as_linear(
        module,
        weight,
        bias=bias,
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        force_cpu=True,
    )
    _mark_dequantized_linear_origin(module, "float")


def gptqmodel_torch_qlinear_converter(self: nn.Module, keep_quant_weight: bool = False):
    # glm_4_moe_lite 默认不保留 quant_weight，避免大 MoE 模型在导出前出现双份权重常驻内存。
    import torch as t

    if self.bits in [2, 4, 8]:
        zeros = t.bitwise_right_shift(
            t.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero,
        ).to(self.dequant_dtype)
        zeros = t.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)

        weight = t.bitwise_and(
            t.bitwise_right_shift(
                t.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one,
            ).to(self.dequant_dtype),
            self.maxq,
        )
    elif self.bits == 3:
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf_unsqueeze_zero
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = t.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        ).reshape(self.scales.shape)

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf_unsqueeze_neg_one) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = t.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
    else:
        raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    if weight.dtype != self.scales.dtype:
        weight = weight.to(self.scales.dtype)

    quant_weight = t.empty_like(weight) if keep_quant_weight else None
    idx = self.g_idx.long()
    maxq = 2 ** (self.bits - 1)
    chunk_rows = 4096
    for start in range(0, weight.shape[0], chunk_rows):
        end = min(start + chunk_rows, weight.shape[0])
        chunk_idx = idx[start:end]
        chunk_weight = weight[start:end]
        chunk_weight.sub_(zeros.index_select(0, chunk_idx).to(chunk_weight.dtype))
        assert chunk_weight.max() < maxq and chunk_weight.min() >= -maxq, (
            f"min={chunk_weight.min()}, max={chunk_weight.max()}, not in [{-maxq}, {maxq})"
        )
        if quant_weight is not None:
            quant_weight[start:end].copy_(chunk_weight)
        chunk_weight.mul_(self.scales.index_select(0, chunk_idx).to(chunk_weight.dtype))
    del zeros
    del idx

    weight = weight.t().contiguous()
    quant_weight = quant_weight.t().contiguous() if quant_weight is not None else None
    in_features, out_features = _get_linear_features(self, weight)
    _drop_quant_attrs(self)
    _finalize_module_as_linear(
        self,
        weight,
        quant_weight=quant_weight,
        in_features=in_features,
        out_features=out_features,
    )
    _mark_dequantized_linear_origin(self, "gptq")


def get_hf_checkpoint_weight_map(hf_model_dir: str | Path) -> dict[str, str]:
    model_dir = Path(hf_model_dir)
    weight_map: dict[str, str] = {}
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, "r") as f:
            data = json.load(f)
        for key, rel_path in data.get("weight_map", {}).items():
            weight_map[key] = str(model_dir / rel_path)
        return weight_map

    safetensors_file = model_dir / "model.safetensors"
    if safetensors_file.exists():
        with safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = str(safetensors_file)
    return weight_map


def load_hf_checkpoint_tensor(tensor_name: str, weight_map: dict[str, str]) -> torch.Tensor | None:
    file_path = weight_map.get(tensor_name)
    if file_path is None:
        return None
    with safe_open(file_path, framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name)


def _gptqmodel_packable_qlinear_classes() -> tuple[type[nn.Module], ...]:
    try:
        from gptqmodel.nn_modules.qlinear import PackableQuantLinear

        return (PackableQuantLinear,)
    except Exception:
        return ()


# =============================================================================
# OBSOLETE PUBLIC ENTRY
# dequantize_gptqmodel_linears() 是旧 GLM 本地 GPTQModel 反量化入口。
# 当前导出链路改为调用 base_model.py 中的公共反量化逻辑。
# =============================================================================
def dequantize_gptqmodel_linears(
    hf_model: nn.Module,
    hf_model_dir: str | Path,
    *,
    keep_quant_weight: bool = False,
) -> nn.Module:
    # 本模型目录内的公共入口：
    # 只适用于“模块拓扑不再额外变化”的 qlinear 反量化。
    # GLM 的 split MoE 恢复由 gptqmodel_compat.convert_gptqmodel_moe_structure 负责。
    if not _is_gptqmodel_available():
        raise ImportError("gptqmodel is required for glm_4_moe_lite GPTQModel dequantization.")

    qlinear_classes = _gptqmodel_packable_qlinear_classes()
    if len(qlinear_classes) == 0:
        return hf_model

    dequant_linears: list[tuple[str, nn.Module]] = []
    for name, module in hf_model.named_modules():
        if isinstance(module, qlinear_classes):
            dequant_linears.append((name, module))

    if len(dequant_linears) == 0:
        return hf_model

    checkpoint_weight_map = get_hf_checkpoint_weight_map(hf_model_dir)
    pbar = tqdm(dequant_linears, desc="Dequantizing GLM GPTQModel linears")
    for name, module in pbar:
        pbar.set_description(f"Dequantizing GPTQModel: {name}")
        float_weight_name = f"{name}.weight"
        quant_weight_name = f"{name}.qweight"
        if float_weight_name in checkpoint_weight_map and quant_weight_name not in checkpoint_weight_map:
            float_weight = load_hf_checkpoint_tensor(float_weight_name, checkpoint_weight_map)
            bias = load_hf_checkpoint_tensor(f"{name}.bias", checkpoint_weight_map)
            assert float_weight is not None, f"Missing checkpoint tensor for {float_weight_name}"
            _convert_float_backed_gptq_linear(module, float_weight, bias)
            continue
        gptqmodel_torch_qlinear_converter(module, keep_quant_weight=keep_quant_weight)

    hf_model.quantization_method = None  # type: ignore[attr-defined]
    hf_model._is_hf_initialized = False  # type: ignore[attr-defined]
    return hf_model
