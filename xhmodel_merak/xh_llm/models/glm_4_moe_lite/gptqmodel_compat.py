import gc
import json
import logging
import re
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 本文件只放 GLM-4.7 Flash / Glm4MoeLite 的 GPTQModel checkpoint 兼容逻辑。
# 可复用边界：
#   1. detect_gptqmodel_moe_format 这类“检测 checkpoint 特殊格式”的函数应由新模型自己实现。
#   2. glm_gptqmodel_load_context 展示了“模型私有 get_hf_model 覆盖里如何包一层加载上下文”的写法：
#      临时 patch，finally 恢复。
#   3. convert_gptqmodel_moe_structure 是 GLM 专属结构转换，不应直接复用到其他 MoE 模型。


# =============================================================================
# OBSOLETE AFTER GPTQMODEL HOOK REFACTOR
# 以下加载期兼容链路不再被 GLM 导出路径调用：
#   - _ensure_transformers_no_init_weights_compat
#   - _get_quant_cfg_attr
#   - _build_gptq_expert_linear
#   - _Glm4MoeLiteNaiveMoeCompat 及其辅助函数
#   - detect_gptqmodel_moe_format
#   - _load_gptqmodel_moe_expert_layout
#   - glm_gptqmodel_load_context
# 当前路径统一走 XHBaseModel._load_gptqmodel() 和公共 GPTQModel 反量化逻辑。
# 暂时保留这些实现仅作回滚/对照，不要在新代码中继续调用。
# =============================================================================
def _ensure_transformers_no_init_weights_compat() -> None:
    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils, "no_init_weights"):
        return

    try:
        from transformers import initialization as init

        modeling_utils.no_init_weights = init.no_init_weights
        return
    except Exception:
        pass

    @contextmanager
    def no_init_weights(_enable: bool = True):
        init_flag = getattr(modeling_utils, "_init_weights", None)
        if init_flag is not None:
            modeling_utils._init_weights = False
        try:
            yield
        finally:
            if init_flag is not None:
                modeling_utils._init_weights = init_flag

    modeling_utils.no_init_weights = no_init_weights


def _get_quant_cfg_attr(config, key: str, default):
    qcfg = getattr(config, "quantization_config", None)
    if qcfg is None:
        return default
    if isinstance(qcfg, dict):
        return qcfg.get(key, default)
    return getattr(qcfg, key, default)


def _build_gptq_expert_linear(config, in_features: int, out_features: int) -> nn.Module:
    bits = int(_get_quant_cfg_attr(config, "bits", 4))
    group_size = int(_get_quant_cfg_attr(config, "group_size", 128))
    if group_size <= 0:
        group_size = in_features
    sym = bool(_get_quant_cfg_attr(config, "sym", True))
    desc_act = bool(_get_quant_cfg_attr(config, "desc_act", False))
    try:
        from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

        return TorchQuantLinear(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            register_buffers=True,
        )
    except Exception as e:
        try:
            from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedQuantLinear

            return TorchFusedQuantLinear(
                bits=bits,
                group_size=group_size,
                sym=sym,
                desc_act=desc_act,
                in_features=in_features,
                out_features=out_features,
                bias=False,
                register_buffers=True,
            )
        except Exception:
            pass
        logger.warning("Falling back to nn.Linear for GPTQModel expert branch: %s", e)
        return nn.Linear(in_features, out_features, bias=False)


def resolve_native_glm_moe_cls() -> type[nn.Module]:
    import transformers.models.glm4_moe_lite.modeling_glm4_moe_lite as glm_module

    return glm_module.Glm4MoeLiteNaiveMoe


class _Glm4MoeLiteNaiveMoeCompat(nn.Module):
    # 仅用于加载 split MoE GPTQModel checkpoint 的临时替代类。
    # 它的职责是让 GPTQModel.load() 能按 checkpoint key 构造 gate/up/down 三组专家模块；
    # 加载完成并通用反量化后，会在 convert_gptqmodel_moe_structure 中恢复为原生 MoE 类。
    _expert_slot_kind: dict[tuple[int, str, int], str] = {}
    _sparse_layer_ids: list[int] = []
    _layer_cursor: int = 0

    @classmethod
    def configure_layout(
        cls,
        sparse_layer_ids: list[int],
        expert_slot_kind: dict[tuple[int, str, int], str],
    ) -> None:
        cls._sparse_layer_ids = list(sparse_layer_ids)
        cls._expert_slot_kind = dict(expert_slot_kind)
        cls._layer_cursor = 0

    @classmethod
    def reset_layout(cls) -> None:
        cls._sparse_layer_ids = []
        cls._expert_slot_kind = {}
        cls._layer_cursor = 0

    @classmethod
    def _consume_layer_id(cls, config) -> int:
        if cls._layer_cursor < len(cls._sparse_layer_ids):
            layer_id = cls._sparse_layer_ids[cls._layer_cursor]
        else:
            dense_prefix = int(getattr(config, "first_k_dense_replace", 1))
            layer_id = dense_prefix + cls._layer_cursor
        cls._layer_cursor += 1
        return layer_id

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = nn.SiLU()
        self.layer_id = self.__class__._consume_layer_id(config)

        self.gate_proj = nn.ModuleList(
            [
                self._build_expert_proj("gate_proj", self.hidden_dim, self.intermediate_dim, expert_idx)
                for expert_idx in range(self.num_experts)
            ]
        )
        self.up_proj = nn.ModuleList(
            [
                self._build_expert_proj("up_proj", self.hidden_dim, self.intermediate_dim, expert_idx)
                for expert_idx in range(self.num_experts)
            ]
        )
        self.down_proj = nn.ModuleList(
            [
                self._build_expert_proj("down_proj", self.intermediate_dim, self.hidden_dim, expert_idx)
                for expert_idx in range(self.num_experts)
            ]
        )

    def _build_expert_proj(
        self,
        proj_name: str,
        in_features: int,
        out_features: int,
        expert_idx: int,
    ) -> nn.Module:
        slot = (self.layer_id, proj_name, expert_idx)
        slot_kind = self.__class__._expert_slot_kind.get(slot, "quant")
        if slot_kind == "fp":
            return nn.Linear(in_features, out_features, bias=False)
        return _build_gptq_expert_linear(self.config, in_features, out_features)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate = self.gate_proj[expert_idx](current_state)
            up = self.up_proj[expert_idx](current_state)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self.down_proj[expert_idx](current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


def detect_gptqmodel_moe_format(model_dir: str | Path) -> bool:
    # GLM 专属格式检测：gptqmodel 保存的 split expert key 形如
    # model.layers.<n>.mlp.experts.gate_proj.<expert>.qweight。
    # 新增其他模型时，应实现自己的 detect_<model>_gptqmodel_format，而不是复用这个判断。
    model_dir = Path(model_dir)
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        try:
            with open(index_file) as f:
                weight_map = json.load(f).get("weight_map", {})
            return any(".experts.gate_proj." in key for key in weight_map)
        except Exception:
            return False

    safetensors_file = model_dir / "model.safetensors"
    if safetensors_file.exists():
        try:
            from safetensors import safe_open

            with safe_open(str(safetensors_file), framework="pt") as f:
                return any(".experts.gate_proj." in key for key in f.keys())
        except Exception:
            return False
    return False


def _load_gptqmodel_moe_expert_layout(
    model_dir: str | Path,
) -> tuple[list[int], dict[tuple[int, str, int], str]]:
    # GLM 专属 expert 槽位解析。
    # 作用是区分 quant slot 与 float-backed slot，保证 compat 类构造出的模块类型能匹配 checkpoint。
    model_dir = Path(model_dir)
    index_file = model_dir / "model.safetensors.index.json"
    if not index_file.exists():
        return [], {}

    try:
        with open(index_file) as f:
            weight_map = json.load(f).get("weight_map", {})
    except Exception as e:
        logger.warning("Failed to parse %s: %s", index_file, e)
        return [], {}

    pattern = re.compile(
        r"model\.layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.(\d+)\.(weight|qweight|qzeros|scales|g_idx)$"
    )
    slot_attrs: dict[tuple[int, str, int], set[str]] = {}
    sparse_layers: set[int] = set()
    for key in weight_map:
        match = pattern.match(key)
        if match is None:
            continue
        layer_id = int(match.group(1))
        proj_name = match.group(2)
        expert_id = int(match.group(3))
        attr_name = match.group(4)
        sparse_layers.add(layer_id)
        slot = (layer_id, proj_name, expert_id)
        slot_attrs.setdefault(slot, set()).add(attr_name)

    slot_kind: dict[tuple[int, str, int], str] = {}
    for slot, attrs in slot_attrs.items():
        if attrs == {"weight"}:
            slot_kind[slot] = "fp"
        elif "qweight" in attrs:
            slot_kind[slot] = "quant"

    return sorted(sparse_layers), slot_kind


@contextmanager
def glm_gptqmodel_load_context(hf_model_dir: str | Path, **kwargs):
    # XHGlm4MoeLiteModel.get_hf_model 的加载期上下文。
    # 重要约束：所有 monkey patch 必须只包住 GPTQModel.load()，并在 finally 中恢复。
    # 后续新模型如果也需要 patch transformers 类，应按这个 contextmanager 模式实现。
    del kwargs
    _ensure_transformers_no_init_weights_compat()
    if not detect_gptqmodel_moe_format(hf_model_dir):
        yield
        return

    sparse_layer_ids, expert_slot_kind = _load_gptqmodel_moe_expert_layout(hf_model_dir)
    _Glm4MoeLiteNaiveMoeCompat.configure_layout(sparse_layer_ids, expert_slot_kind)

    import transformers.models.glm4_moe_lite.modeling_glm4_moe_lite as glm_module

    original_cls = glm_module.Glm4MoeLiteNaiveMoe
    original_init_weights = glm_module.Glm4MoeLitePreTrainedModel._init_weights
    original_initialize_missing_keys = glm_module.Glm4MoeLitePreTrainedModel._initialize_missing_keys
    patched_moe_cls = _Glm4MoeLiteNaiveMoeCompat

    glm_module.Glm4MoeLiteNaiveMoe = patched_moe_cls

    def _patched_init_weights(self_model, module):
        if isinstance(module, patched_moe_cls):
            return
        return original_init_weights(self_model, module)

    def _patched_initialize_missing_keys(self_model, is_quantized: bool):
        del self_model, is_quantized
        return

    glm_module.Glm4MoeLitePreTrainedModel._init_weights = _patched_init_weights
    glm_module.Glm4MoeLitePreTrainedModel._initialize_missing_keys = _patched_initialize_missing_keys
    try:
        yield
    finally:
        glm_module.Glm4MoeLiteNaiveMoe = original_cls
        glm_module.Glm4MoeLitePreTrainedModel._init_weights = original_init_weights
        glm_module.Glm4MoeLitePreTrainedModel._initialize_missing_keys = original_initialize_missing_keys
        _Glm4MoeLiteNaiveMoeCompat.reset_layout()


# =============================================================================
# ACTIVE PATH
# 该函数仍然需要：GLM 通过 XHBaseModel.postprocess_gptqmodel_structure()
# 在公共 GPTQModel 反量化后调用它，完成 split-MoE 到 fused-MoE 的结构归一。
# =============================================================================
def convert_gptqmodel_moe_structure(
    hf_model: nn.Module,
    target_moe_cls: type[nn.Module] | None = None,
) -> int:
    # XHGlm4MoeLiteModel.get_hf_model 的结构归一化步骤。
    # 输入前提：base_model.py 中的公共 GPTQModel 反量化逻辑已经把专家 qlinear 转成 nn.Linear。
    # 输出契约：恢复为原生 Glm4MoeLiteNaiveMoe，且暴露 gate_up_proj/down_proj 参数供 Merak wrapper 使用。
    if target_moe_cls is None:
        target_moe_cls = resolve_native_glm_moe_cls()

    def _is_split_layout_module(module: nn.Module) -> bool:
        return (
            hasattr(module, "gate_proj")
            and isinstance(getattr(module, "gate_proj"), nn.ModuleList)
            and hasattr(module, "up_proj")
            and isinstance(getattr(module, "up_proj"), nn.ModuleList)
            and hasattr(module, "down_proj")
            and isinstance(getattr(module, "down_proj"), nn.ModuleList)
        )

    modules_to_convert = [(name, module) for name, module in hf_model.named_modules() if _is_split_layout_module(module)]

    for name, module in modules_to_convert:
        gate_modules: nn.ModuleList = getattr(module, "gate_proj")
        up_modules: nn.ModuleList = getattr(module, "up_proj")
        down_modules: nn.ModuleList = getattr(module, "down_proj")
        num_experts = len(gate_modules)
        logger.info("Converting GPTQModel split MoE to fused structure: %s (%d experts)", name, num_experts)

        ref = gate_modules[0].weight
        intermediate = ref.shape[0]
        hidden = ref.shape[1]
        dtype = ref.dtype
        device = ref.device

        offload_to_cpu = device.type == "cuda"
        if offload_to_cpu:
            gate_src = [gate_modules[i].weight.detach().to(device="cpu", copy=True) for i in range(num_experts)]
            up_src = [up_modules[i].weight.detach().to(device="cpu", copy=True) for i in range(num_experts)]
            down_src = [down_modules[i].weight.detach().to(device="cpu", copy=True) for i in range(num_experts)]
            del module._modules["gate_proj"]
            del module._modules["up_proj"]
            del module._modules["down_proj"]
            del gate_modules
            del up_modules
            del down_modules
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            gate_src = [gate_modules[i].weight.data for i in range(num_experts)]
            up_src = [up_modules[i].weight.data for i in range(num_experts)]
            down_src = [down_modules[i].weight.data for i in range(num_experts)]

        gate_up_proj = torch.empty(num_experts, 2 * intermediate, hidden, dtype=dtype, device=device)
        down_proj = torch.empty(num_experts, hidden, intermediate, dtype=dtype, device=device)
        with torch.no_grad():
            for i in range(num_experts):
                gate_w = gate_src[i]
                up_w = up_src[i]
                down_w = down_src[i]
                if gate_w.device != device or gate_w.dtype != dtype:
                    gate_w = gate_w.to(device=device, dtype=dtype)
                if up_w.device != device or up_w.dtype != dtype:
                    up_w = up_w.to(device=device, dtype=dtype)
                if down_w.device != device or down_w.dtype != dtype:
                    down_w = down_w.to(device=device, dtype=dtype)
                gate_up_proj[i, :intermediate, :].copy_(gate_w)
                gate_up_proj[i, intermediate:, :].copy_(up_w)
                down_proj[i].copy_(down_w)

        if not offload_to_cpu:
            del module._modules["gate_proj"]
            del module._modules["up_proj"]
            del module._modules["down_proj"]
        else:
            del gate_src
            del up_src
            del down_src

        module.register_parameter("gate_up_proj", nn.Parameter(gate_up_proj, requires_grad=False))
        module.register_parameter("down_proj", nn.Parameter(down_proj, requires_grad=False))
        module.__class__ = target_moe_cls

    return len(modules_to_convert)
