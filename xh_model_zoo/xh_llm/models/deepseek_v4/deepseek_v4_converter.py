# ================================================================== #
#  File: deepseek_v4_converter.py                                     #
#  Description:                                                       #
#    DeepSeek-V4 converter: HF -> wrap -> quant -> HMONNX export.    #
#                                                                     #
#    Handles FP4 pre-quantized model loading (via compat patch)       #
#    and standard float model loading.                                #
# ================================================================== #

import copy
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from onnx import numpy_helper
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from xhquant.api import (
    CacheTensor,
    Config,
    ConfigDict,
    DeviceType,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
)

from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model
from .deepseek_v4_convert_config import DeepseekV4ConvertConfig


# 兼容性 patch：xhquanttool 的 pad_concat_inputchannel_to_64_v2 引用 torch.gelu /
# torch.silu，但 torch 无此顶层属性（实为 torch.nn.functional.gelu/silu）。补齐
# 以避免 AttributeError 阻断 export。仅在缺失时注入，不影响其他逻辑。
# 放在 import 之后：gelu/silu 仅在该 transform pass 运行时被 node.target 比对引用，
# import 时不触及，故 patch 时机无影响。
for _act_name in ("gelu", "silu"):
    if not hasattr(torch, _act_name):
        setattr(torch, _act_name, getattr(torch.nn.functional, _act_name))


# ================================================================== #
#  FP4 (E2M1 + UE8M0) dequantization for expert down_proj weights   #
# ================================================================== #

_FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    """Unpack int8 FP4 -> float32 (2 values per byte)."""
    u8 = x.contiguous().view(torch.uint8)
    lut = _FP4_LUT.to(x.device)
    return torch.stack(
        [lut[(u8 & 0xF).long()], lut[((u8 >> 4) & 0xF).long()]],
        -1,
    ).reshape(*x.shape[:-1], -1)


def _fp4_scale(s: torch.Tensor) -> torch.Tensor:
    """UE8M0 float8 scale -> float32 power-of-2.

    UE8M0 格式: 纯指数，bias=127（与 float32 指数位一致）。
    """
    _UE8M0_BIAS = 127
    return torch.pow(2.0, (s.view(torch.uint8).to(torch.int32) - _UE8M0_BIAS).float())


def _dequantize_fp4(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Dequantize FP4 weight + UE8M0 scale -> float32."""
    v = _unpack_fp4(w)
    sc = _fp4_scale(s)
    R, C = v.shape
    bn = C // sc.shape[-1]
    return (v.reshape(R, -1, bn) * sc.unsqueeze(-1)).reshape(R, C)


# ================================================================== #
#  ONNX post-export: safe domain fixup                               #
# ================================================================== #

# Ops that only have XH2a:: parsers and can be safely domain-rewritten
# (no special attributes needed).
_SAFE_DOMAIN_REWRITE = frozenset(
    {
        "Add",
        "Sub",
        "Div",
        "Sqrt",
        "Reciprocal",
        # Element-wise ops：xh2a parser 零属性依赖
        "Mul",
        "Neg",
    }
)


def _rewrite_onnx_domains(onnx_path: str):
    """Post-export ONNX fixup.

    The onnxscript optimizer is bypassed (see _disable_onnx_optimizer),
    so all xh2a attributes survive intact.  This function handles:
      1. Pow(x, 2) → Mul(x, x) + domain rewrite (simpler than LUT)
      2. Safe domain rewrites (Add/Sub/Div/Sqrt/Reciprocal)
      3. Cast-to-float32 removal (xh2a ops require float16)
      4. ConstantOfShape → initializer preserving value attr (incl. -inf masks)

    Non-quantized standard-domain ops (MatMul, Sin, Cos, Relu, etc.)
    remain in standard domain.  The HMONNX runtime must have standard
    parsers registered for them (see _register_std_parsers).
    """
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)

    # -- Build initializer lookup --
    init_map = {}
    for init in model.graph.initializer:
        try:
            init_map[init.name] = numpy_helper.to_array(init)
        except Exception:
            pass

    stats = {"pow": 0, "domain": 0, "cast": 0, "cos_shape": 0}

    # ------------------------------------------------------------------
    #  Pass 1: Pow(x, 2) → Mul(x, x) in xh2a domain
    # ------------------------------------------------------------------
    for node in list(model.graph.node):
        if node.op_type != "Pow" or node.domain != "":
            continue
        exp_in = node.input[1]
        if exp_in not in init_map:
            continue
        if float(init_map[exp_in]) != 2.0:
            continue
        x_in = node.input[0]
        node.op_type = "Mul"
        node.domain = "ai.houmo.xh2a"
        del node.input[:]
        node.input.append(x_in)
        node.input.append(x_in)
        del node.attribute[:]
        stats["pow"] += 1

    # ------------------------------------------------------------------
    #  Pass 2: Safe domain rewrites
    # ------------------------------------------------------------------
    for node in model.graph.node:
        if node.domain == "" and node.op_type in _SAFE_DOMAIN_REWRITE:
            node.domain = "ai.houmo.xh2a"
            stats["domain"] += 1

    # ------------------------------------------------------------------
    #  Pass 3: Remove Cast-to-float32 feeding xh2a ops
    # ------------------------------------------------------------------
    xh2a_inputs = set()
    for node in model.graph.node:
        if node.domain == "ai.houmo.xh2a":
            for inp in node.input:
                xh2a_inputs.add(inp)

    for node in list(model.graph.node):
        if node.op_type != "Cast" or node.domain != "":
            continue
        to_attr = next((a.i for a in node.attribute if a.name == "to"), None)
        if to_attr != 1:  # 1 = float32
            continue
        out_name = node.output[0]
        if out_name not in xh2a_inputs:
            continue
        # Rewire: consumers now get the Cast's input directly
        cast_in = node.input[0]
        for other in model.graph.node:
            for i, inp in enumerate(other.input):
                if inp == out_name:
                    other.input[i] = cast_in
        model.graph.node.remove(node)
        stats["cast"] += 1

    # ------------------------------------------------------------------
    #  Pass 4: ConstantOfShape → initializer（保留 value 属性）
    # ------------------------------------------------------------------
    # ConstantOfShape 的可选 value 属性缺省为 0；CSA/HCA 的 -inf mask 填充
    # （_compressor_nl.py 的 new_full(-inf)）会导出为 value=-inf 的节点。
    # 盲目物化为 zeros 会摧毁 attention/compressor 的因果掩码——必须按原
    # fill value 保留；value=0 时与旧行为完全一致（消除「一律 zeros」这个
    # 会误伤非零 fill 的特殊情况）。
    nonzero_kept = 0
    for node in list(model.graph.node):
        if node.op_type != "ConstantOfShape" or node.domain != "":
            continue
        shape_in = node.input[0]
        out_name = node.output[0]
        if shape_in not in init_map:
            continue
        shape = init_map[shape_in].tolist()
        fill_val = 0.0
        value_attr = next((a for a in node.attribute if a.name == "value"), None)
        if value_attr is not None and value_attr.HasField("t"):
            try:
                fill_val = float(numpy_helper.to_array(value_attr.t).item())
            except Exception:
                pass
        arr = numpy_helper.from_array(
            np.full(shape, fill_val, dtype=np.float16),
            name=out_name,
        )
        model.graph.initializer.append(arr)
        model.graph.node.remove(node)
        stats["cos_shape"] += 1
        if fill_val != 0.0:
            nonzero_kept += 1
    if nonzero_kept:
        print(f"  [Pass5] 保留 {nonzero_kept} 个非零 ConstantOfShape（含 -inf mask），避免被误转为 zeros 破坏因果掩码")

    onnx.save(model, onnx_path)
    del model
    return stats


def _cast_model_to_fp16(model):
    """cast float 参数/buffer → fp16（保留 int/bool），对齐 quant fake_dtype。

    w8a8h1_sefp 的 quant fake_dtype=fp16，但 HF DeepSeek-V4 是 bf16。frontend 的
    remove_float 会删除 .float()，使 softmax/RMSNorm/sigmoid/RoPE 收到 model
    dtype —— bf16 与 quant fp16 mismatch，导致量化发散。wrap 后统一 cast fp16，
    消除该 mismatch（与 _verify_quanted_pt.py 的做法一致）。
    """
    for p in model.parameters():
        if p.is_floating_point() and p.dtype != torch.float16:
            p.data = p.data.to(torch.float16)
    for buf in model.buffers():
        if buf.is_floating_point() and buf.dtype != torch.float16:
            buf.data = buf.data.to(torch.float16)


# ================================================================== #
#  Standard-domain parser registration for non-quantized ops         #
# ================================================================== #


def _register_std_parsers():
    """Register standard-domain HMONNX parsers for non-quantized ops.

    These ops appear in the ONNX graph as standard-domain nodes because
    they were not quantized (e.g., HC mixing matmul, RoPE sin/cos).
    The HMONNX runtime needs parsers for them.
    """

    from xhquant.xhonnxruntime.base_module import BaseModule
    from xhquant.xhonnxruntime.hmonnx_inference import HMONNX_PARSERS

    _md = HMONNX_PARSERS.module_dict
    if "MatMul" in _md:
        return  # Already registered

    from typing import Dict

    from torch import Tensor

    def _get_axes_from_context(node, context):
        """Get axes from ONNX node attrs or from graph initializer (opset 18+)."""
        if hasattr(node, "attrs") and "axes" in node.attrs:
            v = node.attrs["axes"]
            return list(v) if isinstance(v, (list, tuple)) else [int(v)]
        # opset>=18: axes as second input (constant tensor)
        if len(node.inputs) > 1 and node.inputs[1] is not None:
            axes_tensor = node.inputs[1]
            # gs.Constant stores value in .values attribute
            if hasattr(axes_tensor, "values") and axes_tensor.values is not None:
                val = axes_tensor.values
                if hasattr(val, "tolist"):
                    return val.tolist()
                return [int(val)]
            # Fallback: check context.onnx_graph consts
            graph = getattr(context, "onnx_graph", None)
            if graph is not None:
                axes_name = axes_tensor.name
                for const in getattr(graph, "consts", []):
                    if const.name == axes_name and hasattr(const, "values"):
                        return const.values.tolist()
        return [-1]

    class _StdMatMul(BaseModule):
        def __init__(self, **kw):
            super().__init__()

        @classmethod
        def from_onnx_node(cls, node, context):
            m = cls()
            m.input_names = [i.name for i in node.inputs]
            m.output_names = [o.name for o in node.outputs]
            return m

        def _forward_aligned(self, a: Tensor, b: Tensor) -> Dict[str, Tensor]:
            return {self.output_names[0]: torch.matmul(a, b)}

    class _StdReduceMean(BaseModule):
        def __init__(self, axes, keepdims=1):
            super().__init__()
            self.axes = list(axes) if isinstance(axes, (list, tuple)) else [int(axes)]
            self.keepdims = bool(keepdims)

        @classmethod
        def from_onnx_node(cls, node, context):
            axes = _get_axes_from_context(node, context)
            kd = node.attrs.get("keepdims", 1) if hasattr(node, "attrs") else 1
            m = cls(axes=axes, keepdims=kd)
            m.input_names = [node.inputs[0].name]
            m.output_names = [o.name for o in node.outputs]
            return m

        def _forward_aligned(self, x: Tensor) -> Dict[str, Tensor]:
            return {self.output_names[0]: torch.mean(x, dim=self.axes, keepdim=self.keepdims)}

    class _StdReduceSum(BaseModule):
        def __init__(self, axes, keepdims=0):
            super().__init__()
            self.axes = list(axes) if isinstance(axes, (list, tuple)) else [int(axes)]
            self.keepdims = bool(keepdims)

        @classmethod
        def from_onnx_node(cls, node, context):
            axes = _get_axes_from_context(node, context)
            kd = node.attrs.get("keepdims", 0) if hasattr(node, "attrs") else 0
            m = cls(axes=axes, keepdims=kd)
            m.input_names = [node.inputs[0].name]
            m.output_names = [o.name for o in node.outputs]
            return m

        def _forward_aligned(self, x: Tensor) -> Dict[str, Tensor]:
            return {self.output_names[0]: torch.sum(x, dim=self.axes, keepdim=self.keepdims)}

    class _StdTopK(BaseModule):
        def __init__(self, k, axis=-1, largest=1, sorted=1):
            super().__init__()
            self.k = int(k)
            self.axis = int(axis)
            self.largest = bool(largest)
            self.sorted = bool(sorted)

        @classmethod
        def from_onnx_node(cls, node, context):
            k = node.attrs.get("k", 1) if hasattr(node, "attrs") else 1
            m = cls(
                k=k,
                axis=node.attrs.get("axis", -1) if hasattr(node, "attrs") else -1,
                largest=node.attrs.get("largest", 1) if hasattr(node, "attrs") else 1,
                sorted=node.attrs.get("sorted", 1) if hasattr(node, "attrs") else 1,
            )
            m.input_names = [i.name for i in node.inputs if i is not None]
            # 只暴露 indices 输出（node.outputs[1]）。
            # values 输出（node.outputs[0]）无人消费，跳过它以避免 xhquant runtime
            # 的 output validation IndexError（inputs[index]=t 越界）。
            m.output_names = [node.outputs[1].name]
            return m

        def _forward_aligned(self, x: Tensor) -> Dict[str, Tensor]:
            _, idxs = torch.topk(x, self.k, dim=self.axis, largest=self.largest, sorted=self.sorted)
            return {self.output_names[0]: idxs}

    class _StdSoftmax(BaseModule):
        def __init__(self, axis=-1):
            super().__init__()
            self.axis = int(axis)

        @classmethod
        def from_onnx_node(cls, node, context):
            m = cls(axis=node.attrs.get("axis", -1))
            m.input_names = [i.name for i in node.inputs]
            m.output_names = [o.name for o in node.outputs]
            return m

        def _forward_aligned(self, x: Tensor) -> Dict[str, Tensor]:
            return {self.output_names[0]: torch.softmax(x.float(), dim=self.axis).to(x.dtype)}

    class _StdSin(BaseModule):
        def __init__(self, **kw):
            super().__init__()

        @classmethod
        def from_onnx_node(cls, node, context):
            m = cls()
            m.input_names = [i.name for i in node.inputs]
            m.output_names = [o.name for o in node.outputs]
            return m

        def _forward_aligned(self, x: Tensor) -> Dict[str, Tensor]:
            return {self.output_names[0]: torch.sin(x.float()).to(x.dtype)}

    class _StdCos(BaseModule):
        def __init__(self, **kw):
            super().__init__()

        @classmethod
        def from_onnx_node(cls, node, context):
            m = cls()
            m.input_names = [i.name for i in node.inputs]
            m.output_names = [o.name for o in node.outputs]
            return m

        def _forward_aligned(self, x: Tensor) -> Dict[str, Tensor]:
            return {self.output_names[0]: torch.cos(x.float()).to(x.dtype)}

    class _StdRelu(BaseModule):
        def __init__(self, **kw):
            super().__init__()

        @classmethod
        def from_onnx_node(cls, node, context):
            m = cls()
            m.input_names = [i.name for i in node.inputs]
            m.output_names = [o.name for o in node.outputs]
            return m

        def _forward_aligned(self, x: Tensor) -> Dict[str, Tensor]:
            return {self.output_names[0]: torch.relu(x)}

    class _StdPow(BaseModule):
        def __init__(self, **kw):
            super().__init__()

        @classmethod
        def from_onnx_node(cls, node, context):
            m = cls()
            m.input_names = [i.name for i in node.inputs]
            m.output_names = [o.name for o in node.outputs]
            return m

        def _forward_aligned(self, x: Tensor, exp: Tensor) -> Dict[str, Tensor]:
            return {self.output_names[0]: torch.pow(x, exp)}

    class _StdMul(BaseModule):
        def __init__(self, **kw):
            super().__init__()

        @classmethod
        def from_onnx_node(cls, node, context):
            m = cls()
            m.input_names = [i.name for i in node.inputs]
            m.output_names = [o.name for o in node.outputs]
            return m

        def _forward_aligned(self, a: Tensor, b: Tensor) -> Dict[str, Tensor]:
            return {self.output_names[0]: a * b}

    class _StdNeg(BaseModule):
        def __init__(self, **kw):
            super().__init__()

        @classmethod
        def from_onnx_node(cls, node, context):
            m = cls()
            m.input_names = [i.name for i in node.inputs]
            m.output_names = [o.name for o in node.outputs]
            return m

        def _forward_aligned(self, x: Tensor) -> Dict[str, Tensor]:
            return {self.output_names[0]: torch.neg(x)}

    _REG = {
        "MatMul": _StdMatMul,
        "ReduceMean": _StdReduceMean,
        "ReduceSum": _StdReduceSum,
        "TopK": _StdTopK,
        "Softmax": _StdSoftmax,
        "Sin": _StdSin,
        "Cos": _StdCos,
        "Relu": _StdRelu,
        "Pow": _StdPow,
        "Mul": _StdMul,
        "Neg": _StdNeg,
    }
    for name, cls in _REG.items():
        if name not in HMONNX_PARSERS.module_dict:
            HMONNX_PARSERS.register_module(name=name, module=cls)


# ================================================================== #
#  Disable ONNXScript optimizer to preserve xh2a attributes          #
# ================================================================== #

_original_optimize = None


def _disable_onnx_optimizer():
    """Monkey-patch onnxscript.optimizer.optimize to a no-op.

    The optimizer converts xh2a custom ops back to standard ONNX domain,
    stripping their LUT/mode attributes. By bypassing it, all xh2a
    attributes survive the export pipeline intact.

    The custom optimization passes in xhquant's optimize_v2 (RemoveCast,
    INT64toINT32, Constant2Initializer, CastFolder, RemoveIdentity)
    still run — they don't strip xh2a attributes.
    """
    global _original_optimize
    try:
        import onnxscript.optimizer

        _original_optimize = onnxscript.optimizer.optimize
        onnxscript.optimizer.optimize = lambda *a, **kw: None
    except ImportError:
        pass


def _restore_onnx_optimizer():
    """Restore the original onnxscript optimizer after export."""
    global _original_optimize
    if _original_optimize is not None:
        try:
            import onnxscript.optimizer

            onnxscript.optimizer.optimize = _original_optimize
        except ImportError:
            pass
        _original_optimize = None


# ================================================================== #
#  Converter                                                          #
# ================================================================== #


class DeepseekV4ConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: DeepseekV4ConvertConfig):
        super().__init__()
        self.config = config

    def load_hf_model(self, hf_model_dir: str, **kwargs):
        # -- FP4 compat patch (must be called before import) --
        try:
            from xhquanttool.tests.testing.tools.deepseek_v4.compat import (
                patch_fp8_quantizer,
            )

            patch_fp8_quantizer()
        except ImportError:
            pass

        native_model = AutoModelForCausalLM.from_pretrained(
            hf_model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        if native_model.config.tie_word_embeddings:
            native_model.config.torchscript = True
            native_model.tie_weights()
            native_model.config.tie_word_embeddings = False

        return native_model

    def _fix_fp4_experts(self, model, hf_model_path: str):
        """Dequantize FP4 expert down_proj weights."""
        num_layers = model.config.num_hidden_layers
        num_experts = model.config.n_routed_experts
        idx = json.load(open(os.path.join(hf_model_path, "model.safetensors.index.json")))
        wm = idx["weight_map"]

        for li in range(num_layers):
            ws = []
            for ei in range(num_experts):
                wk = f"layers.{li}.ffn.experts.{ei}.w2.weight"
                sk = f"layers.{li}.ffn.experts.{ei}.w2.scale"
                sw = safe_open(os.path.join(hf_model_path, wm[wk]), framework="pt")
                ss = safe_open(os.path.join(hf_model_path, wm[sk]), framework="pt")
                ws.append(_dequantize_fp4(sw.get_tensor(wk), ss.get_tensor(sk)).to(torch.bfloat16))
            model.model.layers[li].mlp.experts.down_proj = nn.Parameter(torch.stack(ws, 0))
        return model

    def _convert(self, hf_model_path: str, output_dir: str):
        logger = get_root_logger()
        config = self.config

        native_model = self.load_hf_model(hf_model_path)
        native_model = self._fix_fp4_experts(native_model, hf_model_path)
        lm_head = native_model.lm_head
        if not hasattr(lm_head, "quant_weight"):
            config.quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"

        model_name = Path(hf_model_path).name
        target_device = config.quant_scheme.target_device
        batch_size = config.batch_size
        context_length = config.context_length
        input_sequence_length = config.input_sequence_length
        quant_type = config.quant_scheme.quant_type
        quant_config = create_quant_config(config.quant_scheme)
        quant_config = ConfigDict(quant_config)

        meta_info: Dict[str, Any] = dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            device=str(target_device),
            model_name=model_name,
            hf_model_path=hf_model_path,
            quant_scheme=config.quant_scheme.to_dict(),
        )

        work_dir = Path(output_dir)

        # -- Copy HF config --
        hf_config_dir = work_dir / "hf_config"
        hf_config_dir.mkdir(exist_ok=True, parents=True)
        for f in [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
        ]:
            src = Path(hf_model_path) / f
            if src.exists():
                shutil.copyfile(src, hf_config_dir / f)
        meta_info["hf_config"] = str(hf_config_dir.relative_to(work_dir))

        # -- Save token embedding --
        token_embedding = native_model.model.get_input_embeddings()
        tok_emb_file = work_dir / "token_embedding.pt"
        torch.save(token_embedding.state_dict(), str(tok_emb_file))
        meta_info["token_embedding_file"] = str(tok_emb_file.relative_to(work_dir))

        # -- Register wrap modules --
        from ._layers import register_wrap_modules

        register_wrap_modules()

        # -- Read compress_rate from compressor modules BEFORE wrapping --
        # (wrapping decomposes and deletes self.compressor)
        compressor_layers = []
        compress_rates = {}
        for i, lt in enumerate(native_model.config.layer_types):
            if lt != "sliding_attention":
                cr = native_model.model.layers[i].self_attn.compressor.compress_rate
                compressor_layers.append(i)
                compress_rates[i] = cr

        # -- Build wrap config --
        # -- compressed_kv_seq_lens: prefill Attention._setup 也需要 --
        compressed_kv_seq_lens = {i: input_sequence_length // compress_rates[i] for i in compressor_layers}
        wrap_cfg = Config(
            dict(
                batch_size=batch_size,
                max_sequence_length=context_length,
                input_sequence_length=input_sequence_length,
                use_cache=True,
                num_logits_to_keep=config.num_logits_to_keep,
                kv_cache=dict(cache_axis=2, context_length=context_length),
                compressed_kv_seq_lens=compressed_kv_seq_lens,
            )
        )
        meta_info["wrap_cfg"] = wrap_cfg.to_dict()

        # -- Save native model copy for decode BEFORE any mutation --
        decode_native = copy.deepcopy(native_model)

        wrapped_model = wrap_llm_model(native_model, wrap_cfg)
        _cast_model_to_fp16(wrapped_model)  # 对齐 quant fake_dtype(fp16)

        # -- Build KV caches (shared KV: K==V, single head) --
        num_layers = wrapped_model.config.num_hidden_layers
        head_dim = wrapped_model.config.head_dim
        num_kv_heads = wrapped_model.config.num_key_value_heads

        kv_cache_shape = [1, num_kv_heads, context_length, head_dim]
        meta_info["kv_cache"] = dict(
            shape=kv_cache_shape,
            num_decoder_layers=num_layers,
        )

        past_kv_caches = []
        for _ in range(num_layers):
            past_kv_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))

        # -- Build calibration inputs --
        input_ids = torch.randint(0, 1000, (1, input_sequence_length), dtype=torch.long)
        position_ids = torch.arange(0, input_sequence_length, dtype=torch.long).unsqueeze(0)
        inputs_embeds = token_embedding(input_ids)
        past_seq_length = torch.tensor([0], dtype=torch.int32)
        current_input_length = torch.tensor([input_sequence_length], dtype=torch.int32)

        inputs = (
            inputs_embeds,
            position_ids,
            past_seq_length,
            current_input_length,
            input_ids,
            past_kv_caches,
        )
        input_names = [
            "inputs_embeds",
            "position_ids",
            "past_seq_length",
            "current_input_length",
            "input_ids",
        ]
        for i in range(num_layers):
            input_names.append(f"past_key_cache_{i}")
        output_names = ["logits"]
        for i in compressor_layers:
            output_names.append(f"compressed_kv_{i}")

        prefix = f"{model_name}-{target_device}-{context_length // 1024}k-{quant_type}"

        # -- Disable onnxscript optimizer to preserve xh2a attributes --
        _disable_onnx_optimizer()

        try:
            # -- Export prefill --
            prefill_onnx = work_dir / "hmonnx" / "prefill" / f"{prefix}_prefill.onnx"
            prefill_onnx.parent.mkdir(exist_ok=True, parents=True)
            meta_info["prefill_onnx"] = str(prefill_onnx.relative_to(work_dir))

            logger.info("***** start export prefill model *****")
            quanted_model = convert_fx_model_to_quanted_model(
                wrapped_model,
                inputs,
                target_device,
                quant_config=quant_config,
            )

            input_names_compat = BaseConverter.xh1_hmonnx_compatible(input_names)

            convert_quanted_model_to_hmonnx(
                quanted_model,
                inputs,
                str(prefill_onnx),
                input_names=input_names_compat,
                output_names=output_names,
            )
            stats = _rewrite_onnx_domains(str(prefill_onnx))
            logger.info(f"Export prefill to {prefill_onnx} (fixup: {stats})")

            # -- 释放 prefill GPU 资源，避免 decode 量化 OOM --
            del quanted_model, wrapped_model
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Prefill resources released, GPU cache cleared")

            # -- Export decode --
            logger.info("***** start export decode model *****")

            decode_kv_caches = [
                CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_layers)
            ]

            # -- Compressed KV cache: 打包进 past_kv_caches 列表 --
            # past_key_caches = [kv_0..kv_N, ckv_0..ckv_N]
            # Model 用 past_key_caches[num_layers + idx] 访问
            decode_ckv_caches = []
            for i in range(num_layers):
                if i in set(compressor_layers):
                    cr = compress_rates[i]
                    n_windows = input_sequence_length // cr
                    ckv_shape = [1, 1, n_windows, head_dim]
                else:
                    ckv_shape = [1, 1, 1, head_dim]
                decode_ckv_caches.append(torch.zeros(ckv_shape, dtype=torch.float16))

            decode_past_caches = decode_kv_caches + decode_ckv_caches

            decode_inputs = (
                inputs_embeds[:, :1, :],
                position_ids[:, :1],
                past_seq_length,
                torch.ones_like(current_input_length),
                input_ids[:, :1],
                decode_past_caches,
            )

            decode_wrap_cfg = Config(
                dict(
                    batch_size=batch_size,
                    max_sequence_length=context_length,
                    input_sequence_length=1,
                    use_cache=True,
                    num_logits_to_keep=config.num_logits_to_keep,
                    kv_cache=dict(cache_axis=2, context_length=context_length),
                    compressed_kv_seq_lens=compressed_kv_seq_lens,
                )
            )
            decode_wrapped = wrap_llm_model(decode_native, decode_wrap_cfg)
            _cast_model_to_fp16(decode_wrapped)  # 对齐 quant fake_dtype(fp16)

            decode_quanted = convert_fx_model_to_quanted_model(
                decode_wrapped,
                decode_inputs,
                target_device,
                quant_config=quant_config,
            )

            decode_onnx = work_dir / "hmonnx" / "decode" / f"{prefix}_decoder.onnx"
            decode_onnx.parent.mkdir(exist_ok=True, parents=True)
            meta_info["decode_onnx"] = str(decode_onnx.relative_to(work_dir))

            # -- Decode input_names: KV cache + compressed_kv_cache --
            decode_input_names = list(input_names_compat)
            for i in range(num_layers):
                decode_input_names.append(f"compressed_kv_cache_{i}")

            convert_quanted_model_to_hmonnx(
                decode_quanted,
                decode_inputs,
                str(decode_onnx),
                input_names=decode_input_names,
                output_names=["logits"],
            )
            stats = _rewrite_onnx_domains(str(decode_onnx))
            logger.info(f"Export decode to {decode_onnx} (fixup: {stats})")

        finally:
            _restore_onnx_optimizer()

        # -- Compressed KV cache metadata --
        if compressor_layers:
            ckv_shapes = {}
            for i in compressor_layers:
                cr = compress_rates[i]
                n_windows = input_sequence_length // cr
                ckv_shapes[str(i)] = [1, 1, n_windows, head_dim]
            meta_info["compressed_kv_cache"] = dict(
                layer_indices=compressor_layers,
                shapes=ckv_shapes,
            )

        json.dump(meta_info, open(work_dir / "meta.json", "w"), indent=4)

    @classmethod
    def convert(
        cls,
        hf_model_path: str,
        config: DeepseekV4ConvertConfig,
        output_dir: str,
    ):
        cls(config)._convert(hf_model_path, output_dir)
