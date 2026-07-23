"""CosyVoice3 HMONNX utils: load export meta + build unified inference model.

Mirrors qwen3_tts/hmonnx_utils.py: reads export_meta_info.json to locate all
exported components, then builds a CosyVoice3HMONNXInference config dict.
"""

import json
from pathlib import Path


DEFAULT_TEXT = "基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地"


def load_export_meta(work_dir):
    work_dir = Path(work_dir).resolve()
    meta_file = work_dir / "export_meta_info.json"
    if not meta_file.is_file():
        raise FileNotFoundError(
            f"export_meta_info.json not found in {work_dir}. "
            "Only pass a trusted export directory produced by CosyVoice3Workflow.export()."
        )
    return json.loads(meta_file.read_text(encoding="utf-8"))


def load_component_meta(work_dir, export_meta, name):
    work_dir = Path(work_dir)
    comp = export_meta.get("components", {}).get(name)
    if not comp:
        raise KeyError(f"component {name!r} missing in export_meta_info.json")
    if "meta_file" in comp:
        meta_file = work_dir / comp["meta_file"]
        return meta_file, json.loads(meta_file.read_text(encoding="utf-8"))
    return work_dir / comp["component_dir"], comp


def _onnx_path(work_dir, export_meta, name):
    comp = export_meta.get("components", {}).get(name)
    if not comp:
        return {"onnx_file": None}
    if "meta_file" in comp:
        meta_file = work_dir / comp["meta_file"]
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        return {"onnx_file": str(meta_file.parent / meta["hmonnx"])}
    comp_dir = work_dir / comp["component_dir"]
    return {"onnx_file": str(comp_dir / comp["hmonnx"])}


def build_cosyvoice3_model_cfg(work_dir):
    work_dir = Path(work_dir)
    export_meta = load_export_meta(work_dir)
    comps = export_meta.get("components", {})

    llm_comp = comps.get("llm")
    if not llm_comp:
        raise KeyError("llm component missing in export_meta_info.json")
    llm_model_dir = str(work_dir / llm_comp["component_dir"])

    # Prefer hf_model from LLM meta_info.json (always the actual HF model dir),
    # fall back to export_meta_info.json (may be the parent --model-dir arg).
    llm_meta_file = work_dir / llm_comp.get("meta_file", "LLM/meta_info.json")
    hf_model = export_meta.get("hf_model")
    if llm_meta_file.is_file():
        llm_meta = json.loads(llm_meta_file.read_text(encoding="utf-8"))
        hf_model = llm_meta.get("hf_model", hf_model)

    cfg = {
        "type": "CosyVoice3HMONNXInference",
        "hf_model": hf_model,
        "llm": {"model_dir": llm_model_dir},
        "input_embedding": {"file": str(work_dir / export_meta.get("input_embedding_file", "input_embedding.pt"))},
    }

    for name in (
        "campplus",
        "speech_tokenizer_v3",
        "flow_decoder",
        "hift",
        "pre_lookahead_layer",
        "spk_embed_affine_layer",
    ):
        if name in comps:
            cfg[name] = _onnx_path(work_dir, export_meta, name)
        else:
            cfg[name] = {"onnx_file": None}

    sos_path = Path(llm_model_dir) / "sos_emb.pt"
    tid_path = Path(llm_model_dir) / "task_id_emb.pt"
    if sos_path.is_file() and tid_path.is_file():
        cfg["sos_eos_emb"] = {"file": str(sos_path)}
        cfg["task_id_emb"] = {"file": str(tid_path)}
    else:
        cfg["sos_eos_emb"] = None
        cfg["task_id_emb"] = None

    return cfg


def build_cosyvoice3_model(work_dir, device):
    from xhmodel_merak.xh_other_model.builder import MODELS
    from xhmodel_merak.xh_other_model.models.cosyvoice3 import CosyVoice3HMONNXInference

    cfg = build_cosyvoice3_model_cfg(work_dir)
    model = MODELS.build(cfg)
    if not isinstance(model, CosyVoice3HMONNXInference):
        raise TypeError(f"expected CosyVoice3HMONNXInference, got {type(model)!r}")
    model = model.to(device)
    llm = getattr(model, "llm_model", None) or getattr(model, "hf_model_wrapped", None)
    if llm is not None:
        inner = getattr(llm, "_llm_model", llm)
        for attr in ("token_embedding", "speech_embedding"):
            emb = getattr(inner, attr, None)
            if emb is not None:
                setattr(inner, attr, emb.to(device))
    return model


def infer_request(export_meta, args):
    return {
        "text": getattr(args, "text", None) or export_meta.get("tts_text") or DEFAULT_TEXT,
        "prompt_wav": getattr(args, "prompt_wav", None),
        "prompt_text": getattr(args, "prompt_text", None) or "",
    }
