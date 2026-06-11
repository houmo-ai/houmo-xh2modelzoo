import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from xh_model_zoo.xh_llm.models.hidream_o1 import (
    PATCH_SIZE,
    HiDreamO1DenoiseExportWrapper,
    build_rotary_inputs,
    build_t2i_sample_inputs,
    ensure_hidream_o1_imports,
    register_hidream_o1_wrap_modules,
)
from xhquant.api import (
    ConfigDict,
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_dynamo_model_to_quanted_model,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)


ensure_hidream_o1_imports()

if not hasattr(torch, "gelu"):
    torch.gelu = F.gelu
if not hasattr(torch, "silu"):
    torch.silu = F.silu


def load_hidream_model_cls():
    ensure_hidream_o1_imports()
    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration  # pyright: ignore[reportMissingImports]

    return Qwen3VLForConditionalGeneration


def export_component(
    model: torch.nn.Module,
    inputs,
    input_names,
    output_names,
    onnx_file: Path,
    golden_dir: Path,
    quant_scheme: QuantScheme,
    trace_backend: str,
    device: torch.device,
):
    quant_config = ConfigDict(create_quant_config(quant_scheme))
    if trace_backend == "dynamo":
        quant_graph_model = convert_dynamo_model_to_quanted_model(
            model,
            list(inputs),
            quant_scheme.target_device,
            quant_config,
        )
    else:
        quant_graph_model = convert_fx_model_to_quanted_model(
            model,
            list(inputs),
            quant_scheme.target_device,
            quant_config,
        )

    onnx_file.parent.mkdir(parents=True, exist_ok=True)
    convert_quanted_model_to_hmonnx(
        quant_graph_model,
        list(inputs),
        str(onnx_file),
        list(input_names),
        list(output_names),
    )

    session = HMONNXGoldenInference(str(onnx_file))
    session.exec_device = device
    session.save_golden = True
    golden_dir.mkdir(parents=True, exist_ok=True)
    session.golden_dir = str(golden_dir)
    with torch.no_grad():
        session.forward(*list(inputs))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="/data02/datasets/HiDream-O1-Image")
    parser.add_argument("--prompt", type=str, default="A beautiful castle beside a lake, cinematic, highly detailed")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--text-seq-len", type=int, default=512)
    parser.add_argument("--quant-type", type=str, default="w8a16h1_sefp")
    parser.add_argument("--trace-backend", choices=["fx", "dynamo"], default="dynamo")
    parser.add_argument("--work-dir", type=str, default="work_dirs/hidream_o1_w8a16")
    parser.add_argument("--forward-only", action="store_true", help="只验证 wrap 后 forward，不执行量化和 HMONNX 导出")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.height % PATCH_SIZE != 0 or args.width % PATCH_SIZE != 0:
        raise ValueError(f"height/width 必须能被 PATCH_SIZE={PATCH_SIZE} 整除")

    model_dir = Path(args.model).resolve()
    model_name = model_dir.name
    target_device = DeviceType.XH2a
    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else Path("work_dirs") / f"{model_name}_{target_device.name}_{args.width}x{args.height}_hidream_o1"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(work_dir / "convert.log", debug=args.debug)
    logger = get_root_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    logger.info("Loading HiDream-O1 processor/model from %s", model_dir)
    processor = AutoProcessor.from_pretrained(str(model_dir))
    model_cls = load_hidream_model_cls()
    model = model_cls.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        device_map="cuda" if device.type == "cuda" else None,
    ).eval()

    sample, vinputs = build_t2i_sample_inputs(
        model=model,
        processor=processor,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
        dtype=dtype,
        device=device,
        text_seq_len=args.text_seq_len,
    )
    txt_seq_len = int(sample["input_ids"].shape[-1])

    # RoPE only depends on position_ids and model rotary constants. Precompute it
    # before wrapping to preserve the original model's exact rotary path while
    # keeping cos/sin out of the exported graph.
    with torch.no_grad():
        inputs_embeds = model.model.get_input_embeddings()(sample["input_ids"])
        t_emb = model.model.t_embedder1(sample["timestep"].to(inputs_embeds.device))
        text_embeds_for_rope = torch.cat([inputs_embeds[:, :-1, :], t_emb.unsqueeze(1)], dim=1)
        vinputs_embedded = model.model.x_embedder(vinputs.to(inputs_embeds.device)).to(inputs_embeds.dtype)
        rotary_dummy_embeds = torch.cat([text_embeds_for_rope, vinputs_embedded], dim=1)
        rotary_position_ids = sample["position_ids"]
        if rotary_position_ids.ndim == 2:
            rotary_position_ids = rotary_position_ids[None, ...].expand(3, rotary_position_ids.shape[0], -1)
        elif rotary_position_ids.ndim == 3 and rotary_position_ids.shape[0] == 4:
            rotary_position_ids = rotary_position_ids[1:]
        rotary_cos, rotary_sin = build_rotary_inputs(
            model.model.language_model.rotary_emb,
            rotary_dummy_embeds,
            rotary_position_ids,
            dtype=dtype,
        )

    register_hidream_o1_wrap_modules(model)
    wrapper = HiDreamO1DenoiseExportWrapper(model, txt_seq_len=txt_seq_len).to(device).eval()

    inputs = [
        inputs_embeds,
        sample["attention_mask"],
        rotary_cos,
        rotary_sin,
        vinputs,
        sample["timestep_index"],
        # sample["token_types"].to(dtype=torch.int32),
    ]
    input_names = [
        "inputs_embeds",
        "attention_mask",
        "rotary_cos",
        "rotary_sin",
        "vinputs",
        "timestep",
    ]  # , "token_types"
    output_names = ["x_pred"]
    prefix = f"hidream_o1_denoise-{target_device}-{args.quant_type}-{args.width}x{args.height}"
    onnx_file = work_dir / "hmonnx" / f"{prefix}.onnx"
    golden_dir = work_dir / "golden" / prefix
    quant_scheme = QuantScheme(target_device=target_device, quant_type=args.quant_type)

    with torch.no_grad():
        ref = wrapper(*inputs)
    logger.info("torch reference x_pred shape: %s", list(ref.shape))
    if args.forward_only:
        logger.info(
            "forward-only done: dtype=%s mean=%.6f std=%.6f",
            ref.dtype,
            float(ref.float().mean()),
            float(ref.float().std()),
        )
        return

    export_component(
        model=wrapper,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        onnx_file=onnx_file,
        golden_dir=golden_dir,
        quant_scheme=quant_scheme,
        trace_backend=args.trace_backend,
        device=device,
    )

    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "model_name": model_name,
        "source_model_dir": str(model_dir),
        "hmonnx_file": str(onnx_file.relative_to(work_dir)),
        "golden_dir": str(golden_dir.relative_to(work_dir)),
        "target_device": str(target_device),
        "quant_type": args.quant_type,
        "trace_backend": args.trace_backend,
        "height": args.height,
        "width": args.width,
        "patch_size": PATCH_SIZE,
        "txt_seq_len": txt_seq_len,
        "prompt": args.prompt,
        "seed": args.seed,
        "input_names": input_names,
        "output_names": output_names,
        "sample_input_shapes": {name: list(t.shape) for name, t in zip(input_names, inputs, strict=True)},
    }
    json.dump(meta, open(work_dir / "meta.json", "w"), indent=4)
    logger.info("Saved meta to %s", work_dir / "meta.json")


if __name__ == "__main__":
    main()
