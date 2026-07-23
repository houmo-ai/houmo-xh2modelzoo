"""CosyVoice3 HMONNX runtime smoke demo.

Loads every exported CosyVoice3 hmonnx graph referenced by
``export_meta_info.json`` and runs each one on dummy inputs. This validates
that the exported graphs are loadable and runnable at the HMONNX runtime level.

Usage::

    CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=$PWD python \
        examples_merak/tts/cosyvoice3/hmonnx_demo.py \
        --export-dir work_dirs/CosyVoice3-0.5B_XH2a --device cuda:0
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CosyVoice3 HMONNX runtime smoke demo.")
    parser.add_argument(
        "--export-dir",
        required=True,
        help="CosyVoice3 export output directory (contains export_meta_info.json).",
    )
    parser.add_argument("--device", default="cuda", help="Runtime device, e.g. cuda:0 or cpu.")
    return parser.parse_args()


def _select_torch_device(device: str) -> str:
    import torch

    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, torch.device):
        return str(device)
    return str(device)


def _build_inputs(name: str, comp_meta: dict, torch_device: str):
    import torch

    if name == "llm_decoder":
        return [torch.randn(1, 896, dtype=torch.float16, device=torch_device)]
    if name == "spk_embed_affine_layer":
        return [torch.randn(1, 192, dtype=torch.float16, device=torch_device)]
    if name == "pre_lookahead_layer":
        return [torch.randn(1, 1024, 80, dtype=torch.float16, device=torch_device)]
    if name == "campplus":
        seq = int(comp_meta.get("fixed_dims", {}).get("sequence_length", 1000))
        return [torch.randn(1, seq, 80, dtype=torch.float16, device=torch_device)]
    if name == "flow_decoder":
        seq = int(comp_meta.get("seq_len", 2048))
        b = int(comp_meta.get("batch_size", 2))
        c = int(comp_meta.get("out_channels", 80))
        return [
            torch.rand((b, c, seq), dtype=torch.float16, device=torch_device),
            torch.ones((b, 1, seq), dtype=torch.float16, device=torch_device),
            torch.rand((b, c, seq), dtype=torch.float16, device=torch_device),
            torch.rand((b,), dtype=torch.float16, device=torch_device),
            torch.rand((b, c), dtype=torch.float16, device=torch_device),
            torch.rand((b, c, seq), dtype=torch.float16, device=torch_device),
        ]
    if name == "hift":
        return [torch.randn(1, 80, 1024, dtype=torch.float16, device=torch_device)]
    if name == "speech_tokenizer_v3":
        return [
            torch.randn(1, 128, 3000, dtype=torch.float16, device=torch_device),
            torch.randn(1, 20, 750, 750, dtype=torch.float16, device=torch_device),
            torch.randn(1, 750, 1280, dtype=torch.float16, device=torch_device),
        ]
    raise ValueError(f"Unknown component for demo input construction: {name}")


def main() -> None:
    args = parse_args()
    work_dir = Path(args.export_dir)
    meta_file = work_dir / "export_meta_info.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"export_meta_info.json not found in {work_dir}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    torch_device = _select_torch_device(args.device)

    from xhquant.api import HMONNXInference

    components = meta.get("components", {})
    print(f"Found {len(components)} components in {meta_file}")

    for name, comp in components.items():
        comp_dir = work_dir / comp["component_dir"]
        if "meta_file" in comp:
            comp_meta = json.loads((work_dir / comp["meta_file"]).read_text(encoding="utf-8"))
        else:
            comp_meta = comp
        print(f"\n[{name}] component_dir={comp['component_dir']}")

        if name == "llm":
            import torch

            from xhquant.core import CacheTensor

            kv_shape = tuple(int(d) for d in comp_meta["kv_cache_shape"])
            layers = int(comp_meta["num_hidden_layers"])
            prefill_len = int(comp_meta["wrap_cfg"]["input_sequence_length"])

            for phase, seq_len, past_sl in [
                ("prefill_onnx_file", prefill_len, 0),
                ("decode_onnx_file", 1, prefill_len),
            ]:
                hmonnx = comp_dir / comp_meta[phase]
                session = HMONNXInference(str(hmonnx))
                session.to(torch_device)
                session.save_golden = False
                inputs_embeds = torch.randn(1, seq_len, 896, dtype=torch.float16, device=torch_device)
                past_seq_length = torch.tensor([past_sl], dtype=torch.int32, device=torch_device)
                current_input_length = torch.tensor([seq_len], dtype=torch.int32, device=torch_device)
                past_key_caches = [
                    CacheTensor(torch.zeros(kv_shape, dtype=torch.float16, device=torch_device))
                    for _ in range(layers)
                ]
                past_value_caches = [
                    CacheTensor(torch.zeros(kv_shape, dtype=torch.float16, device=torch_device))
                    for _ in range(layers)
                ]
                inputs = [inputs_embeds, past_seq_length, current_input_length, *past_key_caches, *past_value_caches]
                output = session(*inputs)
                print(f"  executed {phase}: {hmonnx.name}, output type={type(output).__name__}")
            continue

        hmonnx = comp_dir / comp_meta["hmonnx"]
        inputs = _build_inputs(name, comp_meta, torch_device)
        session = HMONNXInference(str(hmonnx))
        session.to(torch_device)
        session.save_golden = False
        output = session(*inputs)
        print(f"  loaded {hmonnx.name}, ran OK, output type={type(output).__name__}")

    print("\nAll component graphs loaded and executed successfully.")


if __name__ == "__main__":
    main()
