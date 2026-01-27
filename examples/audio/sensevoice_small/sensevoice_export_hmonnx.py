import argparse
from pathlib import Path
from typing import List


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--onnx", type=str, required=True)
    
    # Calibration data options (Mutually exclusive)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--calib", type=str, help="Path to pre-generated calibration .pth file")
    g.add_argument("--hf-dataset", type=str, help="HuggingFace dataset name (e.g. openslr/librispeech_asr)")
    
    # HF dataset options
    p.add_argument("--hf-config", type=str, default="clean", help="Dataset config name")
    p.add_argument("--hf-split", type=str, default="validation", help="Dataset split")
    p.add_argument("--hf-audio-field", type=str, default="audio", help="Audio field name")
    p.add_argument("--hf-streaming", action="store_true", help="Use streaming mode for HF dataset")
    p.add_argument("--model-dir", type=str, default="/data01/nfs_shared/ASR_TTS/SenseVoiceSmall", help="Path to SenseVoice model for frontend")
    
    p.add_argument("--calib-samples", type=int, default=128, help="Number of calibration samples")
    p.add_argument("--calib-metric", type=str, default="minmax", choices=["minmax", "mse", "kl"], help="Calibration metric")

    p.add_argument("--quant-type", type=str, default="w8a8h1_sefp")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--out-dir", type=str, default="work_dirs/sensevoice_small/export_xh2a")
    return p.parse_args()

def _make_calib_data_from_dataset(args, input_names: List[str], target_shapes: dict[str, tuple[int, ...]]) -> List["object"]:
    import torch
    import torch.nn.functional as F
    import sensevoice_common as sc
    from tqdm import tqdm

    print(f"Loading calibration data from HF dataset: {args.hf_dataset} (split={args.hf_split}, limit={args.calib_samples})")
    
    # Load samples
    samples = sc.load_hf_dataset(
        dataset=args.hf_dataset,
        config=args.hf_config,
        split=args.hf_split,
        limit=args.calib_samples,
        streaming=args.hf_streaming,
        audio_field=args.hf_audio_field
    )
    
    if not samples:
        raise ValueError("No samples found in dataset!")

    # Build frontend
    model_dir = Path(args.model_dir).expanduser().resolve()
    frontend = sc.build_frontend(model_dir)
    target_sr = int(frontend.cfg.fs)

    # Process all samples
    # Result: list of dicts (inputs)
    all_inputs_dict = {name: [] for name in input_names}
    
    print("Processing audio samples...")
    for s in tqdm(samples):
        try:
            wav = sc.load_audio_any(s, target_sr=target_sr)
            feat, feat_len = sc.extract_features(frontend, wav)
            # Make inputs (returns dict of tensors with batch=1)
            inputs = sc.make_inputs_for_sample(feat, feat_len, s.language, s.textnorm)
            
            # Align with target shapes if needed (e.g. padding)
            for name, t in inputs.items():
                if name not in all_inputs_dict:
                    continue
                    
                t = torch.as_tensor(t)
                if name == "speech":
                    t = t.to(torch.float32)
                else:
                    t = t.to(torch.int32)
                
                if name in target_shapes:
                    target_shape = target_shapes[name]
                    if name == "speech" and t.ndim == 3 and len(target_shape) == 3:
                        cur_t = t.shape[1]
                        tgt_t = target_shape[1]
                        if cur_t != tgt_t:
                            if cur_t < tgt_t:
                                t = F.pad(t, (0, 0, 0, tgt_t - cur_t))
                            else:
                                t = t[:, :tgt_t, :]
                
                all_inputs_dict[name].append(t)
                
        except Exception as e:
            print(f"Warning: Failed to process sample {s.audio_id}: {e}")
            continue

    # Stack into a single batch tensor for each input
    # Assuming the first dimension is batch size 1, we cat them to make batch size N
    final_inputs = []
    batch_size = len(all_inputs_dict[input_names[0]])
    print(f"Collected {batch_size} valid calibration samples.")
    
    for name in input_names:
        t_list = all_inputs_dict[name]
        if not t_list:
             raise ValueError(f"No data for input: {name}")
        
        # Handle variable length inputs (like speech) by padding to max length in the batch
        if name == "speech" and len(t_list) > 0 and t_list[0].ndim == 3:
            max_t = max(t.shape[1] for t in t_list)
            padded_list = []
            for t in t_list:
                cur_t = t.shape[1]
                if cur_t < max_t:
                    # Pad time dimension (dim 1) for (1, T, D) tensor
                    # F.pad arg order: (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right, ...)
                    t = F.pad(t, (0, 0, 0, max_t - cur_t))
                padded_list.append(t)
            t_list = padded_list

        # Cat along dim 0
        batched = torch.cat(t_list, dim=0)
        final_inputs.append(batched)
        
    return final_inputs

def _onnx_io_names(onnx_path: Path) -> tuple[list[str], list[str]]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    g = model.graph
    init_names = {i.name for i in g.initializer}
    inputs = [vi.name for vi in g.input if vi.name not in init_names]
    outputs = [vi.name for vi in g.output]
    return inputs, outputs


def _get_static_input_shapes(onnx_path: Path) -> dict[str, tuple[int, ...]]:
    import onnx
    # Load only structure for speed
    model = onnx.load(str(onnx_path), load_external_data=False)
    shapes = {}
    for i in model.graph.input:
        shape = []
        is_static = True
        for d in i.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                shape.append(d.dim_value)
            else:
                is_static = False
                break
        if is_static and shape:
            shapes[i.name] = tuple(shape)
    return shapes


def _make_input_list(calib_path: Path, input_names: List[str], target_shapes: dict[str, tuple[int, ...]]) -> List["object"]:
    import torch
    import torch.nn.functional as F

    payload = torch.load(calib_path, map_location="cpu")
    out: List[object] = []
    for name in input_names:
        if name not in payload:
            raise KeyError(f"missing input in calib: {name}; available: {sorted(payload.keys())}")
        t = payload[name]
        if name in ("speech",):
            t = t.to(torch.float32)
        else:
            t = t.to(torch.int32)
        
        # Auto-adjust shape if static target requires it
        if name in target_shapes:
            target_shape = target_shapes[name]
            if name == "speech" and t.ndim == 3 and len(target_shape) == 3:
                # Assumes [B, T, D] layout
                cur_t = t.shape[1]
                tgt_t = target_shape[1]
                if cur_t != tgt_t:
                    if cur_t < tgt_t:
                        t = F.pad(t, (0, 0, 0, tgt_t - cur_t))
                    else:
                        t = t[:, :tgt_t, :]

        out.append(t)
    return out


def main() -> None:
    args = _parse_args()
    onnx_path = Path(args.onnx).expanduser().resolve()
    
    # Removed explicit calib_path resolution here as it depends on args.calib
    
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_names, output_names = _onnx_io_names(onnx_path)
    target_shapes = _get_static_input_shapes(onnx_path)

    from xhquant.api import DeviceType, QuantScheme, create_quant_config, convert_onnx_to_hmonnx, get_root_logger, xhquant_init

    log_file = out_dir / "convert.log"
    xhquant_init(str(log_file), debug=bool(args.debug))

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)
    quant_config.ops_cfg["LayerNorm"] = dict(force_fp32=True)

    if "w_cfg" in quant_config and "quantizer" in quant_config["w_cfg"]:
        # Move calib_metric to outer level of quantizer config, not inside qspec
        quant_config["w_cfg"]["quantizer"]["calib_metric"] = args.calib_metric
            
    if "i_cfg" in quant_config and "quantizer" in quant_config["i_cfg"]:
        quant_config["i_cfg"]["quantizer"]["calib_metric"] = args.calib_metric

    work_dirs = out_dir / "hmonnx"
    work_dirs.mkdir(parents=True, exist_ok=True)
    out_hmonnx = work_dirs / f"{onnx_path.stem}_{DeviceType.XH2a}.onnx"

    if args.calib:
        calib_path = Path(args.calib).expanduser().resolve()
        input_list = _make_input_list(calib_path, input_names, target_shapes)
    else:
        # Use HF dataset
        input_list = _make_calib_data_from_dataset(args, input_names, target_shapes)

    batch_size = 1
    if len(input_list) > 0 and hasattr(input_list[0], "shape"):
        batch_size = input_list[0].shape[0]

    if batch_size > 1:
        print(f"Detected multiple calibration samples (batch={batch_size}). Using multi-batch calibration.")
        
        # Import lower-level APIs
        from xhquant.api.ptq_export_hmonnx import (
            _convert_model_to_quanted_model, 
            convert_quanted_model_to_hmonnx,
        )
        from xhquant.common.types import FrontendType, PrecisionMode
        from xhquant.quantization import ptq_quantize
        import torch

        # 1. Convert to quanted graph (without running PTQ yet)
        # Use the first sample for graph construction and shape inference
        first_sample_args = [t[0:1] if isinstance(t, torch.Tensor) else t for t in input_list]
        
        quanted_graph_module = _convert_model_to_quanted_model(
            str(onnx_path),
            FrontendType.ONNX,
            first_sample_args,
            DeviceType.XH2a,
            quant_config,
            use_ptq=False,
            input_names=input_names,
        )
        
        # 2. Prepare calibration data list (List[List[Tensor]])
        calib_data = []
        for i in range(batch_size):
            sample_args = []
            for t in input_list:
                if isinstance(t, torch.Tensor):
                    # Slice the batch dimension, keeping it 1
                    sample_args.append(t[i:i+1].cpu())
                else:
                    sample_args.append(t)
            calib_data.append(sample_args)
            
        execution_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # 3. Run PTQ with multiple samples
        ptq_quantize(
            quanted_graph_module, 
            calib_data, 
            PrecisionMode.ALIGNED, 
            execution_device
        )
        
        # 4. Export to HMONNX
        convert_quanted_model_to_hmonnx(
            quanted_graph_module,
            first_sample_args, # Use first sample for tracing
            str(out_hmonnx),
            input_names,
            output_names,
        )
    else:
        convert_onnx_to_hmonnx(
            str(onnx_path),
            input_list,
            DeviceType.XH2a,
            str(out_hmonnx),
            quant_config=quant_config,
            input_names=input_names,
            output_names=output_names,
        )

    logger = get_root_logger()
    logger.info(f"Converted to HMONNX: {out_hmonnx}")
    print(f"Converted to HMONNX: {out_hmonnx}")


if __name__ == "__main__":
    main()
