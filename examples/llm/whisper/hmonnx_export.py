import argparse
import os
import tempfile
from pathlib import Path

import onnx
import onnxsim
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    ptq_quantize,
    to_frontend_graph,
    to_quant_graph,
    xhquant_init,
)
from xhquant.utils.config import Config, ConfigDict


class Decoder(nn.Module):
    def __init__(self, model, config=None):
        super().__init__()
        self.config = config
        self.model = model

    def forward(self, decoder_input_ids, encoder_outputs):  # , cache_position
        hidden_state = self.model(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            # cache_position=cache_position,
        )
        return hidden_state


def main(args):
    # load model and processor
    processor = WhisperProcessor.from_pretrained("/data02/datasets/whisper_medium")
    model = WhisperForConditionalGeneration.from_pretrained("/data02/datasets/whisper_medium")
    model.config.forced_decoder_ids = None

    # load dummy dataset and read audio files
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[0]["audio"]
    input_features = processor(
        sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
    ).input_features
    # [1,80,3000]

    work_dirs = Path("work_dirs") / "whisper" / "encoder"
    work_dirs.mkdir(exist_ok=True, parents=True)
    onnx_file = work_dirs / "whisper_meduim.onnx"

    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_file = work_dirs / "hmonnx" / f"whisper_meduim_xh2a_{quant_type}.onnx"
    golden_path = work_dirs / "hmonnx/golden"

    input_features = torch.randn(1, 80, 3000)

    # encoder ============================================
    if not Path(onnx_file).exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
            torch.onnx.export(
                model.model.encoder,
                input_features,  # inputs[0], #
                temp_onnx_file,
                input_names=["input_features"],
                output_names=[
                    "hidden_state",
                ],
            )
            onnx_model = onnx.load(temp_onnx_file)
            onnx_model_sim, checked = onnxsim.simplify(onnx_model)
            if checked:
                onnx_model = onnx_model_sim
    else:
        onnx_model = onnx.load(onnx_file)

    if not os.path.exists(onnx_file):
        onnx.save(
            onnx_model,
            onnx_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{Path(onnx_file).stem}_external_data",
        )

    if not Path(hmonnx_file).exists():
        convert_onnx_to_hmonnx(
            str(onnx_file),
            [input_features],
            DeviceType.XH2a,
            hmonnx_file,
            quant_config=quant_config,
            input_names=["input_ids"],
            output_names=[
                "hidden_state",
            ],
        )

    if args.gen_golden and not Path(golden_path).exists():
        session = HMONNXGoldenInference(hmonnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = work_dirs / "hmonnx/golden"
        session.step = 0
        session(input_features.half().to("cuda"))

    work_dirs = Path("work_dirs") / "whisper" / "decoder"
    quant_config = create_quant_config(quant_scheme)
    onnx_file = work_dirs / "whisper_meduim_decoder.onnx"
    hmonnx_file = work_dirs / "hmonnx" / f"whisper_meduim_decoder_xh2a_{quant_type}.onnx"
    golden_path = work_dirs / "hmonnx / golden"

    decoder_input_ids = torch.randint(0, 10, (1, 4))
    encoder_outputs = torch.randn(1, 1500, 1024)
    cache_position = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

    model_cus = Decoder(model.model.decoder, config=model.config)

    # encoder ============================================
    if not Path(onnx_file).exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
            torch.onnx.export(
                model_cus,
                (decoder_input_ids, encoder_outputs),  # , cache_position
                temp_onnx_file,
                input_names=["decoder_input_ids", "encoder_outputs"],  # , "cache_position"
                # output_names=[
                #     "hidden_state",
                # ],
            )
            onnx_model = onnx.load(temp_onnx_file)
            onnx_model_sim, checked = onnxsim.simplify(onnx_model)
            if checked:
                onnx_model = onnx_model_sim
    else:
        onnx_model = onnx.load(onnx_file)

    if not os.path.exists(onnx_file):
        onnx.save(
            onnx_model,
            onnx_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{Path(onnx_file).stem}_external_data",
        )

    if not Path(hmonnx_file).exists():
        convert_onnx_to_hmonnx(
            str(onnx_file),
            [decoder_input_ids, encoder_outputs],
            DeviceType.XH2a,
            hmonnx_file,
            quant_config=quant_config,
            input_names=["decoder_input_ids", "encoder_outputs"],  # , "cache_position"
            # output_names=[
            #     "hidden_state",
            # ],
        )

    if args.gen_golden and not Path(golden_path).exists():
        session = HMONNXGoldenInference(hmonnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = work_dirs / "hmonnx/golden"
        session.step = 0
        session(decoder_input_ids.to(torch.int32).to("cuda"), encoder_outputs.half().to("cuda"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--onnx", type=str, default="data/model_zoo2/houmo/yolo12m/yolo12m.onnx")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type, default is w8a8")
    parser.add_argument("--gen_golden", action="store_true", help="generate golden data")
    args = parser.parse_args()
    main(args)
