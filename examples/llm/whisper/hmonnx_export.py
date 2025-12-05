import argparse
import os
import tempfile
from pathlib import Path
from tkinter import NO

from librosa import cache
import onnx
import onnxsim
from sympy import N, false
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
    convert_fx_model_to_quanted_model,
    convert_dynamo_model_to_hmonnx

)
from xhquant.frontend.convert import to_frontend_graph

from xhquant.utils.config import Config, ConfigDict
from xh_model_zoo.xh_llm.models.whisper._model_opt import *
from xhquant.patch.core import RewriterContext
from xh_model_zoo.xh_llm.models.builder import wrap_llm_model

from xhquant.api.ptq_export_hmonnx import _convert_model_to_quanted_model, convert_quanted_model_to_hmonnx, FrontendType
from xhquant.core.datatype_mapping import TORCH_DTYPE_TO_FAKE_DTYPE
from xhquant.common.types import PrecisionMode

class Decoder(nn.Module):
    def __init__(self, model, proj_out, config=None):
        super().__init__()
        self.config = config
        self.model = model
        self.proj_out = proj_out

    def forward(self, decoder_input_ids, cache_position, past_len, current_len, mask_atten=None, k_cache_list=None, v_cache_list=None, k_list=None, v_list=None):  # , cache_position
        hidden_state, k_cache_list, v_cache_list = self.model.decoder( # [1,1,1024]
            input_ids=decoder_input_ids,
            k_list=k_list,
            v_list=v_list,
            position_ids=cache_position,
            k_cache=k_cache_list,
            v_cache=v_cache_list,
            past_len=past_len,
            current_len=current_len,
            mask_atten=mask_atten,
            # past_key_values_length=past_key_len,
        )
        output = self.proj_out(hidden_state)
        return output, k_cache_list, v_cache_list 


def main(args):
    # load model and processor
    processor = WhisperProcessor.from_pretrained("/data02/datasets/whisper_medium")
    model = WhisperForConditionalGeneration.from_pretrained("/data02/datasets/whisper_medium")
    model.config.forced_decoder_ids = None

    model.model.encoder.decoder_m = model.model.decoder

    # load dummy dataset and read audio files
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[1]["audio"]
    # input_features = processor(
    #     sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
    # ).input_features # -142334.8125
    # [1,80,3000]

    work_dirs = Path("work_dirs") / "whisper" / "encoder"
    work_dirs.mkdir(exist_ok=True, parents=True)
    onnx_file = work_dirs / "whisper_meduim.onnx"

    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_file = work_dirs / "hmonnx" / f"whisper_meduim_xh2a_{quant_type}.onnx"
    golden_path = work_dirs / "hmonnx/golden"

    # input_features = torch.randn(1, 80, 3000)
    input_features = torch.load("/data01/home/xuchen/xh2/xh2_model_zoo/work_dirs/whisper/input.pt")

    # encoder ============================================
    if not Path(onnx_file).exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            with RewriterContext(None, backend='onnxruntime'):
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

    output_names = []
    for i in range(24):
        output_names.append(f"key_state_{i}")
    for i in range(24):
        output_names.append(f"value_state_{i}")
        
    if not Path(hmonnx_file).exists():
        convert_onnx_to_hmonnx(
            str(onnx_file),
            [input_features],
            DeviceType.XH2a,
            hmonnx_file,
            quant_config=quant_config,
            input_names=["input_features"],
            output_names=output_names,
        )

    if args.gen_golden and not Path(golden_path).exists():
        session = HMONNXGoldenInference(hmonnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = work_dirs / "hmonnx/golden"
        session.step = 0
        session(input_features.half().to("cuda"))


    # decoder ===========================================
    name = "decoder"

    work_dirs = Path("work_dirs") / "whisper" / name
    quant_config = create_quant_config(quant_scheme)
    onnx_file = work_dirs / f"whisper_meduim_{name}.onnx"
    hmonnx_file = work_dirs / "hmonnx" / f"whisper_meduim_{name}_xh2a_{quant_type}.onnx"
    golden_path = work_dirs / "hmonnx / golden"

    # decoder_input_ids = torch.randint(0, 10, (1, 1)) # (1,1)  (1,4)
    decoder_input_ids = torch.tensor([[2221]])
    cache_position = torch.tensor([[4]])
    # cache_position = torch.tensor([[0, 1, 2, 3]])
    encoder_outputs_kv = torch.randn([1, 1500, 16, 64]).transpose(1, 2).contiguous()
    past_ket_length = torch.tensor([0])
    past_len = torch.tensor([4])

    k_cache_past, v_cache_past = torch.load("/data01/home/xuchen/xh2/xh2_model_zoo/work_dirs/whisper/kv_cache.pt", weights_only=False)
    k_cache = [ torch.ones([1, 16, 1024, 64], dtype=torch.float16)*(-65504) for i in range(24) ]
    v_cache = [ torch.ones([1, 16, 1024, 64], dtype=torch.float16)*(-65504) for i in range(24) ]
    for i in range(24):
        k_cache[i][:, :, :4, :] = k_cache_past[i].half()
        v_cache[i][:, :, :4, :] = v_cache_past[i].half()

    k_list = []
    v_list = []
    # for i in range(24):
    #     kv_list.append((encoder_outputs_kv, encoder_outputs_kv))
    kv = torch.load("/data01/home/xuchen/xh2/xh2_model_zoo/work_dirs/whisper/kv_data.pt", weights_only=False)
    for i in range(24):
        k_list.append( torch.tensor(kv[i*2]).half()   )
        v_list.append( torch.tensor(kv[i*2+1]).half() )
    
    inputs_names = ["decoder_input_ids", "cache_position", "past_len", "current_len", "mask_atten"] # , "past_key_length"

    for i in range(24):
        inputs_names.append(f"k_cache_{i}")
    for i in range(24):
        inputs_names.append(f"v_cache_{i}")

    inputs_names += output_names

    model_cus = Decoder(model.model,  model.proj_out, config=model.config)


    output_names = ["logits"]
    for i in range(24):
        output_names.append(f"newk_cache_{i}")
    for i in range(24):
        output_names.append(f"newv_cache_{i}")

    # encoder ============================================

    cache_len = decoder_input_ids.shape[0]
    mask_atten = torch.ones( ([1, 16, cache_len, 1024]) ).half()
    mask_atten[:,:,:,  past_len+cache_len: ] *= -65504
    current_len = torch.tensor([cache_len])
    
    # warp
    warp_inp = (
        decoder_input_ids, cache_position, past_len, current_len, mask_atten, k_cache, v_cache, k_list, v_list
    ) 

    trace_inp = (decoder_input_ids, cache_position, past_len, current_len, mask_atten) + \
        tuple(k_cache) + tuple(v_cache) + tuple(k_list) + tuple(v_list)


    with RewriterContext(None, backend='onnxruntime'):
        warp_model_cus =  wrap_llm_model(model_cus)
        warp_model_cus = warp_model_cus.half()
        # output, k_cache_list, v_cache_list = warp_model_cus(*warp_inp)
        fronted_graph_module = to_frontend_graph(
            warp_model_cus, 'DynamoFX', warp_inp
        )

        # output, k_cache_list, v_cache_list = frontend_model(*warp_inp)

    '''
    if not Path(onnx_file).exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            with RewriterContext(None, backend='onnxruntime'):
                temp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
                torch.onnx.export(
                    model_cus,
                    (decoder_input_ids, cache_position, current_len, k_cache, v_cache, kv_list),  # , cache_position
                    temp_onnx_file,
                    input_names=inputs_names,  # , "cache_position"
                    output_names=output_names,
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
    '''

    hm_inputs = [decoder_input_ids.to(torch.int32), cache_position.to(torch.int32), past_len.to(torch.int32), current_len.to(torch.int32), mask_atten,] 
    for i in range(24):
        hm_inputs.append( k_cache[i] )     

    for i in range(24):
        hm_inputs.append( v_cache[i] )   

    for i in range(24):
        hm_inputs.append( k_list[i] )
        hm_inputs.append( v_list[i] )

    if not Path(hmonnx_file).exists():
         
        # quanted_graph_module = _convert_model_to_quanted_model(
        #     model, FrontendType.DynamoFX, warp_inp, DeviceType.XH2a, quant_config
        # )

        _input_names = fronted_graph_module.get_input_names()
        if quant_config is None:
            quant_config = ConfigDict()
        if isinstance(quant_config, dict):
            quant_config = ConfigDict(quant_config)
        
        if "inputs" not in quant_config:
            quant_config.inputs = ConfigDict()

        ## 将输入的List展开
        input_args = []
        for arg in warp_inp:
            if isinstance(arg, (list, tuple)):
                input_args.extend(arg)
            else:
                input_args.append(arg)

        assert len(_input_names) == len(input_args), f"input_names: {len(_input_names)}, input_args: {len(input_args)}"
        for input_name, input_arg in zip(_input_names, input_args):
            input_qconfig = ConfigDict(
                dict(
                    quantizer=dict(
                        qspec=dict(),
                    ),
                )
            )
            if isinstance(input_arg, torch.Tensor):
                if input_arg.dtype in TORCH_DTYPE_TO_FAKE_DTYPE:
                    input_qconfig.quantizer.qspec.fake_dtype = TORCH_DTYPE_TO_FAKE_DTYPE[input_arg.dtype]
                else:
                    raise ValueError(f"Unsupported dtype: {input_arg.dtype}")

            quant_config.inputs[input_name] = input_qconfig

        quanted_graph_module = to_quant_graph(fronted_graph_module, DeviceType.XH2a.name, quant_config)
        execution_devce = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ptq_quantize(quanted_graph_module, [input_args], PrecisionMode.ALIGNED, execution_devce)

        convert_quanted_model_to_hmonnx(
            quanted_graph_module,
            warp_inp,
            hmonnx_file,
            inputs_names,
            output_names,
        )


        # convert_dynamo_model_to_hmonnx(
        #     warp_model_cus,
        #     hm_inputs,
        #     DeviceType.XH2a,
        #     hmonnx_file,
        #     quant_config=quant_config,
        #     input_names=inputs_names,  # , "cache_position"
        #     output_names=output_names,
        # )


        # convert_onnx_to_hmonnx(
        #     str(onnx_file),
        #     hm_inputs,
        #     DeviceType.XH2a,
        #     hmonnx_file,
        #     quant_config=quant_config,
        #     input_names=inputs_names,  # , "cache_position"
        #     output_names=output_names,
        # )

    if args.gen_golden and not Path(golden_path).exists():
        session = HMONNXGoldenInference(hmonnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = work_dirs / "hmonnx/golden"
        session.step = 0
        session(*hm_inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--onnx", type=str, default="data/model_zoo2/houmo/yolo12m/yolo12m.onnx")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type, default is w8a8")
    parser.add_argument("--gen_golden", action="store_true", help="generate golden data")
    args = parser.parse_args()
    main(args)
