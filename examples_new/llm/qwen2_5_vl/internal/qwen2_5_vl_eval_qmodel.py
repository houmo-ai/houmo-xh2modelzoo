from vlmeval.dataset import build_dataset
from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin
from vlmeval.config import qwen2vl_series, Qwen2VLChat
from vlmeval.smp import dump, tabulate, pd
import json
import argparse
import os
import datetime
import os.path as osp
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm
from torch import Tensor

from qwen_vl_utils import process_vision_info
from transformers import AutoConfig
from xh_model_zoo.xh_llm.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration

from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger  # isort:skip
from xh_model_zoo_new.utils import MemoryTracker  # isort:skip
from xh_model_zoo_new.utils import TimeProfiler  # isort:skip
from xh_model_zoo_new.xh_llm.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from xh_model_zoo_new.xh_llm.models.qwen2_5_vl.qwen2_5_vl_demo import (
    Qwen2_5_VLLLMInferHMONNXImpl,
    Qwen2_5_VLHFCompatible,
)
from xh_model_zoo_new.xh_llm.models.qwen2_5_vl.qwen2_5_vl_converter_v2 import (
    Qwen25VLConverter,
    Qwen25VLConverterConfig,
    VisionConfig,
)

torch.set_grad_enabled(False)


class XH2LLM(Qwen2VLChat):
    def __init__(
        self,
        model: Qwen2_5_VLHFCompatible,
        processor,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        # if True, will try to only extract stuff in the last \boxed{}.
        post_process: bool = False,
        verbose: bool = False,
        use_audio_in_video: bool = False,
        **kwargs,
    ):
        Qwen2VLPromptMixin.__init__(self, use_custom_prompt=use_custom_prompt)
        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        if self.total_pixels and self.total_pixels > 24576 * 28 * 28:
            print(
                "The total number of video tokens might become too large, resulting in an overly long input sequence. We recommend lowering **total_pixels** to below **24576 × 28 × 28**."
            )  # noqa: E501
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process

        self.use_vllm = kwargs.get("use_vllm", False)
        self.use_lmdeploy = kwargs.get("use_lmdeploy", False)
        self.fps = kwargs.pop("fps", 2)
        self.nframe = kwargs.pop("nframe", 128)
        if self.fps is None and self.nframe is None:
            print(
                "Warning: fps and nframe are both None, \
                  using default nframe/fps setting in qwen-vl-utils/qwen-omni-utils, \
                  the fps/nframe setting in video dataset is omitted"
            )
        self.use_audio_in_video = use_audio_in_video
        self.FRAME_FACTOR = 2
        if self.fps is None and self.nframe is None:
            print(
                "Warning: fps and nframe are both None, \
                  using default nframe/fps setting in qwen-vl-utils/qwen-omni-utils, \
                  the fps/nframe setting in video dataset is omitted"
            )
        self.use_audio_in_video = use_audio_in_video
        self.FRAME_FACTOR = 2
        self.model = model
        self.processor = processor
        torch.cuda.empty_cache()


def to_device(inputs, device):
    if isinstance(inputs, Tensor):
        return inputs.to(device)
    elif isinstance(inputs, (list, tuple)):
        return type(inputs)([to_device(x, device) for x in inputs])
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    # elif isinstance(inputs, QTensor):
    #     return inputs.to(device)
    else:
        return inputs


@torch.no_grad()
def eval_model(model, dataset, dataset_name, model_name, verbose=False):
    """
    Evaluate the model on the dataset.
    Args:
        model: The model to evaluate.
        dataset: The dataset to evaluate on.
        dataset_name: The name of the dataset.
        model_name: The name of the model.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_file = f"output/{model_name}_{dataset_name}_{timestamp}.xlsx"
    os.makedirs("output", exist_ok=True)
    res = {}
    lt = len(dataset.data)

    struct_data_file = f"output/{model_name}_{dataset_name}_{timestamp}_struct.json"

    struct_data = []
    data_indices = [i for i in dataset.data["index"]]
    for i in tqdm(range(lt)):
        idx = dataset.data.iloc[i]["index"]
        if idx in res:
            continue

        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])
        response = model.generate(message=struct, dataset=dataset_name)
        if verbose:
            print(response, flush=True)
        res[idx] = response

        struct_data.append(
            {
                "index": idx,
                "struct": struct,
                "response": response,
            }
        )
    dump(struct_data, struct_data_file)
    res = {k: res[k] for k in data_indices}

    data = dataset.data
    for x in data["index"]:
        assert x in res
    data["prediction"] = [str(res[x]) for x in data["index"]]
    if "image" in data:
        data.pop("image")

    dump(data, result_file)

    judge_kwargs = dict()
    eval_results = dataset.evaluate(result_file, **judge_kwargs)
    if eval_results is not None:
        assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
        print(f"The evaluation of model {model_name} x dataset {dataset_name} has finished! ")
        print("Evaluation Results:")
    if isinstance(eval_results, dict):
        print("\n" + json.dumps(eval_results, indent=4))
    elif isinstance(eval_results, pd.DataFrame):
        if len(eval_results) < len(eval_results.columns):
            eval_results = eval_results.T
        print("\n" + tabulate(eval_results))


def load(export_dir):
    with open(osp.join(export_dir, "meta.json"), "r") as f:
        meta_info = json.load(f)
    # Paths exported by converter
    vision_path = osp.join(export_dir, meta_info["vision_onnx"])
    prefill_path = osp.join(export_dir, meta_info["prefill_onnx"])
    decode_path = osp.join(export_dir, meta_info["decode_onnx"])
    hf_config_dir = osp.join(export_dir, meta_info["hf_config"])
    token_embedding_file = osp.join(export_dir, meta_info["token_embedding_file"])
    wrap_cfg = meta_info["wrap_cfg"]

    hf_config = AutoConfig.from_pretrained(hf_config_dir)
    # Load embedding weights exported together with ONNX to avoid touching native weights.
    token_embedding_state = torch.load(token_embedding_file, weights_only=True)
    vocab_size, hidden_size = token_embedding_state["weight"].shape
    token_embedding = nn.Embedding(vocab_size, hidden_size)
    token_embedding.load_state_dict(token_embedding_state)

    native_model = Qwen2_5_VLForConditionalGeneration._from_config(hf_config)
    processor = Qwen2_5_VLProcessor.from_pretrained(hf_config_dir)

    model = (
        Qwen2_5_VLHFCompatible.from_hmonnx(
            vision_path=vision_path,
            prefill_path=prefill_path,
            decoder_path=decode_path,
            wrap_cfg=wrap_cfg,
            hf_config=hf_config,
            token_embedding=token_embedding,
            native_model_or_path=native_model,
            processor=processor,
        )
        .cuda()
        .half()
        .eval()
    )
    return processor, model


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    ops = dict(
        MatMul=dict(
            act_scheme=dict(
                bits=8,
                fp_mode="sefp",
            ),
            act_schema_2=dict(
                bits=16,
                fp_mode="sefp",
            ),
        )
    )
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type, ops=ops)
    config = Qwen25VLConverterConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        quant_scheme=quant_scheme,
        quant_weight=None,
        # gptqmodel_cfg=args.use_gptqmodel,
        max_pe_length=args.max_pe_length,
        vision_config=VisionConfig(
            image_max_size_h=args.image_max_size_h,
            image_max_size_w=args.image_max_size_w,
            image_max_size_t=args.image_max_size_t,
            temporal_patch_size=args.temporal_patch_size,
            patch_size=args.patch_size,
            sample_image_path=args.sample_image_path,
        ),
    )
    prefix = f"{model_name}-{target_device}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)

    C = Qwen25VLConverter(hf_model_path, config)
    native_model = C.native_model
    quanted_llm_model = C.quanted_llm_model

    # from transformers import Qwen2_5_VLProcessor
    # processor = Qwen2_5_VLProcessor.from_pretrained(hf_model_path)
    if not args.align:
        quanted_llm_model.enable_fast_precision_mode()
    hf_compatible_model = Qwen2_5_VLHFCompatible.from_qmodel(
        quanted_llm_model,
        C.native_model.visual,
        C.hf_config,
        C.wrap_cfg,
        C.token_embedding,
        C.native_model,
        C.processor,
        is_native_vision_model=True,
    )
    hf_compatible_model.cuda().half().eval()
    hf_compatible_model.demo("Describe this image.", "data/images/ILSVRC2012_val_00002031.JPEG")

    model = XH2LLM(
        hf_compatible_model,
        C.processor,
        "Qwen2.5-VL",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        max_new_tokens=2048,
    )
    eval_model(model, build_dataset(args.dataset_name), args.dataset_name, "Qwen2.5-VL-7B", False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--model", type=str, default="/data01/datasets/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--context-length", type=int, default=8192, help="max sequence length")
    parser.add_argument("--max_pe_length", type=int, default=32768, help="max pe length")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument("--image_max_size_h", type=int, default=448, help="image max size height")
    parser.add_argument("--image_max_size_w", type=int, default=448, help="image max size width")
    parser.add_argument(
        "--image_max_size_t",
        type=int,
        default=2,
        help="if image, temporal max size is 2, if video, temporal max size is fps",
    )
    parser.add_argument("--patch_size", type=int, default=14, help="patch size")
    parser.add_argument("--temporal_patch_size", type=int, default=2, help="temporal patch size")
    parser.add_argument(
        "--sample_image_path",
        type=str,
        default="data/images/qwen2_vl_demo.jpeg",
        help="sample image path for generate golden",
    )
    # for eval
    parser.add_argument("--dataset-name", type=str, default="CMMMU_VAL", help="dataset name")
    parser.add_argument("--align",type=bool,default=False,help="align mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
