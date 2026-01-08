from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
import torch 
import math
from curses import meta
from xh_model_zoo.xh_llm.models.qwen_image.qwen2_5_vl_converter import Qwen2_5_VLConverterXH2a
from xh_model_zoo.xh_llm.models.qwen_image.pipeline_cus import cus_QwenImagePipeline
from pathlib import Path
from xh_model_zoo.xh_llm.models.qwen2_5_vl import Qwen2_5_VLConvertConfig, VisualConfig
import argparse
from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger, HMONNXGoldenInference
from xh_model_zoo.xh_llm.models.builder import wrap_llm_model


def main(args):
    model_name = "/data02/datasets/qwen-image"

    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    ops=dict(MatMul=dict(
                act_scheme=dict(
                    bits=8,
                    fp_mode="sefp",
                ),
                act_schema_2=dict(
                    bits=16,
                    fp_mode="sefp",
                ),))

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type, ops=ops)
    config = Qwen2_5_VLConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        gptqmodel_cfg=args.use_gptqmodel,
        max_pe_length=args.max_pe_length,
        visual_config=VisualConfig(
            image_max_size_h=args.image_max_size_h,
            image_max_size_w=args.image_max_size_w,
            image_max_size_t=args.image_max_size_t,
            temporal_patch_size=args.temporal_patch_size,
            patch_size=args.patch_size,
            sample_image_path=args.sample_image_path,
        ),
    )


    # From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),  # We use shift=3 in distillation
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),  # We use shift=3 in distillation
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,  # set shift_terminal to None
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    pipe = DiffusionPipeline.from_pretrained(
        "/data02/datasets/qwen-image", scheduler=scheduler, torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.load_lora_weights(
        "/data02/datasets/qwen_image_fp8/Qwen-Image-fp8-e4m3fn-Lightning-4steps-V1.0-fp32.safetensors"
    )

    prompt = "a tiny astronaut hatching from an egg on the moon, Ultra HD, 4K, cinematic composition."
    negative_prompt = " "

    # from xh_model_zoo.xh_llm.models.qwen_image._mmdit_model_impl import register_wrap_cls as llm_register_wrap_cls 
    # llm_register_wrap_cls(pipe.transformer)
    # wraped_llm_model = wrap_llm_model(pipe.transformer, {})
    # wraped_llm_model.cuda()
    # wraped_llm_model.to(torch.float16)

    work_dir = Path("work_dirs") / "qwen-image"
    work_dir.mkdir(exist_ok=True, parents=True)

    text_encoder_prefill = "work_dirs/qwen-image/hmonnx/qwen_image_text_encoder-XH2a-w8a8h1_sefp-llm-prefill.onnx"
    text_encoder = HMONNXGoldenInference(text_encoder_prefill)
    text_encoder.exec_device = torch.device("cuda:0")

    vae_hmonnx_path = "work_dirs/qwen-image/hmonnx/qwen_image_vae-XH2a-w8a8h1_sefp.onnx"
    vae = HMONNXGoldenInference(vae_hmonnx_path)
    vae.exec_device = torch.device("cuda:1")
    pipe.text_encoder = None
    pipe.vae = None
    del pipe.text_encoder
    del pipe.vae
    torch.cuda.empty_cache()

    xhmodel = cus_QwenImagePipeline.to_hf_compatible(pipe, text_encoder=text_encoder, vae=vae, transformers=None)

    image = xhmodel(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        num_inference_steps=4,
        true_cfg_scale=1.0,
        generator=torch.manual_seed(0),
        meta_info="work_dirs/qwen-image/meta.json",
    ).images[0]
    image.save("qwen_fewsteps.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--model", type=str, default="weights/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--context-length", type=int, default=2048, help="max sequence length")
    parser.add_argument("--max_pe_length", type=int, default=32768, help="max pe length")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument("--image_max_size_h", type=int, default=448, help="image max size height")
    parser.add_argument("--image_max_size_w", type=int, default=448, help="image max size width")
    parser.add_argument("--image_max_size_t", type=int, default=2, help="if image, temporal max size is 2, if video, temporal max size is fps")
    parser.add_argument("--patch_size", type=int, default=14, help="patch size")
    parser.add_argument("--temporal_patch_size", type=int, default=2, help="temporal patch size")
    parser.add_argument("--sample_image_path", type=str, default="data/images/qwen2_vl_demo.jpeg", help="sample image path for generate golden")
    parser.add_argument("--use_gptqmodel", action="store_true", help="use gptqmodel quanted model")
    parser.add_argument(
        "--quant_weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    args = parser.parse_args()
    main(args)
