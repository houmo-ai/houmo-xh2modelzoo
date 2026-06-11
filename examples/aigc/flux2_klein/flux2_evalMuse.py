import argparse
import base64
import io
import sys
from pathlib import Path
from typing import List, Optional

import torch
from evalscope import TaskConfig, run_task
from evalscope.api.messages import ChatMessage
from evalscope.api.messages.content import ContentImage
from evalscope.api.model import ChatCompletionChoice, ModelAPI, ModelOutput
from evalscope.api.model.generate_config import GenerateConfig
from evalscope.api.registry import register_model_api
from evalscope.constants import EvalType, ModelTask

from xh_model_zoo.xh_aigc.models.flux2_klein import (
    Flux2KleinHMONNXPipeline,
    attach_flux2_klein_hmonnx_components,
)


FLUX2_HMONNX_EVAL_TYPE = "flux2_hmonnx_text2image"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def import_flux2_klein_pipeline() -> None:
    vendored_diffusers = str(_repo_root() / "data" / "difs" / "diffusers-main" / "src")
    if vendored_diffusers in sys.path:
        sys.path.remove(vendored_diffusers)
    for module_name in list(sys.modules):
        if module_name == "diffusers" or module_name.startswith("diffusers."):
            del sys.modules[module_name]
    from diffusers import Flux2KleinPipeline  # noqa: F401


def parse_components(value: str) -> set[str]:
    components = {item.strip().lower() for item in value.split(",") if item.strip()}
    aliases = {"te": "text_encoder", "text": "text_encoder", "trans": "transformer", "vae_decoder": "vae"}
    return {aliases.get(item, item) for item in components}


@register_model_api(FLUX2_HMONNX_EVAL_TYPE)
class Flux2KleinHMONNXEvalMuseAPI(ModelAPI):
    """EvalScope TEXT2IMAGE adapter for flux2_klein_hmonnx_demo.py generation path."""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        meta: Optional[str] = None,
        root_meta: Optional[str] = None,
        components: str = "text_encoder,transformer,vae",
        torch_dtype: str = "torch.float16",
        **kwargs,
    ) -> None:
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config, **kwargs)
        if meta is None:
            raise ValueError("model_args 需要提供 meta，指向 HMONNX 导出的 text_encoder meta.json 或根 meta.json")
        import_flux2_klein_pipeline()

        self.meta_path = Path(meta)
        self.root_meta_path = Path(root_meta) if root_meta is not None else self.meta_path
        self.components = parse_components(components)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch_dtype == "torch.float16" and self.device.type == "cuda" else torch.float32

        pipe = Flux2KleinHMONNXPipeline.from_pretrained(model_name, torch_dtype=self.dtype)
        pipe = pipe.to(self.device)
        pipe, _ = attach_flux2_klein_hmonnx_components(
            pipe,
            components=self.components,
            meta_path=self.meta_path,
            root_meta_path=self.root_meta_path,
            device=self.device,
            dtype=self.dtype,
        )
        self.pipe = pipe

    def generate(
        self,
        input: List[ChatMessage],
        tools,
        tool_choice,
        config: GenerateConfig,
    ) -> ModelOutput:
        del tools, tool_choice
        prompt = input[0].text
        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        seed = 1 if config.seed is None else int(config.seed)
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                height=config.height or 1024,
                width=config.width or 1024,
                guidance_scale=1.0 if config.guidance_scale is None else float(config.guidance_scale),
                num_inference_steps=4 if config.num_inference_steps is None else int(config.num_inference_steps),
                generator=generator,
                use_hmonnx_text_encoder="text_encoder" in self.components,
                use_hmonnx_transformer="transformer" in self.components,
                use_hmonnx_vae="vae" in self.components,
            ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        choice = ChatCompletionChoice.from_content([ContentImage(image=image_base64)])
        return ModelOutput(model=self.model_name, choices=[choice])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="/data02/datasets/flux-4b")
    parser.add_argument("--meta", type=str, required=True, help="HMONNX 导出的 text_encoder meta.json 或根 meta.json")
    parser.add_argument("--root-meta", type=str, default=None, help="HMONNX 导出的根 meta.json，用于 transformer/vae")
    parser.add_argument("--components", type=str, default="text_encoder,transformer,vae")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=float, default=None, help="EvalMuse 样本数；调试可设 1/5/10")
    parser.add_argument("--work-dir", type=str, default="outputs/flux2_klein_hmonnx_evalmuse")
    parser.add_argument("--analysis-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_cfg = TaskConfig(
        model=args.model,
        model_id="flux2-klein-hmonnx",
        model_task=ModelTask.IMAGE_GENERATION,
        eval_type=FLUX2_HMONNX_EVAL_TYPE,
        model_args={
            "meta": args.meta,
            "root_meta": args.root_meta,
            "components": args.components,
            "torch_dtype": "torch.float16",
        },
        datasets=["evalmuse"],
        generation_config={
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "batch_size": 1,
        },
        limit=args.limit,
        eval_batch_size=1,
        work_dir=args.work_dir,
        analysis_report=args.analysis_report,
    )
    run_task(task_cfg=task_cfg)


if __name__ == "__main__":
    main()
