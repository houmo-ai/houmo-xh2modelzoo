import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...workflows.base import BaseLLMWorkflow
from ...workflows.result import ExportResult, QuantResult


class XHUnlimitedOCRWorkflow(BaseLLMWorkflow):
    expected_model_config_cls_name = "XHUnlimitedOCRModelConfig"
    expected_model_cls_name = "XHUnlimitedOCRModel"

    def __init__(
        self,
        model_dir: str,
        config_path: str,
        seed: int = 1024,
        debug: bool = False,
    ):
        super().__init__(model_dir=model_dir, config_path=config_path, seed=seed, debug=debug)
        self._validate_export_mode(self.workflow_config.export["model"])
        self._validate_model_dir_override()
        self._validate_calibration_source(self.workflow_config.export["model"])

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        self._validate_export_mode(workflow_config.export["model"])
        self._validate_calibration_source(workflow_config.export["model"])
        return super().export(
            quant_result=quant_result,
            output_dir=output_dir,
            device=device,
            config_overrides=config_overrides,
        )

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any,
    ) -> str:
        from transformers import TextStreamer

        from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
        from xhquant.api import get_xhquant_logger
        from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler

        meta_file = self._find_golden_meta_file(export_result)
        image_path, prompt = self.build_input(input_messages)
        logger = get_xhquant_logger()
        hmonnx_model = AutoLLMHONNXModel.from_pretrained(meta_file)
        if type(hmonnx_model).__name__ != "XHUnlimitedOCRHMONNXModel":
            raise TypeError(
                "Expected model type XHUnlimitedOCRHMONNXModel, "
                f"but got {type(hmonnx_model).__name__}"
            )

        processor = hmonnx_model.get_tf_processor()
        tokenizer = processor.tokenizer
        model_inputs = processor.process(prompt, image_path, device=device)
        hmonnx_model.to(device)
        hmonnx_model.enable_golden = True
        logger.warning("Golden outputs should be generated in aligned precision for stability.")

        contexts = [
            TimeProfiler("hmonnx_generate_golden", logger),
            MemoryTracker(device=device, name="generate_golden", logger=logger),
            LLMInferenceContextManager(hmonnx_model),
        ]
        with ContextManagers(contexts):
            generated_ids = hmonnx_model.generate(
                input_ids=model_inputs["input_ids"],
                images_ori=model_inputs["images_ori"],
                images_crop=model_inputs.get("images_crop"),
                images_seq_mask=model_inputs["images_seq_mask"],
                images_spatial_crop=model_inputs["images_spatial_crop"],
                max_new_tokens=2,
                streamer=TextStreamer(tokenizer),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(model_inputs["input_ids"], generated_ids, strict=False)
        ]
        output_text = tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        logger.info(f"{'-' * 20} Golden output {'-' * 20}")
        logger.info(f"{output_text}")
        return meta_file

    @staticmethod
    def build_input(input_messages: Any) -> tuple[str, str]:
        if not isinstance(input_messages, Mapping):
            raise ValueError("Unlimited-OCR input_messages must be a mapping with 'image' and 'prompt'")
        image_path = input_messages.get("image")
        prompt = input_messages.get("prompt", input_messages.get("text"))
        if not isinstance(image_path, str) or not image_path:
            raise ValueError("Unlimited-OCR input_messages must contain a non-empty 'image'")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Unlimited-OCR input_messages must contain a non-empty 'prompt'")
        return image_path, prompt

    def _validate_model_dir_override(self) -> None:
        env_model_dir = os.environ.get("UNLIMITED_OCR_HF_MODEL")
        if not env_model_dir:
            return
        env_model_dir = os.path.abspath(os.path.normpath(env_model_dir))
        if env_model_dir != self.model_dir:
            raise ValueError(
                "UNLIMITED_OCR_HF_MODEL conflicts with the workflow model_dir: "
                f"{env_model_dir!r} != {self.model_dir!r}. Unset the environment variable or pass the same path."
            )

    @staticmethod
    def _validate_calibration_source(model_config: Mapping[str, Any]) -> None:
        calib_config = model_config.get("calib_config")
        if not isinstance(calib_config, Mapping) or not calib_config.get("enable", False):
            return
        image_dir = os.environ.get("UNLIMITED_OCR_CALIB_IMAGE_DIR") or calib_config.get("image_dir")
        image_path = Path(str(image_dir)).expanduser() if image_dir else None
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        if image_path is None or not image_path.is_dir() or not any(
            path.is_file() and path.suffix.lower() in image_extensions for path in image_path.iterdir()
        ):
            raise FileNotFoundError(
                "Unlimited-OCR real-image calibration is enabled, but no calibration images were found in "
                f"{str(image_path)!r}. Set UNLIMITED_OCR_CALIB_IMAGE_DIR to a directory containing images."
            )

    @staticmethod
    def _validate_export_mode(model_config: Mapping[str, Any]) -> None:
        visual_config = model_config.get("visual_config")
        if not isinstance(visual_config, Mapping):
            raise ValueError("Unlimited-OCR workflow requires export.model.visual_config")
        export_mode = visual_config.get("export_mode")
        crop_mode = bool(visual_config.get("crop_mode", False))
        hmonnx_export = bool(visual_config.get("hmonnx_export", False))
        if export_mode != "base" or crop_mode or not hmonnx_export:
            raise NotImplementedError(
                "Unlimited-OCR workflow only supports base/no-crop HMONNX export; "
                f"export_mode={export_mode!r}, crop_mode={crop_mode}, hmonnx_export={hmonnx_export}."
            )


__all__ = ["XHUnlimitedOCRWorkflow"]