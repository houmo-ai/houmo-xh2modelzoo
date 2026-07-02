import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...workflows.base import BaseLLMWorkflow
from ...workflows.result import ExportResult, QuantResult


MINERU_VISUAL_BUCKETS_MANIFEST = "mineru_visual_buckets.json"


class XHQwen2VLHMONNXWorkflow(BaseLLMWorkflow):
    expected_model_config_cls_name = "XHQwen2VLModelConfig"
    expected_model_cls_name = "XHQwen2VLModel"

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        export_result = super().export(
            quant_result=quant_result,
            output_dir=output_dir,
            device=device,
            config_overrides=config_overrides,
        )

        visual_buckets_cfg = workflow_config.export.get("visual_buckets")
        if visual_buckets_cfg is None:
            return export_result
        if not isinstance(visual_buckets_cfg, Mapping):
            raise TypeError("export.visual_buckets must be a mapping when provided")

        self._write_visual_bucket_manifest(
            export_result=export_result,
            visual_buckets_cfg=visual_buckets_cfg,
        )
        return export_result

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
        logger = get_xhquant_logger()
        hmonnx_model = AutoLLMHONNXModel.from_pretrained(meta_file)
        processor = hmonnx_model.get_tf_processor()
        tokenizer = processor.tokenizer

        messages = self.build_input_message(input_messages)
        model_inputs = processor.apply_chat_template(messages).to(device)
        streamer = TextStreamer(tokenizer)
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
                **model_inputs,
                max_new_tokens=2,
                streamer=streamer,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        logger.info(f"{'-' * 20} Golden output {'-' * 20}")
        logger.info(f"{output_text}")
        return meta_file

    def build_input_message(self, input_messages: Any) -> list[dict[str, Any]]:
        if not isinstance(input_messages, Mapping):
            raise ValueError("Qwen2-VL input_messages must be a mapping with 'image' and 'text'")
        if "image" not in input_messages:
            raise ValueError("Qwen2-VL input_messages must contain 'image'")
        if "text" not in input_messages:
            raise ValueError("Qwen2-VL input_messages must contain 'text'")
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": input_messages["image"],
                    },
                    {"type": "text", "text": input_messages["text"]},
                ],
            }
        ]

    def _write_visual_bucket_manifest(
        self,
        export_result: ExportResult,
        visual_buckets_cfg: Mapping[str, Any],
    ) -> str:
        from qwen_vl_utils.vision_process import SPATIAL_MERGE_SIZE
        from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel
        from xhquant.api import get_xhquant_logger

        logger = get_xhquant_logger()
        exported_dir = self._resolve_exported_dir(export_result)
        visual_cfg_model = self._build_visual_config_model(visual_buckets_cfg)
        buckets = self._parse_visual_buckets(visual_buckets_cfg)

        default_visual_meta = getattr(export_result.meta, "visual_config", None)
        default_hmonnx = getattr(default_visual_meta, "hmonnx", None)
        if default_hmonnx is None:
            raise RuntimeError("Default visual HMONNX path is missing from exported metadata.")
        default_bucket = (
            int(getattr(default_visual_meta, "image_size_h")),
            int(getattr(default_visual_meta, "image_size_w")),
        )

        if default_bucket not in buckets:
            buckets.append(default_bucket)
        buckets = sorted(set(buckets), key=lambda item: (item[0] * item[1], item[0], item[1]))
        patch_size = int(visual_cfg_model.get("patch_size", getattr(default_visual_meta, "patch_size", 14)))
        self._validate_buckets(buckets, patch_size, SPATIAL_MERGE_SIZE)

        default_visual_model_name = exported_dir.name

        manifest_buckets = []
        for bucket in buckets:
            if bucket == default_bucket:
                hmonnx = self._relative_to_export_dir(default_hmonnx, exported_dir)
            else:
                hmonnx = self._export_visual_bucket(
                    visual_cfg_model=visual_cfg_model,
                    bucket=bucket,
                    default_bucket=default_bucket,
                    default_visual_model_name=default_visual_model_name,
                    exported_dir=exported_dir,
                    auto_llm_config_cls=AutoLLMConfig,
                    auto_llm_model_cls=AutoLLMModel,
                    logger=logger,
                )
            manifest_buckets.append(
                {
                    "max_size_h": bucket[0],
                    "max_size_w": bucket[1],
                    "hmonnx": hmonnx,
                }
            )

        manifest = {
            "buckets": manifest_buckets,
            "fallback_bucket": {
                "max_size_h": default_bucket[0],
                "max_size_w": default_bucket[1],
            },
            "patch_size": int(getattr(default_visual_meta, "patch_size", visual_cfg_model.get("patch_size", 14))),
            "spatial_merge_size": int(getattr(default_visual_meta, "spatial_merge_size", SPATIAL_MERGE_SIZE)),
            "temporal_patch_size": int(
                getattr(default_visual_meta, "temporal_patch_size", visual_cfg_model.get("temporal_patch_size", 2))
            ),
        }

        manifest_path = exported_dir / MINERU_VISUAL_BUCKETS_MANIFEST
        with manifest_path.open("w", encoding="utf-8") as fout:
            json.dump(manifest, fout, indent=4)
        logger.info(f"MinerU static visual bucket manifest saved to {manifest_path}")
        return str(manifest_path)

    def _build_visual_config_model(self, visual_buckets_cfg: Mapping[str, Any]) -> dict[str, Any]:
        visual_cfg_model = visual_buckets_cfg.get("model")
        if not isinstance(visual_cfg_model, Mapping) or not visual_cfg_model:
            raise ValueError("export.visual_buckets.model must be a non-empty mapping")
        visual_cfg_model = copy.deepcopy(dict(visual_cfg_model))
        visual_cfg_model["hf_model"] = self.model_dir
        return visual_cfg_model

    @staticmethod
    def _parse_visual_buckets(visual_buckets_cfg: Mapping[str, Any]) -> list[tuple[int, int]]:
        buckets_cfg = visual_buckets_cfg.get("buckets")
        if not isinstance(buckets_cfg, Sequence) or isinstance(buckets_cfg, (str, bytes)):
            raise ValueError("export.visual_buckets.buckets must be a non-empty sequence")
        buckets = [XHQwen2VLHMONNXWorkflow._parse_bucket(bucket) for bucket in buckets_cfg]
        if not buckets:
            raise ValueError("export.visual_buckets.buckets must contain at least one bucket")
        return buckets

    @staticmethod
    def _parse_bucket(bucket: Any) -> tuple[int, int]:
        if isinstance(bucket, Mapping):
            return int(bucket["max_size_h"]), int(bucket["max_size_w"])
        if isinstance(bucket, Sequence) and not isinstance(bucket, (str, bytes)) and len(bucket) == 2:
            return int(bucket[0]), int(bucket[1])
        raise ValueError(f"Invalid static visual bucket: {bucket!r}")

    @staticmethod
    def _validate_buckets(buckets: list[tuple[int, int]], patch_size: int, spatial_merge_size: int) -> None:
        factor = patch_size * spatial_merge_size
        for bucket_h, bucket_w in buckets:
            if bucket_h <= 0 or bucket_w <= 0:
                raise ValueError(f"Invalid static visual bucket {(bucket_h, bucket_w)}")
            if bucket_h % factor != 0 or bucket_w % factor != 0:
                raise ValueError(f"Static visual bucket {(bucket_h, bucket_w)} must be divisible by {factor}")

    @staticmethod
    def _resolve_exported_dir(export_result: ExportResult) -> Path:
        work_dir = Path(export_result.work_dir)
        if not work_dir.is_dir():
            raise FileNotFoundError(f"Export work_dir does not exist or is not a directory: {export_result.work_dir}")

        prefill_hmonnx = getattr(export_result.meta, "prefill_hmonnx", None)
        if prefill_hmonnx:
            matches = [
                path
                for path in work_dir.iterdir()
                if path.is_dir() and path.name.startswith("hmquant") and (path / str(prefill_hmonnx)).exists()
            ]
            if len(matches) == 1:
                return matches[0]

        meta_files = []
        for path in work_dir.iterdir():
            if path.is_dir() and path.name.startswith("hmquant") and (path / "golden_meta_info.json").is_file():
                meta_files.append(path / "golden_meta_info.json")
        if not meta_files:
            raise RuntimeError(f"Cannot locate exported HMONNX directory under {work_dir}")
        if len(meta_files) > 1:
            meta_file_list = ", ".join(str(path) for path in meta_files)
            raise RuntimeError(f"Found multiple exported HMONNX directories under {work_dir}: {meta_file_list}")
        return meta_files[0].parent

    @staticmethod
    def _relative_to_export_dir(path: str | Path, exported_dir: Path) -> str:
        path = Path(path)
        exported_dir_abs = exported_dir.resolve()
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend([exported_dir / path, Path.cwd() / path])
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                return candidate.resolve().relative_to(exported_dir_abs).as_posix()
            except ValueError:
                continue
        return path.as_posix()

    @staticmethod
    def _bucket_resolution_name(bucket: tuple[int, int]) -> str:
        max_size_h, max_size_w = bucket
        return f"{max_size_w}x{max_size_h}"

    @classmethod
    def _replace_model_name_resolution(
        cls,
        model_name: str,
        source_bucket: tuple[int, int],
        target_bucket: tuple[int, int],
    ) -> str:
        source_resolution = cls._bucket_resolution_name(source_bucket)
        target_resolution = cls._bucket_resolution_name(target_bucket)
        if source_resolution not in model_name:
            raise ValueError(
                f"Cannot find default visual resolution {source_resolution!r} in model name {model_name!r}"
            )
        return model_name.replace(source_resolution, target_resolution, 1)

    def _export_visual_bucket(
        self,
        visual_cfg_model: Mapping[str, Any],
        bucket: tuple[int, int],
        default_bucket: tuple[int, int],
        default_visual_model_name: str,
        exported_dir: Path,
        auto_llm_config_cls: Any,
        auto_llm_model_cls: Any,
        logger: Any,
    ) -> str:
        max_size_h, max_size_w = bucket
        bucket_resolution = self._bucket_resolution_name(bucket)
        bucket_dir = exported_dir / f"visual_{bucket_resolution}"

        cfg_model = copy.deepcopy(dict(visual_cfg_model))
        cfg_model["max_size_h"] = max_size_h
        cfg_model["max_size_w"] = max_size_w
        cfg_model["model_name"] = self._replace_model_name_resolution(
            default_visual_model_name,
            source_bucket=default_bucket,
            target_bucket=bucket,
        )
        model_cfg = auto_llm_config_cls.from_pretrained(cfg_model)
        model_cfg.work_dir = str(bucket_dir)
        visual_model = auto_llm_model_cls.from_pretrained(config=model_cfg)
        if type(visual_model).__name__ != "XHQwen2VLVisualModel":
            raise TypeError(f"Expected model type XHQwen2VLVisualModel, but got {type(visual_model).__name__}")
        logger.info(f"Exporting static visual bucket {bucket} to {bucket_dir}")
        visual_meta = visual_model.export_hmonnx(str(bucket_dir))
        return self._relative_to_export_dir(visual_meta.hmonnx, exported_dir)


__all__ = ["XHQwen2VLHMONNXWorkflow"]
