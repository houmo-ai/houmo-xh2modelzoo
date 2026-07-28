import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.xh_llm.workflows.base import BaseLLMWorkflow
from xhmodel_merak.xh_llm.workflows.result import (
    ExportResult,
    QuantResult,
)


class XHQwen3VLEmbeddingWorkflow(BaseLLMWorkflow):
    expected_model_config_cls_name = (
        "XHQwen3VLEmbeddingModelConfig"
    )
    expected_model_cls_name = "XHQwen3VLEmbeddingModel"

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        export_result = super().export(
            quant_result=quant_result,
            output_dir=output_dir,
            device=device,
            config_overrides=config_overrides,
        )
        workflow_config = self.workflow_config.with_overrides(
            config_overrides
        )
        work_dir = Path(export_result.work_dir).resolve()
        if not hasattr(export_result.meta, "to_dict"):
            raise TypeError(
                "Embedding export metadata must support to_dict()"
            )
        model_meta = export_result.meta.to_dict()
        artifact_dirs = [
            path
            for path in work_dir.iterdir()
            if path.is_dir() and path.name.startswith("hmquant_")
        ]
        if len(artifact_dirs) != 1:
            raise ValueError(
                "Expected exactly one hmquant artifact directory "
                f"under {work_dir}, found {len(artifact_dirs)}"
            )
        artifact_dir = artifact_dirs[0]

        def artifact_path(relative_path: str) -> str:
            if not relative_path:
                return relative_path
            return str(
                (artifact_dir / relative_path).relative_to(work_dir)
            )

        visual_meta = dict(model_meta["visual_config"])
        visual_meta["hmonnx"] = artifact_path(
            visual_meta["hmonnx"]
        )
        visual_meta.pop("onnx", None)

        root_meta = dict(model_meta)
        root_meta["hf_config"] = artifact_path(
            model_meta["hf_config"]
        )
        root_meta["quant_embedding"] = artifact_path(
            model_meta["quant_embedding"]
        )
        root_meta["prefill_hmonnx"] = artifact_path(
            model_meta["prefill_hmonnx"]
        )
        root_meta["decode_hmonnx"] = artifact_path(
            model_meta.get("decode_hmonnx", "")
        )
        root_meta["visual_config"] = visual_meta
        root_meta.update(
            {
                "config": str(
                    Path(export_result.config_file)
                    .resolve()
                    .relative_to(work_dir)
                ),
                "hf_model": self.model_dir,
                "target_device": workflow_config.export["model"][
                    "chip_arch"
                ],
                "components": ["vision", "prefill"],
                "artifact_dir": str(
                    artifact_dir.relative_to(work_dir)
                ),
                "vision_hmonnx": visual_meta["hmonnx"],
                "embedding": {
                    "output": "hidden_states",
                    "pooling": "last_token",
                    "normalize": True,
                },
            }
        )
        root_meta_file = work_dir / "export_meta_info.json"
        root_meta_file.write_text(
            json.dumps(root_meta, indent=4),
            encoding="utf-8",
        )

        visual_onnx_dir = work_dir / "visual" / "onnx"
        if visual_onnx_dir.is_dir():
            shutil.rmtree(visual_onnx_dir)
            visual_work_dir = visual_onnx_dir.parent
            if not any(visual_work_dir.iterdir()):
                visual_work_dir.rmdir()

        export_result.meta = root_meta
        return export_result

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        import torch
        from PIL import Image

        from xhmodel_merak.xh_llm import AutoLLMHONNXModel

        work_dir = Path(export_result.work_dir)
        root_meta_file = work_dir / "export_meta_info.json"
        if not root_meta_file.is_file():
            raise FileNotFoundError(
                f"export_meta_info.json not found under {work_dir}"
            )
        root_meta = json.loads(
            root_meta_file.read_text(encoding="utf-8")
        )
        visual_meta = root_meta["visual_config"]
        prefill_golden_dir = (
            work_dir / root_meta["prefill_hmonnx"]
        ).parent / "golden"
        visual_golden_dir = (
            work_dir / root_meta["vision_hmonnx"]
        ).parent / "golden"
        for golden_dir in (
            prefill_golden_dir,
            visual_golden_dir,
        ):
            if golden_dir.exists():
                shutil.rmtree(golden_dir)
            golden_dir.mkdir(parents=True)

        default_image = Image.new(
            "RGB",
            (
                int(visual_meta["image_size_w"]),
                int(visual_meta["image_size_h"]),
            ),
            color=(114, 114, 114),
        )

        if input_messages is None:
            input_messages = {
                "image": default_image,
                "text": "A dog playing in the park",
            }
        elif isinstance(input_messages, str):
            input_messages = {
                "image": default_image,
                "text": input_messages,
            }
        elif not isinstance(input_messages, Mapping):
            raise TypeError(
                "input_messages must be a string or a text/image mapping"
            )
        else:
            input_messages = dict(input_messages)
            input_messages.setdefault("image", default_image)

        hmonnx_model = AutoLLMHONNXModel.from_pretrained(
            str(root_meta_file),
            device_map=[device],
        )
        hmonnx_model.to(torch.device(device))
        hmonnx_model.prefill_model.hmonnx_session.save_golden_dir = (
            str(prefill_golden_dir)
        )
        hmonnx_model.visual.hmonnx_session.save_golden_dir = str(
            visual_golden_dir
        )
        hmonnx_model.enable_golden = True
        hmonnx_model.embed_items([input_messages])

        missing_golden_dirs = [
            str(golden_dir / "step_0")
            for golden_dir in (
                prefill_golden_dir,
                visual_golden_dir,
            )
            if not (golden_dir / "step_0").is_dir()
        ]
        if missing_golden_dirs:
            raise RuntimeError(
                "Golden generation did not create: "
                + ", ".join(missing_golden_dirs)
            )
        return str(work_dir)


__all__ = ["XHQwen3VLEmbeddingWorkflow"]
