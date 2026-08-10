# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
import types
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


GENERATE_SCRIPT = Path("examples_merak/llm/hunyuan_ocr/hunyuan_ocr_xh_hmonnx_generate.py")


def _install_hmonnx_optimizer_stub() -> None:
    module = types.ModuleType("xhquant.xhonnxruntime.hmonnx_optimizer")
    module.materialize_parallel_linear_fusion = lambda path: path
    sys.modules.setdefault("xhquant.xhonnxruntime.hmonnx_optimizer", module)


def _load_generate_script():
    spec = importlib.util.spec_from_file_location("hunyuan_ocr_hmonnx_generate_testmod", GENERATE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hunyuan_ocr_registers_autoregressive_hmonnx_runtime() -> None:
    _install_hmonnx_optimizer_stub()

    from xhmodel_merak.xh_llm.models.hunyuan_ocr import XHHunYuanOCRModel
    from xhmodel_merak.xh_llm.models.hunyuan_ocr.hunyuan_ocr_hmonnx_inference import XHHunYuanOCRHMONNXModel

    runtime_cls = XHHunYuanOCRModel.get_hmonnx_inference_cls()

    assert runtime_cls is XHHunYuanOCRHMONNXModel
    assert runtime_cls.LLM_MODEL_CLS is XHHunYuanOCRModel


def test_hunyuan_ocr_hmonnx_runtime_wraps_processor_when_manifest_is_approved(monkeypatch) -> None:
    _install_hmonnx_optimizer_stub()

    from xhmodel_merak.xh_llm.models.hunyuan_ocr.hunyuan_ocr_hmonnx_inference import XHHunYuanOCRHMONNXModel
    from xhmodel_merak.xh_llm.models.hunyuan_ocr.hunyuan_ocr_processor import HunyuanOCRMultiBucketProcessor

    runtime = object.__new__(XHHunYuanOCRHMONNXModel)
    runtime.hf_model_dir = "checkpoint"
    runtime.meta_info = SimpleNamespace(
        resolution_bucket_manifest={
            "status": "approved",
            "input_policy": {"max_images_per_request": 3, "padding_color_rgb": [255, 255, 255]},
            "routing": {"min_content_ratio": 0.8, "min_downscale": 0.25},
            "resource_limits": {
                "max_image_tokens": 1400,
                "max_canonical_prefill_length": 1446,
                "max_vision_attention_pair_count": 27040000,
            },
            "buckets": [
                {
                    "id": "bucket_1440x896",
                    "width": 1440,
                    "height": 896,
                    "image_grid_thw": [1, 56, 90],
                    "image_token_count": 1290,
                    "canonical_prefill_length": 1336,
                    "vision_attention_pair_count": 25401600,
                }
            ],
        }
    )
    source_processor = object()
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.hunyuan_ocr.hunyuan_ocr_hmonnx_inference.AutoProcessor.from_pretrained",
        lambda *args, **kwargs: source_processor,
    )

    processor = runtime.get_tf_processor()

    assert isinstance(processor, HunyuanOCRMultiBucketProcessor)
    assert processor.processor is source_processor


def test_hunyuan_ocr_hmonnx_runtime_builds_text_preprocessor() -> None:
    _install_hmonnx_optimizer_stub()

    from xhmodel_merak.xh_llm.models.hunyuan_ocr.hunyuan_ocr_hmonnx_inference import XHHunYuanOCRHMONNXModel

    runtime = object.__new__(XHHunYuanOCRHMONNXModel)
    runtime.meta_info = SimpleNamespace(
        model_config=SimpleNamespace(
            context_max_length=131072,
            visual_config=SimpleNamespace(spatial_merge_size=2),
            image_start_token_id=120118,
            image_end_token_id=120119,
        ),
        pad_token_id=120002,
        image_token_id=120120,
    )
    runtime._device = torch.device("cpu")
    runtime._dtype = torch.float32
    runtime._kvcache_mixin = SimpleNamespace(past_key_caches=None, past_value_caches=None)
    runtime.get_input_embeddings = lambda: torch.nn.Embedding(16, 8, padding_idx=0)
    runtime.get_input_sequence_length = lambda: 5

    processor = runtime._get_data_preprocessor()

    assert processor.context_max_length == 131072
    assert processor.input_sequence_length == 5
    assert processor.image_token_id == 120120


def test_hunyuan_ocr_hmonnx_runtime_routes_visual_graph_by_image_grid() -> None:
    _install_hmonnx_optimizer_stub()

    from xhmodel_merak.xh_llm.models.hunyuan_ocr.hunyuan_ocr_hmonnx_inference import XHHunYuanOCRHMONNXModel

    runtime = object.__new__(XHHunYuanOCRHMONNXModel)
    default_meta = SimpleNamespace(
        image_grid_thw=[1, 80, 80],
        input_shape=[6400, 768],
        output_shape=[1, 1682, 1024],
    )
    routed_meta = SimpleNamespace(
        image_grid_thw=[1, 82, 58],
        input_shape=[4756, 768],
        output_shape=[1, 1232, 1024],
    )
    calls = []

    class FakeVisual:
        def __init__(self, name: str, output_shape: tuple[int, ...]) -> None:
            self.name = name
            self.output_shape = output_shape

        def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
            calls.append((self.name, tuple(pixels.shape)))
            return torch.zeros(self.output_shape)

    runtime.visual_meta = default_meta
    runtime.visual_buckets = {"bucket_1280x960": default_meta, "bucket_928x1312": routed_meta}
    runtime.visual = FakeVisual("default", (1, 1682, 1024))
    runtime.visual_by_bucket = {
        "bucket_1280x960": runtime.visual,
        "bucket_928x1312": FakeVisual("routed", (1, 1232, 1024)),
    }
    runtime._device = torch.device("cpu")
    runtime._dtype = torch.float16

    features, selected_meta = runtime.run_visual_bucket(
        torch.zeros((4756, 768)),
        torch.tensor([[1, 82, 58]]),
    )

    assert selected_meta is routed_meta
    assert features.shape == (1, 1232, 1024)
    assert calls == [("routed", (4756, 768))]

    with pytest.raises(ValueError, match="approved bucket"):
        runtime.run_visual_bucket(torch.zeros((4, 768)), torch.tensor([[1, 2, 2]]))


def test_hunyuan_ocr_target_verify_normalizes_int64_graph_inputs() -> None:
    _install_hmonnx_optimizer_stub()

    from xhmodel_merak.xh_llm.models.hunyuan_ocr.hunyuan_ocr_hmonnx_inference import XHHunYuanOCRHMONNXModel

    runtime = object.__new__(XHHunYuanOCRHMONNXModel)
    received_dtypes = []

    class FakeProcessor:
        input_sequence_length = 1

        def __call__(self, _data):
            return (
                torch.zeros((1, 4, 8), dtype=torch.float16),
                torch.zeros((1, 4), dtype=torch.int64),
                torch.zeros((1, 4), dtype=torch.int64),
            )

    processor = FakeProcessor()
    runtime.get_data_preprocessor = lambda: processor

    def verify_model(*args):
        received_dtypes.extend(arg.dtype for arg in args)
        return torch.zeros((1, 4, 16)), torch.zeros((1, 4, 8))

    runtime.verify_model = verify_model

    runtime._execute_target_verify(
        input_token_ids=torch.tensor([[1, 2, 3, 4]]),
        position_ids=torch.zeros((4, 1, 4), dtype=torch.int64),
        past_seq_length=0,
        current_input_length=4,
    )

    assert received_dtypes == [torch.float16, torch.int32, torch.int32]
    assert processor.input_sequence_length == 1


def test_generate_cli_rejects_invalid_args_before_loading_runtime(tmp_path: Path) -> None:
    module = _load_generate_script()
    args = Namespace(
        hmonnx_config=tmp_path / "missing.json",
        image=tmp_path / "image.png",
        prompt="Extract text",
        device="cpu",
        max_new_tokens=1,
        dflash=False,
        num_draft_tokens=None,
        stream=False,
        legacy_runtime=False,
        cuda_graph=False,
        output=None,
    )

    try:
        module.main(args)
    except FileNotFoundError as error:
        assert "metadata" in str(error)
    else:
        raise AssertionError("missing metadata unexpectedly succeeded")

    parser_destinations = {action.dest for action in module.build_parser()._actions}
    assert {"dflash", "num_draft_tokens"} <= parser_destinations
    assert module._runtime_device_map("cuda") == ["cuda:0"]
    assert module._runtime_device_map("cuda:3") == ["cuda:3"]
    assert module._runtime_device_map("cpu") == ["cpu"]


def test_generate_cli_reports_incompatible_hmonnx_dependency(monkeypatch, tmp_path: Path) -> None:
    module = _load_generate_script()
    metadata = tmp_path / "golden_meta_info.json"
    image = tmp_path / "page.png"
    metadata.write_text("{}", encoding="utf-8")
    image.write_bytes(b"image")
    monkeypatch.setattr("xhmodel_merak.xh_llm.AutoLLMHONNXModel", None)

    with pytest.raises(RuntimeError, match="xhquanttool build paired"):
        module.main(
            Namespace(
                hmonnx_config=metadata,
                image=image,
                prompt="Extract text",
                device="cpu",
                max_new_tokens=1,
                dflash=False,
                num_draft_tokens=None,
                stream=False,
                legacy_runtime=False,
                cuda_graph=False,
                output=None,
            )
        )


def test_generate_cli_rejects_unsupported_dflash_combinations(tmp_path: Path) -> None:
    module = _load_generate_script()
    metadata = tmp_path / "golden_meta_info.json"
    image = tmp_path / "page.png"
    metadata.write_text("{}", encoding="utf-8")
    image.write_bytes(b"image")

    common = {
        "hmonnx_config": metadata,
        "image": image,
        "prompt": "Extract text",
        "device": "cpu",
        "max_new_tokens": 1,
        "legacy_runtime": False,
        "cuda_graph": False,
        "output": None,
    }
    with pytest.raises(ValueError, match="--num-draft-tokens requires --dflash"):
        module._validate_args(
            Namespace(**common, dflash=False, num_draft_tokens=4, stream=False)
        )
    with pytest.raises(ValueError, match="--num-draft-tokens must be positive"):
        module._validate_args(
            Namespace(**common, dflash=True, num_draft_tokens=0, stream=False)
        )
    with pytest.raises(ValueError, match="--no-stream"):
        module._validate_args(
            Namespace(**common, dflash=True, num_draft_tokens=None, stream=True)
        )


def test_generate_cli_runs_autoregressive_runtime(monkeypatch, tmp_path: Path) -> None:
    module = _load_generate_script()
    metadata = tmp_path / "golden_meta_info.json"
    image = tmp_path / "page.png"
    output = tmp_path / "report.json"
    metadata.write_text(json.dumps({"model_config": {"model_type": "HunYuanVLForConditionalGeneration"}}))
    image.write_bytes(b"image")
    calls = []

    class FakeInputs(dict):
        def to(self, device):
            calls.append(("inputs.to", device))
            return self

    class FakeProcessor:
        tokenizer = object()

        def apply_chat_template(self, messages, **kwargs):
            calls.append(("apply_chat_template", messages, kwargs))
            return FakeInputs({"input_ids": torch.tensor([[10, 11, 12]])})

        @staticmethod
        def decode(tokens, **kwargs):
            calls.append(("decode", tokens, kwargs))
            return "decoded text"

    class FakeRuntime:
        meta_info = SimpleNamespace(generation_eos_token_id=120020, pad_token_id=120002)
        last_request_summary = {"mode": "autoregressive"}

        def to(self, device):
            calls.append(("runtime.to", device))
            return self

        def get_tf_processor(self):
            return FakeProcessor()

        def generate(self, **kwargs):
            calls.append(("generate", sorted(kwargs)))
            return SimpleNamespace(sequences=torch.tensor([[10, 11, 12, 99, 100]]))

    class FakeAuto:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("from_pretrained", path, kwargs))
            return FakeRuntime()

    monkeypatch.setattr("xhmodel_merak.xh_llm.AutoLLMHONNXModel", FakeAuto)
    monkeypatch.setattr("xhmodel_merak.xh_llm.LLMInferenceContextManager", lambda _runtime: nullcontext())

    report = module.main(
        Namespace(
            hmonnx_config=metadata,
            image=image,
            prompt="Extract text",
            device="cpu",
            max_new_tokens=2,
            dflash=False,
            num_draft_tokens=None,
            stream=False,
            legacy_runtime=False,
            cuda_graph=False,
            output=output,
        )
    )

    assert report["generated_tokens"] == [99, 100]
    assert report["text"] == "decoded text"
    assert report["runtime_summary"] == {"mode": "autoregressive"}
    assert json.loads(output.read_text(encoding="utf-8"))["generated_tokens"] == [99, 100]
    assert ("from_pretrained", str(metadata), {"device_map": ["cpu"], "enable_cuda_graph": False}) in calls


def test_generate_cli_runs_dflash_without_unsupported_generation_options(monkeypatch, tmp_path: Path) -> None:
    module = _load_generate_script()
    metadata = tmp_path / "golden_meta_info.json"
    image = tmp_path / "page.png"
    metadata.write_text("{}", encoding="utf-8")
    image.write_bytes(b"image")
    generation_kwargs = None

    class FakeInputs(dict):
        def to(self, _device):
            return self

    class FakeProcessor:
        def apply_chat_template(self, _messages, **_kwargs):
            return FakeInputs({"input_ids": torch.tensor([[10, 11, 12]])})

        @staticmethod
        def decode(_tokens, **_kwargs):
            return "decoded text"

    class FakeRuntime:
        meta_info = SimpleNamespace(generation_eos_token_id=120020, pad_token_id=120002)
        last_request_summary = {"mode": "dflash"}

        def to(self, _device):
            return self

        def get_tf_processor(self):
            return FakeProcessor()

        def generate(self, **kwargs):
            nonlocal generation_kwargs
            generation_kwargs = kwargs
            return torch.tensor([[10, 11, 12, 99]])

    class FakeAuto:
        @staticmethod
        def from_pretrained(_path, **_kwargs):
            return FakeRuntime()

    monkeypatch.setattr("xhmodel_merak.xh_llm.AutoLLMHONNXModel", FakeAuto)
    monkeypatch.setattr("xhmodel_merak.xh_llm.LLMInferenceContextManager", lambda _runtime: nullcontext())

    report = module.main(
        Namespace(
            hmonnx_config=metadata,
            image=image,
            prompt="Extract text",
            device="cpu",
            max_new_tokens=1,
            dflash=True,
            num_draft_tokens=4,
            stream=False,
            legacy_runtime=False,
            cuda_graph=False,
            output=None,
        )
    )

    assert generation_kwargs is not None
    assert generation_kwargs["dflash_enabled"] is True
    assert generation_kwargs["dflash_num_draft_tokens"] == 4
    assert "eos_token_id" not in generation_kwargs
    assert "streamer" not in generation_kwargs
    assert report["dflash_enabled"] is True
    assert report["runtime_summary"] == {"mode": "dflash"}
