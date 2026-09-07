"""CPU contracts for visual gear packaging, relocation and LoRA shared files.

Run the real config loaders, multi-gear exporters, metadata writers/readers and
filesystem operations. Only unrelated checkpoint quantization, single-graph
lowering and HMONNX session construction are replaced at their boundaries.
The graph fixture is a valid one-element ONNX Add with real external data;
these tests do not claim model numerical or hardware-execution coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

from examples_merak.llm.qwen3_5.debug_scripts.validate_hm_release_layout import (
    ensure_step_artifact_links,
    validate_release_layout,
)
from xhmodel_merak.xh_llm.models.qwen3_5.lora import (
    XHQwen3_5LoRAConfig,
    finalize_lora_metadata,
    inspect_lora_adapter,
)
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import VisualTokenGearHMONNXModel
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_vision_model import XHQwen3_5VisionModel
from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_model import (
    XHQwen3_5MoeModel,
    build_qwen3_5_moe_hf_compatible_model,
)
from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_vision_model import XHQwen3_5MoeVisionModel
from xhmodel_merak.xh_llm.types import ExportData, VisualModelMeta, VLLMModelMeta


GEARS = (96, 196, 384, 704, 1536)


@pytest.mark.parametrize("unknown_kwarg", [False, True])
def test_moe_golden_generation_preserves_image_kwargs_and_reaches_gear_encoder(unknown_kwarg):
    """Run real HF generation/validation; stop at the expensive visual kernel."""
    config = Qwen3_5MoeConfig(
        image_token_id=30,
        text_config={
            "vocab_size": 32,
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
            "layer_types": ["full_attention"],
            "pad_token_id": 0,
            "eos_token_id": 31,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "out_hidden_size": 8,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 16,
        },
    )
    pixels = torch.ones(4, 24)
    grid = torch.tensor([[1, 2, 2]])

    class EncoderReachedError(RuntimeError):
        pass

    class VisualKernelBoundary:
        def encode_many(self, pixel_values, image_grid_thw):
            torch.testing.assert_close(pixel_values, pixels)
            torch.testing.assert_close(image_grid_thw, grid)
            raise EncoderReachedError("image kwargs reached the token-gear encoder")

    class RuntimeBoundary:
        def __init__(self):
            self.visual = VisualKernelBoundary()
            self.embedding = torch.nn.Embedding(32, 8)
            self.input_length = 8

        def get_input_embeddings(self):
            return self.embedding

        def get_input_sequence_length(self):
            return self.input_length

        def set_input_sequence_length(self, length):
            self.input_length = length

        def set_prefill(self):
            pass

        def set_decode(self):
            pass

        def is_support_dynamic_input(self):
            return False

    runtime = RuntimeBoundary()
    model = build_qwen3_5_moe_hf_compatible_model(Qwen3_5MoeForConditionalGeneration(config), runtime)
    original_forward = model.forward
    kwargs = {"misspelled_image_kwarg": torch.ones(1)} if unknown_kwarg else {}
    error = ValueError if unknown_kwarg else EncoderReachedError
    match = "misspelled_image_kwarg" if unknown_kwarg else "token-gear encoder"
    with pytest.raises(error, match=match):
        model.generate(
            input_ids=torch.tensor([[2, 30, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            pixel_values=pixels,
            image_grid_thw=grid,
            max_new_tokens=2,
            do_sample=False,
            **kwargs,
        )
    assert model.forward == original_forward
    assert not hasattr(model, "_xh_orig_forward")
    assert runtime.input_length == 8


def _write_graph(path: Path, external_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["x", "weight"], ["y"])],
        "visual_package_fixture",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
        [onnx.numpy_helper.from_array(np.array([1.0], dtype=np.float32), "weight")],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_name,
        size_threshold=0,
    )
    onnx.checker.check_model(str(path))


class _MetadataOnlyRuntime(VisualTokenGearHMONNXModel):
    """Leave the production metadata parser intact; do not open GPU sessions."""

    def __init__(self, visual_meta, *, device, enable_golden=False):
        self.visual_meta = visual_meta


@pytest.mark.parametrize("export_kind", ["standalone_dense", "standalone_moe", "dense", "moe"])
def test_visual_export_metadata_resolves_flat_gears_after_relocation(tmp_path, monkeypatch, export_kind):
    vision_cls = XHQwen3_5MoeVisionModel if export_kind.endswith("moe") else XHQwen3_5VisionModel
    hf_dir = tmp_path / "source"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe" if export_kind.endswith("moe") else "qwen3_5",
                "vision_config": {
                    "depth": 1,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "out_hidden_size": 8,
                    "num_heads": 2,
                    "patch_size": 2,
                    "temporal_patch_size": 2,
                    "spatial_merge_size": 2,
                    "num_position_embeddings": 16,
                },
            }
        )
    )
    config = vision_cls.CONFIG_CLS(
        hf_model=str(hf_dir),
        model_name="hmquant_fixture_visual",
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        visual_input_mode="patches",
        image_token_gears=list(GEARS),
        image_token_capacity=GEARS[-1],
    )

    def export_one_graph(self, output_dir):
        # Model lowering is unrelated to path composition. Emit a real ONNX
        # artifact at the single-graph boundary without mocking gear routing.
        self.config.work_dir = output_dir
        graph = Path(output_dir) / f"{self.config.model_name}_with_act.onnx"
        _write_graph(graph, f"{self.config.model_name}_external_data")
        meta = self.create_export_metadata(output_dir)
        meta.hmonnx = str(graph)
        return meta

    monkeypatch.setattr(XHQwen3_5VisionModel, "_export_single_hmonnx", export_one_graph)
    monkeypatch.setattr(XHQwen3_5VisionModel, "to_quanted_aligned", lambda self: None)
    model = vision_cls(config)
    original_work_dir = model.config.work_dir
    output_dir = tmp_path / "hmquant_fixture"
    standalone = export_kind.startswith("standalone")
    if standalone:
        model.export_hmonnx(str(output_dir))
    else:
        model_cls = XHQwen3_5Model if export_kind == "dense" else XHQwen3_5MoeModel
        full_model = model_cls.__new__(model_cls)
        full_model._models = {}
        full_model.visual = model
        for stage in ("prefill", "decode"):
            (output_dir / stage).mkdir(parents=True)
        exported_info = ExportData()
        exported_info.exported_dir = str(output_dir)
        exported_info.model_name = output_dir.name
        exported_info.meta = VLLMModelMeta(
            hf_config="hf_config",
            quant_embedding="quant_embedding.pt",
            prefill_hmonnx="prefill/model.onnx",
            decode_hmonnx="decode/model.onnx",
        )
        full_model._export_visual_hmonnx_impl(exported_info)
        assert full_model.visual is model

    assert model.config.work_dir == original_work_dir
    assert model.config.model_name == "hmquant_fixture_visual"
    relocated = output_dir.rename(tmp_path / "relocated_release")
    if standalone:
        meta_file = relocated / "visual_meta_info.json"
        metadata = json.loads(meta_file.read_text())
        resolved = _MetadataOnlyRuntime.from_meta_file(meta_file, device="cpu").visual_meta
    else:
        meta_file = relocated / "golden_meta_info.json"
        payload = json.loads(meta_file.read_text())
        metadata = payload["visual_config"]
        resolved = VLLMModelMeta.from_dict({**payload, "_meta_path_": str(meta_file)}).visual_config
        assert not (relocated / "visual_meta_info.json").exists()
    manifest = json.loads((relocated / metadata["gear_manifest"]).read_text())
    expected_paths = [f"visual_m{gear}/hmquant_fixture_visual_m{gear}_with_act.onnx" for gear in GEARS]
    assert [entry["hmonnx"] for entry in metadata["gears"]] == expected_paths
    assert manifest["gears"] == metadata["gears"]
    assert metadata["hmonnx"] == expected_paths[-1]
    assert metadata["gear_manifest"] == "visual_gears.json"
    expected_dirs = {f"visual_m{gear}" for gear in GEARS}
    if not standalone:
        expected_dirs.update({"prefill", "decode"})
    assert {path.name for path in relocated.iterdir() if path.is_dir()} == expected_dirs
    assert [Path(gear.hmonnx) for gear in resolved.gears] == [relocated / path for path in expected_paths]
    for gear in resolved.gears:
        loaded = onnx.load(gear.hmonnx)
        np.testing.assert_array_equal(onnx.numpy_helper.to_array(loaded.graph.initializer[0]), [1.0])


@pytest.fixture
def flat_visual_release(tmp_path):
    export_dir = tmp_path / "hmquant_fixture"
    prefix = export_dir.name
    (export_dir / "hf_config").mkdir(parents=True)
    (export_dir / "hf_config" / "config.json").write_text("{}")
    (export_dir / "quant_embedding.pt").write_bytes(b"embedding")
    gears = [
        {"image_token_capacity": gear, "hmonnx": f"visual_m{gear}/{prefix}_visual_m{gear}_with_act.onnx"}
        for gear in GEARS
    ]
    for stage in ["prefill", "decode", *(f"visual_m{gear}" for gear in GEARS)]:
        _write_graph(export_dir / stage / f"{prefix}_{stage}_with_act.onnx", f"{prefix}_{stage}_external_data")
    (export_dir / "golden_meta_info.json").write_text(
        json.dumps(
            {
                "hf_config": "hf_config",
                "quant_embedding": "quant_embedding.pt",
                "prefill_hmonnx": f"prefill/{prefix}_prefill_with_act.onnx",
                "decode_hmonnx": f"decode/{prefix}_decode_with_act.onnx",
                "visual_config": {"hmonnx": gears[-1]["hmonnx"], "gears": gears},
            }
        )
    )
    return export_dir


def test_release_layout_validates_and_repairs_every_flat_visual_gear(flat_visual_release, tmp_path):
    export_dir = flat_visual_release
    for gear in GEARS:
        (export_dir / f"visual_m{gear}" / "step_0").mkdir()
    ensure_step_artifact_links(export_dir)
    ensure_step_artifact_links(export_dir)  # Repair must be idempotent.
    assert validate_release_layout(export_dir) == []
    for gear in GEARS:
        stage = export_dir / f"visual_m{gear}"
        for suffix in ("with_act.onnx", "external_data"):
            name = f"{export_dir.name}_{stage.name}_{suffix}"
            link = stage / "step_0" / name
            assert link.is_symlink()
            assert not link.readlink().is_absolute()
            assert link.resolve() == stage / name
    relocated_parent = tmp_path / "relocated"
    relocated_parent.mkdir()
    relocated = export_dir.rename(relocated_parent / export_dir.name)
    assert validate_release_layout(relocated) == []
    assert all(path.is_file() for path in relocated.glob("visual_m*/step_0/*"))


@pytest.mark.parametrize("suffix", ["with_act.onnx", "external_data"])
def test_release_layout_rejects_missing_visual_gear_artifacts(flat_visual_release, suffix):
    artifact = flat_visual_release / "visual_m96" / f"{flat_visual_release.name}_visual_m96_{suffix}"
    artifact.unlink()
    assert f"missing {artifact.relative_to(flat_visual_release)}" in validate_release_layout(flat_visual_release)


@pytest.mark.parametrize(
    "directory", ["visual/m96", "visual", "m96", "visual_m196", "/visual_m96", "visual_m96/..", "other"]
)
def test_release_layout_rejects_stale_visual_gear_metadata(flat_visual_release, directory):
    meta_file = flat_visual_release / "golden_meta_info.json"
    meta = json.loads(meta_file.read_text())
    meta["visual_config"]["gears"][0]["hmonnx"] = f"{directory}/graph.onnx"
    meta_file.write_text(json.dumps(meta))
    assert any("visual_config.gears[0].hmonnx" in failure for failure in validate_release_layout(flat_visual_release))


def test_release_layout_ignores_non_directory_optional_stages(flat_visual_release):
    (flat_visual_release / "visual_m2048").write_text("not an exported stage")
    assert validate_release_layout(flat_visual_release) == []


@pytest.mark.parametrize("directory", ["visual/m96", "visual", "m96", "other", "../outside", "/outside"])
def test_step_link_repair_ignores_unsupported_or_escaping_metadata(tmp_path, directory):
    root = tmp_path / "release"
    root.mkdir()
    outside = tmp_path / "outside"
    directory = str(outside) if directory == "/outside" else directory
    target = root / directory
    _write_graph(target / "graph.onnx", "graph_external_data")
    step = target / "step_0"
    step.mkdir()
    (root / "golden_meta_info.json").write_text(json.dumps({"visual_config": {"hmonnx": f"{directory}/graph.onnx"}}))
    ensure_step_artifact_links(root)
    assert list(step.iterdir()) == []


def _write_adapter(adapter_dir):
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 2,
                "lora_alpha": 4,
                "bias": "none",
            }
        )
    )
    prefix = "base_model.model.model.language_model.layers.0.self_attn.q_proj"
    save_file(
        {
            f"{prefix}.lora_A.weight": torch.ones(2, 4),
            f"{prefix}.lora_B.weight": torch.ones(3, 2),
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )


def test_lora_child_metadata_reuses_root_artifacts(tmp_path: Path):
    visual_dirs = ("visual_m96", "visual_m1536")
    adapter_dir = tmp_path / "source-adapter"
    _write_adapter(adapter_dir)
    adapter = inspect_lora_adapter(str(adapter_dir))

    root_dir = tmp_path / "hmquant-model"
    child_dir = root_dir / "lora" / adapter.name
    child_dir.mkdir(parents=True)
    (root_dir / "hf_config").mkdir()
    (root_dir / "hf_config" / "config.json").write_text("{}", encoding="utf-8")
    (root_dir / "quant_embedding.pt").write_bytes(b"embedding")
    for visual_dir in visual_dirs:
        (root_dir / visual_dir).mkdir(parents=True, exist_ok=True)
        _write_graph(root_dir / visual_dir / "visual.onnx", "visual_external_data")
        (root_dir / visual_dir / "step_0").mkdir()
    (root_dir / "mtp_draft_prefill").mkdir()
    (root_dir / "mtp_draft_prefill" / "mtp.onnx").write_bytes(b"mtp")
    root_meta = VLLMModelMeta(
        hf_config="hf_config",
        quant_embedding="quant_embedding.pt",
        prefill_hmonnx="prefill/base.onnx",
        decode_hmonnx="decode/base.onnx",
    )
    root_meta.visual_config = VisualModelMeta()
    root_meta.visual_config.hmonnx = f"{visual_dirs[-1]}/visual.onnx"
    root_meta.visual_config.gears = [
        {"image_token_capacity": int(name.rsplit("m", 1)[-1]), "hmonnx": f"{name}/visual.onnx"} for name in visual_dirs
    ]
    root_meta.visual_config.gear_manifest = "visual_gears.json"
    (root_dir / root_meta.visual_config.gear_manifest).write_text(
        json.dumps({"gears": root_meta.visual_config.gears}), encoding="utf-8"
    )
    root_meta.mtp_prefill_config = VisualModelMeta()
    root_meta.mtp_prefill_config.hmonnx = "mtp_draft_prefill/mtp.onnx"
    root_meta.spec_decode = {"mode": "mtp", "draft_prefill_onnx": "mtp_draft_prefill/mtp.onnx"}

    graph_meta = VLLMModelMeta(
        prefill_hmonnx="prefill/adapter.onnx",
        prefill_hmonnx_md5="prefill-md5",
        decode_hmonnx="decode/adapter.onnx",
        decode_hmonnx_md5="decode-md5",
    )
    exported_info = ExportData()
    exported_info.exported_dir = str(root_dir)
    exported_info.meta = root_meta
    adapter_export = ExportData()
    adapter_export.exported_dir = str(child_dir)
    adapter_export.meta = graph_meta

    lora_config = XHQwen3_5LoRAConfig(
        path=[str(adapter_dir)],
        w_schema={"bits": 4, "fp_mode": "ssfp"},
    )
    finalize_lora_metadata(root_meta, exported_info, [(adapter, adapter_export)], lora_config)

    child_meta = json.loads((child_dir / "golden_meta_info.json").read_text(encoding="utf-8"))
    assert child_meta["prefill_hmonnx"] == "prefill/adapter.onnx"
    assert child_meta["decode_hmonnx"] == "decode/adapter.onnx"
    assert child_meta["quant_embedding"] == "quant_embedding.pt"
    assert child_meta["hf_config"] == "hf_config"
    assert child_meta["visual_config"]["hmonnx"] == f"{visual_dirs[-1]}/visual.onnx"
    assert child_meta["mtp_prefill_config"]["hmonnx"] == "mtp_draft_prefill/mtp.onnx"
    assert child_meta["spec_decode"]["draft_prefill_onnx"] == "mtp_draft_prefill/mtp.onnx"
    assert child_meta["active_lora"]["scale"] == 2.0
    assert (child_dir / "hf_config").is_dir()
    assert not (child_dir / "hf_config").is_symlink()
    assert not (child_dir / "hf_config" / "config.json").is_symlink()
    assert (child_dir / "quant_embedding.pt").is_symlink()
    for visual_dir in visual_dirs:
        assert (child_dir / visual_dir).is_dir()
        assert not (child_dir / visual_dir).is_symlink()
        assert (child_dir / visual_dir / "visual.onnx").is_symlink()
        assert (child_dir / visual_dir / "visual_external_data").is_symlink()
        assert not (child_dir / visual_dir / "step_0").exists()
    manifest_path = child_dir / child_meta["visual_config"]["gear_manifest"]
    assert manifest_path.is_symlink()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["gears"] == child_meta["visual_config"]["gears"]
    assert all((child_dir / gear["hmonnx"]).is_file() for gear in manifest["gears"])
    assert (child_dir / "mtp_draft_prefill" / "mtp.onnx").is_symlink()
    assert root_meta.quant_embedding == "quant_embedding.pt"
    assert root_meta.lora_adapters[0]["meta_file"] == f"lora/{adapter.name}/golden_meta_info.json"

    for visual_dir in visual_dirs:
        (child_dir / visual_dir / "step_0").mkdir()
    ensure_step_artifact_links(child_dir)
    for visual_dir in visual_dirs:
        child_visual_step = child_dir / visual_dir / "step_0"
        assert (child_visual_step / "visual.onnx").is_symlink()
        assert (child_visual_step / "visual_external_data").is_symlink()
        assert (child_visual_step / "visual.onnx").resolve() == root_dir / visual_dir / "visual.onnx"
        assert (child_visual_step / "visual_external_data").resolve() == root_dir / visual_dir / "visual_external_data"

    relocated_root = root_dir.rename(tmp_path / "relocated-model")
    relocated_child = relocated_root / child_dir.relative_to(root_dir)
    for visual_dir in visual_dirs:
        assert (relocated_child / visual_dir / "step_0" / "visual.onnx").is_file()
        assert (relocated_child / visual_dir / "step_0" / "visual_external_data").is_file()
    manifest = json.loads((relocated_child / root_meta.visual_config.gear_manifest).read_text())
    assert all((relocated_child / gear["hmonnx"]).is_file() for gear in manifest["gears"])
