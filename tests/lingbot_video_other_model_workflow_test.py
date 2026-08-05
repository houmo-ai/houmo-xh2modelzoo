import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from xhmodel_merak.workflows import AutoWorkflow


CONFIG_ROOT = Path("configs_merak/workflows/xh2a/other_models/lingbot_video/dense_1_3b")
CONFIG_PATHS = (
    CONFIG_ROOT / "lingbot_video_dense_1_3b_w8a8.yaml",
    CONFIG_ROOT / "lingbot_video_dense_1_3b_w8a16.yaml",
    CONFIG_ROOT / "lingbot_video_dense_1_3b_w16a16.yaml",
)
EXPECTED_COMPONENTS = {
    "text_encoder",
    "visual_encoder",
    "transformer",
    "vae_encoder",
    "vae_decoder",
}
LINGBOT_MODEL_ROOT = Path("xhmodel_merak/xh_other_model/models/lingbot_video")


@pytest.mark.parametrize(
    ("config_path", "quant_type", "activation_bits"),
    [
        (CONFIG_PATHS[0], "w8a8h1_sefp", 8),
        (CONFIG_PATHS[1], "w8a16h1_sefp", 16),
        (CONFIG_PATHS[2], "w16a16h1_sefp", 16),
    ],
)
def test_lingbot_video_config_declares_full_model_quantization(
    config_path: Path,
    quant_type: str,
    activation_bits: int,
):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    export = config["export"]
    components = export["components"]

    assert config["quant"] is None
    assert export["model"]["type"] == "XHLingBotVideoModel"
    assert set(components) == EXPECTED_COMPONENTS
    assert all(component["enabled"] for component in components.values())
    assert all(component["quant_type"] == quant_type for component in components.values())

    for name in ("text_encoder", "visual_encoder", "transformer"):
        matmul = components[name]["ops"]["MatMul"]
        assert matmul["act_scheme"]["bits"] == activation_bits
        assert matmul["act_schema_2"]["bits"] == activation_bits

    flash_attention = components["transformer"]["flash_attention"]
    assert flash_attention["enable"] is True
    assert {flash_attention[name] for name in ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")} == {activation_bits}
    assert components["vae_encoder"]["ops"]["Normalize"]["force_fp32"] is True
    assert components["vae_decoder"]["ops"]["Normalize"]["force_fp32"] is True


def test_auto_workflow_binds_lingbot_video_workflow():
    workflow = AutoWorkflow.from_config(
        model_dir="/models/lingbot-video-dense-1.3b",
        config_path=str(CONFIG_PATHS[0]),
    )

    from xhmodel_merak.xh_other_model.models.lingbot_video.workflow import (
        LingBotVideoWorkflow,
    )

    assert isinstance(workflow, LingBotVideoWorkflow)
    quant_result = workflow.quant(output_dir="unused", device="cpu")
    assert quant_result.skipped is True
    assert quant_result.raw_model_dir == "/models/lingbot-video-dense-1.3b"
    assert quant_result.meta["quantization"] == "integrated_into_each_static_graph_export"


def test_lingbot_video_package_import_does_not_import_workflow():
    package_name = "xhmodel_merak.xh_other_model.models.lingbot_video"
    workflow_module_name = f"{package_name}.workflow"
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            del sys.modules[module_name]

    package = importlib.import_module(package_name)

    assert package.XHLingBotVideoModel.__name__ == "XHLingBotVideoModel"
    assert package.__all__ == ["XHLingBotVideoModel"]
    assert workflow_module_name not in sys.modules


def test_lingbot_video_does_not_depend_on_xh_llm():
    import_pattern = re.compile(r"xhmodel_merak\.xh_llm|(^|\.)xh_llm(\.|\b)", re.MULTILINE)
    matches = []
    for source_file in LINGBOT_MODEL_ROOT.rglob("*.py"):
        if import_pattern.search(source_file.read_text(encoding="utf-8")):
            matches.append(str(source_file))
    assert matches == []

    script = """
import builtins

original_import = builtins.__import__

def isolated_import(name, *args, **kwargs):
    if name == "xhmodel_merak.xh_llm" or name.startswith("xhmodel_merak.xh_llm."):
        raise ImportError(f"blocked architecture import: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = isolated_import
import xhmodel_merak.xh_other_model.models.lingbot_video.components_hmonnx
import xhmodel_merak.xh_other_model.models.lingbot_video.text_encoder
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path.cwd())
    subprocess.run([sys.executable, "-c", script], check=True, env=environment)


def test_lingbot_qwen_text_and_visual_artifact_names_use_independent_quant_types():
    from xhmodel_merak.xh_other_model.models.lingbot_video.text_encoder import (
        LingBotQwen3VLEncoder,
    )

    text_encoder = LingBotQwen3VLEncoder(
        text_encoder_dir=Path("unused"),
        target_device="XH2a",
        text_cfg={"quant_type": "w8a8h1_sefp", "sequence_length": 2048},
        visual_cfg={
            "quant_type": "w8a16h1_sefp",
            "max_size_w": 448,
            "max_size_h": 448,
        },
    )

    assert text_encoder.language_model_name == "lingbot_qwen3_vl_XH2a_w8a8h1_sefp"
    assert text_encoder.visual_model_name == "lingbot_qwen3_vl_XH2a_w8a16h1_sefp_448x448"
    assert text_encoder.language_quant_type == "w8a8h1_sefp"
    assert text_encoder.visual_quant_type == "w8a16h1_sefp"


def test_lingbot_pipeline_uses_exported_text_sequence_length():
    from xhmodel_merak.xh_other_model.models.lingbot_video.components_hmonnx import (
        configure_pipeline_token_length,
    )

    pipeline = type("Pipeline", (), {"token_length": 37698})()
    text_encoder = type("TextEncoder", (), {"sequence_length": 1536})()
    configure_pipeline_token_length(pipeline, text_encoder)
    assert pipeline.token_length == 1536


@pytest.mark.parametrize(
    ("sequence_length", "expected_scale"),
    [
        (3608, 1.0),
        (50408, 16.0),
        (97208, 32.0),
    ],
)
def test_lingbot_flash_attention_value_scale(sequence_length: int, expected_scale: float):
    from xhmodel_merak.xh_other_model.models.lingbot_video.transformer_wrapper import (
        flash_attention_value_scale,
    )

    assert flash_attention_value_scale(sequence_length) == expected_scale


def test_lingbot_attention_setup_uses_current_flash_attention_contract():
    from types import SimpleNamespace

    from xhmodel_merak.xh_other_model.models.lingbot_video.transformer_wrapper import (
        _LingBotVideoAttention,
        xhnn,
    )

    attention = SimpleNamespace(
        head_dim=64,
        num_heads=4,
        lingbot_flash_attention_bits={name: 8 for name in ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")},
    )

    assert _LingBotVideoAttention._setup(attention) is attention
    assert isinstance(attention.flash_attn, xhnn.FlashAttention)
    assert attention.flash_attn.num_heads == 4
    assert attention.flash_attn.num_kv_heads == 4
    assert attention.flash_attn.is_causal is False
    assert attention.flash_attn.scale == 0.125
    assert {
        name: getattr(attention.flash_attn, name) for name in ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")
    } == {name: 8 for name in ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")}


def test_lingbot_attention_restores_sequence_major_output_layout():
    from types import SimpleNamespace

    from xhmodel_merak.xh_other_model.models.lingbot_video.transformer_wrapper import (
        _LingBotVideoAttention,
    )

    batch_size, sequence_length, num_heads, head_dim = 1, 3, 2, 2
    canonical_bhsd = torch.arange(
        batch_size * num_heads * sequence_length * head_dim,
        dtype=torch.float32,
    ).reshape(batch_size, num_heads, sequence_length, head_dim)
    current_input_length = torch.tensor([sequence_length], dtype=torch.int32)

    class FakeFlashAttention:
        def __call__(self, query, key, value, *, current_input_length):
            assert query.shape == key.shape == value.shape == canonical_bhsd.shape
            assert current_input_length is expected_current_input_length
            return canonical_bhsd

    expected_current_input_length = current_input_length
    attention = SimpleNamespace(
        lingbot_export_batch_size=batch_size,
        lingbot_export_sequence_length=sequence_length,
        lingbot_flash_attention_value_scale=1.0,
        num_heads=num_heads,
        head_dim=head_dim,
        to_q=torch.nn.Identity(),
        to_k=torch.nn.Identity(),
        to_v=torch.nn.Identity(),
        norm_q=torch.nn.Identity(),
        norm_k=torch.nn.Identity(),
        flash_attn=FakeFlashAttention(),
        to_out=torch.nn.Identity(),
        _apply_rotary=lambda hidden_states, rotary_emb: hidden_states,
    )
    hidden_states = torch.zeros(batch_size, sequence_length, num_heads * head_dim)

    actual = _LingBotVideoAttention.forward(
        attention,
        hidden_states,
        rotary_emb=(torch.empty(0), torch.empty(0)),
        current_input_length=current_input_length,
    )

    expected = canonical_bhsd.transpose(1, 2).reshape(batch_size, sequence_length, num_heads * head_dim)
    torch.testing.assert_close(actual, expected)


def test_lingbot_export_geometry_applies_attention_value_scale():
    from types import SimpleNamespace

    from xhmodel_merak.xh_other_model.models.lingbot_video.transformer_wrapper import (
        install_export_geometry,
    )

    class LingBotVideoAttention(torch.nn.Module):
        pass

    transformer = torch.nn.Module()
    transformer.config = SimpleNamespace(patch_size=(1, 2, 2))
    transformer.attention = LingBotVideoAttention()
    install_export_geometry(
        transformer,
        hidden_state_shape=(1, 16, 31, 60, 104),
        padded_text_length=2048,
    )

    assert transformer.lingbot_export_sequence_length == 50408
    assert transformer.lingbot_flash_attention_value_scale == 16.0
    assert transformer.attention.lingbot_flash_attention_value_scale == 16.0


def test_lingbot_qwen_image_processor_produces_hmonnx_pixels():
    from PIL import Image

    from xhmodel_merak.xh_other_model.models.lingbot_video.qwen3_vl_preprocess import (
        LingBotQwen3VLImageProcessor,
    )

    processor = LingBotQwen3VLImageProcessor(
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
    )
    outputs = processor(
        images=[Image.new("RGB", (8, 8), color=(17, 83, 149))],
        do_resize=False,
        return_tensors="pt",
    )

    assert outputs["pixel_values"].shape == (16, 24)
    torch.testing.assert_close(outputs["image_grid_thw"], torch.tensor([[1, 4, 4]]))
    assert len(outputs["hm_pixel_values"]) == 1
    assert outputs["hm_pixel_values"][0].shape == (1, 3, 2, 8, 8)
    assert outputs["hm_pixel_values"][0].dtype == torch.uint8


def test_lingbot_qwen_data_preprocess_builds_multimodal_graph_inputs():
    from xhmodel_merak.xh_other_model.models.lingbot_video.qwen3_vl_preprocess import (
        LingBotQwen3VLDataPreprocess,
    )

    embedding = torch.nn.Embedding(32, 8)
    preprocessor = LingBotQwen3VLDataPreprocess(
        token_embedding=embedding,
        input_sequence_length=8,
        image_token_id=11,
        video_token_id=12,
        vision_start_token_id=10,
        spatial_merge_size=2,
    )
    input_ids = torch.tensor([[10, 11, 11, 11, 11, 7]])
    image_embeds = torch.randn(4, 8)
    deepstack = [torch.randn(4, 8) for _ in range(3)]

    outputs = preprocessor(
        {
            "input_ids": input_ids,
            "past_seq_length": 0,
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
            "image_embeds": image_embeds,
            "deepstack_image_embeds": deepstack,
        }
    )

    assert len(outputs) == 9
    assert outputs[0].shape == (1, 8, 8)
    torch.testing.assert_close(outputs[0][0, 1:5], image_embeds)
    for index, feature in enumerate(deepstack):
        torch.testing.assert_close(outputs[6 + index][0, 1:5], feature)
    assert all(position_ids.shape == (8,) for position_ids in outputs[1:4])
    torch.testing.assert_close(outputs[4], torch.tensor([0], dtype=torch.int32))
    torch.testing.assert_close(outputs[5], torch.tensor([6], dtype=torch.int32))


@pytest.mark.parametrize("num_frames", [0, 2, 8, 120])
def test_lingbot_video_rejects_invalid_static_frame_profiles(num_frames: int):
    from xhmodel_merak.xh_other_model.models.lingbot_video.workflow import (
        LingBotVideoWorkflow,
    )

    config = yaml.safe_load(CONFIG_PATHS[0].read_text(encoding="utf-8"))["export"]
    geometry = dict(config["geometry"], mode="t2v", num_frames=num_frames)

    with pytest.raises(ValueError, match=r"num_frames must be 1 or 4n\+1"):
        LingBotVideoWorkflow._validate_static_profile(config["components"], geometry)


def test_lingbot_video_export_dispatches_all_components_and_copies_runtime(tmp_path, monkeypatch):
    from xhmodel_merak.xh_other_model.models.lingbot_video import text_encoder, transformer, vae

    model_dir = tmp_path / "model"
    required_files = (
        "model_index.json",
        "processor/config.json",
        "processor/preprocessor_config.json",
        "processor/tokenizer_config.json",
        "processor/video_preprocessor_config.json",
        "text_encoder/config.json",
        "transformer/config.json",
        "vae/config.json",
        "scheduler/scheduler_config.json",
    )
    for relative_path in required_files:
        path = model_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    calls = []

    class FakeQwenModel:
        def to(self, *args):
            calls.append(("qwen_to", args))
            return self

        def export_lingbot_hmonnx(self, output_dir):
            calls.append(("text_encoder", Path(output_dir)))
            return {"language_hmonnx": "language/model.onnx"}

    def fake_build_text_encoder(**kwargs):
        calls.append(("build_text_encoder", kwargs))
        return FakeQwenModel()

    def fake_export_transformer(**kwargs):
        calls.append(("transformer", kwargs))
        return {"hmonnx_file": "hmonnx/transformer.onnx"}

    def fake_export_vae_components(**kwargs):
        calls.append(("vae", kwargs))
        return {"encoder": {}, "decoder": {}}

    monkeypatch.setattr(text_encoder, "build_text_encoder", fake_build_text_encoder)
    monkeypatch.setattr(transformer, "export_transformer", fake_export_transformer)
    monkeypatch.setattr(vae, "export_vae_components", fake_export_vae_components)

    workflow = AutoWorkflow.from_config(
        model_dir=str(model_dir),
        config_path=str(CONFIG_PATHS[0]),
    )
    quant_result = workflow.quant(output_dir="unused", device="cpu")
    output_dir = tmp_path / "export"
    result = workflow.export(quant_result, str(output_dir), "cpu")

    assert [call[0] for call in calls] == [
        "build_text_encoder",
        "qwen_to",
        "text_encoder",
        "transformer",
        "vae",
    ]
    assert result.work_dir == str(output_dir)
    meta = json.loads((output_dir / "export_meta_info.json").read_text(encoding="utf-8"))
    assert meta["components"] == [
        "text_encoder",
        "visual_encoder",
        "transformer",
        "vae_encoder",
        "vae_decoder",
    ]
    for relative_path in (
        "runtime/processor/config.json",
        "runtime/scheduler/scheduler_config.json",
        "runtime/text_encoder/config.json",
        "runtime/transformer/config.json",
        "runtime/vae/config.json",
    ):
        assert (output_dir / relative_path).is_file()


def test_wan_causal_conv3d_decomposition_matches_source():
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    from xhmodel_merak.xh_other_model.models.lingbot_video.vae import (
        WanCausalConv3dAs2d,
    )

    torch.manual_seed(7)
    source = WanCausalConv3d(
        in_channels=2,
        out_channels=3,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1),
    ).eval()
    replacement = WanCausalConv3dAs2d(source).eval()
    sample = torch.randn(1, 2, 4, 7, 9)

    torch.testing.assert_close(replacement(sample), source(sample), rtol=1e-5, atol=1e-6)

    cache = torch.randn(1, 2, 2, 7, 9)
    next_sample = sample[:, :, :1]
    torch.testing.assert_close(
        replacement(next_sample, cache),
        source(next_sample, cache),
        rtol=1e-5,
        atol=1e-6,
    )


def test_qwen3_vl_text_rope_cache_uses_export_compute_device():
    from xhmodel_merak.xh_other_model.models.lingbot_video._qwen3_vl_modeling import (
        _LingBotQwen3VLTextRotaryEmbedding,
    )

    inv_freq = torch.tensor([1.0, 0.25, 0.0625], dtype=torch.float16)
    rotary = type("Rotary", (), {"inv_freq": inv_freq, "attention_scaling": 1.0})()
    cos, sin = _LingBotQwen3VLTextRotaryEmbedding.forward(rotary, max_seq_len_cached=32)

    compute_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    compute_freq = inv_freq.to(compute_device)
    expanded = compute_freq[None, None, :, None].float().expand(1, 32, -1, 1)
    positions = torch.arange(32, device=compute_device).float()[None, :, None, None]
    frequencies = (expanded @ positions).transpose(2, 3)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    expected_cos = embedding.cos().to(inv_freq.dtype).cpu()
    expected_sin = embedding.sin().to(inv_freq.dtype).cpu()

    torch.testing.assert_close(cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(sin, expected_sin, rtol=0, atol=0)


def test_qwen3_vl_visual_patch_decomposition_preserves_bias_and_layout():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed

    from xhmodel_merak.xh_other_model.builder import wrap_llm_model
    from xhmodel_merak.xh_other_model.models.lingbot_video._qwen3_vl_modeling import (
        register_qwen3_vl_wrappers,
    )
    from xhmodel_merak.xh_other_model.models.lingbot_video.text_encoder import (
        _visual_wrap_config,
    )

    torch.manual_seed(13)
    config = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        out_hidden_size=8,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    patch_embed = Qwen3VLVisionPatchEmbed(config).eval()
    patch_embed.proj.bias.data.uniform_(-1.0, 1.0)
    pixels = torch.randint(0, 256, (1, 3, 2, 8, 8), dtype=torch.int32).float()

    normalized = (pixels / 255.0 - 0.5) / 0.5
    expected = patch_embed.proj(normalized)
    batch_size, channels, temporal_patches, height, width = expected.shape
    expected = expected.permute(0, 2, 1, 3, 4).reshape(
        batch_size,
        temporal_patches,
        channels,
        height // 2,
        2,
        width // 2,
        2,
    )
    expected = expected.permute(0, 1, 3, 5, 4, 6, 2).reshape(batch_size, -1, channels)

    register_qwen3_vl_wrappers()
    wrapped = wrap_llm_model(
        patch_embed,
        _visual_wrap_config(
            {
                "max_size_w": 8,
                "max_size_h": 8,
                "max_size_t": 2,
                "patch_size": 2,
                "temporal_patch_size": 2,
            },
            "visual_patch_test",
        ),
    )

    torch.testing.assert_close(wrapped(pixels), expected, rtol=1e-5, atol=1e-5)


def test_lingbot_video_golden_inputs_follow_hmonnx_dtype_contract(tmp_path):
    from xhmodel_merak.xh_other_model.models.lingbot_video.workflow import (
        _dump_one_golden,
    )

    hmonnx_file = tmp_path / "model.onnx"
    hmonnx_file.touch()
    inputs_file = tmp_path / "calibration_inputs.pt"
    torch.save(
        [
            torch.randn(1, 3, dtype=torch.float32),
            torch.tensor([7], dtype=torch.int64),
            torch.tensor([True], dtype=torch.bool),
        ],
        inputs_file,
    )
    captured = {}

    class InputInfo:
        def __init__(self, dtype):
            self.dtype = dtype

    class FakeSession:
        expected_dtypes = {
            "pixels": torch.float16,
            "length": torch.int32,
            "mask": torch.bool,
        }

        def __init__(self, hmonnx_path):
            assert hmonnx_path == str(hmonnx_file)
            self.save_golden = False

        def to(self, device):
            captured["device"] = device
            return self

        def initialize(self):
            captured["initialized"] = True

        def get_input_names(self):
            return list(self.expected_dtypes)

        def get_input(self, name):
            return InputInfo(self.expected_dtypes[name])

        def __call__(self, *inputs):
            captured["inputs"] = inputs
            captured["save_golden"] = self.save_golden
            captured["golden_dir"] = self.golden_dir

    golden_dir = tmp_path / "golden"
    _dump_one_golden(
        runtime_cls=FakeSession,
        hmonnx_file=hmonnx_file,
        inputs_file=inputs_file,
        golden_dir=golden_dir,
        device="cpu",
    )

    assert captured["initialized"] is True
    assert captured["device"] == torch.device("cpu")
    assert captured["save_golden"] is True
    assert captured["golden_dir"] == str(golden_dir)
    assert [tensor.dtype for tensor in captured["inputs"]] == [torch.float16, torch.int32, torch.bool]
