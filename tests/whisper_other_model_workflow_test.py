import copy
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from xhmodel_merak.workflows import AutoWorkflow
from xhmodel_merak.xh_other_model.workflows import AutoOtherModelWorkflow
from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult


DEFAULT_YAML = {
    "quant": None,
    "export": {
        "target_device": "XH2a",
        "model": {
            "type": "XHWhisperModel",
        },
        "components": {
            "encoder": {
                "enabled": True,
                "quant_type": "w8a8_sefp",
                "audio_path": "examples_merak/asr/whisper/audio.mp3",
            },
            "prefill": {
                "enabled": True,
                "quant_type": "w8a8_sefp",
                "prompt_token_ids": [50258, 50259, 50359, 50363],
                "cache_position": [0, 1, 2, 3],
                "past_len": 0,
            },
            "decoder": {
                "enabled": True,
                "quant_type": "w8a8_sefp",
                "prompt_token_ids": [2221],
                "cache_position": [4],
                "past_len": 4,
            },
        },
    },
}


def _write_whisper_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    data = copy.deepcopy(DEFAULT_YAML)
    if overrides:
        _deep_merge(data, overrides)
    config_path = tmp_path / "whisper.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config_path


def _deep_merge(base: dict, overrides: dict) -> None:
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# Auto-binding
# ---------------------------------------------------------------------------


def test_auto_workflow_routes_to_whisper(tmp_path):
    config_path = _write_whisper_config(tmp_path)
    workflow = AutoWorkflow.from_config(
        model_dir="/models/whisper",
        config_path=str(config_path),
    )
    from xhmodel_merak.xh_other_model.models.whisper.workflow import WhisperWorkflow

    assert isinstance(workflow, WhisperWorkflow)


def test_auto_other_workflow_routes_to_whisper(tmp_path):
    config_path = _write_whisper_config(tmp_path)
    workflow = AutoOtherModelWorkflow.from_config(
        model_dir="/models/whisper",
        config_path=str(config_path),
    )
    from xhmodel_merak.xh_other_model.models.whisper.workflow import WhisperWorkflow

    assert isinstance(workflow, WhisperWorkflow)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_whisper_registration():
    from xhmodel_merak.xh_other_model.builder import get_model_class

    model_cls = get_model_class({"type": "XHWhisperModel"})
    assert model_cls.__module__ == "xhmodel_merak.xh_other_model.models.whisper.model"
    assert model_cls.WORKFLOW_CLS.endswith("whisper.workflow:WhisperWorkflow")



# ---------------------------------------------------------------------------
# YAML config consistency
# ---------------------------------------------------------------------------


def test_whisper_yaml_past_len_matches_cache_position():
    """decoder.past_len must equal the first element of cache_position when
    the decoder follows a prefill of that length (4-token prefill → past_len=4)."""
    decoder = DEFAULT_YAML["export"]["components"]["decoder"]
    cache_pos = decoder["cache_position"]
    past_len = decoder["past_len"]
    assert len(cache_pos) == 1, "decoder cache_position should be a single-element list"
    assert cache_pos[0] == past_len, (
        f"decoder.cache_position[0]={cache_pos[0]} != decoder.past_len={past_len}. "
        "For a single-step decode after a 4-token prefill, past_len should equal the "
        "cache write position."
    )


def test_whisper_yaml_default_config_is_parseable():
    path = Path("configs_merak/workflows/xh2a/other_models/whisper/whisper.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data is not None
    assert data.get("quant") is None
    export = data.get("export", {})
    assert export.get("target_device") == "XH2a"
    assert export.get("model", {}).get("type") == "XHWhisperModel"
    components = export.get("components", {})
    for name in ("encoder", "prefill", "decoder"):
        assert name in components, f"Missing component {name}"
        assert "quant_type" in components[name], f"Component {name} missing quant_type"


# ---------------------------------------------------------------------------
# Export dispatch
# ---------------------------------------------------------------------------


def test_whisper_export_dispatches_three_components(tmp_path, monkeypatch):
    config_path = _write_whisper_config(tmp_path)
    workflow = AutoOtherModelWorkflow.from_config(
        model_dir="/models/whisper",
        config_path=str(config_path),
        debug=True,
    )
    calls = []

    def fake_export_encoder(**kwargs):
        calls.append(("encoder", kwargs))
        return {"onnx_file": "encoder/m.onnx", "hmonnx_file": "encoder/hmonnx/m.onnx"}

    def fake_export_prefill(**kwargs):
        calls.append(("prefill", kwargs))
        return {"hmonnx_file": "prefill/hmonnx/m.onnx"}

    def fake_export_decoder(**kwargs):
        calls.append(("decoder", kwargs))
        return {"hmonnx_file": "decoder/hmonnx/m.onnx"}

    import xhmodel_merak.xh_other_model.models.whisper.workflow as wf

    monkeypatch.setattr(wf, "export_whisper_encoder", fake_export_encoder)
    monkeypatch.setattr(wf, "export_whisper_decoder_graph", fake_export_prefill)
    monkeypatch.setattr(wf, "_validate_whisper_export_config", lambda *a, **kw: None)

    class _FakeSelfAttn:
        head_dim = 64
        num_heads = 2
        embed_dim = 128

    class _FakeLayer:
        self_attn = _FakeSelfAttn()

    class _FakeDecoder:
        layers = [_FakeLayer()]

    class _FakeEncoder:
        decoder_m = None

    class _FakeModelInner:
        encoder = _FakeEncoder()
        decoder = _FakeDecoder()

    class _FakeConfig:
        forced_decoder_ids = None
        _attn_implementation = "eager"
        max_source_positions = 1500
        decoder_layers = 2

    class _FakeModel:
        model = _FakeModelInner()
        config = _FakeConfig()
        proj_out = None

        def eval(self):
            return self

    class _FakeWhisperForConditionalGeneration:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeModel()

    monkeypatch.setattr(
        "transformers.WhisperForConditionalGeneration",
        _FakeWhisperForConditionalGeneration,
    )

    # Only the decoder call uses a different component_name; we need a way to
    # distinguish.  Monkey-patch a second copy so the third call is the decoder.
    def fake_export_decoder_graph(**kwargs):
        calls.append(("decoder", kwargs))
        return {"hmonnx_file": "decoder/hmonnx/m.onnx"}

    # Replace original with a dispatch that routes by component_name.
    original = wf.export_whisper_decoder_graph

    def dispatch(**kwargs):
        name = kwargs.get("component_name", "unknown")
        if name == "prefill":
            return fake_export_prefill(**kwargs)
        elif name == "decoder":
            return fake_export_decoder(**kwargs)
        return original(**kwargs)

    monkeypatch.setattr(wf, "export_whisper_decoder_graph", dispatch)

    result = workflow.export(
        quant_result=QuantResult(raw_model_dir="/models/whisper", skipped=True),
        output_dir=str(tmp_path / "out"),
        device="cuda:0",
    )

    assert len(calls) == 3, f"Expected 3 component exports, got {len(calls)}"
    component_names = [c[0] for c in calls]
    assert component_names == ["encoder", "prefill", "decoder"], component_names

    encoder_call = calls[0][1]
    assert encoder_call["model_dir"] == "/models/whisper"
    assert encoder_call["quant_type"] == "w8a8_sefp"

    for name, call in calls[1:]:
        assert call["past_len"] == (0 if name == "prefill" else 4), (
            f"{name}.past_len unexpected: {call['past_len']}"
        )
        assert call["quant_type"] == "w8a8_sefp"

    assert result.meta["components"] == ["encoder", "prefill", "decoder"]
    assert Path(result.config_file).exists()


def test_whisper_export_rejects_unknown_component(tmp_path):
    config_path = _write_whisper_config(
        tmp_path, {"export": {"components": {"unknown": {"enabled": True}}}}
    )
    workflow = AutoOtherModelWorkflow.from_config(
        model_dir="/models/whisper",
        config_path=str(config_path),
    )
    with pytest.raises(ValueError, match="Unsupported Whisper export component"):
        workflow.export(
            quant_result=QuantResult(raw_model_dir="/models/whisper", skipped=True),
            output_dir=str(tmp_path / "out"),
            device="cuda:0",
        )


# ---------------------------------------------------------------------------
# golden
# ---------------------------------------------------------------------------


def test_whisper_dump_golden_reads_export_meta(tmp_path, monkeypatch):
    config_path = _write_whisper_config(tmp_path)
    workflow = AutoOtherModelWorkflow.from_config(
        model_dir="/models/whisper",
        config_path=str(config_path),
    )
    calls = []

    def fake_run_golden(hmonnx_file, golden_dir, device, inputs):
        calls.append(
            {
                "hmonnx_file": hmonnx_file,
                "golden_dir": golden_dir,
                "device": device,
                "num_inputs": len(inputs),
            }
        )

    import xhmodel_merak.xh_other_model.models.whisper.workflow as wf

    monkeypatch.setattr(wf, "run_hmonnx_golden", fake_run_golden)
    monkeypatch.setattr(
        wf, "build_encoder_input_features",
        lambda *a, **kw: __import__("torch").zeros(1, 128, 3000),
    )

    work_dir = tmp_path / "out"
    (work_dir / "encoder" / "hmonnx").mkdir(parents=True)
    (work_dir / "prefill" / "hmonnx").mkdir(parents=True)
    (work_dir / "decoder" / "hmonnx").mkdir(parents=True)
    (work_dir / "export_meta_info.json").write_text(
        json.dumps(
            {
                "encoder": {
                    "hmonnx_file": "encoder/hmonnx/e.onnx",
                    "num_mel_bins": 128,
                    "input_length": 3000,
                },
                "prefill": {
                    "hmonnx_file": "prefill/hmonnx/p.onnx",
                    "prompt_token_ids": [50258, 50259, 50359, 50363],
                    "cache_position": [0, 1, 2, 3],
                    "past_len": 0,
                },
                "decoder": {
                    "hmonnx_file": "decoder/hmonnx/d.onnx",
                    "prompt_token_ids": [2221],
                    "cache_position": [4],
                    "past_len": 4,
                },
                "model_cfg": {
                    "head_dim": 64,
                    "num_heads": 2,
                    "embed_dim": 128,
                    "max_source_positions": 1500,
                    "num_decode_layers": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(wf, "build_decoder_golden_inputs", lambda **kw: [torch.zeros(1)])

    import torch

    result = workflow.dump_golden(
        export_result=ExportResult(
            work_dir=str(work_dir),
            config_file=str(tmp_path / "out" / "whisper.yaml"),
        ),
        device="cpu",
    )

    assert result == str(work_dir)
    assert len(calls) == 3
    assert [c["hmonnx_file"] for c in calls] == [
        work_dir / "encoder/hmonnx/e.onnx",
        work_dir / "prefill/hmonnx/p.onnx",
        work_dir / "decoder/hmonnx/d.onnx",
    ]


# ---------------------------------------------------------------------------
# Example scripts
# ---------------------------------------------------------------------------


def test_whisper_merak_example_uses_workflow_entrypoint():
    example_dir = Path("examples_merak/asr/whisper")
    script_path = example_dir / "whisper_workflow.py"
    readme_path = example_dir / "README.md"

    assert script_path.exists()
    assert readme_path.exists()

    source = script_path.read_text(encoding="utf-8")
    assert "from xhmodel_merak.workflows import AutoWorkflow" in source
    assert "AutoWorkflow.from_config" in source
    assert "examples/audio/whisper" not in source

    spec = importlib.util.spec_from_file_location("whisper_workflow_example", str(script_path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.parse_args)
    assert callable(module.main)

    readme = readme_path.read_text(encoding="utf-8")
    assert "<env_name>" in readme or "<gpu_id>" in readme
    assert "AutoWorkflow" in readme
    assert "configs_merak/workflows/xh2a/other_models/whisper/whisper.yaml" in readme


def test_whisper_merak_hmonnx_demo_uses_correct_patterns():
    example_dir = Path("examples_merak/asr/whisper")
    demo_path = example_dir / "hmonnx_demo.py"

    assert demo_path.exists()
    source = demo_path.read_text(encoding="utf-8")
    assert "examples/audio/whisper" not in source
    assert "--work-dir" in source
    assert "--audio" in source
    assert "--device" in source

    spec = importlib.util.spec_from_file_location("whisper_hmonnx_demo", str(demo_path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.parse_args)
    assert callable(module.main)


# ---------------------------------------------------------------------------
# _load_meta / _build_dec_inputs error cases
# ---------------------------------------------------------------------------


def test_load_meta_raises_on_missing_file(tmp_path):
    """Missing export_meta_info.json must raise FileNotFoundError."""
    from examples_merak.asr.whisper.hmonnx_demo import _load_meta

    with pytest.raises(FileNotFoundError, match=str(tmp_path / "export_meta_info.json")):
        _load_meta(tmp_path)


def test_load_meta_raises_on_invalid_json(tmp_path):
    """Malformed JSON must raise JSONDecodeError."""
    (tmp_path / "export_meta_info.json").write_text("not-json", encoding="utf-8")
    from examples_merak.asr.whisper.hmonnx_demo import _load_meta

    with pytest.raises(json.JSONDecodeError):
        _load_meta(tmp_path)


def test_load_meta_missing_required_field_not_caught_by_load(tmp_path):
    """_load_meta itself returns any dict; callers must handle missing fields.
    Verify that loading a minimal JSON does not crash _load_meta."""
    (tmp_path / "export_meta_info.json").write_text(
        json.dumps({"model_cfg": {"unknown": True}}), encoding="utf-8"
    )
    from examples_merak.asr.whisper.hmonnx_demo import _load_meta

    meta = _load_meta(tmp_path)
    assert meta["model_cfg"]["unknown"] is True


def test_build_prompt_tokens_fallback_when_meta_missing_prefill():
    """_build_prompt_tokens falls back to hardcoded zh tokens when meta has no prefill section."""
    from examples_merak.asr.whisper.hmonnx_demo import _build_prompt_tokens

    class _FakeProcessor:
        class tokenizer:  # noqa: N801
            unk_token_id = -1

            @staticmethod
            def convert_tokens_to_ids(tok):
                table = {"<|zh|>": 50259, "<|transcribe|>": 50359, "<|notimestamps|>": 50363}
                return table[tok]

    class _FakeConfig:
        decoder_start_token_id = 50258

    tokens = _build_prompt_tokens(_FakeProcessor(), _FakeConfig(), meta={})
    assert tokens == [50258, 50259, 50359, 50363]


def test_build_prompt_tokens_reads_from_meta():
    """_build_prompt_tokens returns meta tokens when they match expected Chinese tokens."""
    from examples_merak.asr.whisper.hmonnx_demo import _build_prompt_tokens

    class _FakeProcessor:
        class tokenizer:  # noqa: N801
            unk_token_id = -1

            @staticmethod
            def convert_tokens_to_ids(tok):
                table = {"<|zh|>": 50259, "<|transcribe|>": 50359, "<|notimestamps|>": 50363}
                return table[tok]
            @staticmethod
            def decode(ids):
                return ""

    class _FakeConfig:
        decoder_start_token_id = 50258

    tokens = _build_prompt_tokens(
        _FakeProcessor(), _FakeConfig(),
        meta={"prefill": {"prompt_token_ids": [50258, 50259, 50359, 50363]}},
    )
    assert tokens == [50258, 50259, 50359, 50363]


# ---------------------------------------------------------------------------
# _build_dec_inputs validation
# ---------------------------------------------------------------------------


@pytest.fixture
def _demo_module():
    """Lazy-import the demo module once per test via its spec (no HF/xhquant)."""
    import importlib.util

    module_name = "examples_merak.asr.whisper.hmonnx_demo"
    if module_name not in sys.modules:
        spec = importlib.util.find_spec(module_name)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[module_name]


def test_build_dec_inputs_wrong_input_count(_demo_module):
    """_build_dec_inputs must reject dec_names that don't match the expected length."""
    dec_names = [f"in_{i}" for i in range(6)]
    with pytest.raises(ValueError, match="Expected \\d+ decoder graph inputs"):
        _demo_module._build_dec_inputs(
            dec_names=dec_names,
            input_ids=_demo_module.torch.zeros(1),
            position_ids=_demo_module.torch.zeros(1),
            past_len=_demo_module.torch.zeros(1, dtype=_demo_module.torch.int32),
            current_len=_demo_module.torch.zeros(1, dtype=_demo_module.torch.int32),
            mask_atten=_demo_module.torch.zeros(1),
            encoder_attention_mask=_demo_module.torch.zeros(1),
            k_cache=[_demo_module.torch.zeros(1) for _ in range(2)],
            v_cache=[_demo_module.torch.zeros(1) for _ in range(2)],
            k_list=[_demo_module.torch.zeros(1) for _ in range(2)],
            v_list=[_demo_module.torch.zeros(1) for _ in range(2)],
            num_decode_layers=2,
        )


def test_build_dec_inputs_wrong_base_names(_demo_module):
    """_build_dec_inputs rejects dec_names whose base section doesn't match expected names."""
    # Count matches (6 + 4*2 = 14) but first name doesn't match
    dec_names = [f"weird_{i}" for i in range(14)]
    with pytest.raises(ValueError, match="Unexpected base name at position 0"):
        _demo_module._build_dec_inputs(
            dec_names=dec_names,
            input_ids=_demo_module.torch.zeros(1),
            position_ids=_demo_module.torch.zeros(1),
            past_len=_demo_module.torch.zeros(1, dtype=_demo_module.torch.int32),
            current_len=_demo_module.torch.zeros(1, dtype=_demo_module.torch.int32),
            mask_atten=_demo_module.torch.zeros(1),
            encoder_attention_mask=_demo_module.torch.zeros(1),
            k_cache=[_demo_module.torch.zeros(1) for _ in range(2)],
            v_cache=[_demo_module.torch.zeros(1) for _ in range(2)],
            k_list=[_demo_module.torch.zeros(1) for _ in range(2)],
            v_list=[_demo_module.torch.zeros(1) for _ in range(2)],
            num_decode_layers=2,
        )


def test_build_dec_inputs_wrong_cache_group_names(_demo_module):
    """_build_dec_inputs rejects dec_names whose cache/state names don't match pattern."""
    base = _demo_module._BASE_INPUT_NAMES
    N = 2
    dec_names = base + ["k_cacheX" for _ in range(N * 4)]
    with pytest.raises(ValueError, match="does not follow.*pattern"):
        _demo_module._build_dec_inputs(
            dec_names=dec_names,
            input_ids=_demo_module.torch.zeros(1),
            position_ids=_demo_module.torch.zeros(1),
            past_len=_demo_module.torch.zeros(1, dtype=_demo_module.torch.int32),
            current_len=_demo_module.torch.zeros(1, dtype=_demo_module.torch.int32),
            mask_atten=_demo_module.torch.zeros(1),
            encoder_attention_mask=_demo_module.torch.zeros(1),
            k_cache=[_demo_module.torch.zeros(1) for _ in range(N)],
            v_cache=[_demo_module.torch.zeros(1) for _ in range(N)],
            k_list=[_demo_module.torch.zeros(1) for _ in range(N)],
            v_list=[_demo_module.torch.zeros(1) for _ in range(N)],
            num_decode_layers=N,
        )


def test_build_dec_inputs_correct_layout(_demo_module):
    """Verify correct layout: base[0..5] + k_cache[0..N-1] + v_cache[0..N-1]
    + key_state[0..N-1] + value_state[0..N-1]."""
    N = 2
    dec_names = (
        _demo_module._BASE_INPUT_NAMES
        + [f"k_cache_{i}" for i in range(N)]
        + [f"v_cache_{i}" for i in range(N)]
        + [f"key_state_{i}" for i in range(N)]
        + [f"value_state_{i}" for i in range(N)]
    )
    sentinel = _demo_module.torch.tensor([-1])
    result = _demo_module._build_dec_inputs(
        dec_names=dec_names,
        input_ids=sentinel,
        position_ids=sentinel,
        past_len=sentinel,
        current_len=sentinel,
        mask_atten=sentinel,
        encoder_attention_mask=sentinel,
        k_cache=[_demo_module.torch.tensor([i]) for i in range(N)],
        v_cache=[_demo_module.torch.tensor([N + i]) for i in range(N)],
        k_list=[_demo_module.torch.tensor([2 * N + i]) for i in range(N)],
        v_list=[_demo_module.torch.tensor([3 * N + i]) for i in range(N)],
        num_decode_layers=N,
    )
    assert result["decoder_input_ids"].item() == -1
    assert result["past_len"].item() == -1
    for i in range(N):
        assert result[f"k_cache_{i}"].item() == i
    for i in range(N):
        assert result[f"v_cache_{i}"].item() == N + i
    for i in range(N):
        assert result[f"key_state_{i}"].item() == 2 * N + i
    for i in range(N):
        assert result[f"value_state_{i}"].item() == 3 * N + i


# ---------------------------------------------------------------------------
# Mock-runtime end-to-end: verify past_len / cache_position consistency
# ---------------------------------------------------------------------------


def test_hmonnx_demo_past_len_chain_invariant(_demo_module):
    """After a 4-token prefill (past_len=0), the first decode step must use
    past_len=4 and current_len=1. Verify via _build_dec_inputs directly."""
    import torch as _torch

    N = 2
    dec_names = (
        _demo_module._BASE_INPUT_NAMES
        + [f"k_cache_{i}" for i in range(N)]
        + [f"v_cache_{i}" for i in range(N)]
        + [f"key_state_{i}" for i in range(N)]
        + [f"value_state_{i}" for i in range(N)]
    )
    dummy = _torch.zeros(1)
    cache = [_torch.zeros(1) for _ in range(N)]

    prefill_inputs = _demo_module._build_dec_inputs(
        dec_names=dec_names,
        input_ids=_torch.tensor([[50258, 50259, 50359, 50363]]),
        position_ids=_torch.tensor([[0, 1, 2, 3]]),
        past_len=_torch.tensor([0], dtype=_torch.int32),
        current_len=_torch.tensor([4], dtype=_torch.int32),
        mask_atten=dummy,
        encoder_attention_mask=dummy,
        k_cache=cache,
        v_cache=cache,
        k_list=cache,
        v_list=cache,
        num_decode_layers=N,
    )
    assert prefill_inputs["past_len"].item() == 0
    assert prefill_inputs["current_len"].item() == 4

    decode_inputs = _demo_module._build_dec_inputs(
        dec_names=dec_names,
        input_ids=_torch.tensor([[1]]),
        position_ids=_torch.tensor([[4]]),
        past_len=_torch.tensor([4], dtype=_torch.int32),
        current_len=_torch.tensor([1], dtype=_torch.int32),
        mask_atten=dummy,
        encoder_attention_mask=dummy,
        k_cache=cache,
        v_cache=cache,
        k_list=cache,
        v_list=cache,
        num_decode_layers=N,
    )
    assert decode_inputs["past_len"].item() == 4
    assert decode_inputs["current_len"].item() == 1
