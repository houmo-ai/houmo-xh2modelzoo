from __future__ import annotations

from types import SimpleNamespace

import torch


def _run_llm_decoder_graph(*args, **kwargs):
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_text_decoder import _iter_decoder_graph_outputs
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_llm import _collect_llm_decoder_outputs

    outputs = _iter_decoder_graph_outputs(*args, **kwargs)
    return _collect_llm_decoder_outputs(outputs, args[2].shape[1])


def _run_tts_decoder_graph(*args, **kwargs):
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_text_decoder import _iter_decoder_graph_outputs
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_tts import _last_tts_decoder_output

    return _last_tts_decoder_output(_iter_decoder_graph_outputs(*args, **kwargs))


def test_official_tts_gen_logits_resolves_loaded_modeling_module(monkeypatch) -> None:
    """gen_logits must be resolved from the already-imported official modeling module,
    not a brittle hard-coded hyphen-encoded module path."""
    import sys
    import types

    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import hf_compatible

    calls: list[tuple[int, float]] = []

    def gen_logits(num_code, repetition_penalty, **kwargs):
        del kwargs
        calls.append((num_code, repetition_penalty))
        return ["warper"], ["processor"]

    fake = types.ModuleType("transformers_modules.Some_hyphen_name.modeling_minicpmo")
    fake.gen_logits = gen_logits
    monkeypatch.setitem(sys.modules, fake.__name__, fake)

    result = hf_compatible._official_tts_gen_logits(4096, 1.05)

    assert calls == [(4096, 1.05)]
    assert result == (["warper"], ["processor"])


def test_vision_runtime_executes_exported_batch_one_serially() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_vision import run_vision_slices

    calls: list[int] = []

    def session(*inputs: torch.Tensor) -> torch.Tensor:
        calls.append(inputs[0].shape[0])
        return inputs[0].sum(dim=(-1, -2), keepdim=True)

    pixels = torch.ones((3, 3, 2, 2), dtype=torch.float16)
    position_ids = torch.zeros((3, 4), dtype=torch.int32)
    attention_mask = torch.zeros((3, 1, 4, 4), dtype=torch.float16)
    target_sizes = torch.ones((3, 2), dtype=torch.int32)

    output = run_vision_slices(session, pixels, position_ids, attention_mask, target_sizes)

    assert calls == [1, 1, 1]
    assert output.shape[0] == 3


def test_vision_runtime_prepares_padded_hmonnx_inputs() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_vision import prepare_vision_inputs

    data = {
        "pixel_values": [[torch.ones((1, 3, 4), dtype=torch.float16)]],
        "tgt_sizes": [torch.tensor([[2, 2]], dtype=torch.int32)],
    }

    pixels, positions, mask, sizes, image_counts = prepare_vision_inputs(
        data,
        patch_size=1,
        num_patches_per_side=4,
        image_slice_max_size=(2, 2),
    )

    assert pixels.shape == (1, 3, 1, 4)
    assert positions.shape == (1, 4)
    assert mask.shape == (1, 1, 4, 4)
    assert sizes.tolist() == [[2, 2]]
    assert image_counts == [1]


def test_vision_runtime_preserves_navit_patch_layout() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_vision import prepare_vision_inputs

    source = torch.arange(24, dtype=torch.float16).reshape(3, 2, 4)
    data = {
        "pixel_values": [[source]],
        "tgt_sizes": [torch.tensor([[2, 2]], dtype=torch.int32)],
    }

    pixels, _, _, _, _ = prepare_vision_inputs(
        data,
        patch_size=2,
        num_patches_per_side=4,
        image_slice_max_size=(2, 2),
    )

    flattened = source.flatten(end_dim=1).permute(1, 0)
    padded = torch.zeros((8, 6), dtype=torch.float16)
    padded[: flattened.shape[0]] = flattened
    expected = padded.unsqueeze(0).permute(0, 2, 1).reshape(1, 3, 2, 8)
    assert torch.equal(pixels, expected)


def test_vision_export_dummy_uses_fixed_navit_layout() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_vision import capture_vision_inputs

    host = SimpleNamespace(vpm=SimpleNamespace(embeddings=SimpleNamespace(patch_size=14)))
    model = SimpleNamespace(wrap_cfg=SimpleNamespace(image_slice_max_size=[40, 40]))

    pixels, position_ids, attention_mask, sizes = capture_vision_inputs(host, model)

    assert pixels.shape == (1, 3, 14, 22400)
    assert position_ids.shape == (1, 1600)
    assert attention_mask.shape == (1, 1, 1600, 1600)
    assert sizes.tolist() == [[40, 40]]


def test_vision_runtime_wrap_model_delegates_to_forward() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_vision import MiniCPMO45VisionHMONNXRuntime

    runtime = object.__new__(MiniCPMO45VisionHMONNXRuntime)
    expected = torch.ones((1, 2, 3), dtype=torch.float16)
    runtime.forward = lambda *args: expected

    result = runtime._wrap_model(
        torch.zeros((1, 3, 2, 2)),
        torch.zeros((1, 4), dtype=torch.int64),
        torch.zeros((1, 1, 4, 4)),
        torch.ones((1, 2), dtype=torch.int32),
    )

    assert result is expected


def test_audio_runtime_pads_static_batch_and_trims_output() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_audio import run_static_audio_batch

    observed: list[tuple[int, int]] = []

    def session(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        observed.append((features.shape[0], mask.shape[0]))
        return torch.arange(features.shape[0] * 2, dtype=torch.float16).reshape(features.shape[0], 2)

    output = run_static_audio_batch(
        session,
        torch.ones((1, 80, 16), dtype=torch.float32),
        torch.ones((1, 1, 8, 8), dtype=torch.float32),
        exported_batch=4,
    )

    assert observed == [(4, 4)]
    assert output.shape == (1, 2)


def test_audio_runtime_pads_static_time_dimension() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_audio import run_static_audio_batch

    observed: list[tuple[torch.Size, torch.Size]] = []

    def session(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        observed.append((features.shape, mask.shape))
        return features[:, :, :2]

    run_static_audio_batch(
        session,
        torch.ones((1, 80, 9), dtype=torch.float32),
        torch.ones((1, 1, 4, 4), dtype=torch.float32),
        exported_batch=4,
        exported_frames=12,
        exported_mask_size=6,
    )

    assert observed == [(torch.Size([4, 80, 12]), torch.Size([4, 1, 6, 6]))]


def test_audio_runtime_splits_batches_larger_than_exported_capacity() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_audio import run_static_audio_batch

    observed: list[tuple[int, int]] = []

    def session(features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observed.append((features.shape[0], mask.shape[0]))
        return features[:, 0, :1], mask[:, 0, :1, :1]

    sample_ids = torch.arange(9, dtype=torch.float32).reshape(9, 1, 1).expand(-1, 80, 16)
    masks = torch.arange(9, dtype=torch.float32).reshape(9, 1, 1, 1).expand(-1, 1, 8, 8)
    output, output_mask = run_static_audio_batch(
        session,
        sample_ids,
        masks,
        exported_batch=4,
    )

    assert observed == [(4, 4), (4, 4), (4, 4)]
    assert output[:, 0].tolist() == list(range(9))
    assert output_mask[:, 0, 0].tolist() == list(range(9))


def test_llm_runtime_splits_long_prefill_and_concatenates_valid_hidden_states() -> None:
    calls: list[tuple[int, int, int]] = []

    def prefill(*inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        past = int(inputs[1].item())
        current = int(inputs[2].item())
        calls.append((past, current, inputs[0].shape[1]))
        logits = torch.full((1, 1, 2), float(len(calls)), dtype=torch.float16)
        hidden = torch.full_like(inputs[0], float(len(calls)))
        return logits, hidden

    def decode(*inputs: torch.Tensor) -> torch.Tensor:
        raise AssertionError("decode must not run for an empty-past multi-token input")

    logits, hidden = _run_llm_decoder_graph(
        prefill,
        decode,
        torch.ones((1, 10, 8), dtype=torch.float16),
        past_seq_length=0,
        current_input_length=10,
        past_key_caches=[torch.zeros((1, 2, 16, 4), dtype=torch.float16)],
        past_value_caches=[torch.zeros((1, 2, 16, 4), dtype=torch.float16)],
        prefill_length=4,
    )

    assert calls == [(0, 4, 4), (4, 4, 4), (8, 2, 4)]
    assert logits.shape == (1, 1, 2)
    assert hidden.shape == (1, 10, 8)
    assert torch.all(hidden[:, :4] == 1)
    assert torch.all(hidden[:, 4:8] == 2)
    assert torch.all(hidden[:, 8:] == 3)


def test_llm_runtime_exposes_hf_generation_length_controls() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_llm import MiniCPMO45LLMHMONNXRuntime

    runtime = object.__new__(MiniCPMO45LLMHMONNXRuntime)
    runtime.prefill_length = 256

    assert runtime.get_input_sequence_length() == 256
    runtime.set_input_sequence_length(64)
    runtime.set_num_logits_to_keep(1)
    assert runtime.get_input_sequence_length() == 64


def test_tts_runtime_splits_long_prefill_and_masks_padded_query_rows() -> None:
    calls: list[tuple[int, int, int, torch.Tensor]] = []

    def prefill(*inputs: torch.Tensor) -> torch.Tensor:
        calls.append((int(inputs[1].item()), int(inputs[2].item()), inputs[0].shape[1], inputs[-1].clone()))
        return torch.full((1, inputs[0].shape[1], 8), float(len(calls)), dtype=torch.float16)

    def decode(*inputs: torch.Tensor) -> torch.Tensor:
        raise AssertionError("decode must not run for an empty-past multi-token input")

    output = _run_tts_decoder_graph(
        prefill,
        decode,
        torch.ones((1, 10, 8), dtype=torch.float16),
        past_seq_length=0,
        current_input_length=10,
        past_key_caches=[torch.zeros((1, 2, 16, 4), dtype=torch.float16)],
        past_value_caches=[torch.zeros((1, 2, 16, 4), dtype=torch.float16)],
        prefill_length=4,
        attention_mask=torch.zeros((1, 1, 10, 16), dtype=torch.float16),
    )

    assert [(past, current, sequence) for past, current, sequence, _ in calls] == [
        (0, 4, 4),
        (4, 4, 4),
        (8, 2, 4),
    ]
    assert output.shape == (1, 4, 8)
    assert calls[-1][3].shape == (1, 1, 4, 16)
    assert torch.all(calls[-1][3][:, :, 2:, :] == torch.finfo(torch.float16).min)


def test_tts_logits_normalizer_projects_hidden_tensor() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import create_tts_model_wraped_cls

    class Base:
        pass

    wrapped = object.__new__(create_tts_model_wraped_cls(Base))
    wrapped.config = SimpleNamespace(num_audio_tokens=5)
    wrapped.num_vq = 1
    wrapped.head_code = [object()]
    wrapped.model = SimpleNamespace(
        project_head_code=lambda hidden: torch.ones((*hidden.shape[:-1], 5), dtype=hidden.dtype)
    )

    logits = wrapped._normalize_tts_logits(torch.ones((1, 3, 8), dtype=torch.float16))

    assert logits.shape == (1, 3, 5, 1)


def test_decoder_routes_one_token_to_decode_and_orders_tts_mask_last() -> None:
    calls: list[tuple[str, tuple[torch.dtype, ...]]] = []

    def session(name: str):
        def run(*inputs: torch.Tensor) -> torch.Tensor:
            calls.append((name, tuple(value.dtype for value in inputs)))
            return inputs[0]

        return run

    caches = [torch.zeros((1, 2, 8, 4), dtype=torch.float16)]
    attention_mask = torch.ones((1, 1, 2, 2), dtype=torch.float16)
    _run_tts_decoder_graph(
        session("prefill"),
        session("decode"),
        torch.ones((1, 1, 8), dtype=torch.float16),
        past_seq_length=4,
        current_input_length=1,
        past_key_caches=caches,
        past_value_caches=caches,
        prefill_length=4,
        attention_mask=attention_mask,
    )

    assert [name for name, _ in calls] == ["decode"]


def test_tts_multi_token_continuation_uses_prefill_not_decode() -> None:
    """A multi-token condition chunk arriving with a non-empty TTS cache must still be
    prefilled; the decode graph only accepts a single token."""
    calls: list[str] = []

    def session(name: str):
        def run(*inputs: torch.Tensor) -> torch.Tensor:
            calls.append(name)
            return inputs[0]

        return run

    caches = [torch.zeros((1, 2, 16, 4), dtype=torch.float16)]
    _run_tts_decoder_graph(
        session("prefill"),
        session("decode"),
        torch.ones((1, 12, 8), dtype=torch.float16),
        past_seq_length=12,
        current_input_length=12,
        past_key_caches=caches,
        past_value_caches=caches,
        prefill_length=12,
        attention_mask=torch.zeros((1, 1, 12, 16), dtype=torch.float16),
    )

    assert calls == ["prefill"]


def test_host_attachment_omits_tts_when_host_did_not_initialize_it(monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime as runtime_module

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        runtime_module.MiniCPMOHFCompatible,
        "to_hf_compatible",
        lambda host, **kwargs: captured.update(kwargs) or host,
    )
    runtime = object.__new__(runtime_module.MiniCPMO45HMONNXRuntime)
    runtime.vision = SimpleNamespace()
    runtime.audio = SimpleNamespace()
    runtime.llm = SimpleNamespace()
    runtime.tts = SimpleNamespace()
    host = SimpleNamespace(vpm=SimpleNamespace(embeddings=SimpleNamespace(patch_size=14, num_patches_per_side=70)))

    runtime.attach_host(host)

    assert captured["tts_llama_model"] is None
    assert runtime.vision.patch_size == 14
    assert runtime.vision.num_patches_per_side == 70


def test_host_attachment_supplies_vision_wrap_config(monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime as runtime_module

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        runtime_module.MiniCPMOHFCompatible,
        "to_hf_compatible",
        lambda host, **kwargs: captured.update(kwargs) or host,
    )
    runtime = object.__new__(runtime_module.MiniCPMO45HMONNXRuntime)
    runtime.vision = SimpleNamespace()
    runtime.audio = SimpleNamespace()
    runtime.llm = SimpleNamespace()
    runtime.tts = SimpleNamespace()
    host = SimpleNamespace(vpm=SimpleNamespace(embeddings=SimpleNamespace(patch_size=14, num_patches_per_side=70)))

    runtime.attach_host(host)

    assert runtime.vision.wrap_cfg.image_slice_max_size == [40, 40]


def test_host_attachment_bypasses_native_resampler_for_post_resampler_graph(monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime as runtime_module

    host = SimpleNamespace(
        vpm=SimpleNamespace(embeddings=SimpleNamespace(patch_size=14, num_patches_per_side=70)),
        resampler=lambda value, _target_sizes: value,
    )
    runtime = object.__new__(runtime_module.MiniCPMO45HMONNXRuntime)
    runtime.vision = SimpleNamespace()
    runtime.audio = SimpleNamespace()
    runtime.llm = SimpleNamespace()
    runtime.tts = SimpleNamespace()
    monkeypatch.setattr(runtime_module.MiniCPMOHFCompatible, "to_hf_compatible", lambda model, **_kwargs: model)

    runtime.attach_host(host)

    value = torch.zeros((1, 64, 4096))
    assert runtime.host_model.resampler(value, torch.tensor([[40, 40]])) is value


def test_vision_wrapper_accepts_official_keyword_inputs() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import create_vision_wraped_cls

    class Base:
        pass

    observed: dict[str, object] = {}

    class Runtime:
        def prepare_inputs(self, data):
            observed["data"] = data
            return (
                torch.zeros((1, 3, 2, 2)),
                torch.zeros((1, 4), dtype=torch.int32),
                torch.ones((1, 4), dtype=torch.bool),
                torch.tensor([[2, 2]], dtype=torch.int32),
                [1],
            )

        def _wrap_model(self, *inputs):
            observed["inputs"] = inputs
            return torch.zeros((1, 4, 8))

    wrapped = object.__new__(create_vision_wraped_cls(Base))
    wrapped._vision_model = Runtime()
    pixel_values = [[torch.zeros((3, 2, 2))]]
    patch_attention_mask = torch.ones((1, 2, 2), dtype=torch.bool)
    tgt_sizes = torch.tensor([[2, 2]], dtype=torch.int32)

    result = wrapped.forward(
        pixel_values=pixel_values,
        patch_attention_mask=patch_attention_mask,
        tgt_sizes=tgt_sizes,
    )

    assert observed["data"] == {
        "pixel_values": pixel_values,
        "patch_attention_mask": patch_attention_mask,
        "tgt_sizes": tgt_sizes,
    }
    assert len(result) == 1


def test_vision_wrapper_uses_official_tensor_inputs_without_reprocessing() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import create_vision_wraped_cls

    class Base:
        pass

    observed: list[tuple[torch.Tensor, ...]] = []
    wrapped = object.__new__(create_vision_wraped_cls(Base))
    wrapped._vision_model = SimpleNamespace(
        prepare_inputs=lambda data: (
            pixel_values,
            torch.zeros((1, 560), dtype=torch.int32),
            torch.zeros((1, 1, 560, 560), dtype=torch.float16),
            tgt_sizes,
            [1],
        ),
        _wrap_model=lambda *inputs: observed.append(inputs) or torch.zeros((1, 4, 8)),
    )
    pixel_values = torch.zeros((1, 3, 14, 560), dtype=torch.float16)
    patch_attention_mask = torch.ones((1, 1600), dtype=torch.bool)
    tgt_sizes = torch.tensor([[40, 40]], dtype=torch.int32)

    result = wrapped.forward(
        pixel_values,
        patch_attention_mask=patch_attention_mask,
        tgt_sizes=tgt_sizes,
    )

    assert len(observed) == 1
    assert observed[0][0] is pixel_values
    assert observed[0][3] is tgt_sizes
    assert len(result) == 1


def test_vision_input_preparation_pads_hmonnx_pixel_width_to_capacity() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_vision import prepare_vision_inputs

    pixel_values = [[torch.zeros((3, 14, 1032), dtype=torch.float16)]]
    pixels, _positions, _mask, _sizes, _counts = prepare_vision_inputs(
        {"pixel_values": pixel_values, "tgt_sizes": [torch.tensor([[40, 40]])]},
        patch_size=14,
        num_patches_per_side=40,
        image_slice_max_size=(40, 40),
    )

    assert pixels.shape == (1, 3, 14, 22400)


def test_runtime_reset_state_clears_decoder_caches_and_captured_outputs() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import MiniCPMO45HMONNXRuntime

    reset_calls: list[str] = []
    runtime = object.__new__(MiniCPMO45HMONNXRuntime)
    runtime.audio = SimpleNamespace(reset_state=lambda: None)
    runtime.llm = SimpleNamespace(reset_state=lambda: reset_calls.append("llm"))
    runtime.tts = SimpleNamespace(reset_state=lambda: reset_calls.append("tts"))
    runtime.token2wav = SimpleNamespace(reset_state=lambda: None)
    runtime.host_model = SimpleNamespace(
        _xh_last_speech_tokens=torch.tensor([1, 2]),
        _xh_last_waveform=torch.tensor([0.1]),
    )

    runtime.reset_state()

    assert reset_calls == ["llm", "tts"]
    assert runtime.last_speech_token_ids is None
    assert runtime.last_waveform is None


def test_runtime_places_native_tts_generation_modules_on_hmonnx_device() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import move_native_tts_modules

    moves: list[tuple[str, str]] = []

    class Movable:
        def __init__(self, name: str) -> None:
            self.name = name

        def to(self, device: str):
            moves.append((self.name, device))
            return self

    host = SimpleNamespace(
        tts=SimpleNamespace(
            emb_text=Movable("emb_text"),
            projector_semantic=Movable("projector_semantic"),
            emb_code=Movable("emb_code"),
            head_code=Movable("head_code"),
            audio_tokenizer=SimpleNamespace(),
        )
    )

    move_native_tts_modules(host, "cuda:1")

    assert moves == [
        ("emb_text", "cuda:1"),
        ("projector_semantic", "cuda:1"),
        ("emb_code", "cuda:1"),
        ("head_code", "cuda:1"),
    ]


def test_host_attachment_preserves_native_tts_model_config(monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime as runtime_module

    native_config = SimpleNamespace(rope_theta=10000.0, hidden_size=3584, num_attention_heads=28)
    host = SimpleNamespace(
        vpm=SimpleNamespace(embeddings=SimpleNamespace(patch_size=14, num_patches_per_side=70)),
        tts=SimpleNamespace(model=SimpleNamespace(config=native_config)),
    )
    runtime = object.__new__(runtime_module.MiniCPMO45HMONNXRuntime)
    runtime.vision = SimpleNamespace()
    runtime.audio = SimpleNamespace()
    runtime.llm = SimpleNamespace()
    runtime.tts = SimpleNamespace()
    monkeypatch.setattr(runtime_module.MiniCPMOHFCompatible, "to_hf_compatible", lambda model, **_kwargs: model)

    runtime.attach_host(host)

    assert runtime.tts.config is native_config


def test_runtime_moves_native_llm_embedding_to_hmonnx_device() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import move_native_llm_modules

    moves: list[tuple[str, str]] = []
    host = SimpleNamespace(
        llm=SimpleNamespace(
            embed_tokens=SimpleNamespace(to=lambda device: moves.append(("embed_tokens", device))),
            lm_head=SimpleNamespace(to=lambda device: moves.append(("lm_head", device))),
        )
    )

    move_native_llm_modules(host, "cuda:1")

    assert moves == [("embed_tokens", "cuda:1"), ("lm_head", "cuda:1")]


def test_speech_capture_records_tokens_and_waveform_without_changing_result() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import patch_speech_generation_capture

    class FakeTTS:
        def generate(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(new_ids=torch.tensor([[[3], [4]]]))

    class FakeHost:
        def __init__(self) -> None:
            self.tts = FakeTTS()

        def _generate_speech_non_streaming(self, *args, **kwargs):
            del args, kwargs
            self.tts.generate()
            return torch.tensor([0.25, -0.25])

    host = FakeHost()
    patch_speech_generation_capture(host)
    patch_speech_generation_capture(host)

    result = host._generate_speech_non_streaming()

    assert torch.equal(result, torch.tensor([0.25, -0.25]))
    assert torch.equal(host._xh_last_speech_tokens, torch.tensor([[[3], [4]]]))
    assert torch.equal(host._xh_last_waveform, result)


def test_tts_alignment_maps_full_sequence_bounds_to_decode_hidden_states() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import extract_tts_decode_alignment

    prefill = (torch.zeros((1, 20, 4)),)
    decode = tuple((torch.full((1, 1, 4), float(index)),) for index in range(1, 5))
    outputs = SimpleNamespace(
        hidden_states=(prefill, *decode),
        full_sequences=torch.tensor([[10, 11, 12, 13, 14, 15, 16]]),
    )

    token_ids, hidden = extract_tts_decode_alignment(outputs, (3, None), 0)

    assert token_ids.tolist() == [13, 14, 15, 16]
    assert hidden.shape == (4, 4)
    assert hidden[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_tts_alignment_normalizes_generation_leading_dimensions() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import extract_tts_decode_alignment

    prefill = (torch.zeros((3, 1, 20, 4)),)
    decode = tuple((torch.full((3, 1, 1, 4), float(index)),) for index in range(1, 5))
    outputs = SimpleNamespace(
        hidden_states=(prefill, *decode),
        full_sequences=torch.tensor([[10, 11, 12, 13, 14, 15, 16]]),
    )

    token_ids, hidden = extract_tts_decode_alignment(outputs, (3, None), 0)

    assert token_ids.tolist() == [13, 14, 15, 16]
    assert hidden.shape == (4, 4)
    assert hidden[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_tts_cache_position_compat_flattens_single_batch_position() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import (
        patch_tts_cache_position_compat,
    )

    class FakeModel:
        def __init__(self) -> None:
            self.received: torch.Tensor | None = None

        def forward(self, **kwargs):
            self.received = kwargs["cache_position"]
            return kwargs["inputs_embeds"]

    model = FakeModel()
    host = SimpleNamespace(tts=SimpleNamespace(model=model))
    patch_tts_cache_position_compat(host)

    result = model.forward(
        inputs_embeds=torch.ones((1, 1, 8)),
        position_ids=torch.tensor([[31]]),
        cache_position=torch.tensor([[31]]),
    )

    assert result.shape == (1, 1, 8)
    assert model.received is not None
    assert model.received.shape == (1,)
    assert model.received.tolist() == [31]


def test_tts_sampling_defaults_fill_only_missing_config_values() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import ensure_tts_sampling_config

    config = SimpleNamespace(tts_config=SimpleNamespace(top_p=0.9))

    result = ensure_tts_sampling_config(config)

    assert result is config
    assert config.tts_config.top_p == 0.9
    assert config.tts_config.top_k == 25
    assert config.tts_config.repetition_penalty == 1.05


def test_static_cache_attention_mask_helper_builds_expected_causal_mask() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import (
        _prepare_4d_causal_attention_mask_with_cache_position,
    )

    attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.int64)
    cache_position = torch.tensor([0, 1], dtype=torch.int64)

    result = _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=2,
        target_length=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        min_dtype=torch.finfo(torch.float32).min,
        cache_position=cache_position,
        batch_size=1,
    )

    assert result.shape == (1, 1, 2, 4)
    assert result[0, 0, 0, 0] == 0
    assert result[0, 0, 0, 1] == torch.finfo(torch.float32).min
    assert result[0, 0, 1, 0] == 0
    assert result[0, 0, 1, 1] == 0
    assert torch.all(result[0, 0, :, 2:] == torch.finfo(torch.float32).min)
