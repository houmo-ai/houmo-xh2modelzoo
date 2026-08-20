from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from onnx import TensorProto, helper, numpy_helper


def test_flow_frontend_inputs_pad_tokens_and_prompt_mel_to_exported_shapes() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        prepare_flow_frontend_inputs,
    )

    inputs = prepare_flow_frontend_inputs(
        token=torch.tensor([[4, 5, 6]], dtype=torch.int32),
        prompt_token=torch.tensor([[1, 2]], dtype=torch.int32),
        prompt_feat=torch.ones((1, 4, 80), dtype=torch.float32),
        embedding=torch.ones((1, 192), dtype=torch.float32),
        token_capacity=8,
        mel_capacity=16,
        up_rate=2,
    )

    assert inputs.tokens.shape == (1, 8)
    assert inputs.tokens[0, :5].tolist() == [1, 2, 4, 5, 6]
    assert inputs.token_length.tolist() == [5]
    assert inputs.prompt_feat.shape == (1, 16, 80)
    assert inputs.prompt_feat_length.tolist() == [4]
    assert inputs.output_mel_length == 6


def test_flow_frontend_inputs_reject_negative_token_ids_before_hmonnx() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        prepare_flow_frontend_inputs,
    )

    with pytest.raises(RuntimeError, match="non-negative"):
        prepare_flow_frontend_inputs(
            token=torch.tensor([[-1]], dtype=torch.int32),
            prompt_token=torch.tensor([[1]], dtype=torch.int32),
            prompt_feat=torch.ones((1, 2, 80), dtype=torch.float32),
            embedding=torch.ones((1, 192), dtype=torch.float32),
            token_capacity=4,
            mel_capacity=8,
            up_rate=2,
        )


def test_flow_frontend_inputs_cast_float_boundaries_to_float16() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        prepare_flow_frontend_inputs,
    )

    inputs = prepare_flow_frontend_inputs(
        token=torch.tensor([[2]], dtype=torch.int32),
        prompt_token=torch.tensor([[1]], dtype=torch.int32),
        prompt_feat=torch.ones((1, 2, 80), dtype=torch.float32),
        embedding=torch.ones((1, 192), dtype=torch.float32),
        token_capacity=4,
        mel_capacity=8,
        up_rate=2,
    )

    assert inputs.prompt_feat.dtype == torch.float16
    assert inputs.embedding.dtype == torch.float16


def test_cfm_scheduler_calls_decoder_ten_times_and_applies_cfg() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import run_cfm

    calls: list[tuple[torch.Size, torch.Size]] = []

    def decoder(
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        del mu, t, spks, cond
        calls.append((x.shape, mask.shape))
        conditional = torch.ones_like(x[:1])
        unconditional = torch.zeros_like(x[:1])
        return torch.cat((conditional, unconditional), dim=0)

    output = run_cfm(
        decoder,
        mu=torch.zeros((1, 80, 4)),
        mask=torch.ones((1, 1, 4)),
        spks=torch.zeros((1, 80)),
        cond=torch.zeros((1, 80, 4)),
        noise=torch.zeros((1, 80, 4)),
        n_timesteps=10,
        cfg_rate=0.7,
    )

    assert len(calls) == 10
    assert calls == [(torch.Size([2, 80, 4]), torch.Size([2, 1, 4]))] * 10
    assert torch.allclose(output, torch.full_like(output, 1.7), atol=1e-5)


def test_hift_input_padding_and_waveform_crop_use_real_mel_length() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        crop_hift_waveform,
        prepare_hift_input,
    )

    mel = torch.ones((1, 80, 6), dtype=torch.float32)
    padded = prepare_hift_input(mel, frame_capacity=10)
    waveform = torch.arange(6000, dtype=torch.float32).unsqueeze(0)

    assert padded.shape == (1, 80, 10)
    assert torch.equal(padded[:, :, :6], mel)
    assert torch.count_nonzero(padded[:, :, 6:]) == 0
    assert crop_hift_waveform(waveform, mel_frames=6).shape == (1, 2880)


def test_deterministic_hift_source_matches_official_sinegen2_with_supplied_randomness(monkeypatch) -> None:
    pytest.importorskip("stepaudio2.flashcosyvoice.modules.hifigan_components.layers")
    from stepaudio2.flashcosyvoice.modules.hifigan_components.layers import SineGen2

    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.minicpmo_token2wav_modules import (
        deterministic_hift_source,
    )

    f0 = torch.linspace(80.0, 160.0, 960).reshape(1, 960, 1)
    phase_noise = torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float32)
    gaussian_noise = torch.linspace(-1.0, 1.0, 960 * 3).reshape(1, 960, 3)
    generator = SineGen2(
        samp_rate=24000,
        upsample_scale=480,
        harmonic_num=2,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=10,
    )
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: phase_noise.clone())
    monkeypatch.setattr(torch, "randn_like", lambda value: gaussian_noise.to(value))

    expected, expected_uv, _ = generator(f0.clone())
    actual, actual_uv = deterministic_hift_source(
        f0,
        phase_noise,
        gaussian_noise,
        sampling_rate=24000,
        upsample_scale=480,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=10,
    )

    assert torch.allclose(actual, expected)
    assert torch.equal(actual_uv, expected_uv)


def test_deterministic_hift_source_onnx_avoids_scatternd(tmp_path) -> None:
    import onnx

    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.minicpmo_token2wav_modules import (
        deterministic_hift_source,
    )

    class SourceWrapper(torch.nn.Module):
        def forward(
            self,
            f0: torch.Tensor,
            phase_noise: torch.Tensor,
            gaussian_noise: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return deterministic_hift_source(
                f0,
                phase_noise,
                gaussian_noise,
                sampling_rate=24000,
                upsample_scale=4,
                sine_amp=0.1,
                noise_std=0.003,
                voiced_threshold=10,
            )

    output = tmp_path / "deterministic_hift_source.onnx"
    torch.onnx.export(
        SourceWrapper(),
        (
            torch.ones(1, 16, 1),
            torch.zeros(1, 3),
            torch.zeros(1, 16, 3),
        ),
        output,
        opset_version=17,
    )

    model = onnx.load(output)
    assert all(node.op_type != "ScatterND" for node in model.graph.node)


def test_hift_random_inputs_match_exported_source_contract() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_token2wav import (
        create_hift_random_inputs,
    )

    phase_noise, source_noise = create_hift_random_inputs(
        harmonic_count=9,
        source_length=491040,
        seed=1024,
    )

    assert phase_noise.shape == (1, 9)
    assert phase_noise[0, 0].item() == 0.0
    assert source_noise.shape == (1, 491040, 9)
    assert torch.equal(
        source_noise,
        create_hift_random_inputs(harmonic_count=9, source_length=491040, seed=1024)[1],
    )


def test_hift_onnx_rejects_graph_level_random_operators() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_token2wav import (
        require_deterministic_onnx,
    )

    input_value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    output_value = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    random_node = helper.make_node("RandomNormalLike", ["input"], ["output"])
    model = helper.make_model(helper.make_graph([random_node], "random", [input_value], [output_value]))

    with pytest.raises(RuntimeError, match="RandomNormalLike"):
        require_deterministic_onnx(model)


def test_hift_runtime_supplies_persisted_source_inputs() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    runtime = object.__new__(runtime_token2wav.MiniCPMO45Token2WavHMONNXRuntime)
    runtime.report = runtime_token2wav.Token2WavExecutionReport()
    runtime.hift_frame_capacity = 10
    runtime.hift_phase_noise = torch.arange(9, dtype=torch.float16).reshape(1, 9)
    runtime.hift_source_noise = torch.arange(10 * 480 * 9, dtype=torch.float16).reshape(1, 10 * 480, 9)
    observed: list[tuple[torch.Tensor, ...]] = []
    runtime.hift_session = lambda *values: observed.append(values) or torch.ones((1, 4800))

    waveform = runtime.hift_inference(torch.ones((1, 80, 6)))

    assert len(observed[0]) == 3
    assert observed[0][0].shape == (1, 80, 10)
    assert torch.equal(observed[0][1], runtime.hift_phase_noise)
    assert torch.equal(observed[0][2], runtime.hift_source_noise)
    assert waveform.shape == (1, 2880)


def test_hift_onnx_preparation_fixes_source_resize_to_frame_capacity() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_token2wav import prepare_hift_onnx

    input_value = helper.make_tensor_value_info("speech_feat", TensorProto.FLOAT, [1, 80, 1024])
    output_value = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 9, 1024])
    scales = numpy_helper.from_array(torch.tensor([1.0, 1.0, 480.0]).numpy(), name="scales")
    resize = helper.make_node(
        "Resize",
        ["speech_feat", "", "scales"],
        ["output"],
        name="/m_source/l_sin_gen/Resize",
    )
    model = helper.make_model(helper.make_graph([resize], "hift", [input_value], [output_value], [scales]))

    prepared = prepare_hift_onnx(model, frame_capacity=1024)
    prepared_resize = prepared.graph.node[0]
    initializers = {value.name: numpy_helper.to_array(value) for value in prepared.graph.initializer}

    assert prepared_resize.input[2] == ""
    assert prepared_resize.input[3] == "token2wav_hift_source_sizes"
    assert initializers["token2wav_hift_source_sizes"].tolist() == [1, 9, 1024]


def test_token2wav_adapter_replaces_native_flow_and_hift_without_fallback() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        install_hmonnx_token2wav,
    )

    native_flow = SimpleNamespace(inference=lambda *args: (_ for _ in ()).throw(AssertionError("native flow ran")))

    def native_hift(*args, **kwargs):
        del args, kwargs
        raise AssertionError("native hift ran")

    tokenizer = SimpleNamespace(flow=native_flow, hift=native_hift)
    runtime = SimpleNamespace(
        up_rate=2,
        flow_inference=lambda *args: torch.ones((1, 80, 3)),
        hift_inference=lambda mel: torch.ones((1, 1440)),
    )

    install_hmonnx_token2wav(tokenizer, runtime)
    mel = tokenizer.flow.inference(
        torch.tensor([[3]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([[1, 2]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.ones((1, 4, 80)),
        torch.tensor([4], dtype=torch.int32),
        torch.ones((1, 192)),
        10,
    )
    # Non-streaming entry (no cache source) routes to the offline HiFT main graph,
    # never to native hift; returns (waveform, None) matching official unpacking.
    waveform, source = tokenizer.hift(speech_feat=mel)
    assert waveform.shape == (1, 1440)
    assert source is None

    assert mel.shape == (1, 80, 3)
    assert tokenizer.flow is not native_flow
    assert tokenizer.hift is not native_hift
    assert tokenizer.flow.up_rate == 2


def test_flow_adapter_disables_outer_cuda_autocast() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise CUDA autocast")

    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        install_hmonnx_token2wav,
    )

    observed: list[bool] = []
    runtime = SimpleNamespace(
        up_rate=2,
        flow_inference=lambda *args: observed.append(torch.is_autocast_enabled("cuda")) or torch.ones((1, 80, 3)),
        hift_inference=lambda mel: torch.ones((1, 1440)),
    )
    tokenizer = SimpleNamespace(flow=SimpleNamespace(), hift=lambda *args: None)
    install_hmonnx_token2wav(tokenizer, runtime)

    with torch.autocast("cuda", dtype=torch.float32):
        tokenizer.flow.inference(torch.tensor([[1]], device="cuda"))

    assert observed == [False]


def test_hift_adapter_returns_float32_waveform_for_wav_encoding() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        install_hmonnx_token2wav,
    )

    runtime = SimpleNamespace(
        up_rate=2,
        flow_inference=lambda *args: torch.ones((1, 80, 3)),
        hift_inference=lambda mel: torch.ones((1, 1440), dtype=torch.float16),
    )
    tokenizer = SimpleNamespace(flow=SimpleNamespace(), hift=lambda *args: None)
    install_hmonnx_token2wav(tokenizer, runtime)

    # Non-streaming HiFT must return float32 so torchaudio can encode WAV
    # (official path does torchaudio.save(wav, 24000, format="wav")).
    waveform, _ = tokenizer.hift(torch.ones((1, 80, 3)))
    assert waveform.dtype == torch.float32


def test_backend_report_requires_all_token2wav_hmonnx_components_to_execute() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        Token2WavExecutionReport,
    )

    report = Token2WavExecutionReport()
    report.record_flow_frontend()
    report.record_flow_decoder()
    report.record_hift()

    assert report.backends == {
        "token2wav_flow_frontend": "hmonnx",
        "token2wav_flow_decoder": "hmonnx",
        "token2wav_hift": "hmonnx",
    }
    assert report.execution_counts == {
        "token2wav_flow_frontend": 1,
        "token2wav_flow_decoder": 1,
        "token2wav_hift": 1,
    }
    report.require_full_hmonnx_execution()


def test_token2wav_runtime_casts_frontend_outputs_before_cfm() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    runtime = object.__new__(runtime_token2wav.MiniCPMO45Token2WavHMONNXRuntime)
    runtime.token_capacity = 8
    runtime.mel_capacity = 16
    runtime.up_rate = 2
    runtime.n_timesteps = 10
    runtime.cfg_rate = 0.7
    runtime.report = runtime_token2wav.Token2WavExecutionReport()
    runtime.decoder_session = lambda *args: args[0]
    runtime.rand_noise = torch.zeros((1, 80, 16), dtype=torch.float16)
    runtime.frontend_session = lambda *args: (
        torch.zeros((1, 80, 16), dtype=torch.float32),
        torch.ones((1, 1, 16), dtype=torch.float32),
        torch.zeros((1, 80), dtype=torch.float32),
        torch.zeros((1, 80, 16), dtype=torch.float32),
    )
    observed: dict[str, torch.dtype] = {}

    def fake_run_cfm(decoder, **kwargs):
        del decoder
        observed.update({name: kwargs[name].dtype for name in ("mu", "mask", "spks", "cond", "noise")})
        return kwargs["noise"]

    original = runtime_token2wav.run_cfm
    runtime_token2wav.run_cfm = fake_run_cfm
    try:
        runtime.flow_inference(
            torch.tensor([[3]], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([[1, 2]], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            torch.ones((1, 4, 80)),
            torch.tensor([4], dtype=torch.int32),
            torch.ones((1, 192)),
            10,
        )
    finally:
        runtime_token2wav.run_cfm = original

    assert observed == {name: torch.float16 for name in ("mu", "mask", "spks", "cond", "noise")}


def test_token2wav_runtime_uses_persistent_cfm_noise_slice() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    runtime = object.__new__(runtime_token2wav.MiniCPMO45Token2WavHMONNXRuntime)
    runtime.token_capacity = 8
    runtime.mel_capacity = 16
    runtime.up_rate = 2
    runtime.n_timesteps = 10
    runtime.cfg_rate = 0.7
    runtime.report = runtime_token2wav.Token2WavExecutionReport()
    runtime.decoder_session = lambda *args: args[0]
    runtime.rand_noise = torch.arange(16 * 80, dtype=torch.float16).reshape(1, 80, 16)
    runtime.frontend_session = lambda *args: (
        torch.zeros((1, 80, 16), dtype=torch.float16),
        torch.ones((1, 1, 16), dtype=torch.float16),
        torch.zeros((1, 80), dtype=torch.float16),
        torch.zeros((1, 80, 16), dtype=torch.float16),
    )
    observed: dict[str, torch.Tensor] = {}

    def fake_run_cfm(decoder, **kwargs):
        del decoder
        observed["noise"] = kwargs["noise"].clone()
        return kwargs["noise"]

    original = runtime_token2wav.run_cfm
    runtime_token2wav.run_cfm = fake_run_cfm
    try:
        runtime.flow_inference(
            torch.tensor([[3]], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([[1, 2]], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            torch.ones((1, 4, 80)),
            torch.tensor([4], dtype=torch.int32),
            torch.ones((1, 192)),
            10,
        )
    finally:
        runtime_token2wav.run_cfm = original

    assert torch.equal(observed["noise"], runtime.rand_noise)
