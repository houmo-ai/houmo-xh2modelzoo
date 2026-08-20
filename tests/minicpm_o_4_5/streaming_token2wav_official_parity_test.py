from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


MODEL_DIR = Path(os.environ["MINICPM_O45_MODEL_DIR"]) if os.environ.get("MINICPM_O45_MODEL_DIR") else None


@pytest.fixture(scope="module")
def official_flow():
    if MODEL_DIR is None:
        pytest.skip("set MINICPM_O45_MODEL_DIR to run Token2Wav official parity tests")
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_token2wav import load_token2wav_modules

    flow, _ = load_token2wav_modules(str(MODEL_DIR))
    return flow.eval()


def _conformer_cache(valid_length: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17 + valid_length)
    base_half = torch.randn((6, 1, 8, valid_length // 2, 128), generator=generator, dtype=dtype) * 0.01
    up = torch.randn((4, 1, 8, valid_length, 128), generator=generator, dtype=dtype) * 0.01
    return torch.cat((base_half.repeat(1, 1, 1, 2, 1), up), dim=0)


@pytest.mark.parametrize(
    "past_valid,last_chunk,token_valid_length",
    [(160, False, 28), (160, True, 7), (892, False, 28), (892, True, 7)],
)
def test_fixed_conformer_frontend_matches_official_dynamic_cache(
    official_flow,
    past_valid: int,
    last_chunk: bool,
    token_valid_length: int,
) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.minicpmo_token2wav_modules import (
        FlowStreamingFrontendWrapper,
    )
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        compact_conformer_attention_cache,
        pack_conformer_attention_cache,
    )

    torch.manual_seed(23)
    capacity = 942
    tokens = torch.randint(0, 6000, (1, token_valid_length), dtype=torch.int64)
    padded_tokens = F.pad(tokens, (0, 28 - token_valid_length), value=4218)
    embedding = torch.randn(1, 192)
    cnn_cache = torch.randn(1, 512, 6) * 0.01
    att_cache = _conformer_cache(past_valid)

    with torch.inference_mode():
        official_spks = official_flow.spk_embed_affine_layer(F.normalize(embedding, dim=1))
        official_hidden = official_flow.input_embedding(tokens)
        official_hidden, official_cnn, official_att = official_flow.encoder.forward_chunk(
            xs=official_hidden,
            last_chunk=last_chunk,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )
        official_mu = official_flow.encoder_proj(official_hidden).transpose(1, 2).contiguous()

        fixed = FlowStreamingFrontendWrapper(official_flow, last_chunk=last_chunk)(
            padded_tokens,
            torch.tensor([token_valid_length], dtype=torch.int32),
            embedding,
            cnn_cache,
            pack_conformer_attention_cache(
                att_cache,
                valid_length=past_valid,
                capacity=capacity,
                base_layer_count=6,
            ),
            torch.tensor([past_valid], dtype=torch.int32),
        )

    current_valid = token_valid_length * 2 if last_chunk else (token_valid_length - 3) * 2
    fixed_att = compact_conformer_attention_cache(
        fixed[3],
        past_valid_length=past_valid,
        current_valid_length=current_valid,
        input_capacity=capacity,
        base_layer_count=6,
    )

    assert int(fixed[4].item()) == past_valid + current_valid
    assert torch.allclose(fixed[0][:, :, :current_valid], official_mu, atol=2e-5, rtol=2e-5)
    assert torch.allclose(fixed[1], official_spks, atol=1e-6, rtol=1e-6)
    if not last_chunk:
        assert torch.allclose(fixed[2], official_cnn, atol=2e-5, rtol=2e-5)
    assert torch.allclose(fixed_att, official_att, atol=3e-5, rtol=3e-5)


def test_fixed_conformer_frontend_matches_official_across_cache_truncation(official_flow) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.minicpmo_token2wav_modules import (
        FlowStreamingFrontendWrapper,
    )
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        _truncate_streaming_cache,
        compact_conformer_attention_cache,
        pack_conformer_attention_cache,
    )

    torch.manual_seed(41)
    prompt_length = 842
    capacity = 942
    current_valid = 50
    embedding = torch.randn(1, 192)
    official_cnn = torch.randn(1, 512, 6) * 0.01
    fixed_cnn = official_cnn.clone()
    official_att = _conformer_cache(prompt_length)
    fixed_att = official_att.clone()
    valid_length = prompt_length
    observed_lengths: list[int] = []

    for _ in range(4):
        tokens = torch.randint(0, 6000, (1, 28), dtype=torch.int64)
        with torch.inference_mode():
            official_hidden = official_flow.input_embedding(tokens)
            official_hidden, official_present_cnn, official_present_att = official_flow.encoder.forward_chunk(
                xs=official_hidden,
                last_chunk=False,
                cnn_cache=official_cnn,
                att_cache=official_att,
            )
            official_mu = official_flow.encoder_proj(official_hidden).transpose(1, 2).contiguous()
            fixed = FlowStreamingFrontendWrapper(official_flow, last_chunk=False)(
                tokens,
                torch.tensor([tokens.shape[1]], dtype=torch.int32),
                embedding,
                fixed_cnn,
                pack_conformer_attention_cache(
                    fixed_att,
                    valid_length=valid_length,
                    capacity=capacity,
                    base_layer_count=6,
                ),
                torch.tensor([valid_length], dtype=torch.int32),
            )

        assert int(fixed[4].item()) == valid_length + current_valid
        fixed_present_att = compact_conformer_attention_cache(
            fixed[3],
            past_valid_length=valid_length,
            current_valid_length=current_valid,
            input_capacity=capacity,
            base_layer_count=6,
        )
        assert torch.allclose(fixed[0][:, :, :current_valid], official_mu, atol=2e-5, rtol=2e-5)
        assert torch.allclose(fixed[2], official_present_cnn, atol=2e-5, rtol=2e-5)
        assert torch.allclose(fixed_present_att, official_present_att, atol=3e-5, rtol=3e-5)

        official_att = _truncate_streaming_cache(
            official_present_att,
            prompt_length=prompt_length,
            tail=100,
            capacity=capacity,
        )
        fixed_att = _truncate_streaming_cache(
            fixed_present_att,
            prompt_length=prompt_length,
            tail=100,
            capacity=capacity,
        )
        official_cnn = official_present_cnn
        fixed_cnn = fixed[2]
        valid_length = int(official_att.shape[3])
        observed_lengths.append(valid_length)
        assert torch.allclose(fixed_att, official_att, atol=3e-5, rtol=3e-5)

    assert observed_lengths == [892, 942, 942, 942]


@pytest.mark.parametrize(
    "past_valid,current_valid",
    [(160, 50), (160, 14), (892, 50), (892, 14), (942, 50), (942, 14)],
)
def test_fixed_estimator_step_matches_official_dynamic_cache(
    official_flow,
    past_valid: int,
    current_valid: int,
) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.minicpmo_token2wav_modules import (
        FlowEstimatorStepWrapper,
    )
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        compact_estimator_attention_cache,
        pack_streaming_attention_cache,
    )

    torch.manual_seed(31 + current_valid)
    estimator = official_flow.decoder.estimator
    cache_capacity = 942
    frame_capacity = 56
    x = torch.randn(2, 80, current_valid) * 0.01
    mu = torch.randn(2, 80, current_valid) * 0.01
    t = torch.tensor([0.2, 0.2])
    spks = torch.randn(2, 80) * 0.01
    cond = torch.zeros_like(mu)
    cnn_cache = torch.randn(16, 2, 1024, 2) * 0.01
    att_cache = torch.randn(16, 2, 8, past_valid, 128) * 0.01

    with torch.inference_mode():
        official = estimator.forward_chunk(x, mu, t, spks, cond, cnn_cache, att_cache)
        fixed = FlowEstimatorStepWrapper(estimator)(
            F.pad(x, (0, frame_capacity - current_valid)),
            F.pad(mu, (0, frame_capacity - current_valid)),
            t,
            spks,
            F.pad(cond, (0, frame_capacity - current_valid)),
            cnn_cache,
            pack_streaming_attention_cache(
                att_cache,
                valid_length=past_valid,
                capacity=cache_capacity,
                axis=3,
            ),
            torch.tensor([past_valid], dtype=torch.int32),
            torch.tensor([current_valid], dtype=torch.int32),
        )

    fixed_att = compact_estimator_attention_cache(
        fixed[2],
        past_valid_length=past_valid,
        current_valid_length=current_valid,
        input_capacity=cache_capacity,
        frame_capacity=frame_capacity,
    )
    assert torch.allclose(fixed[0][:, :, :current_valid], official[0], atol=3e-5, rtol=3e-5)
    assert torch.allclose(fixed[1], official[1], atol=3e-5, rtol=3e-5)
    assert torch.allclose(fixed_att, official[2], atol=3e-5, rtol=3e-5)
