from __future__ import annotations

import pytest
import torch

from xhmodel_merak.xh_llm.models.ling_3_flash._kda_rule import (
    chunk_kda,
    recurrent_kda_reference,
)
from xhquant.backend.xh2a.functions.gdr_chunk_scan import gdr_chunk_scan_xh2a_default
from xhquant.backend.xh2a.ir import generate_exp_lut_table
from xhquant.core.cache_tensor import CacheTensor
from xhquant.lib import hsum
from xhquant.nn.modules import GDRChunkScan
from xhquant.ops.xh.gdr_chunk_scan import gdr_chunk_scan_default
from xhquant.quantization.xh2a.qmodules._gdr_reference import (
    reference_gdr_chunk_scan_chain,
)


def _inputs():
    torch.manual_seed(20260811)
    batch, heads, chunks, chunk_size, key_dim, value_dim = 1, 2, 2, 4, 3, 5
    query = torch.randn(batch, heads, chunks, chunk_size, key_dim)
    key = torch.randn_like(query)
    value = torch.randn(batch, heads, chunks, chunk_size, value_dim)
    k_cumdecay = torch.randn(batch, heads, chunks, chunk_size, key_dim)
    initial_state = torch.randn(batch, heads, key_dim, value_dim)
    mask_incl = torch.tril(torch.ones(chunk_size, chunk_size))
    attrs = (chunks, heads, key_dim, value_dim, chunk_size)
    return query, key, value, k_cumdecay, mask_incl, initial_state, attrs


def _legacy_gdr(query, key, value, k_cumdecay, decay_mask, mask_incl, g, state):
    outputs = []
    for chunk_idx in range(query.shape[2]):
        q_i = query[:, :, chunk_idx]
        k_i = key[:, :, chunk_idx]
        v_i = value[:, :, chunk_idx]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, chunk_idx]) * mask_incl
        v_new = v_i - k_cumdecay[:, :, chunk_idx] @ state
        outputs.append((q_i * g[:, :, chunk_idx, :, None].exp()) @ state + attn @ v_new)
        state = (
            state * g[:, :, chunk_idx, -1, None, None].exp()
            + (
                k_i
                * (g[:, :, chunk_idx, -1, None] - g[:, :, chunk_idx]).exp()[..., None]
            ).transpose(-1, -2)
            @ v_new
        )
    return torch.stack(outputs, dim=2), state


def _kda_reference(query, key, value, k_cumdecay, pair_decay, mask_incl, g, state):
    outputs = []
    for chunk_idx in range(query.shape[2]):
        q_i = query[:, :, chunk_idx]
        k_i = key[:, :, chunk_idx]
        v_i = value[:, :, chunk_idx]
        g_i = g[:, :, chunk_idx]
        aqk = (
            q_i.unsqueeze(-2)
            * k_i.unsqueeze(-3)
            * pair_decay[:, :, chunk_idx]
        ).sum(dim=-1)
        v_new = v_i - k_cumdecay[:, :, chunk_idx] @ state
        outputs.append((q_i * g_i.exp()) @ state + (aqk * mask_incl) @ v_new)
        g_last = g_i[:, :, -1]
        state = (
            state * g_last.exp().unsqueeze(-1)
            + (k_i * (g_last.unsqueeze(-2) - g_i).exp()).transpose(-1, -2) @ v_new
        )
    return torch.stack(outputs, dim=2), state


def test_original_gdr_rank_and_order_are_bit_identical():
    query, key, value, k_cumdecay, mask_incl, initial_state, attrs = _inputs()
    g = -torch.rand(*query.shape[:-1]).cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)) * mask_incl).exp() * mask_incl

    expected = _legacy_gdr(
        query, key, value, k_cumdecay, decay_mask, mask_incl, g, initial_state
    )
    actual = gdr_chunk_scan_default(
        query,
        key,
        value,
        k_cumdecay,
        decay_mask,
        mask_incl,
        g,
        initial_state,
        *attrs,
    )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_kda_per_key_decay_matches_reference():
    query, key, value, k_cumdecay, mask_incl, initial_state, attrs = _inputs()
    g = -torch.rand_like(query).cumsum(dim=-2)
    pair_decay = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp()

    expected = _kda_reference(
        query, key, value, k_cumdecay, pair_decay, mask_incl, g, initial_state
    )
    actual = gdr_chunk_scan_default(
        query,
        key,
        value,
        k_cumdecay,
        pair_decay,
        mask_incl,
        g,
        initial_state,
        *attrs,
    )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_module_publishes_kda_state_to_cache_tensor():
    query, key, value, k_cumdecay, mask_incl, initial_state, attrs = _inputs()
    g = -torch.rand_like(query).cumsum(dim=-2)
    pair_decay = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp()
    expected_output, expected_state = gdr_chunk_scan_default(
        query,
        key,
        value,
        k_cumdecay,
        pair_decay,
        mask_incl,
        g,
        initial_state,
        *attrs,
    )
    cache = CacheTensor(initial_state.clone())

    output = GDRChunkScan(*attrs)(
        query,
        key,
        value,
        k_cumdecay,
        pair_decay,
        mask_incl,
        g,
        cache,
    )

    assert torch.equal(output, expected_output)
    assert torch.equal(cache, expected_state)


def test_kda_trace_keeps_one_shared_gdr_chunk_scan_node():
    query, key, value, k_cumdecay, mask_incl, initial_state, attrs = _inputs()
    g = -torch.rand_like(query).cumsum(dim=-2)
    pair_decay = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp()

    traced = torch.jit.trace(
        GDRChunkScan(*attrs),
        (
            query,
            key,
            value,
            k_cumdecay,
            pair_decay,
            mask_incl,
            g,
            initial_state,
        ),
        check_trace=False,
    )

    assert "xh::GDRChunkScan" in str(traced.graph)
    assert traced(
        query,
        key,
        value,
        k_cumdecay,
        pair_decay,
        mask_incl,
        g,
        initial_state,
    ).shape == value.shape


@pytest.mark.skipif(
    getattr(hsum, "_module", None) is None,
    reason="requires the hsum extension for aligned XH2a LUT arithmetic",
)
def test_kda_xh2a_chain_matches_quant_reference():
    query, key, value, k_cumdecay, mask_incl, initial_state, attrs = _inputs()
    query, key, value, k_cumdecay, mask_incl, initial_state = (
        tensor.to(torch.float16)
        for tensor in (query, key, value, k_cumdecay, mask_incl, initial_state)
    )
    g = -torch.rand_like(query).cumsum(dim=-2)
    pair_decay = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp()
    lut = tuple(
        torch.as_tensor(part, dtype=torch.float32)
        for part in generate_exp_lut_table()
    )
    expected_output, expected_state = reference_gdr_chunk_scan_chain(
        query,
        key,
        value,
        k_cumdecay,
        pair_decay,
        mask_incl,
        g,
        initial_state,
        *attrs,
        mode="aligned",
        exp_lut_cut_points=lut[0],
        exp_lut_table=lut[1],
        exp_lut_scale=lut[2],
    )
    cache = CacheTensor(initial_state.clone())

    output = gdr_chunk_scan_xh2a_default(
        query,
        key,
        value,
        k_cumdecay,
        pair_decay,
        mask_incl,
        g,
        cache,
        *lut,
        *attrs,
    )

    assert torch.equal(output, expected_output)
    assert torch.equal(cache, expected_state)


def test_chunk_kda_keeps_original_fp16_cache_and_masks_noncausal_exp():
    torch.manual_seed(17)
    batch, sequence, heads, key_dim, value_dim, chunk_size = 1, 8, 2, 4, 5, 8
    query = torch.randn(batch, sequence, heads, key_dim, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn(batch, sequence, heads, value_dim, dtype=torch.float16)
    # The unmasked upper triangle reaches exp(35) and overflows fp16.  The
    # causal log-domain mask must prevent that intermediate from becoming inf.
    log_decay = torch.full_like(query, -5.0)
    beta = torch.rand(batch, sequence, heads, dtype=torch.float16)
    cache = CacheTensor(
        torch.zeros(batch, heads, key_dim, value_dim, dtype=torch.float16)
    )
    scan = GDRChunkScan(1, heads, key_dim, value_dim, chunk_size)

    output, returned_state = chunk_kda(
        query,
        key,
        value,
        log_decay,
        beta,
        cache,
        chunk_size=chunk_size,
        block_tri_inverse_op=None,
        chunk_scan_op=scan,
        eye_block_batched=torch.eye(chunk_size, dtype=torch.float16).expand(
            batch * heads, chunk_size, chunk_size
        ),
    )

    assert returned_state is cache
    assert torch.isfinite(output).all()
    assert torch.isfinite(cache).all()
    assert torch.count_nonzero(cache) > 0


def test_chunk_kda_shared_scan_matches_token_recurrence_with_distinct_k_v_dims():
    torch.manual_seed(29)
    batch, sequence, heads, key_dim, value_dim, chunk_size = 1, 8, 2, 3, 5, 4
    query = torch.randn(batch, sequence, heads, key_dim)
    key = torch.randn_like(query)
    value = torch.randn(batch, sequence, heads, value_dim)
    log_decay = -torch.rand_like(query) * 0.5
    beta = torch.rand(batch, sequence, heads)
    initial_state = torch.randn(batch, heads, key_dim, value_dim) * 0.1
    scale = key_dim**-0.5

    expected_output, expected_state = recurrent_kda_reference(
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state,
        scale=scale,
    )
    cache = CacheTensor(initial_state.clone())
    actual_output, returned_state = chunk_kda(
        query,
        key,
        value,
        log_decay,
        beta,
        cache,
        chunk_size=chunk_size,
        block_tri_inverse_op=None,
        chunk_scan_op=GDRChunkScan(
            sequence // chunk_size,
            heads,
            key_dim,
            value_dim,
            chunk_size,
        ),
        eye_block_batched=torch.empty(0),
        scale=scale,
    )

    assert returned_state is cache
    torch.testing.assert_close(actual_output, expected_output, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(cache, expected_state, atol=2e-6, rtol=2e-6)
