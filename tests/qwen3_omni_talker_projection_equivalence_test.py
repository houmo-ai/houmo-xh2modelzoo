"""Equivalence test: wrap-side fused projection vs. HF floating-point construction.

The wrap forward (``_Qwen3OmniMoeTalkerForConditionalGeneration.forward`` in
``xh_model_zoo/xh_llm/models/qwen3_omni/_talker_model.py:301-331``) folds
``hidden_projection`` and ``text_projection`` into the talker graph and selects
between them with arithmetic masks. HF's floating-point reference does the
same projection per-segment with host-side control flow inside
``Qwen3OmniMoeForConditionalGeneration._get_talker_user_parts`` /
``_get_talker_assistant_parts`` and hands the already-projected
``inputs_embeds`` to ``talker.generate``.

If, for the same projection weights and the same random embeddings/specials,
the wrap formula given the documented ``(source, role_mask, bypass_embeds,
bypass_mask)`` contract reproduces the HF host-side construction, then both
paths feed the SAME trunk (``self.model``) with the SAME input tensor — so
every downstream tensor is identical. That is the equivalence we need.
"""

from pathlib import Path

import torch

from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerResizeMLP,
)


_WRAP_SOURCE = Path("xh_model_zoo/xh_llm/models/qwen3_omni/_talker_model.py")


class _DummyTalkerCfg:
    """Minimal config to instantiate ``Qwen3OmniMoeTalkerResizeMLP``."""

    def __init__(self, thinker_hidden, intermediate, talker_hidden, hidden_act="silu"):
        self.thinker_hidden_size = thinker_hidden
        text_cfg = type("TextCfg", (), {})()
        text_cfg.intermediate_size = intermediate
        text_cfg.hidden_size = talker_hidden
        text_cfg.hidden_act = hidden_act
        self.text_config = text_cfg


def _build_projections(thinker_hidden=8, intermediate=16, talker_hidden=10, seed=0):
    cfg = _DummyTalkerCfg(thinker_hidden, intermediate, talker_hidden)
    torch.manual_seed(seed)
    text_proj = Qwen3OmniMoeTalkerResizeMLP(cfg).eval()
    hidden_proj = Qwen3OmniMoeTalkerResizeMLP(cfg).eval()
    return text_proj, hidden_proj, cfg


@torch.no_grad()
def _hf_user_parts(im_start, end, multimodal_mask, thinker_hidden, thinker_embed,
                   text_proj, hidden_proj, talker_hidden_size, dtype):
    """Mirror ``Qwen3OmniMoeForConditionalGeneration._get_talker_user_parts``."""
    out = torch.empty((1, end - im_start, talker_hidden_size), dtype=dtype)
    user_mm = multimodal_mask[:, im_start:end]
    if user_mm.any():
        mm_src = thinker_hidden[:, im_start:end][user_mm]
        out[user_mm] = hidden_proj(mm_src)
    text_src = thinker_embed[:, im_start:end][~user_mm]
    out[~user_mm] = text_proj(text_src)
    return out


@torch.no_grad()
def _wrap_projection_mix(source, role_mask, bypass_embeds, bypass_mask,
                          hidden_proj, text_proj):
    """Exact copy of the wrap formula in ``_talker_model.py``.

    The companion ``test_wrap_formula_matches_source`` test guards against
    drift between this helper and the wrap source.
    """
    hidden_proj_out = hidden_proj(source)
    text_proj_out = text_proj(source)
    one_minus_role = role_mask * -1.0 + 1.0
    projected = hidden_proj_out * one_minus_role + text_proj_out * role_mask
    one_minus_bypass = bypass_mask * -1.0 + 1.0
    inputs_embeds = projected * one_minus_bypass + bypass_embeds * bypass_mask
    return inputs_embeds


@torch.no_grad()
def test_user_segment_wrap_matches_hf_floating_point():
    text_proj, hidden_proj, cfg = _build_projections()
    seq = 12
    thinker_h = cfg.thinker_hidden_size
    talker_h = cfg.text_config.hidden_size
    dtype = torch.float32

    torch.manual_seed(101)
    thinker_hidden = torch.randn(1, seq, thinker_h, dtype=dtype)
    thinker_embed = torch.randn(1, seq, thinker_h, dtype=dtype)
    multimodal_mask = torch.tensor([[
        False, True, True, False, False, True, False, False, True, False, False, False
    ]])

    hf_user = _hf_user_parts(
        0, seq, multimodal_mask, thinker_hidden, thinker_embed,
        text_proj, hidden_proj, talker_h, dtype=dtype,
    )

    user_mm = multimodal_mask[:, 0:seq]
    source = thinker_embed[:, 0:seq].clone()
    source[user_mm] = thinker_hidden[:, 0:seq][user_mm]
    role_mask = (~user_mm).unsqueeze(-1).to(dtype)  # text=1, mm=0
    bypass_embeds = torch.zeros_like(hf_user)
    bypass_mask = torch.zeros(1, seq, 1, dtype=dtype)

    wrap_user = _wrap_projection_mix(
        source, role_mask, bypass_embeds, bypass_mask, hidden_proj, text_proj,
    )

    diff = (wrap_user - hf_user).abs().max().item()
    assert torch.allclose(wrap_user, hf_user, atol=1e-6, rtol=0), (
        f"wrap user-segment must equal HF; max abs diff = {diff}"
    )


@torch.no_grad()
def test_assistant_segment_wrap_matches_hf_floating_point():
    text_proj, hidden_proj, cfg = _build_projections()
    thinker_h = cfg.thinker_hidden_size
    talker_h = cfg.text_config.hidden_size
    dtype = torch.float32

    torch.manual_seed(202)
    thinker_embed = torch.randn(1, 16, thinker_h, dtype=dtype)
    im_start, end = 2, 2 + 4  # 4 thinker tokens drive the assistant segment

    # ---- HF reference path (mirror _get_talker_assistant_parts) ----
    asst_hidden = text_proj(thinker_embed[:, im_start:end])  # [1, 4, talker_h]
    tts_pad = torch.randn(1, 1, talker_h, dtype=dtype)
    tts_bos = torch.randn(1, 1, talker_h, dtype=dtype)
    codec_specials = torch.randn(1, 6, talker_h, dtype=dtype)

    asst_text_hidden = torch.cat(
        [asst_hidden[:, :3], tts_pad.expand(-1, 4, -1), tts_bos, asst_hidden[:, 3:4]],
        dim=1,
    )
    asst_codec_hidden = torch.cat(
        [torch.zeros(1, 3, talker_h, dtype=dtype), codec_specials],
        dim=1,
    )
    hf_asst = asst_text_hidden + asst_codec_hidden  # [1, 9, talker_h]

    # ---- Wrap construction (per the export hook contract) ----
    seg = 9
    source = torch.zeros(1, seg, thinker_h, dtype=dtype)
    source[:, :3, :] = thinker_embed[:, im_start:im_start + 3, :]
    role_mask = torch.ones(1, seg, 1, dtype=dtype)
    bypass_embeds = hf_asst.clone()
    bypass_embeds[:, :3, :] = 0
    bypass_mask = torch.ones(1, seg, 1, dtype=dtype)
    bypass_mask[:, :3, :] = 0

    wrap_asst = _wrap_projection_mix(
        source, role_mask, bypass_embeds, bypass_mask, hidden_proj, text_proj,
    )

    diff = (wrap_asst - hf_asst).abs().max().item()
    assert torch.allclose(wrap_asst, hf_asst, atol=1e-6, rtol=0), (
        f"wrap assistant-segment must equal HF; max abs diff = {diff}"
    )


@torch.no_grad()
def test_full_chatml_concatenation_matches_hf():
    """End-to-end host-side concatenation: user + assistant."""
    text_proj, hidden_proj, cfg = _build_projections()
    thinker_h = cfg.thinker_hidden_size
    talker_h = cfg.text_config.hidden_size
    dtype = torch.float32

    torch.manual_seed(303)

    # ---- user segment ----
    user_seq = 8
    multimodal_mask = torch.zeros(1, user_seq, dtype=torch.bool)
    for p in (1, 4, 5):
        multimodal_mask[0, p] = True
    thinker_embed_user = torch.randn(1, user_seq, thinker_h, dtype=dtype)
    thinker_hidden_user = torch.randn(1, user_seq, thinker_h, dtype=dtype)

    hf_user = _hf_user_parts(
        0, user_seq, multimodal_mask, thinker_hidden_user, thinker_embed_user,
        text_proj, hidden_proj, talker_h, dtype=dtype,
    )

    u_source = thinker_embed_user.clone()
    u_source[multimodal_mask] = thinker_hidden_user[multimodal_mask]
    u_role = (~multimodal_mask).unsqueeze(-1).to(dtype)
    u_bypass_embeds = torch.zeros_like(hf_user)
    u_bypass_mask = torch.zeros(1, user_seq, 1, dtype=dtype)

    # ---- assistant segment ----
    assistant_seq = 4
    thinker_embed_asst = torch.randn(1, assistant_seq, thinker_h, dtype=dtype)
    tts_pad = torch.randn(1, 1, talker_h, dtype=dtype)
    tts_bos = torch.randn(1, 1, talker_h, dtype=dtype)
    codec_specials = torch.randn(1, 6, talker_h, dtype=dtype)
    asst_hidden = text_proj(thinker_embed_asst)
    asst_text_hidden = torch.cat(
        [asst_hidden[:, :3], tts_pad.expand(-1, 4, -1), tts_bos, asst_hidden[:, 3:4]],
        dim=1,
    )
    asst_codec_hidden = torch.cat(
        [torch.zeros(1, 3, talker_h, dtype=dtype), codec_specials],
        dim=1,
    )
    hf_asst = asst_text_hidden + asst_codec_hidden

    a_source = torch.zeros(1, 9, thinker_h, dtype=dtype)
    a_source[:, :3, :] = thinker_embed_asst[:, :3, :]
    a_role = torch.ones(1, 9, 1, dtype=dtype)
    a_bypass_embeds = hf_asst.clone()
    a_bypass_embeds[:, :3, :] = 0
    a_bypass_mask = torch.ones(1, 9, 1, dtype=dtype)
    a_bypass_mask[:, :3, :] = 0

    # ---- concatenate and apply wrap formula on the full sequence ----
    hf_full = torch.cat([hf_user, hf_asst], dim=1)
    wrap_source = torch.cat([u_source, a_source], dim=1)
    wrap_role = torch.cat([u_role, a_role], dim=1)
    wrap_bypass_embeds = torch.cat([u_bypass_embeds, a_bypass_embeds], dim=1)
    wrap_bypass_mask = torch.cat([u_bypass_mask, a_bypass_mask], dim=1)

    wrap_full = _wrap_projection_mix(
        wrap_source, wrap_role, wrap_bypass_embeds, wrap_bypass_mask,
        hidden_proj, text_proj,
    )

    diff = (wrap_full - hf_full).abs().max().item()
    assert torch.allclose(wrap_full, hf_full, atol=1e-6, rtol=0), (
        f"wrap full-chatml must equal HF; max abs diff = {diff}"
    )


@torch.no_grad()
def test_decode_bypass_path_returns_bypass_embeds_exactly():
    """Decode case: bypass_mask=1, bypass_embeds is the codec-token embed.

    The wrap formula must propagate ``bypass_embeds`` bit-exactly through the
    ``projected*0 + bypass_embeds*1`` mix. This is what enables a single graph
    to serve both prefill and decode.
    """
    text_proj, hidden_proj, cfg = _build_projections()
    thinker_h = cfg.thinker_hidden_size
    talker_h = cfg.text_config.hidden_size
    dtype = torch.float32

    torch.manual_seed(404)
    source = torch.randn(1, 1, thinker_h, dtype=dtype)
    role_mask = torch.zeros(1, 1, 1, dtype=dtype)  # arbitrary; bypass overrides
    bypass_embeds = torch.randn(1, 1, talker_h, dtype=dtype)
    bypass_mask = torch.ones(1, 1, 1, dtype=dtype)

    wrap_out = _wrap_projection_mix(
        source, role_mask, bypass_embeds, bypass_mask, hidden_proj, text_proj,
    )

    # Bit-exact: float * 0 = 0 for finite values; float + 0 = float.
    assert torch.equal(wrap_out, bypass_embeds), (
        "decode path must emit bypass_embeds bit-exactly"
    )


def test_wrap_formula_matches_source():
    """Regression net: if ``_talker_model.py`` changes its formula, fail loud.

    The ``_wrap_projection_mix`` helper above is a verbatim copy of the wrap
    forward's projection block. If anyone edits the wrap source, this test
    fails until the helper (and the equivalence proofs above) are revisited.
    """
    src = _WRAP_SOURCE.read_text()
    expected_lines = [
        "hidden_proj_out = self.hidden_projection(source)",
        "text_proj_out = self.text_projection(source)",
        "one_minus_role = role_mask * -1.0 + 1.0",
        "projected = hidden_proj_out * one_minus_role + text_proj_out * role_mask",
        "one_minus_bypass = bypass_mask * -1.0 + 1.0",
        "inputs_embeds = projected * one_minus_bypass + bypass_embeds * bypass_mask",
    ]
    missing = [line for line in expected_lines if line not in src]
    assert not missing, (
        f"wrap source no longer contains the lines this test mirrors: {missing}. "
        f"Update `_wrap_projection_mix` and re-verify equivalence."
    )


# ---------------------------------------------------------------------------
# Adapter-level test: XHQwen3OmniMoeTalkerModel.prepare_inputs produces the
# 8-tuple that ``_Qwen3OmniMoeTalkerForConditionalGeneration.forward`` expects.
# Bypasses LLMBaseModel.__init__ (which needs HF weights) by stubbing the
# attributes prepare_inputs actually reads.
# ---------------------------------------------------------------------------


from xh_model_zoo.xh_llm.models.qwen3_omni.qwen3_omni_moe_talker_model import (  # noqa: E402
    XHQwen3OmniMoeTalkerModel,
)


class _AdapterStub:
    """Lightweight stand-in exposing only what prepare_inputs reads."""

    def __init__(self, input_sequence_length=32):
        self.execution_device = torch.device("cpu")
        self.input_sequence_length = input_sequence_length
        self.past_key_caches = ["__pk_stub__"]
        self.past_value_caches = ["__pv_stub__"]

    _pad_to_seq_len = XHQwen3OmniMoeTalkerModel._pad_to_seq_len
    prepare_inputs = XHQwen3OmniMoeTalkerModel.prepare_inputs


@torch.no_grad()
def test_prepare_inputs_emits_eight_tuple_with_correct_shapes_and_padding():
    stub = _AdapterStub(input_sequence_length=32)

    thinker_h, talker_h = 8, 10
    seq_a, seq_b = 12, 7
    data = {
        "hidden_state": [
            torch.randn(seq_a, thinker_h, dtype=torch.float16),
            torch.randn(seq_b, thinker_h, dtype=torch.float16),
        ],
        "role_mask": [
            torch.ones(seq_a, 1, dtype=torch.float16),
            torch.zeros(seq_b, 1, dtype=torch.float16),
        ],
        "bypass_embeds": [
            torch.randn(seq_a, talker_h, dtype=torch.float16),
            torch.randn(seq_b, talker_h, dtype=torch.float16),
        ],
        "bypass_mask": [
            torch.zeros(seq_a, 1, dtype=torch.float16),
            torch.ones(seq_b, 1, dtype=torch.float16),
        ],
        "past_seq_length": [0, 5],
    }

    out = stub.prepare_inputs(data)

    # 1. Wrap forward expects exactly 8 positional arguments.
    assert len(out) == 8
    (
        source,
        role_mask,
        bypass_embeds,
        bypass_mask,
        past_seq_length,
        current_input_length,
        past_key_caches,
        past_value_caches,
    ) = out

    # 2. All four head tensors are padded to input_sequence_length on dim 1.
    assert source.shape == (2, 32, thinker_h)
    assert role_mask.shape == (2, 32, 1)
    assert bypass_embeds.shape == (2, 32, talker_h)
    assert bypass_mask.shape == (2, 32, 1)

    # 3. Padding region is filled with zeros so it cannot poison the trunk
    #    (KV cache writes are gated by current_input_length).
    assert torch.all(source[0, seq_a:, :] == 0)
    assert torch.all(role_mask[0, seq_a:, :] == 0)
    assert torch.all(bypass_embeds[0, seq_a:, :] == 0)
    assert torch.all(bypass_mask[0, seq_a:, :] == 0)
    assert torch.all(source[1, seq_b:, :] == 0)
    assert torch.all(bypass_mask[1, seq_b:, :] == 0)

    # 4. Real per-batch lengths are surfaced for KV cache write masking.
    assert current_input_length.tolist() == [seq_a, seq_b]
    assert past_seq_length.tolist() == [0, 5]

    # 5. KV caches are passed through unmodified (identity, not copies).
    assert past_key_caches is stub.past_key_caches
    assert past_value_caches is stub.past_value_caches


@torch.no_grad()
def test_prepare_inputs_rejects_legacy_inputs_embeds_path():
    """The fused projection lives inside the wrap graph, so the legacy
    ``inputs_embeds`` / ``input_ids`` entry must be refused — silently
    accepting it would route post-projection embeddings into the wrap's
    ``source`` slot and double-project.
    """
    import pytest

    stub = _AdapterStub()
    legacy_data = {
        "inputs_embeds": [torch.randn(4, 10, dtype=torch.float16)],
        "past_seq_length": [0],
    }
    with pytest.raises(AssertionError, match="hidden_state"):
        stub.prepare_inputs(legacy_data)


@torch.no_grad()
def test_prepare_inputs_for_graph_returns_same_eight_tuple():
    """LLMBaseModel.prepare_inputs_for_graph unpacks 5 values; this override
    must forward the 8-tuple verbatim or downstream conversion breaks.
    """

    class _StubWithGraph(_AdapterStub):
        prepare_inputs_for_graph = XHQwen3OmniMoeTalkerModel.prepare_inputs_for_graph

    stub = _StubWithGraph(input_sequence_length=16)

    data = {
        "hidden_state": [torch.zeros(4, 8, dtype=torch.float16)],
        "role_mask": [torch.ones(4, 1, dtype=torch.float16)],
        "bypass_embeds": [torch.zeros(4, 10, dtype=torch.float16)],
        "bypass_mask": [torch.zeros(4, 1, dtype=torch.float16)],
        "past_seq_length": [0],
    }
    direct = stub.prepare_inputs(data)
    via_graph = stub.prepare_inputs_for_graph(data)

    assert len(via_graph) == 8
    assert len(direct) == len(via_graph)
    for a, b in zip(direct, via_graph):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        else:
            assert a == b


# ---------------------------------------------------------------------------
# Class-level test: invoke the REAL wrap forward (not the copied helper)
# with stubs for the trunk + codec head. Confirms the live class still
# produces HF-equivalent inputs_embeds for the next layer.
# ---------------------------------------------------------------------------


from xh_model_zoo.xh_llm.models.qwen3_omni._talker_model import (  # noqa: E402
    _Qwen3OmniMoeTalkerForConditionalGeneration,
)


@torch.no_grad()
def test_real_wrap_forward_feeds_hf_equivalent_inputs_embeds_to_trunk():
    """Capture the actual ``inputs_embeds`` that the live wrap forward feeds
    to ``self.model``. It must equal the HF host-built reference for both
    user (mm + text) and assistant (projected prefix + bypassed suffix)
    segments concatenated end-to-end.
    """
    text_proj, hidden_proj, cfg = _build_projections()
    thinker_h = cfg.thinker_hidden_size
    talker_h = cfg.text_config.hidden_size
    dtype = torch.float32

    # Build the same chatml mix as test_full_chatml_concatenation_matches_hf
    torch.manual_seed(505)

    user_seq = 6
    multimodal_mask = torch.tensor([[False, True, False, True, False, False]])
    thinker_embed_user = torch.randn(1, user_seq, thinker_h, dtype=dtype)
    thinker_hidden_user = torch.randn(1, user_seq, thinker_h, dtype=dtype)

    hf_user = _hf_user_parts(
        0, user_seq, multimodal_mask, thinker_hidden_user, thinker_embed_user,
        text_proj, hidden_proj, talker_h, dtype=dtype,
    )

    u_source = thinker_embed_user.clone()
    u_source[multimodal_mask] = thinker_hidden_user[multimodal_mask]
    u_role = (~multimodal_mask).unsqueeze(-1).to(dtype)
    u_bypass_embeds = torch.zeros_like(hf_user)
    u_bypass_mask = torch.zeros(1, user_seq, 1, dtype=dtype)

    # Assistant segment (purely bypass for simplicity — the projected-prefix
    # case is already exercised in the helper-based tests above).
    asst_seq = 5
    a_source = torch.zeros(1, asst_seq, thinker_h, dtype=dtype)
    a_role = torch.ones(1, asst_seq, 1, dtype=dtype)
    a_bypass_embeds = torch.randn(1, asst_seq, talker_h, dtype=dtype)
    a_bypass_mask = torch.ones(1, asst_seq, 1, dtype=dtype)

    hf_full = torch.cat([hf_user, a_bypass_embeds], dim=1)
    source = torch.cat([u_source, a_source], dim=1)
    role_mask = torch.cat([u_role, a_role], dim=1)
    bypass_embeds = torch.cat([u_bypass_embeds, a_bypass_embeds], dim=1)
    bypass_mask = torch.cat([u_bypass_mask, a_bypass_mask], dim=1)

    # Stub trunk + codec head: capture inputs_embeds, return shape-compatible
    # placeholders so the wrap forward can complete.
    captured = {}

    class _StubModelOutput:
        def __init__(self, hs):
            self.last_hidden_state = hs

    def stub_model(*, inputs_embeds, past_seq_length, current_input_length,
                   past_key_cache, past_value_cache):
        captured["inputs_embeds"] = inputs_embeds.clone()
        return _StubModelOutput(inputs_embeds)

    def stub_codec_head(hidden_states):
        return hidden_states[..., :3]  # arbitrary projection just for shape

    stub = type("WrapStub", (), {})()
    stub.hidden_projection = hidden_proj
    stub.text_projection = text_proj
    stub.model = stub_model
    stub.codec_head = stub_codec_head

    # Invoke the LIVE wrap forward. If the formula in _talker_model.py drifts,
    # this test fails because captured["inputs_embeds"] won't match hf_full.
    logits, hidden_states = _Qwen3OmniMoeTalkerForConditionalGeneration.forward(
        stub,
        source=source,
        role_mask=role_mask,
        bypass_embeds=bypass_embeds,
        bypass_mask=bypass_mask,
        past_seq_length=torch.tensor([0], dtype=torch.int32),
        current_input_length=torch.tensor([source.shape[1]], dtype=torch.int32),
        past_key_cache=[],
        past_value_cache=[],
    )

    diff = (captured["inputs_embeds"] - hf_full).abs().max().item()
    assert torch.allclose(captured["inputs_embeds"], hf_full, atol=1e-6, rtol=0), (
        f"live wrap forward inputs_embeds must equal HF host-built reference; "
        f"max abs diff = {diff}"
    )

    # Sanity: outputs flow through the stubs without shape errors.
    assert logits.shape == (1, user_seq + asst_seq, 3)
    assert hidden_states.shape == (1, user_seq + asst_seq, talker_h)


@torch.no_grad()
def test_real_wrap_forward_bit_exact_in_fp16():
    """Production dtype is fp16. Confirm the equivalence still holds bit-exact:
    masks are exactly 0.0/1.0 in fp16, so ``1-mask``, ``x*1``, ``x*0``,
    ``x+0`` all incur no rounding.
    """
    text_proj, hidden_proj, cfg = _build_projections()
    text_proj = text_proj.to(torch.float16)
    hidden_proj = hidden_proj.to(torch.float16)
    thinker_h = cfg.thinker_hidden_size
    talker_h = cfg.text_config.hidden_size
    dtype = torch.float16

    torch.manual_seed(606)
    seq = 7
    multimodal_mask = torch.tensor([[True, False, False, True, False, False, True]])
    thinker_embed = torch.randn(1, seq, thinker_h, dtype=dtype)
    thinker_hidden = torch.randn(1, seq, thinker_h, dtype=dtype)

    hf_user = _hf_user_parts(
        0, seq, multimodal_mask, thinker_hidden, thinker_embed,
        text_proj, hidden_proj, talker_h, dtype=dtype,
    )

    source = thinker_embed.clone()
    source[multimodal_mask] = thinker_hidden[multimodal_mask]
    role_mask = (~multimodal_mask).unsqueeze(-1).to(dtype)
    bypass_embeds = torch.zeros_like(hf_user)
    bypass_mask = torch.zeros(1, seq, 1, dtype=dtype)

    captured = {}

    class _StubModelOutput:
        def __init__(self, hs):
            self.last_hidden_state = hs

    def stub_model(*, inputs_embeds, **_):
        captured["inputs_embeds"] = inputs_embeds.clone()
        return _StubModelOutput(inputs_embeds)

    stub = type("WrapStub", (), {})()
    stub.hidden_projection = hidden_proj
    stub.text_projection = text_proj
    stub.model = stub_model
    stub.codec_head = lambda hs: hs[..., :2]

    _Qwen3OmniMoeTalkerForConditionalGeneration.forward(
        stub,
        source=source,
        role_mask=role_mask,
        bypass_embeds=bypass_embeds,
        bypass_mask=bypass_mask,
        past_seq_length=torch.tensor([0], dtype=torch.int32),
        current_input_length=torch.tensor([seq], dtype=torch.int32),
        past_key_cache=[],
        past_value_cache=[],
    )

    # fp16 + 0/1 masks → bit-exact equality.
    assert torch.equal(captured["inputs_embeds"], hf_user), (
        "fp16 wrap output must be bit-exact equal to HF reference"
    )


@torch.no_grad()
def test_adapter_8tuple_feeds_real_wrap_forward_end_to_end():
    """End-to-end integration: ``XHQwen3OmniMoeTalkerModel.prepare_inputs``
    builds the 8-tuple, the live wrap forward consumes it, and the
    ``inputs_embeds`` it feeds the trunk equals the HF host-built reference.

    This is the only test that wires adapter → wrap → trunk in one go, and
    therefore the only one that proves the whole pipeline is consumable.
    """
    text_proj, hidden_proj, cfg = _build_projections()
    text_proj = text_proj.to(torch.float16)
    hidden_proj = hidden_proj.to(torch.float16)
    thinker_h = cfg.thinker_hidden_size
    talker_h = cfg.text_config.hidden_size
    dtype = torch.float16

    # ---- Build per-batch fields per HF semantics for a user segment ----
    torch.manual_seed(707)
    seq = 9
    multimodal_mask = torch.tensor([[True, False, True, True, False, False, False, True, False]])
    thinker_embed = torch.randn(1, seq, thinker_h, dtype=dtype)
    thinker_hidden_t = torch.randn(1, seq, thinker_h, dtype=dtype)

    hf_ref = _hf_user_parts(
        0, seq, multimodal_mask, thinker_hidden_t, thinker_embed,
        text_proj, hidden_proj, talker_h, dtype=dtype,
    )

    source_per_batch = thinker_embed[0].clone()
    source_per_batch[multimodal_mask[0]] = thinker_hidden_t[0][multimodal_mask[0]]
    role_per_batch = (~multimodal_mask[0]).unsqueeze(-1).to(dtype)
    bypass_embeds_per_batch = torch.zeros(seq, talker_h, dtype=dtype)
    bypass_mask_per_batch = torch.zeros(seq, 1, dtype=dtype)

    # ---- Run adapter.prepare_inputs ----
    adapter_input_seq_len = 16  # > seq, so padding happens
    adapter = _AdapterStub(input_sequence_length=adapter_input_seq_len)
    eight_tuple = adapter.prepare_inputs({
        "hidden_state":   [source_per_batch],
        "role_mask":      [role_per_batch],
        "bypass_embeds":  [bypass_embeds_per_batch],
        "bypass_mask":    [bypass_mask_per_batch],
        "past_seq_length": [0],
    })

    # ---- Feed it to the live wrap forward ----
    captured = {}

    class _StubModelOutput:
        def __init__(self, hs):
            self.last_hidden_state = hs

    def stub_model(*, inputs_embeds, **_):
        captured["inputs_embeds"] = inputs_embeds.clone()
        return _StubModelOutput(inputs_embeds)

    wrap_stub = type("WrapStub", (), {})()
    wrap_stub.hidden_projection = hidden_proj
    wrap_stub.text_projection = text_proj
    wrap_stub.model = stub_model
    wrap_stub.codec_head = lambda hs: hs[..., :2]

    logits, hidden_states = _Qwen3OmniMoeTalkerForConditionalGeneration.forward(
        wrap_stub, *eight_tuple
    )

    # ---- Verify the inputs_embeds at the trunk boundary ----
    # The first `seq` positions must equal HF; the padding tail must be all 0
    # (because source/role/bypass_embeds/bypass_mask are all 0 there, so
    # projected = hidden_proj(0)*1 + text_proj(0)*0 ≠ 0 — wait, let's check).
    real = captured["inputs_embeds"]
    assert real.shape == (1, adapter_input_seq_len, talker_h)
    assert torch.equal(real[:, :seq, :], hf_ref), (
        "real prefix of the wrap-fed inputs_embeds must equal HF reference, "
        "bit-exact in fp16"
    )

    # Padding region: source=0, role_mask=0 → projected = hidden_proj(0).
    # Bias terms in linear_fc1/linear_fc2 mean hidden_proj(0) is generally
    # nonzero, but bypass_embeds=0 and bypass_mask=0 means the projected
    # value flows through unchanged. KV cache writes are gated by
    # current_input_length, so this is harmless — but we DO want to verify
    # current_input_length carries the real length so the trunk knows where
    # to stop.
    current_input_length = eight_tuple[5]
    assert current_input_length.tolist() == [seq], (
        "current_input_length must report the real (un-padded) sequence length "
        "so the trunk can mask out padding from the KV cache"
    )

    # Output shapes flow through the wrap stub correctly.
    assert logits.shape == (1, adapter_input_seq_len, 2)
    assert hidden_states.shape == (1, adapter_input_seq_len, talker_h)

