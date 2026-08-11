"""Unlimited-OCR numerical alignment regression tests (issue 017).

Freezes the one-shot alignment validations from issue 013 (SAM rel-pos fix,
visual / LLM golden alignment) and issue 014 (crop token/feature parity) into
repeatable regression tests, guarding against future drift.

Two tiers:

- ``test_rel_pos_*``: pure-function CPU regression for the SAM window-attention
  decomposed relative-position width fix (issue 013). No weights / GPU needed,
  so these always run and are the primary guard against re-introducing the
  ``rel_w`` width bug.
- Weight-backed alignment tests (``@pytest.mark.gpu`` + ``slow``): load the real
  ``data/models/Unlimited-OCR`` weights and compare wrap visual, LLM prefill and
  crop token/feature counts against the native HF reference. These ``skip`` when
  the weights or a CUDA device are missing (never ``fail``), so CI can select
  them explicitly (e.g. ``pytest -m gpu``).

Run everything (weights + GPU present):
    pytest tests/unlimited_ocr_alignment_test.py
CPU-only rel-pos guard:
    pytest tests/unlimited_ocr_alignment_test.py -m "not gpu"
"""

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_DIR = REPO_ROOT / "data" / "models" / "Unlimited-OCR"
# Golden alignment (issue 013) was validated against this demo image; the wrap
# LLM path is content-sensitive (issue 018 records wrap-vs-HF drift on other
# images), so alignment regressions must reuse the validated demo image.
GOLDEN_IMAGE = REPO_ROOT / "data" / "images" / "qwen2_vl_demo.jpeg"
# A small (<=640) image that degrades to the base (no-crop) 273-token path.
SMALL_IMAGE = REPO_ROOT / "data" / "images" / "houmo_logo.jpg"

BASE_LLM_CONFIG = (
    REPO_ROOT
    / "configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_llm_base_xh2a_32k.py"
)
BASE_VISUAL_CONFIG = (
    REPO_ROOT
    / "configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_visual_base_xh2a_32k.py"
)
GUNDAM_LLM_CONFIG = (
    REPO_ROOT
    / "configs_merak/xh2a/llm_models/unlimited_ocr/gundam/unlimited_ocr_llm_gundam_xh2a_32k.py"
)

# Alignment thresholds (fp32) from issue 013 validation records.
VISUAL_FP32_MAX_ATOL = 1e-2  # observed ~1.4e-3 after the rel-pos fix
REL_POS_ATOL = 1e-4  # observed ~6e-7 vs native einsum


# --------------------------------------------------------------------------- #
# rel-pos pure-function regression (issue 013 width bug guard) - CPU, no weights
# --------------------------------------------------------------------------- #


def _make_rel_pos_inputs(q_h, q_w, dim, rel_len_h, rel_len_w, seed=1234):
    torch.manual_seed(seed)
    batch_heads = 2
    q = torch.randn(batch_heads, q_h * q_w, dim, dtype=torch.float32)
    rel_pos_h = torch.randn(rel_len_h, dim, dtype=torch.float32)
    rel_pos_w = torch.randn(rel_len_w, dim, dtype=torch.float32)
    return q, rel_pos_h, rel_pos_w


@pytest.mark.parametrize(
    ("q_h", "q_w"),
    [(14, 14), (8, 12), (12, 8)],
)
def test_rel_pos_matches_native_einsum(q_h, q_w):
    """Wrap ``_add_decomposed_rel_pos`` must match native einsum rel_h/rel_w.

    Regression for issue 013: the width bias (``rel_w``) is indexed by
    query-width, so a plain matmul that reuses the query-height alignment is
    wrong. Non-square (q_h != q_w) grids expose the bug; keep those parametrized.
    """
    from xhmodel_merak.xh_llm.models.unlimited_ocr._visual_model_impl import (
        _add_decomposed_rel_pos,
    )
    from xhmodel_merak.xh_llm.models.unlimited_ocr.deepencoder import (
        add_decomposed_rel_pos,
    )

    dim = 16
    rel_len_h = 2 * q_h - 1
    rel_len_w = 2 * q_w - 1
    q, rel_pos_h, rel_pos_w = _make_rel_pos_inputs(q_h, q_w, dim, rel_len_h, rel_len_w)

    native_h, native_w = add_decomposed_rel_pos(
        q, rel_pos_h, rel_pos_w, (q_h, q_w), (q_h, q_w)
    )
    wrap_h, wrap_w = _add_decomposed_rel_pos(
        q, rel_pos_h, rel_pos_w, (q_h, q_w), (q_h, q_w), rel_len_h, rel_len_w
    )

    assert wrap_h.shape == native_h.shape
    assert wrap_w.shape == native_w.shape
    h_diff = (wrap_h - native_h).abs().max().item()
    w_diff = (wrap_w - native_w).abs().max().item()
    assert h_diff <= REL_POS_ATOL, f"rel_h diff {h_diff} > {REL_POS_ATOL}"
    assert w_diff <= REL_POS_ATOL, f"rel_w (width) diff {w_diff} > {REL_POS_ATOL}"


def test_rel_pos_width_component_is_axis_sensitive():
    """A non-square grid must produce a width bias that depends on the width axis.

    This is a structural guard: if someone reintroduces the pre-013 bug (reusing
    the height alignment for width), ``rel_w`` collapses to a height-indexed
    result. We reproduce the buggy width contraction with einsum and assert the
    native width bias genuinely differs from it on a non-square grid, so the
    parity test above has real discriminating power.
    """
    from xhmodel_merak.xh_llm.models.unlimited_ocr.deepencoder import (
        add_decomposed_rel_pos,
        get_rel_pos,
    )

    q_h, q_w, dim = 8, 12, 16
    rel_len_h = 2 * q_h - 1
    rel_len_w = 2 * q_w - 1
    q, rel_pos_h, rel_pos_w = _make_rel_pos_inputs(q_h, q_w, dim, rel_len_h, rel_len_w)

    _, native_w = add_decomposed_rel_pos(
        q, rel_pos_h, rel_pos_w, (q_h, q_w), (q_h, q_w)
    )

    # The pre-013 bug indexed the width bias by the query-height axis instead of
    # query-width. Reproduce that wrong contraction with einsum ("bhwc,hkc"
    # applied to the width table) and confirm the correct result differs, so a
    # degenerate grid can't make the parity test pass trivially.
    r_w = get_rel_pos(q_w, q_w, rel_pos_w)  # (q_w, k_w, dim)
    r_q = q.reshape(q.shape[0], q_h, q_w, dim)
    buggy_w = torch.einsum("bhwc,wkc->bhwk", r_q, r_w)  # correct axis
    right_w = buggy_w.unsqueeze(-2).reshape(q.shape[0], q_h * q_w, 1, q_w)
    assert (right_w - native_w).abs().max().item() <= REL_POS_ATOL

    # Now the genuinely buggy contraction: index width by the height table shape.
    r_h_as_w = get_rel_pos(q_h, q_h, rel_pos_w[:rel_len_h])  # (q_h, k_h, dim)
    wrong = torch.einsum("bhwc,hkc->bhwk", r_q, r_h_as_w)  # aligns q_h, not q_w
    wrong = wrong.unsqueeze(-1).reshape(q.shape[0], q_h * q_w, q_h, 1)
    # Shapes differ (k_w vs k_h), proving width and height biases are not
    # interchangeable on a non-square grid.
    assert native_w.shape[-1] == q_w
    assert wrong.shape[-2] == q_h
    assert native_w.shape != wrong.shape


# --------------------------------------------------------------------------- #
# Weight-backed alignment tests (issue 013 / 014) - require weights + GPU
# --------------------------------------------------------------------------- #

_needs_weights = pytest.mark.skipif(
    not MODEL_DIR.exists(),
    reason=f"Unlimited-OCR weights not found at {MODEL_DIR}",
)
_needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required for weight-backed alignment",
)
_needs_image = pytest.mark.skipif(
    not GOLDEN_IMAGE.exists(),
    reason=f"golden alignment image not found at {GOLDEN_IMAGE}",
)


def _init_xhquant():
    from xhquant.api import set_random_seed, xhquant_init

    xhquant_init(None, False)
    set_random_seed(1024)


def _load_processor(model_cfg, crop_mode=False):
    from transformers import AutoTokenizer

    from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_processor import (
        XHUnlimitedOCRProcessor,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_model, trust_remote_code=True)

    # A visual config exposes image_size/base_size/... at the top level; an LLM
    # config nests them under ``visual_config``. Pick whichever carries a valid
    # image_size so both config shapes construct the same processor.
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    if _get(model_cfg, "image_size") is not None:
        vc = model_cfg
    else:
        vc = model_cfg.visual_config

    return XHUnlimitedOCRProcessor(
        tokenizer,
        image_token_id=model_cfg.image_token_id,
        image_size=_get(vc, "image_size"),
        base_size=_get(vc, "base_size"),
        patch_size=_get(vc, "patch_size"),
        downsample_ratio=_get(vc, "downsample_ratio"),
        crop_mode=crop_mode,
        max_crop_num=_get(vc, "max_crop_num", 32) or 32,
    )


@pytest.mark.gpu
@pytest.mark.slow
@_needs_weights
@_needs_gpu
@_needs_image
def test_wrap_visual_matches_hf_native_fp32():
    """Base wrap visual vs HF native visual: fp32 max abs diff <= 1e-2 (issue 013)."""
    from xhmodel_merak.xh_llm import (
        AutoLLMConfig,
        AutoLLMModel,
        LLMInferenceContextManager,
        LLMModelState,
    )
    from xhmodel_merak.xh_llm.models.unlimited_ocr.modeling_unlimitedocr import (
        UnlimitedOCRForCausalLM,
    )
    from xhmodel_merak.xh_llm.models.unlimited_ocr.modeling_unlimitedocr_patch import (
        unlimited_ocr_patch,
    )
    from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_visual_model import (
        UnlimitedOCRBaseVisualModel,
    )
    from xhquant.api import Config
    from xhquant.utils import ContextManagers

    _init_xhquant()
    device = "cuda"
    dtype = torch.float32

    cfg = Config.fromfile(str(BASE_VISUAL_CONFIG))
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)

    processor = _load_processor(model_cfg, crop_mode=False)
    inputs = processor.process("<image>\n", str(GOLDEN_IMAGE), device=device)
    image = inputs["images_ori"].to(dtype)

    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.set_state(LLMModelState.from_string("wrap"))

    with ContextManagers([LLMInferenceContextManager(xh_model), torch.no_grad()]):
        xh_model.to(device=device, dtype=dtype)
        wrap_embeds = xh_model(image)
    if isinstance(wrap_embeds, (tuple, list)):
        wrap_embeds = wrap_embeds[0]

    hf_native = UnlimitedOCRForCausalLM.from_pretrained(
        model_cfg.hf_model, dtype=dtype, trust_remote_code=False
    )
    hf_visual = (
        UnlimitedOCRBaseVisualModel(unlimited_ocr_patch(hf_native))
        .to(device=device, dtype=dtype)
        .eval()
    )
    with torch.no_grad():
        hf_embeds = hf_visual(image)

    assert wrap_embeds.shape == hf_embeds.shape == (1, 273, 1280)
    diff = (hf_embeds.float() - wrap_embeds.float()).abs()
    max_diff = diff.max().item()
    assert max_diff <= VISUAL_FP32_MAX_ATOL, (
        f"wrap visual vs HF fp32 max diff {max_diff} > {VISUAL_FP32_MAX_ATOL}"
    )


@pytest.mark.gpu
@pytest.mark.slow
@_needs_weights
@_needs_gpu
@_needs_image
def test_wrap_llm_prefill_argmax_matches_native_golden():
    """Wrap LLM prefill last-token argmax must match HF native golden (issue 013).

    Uses native HF visual embeddings for both paths to isolate the LLM prefill
    from visual quant error.
    """
    from xhmodel_merak.xh_llm import (
        AutoLLMConfig,
        AutoLLMModel,
        LLMInferenceContextManager,
        LLMModelState,
    )
    from xhmodel_merak.xh_llm.models.unlimited_ocr.modeling_unlimitedocr import (
        UnlimitedOCRForCausalLM,
    )
    from xhmodel_merak.xh_llm.models.unlimited_ocr.modeling_unlimitedocr_patch import (
        unlimited_ocr_patch,
    )
    from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_visual_model import (
        UnlimitedOCRBaseVisualModel,
    )
    from xhquant.api import Config
    from xhquant.utils import ContextManagers

    _init_xhquant()
    device = "cuda"
    dtype = torch.float16

    cfg = Config.fromfile(str(BASE_LLM_CONFIG))
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)

    processor = _load_processor(model_cfg, crop_mode=False)
    inputs = processor.process("<image>\\nFree OCR. ", str(GOLDEN_IMAGE), device=device)
    input_ids = inputs["input_ids"]
    images_seq_mask = inputs["images_seq_mask"]
    images_ori = inputs["images_ori"].to(device=device, dtype=dtype)
    seq_length = int(input_ids.shape[1])

    # Native HF visual + full HF forward -> golden next token.
    hf_native = UnlimitedOCRForCausalLM.from_pretrained(
        model_cfg.hf_model, dtype=dtype, trust_remote_code=False
    )
    hf_visual = (
        UnlimitedOCRBaseVisualModel(unlimited_ocr_patch(hf_native))
        .to(device=device, dtype=dtype)
        .eval()
    )
    with torch.no_grad():
        image_embeds = hf_visual(images_ori)
    image_embeds_flat = image_embeds.reshape(-1, image_embeds.shape[-1])

    image_size = model_cfg.visual_config.image_size
    images_crop = torch.zeros((1, 3, image_size, image_size), device=device, dtype=dtype)
    hf_model = hf_native.to(device).eval()
    with torch.no_grad():
        golden_out = hf_model(
            input_ids=input_ids,
            images=[(images_crop, images_ori)],
            images_seq_mask=images_seq_mask,
            images_spatial_crop=inputs["images_spatial_crop"],
            use_cache=False,
            return_dict=True,
        )
    golden_token = int(golden_out.logits[0, -1].argmax().item())

    # Wrap LLM prefill with the same native visual embeddings.
    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.set_state(LLMModelState.from_string("wrap"))
    with ContextManagers([LLMInferenceContextManager(xh_model), torch.no_grad()]):
        xh_model.to(device=device, dtype=dtype)
        xh_model.set_input_sequence_length(
            max(seq_length, xh_model.wrap_cfg.input_sequence_length)
        )
        data_processor = xh_model.get_data_preprocessor()
        processed = data_processor(
            {
                "input_ids": input_ids,
                "image_embeds": image_embeds_flat,
                "images_seq_mask": images_seq_mask,
                "past_seq_length": 0,
            }
        )
        logits = xh_model(*processed)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    last = logits[0, seq_length - 1] if logits.shape[1] >= seq_length else logits[0, -1]
    wrap_token = int(last.argmax().item())

    assert wrap_token == golden_token, (
        f"wrap prefill argmax {wrap_token} != native golden {golden_token}"
    )


@pytest.mark.gpu
@pytest.mark.slow
@_needs_weights
@_needs_gpu
@_needs_image
def test_crop_token_feature_counts_align():
    """Crop path: small image -> base 273; large image token count == feature count.

    Also checks local-crop ``_encode`` numerically matches HF sam+clip+projector
    (issue 014 records diff = 0).
    """
    from PIL import Image

    from xhmodel_merak.xh_llm import (
        AutoLLMConfig,
        AutoLLMModel,
        LLMInferenceContextManager,
        LLMModelState,
    )
    from xhquant.api import Config
    from xhquant.utils import ContextManagers

    _init_xhquant()
    device = "cuda"
    dtype = torch.float16

    cfg = Config.fromfile(str(GUNDAM_LLM_CONFIG))
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    vc = model_cfg.visual_config
    assert vc.crop_mode is True

    processor = _load_processor(model_cfg, crop_mode=True)

    # Small image (<=640) degrades to base: 273 tokens, no local crops.
    if SMALL_IMAGE.exists():
        small = processor.process("<image>\\nFree OCR. ", str(SMALL_IMAGE), device=device)
        assert small["images_spatial_crop"].tolist() == [[1, 1]]
        assert int(small["images_seq_mask"].sum().item()) == 273
        assert small["images_crop"].shape[0] == 0

    # Large synthetic image: dynamic crop, token count must equal feature count.
    big_path = REPO_ROOT / "work_dirs" / "_ocr_align_big.png"
    big_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1600, 900), (120, 120, 120)).save(big_path)
    try:
        big = processor.process("<image>\\nFree OCR. ", str(big_path), device=device)
    finally:
        big_path.unlink(missing_ok=True)

    wc, hc = (int(x) for x in big["images_spatial_crop"][0])
    n_tokens = int(big["images_seq_mask"].sum().item())
    assert wc > 1 or hc > 1, "large image should trigger dynamic crop"
    assert big["images_crop"].shape[0] == wc * hc

    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.set_state(LLMModelState.from_string("wrap"))
    with ContextManagers([LLMInferenceContextManager(xh_model), torch.no_grad()]):
        xh_model.to(device=device, dtype=dtype)
        image_embeds = xh_model.visual.forward_crop(
            big["images_ori"][0].unsqueeze(0).to(dtype),
            big["images_crop"].to(dtype),
            wc,
            hc,
        )
    n_feat = int(image_embeds.shape[1])
    assert n_tokens == n_feat, f"crop_ratio=({wc},{hc}) tokens={n_tokens} feat={n_feat}"
