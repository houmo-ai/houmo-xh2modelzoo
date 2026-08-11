"""Processor contract tests for Unlimited-OCR (CPU, no model weights).

Uses a fake tokenizer and in-memory temp images so the tests do not require the
HF model dir or GPU. Mirrors tests/gemma4_merak_processor_test.py.
"""

import math
from pathlib import Path

import pytest
import torch
from PIL import Image


class _FakeTokenizer:
    """Minimal tokenizer: maps each character to a fixed non-image token id."""

    def encode(self, text, add_special_tokens=False):
        return [10 + (ord(c) % 50) for c in text]


def _make_image(tmp_path: Path, name: str, size) -> str:
    path = tmp_path / name
    Image.new("RGB", size, (128, 128, 128)).save(path)
    return str(path)


def _make_processor(crop_mode=False, image_size=1024, base_size=1024):
    from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_processor import (
        XHUnlimitedOCRProcessor,
    )

    return XHUnlimitedOCRProcessor(
        _FakeTokenizer(),
        image_token_id=128815,
        image_size=image_size,
        base_size=base_size,
        patch_size=16,
        downsample_ratio=4,
        crop_mode=crop_mode,
        max_crop_num=32,
    )


def test_processor_base_single_image_273_tokens(tmp_path):
    proc = _make_processor(crop_mode=False)
    assert proc.num_queries == 16
    assert proc.image_token_count == 273

    image = _make_image(tmp_path, "img.png", (800, 600))
    out = proc.process("<image>\\nFree OCR. ", image, device="cpu")

    assert out["input_ids"].shape[0] == 1
    assert int(out["images_seq_mask"].sum().item()) == 273
    assert tuple(out["images_ori"].shape) == (1, 3, 1024, 1024)
    assert out["images_spatial_crop"].tolist() == [[1, 1]]
    # image tokens count must match the number of True positions in the mask
    n_image_tok = int((out["input_ids"] == 128815).sum().item())
    assert n_image_tok == 273


def test_processor_prompt_image_mismatch_raises(tmp_path):
    proc = _make_processor(crop_mode=False)
    image = _make_image(tmp_path, "img.png", (800, 600))
    # prompt has no <image> marker but an image is supplied
    with pytest.raises(ValueError):
        proc.process("Free OCR without marker", image, device="cpu")


def test_processor_crop_small_image_reduces_to_base(tmp_path):
    proc = _make_processor(crop_mode=True, image_size=640, base_size=1024)
    image = _make_image(tmp_path, "small.png", (500, 400))
    out = proc.process("<image>\\nFree OCR. ", image, device="cpu")

    assert out["images_spatial_crop"].tolist() == [[1, 1]]
    assert int(out["images_seq_mask"].sum().item()) == 273
    assert tuple(out["images_ori"].shape) == (1, 3, 1024, 1024)
    assert out["images_crop"].shape[0] == 0


@pytest.mark.parametrize(
    "size",
    [(1600, 900), (2000, 1000)],
)
def test_processor_crop_large_image_token_feature_count(tmp_path, size):
    proc = _make_processor(crop_mode=True, image_size=640, base_size=1024)
    image = _make_image(tmp_path, f"big_{size[0]}.png", size)
    out = proc.process("<image>\\nFree OCR. ", image, device="cpu")

    wc, hc = (int(x) for x in out["images_spatial_crop"][0])
    n_tokens = int(out["images_seq_mask"].sum().item())

    nqb = proc.num_queries_base  # 16
    nq = proc.num_queries  # 10
    glob = (nqb + 1) * nqb
    sep = 1
    loc = (nq * wc + 1) * (nq * hc) if (wc > 1 or hc > 1) else 0
    expected = glob + sep + loc
    assert n_tokens == expected, f"crop_ratio=({wc},{hc}) tokens={n_tokens} expected={expected}"
    # one local crop tensor per crop block when cropped
    if wc > 1 or hc > 1:
        assert out["images_crop"].shape[0] == wc * hc
        assert tuple(out["images_crop"].shape)[1:] == (3, 640, 640)


def test_processor_crop_token_formula_matches_reference():
    """Independent check of the crop token formula for several ratios."""
    proc = _make_processor(crop_mode=True, image_size=640, base_size=1024)
    nqb = proc.num_queries_base
    nq = proc.num_queries
    for wc, hc in [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (4, 2)]:
        tokens = proc._build_crop_image_tokens(wc, hc)
        glob = (nqb + 1) * nqb
        loc = (nq * wc + 1) * (nq * hc) if (wc > 1 or hc > 1) else 0
        assert len(tokens) == glob + 1 + loc
