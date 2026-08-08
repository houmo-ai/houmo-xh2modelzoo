import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from examples_merak.vla.pi05.pi05_droid_fae_pipeline import VARIANTS, validate_summary
from examples_merak.vla.pi05.pi05_modelscope_droid import verify_snapshot
from xhmodel_merak.xh_other_model.builder import XHLLM_TRACEABLE_MODULES
from xhmodel_merak.xh_other_model.models.pi05._export_utils import Siglip, _load_local_pi05_config_compat
from xhmodel_merak.xh_other_model.models.pi05.gemma_llm_casual_model_05 import (
    XHGemma05CLLMModel,
    _clear_gemma_traceable_modules,
)
from xhmodel_merak.xh_other_model.models.pi05.gemma_llm_model import XHGemmaLLMModel
from xhmodel_merak.xh_other_model.models.pi05.workflow import (
    _build_pi05_calibration_contexts,
    _build_pi05_expert_calibration_inputs,
    _build_pi05_gemma_calibration_inputs,
    _build_pi05_runtime_contract,
    _require_pi05_config_dir,
)
from xhmodel_merak.xh_other_model.workflows.config import WorkflowConfig


class _ScaledEmbedding(nn.Embedding):
    def __init__(self, scale: float):
        super().__init__(8, 4)
        self.scalar_embed_scale = scale

    def forward(self, input_ids):
        return super().forward(input_ids) * self.scalar_embed_scale


class _VisionConfig:
    text_config = SimpleNamespace(hidden_size=49)


class _VisionOutput:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class _VisionModel(nn.Module):
    config = _VisionConfig()

    def __init__(self, return_tensor: bool):
        super().__init__()
        self.return_tensor = return_tensor

    def get_image_features(self, pixel_values):
        output = pixel_values.mean(dim=(-2, -1))
        return output if self.return_tensor else _VisionOutput(output)


class _ExpertPolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_in_proj = nn.Linear(2, 4, bias=False)

    def embed_suffix(self, noisy_actions, timestep):
        suffix_embs = self.action_in_proj(noisy_actions)
        cond = timestep[:, None].expand(-1, 4)
        suffix_pad_masks = torch.ones(suffix_embs.shape[:2], dtype=torch.bool)
        suffix_att_masks = torch.tensor([[1, 0, 0]], dtype=torch.bool)
        return suffix_embs, suffix_pad_masks, suffix_att_masks, cond

    @staticmethod
    def _prepare_attention_masks_4d(attention_mask):
        return torch.where(attention_mask[:, None], 0.0, torch.finfo(torch.float32).min)


class _CompactPrefixPolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def embed_prefix(self, images, image_masks, tokens, token_masks):
        del images, tokens
        image_embs = [
            torch.full((1, 256, 4), float(index + 1), device=self.anchor.device)
            for index in range(len(image_masks))
        ]
        language_embs = torch.full((1, token_masks.shape[1], 4), 9.0, device=self.anchor.device)
        prefix_embs = torch.cat([*image_embs, language_embs], dim=1)
        prefix_pad_masks = torch.cat(
            [mask[:, None].expand(-1, 256) for mask in image_masks] + [token_masks], dim=1
        )
        return prefix_embs, prefix_pad_masks, torch.zeros_like(prefix_pad_masks)

    @staticmethod
    def _prepare_attention_masks_4d(attention_mask):
        return torch.where(attention_mask[:, None], 0.0, torch.finfo(torch.float32).min)


class _CompactTokenizer:
    def __call__(self, *args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 0, 0]]),
            attention_mask=torch.tensor([[1, 1, 0, 0]]),
        )


def test_gemma_prepare_inputs_preserves_scaled_embedding() -> None:
    model = object.__new__(XHGemmaLLMModel)
    nn.Module.__init__(model)
    model.token_embedding = _ScaledEmbedding(scale=3.0)
    model.input_embedding_scale = 3.0
    model.input_sequence_length = 2
    model.cache_length = 8
    model.pad_token_id = 0
    model._device = torch.device("cpu")
    model._exec_device = None
    model.past_key_caches = []
    model.past_value_caches = []

    inputs_embeds = model.prepare_inputs({"input_ids": [[1, 2]], "past_seq_length": [0]})[0]

    torch.testing.assert_close(inputs_embeds, model.token_embedding(torch.tensor([[1, 2]])))


def test_expert_prepare_inputs_applies_saved_embedding_scale() -> None:
    model = object.__new__(XHGemma05CLLMModel)
    nn.Module.__init__(model)
    model.token_embedding = nn.Embedding(8, 4)
    model.input_sequence_length = 2
    model.cache_length = 8
    model.pad_token_id = 0
    model._device = torch.device("cpu")
    model._exec_device = None
    model.past_key_caches = []
    model.past_value_caches = []

    inputs_embeds = model.prepare_inputs({"input_ids": [[1, 2]], "past_seq_length": [0]})[0]

    expected = model.token_embedding(torch.tensor([[1, 2]]))
    torch.testing.assert_close(inputs_embeds, expected)


def test_siglip_preserves_transformers_v4_tensor_contract() -> None:
    model = Siglip(_VisionModel(return_tensor=True))
    pixel_values = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)

    torch.testing.assert_close(model(pixel_values), pixel_values.mean(dim=(-2, -1)))


def test_siglip_preserves_transformers_v5_output_contract() -> None:
    model = Siglip(_VisionModel(return_tensor=False))
    pixel_values = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)

    torch.testing.assert_close(model(pixel_values), pixel_values.mean(dim=(-2, -1)) * 7.0)


def test_local_pi05_config_normalizes_legacy_optional_list(tmp_path) -> None:
    config = {
        "type": "pi05",
        "input_features": {},
        "output_features": {},
        "relative_exclude_joints": None,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    loaded = _load_local_pi05_config_compat(str(tmp_path))

    assert loaded.relative_exclude_joints == []


def test_modelscope_snapshot_verification_records_sha256_and_md5(tmp_path) -> None:
    payload = b"pi05-modelscope-test\n"
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(payload)
    expected = {
        "model.safetensors": {
            "size_bytes": len(payload),
            "sha256": "45b1e2549f812b47a2e031d6825f91975786dabf658eba8c492b010ad86895d4",
            "md5": "1a548c88ebdc540d74a963464a1b858b",
        }
    }

    result = verify_snapshot(tmp_path, expected)

    assert result["passed"] is True
    assert result["files"]["model.safetensors"]["matches"] == {
        "size_bytes": True,
        "sha256": True,
        "md5": True,
    }


def test_expert_calibration_uses_action_suffix_and_time_conditioning() -> None:
    policy = SimpleNamespace(
        config=SimpleNamespace(max_action_dim=2),
        model=_ExpertPolicyModel(),
    )
    contexts = [
        {
            "prefix_pad_masks": torch.tensor([[True, True, True, False]]),
            "valid_prefix_length": 3,
            "prefix_key_values": [
                (
                    torch.full((1, 1, 3, 2), 2.0),
                    torch.full((1, 1, 3, 2), 3.0),
                )
            ],
        }
        for _ in range(3)
    ]
    template_inputs = [
        torch.zeros(1, 3, 4),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.ones(1, 4),
        torch.zeros(1, 1, 1, 8),
        torch.zeros(1, 1, 8, 2),
        torch.zeros(1, 1, 8, 2),
    ]

    batches = _build_pi05_expert_calibration_inputs(
        policy=policy,
        contexts=contexts,
        template_inputs=template_inputs,
        sequence_length=3,
        cache_length=8,
        device=torch.device("cpu"),
    )

    assert len(batches) == 3
    assert [batch[0].shape for batch in batches] == [(1, 3, 4)] * 3
    assert [batch[1].item() for batch in batches] == [3] * 3
    assert [batch[2].item() for batch in batches] == [3] * 3
    torch.testing.assert_close(batches[1][3], torch.full((1, 4), 0.5))
    assert batches[0][4].shape == (1, 1, 1, 8)
    assert torch.all(batches[0][4][..., :6] == 0)
    assert torch.all(batches[0][4][..., 6:] < 0)
    torch.testing.assert_close(
        batches[0][5].data[..., :3, :],
        torch.full((1, 1, 3, 2), 2.0, dtype=torch.float16),
    )
    torch.testing.assert_close(
        batches[0][6].data[..., :3, :],
        torch.full((1, 1, 3, 2), 3.0, dtype=torch.float16),
    )
    assert torch.count_nonzero(batches[0][5].data[..., 3:, :]) == 0
    assert torch.count_nonzero(batches[0][6].data[..., 3:, :]) == 0


def test_gemma_calibration_preserves_prefix_embeddings_and_padding_mask() -> None:
    contexts = [
        {
            "prefix_embs": torch.arange(16, dtype=torch.float32).reshape(1, 4, 4),
            "prefix_attention_mask": torch.tensor(
                [[[[0.0, 0.0, 0.0, -1.0]] * 4]],
                dtype=torch.float32,
            ),
            "prefix_position_ids": torch.tensor([[0, 1, 2, 2]], dtype=torch.int32),
            "valid_prefix_length": 3,
        }
    ]
    template_inputs = [
        torch.zeros(1, 4, 4),
        torch.ones(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, 1, 1, 8),
    ]

    batches = _build_pi05_gemma_calibration_inputs(
        contexts=contexts,
        template_inputs=template_inputs,
        cache_length=8,
        device=torch.device("cpu"),
    )

    assert len(batches) == 1
    torch.testing.assert_close(batches[0][0], contexts[0]["prefix_embs"].half())
    assert batches[0][1].item() == 0
    assert batches[0][2].item() == 3
    assert batches[0][3].shape == (1, 1, 1, 8)
    assert torch.all(batches[0][3][..., :3] == 0)
    assert torch.all(batches[0][3][..., 3:] < 0)


def test_calibration_context_packs_selected_images_before_language_padding() -> None:
    policy = nn.Module()
    policy.config = SimpleNamespace(
        image_resolution=(2, 2),
        image_features=["base", "left_wrist", "right_wrist"],
        tokenizer_max_length=4,
    )
    policy.model = _CompactPrefixPolicyModel()

    contexts = _build_pi05_calibration_contexts(
        policy=policy,
        tokenizer=_CompactTokenizer(),
        prefix_sequence_length=516,
        selected_image_indices=[1, 0],
        text_max_length=4,
    )

    assert len(contexts) == 3
    context = contexts[0]
    assert context["prefix_embs"].shape == (1, 516, 4)
    assert context["valid_prefix_length"] == 514
    assert torch.all(context["prefix_embs"][:, :256] == 2)
    assert torch.all(context["prefix_embs"][:, 256:512] == 1)
    assert torch.all(context["prefix_embs"][:, 512:] == 9)
    assert bool(context["prefix_pad_masks"][:, :514].all())
    assert not bool(context["prefix_pad_masks"][:, 514:].any())


@pytest.mark.parametrize(
    ("relative_path", "image_indices", "prefix_length", "horizon"),
    [
        ("droid/pi05_droid.yaml", [0, 1], 712, 50),
        ("droid/pi05_droid_openpi_h15.yaml", [0, 1], 712, 15),
        ("droid/pi05_droid_customer_h50.yaml", [0, 1], 712, 50),
        ("libero/pi05_libero.yaml", [0, 1], 712, 50),
        ("libero/pi05_libero_openpi_h10.yaml", [0, 1], 712, 10),
        ("libero/pi05_libero_lerobot_h50.yaml", [0, 1], 712, 50),
        ("aloha/pi05_aloha_openpi_h50.yaml", [0, 1, 2], 968, 50),
    ],
)
def test_compact_variant_contracts(
    relative_path: str,
    image_indices: list[int],
    prefix_length: int,
    horizon: int,
) -> None:
    config_root = (
        Path(__file__).resolve().parents[1]
        / "configs_merak/workflows/xh2a/other_models/pi05"
    )
    export_cfg = WorkflowConfig.from_file(str(config_root / relative_path)).build_export_dict()

    assert export_cfg["config_dir"] is None
    contract = _build_pi05_runtime_contract(export_cfg)

    assert contract == {
        "selected_image_indices": image_indices,
        "text_max_length": 200,
        "prefix_sequence_length": prefix_length,
        "action_horizon": horizon,
        "cache_length": 1024,
    }


def test_pi05_config_dir_must_be_supplied_externally() -> None:
    with pytest.raises(ValueError, match="must be supplied externally"):
        _require_pi05_config_dir({"config_dir": None})

    assert _require_pi05_config_dir({"config_dir": "tokenizer"}) == "tokenizer"


def test_compact_graph_signatures_use_one_cache_and_rope_offset() -> None:
    from xhmodel_merak.xh_other_model.models.pi05._llm_model_impl import (
        _GemmaAttention,
        _GemmaModel,
    )

    assert "self.masked_add(attn_weights, attention_mask)" in inspect.getsource(_GemmaAttention.forward)
    _clear_gemma_traceable_modules()
    expert_impl = importlib.import_module(
        "xhmodel_merak.xh_other_model.models.pi05._llm_model_impl_cond"
    )
    expert_attention = expert_impl._GemmaAttention
    expert_model = expert_impl._GemmaModel

    assert "position_ids" not in inspect.signature(_GemmaModel.forward).parameters
    assert "position_seq_length" not in inspect.signature(expert_model.forward).parameters
    assert "self.masked_add(attn_weights, attention_mask)" in inspect.getsource(
        expert_attention.forward
    )


def test_xhquant_masked_add_applies_mask_twice_with_saturation() -> None:
    from xhquant.nn import MaskedAdd

    values = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float16)
    mask = torch.tensor([[[[3.0, torch.finfo(torch.float16).min]]]], dtype=torch.float16)

    output = MaskedAdd()(values, mask)

    torch.testing.assert_close(
        output,
        torch.tensor([[[[7.0, torch.finfo(torch.float16).min]]]], dtype=torch.float16),
    )


@pytest.mark.parametrize(
    ("variant", "metadata_variant", "horizon"),
    [
        ("droid-customer-h50", "droid_customer_h50", 50),
        ("droid-openpi-h15", "droid_openpi_h15", 15),
    ],
)
def test_fae_variant_contracts(variant: str, metadata_variant: str, horizon: int) -> None:
    spec = VARIANTS[variant]

    assert spec.metadata_variant == metadata_variant
    assert spec.contract == {
        "selected_image_indices": [0, 1],
        "text_max_length": 200,
        "prefix_sequence_length": 712,
        "action_horizon": horizon,
        "cache_length": 1024,
    }


def test_fae_validation_acceptance_checks_contract_and_metrics() -> None:
    spec = VARIANTS["droid-openpi-h15"]
    summary = {
        "matched_sample_ids": [0, 1],
        "runtime_contract": {
            "static_prefix_length": 712,
            "selected_image_indices": [0, 1],
            "vision_runs": 2,
            "horizon": 15,
            "noise_sha256": "fixed-noise",
        },
        "aggregate": {
            "num_samples": 2,
            "shape": [15, 8],
            "cosine": {"mean": 0.995},
            "mae": {"mean": 0.016},
        },
    }

    acceptance = validate_summary(
        summary,
        spec,
        sample_ids=[0, 1],
        noise_sha256="fixed-noise",
        min_cosine_mean=0.99,
        max_mae_mean=0.02,
    )

    assert acceptance["passed"] is True
    assert acceptance["observed"]["shape"] == [15, 8]


def test_pi05_runtime_classes_register_for_dynamic_wrapping() -> None:
    from lerobot.policies.pi_gemma import PiGemmaForCausalLM, PiGemmaModel

    from xhmodel_merak.xh_other_model.models.pi05 import _llm_model_impl

    class _Layer(nn.Module):
        pass

    class _Norm(nn.Module):
        pass

    gemma = object.__new__(PiGemmaModel)
    nn.Module.__init__(gemma)
    gemma.layers = nn.ModuleList([_Layer()])
    gemma.norm = _Norm()
    expert = object.__new__(PiGemmaForCausalLM)
    nn.Module.__init__(expert)
    expert.model = gemma
    _llm_model_impl.register_wrap_cls(gemma)
    assert PiGemmaModel in XHLLM_TRACEABLE_MODULES

    _clear_gemma_traceable_modules()
    _llm_model_impl_cond = importlib.import_module(
        "xhmodel_merak.xh_other_model.models.pi05._llm_model_impl_cond"
    )
    _llm_model_impl_cond.register_wrap_cls(expert)

    assert PiGemmaModel in XHLLM_TRACEABLE_MODULES
    assert PiGemmaForCausalLM in XHLLM_TRACEABLE_MODULES
