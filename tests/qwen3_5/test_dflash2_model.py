import inspect
import json
from pathlib import Path

import pytest
import torch

from xhmodel_merak.xh_llm.models.qwen3_5._dflash_model_impl import (
    DFlash2CandidateSelector,
    DFlashMLP,
    GroupedDynamicCausalConv,
    _grouped_dynamic_convolve,
)
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_dflash_model import (
    XHQwen3_5DFlashDraftModel,
    _build_dflash_export_adapter,
)
from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import (
    XHQwen3_5_DFlashConfig,
    XHQwen3_5ModelConfig,
)
from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_dflash2_config(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DFlash2DraftModel"],
                "is_causal": False,
                "hidden_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "vocab_size": 97,
                "sliding_window": 16,
                "input_embedding_scale": 1.25,
                "output_multiplier": 1.5,
                "final_logit_softcapping": 8.0,
                "dflash_config": {
                    "block_size": 8,
                    "conv_kernel_size": 2,
                    "conv_group_size": 8,
                    "selector_rank": 6,
                    "selector_top_k": 5,
                    "mask_token_id": 91,
                    "target_layer_ids": [1, 3],
                },
            }
        ),
        encoding="utf-8",
    )


def test_dflash2_grouped_dynamic_conv_matches_full_block_reference():
    torch.manual_seed(0)
    batch_size, block_size, num_groups, group_size, taps = 2, 8, 3, 4, 2
    hidden_size = num_groups * group_size
    hidden = torch.randn(batch_size, block_size, hidden_size)
    dynamic = torch.randn(
        batch_size,
        block_size,
        taps,
        num_groups,
    )
    base = torch.randn(taps, hidden_size)

    actual = _grouped_dynamic_convolve(
        hidden,
        dynamic,
        base,
        group_size,
        num_groups,
        taps,
    )

    grouped = hidden.reshape(
        batch_size,
        block_size,
        num_groups,
        group_size,
    )
    expected = torch.zeros_like(grouped)
    base_grouped = base.reshape(taps, num_groups, group_size)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base_grouped[tap]
                + dynamic[:, position, tap].unsqueeze(-1)
            ) * grouped[:, position - tap]

    torch.testing.assert_close(actual, expected.reshape_as(hidden))
    # The anchor is row zero. It sees zero padding, while row one consumes it.
    torch.testing.assert_close(
        actual[:, 0].reshape(batch_size, num_groups, group_size),
        (
            base_grouped[0]
            + dynamic[:, 0, 0].unsqueeze(-1)
        )
        * grouped[:, 0],
    )


def test_dflash2_scheme_b_scores_and_walk_match_sequential_reference():
    torch.manual_seed(1)
    batch_size, steps, hidden_size, vocab_size = 2, 5, 12, 31
    rank, top_k = 7, 4
    selector = DFlash2CandidateSelector(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        rank=rank,
        top_k=top_k,
    )
    hidden = torch.randn(batch_size, steps, hidden_size)
    logits = torch.randn(batch_size, steps, vocab_size)
    anchors = torch.randint(vocab_size, (batch_size,))

    candidate_ids, first_scores, transition_scores = selector(
        hidden,
        logits,
        anchors,
    )
    unary = logits.gather(-1, candidate_ids)
    projected = selector.hidden_projection(hidden)
    successor = selector.successor_codebook(candidate_ids)
    predecessor = selector.predecessor_codebook(candidate_ids)
    anchor_predecessor = selector.predecessor_codebook(anchors)
    expected_first = unary[:, 0] + torch.einsum(
        "br,bkr->bk",
        anchor_predecessor * projected[:, 0],
        successor[:, 0],
    )
    expected_transition = unary[:, 1:].unsqueeze(2) + torch.einsum(
        "blpr,blcr->blpc",
        predecessor[:, :-1] * projected[:, 1:].unsqueeze(2),
        successor[:, 1:],
    )

    torch.testing.assert_close(first_scores, expected_first)
    torch.testing.assert_close(transition_scores, expected_transition)

    scheme_b_indices = [first_scores.argmax(-1)]
    for step in range(steps - 1):
        current_row = transition_scores[:, step].gather(
            1,
            scheme_b_indices[-1].reshape(-1, 1, 1).expand(-1, 1, top_k),
        )[:, 0]
        scheme_b_indices.append(current_row.argmax(-1))
    scheme_b_tokens = torch.stack(
        [
            candidate_ids[:, step].gather(
                -1,
                index.unsqueeze(-1),
            )[:, 0]
            for step, index in enumerate(scheme_b_indices)
        ],
        dim=1,
    )

    predecessor_token = anchors
    sequential_tokens = []
    for step in range(steps):
        scores = unary[:, step] + torch.einsum(
            "br,bkr->bk",
            selector.predecessor_codebook(predecessor_token)
            * projected[:, step],
            successor[:, step],
        )
        index = scores.argmax(-1)
        predecessor_token = candidate_ids[:, step].gather(
            -1,
            index.unsqueeze(-1),
        )[:, 0]
        sequential_tokens.append(predecessor_token)
    sequential_tokens = torch.stack(sequential_tokens, dim=1)

    torch.testing.assert_close(scheme_b_tokens, sequential_tokens)


def test_dflash2_scheme_b_walk_is_invariant_to_candidate_permutation():
    """HMONNX's required sorted TopK is only a selector-axis permutation."""

    torch.manual_seed(2)
    batch_size, steps, hidden_size, vocab_size = 2, 4, 12, 37
    rank, top_k = 7, 5
    selector = DFlash2CandidateSelector(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        rank=rank,
        top_k=top_k,
    )
    hidden = torch.randn(batch_size, steps, hidden_size)
    logits = torch.randn(batch_size, steps, vocab_size)
    anchors = torch.randint(vocab_size, (batch_size,))
    candidate_ids, first_scores, transition_scores = selector(
        hidden,
        logits,
        anchors,
    )

    def walk(candidates, first, transitions):
        selected = first.argmax(-1)
        tokens = [candidates[:, 0].gather(-1, selected[:, None])]
        for step in range(1, steps):
            row = transitions[:, step - 1].gather(
                1,
                selected[:, None, None].expand(-1, 1, top_k),
            )[:, 0]
            selected = row.argmax(-1)
            tokens.append(candidates[:, step].gather(-1, selected[:, None]))
        return torch.cat(tokens, dim=-1)

    permutations = torch.stack(
        [torch.randperm(top_k) for _ in range(steps)],
    )
    permuted_candidates = torch.stack(
        [candidate_ids[:, step, permutations[step]] for step in range(steps)],
        dim=1,
    )
    permuted_first = first_scores[:, permutations[0]]
    permuted_transitions = torch.stack(
        [
            transition_scores[:, step - 1][
                :, permutations[step - 1]
            ][:, :, permutations[step]]
            for step in range(1, steps)
        ],
        dim=1,
    )
    torch.testing.assert_close(
        walk(candidate_ids, first_scores, transition_scores),
        walk(
            permuted_candidates,
            permuted_first,
            permuted_transitions,
        ),
    )


class _DecodeCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.export_prepare_calls = 0

    def prepare_for_export(self):
        self.export_prepare_calls += 1

    def forward_decode(
        self,
        noise_embedding,
        past_seq_length,
        current_input_length,
        attn_mask,
        anchor_token_id,
        query_kv_range_abs,
        *cache_tensors,
    ):
        return (
            noise_embedding,
            attn_mask,
            anchor_token_id,
            query_kv_range_abs,
            *cache_tensors,
        )


def test_dflash2_nonflash_export_adapter_keeps_anchor_and_dense_mask():
    core = _DecodeCore()
    adapter = _build_dflash_export_adapter(
        core,
        mode="decode",
        num_hidden_layers=1,
        use_flash_attention=False,
        is_dflash2=True,
    )
    assert core.export_prepare_calls == 1
    assert list(inspect.signature(adapter.forward).parameters) == [
        "noise_embedding",
        "past_seq_length",
        "current_input_length",
        "anchor_token_id",
        "attn_mask",
        "past_key_cache_0",
        "past_value_cache_0",
    ]

    values = [torch.tensor(index) for index in range(7)]
    output = adapter(*values)
    assert output[1] is values[4]
    assert output[2] is values[3]
    assert output[3] is None
    assert output[4:] == tuple(values[5:])


def test_dflash2_flash_export_adapter_uses_query_dependent_ranges():
    core = _DecodeCore()
    adapter = _build_dflash_export_adapter(
        core,
        mode="decode",
        num_hidden_layers=1,
        use_flash_attention=True,
        is_dflash2=True,
    )
    assert core.export_prepare_calls == 1
    assert list(inspect.signature(adapter.forward).parameters) == [
        "noise_embedding",
        "past_seq_length",
        "current_input_length",
        "anchor_token_id",
        "query_kv_range_abs",
        "past_key_cache_0",
        "past_value_cache_0",
    ]

    values = [torch.tensor(index) for index in range(7)]
    output = adapter(*values)
    assert output[1] is None
    assert output[2] is values[3]
    assert output[3] is values[4]
    assert output[4:] == tuple(values[5:])


def test_dflash2_config_uses_checkpoint_block_and_static_decode_abi(tmp_path):
    assistant_dir = tmp_path / "dflash2"
    _write_dflash2_config(assistant_dir)
    config = XHQwen3_5ModelConfig(
        model_name="qwen3_8_27b",
        hf_model="weights/target",
        spec_decode_mode="dflash",
        dflash_config={"hf_model": str(assistant_dir)},
    )

    assert config.num_draft_tokens == 7
    assert config.dflash_config.architecture == "DFlash2DraftModel"
    assert config.dflash_config.sliding_window == 16
    assert config.dflash_config.selector_top_k == 5
    assert config.dflash_config.input_embedding_scale == 1.25
    assert config.dflash_config.output_multiplier == 1.5
    assert config.dflash_config.final_logit_softcapping == 8.0
    assert config.dflash_config.activation_residual_scale == 128.0

    decode_config = XHQwen3_5_DFlashConfig(
        model_name="qwen3_8_27b_dflash2",
        hf_model=str(assistant_dir),
        target_model_dir="weights/target",
        dtype="bfloat16",
        mode="decode",
        input_sequence_length=8,
        max_sequence_length=64,
    )
    draft_model = XHQwen3_5DFlashDraftModel(decode_config)
    dummy = draft_model.get_dummy_inputs()
    assert tuple(dummy["anchor_token_id"].shape) == (1,)
    assert tuple(dummy["attn_mask"].shape) == (1, 8, 64)
    assert dummy["noise_embedding"].dtype is torch.bfloat16
    assert dummy["attn_mask"].dtype is torch.bfloat16
    assert dummy["past_key_cache_0"].dtype is torch.bfloat16
    assert dummy["past_value_cache_0"].dtype is torch.bfloat16
    assert draft_model.get_export_cfg()["output_names"] == [
        "candidate_ids",
        "selector_first_scores",
        "selector_transition_scores",
    ]


def test_dflash2_release_workflows_enable_all_gdr_paths():
    cases = {
        "qwen3_8_27b_full_dflash2_8k.yaml": (8192, False),
        "qwen3_8_27b_full_dflash2_256k_fa.yaml": (262144, True),
    }
    config_dir = (
        REPO_ROOT
        / "configs_merak/workflows/xh2a/llm_models/qwen3_5/27b"
    )
    for filename, (context_length, flash_enabled) in cases.items():
        workflow = WorkflowConfig.from_file(str(config_dir / filename))
        model = workflow.export["model"]
        assert model["context_max_length"] == context_length
        assert model["flash_attention"]["enable"] is flash_enabled
        assert model["fuse_gdr_ops"] is True
        assert model["fuse_gdr_block_recurrent_ops"] is True
        assert model["num_draft_tokens"] == 7
        assert model["dflash_config"]["dtype"] == "bfloat16"
        assert model["dflash_config"]["activation_residual_scale"] == 128


def test_dflash2_mlp_scales_before_down_projection():
    torch.manual_seed(3)
    reference = DFlashMLP(16, 32, branch_output_scale=1.0)
    scaled = DFlashMLP(16, 32, branch_output_scale=128.0)
    scaled.load_state_dict(reference.state_dict())
    hidden = torch.randn(2, 8, 16)
    torch.testing.assert_close(
        scaled(hidden),
        reference(hidden) / 128.0,
        rtol=1e-5,
        atol=1e-6,
    )


def test_dflash2_rejects_non_power_of_two_activation_scale(tmp_path):
    assistant_dir = tmp_path / "dflash2"
    _write_dflash2_config(assistant_dir)
    with pytest.raises(ValueError, match="activation_residual_scale"):
        XHQwen3_5_DFlashConfig(
            model_name="qwen3_8_27b_dflash2",
            hf_model=str(assistant_dir),
            target_model_dir="weights/target",
            activation_residual_scale=100.0,
        )


def test_dflash2_export_preparation_preserves_conv_exactly():
    torch.manual_seed(4)
    conv = GroupedDynamicCausalConv(
        hidden_size=12,
        kernel_size=2,
        group_size=4,
    )
    with torch.no_grad():
        conv.base_kernel.normal_()
    hidden = torch.randn(2, 8, 12)
    branch = torch.randn(2, 8, 12)

    original_kernel = conv.base_kernel.detach().clone()
    expected_pre, expected_dynamic = conv.prepare(hidden)
    expected_output = conv.finish(branch, expected_dynamic)

    conv.prepare_for_export()
    conv.prepare_for_export()

    assert isinstance(conv.base_kernel, torch.nn.ModuleList)
    assert len(conv.base_kernel) == 2
    for branch_index in range(2):
        assert isinstance(
            conv.base_kernel[branch_index],
            torch.nn.ParameterList,
        )
        for tap in range(2):
            prepared = conv.base_kernel[branch_index][tap]
            assert tuple(prepared.shape) == (1, 1, 3, 4)
            assert torch.equal(
                prepared,
                original_kernel[branch_index, tap].reshape(1, 1, 3, 4),
            )

    actual_pre, actual_dynamic = conv.prepare(hidden)
    actual_output = conv.finish(branch, actual_dynamic)
    assert torch.equal(actual_pre, expected_pre)
    assert torch.equal(actual_dynamic, expected_dynamic)
    assert torch.equal(actual_output, expected_output)
