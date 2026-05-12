import torch

from xh_model_zoo.xh_llm.models.qwen3moe.qwen_moe_converter import Qwen3MoeConverterXH2a


def test_unpack_gptq_weight_normalizes_raw_checkpoint_qzeros():
    qweight = torch.tensor([[0]], dtype=torch.int32)
    qzeros = torch.tensor([[0]], dtype=torch.int32)
    scales = torch.ones((8, 1), dtype=torch.float32)
    g_idx = torch.arange(8, dtype=torch.int32)

    unpacked_default = Qwen3MoeConverterXH2a._unpack_gptq_weight(qweight, qzeros, scales, g_idx)
    unpacked_raw = Qwen3MoeConverterXH2a._unpack_gptq_weight(
        qweight,
        qzeros,
        scales,
        g_idx,
        raw_checkpoint_qzeros=True,
    )

    assert unpacked_default.shape == (1, 8)
    assert torch.equal(unpacked_default, torch.zeros((1, 8), dtype=torch.float32))
    assert torch.equal(unpacked_raw, -torch.ones((1, 8), dtype=torch.float32))