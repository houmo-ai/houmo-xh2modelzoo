from contextlib import nullcontext
from pathlib import Path

import torch

from xh2_model_zoo.xh_llm import LLMConverter
from xh2_model_zoo.xh_llm.models.qwen2 import Qwen2ConvertConfig, Qwen2Inference

from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger  # isort:skip


def test_export_qwen2():
    hf_model_path = "data/models/Qwen2.5-7B-Instruct"
    import tempfile

    quant_type = "w4a8h1_sefp"
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    config = Qwen2ConvertConfig(
        batch_size=1,
        context_length=2048,
        input_sequence_length=256,
        quant_scheme=quant_scheme,
        quant_weight=None,
    )
    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你多大了？用中文回答。"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "中国首都是哪里？"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "香港特别行政区是哪一年成立的？"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "神舟五号是哪年发射的？"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "中国第一个进入太空的是谁？"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "人类第一个进入太空的是谁？"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "人类是哪年实现载人登月的？"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "中国第一颗卫星的名字是什么？"},
        ],
    ]

    # work_dir = "work_dirs/Qwen2.5-7B-Instruct-XH2a-batch_1-2k-w4a8h0_ssfp"
    # with nullcontext():
    with tempfile.TemporaryDirectory() as work_dir:
        work_dir_path: Path = Path(work_dir)
        xhquant_init(str(work_dir_path / "convert.log"), debug=False)

        LLMConverter.from_pretrained(hf_model_path, None, config, work_dir)
        model_meta_file = work_dir_path / "meta.json"
        assert model_meta_file.exists()

        session = Qwen2Inference(str(model_meta_file))
        batch_size = session.batch_size

        assert len(messages) >= batch_size
        messages = messages[:batch_size]
        ids, text = session.forward(messages)
        ids == torch.tensor([[35946, 101909]], dtype=torch.int64, device=ids.device)


if __name__ == "__main__":
    test_export_qwen2()
