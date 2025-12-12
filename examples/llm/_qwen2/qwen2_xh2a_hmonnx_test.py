import argparse

from xhquant.api import get_root_logger, xhquant_init

from xh_model_zoo.xh_llm.models._qwen2 import Qwen2Inference


def main(args):
    xhquant_init(None, args.debug)
    inference_engine = Qwen2Inference(args.config)
    batch_size = inference_engine.batch_size
    logger = get_root_logger()
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
    assert len(messages) >= batch_size
    messages = messages[:batch_size]
    ids, text = inference_engine.forward(messages)
    logger.info(f"{ids}, {text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="work_dirs/Qwen2.5-3B-Instruct-XH2a-batch_1-2k-w8a8h1_sefp/meta.json",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--execution_device", type=str, default="cuda:0", help="execution device, default is cuda:0")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    main(args)
