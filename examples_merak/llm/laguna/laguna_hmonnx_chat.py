"""Chat with an exported Laguna HMONNX model using xhquanttool only.

The release directory must contain ``golden_meta_info.json``, prefill/decode
HMONNX graphs, ``quant_embedding.pt``, and the packaged tokenizer files.

Examples:
    python examples_merak/llm/laguna/laguna_hmonnx_chat.py /path/to/hmquant_laguna \
        --devices cuda:0,cuda:1,cuda:2,cuda:3

    python examples_merak/llm/laguna/laguna_hmonnx_chat.py /path/to/hmquant_laguna \
        --devices cuda:0,cuda:1,cuda:2,cuda:3 \
        --prompt "你好，请介绍一下你自己。" \
        --prompt "用一句话概括刚才的回答。"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast

from xhquant.core.cache_tensor import CacheTensor
from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2
from xhquant.xhonnxruntime.llm_hmonnx_loader import LLMHMONNXLoader


@dataclass(frozen=True)
class LagunaRelease:
    prefill_hmonnx: Path
    decode_hmonnx: Path
    hf_config: Path
    quant_embedding: Path
    pad_token_id: int
    eos_token_ids: tuple[int, ...]
    context_length: int
    prefill_length: int
    num_layers: int
    cache_shape: tuple[int, ...]


def _parse_devices(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("--devices must contain at least one device")
    return devices


def _resolve_release(model_dir: Path) -> LagunaRelease:
    meta_file = model_dir if model_dir.is_file() else model_dir / "golden_meta_info.json"
    if not meta_file.is_file():
        raise FileNotFoundError(f"Laguna HMONNX metadata not found: {meta_file}")

    root = meta_file.resolve().parent
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    model_config = meta["model_config"]
    kv_cache = meta["kv_cache"]
    hf_config = root / meta["hf_config"]
    generation_config = json.loads((hf_config / "generation_config.json").read_text(encoding="utf-8"))
    eos_token_id = generation_config["eos_token_id"]
    if isinstance(eos_token_id, int):
        eos_token_ids = (eos_token_id,)
    else:
        eos_token_ids = tuple(int(token_id) for token_id in eos_token_id)

    release = LagunaRelease(
        prefill_hmonnx=root / meta["prefill_hmonnx"],
        decode_hmonnx=root / meta["decode_hmonnx"],
        hf_config=hf_config,
        quant_embedding=root / meta["quant_embedding"],
        pad_token_id=int(meta["pad_token_id"]),
        eos_token_ids=eos_token_ids,
        context_length=int(model_config["context_max_length"]),
        prefill_length=int(model_config["prefill_chunk_length"]),
        num_layers=int(kv_cache["num_layers"]),
        cache_shape=tuple(int(dim) for dim in kv_cache["kv_cache_shape"]),
    )
    for required_file in (
        release.prefill_hmonnx,
        release.decode_hmonnx,
        release.quant_embedding,
        release.hf_config / "tokenizer.json",
        release.hf_config / "chat_template.jinja",
    ):
        if not required_file.is_file():
            raise FileNotFoundError(f"Required Laguna release file not found: {required_file}")
    return release


def _load_tokenizer(release: LagunaRelease) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(release.hf_config / "tokenizer.json"),
        bos_token="〈|EOS|〉",
        eos_token="〈|EOS|〉",
        unk_token="〈|UNK|〉",
        pad_token="〈|PAD|〉",
        cls_token="〈|CLS|〉",
        sep_token="〈|SEP|〉",
        mask_token="〈|MASK|〉",
    )
    tokenizer.chat_template = (release.hf_config / "chat_template.jinja").read_text(encoding="utf-8")
    return tokenizer


def _load_embedding(path: Path, device: torch.device) -> nn.Embedding:
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    embedding = nn.Embedding.from_pretrained(state_dict["weight"], freeze=True)
    return embedding.eval().to(device=device, dtype=torch.float16)


class LagunaHMONNXChat:
    def __init__(self, release: LagunaRelease, devices: list[str]) -> None:
        self.release = release
        exec_devices = [torch.device(device) for device in devices]
        if any(device.type != "cuda" for device in exec_devices):
            raise ValueError("Laguna HMONNX chat currently requires CUDA devices")

        loader = LLMHMONNXLoader(str(release.prefill_hmonnx), str(release.decode_hmonnx))
        prefill_config = HMONNXInferenceConfig()
        prefill_config.enable_auto_offload = len(exec_devices) > 1
        prefill_config.exec_devices = exec_devices
        self.prefill_session = HMONNXInferenceV2.from_onnx_graph(
            str(release.prefill_hmonnx), loader.prefill_graph, prefill_config
        )

        decode_config = HMONNXInferenceConfig()
        decode_config.enable_auto_offload = len(exec_devices) > 1
        decode_config.exec_devices = exec_devices
        decode_config.layers = self.prefill_session.get_layer_infos()
        self.decode_session = HMONNXInferenceV2.from_onnx_graph(
            str(release.decode_hmonnx), loader.decode_graph, decode_config
        )
        self.embedding = _load_embedding(release.quant_embedding, self.prefill_session.device)
        self.key_caches = [
            CacheTensor(torch.zeros(release.cache_shape, dtype=torch.float16)) for _ in range(release.num_layers)
        ]
        self.value_caches = [
            CacheTensor(torch.zeros(release.cache_shape, dtype=torch.float16)) for _ in range(release.num_layers)
        ]

    def _reset_caches(self) -> None:
        for cache in (*self.key_caches, *self.value_caches):
            cache.reset()

    def _feed(
        self,
        input_ids: torch.Tensor,
        past_length: int,
        current_length: int,
    ) -> dict[str, torch.Tensor]:
        embedding_device = self.embedding.weight.device
        input_ids = input_ids.to(embedding_device)
        inputs_embeds = self.embedding(input_ids).to(torch.float16)
        feed: dict[str, torch.Tensor] = {
            "input_1": inputs_embeds,
            "valid_length": torch.tensor([past_length], dtype=torch.int32, device=embedding_device),
            "current_length": torch.tensor([current_length], dtype=torch.int32, device=embedding_device),
        }
        for layer_index, cache in enumerate(self.key_caches):
            feed[f"model_layers_{layer_index}_self_attn_kcache_input"] = cache
        for layer_index, cache in enumerate(self.value_caches):
            feed[f"model_layers_{layer_index}_self_attn_vcache_input"] = cache
        return feed

    @torch.inference_mode()
    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        prompt_length = len(prompt_ids)
        if prompt_length > self.release.prefill_length:
            raise ValueError(
                f"Rendered prompt has {prompt_length} tokens, but this export supports at most "
                f"{self.release.prefill_length} prefill tokens"
            )
        if prompt_length + max_new_tokens > self.release.context_length:
            raise ValueError(
                f"Prompt plus generation exceeds context length {self.release.context_length}: "
                f"{prompt_length} + {max_new_tokens}"
            )

        self._reset_caches()
        padded_ids = prompt_ids + [self.release.pad_token_id] * (self.release.prefill_length - prompt_length)
        prefill_ids = torch.tensor([padded_ids], dtype=torch.long)
        logits = self.prefill_session.run(self._feed(prefill_ids, 0, prompt_length))
        next_token_id = int(torch.argmax(logits[0, -1]).item())
        generated_ids: list[int] = []

        for _ in range(max_new_tokens):
            if next_token_id in self.release.eos_token_ids:
                break
            generated_ids.append(next_token_id)
            past_length = prompt_length + len(generated_ids) - 1
            decode_ids = torch.tensor([[next_token_id]], dtype=torch.long)
            logits = self.decode_session.run(self._feed(decode_ids, past_length, 1))
            next_token_id = int(torch.argmax(logits[0, -1]).item())
        return generated_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a Laguna HMONNX export using xhquanttool only.")
    parser.add_argument("model_dir", type=Path, help="HMQuant release directory or golden_meta_info.json path.")
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=_parse_devices("cuda:0,cuda:1,cuda:2,cuda:3"),
        help="Comma-separated CUDA devices used by HMONNX auto-offload.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to run. Repeat for scripted multi-turn chat; omit for interactive chat.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Laguna HMONNX chat requires CUDA")

    release = _resolve_release(args.model_dir)
    tokenizer = _load_tokenizer(release)
    model = LagunaHMONNXChat(release, args.devices)
    messages: list[dict[str, str]] = []

    scripted_prompts = iter(args.prompt)
    interactive = not args.prompt
    while True:
        if interactive:
            try:
                prompt = input("\nUser> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt:
                continue
            if prompt.lower() in {"exit", "quit", "/exit", "/quit"}:
                break
        else:
            try:
                prompt = next(scripted_prompts)
            except StopIteration:
                break
            print(f"\nUser> {prompt}")

        messages.append({"role": "user", "content": prompt})
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        output_ids = model.generate(prompt_ids, args.max_new_tokens)
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        print(f"Assistant> {response}")
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
