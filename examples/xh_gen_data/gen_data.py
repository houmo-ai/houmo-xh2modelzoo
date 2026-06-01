# Copyright 2025 HOUMO AI
#
# File: gen_data.py
# Description:
#   Example script: xh_gen_data/gen_data.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import random
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed-length calibration dataset for LLM PTQ."
    )

    # Backward-compatible positional arguments:
    # python gen_data.py <prefix_name> <model_path> <gen_seqlen>
    parser.add_argument("legacy_prefix_name", nargs="?", default=None)
    parser.add_argument("legacy_model_path", nargs="?", default=None)
    parser.add_argument("legacy_gen_seqlen", nargs="?", default=None)

    parser.add_argument("--prefix-name", type=str, default=None, help="Output file name prefix")
    parser.add_argument("--model-path", type=str, default=None, help="HuggingFace model path")
    parser.add_argument("--gen-seqlen", type=int, default=None, help="Target token length of each sample")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="Number of generated calibration samples",
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help="Optional questions file (.txt or .jsonl). If provided, prompts come from this file.",
    )
    parser.add_argument("--seed", type=int, default=1024, help="Random seed")
    parser.add_argument("--num-beams", type=int, default=4, help="Beam size for generation")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Generation device",
    )

    args = parser.parse_args()

    # Resolve final args with backward compatibility.
    if args.prefix_name is None:
        args.prefix_name = args.legacy_prefix_name
    if args.model_path is None:
        args.model_path = args.legacy_model_path
    if args.gen_seqlen is None and args.legacy_gen_seqlen is not None:
        args.gen_seqlen = int(args.legacy_gen_seqlen)

    if args.prefix_name is None or args.model_path is None or args.gen_seqlen is None:
        raise ValueError(
            "Missing required arguments. Use either legacy mode: "
            "gen_data.py <prefix_name> <model_path> <gen_seqlen>, "
            "or explicit mode: --prefix-name --model-path --gen-seqlen"
        )

    return args


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_questions(questions_file: Path) -> List[str]:
    prompts: List[str] = []
    with questions_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if questions_file.suffix.lower() == ".jsonl":
                data = json.loads(line)
                prompt = data.get("question") or data.get("prompt") or data.get("text")
                if prompt:
                    prompts.append(str(prompt))
            else:
                prompts.append(line)

    if not prompts:
        raise ValueError(f"No valid prompts found in: {questions_file}")
    return prompts


def sample_vocab_ids(vocab_size: int, num_samples: int, rng: random.Random) -> List[int]:
    if num_samples <= vocab_size:
        return rng.sample(range(vocab_size), num_samples)
    return [rng.randrange(vocab_size) for _ in range(num_samples)]


def generate_fixed_length_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    target_len: int,
    num_beams: int,
) -> str:
    input_len = input_ids.shape[-1]
    if input_len >= target_len:
        output_ids = input_ids[:, :target_len]
    else:
        max_new_tokens = target_len - input_len
        output_ids = model.generate(
            input_ids,
            do_sample=False,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            repetition_penalty=1.5,
            length_penalty=1.0,
        )
        output_ids = output_ids[:, :target_len]

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    output_path = Path("gen_data") / args.prefix_name / f"{args.gen_seqlen}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading tokenizer/model from: {args.model_path}")
    print(f"[INFO] Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    device_map = "cpu" if device == "cpu" else "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=device_map)
    model.eval()
    runtime_device = next(model.parameters()).device

    prompts: List[str] = []
    if args.questions_file:
        prompts = load_questions(Path(args.questions_file))
        print(f"[INFO] Loaded {len(prompts)} prompts from: {args.questions_file}")

    vocab_ids: List[int] = []
    if not prompts:
        vocab_ids = sample_vocab_ids(model.config.vocab_size, args.num_samples, rng)

    print(f"[INFO] Generating {args.num_samples} samples to: {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        for idx in range(args.num_samples):
            if prompts:
                prompt = prompts[idx % len(prompts)]
                encoded = tokenizer(prompt, return_tensors="pt")
                input_ids = encoded["input_ids"].to(runtime_device)
                show_label = f"prompt#{idx + 1}"
            else:
                token_id = vocab_ids[idx]
                input_ids = torch.tensor([[token_id]], dtype=torch.long, device=runtime_device)
                show_label = f"token_id={token_id}"

            text = generate_fixed_length_text(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                target_len=args.gen_seqlen,
                num_beams=args.num_beams,
            )

            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            print(f"[INFO] {idx + 1}/{args.num_samples} done ({show_label})")

    print("[INFO] Generation finished.")


if __name__ == "__main__":
    main()


