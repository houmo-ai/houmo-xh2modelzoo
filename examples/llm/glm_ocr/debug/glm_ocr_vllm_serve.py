# Copyright 2025 HOUMO AI
#
# File: glm_ocr_vllm_serve.py
# Description:
#   Deploy GLM-OCR as a vLLM OpenAI-compatible service, then optionally run a
#   quick OCR test via the exposed /v1/chat/completions endpoint.
#
# Usage:
#   # Start vLLM server (blocking):
#   python examples/llm/glm_ocr/debug/glm_ocr_vllm_serve.py \
#       --model /data02/datasets/GLM-OCR/ \
#       --port 8080 \
#       --api-key my-secret-key
#
#   # Start server + auto-run a test image:
#   python examples/llm/glm_ocr/debug/glm_ocr_vllm_serve.py \
#       --model /data02/datasets/GLM-OCR/ \
#       --port 8080 \
#       --api-key my-secret-key \
#       --test-image examples/llm/glm_ocr/data/img3.png
#
#   # Query the running server from another terminal / script:
#   curl http://localhost:8080/v1/chat/completions \
#       -H "Authorization: Bearer my-secret-key" \
#       -H "Content-Type: application/json" \
#       -d '{
#             "model": "glm-ocr",
#             "messages": [{"role":"user","content":[
#               {"type":"image_url","image_url":{"url":"data:image/png;base64,<B64>"}},
#               {"type":"text","text":"Text Recognition:"}
#             ]}],
#             "max_tokens": 1024
#           }'
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

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# ────────────────────────────────────────────────────────────────────
#  Argument parsing
# ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy GLM-OCR as a vLLM OpenAI-compatible service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model & server ──
    parser.add_argument(
        "--model", type=str, default="/data02/datasets/GLM-OCR/",
        help="HuggingFace model path or hub ID for GLM-OCR",
    )
    parser.add_argument(
        "--served-model-name", type=str, default="glm-ocr",
        help="Model name exposed by the vLLM API",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument(
        "--api-key", type=str, default="EMPTY",
        help="API key required by the OpenAI-compatible endpoint. "
             "Use 'EMPTY' to disable authentication (vLLM default).",
    )

    # ── vLLM tuning ──
    parser.add_argument(
        "--max-model-len", type=int, default=None,
        help="Maximum sequence length (tokens). Leave None for model default.",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="Fraction of GPU memory vLLM may use",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--speculative-config", type=str, default=None,
        help="JSON string for vLLM speculative decoding config, "
             'e.g. \'{"method":"mtp","num_speculative_tokens":3}\'',
    )
    parser.add_argument(
        "--dtype", type=str, default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model data type",
    )
    parser.add_argument(
        "--extra-vllm-args", type=str, default="",
        help="Additional raw arguments passed to `vllm serve` (space-separated)",
    )

    # ── Test client (optional) ──
    parser.add_argument(
        "--test-image", type=str, default=None,
        help="If set, launch the server then run a test OCR request on this image",
    )
    parser.add_argument(
        "--test-prompt", type=str, default="Text Recognition:",
        help="Prompt used for the test OCR request",
    )
    parser.add_argument(
        "--test-max-tokens", type=int, default=1024,
        help="max_tokens for the test request",
    )
    parser.add_argument(
        "--server-ready-timeout", type=int, default=300,
        help="Seconds to wait for the vLLM server to become ready",
    )

    # ── Mode ──
    parser.add_argument(
        "--client-only", action="store_true",
        help="Skip launching the server; only run the test client against an "
             "already-running server (requires --test-image)",
    )

    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────
#  Server launch
# ────────────────────────────────────────────────────────────────────

def build_vllm_command(args: argparse.Namespace) -> list[str]:
    """Build the `vllm serve ...` command."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "serve", args.model,
        "--served-model-name", args.served_model_name,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
    ]

    if args.api_key and args.api_key != "EMPTY":
        cmd += ["--api-key", args.api_key]

    if args.max_model_len is not None:
        cmd += ["--max-model-len", str(args.max_model_len)]

    if args.speculative_config:
        cmd += ["--speculative-config", args.speculative_config]

    if args.extra_vllm_args.strip():
        cmd += args.extra_vllm_args.strip().split()

    return cmd


def launch_server(args: argparse.Namespace) -> subprocess.Popen:
    """Start the vLLM server as a subprocess."""
    cmd = build_vllm_command(args)
    print(f"[glm_ocr_vllm_serve] Launching vLLM server:")
    print(f"  {' '.join(cmd)}\n")
    proc = subprocess.Popen(cmd)
    return proc


def wait_for_server(base_url: str, timeout: int, api_key: str = "EMPTY") -> bool:
    """Poll the /health endpoint until the server is ready."""
    import urllib.request
    import urllib.error
    import ssl

    health_url = f"{base_url}/health"
    deadline = time.time() + timeout
    ctx = ssl.create_default_context()

    print(f"[glm_ocr_vllm_serve] Waiting for server at {health_url} ...", flush=True)
    while time.time() < deadline:
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=5, context=ctx) as resp:
                if resp.status == 200:
                    print("[glm_ocr_vllm_serve] Server is ready.")
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(2)

    print("[glm_ocr_vllm_serve] WARNING: server did not become ready in time.")
    return False


# ────────────────────────────────────────────────────────────────────
#  Test client
# ────────────────────────────────────────────────────────────────────

def encode_image_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def guess_mime(image_path: str) -> str:
    """Guess the MIME type from the file extension."""
    suffix = Path(image_path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }.get(suffix, "image/png")


def run_test_client(
    base_url: str,
    api_key: str,
    model_name: str,
    image_path: str,
    prompt: str,
    max_tokens: int,
) -> str:
    """Send a chat-completion request with an image and return the OCR text."""
    import urllib.request
    import ssl

    b64 = encode_image_base64(image_path)
    mime = guess_mime(image_path)

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
    }

    url = f"{base_url}/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    ctx = ssl.create_default_context()

    print(f"[glm_ocr_vllm_serve] Sending test request to {url} ...")
    with urllib.request.urlopen(req, timeout=300, context=ctx) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    content = body["choices"][0]["message"]["content"]
    return content


# ────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    base_url = f"http://{args.host.replace('0.0.0.0', 'localhost')}:{args.port}"

    # ── Client-only mode: skip server launch ──
    if args.client_only:
        if not args.test_image:
            print("[glm_ocr_vllm_serve] ERROR: --client-only requires --test-image")
            sys.exit(1)
        result = run_test_client(
            base_url=base_url,
            api_key=args.api_key,
            model_name=args.served_model_name,
            image_path=args.test_image,
            prompt=args.test_prompt,
            max_tokens=args.test_max_tokens,
        )
        print("\n" + "=" * 60)
        print("OCR Result:")
        print("=" * 60)
        print(result)
        return

    # ── Start server ──
    server_proc = launch_server(args)

    try:
        ready = wait_for_server(
            base_url=base_url,
            timeout=args.server_ready_timeout,
            api_key=args.api_key,
        )
        if not ready:
            print("[glm_ocr_vllm_serve] Server failed to start, exiting.")
            server_proc.terminate()
            server_proc.wait(timeout=10)
            sys.exit(1)

        # ── Optional test ──
        if args.test_image:
            print()
            result = run_test_client(
                base_url=base_url,
                api_key=args.api_key,
                model_name=args.served_model_name,
                image_path=args.test_image,
                prompt=args.test_prompt,
                max_tokens=args.test_max_tokens,
            )
            print("\n" + "=" * 60)
            print("OCR Result:")
            print("=" * 60)
            print(result)

            # After test, keep server running until Ctrl-C
            print("\n[glm_ocr_vllm_serve] Server is still running. Press Ctrl-C to stop.")

        # Block until interrupted
        server_proc.wait()

    except KeyboardInterrupt:
        print("\n[glm_ocr_vllm_serve] Shutting down server ...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("[glm_ocr_vllm_serve] Server stopped.")


if __name__ == "__main__":
    main()
