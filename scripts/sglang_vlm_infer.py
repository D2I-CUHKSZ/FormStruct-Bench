#!/usr/bin/env python3
"""Single-image FormTSR inference through a local SGLang server.

Designed for formtsr_exp.adapters.LocalHFAdapter. The adapter provides:
  FORMTSR_IMAGE_PATH, FORMTSR_PROMPT, FORMTSR_MODEL

Expected server:
  python -m sglang.launch_server --model-path <model> --host 127.0.0.1 --port 30000

This script calls the OpenAI-compatible endpoint and prints only the raw model
text to stdout, so formtsr_exp.run_main can save and parse it.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one FormTSR VLM request via local SGLang.")
    parser.add_argument("--image", default=os.environ.get("FORMTSR_IMAGE_PATH", ""))
    parser.add_argument("--prompt", default=os.environ.get("FORMTSR_PROMPT", ""))
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--model", default=os.environ.get("FORMTSR_SERVED_MODEL") or os.environ.get("FORMTSR_MODEL", "default"))
    parser.add_argument("--base-url", default=os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000"))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("SGLANG_TEMPERATURE", "0")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("SGLANG_MAX_TOKENS", "4096")))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("SGLANG_TIMEOUT", "600")))
    parser.add_argument("--retries", type=int, default=int(os.environ.get("SGLANG_RETRIES", "0")))
    parser.add_argument("--retry-sleep", type=float, default=float(os.environ.get("SGLANG_RETRY_SLEEP", "5")))
    parser.add_argument("--enable-thinking", default=os.environ.get("SGLANG_ENABLE_THINKING", "false"))
    parser.add_argument("--stop", default=os.environ.get("SGLANG_STOP", ""))
    return parser.parse_args()


def image_data_url(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_payload(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(Path(args.image))}},
                ],
            }
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    enable_thinking = str(args.enable_thinking).strip().lower() in {"1", "true", "yes", "on"}
    payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if args.stop:
        stop_text = args.stop.replace("\\n", "\n")
        payload["stop"] = [item for item in stop_text.split("|") if item]
    return payload


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("SGLang response was not a JSON object")
    return data


def extract_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    chunks: list[str] = []
                    for item in content:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            chunks.append(item["text"])
                    if chunks:
                        return "\n".join(chunks)
                reasoning_content = message.get("reasoning_content")
                if isinstance(reasoning_content, str):
                    return reasoning_content
            if isinstance(first.get("text"), str):
                return first["text"]
    return json.dumps(response, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    if not args.image:
        print("missing --image or FORMTSR_IMAGE_PATH", file=sys.stderr)
        return 2
    if not prompt:
        print("missing --prompt/--prompt-file or FORMTSR_PROMPT", file=sys.stderr)
        return 2

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    payload = build_payload(args, prompt)
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            print(extract_text(post_json(url, payload, args.timeout)))
            return 0
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError, ValueError) as exc:
            last_error = exc
            if attempt < args.retries:
                time.sleep(args.retry_sleep)
                continue
            print(f"SGLang request failed: {exc}", file=sys.stderr)
            return 1
    print(f"SGLang request failed: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
