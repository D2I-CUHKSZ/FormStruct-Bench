#!/usr/bin/env python3
"""Batch FormTSR VLM inference through SGLang Offline Engine.

The runner provides FORMTSR_BATCH as a JSON list of:
  {"sample_id": "...", "image_path": "..."}

The script prints a JSON list of:
  {"sample_id": "...", "status": "ok", "raw_response": "..."}

It uses SGLang native offline batching instead of launching one subprocess per
sample or sending concurrent HTTP requests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batch of FormTSR VLM requests with SGLang Offline Engine.")
    parser.add_argument("--model-path", default=os.environ.get("SGLANG_MODEL_PATH", ""))
    parser.add_argument("--model", default=os.environ.get("FORMTSR_MODEL", ""))
    parser.add_argument("--chat-template", default=os.environ.get("SGLANG_CHAT_TEMPLATE", "qwen2-vl"))
    parser.add_argument("--tp-size", type=int, default=int(os.environ.get("SGLANG_TP_SIZE", "1")))
    parser.add_argument("--mem-fraction-static", type=float, default=float(os.environ.get("SGLANG_MEM_FRACTION_STATIC", "0.92")))
    parser.add_argument("--kv-cache-dtype", default=os.environ.get("SGLANG_KV_CACHE_DTYPE", "auto"))
    parser.add_argument("--context-length", type=int, default=int(os.environ.get("SGLANG_CONTEXT_LENGTH", "16384")))
    parser.add_argument("--max-total-tokens", type=int, default=int(os.environ.get("SGLANG_MAX_TOTAL_TOKENS", "32768")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("SGLANG_TEMPERATURE", "0")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("SGLANG_MAX_TOKENS", "4096")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("FORMTSR_BATCH_SIZE", "1")))
    parser.add_argument("--stop", default=os.environ.get("SGLANG_STOP", ""))
    parser.add_argument("--attention-backend", default=os.environ.get("SGLANG_ATTENTION_BACKEND", "triton"))
    parser.add_argument("--sampling-backend", default=os.environ.get("SGLANG_SAMPLING_BACKEND", "pytorch"))
    parser.add_argument("--submit-mode", choices=["batch", "async"], default=os.environ.get("SGLANG_SUBMIT_MODE", "batch"))
    parser.add_argument("--max-inflight-requests", type=int, default=int(os.environ.get("SGLANG_MAX_INFLIGHT_REQUESTS", "0")))
    parser.add_argument("--max-running-requests", type=int, default=int(os.environ.get("SGLANG_MAX_RUNNING_REQUESTS", "0")))
    parser.add_argument("--max-queued-requests", type=int, default=int(os.environ.get("SGLANG_MAX_QUEUED_REQUESTS", "0")))
    parser.add_argument("--chunked-prefill-size", type=int, default=int(os.environ.get("SGLANG_CHUNKED_PREFILL_SIZE", "0")))
    parser.add_argument("--max-prefill-tokens", type=int, default=int(os.environ.get("SGLANG_MAX_PREFILL_TOKENS", "0")))
    parser.add_argument("--schedule-conservativeness", type=float, default=float(os.environ.get("SGLANG_SCHEDULE_CONSERVATIVENESS", "1.0")))
    parser.add_argument("--cuda-graph-max-bs", type=int, default=int(os.environ.get("SGLANG_CUDA_GRAPH_MAX_BS", "0")))
    parser.add_argument("--disable-cuda-graph", default=os.environ.get("SGLANG_DISABLE_CUDA_GRAPH", "false"))
    parser.add_argument("--disable-overlap-schedule", default=os.environ.get("SGLANG_DISABLE_OVERLAP_SCHEDULE", "false"))
    parser.add_argument("--log-level", default=os.environ.get("SGLANG_LOG_LEVEL", "error"))
    parser.add_argument("--trust-remote-code", default=os.environ.get("SGLANG_TRUST_REMOTE_CODE", "true"))
    parser.add_argument("--enable-thinking", default=os.environ.get("SGLANG_ENABLE_THINKING", "false"))
    parser.add_argument(
        "--keep-mm-feature-on-device",
        default=os.environ.get("SGLANG_KEEP_MM_FEATURE_ON_DEVICE", "true"),
        help="Keep multimodal feature tensors on GPU to avoid large /dev/shm transfers.",
    )
    return parser.parse_args()


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_batch() -> list[dict[str, Any]]:
    raw = os.environ.get("FORMTSR_BATCH", "")
    if not raw:
        raise ValueError("missing FORMTSR_BATCH")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("FORMTSR_BATCH must be a JSON list")
    rows: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each FORMTSR_BATCH item must be an object")
        sample_id = item.get("sample_id")
        image_path = item.get("image_path")
        if not isinstance(sample_id, str) or not isinstance(image_path, str):
            raise ValueError("each FORMTSR_BATCH item requires sample_id and image_path strings")
        rows.append({"sample_id": sample_id, "image_path": image_path})
    return rows


def extract_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "output_text", "content"):
            value = item.get(key)
            if isinstance(value, str):
                return value
        if "choices" in item and isinstance(item["choices"], list) and item["choices"]:
            first = item["choices"][0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(first.get("text"), str):
                    return first["text"]
    return json.dumps(item, ensure_ascii=False)


def build_prompts_with_sglang_template(prompt: str, batch_size: int, chat_template_name: str) -> list[str]:
    from sglang.srt.parser.conversation import chat_templates

    template = chat_templates.get(chat_template_name)
    if template is None:
        known = ", ".join(sorted(chat_templates.keys()))
        raise ValueError(f"unknown SGLang chat template {chat_template_name!r}; known templates: {known}")

    prompts: list[str] = []
    for _ in range(batch_size):
        conv = template.copy()
        conv.append_message(conv.roles[0], conv.image_token + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        prompts.append(conv.get_prompt())
    return prompts


def build_prompts_with_hf_template(prompt: str, batch_size: int, model_path: str, enable_thinking: bool) -> list[str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for _ in range(batch_size)
    ]


def build_prompts(prompt: str, batch_size: int, model_path: str, chat_template_name: str, enable_thinking: bool) -> list[str]:
    if os.environ.get("SGLANG_USE_HF_CHAT_TEMPLATE", "true").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            return build_prompts_with_hf_template(prompt, batch_size, model_path, enable_thinking)
        except Exception as exc:
            if os.environ.get("SGLANG_REQUIRE_HF_CHAT_TEMPLATE", "false").strip().lower() in {"1", "true", "yes", "on"}:
                raise
            print(f"HF chat template failed, falling back to SGLang template: {exc}", file=sys.stderr)
    return build_prompts_with_sglang_template(prompt, batch_size, chat_template_name)


def build_engine_kwargs(args: argparse.Namespace, model_path: str) -> dict[str, Any]:
    engine_kwargs: dict[str, Any] = {
        "model_path": model_path,
        "chat_template": args.chat_template,
        "tp_size": args.tp_size,
        "trust_remote_code": as_bool(args.trust_remote_code),
        "mem_fraction_static": args.mem_fraction_static,
        "kv_cache_dtype": args.kv_cache_dtype,
        "context_length": args.context_length,
        "max_total_tokens": args.max_total_tokens,
        "enable_multimodal": True,
        "keep_mm_feature_on_device": as_bool(args.keep_mm_feature_on_device),
        "attention_backend": args.attention_backend,
        "sampling_backend": args.sampling_backend,
        "schedule_conservativeness": args.schedule_conservativeness,
        "disable_cuda_graph": as_bool(args.disable_cuda_graph),
        "disable_overlap_schedule": as_bool(args.disable_overlap_schedule),
        "log_level": args.log_level,
    }
    optional_ints = {
        "max_running_requests": args.max_running_requests,
        "max_queued_requests": args.max_queued_requests,
        "chunked_prefill_size": args.chunked_prefill_size,
        "max_prefill_tokens": args.max_prefill_tokens,
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
    }
    for key, value in optional_ints.items():
        if value > 0:
            engine_kwargs[key] = value
    return engine_kwargs


async def generate_async_queue(
    llm: Any,
    rows: list[dict[str, Any]],
    prompts: list[str],
    sampling_params: dict[str, Any],
    max_inflight: int,
    result_jsonl_path: Path | None = None,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any] | None] = [None] * len(rows)
    semaphore = asyncio.Semaphore(max(1, max_inflight))
    write_lock = asyncio.Lock()
    started = time.monotonic()

    fsync_results = os.environ.get("FORMTSR_RESULT_FSYNC", "false").strip().lower() in {"1", "true", "yes", "on"}

    async def emit(result: dict[str, Any]) -> None:
        if result_jsonl_path is None:
            return
        line = json.dumps(result, ensure_ascii=False)
        async with write_lock:
            with result_jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                if fsync_results:
                    os.fsync(fh.fileno())

    async def run_one(index: int) -> None:
        async with semaphore:
            row = rows[index]
            try:
                output = await llm.async_generate(
                    prompt=prompts[index],
                    image_data=[str(Path(row["image_path"]).resolve())],
                    sampling_params=sampling_params,
                    rid=str(row["sample_id"]),
                )
                result = {"sample_id": row["sample_id"], "status": "ok", "raw_response": extract_text(output)}
            except Exception as exc:
                result = {"sample_id": row["sample_id"], "status": "error", "raw_response": "", "error": str(exc)}
            outputs[index] = result
            await emit(result)

    tasks = [asyncio.create_task(run_one(index)) for index in range(len(rows))]
    if os.environ.get("SGLANG_PROGRESS", "true").strip().lower() in {"1", "true", "yes", "on"}:
        remaining = set(tasks)
        while remaining:
            done, remaining = await asyncio.wait(remaining, timeout=30, return_when=asyncio.FIRST_COMPLETED)
            completed = len(tasks) - len(remaining)
            elapsed = time.monotonic() - started
            print(f"SGLang async progress: {completed}/{len(tasks)} finished in {elapsed:.1f}s", file=sys.stderr)
    else:
        await asyncio.gather(*tasks)
    return [
        output
        if output is not None
        else {"sample_id": row["sample_id"], "status": "error", "raw_response": "", "error": "missing SGLang output"}
        for row, output in zip(rows, outputs)
    ]


def main() -> int:
    args = parse_args()
    prompt = os.environ.get("FORMTSR_PROMPT", "")
    if not prompt:
        print("missing FORMTSR_PROMPT", file=sys.stderr)
        return 2
    model_path = args.model_path or args.model
    if not model_path:
        print("missing --model-path/SGLANG_MODEL_PATH or FORMTSR_MODEL", file=sys.stderr)
        return 2

    try:
        batch = load_batch()
        from sglang import Engine

        stop = args.stop.replace("\\n", "\n")
        stop_list = [item for item in stop.split("|") if item]
        sampling_params: dict[str, Any] = {
            "temperature": args.temperature,
            "max_new_tokens": args.max_tokens,
        }
        if stop_list:
            sampling_params["stop"] = stop_list

        llm = Engine(**build_engine_kwargs(args, model_path))
        try:
            outputs: list[Any] = []
            batch_size = max(1, args.batch_size)
            enable_thinking = str(args.enable_thinking).lower() in {"1", "true", "yes", "on"}
            if args.submit_mode == "async":
                prompts = build_prompts(prompt, len(batch), model_path, args.chat_template, enable_thinking)
                max_inflight = args.max_inflight_requests or batch_size
                jsonl_output = os.environ.get("FORMTSR_BATCH_OUTPUT_JSONL", "")
                outputs = asyncio.run(
                    generate_async_queue(
                        llm,
                        batch,
                        prompts,
                        sampling_params,
                        max_inflight,
                        Path(jsonl_output) if jsonl_output else None,
                    )
                )
            else:
                for start in range(0, len(batch), batch_size):
                    chunk = batch[start : start + batch_size]
                    prompts = build_prompts(prompt, len(chunk), model_path, args.chat_template, enable_thinking)
                    image_data = [[str(Path(row["image_path"]).resolve())] for row in chunk]
                    chunk_outputs = llm.generate(prompt=prompts, image_data=image_data, sampling_params=sampling_params)
                    if isinstance(chunk_outputs, list):
                        outputs.extend(chunk_outputs)
                    else:
                        outputs.append(chunk_outputs)
        finally:
            shutdown = getattr(llm, "shutdown", None)
            if callable(shutdown):
                shutdown()

        rows = []
        if args.submit_mode == "async":
            rows = [output for output in outputs if isinstance(output, dict)]
        else:
            for row, output in zip(batch, outputs):
                rows.append({"sample_id": row["sample_id"], "status": "ok", "raw_response": extract_text(output)})
            for row in batch[len(rows) :]:
                rows.append({"sample_id": row["sample_id"], "status": "error", "raw_response": "", "error": "missing SGLang output"})
        output_path = os.environ.get("FORMTSR_BATCH_OUTPUT")
        if output_path:
            Path(output_path).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        else:
            print(json.dumps(rows, ensure_ascii=False))
        return 0
    except Exception as exc:
        rows = []
        try:
            batch = load_batch()
        except Exception:
            batch = []
        for row in batch:
            rows.append({"sample_id": row.get("sample_id", ""), "status": "error", "raw_response": "", "error": str(exc)})
        output_path = os.environ.get("FORMTSR_BATCH_OUTPUT")
        if rows and output_path:
            Path(output_path).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        elif rows:
            print(json.dumps(rows, ensure_ascii=False))
        print(f"SGLang offline batch failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
