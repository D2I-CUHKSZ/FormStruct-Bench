from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local HuggingFace InternVL VLM batch inference.")
    parser.add_argument("--model-path", default=os.environ.get("HF_MODEL_PATH") or os.environ.get("FORMTSR_MODEL", ""))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("FORMTSR_BATCH_SIZE", "1")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("HF_MAX_TOKENS", os.environ.get("SGLANG_MAX_TOKENS", "4096"))))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("HF_TEMPERATURE", os.environ.get("SGLANG_TEMPERATURE", "0"))))
    parser.add_argument("--device", default=os.environ.get("HF_DEVICE", "cuda:0"))
    parser.add_argument("--device-map", default=os.environ.get("HF_DEVICE_MAP", ""))
    parser.add_argument("--dtype", default=os.environ.get("HF_DTYPE", "bfloat16"))
    return parser.parse_args()


def load_batch() -> list[dict[str, Any]]:
    batch_file = os.environ.get("FORMTSR_BATCH_FILE") or os.environ.get("FORMTSR_BATCH_PATH")
    if batch_file:
        rows = json.loads(Path(batch_file).read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("FORMTSR_BATCH_FILE must contain a JSON list")
        return [row for row in rows if isinstance(row, dict)]
    batch_json = os.environ.get("FORMTSR_BATCH", "")
    if not batch_json:
        image_path = os.environ.get("FORMTSR_IMAGE_PATH", "")
        sample_id = os.environ.get("FORMTSR_SAMPLE_ID", "")
        if image_path and sample_id:
            return [{"sample_id": sample_id, "image_path": image_path}]
        raise ValueError("missing FORMTSR_BATCH_FILE/FORMTSR_BATCH or FORMTSR_IMAGE_PATH/FORMTSR_SAMPLE_ID")
    rows = json.loads(batch_json)
    if not isinstance(rows, list):
        raise ValueError("FORMTSR_BATCH must be a JSON list")
    return [row for row in rows if isinstance(row, dict)]


def torch_dtype(name: str) -> torch.dtype:
    value = name.strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    return torch.bfloat16


def model_device(model: Any, fallback: str) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device(fallback)


def emit_jsonl(path_value: str, row: dict[str, Any]) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        if os.environ.get("FORMTSR_RESULT_FSYNC", "false").strip().lower() in {"1", "true", "yes", "on"}:
            os.fsync(fh.fileno())


def build_messages(image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def run_chunk(
    *,
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    prompt: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    messages_batch: list[list[dict[str, Any]]] = []
    images: list[Image.Image] = []
    opened: list[Image.Image] = []
    try:
        for row in rows:
            image = Image.open(str(row["image_path"])).convert("RGB")
            opened.append(image)
            images.append(image)
            messages_batch.append(build_messages(image, prompt))

        inputs = processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = model_device(model, args.device)
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        input_len = int(inputs["input_ids"].shape[1])
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_tokens,
            "do_sample": args.temperature > 0,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)
        texts = processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)
        return [
            {"sample_id": str(row["sample_id"]), "status": "ok", "raw_response": text.strip()}
            for row, text in zip(rows, texts)
        ]
    finally:
        for image in opened:
            try:
                image.close()
            except Exception:
                pass


def main() -> int:
    args = parse_args()
    prompt = os.environ.get("FORMTSR_PROMPT", "")
    if not prompt:
        print("missing FORMTSR_PROMPT", file=sys.stderr)
        return 2
    if not args.model_path:
        print("missing --model-path/HF_MODEL_PATH/FORMTSR_MODEL", file=sys.stderr)
        return 2

    batch = load_batch()
    jsonl_path = os.environ.get("FORMTSR_BATCH_OUTPUT_JSONL", "")
    started = time.monotonic()
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": torch_dtype(args.dtype),
        }
        if args.device_map:
            model_kwargs["device_map"] = args.device_map
        model = AutoModelForImageTextToText.from_pretrained(args.model_path, **model_kwargs).eval()
        if not args.device_map:
            model = model.to(args.device)

        rows_out: list[dict[str, Any]] = []
        batch_size = max(1, args.batch_size)
        for start in range(0, len(batch), batch_size):
            chunk = batch[start : start + batch_size]
            try:
                results = run_chunk(model=model, processor=processor, rows=chunk, prompt=prompt, args=args)
            except Exception as exc:
                results = [
                    {
                        "sample_id": str(row.get("sample_id", "")),
                        "status": "error",
                        "raw_response": "",
                        "error": str(exc),
                    }
                    for row in chunk
                ]
            for result in results:
                rows_out.append(result)
                emit_jsonl(jsonl_path, result)
            elapsed = time.monotonic() - started
            print(f"InternVL HF progress: {len(rows_out)}/{len(batch)} finished in {elapsed:.1f}s", file=sys.stderr)

        output_path = os.environ.get("FORMTSR_BATCH_OUTPUT")
        if output_path:
            Path(output_path).write_text(json.dumps(rows_out, ensure_ascii=False), encoding="utf-8")
        else:
            print(json.dumps(rows_out, ensure_ascii=False))
        return 0
    except Exception as exc:
        rows_out = [
            {
                "sample_id": str(row.get("sample_id", "")),
                "status": "error",
                "raw_response": "",
                "error": str(exc),
            }
            for row in batch
        ]
        output_path = os.environ.get("FORMTSR_BATCH_OUTPUT")
        if output_path:
            Path(output_path).write_text(json.dumps(rows_out, ensure_ascii=False), encoding="utf-8")
        else:
            print(json.dumps(rows_out, ensure_ascii=False))
        print(f"InternVL HF batch failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
