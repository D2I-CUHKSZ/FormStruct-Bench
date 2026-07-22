from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local document VLM batch inference for FormTSR.")
    parser.add_argument(
        "--backend",
        required=True,
        choices=["paddleocr_vl", "mineru_qwen2vl", "mineru_client", "unlimited_ocr", "generic_image_text"],
    )
    parser.add_argument("--model-path", default=os.environ.get("HF_MODEL_PATH") or os.environ.get("FORMTSR_MODEL", ""))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("FORMTSR_BATCH_SIZE", "1")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("HF_MAX_TOKENS", "4096")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("HF_TEMPERATURE", "0")))
    parser.add_argument("--device", default=os.environ.get("HF_DEVICE", "cuda:0"))
    parser.add_argument("--dtype", default=os.environ.get("HF_DTYPE", "bfloat16"))
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--paddle-task", default=os.environ.get("PADDLEOCR_VL_TASK", "formtsr"))
    parser.add_argument("--unlimited-image-mode", default=os.environ.get("UNLIMITED_OCR_IMAGE_MODE", "gundam"))
    parser.add_argument("--unlimited-ngram-window", type=int, default=int(os.environ.get("UNLIMITED_OCR_NGRAM_WINDOW", "128")))
    return parser.parse_args()


def load_batch() -> list[dict[str, Any]]:
    batch_file = os.environ.get("FORMTSR_BATCH_FILE") or os.environ.get("FORMTSR_BATCH_PATH")
    if batch_file:
        rows = json.loads(Path(batch_file).read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("FORMTSR_BATCH_FILE must contain a JSON list")
        return [row for row in rows if isinstance(row, dict)]
    batch_json = os.environ.get("FORMTSR_BATCH", "")
    if batch_json:
        rows = json.loads(batch_json)
        if not isinstance(rows, list):
            raise ValueError("FORMTSR_BATCH must be a JSON list")
        return [row for row in rows if isinstance(row, dict)]
    image_path = os.environ.get("FORMTSR_IMAGE_PATH", "")
    sample_id = os.environ.get("FORMTSR_SAMPLE_ID", "")
    if image_path and sample_id:
        return [{"sample_id": sample_id, "image_path": image_path}]
    raise ValueError("missing FORMTSR_BATCH_FILE/FORMTSR_BATCH or FORMTSR_IMAGE_PATH/FORMTSR_SAMPLE_ID")


def torch_dtype(name: str) -> torch.dtype:
    value = name.strip().lower()
    if value in {"bf16", "bfloat16", "auto"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    return torch.bfloat16


def env_bool(name: str, default: bool | None = None) -> bool | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def env_json(name: str) -> Any | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return json.loads(value)


def env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def env_float(name: str, default: float | None = None) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


def normalize_max_memory(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    normalized: dict[Any, Any] = {}
    for key, item in value.items():
        if isinstance(key, str) and key.isdigit():
            normalized[int(key)] = item
        else:
            normalized[key] = item
    return normalized


def patch_mineru_vllm_renderer() -> None:
    """Avoid double-rendering MinerU raw multimodal prompts on newer vLLM."""
    try:
        from mineru_vl_utils.vlm_client import vllm_engine_client
    except Exception:
        return

    def passthrough_batch(self: Any, raw_prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        del self
        return raw_prompts

    vllm_engine_client.VllmEngineVlmClient._render_vllm_cmpl_inputs = passthrough_batch

    try:
        from mineru_vl_utils.vlm_client import vllm_async_engine_client
    except Exception:
        return

    async def passthrough_one(self: Any, raw_prompt: dict[str, Any]) -> dict[str, Any]:
        del self
        return raw_prompt

    vllm_async_engine_client.VllmAsyncEngineVlmClient._render_vllm_cmpl_input = passthrough_one


def input_target_device(model: Any, fallback: str) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for value in hf_device_map.values():
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
            if isinstance(value, str) and value.startswith("cuda"):
                return torch.device(value)
    return fallback


def move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    return inputs


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


def trim_generated(input_ids: torch.Tensor, generated: torch.Tensor) -> list[torch.Tensor]:
    if input_ids.ndim == 1:
        return [generated[0, input_ids.shape[0] :]]
    return [out[len(inp) :] for inp, out in zip(input_ids, generated)]


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class GenericImageTextRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        from transformers import AutoProcessor

        self.args = args
        model_class = os.environ.get("HF_MODEL_CLASS", os.environ.get("HF_AUTO_MODEL_CLASS", "image_text_to_text")).strip().lower()
        if model_class in {"multimodal_lm", "auto_model_for_multimodal_lm"}:
            from transformers import AutoModelForMultimodalLM as AutoModelClass
        elif model_class in {"image_text_to_text", "auto_model_for_image_text_to_text"}:
            from transformers import AutoModelForImageTextToText as AutoModelClass
        else:
            raise ValueError(f"unsupported HF_MODEL_CLASS={model_class!r}")

        self.processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": args.trust_remote_code,
            "dtype": torch_dtype(args.dtype),
        }
        device_map = os.environ.get("HF_DEVICE_MAP", "").strip()
        if device_map and device_map.lower() not in {"none", "false", "0"}:
            model_kwargs["device_map"] = device_map
            max_memory = env_json("HF_MAX_MEMORY")
            if max_memory is not None:
                model_kwargs["max_memory"] = normalize_max_memory(max_memory)
        attn_implementation = os.environ.get("HF_ATTN_IMPLEMENTATION", "").strip()
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelClass.from_pretrained(args.model_path, **model_kwargs).eval()
        if args.device and "device_map" not in model_kwargs:
            self.model = self.model.to(args.device)

    def run_one(self, image_path: str, prompt: str) -> str:
        image_mode = os.environ.get("HF_IMAGE_INPUT", "pil").strip().lower()
        image: Image.Image | None = load_image(image_path) if image_mode == "pil" else None
        if image_mode == "path":
            image_payload: Any = image_path
        elif image_mode == "url":
            image_payload = Path(image_path).resolve().as_uri()
        else:
            image_payload = image
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_payload},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            kwargs: dict[str, Any] = {}
            if self.args.backend == "paddleocr_vl":
                max_pixels = int(os.environ.get("PADDLEOCR_VL_MAX_PIXELS", str(1280 * 28 * 28)))
                kwargs["images_kwargs"] = {
                    "size": {
                        "shortest_edge": self.processor.image_processor.min_pixels,
                        "longest_edge": max_pixels,
                    }
                }
            enable_thinking = env_bool("HF_ENABLE_THINKING")
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **kwargs,
            )
            target_device = input_target_device(self.model, self.args.device)
            inputs = move_inputs_to_device(inputs, target_device)
            gen_kwargs: dict[str, Any] = {"max_new_tokens": self.args.max_tokens, "do_sample": self.args.temperature > 0}
            if self.args.temperature > 0:
                gen_kwargs["temperature"] = self.args.temperature
            with torch.inference_mode():
                generated = self.model.generate(**inputs, **gen_kwargs)
            pieces = trim_generated(inputs["input_ids"], generated)
            skip_special_tokens = env_bool("HF_SKIP_SPECIAL_TOKENS", True)
            clean_up_spaces = env_bool("HF_CLEAN_UP_TOKENIZATION_SPACES", False)
            decode_kwargs: dict[str, Any] = {"skip_special_tokens": bool(skip_special_tokens)}
            if clean_up_spaces is not None:
                decode_kwargs["clean_up_tokenization_spaces"] = bool(clean_up_spaces)
            return self.processor.batch_decode(pieces, **decode_kwargs)[0].strip()
        finally:
            if image is not None:
                image.close()


class MinerUQwen2VLRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.args = args
        self.processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch_dtype(args.dtype),
        ).eval()
        if args.device:
            self.model = self.model.to(args.device)

    def run_one(self, image_path: str, prompt: str) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        gen_kwargs: dict[str, Any] = {"max_new_tokens": self.args.max_tokens, "do_sample": self.args.temperature > 0}
        if self.args.temperature > 0:
            gen_kwargs["temperature"] = self.args.temperature
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **gen_kwargs)
        pieces = trim_generated(inputs["input_ids"], generated)
        return self.processor.batch_decode(pieces, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


class MinerUClientRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        from mineru_vl_utils import MinerUClient

        self.args = args
        try:
            from loguru import logger

            logger.remove()
            logger.add(sys.stderr, level=os.environ.get("MINERU_LOG_LEVEL", "WARNING"))
        except Exception:
            pass
        image_analysis = os.environ.get("MINERU_IMAGE_ANALYSIS", "false").strip().lower() in {"1", "true", "yes", "on"}
        backend = os.environ.get("MINERU_BACKEND", "transformers").strip().lower()
        client_kwargs: dict[str, Any] = {
            "backend": backend,
            "image_analysis": image_analysis,
            "batch_size": max(0, int(args.batch_size)),
            "max_concurrency": int(os.environ.get("MINERU_MAX_CONCURRENCY", str(max(1, int(args.batch_size))))),
            "use_tqdm": env_bool("MINERU_USE_TQDM", False),
        }
        if backend == "transformers":
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model_path,
                dtype=torch_dtype(args.dtype),
            ).eval()
            if args.device:
                self.model = self.model.to(args.device)
            client_kwargs.update({"model": self.model, "processor": self.processor})
        elif backend == "vllm-engine":
            if env_bool("MINERU_VLLM_PATCH_RENDERER", True):
                patch_mineru_vllm_renderer()
            from mineru_vl_utils import MinerULogitsProcessor
            from vllm import LLM

            llm_kwargs: dict[str, Any] = {
                "model": args.model_path,
                "dtype": os.environ.get("MINERU_VLLM_DTYPE", "auto"),
                "trust_remote_code": bool(env_bool("MINERU_VLLM_TRUST_REMOTE_CODE", True)),
                "gpu_memory_utilization": float(env_float("MINERU_VLLM_GPU_MEMORY_UTILIZATION", 0.90)),
                "logits_processors": [MinerULogitsProcessor],
            }
            optional_ints = {
                "tensor_parallel_size": "MINERU_VLLM_TENSOR_PARALLEL_SIZE",
                "max_model_len": "MINERU_VLLM_MAX_MODEL_LEN",
                "max_num_seqs": "MINERU_VLLM_MAX_NUM_SEQS",
            }
            for key, env_name in optional_ints.items():
                value = env_int(env_name)
                if value is not None:
                    llm_kwargs[key] = value
            optional_bools = {
                "enforce_eager": "MINERU_VLLM_ENFORCE_EAGER",
                "disable_custom_all_reduce": "MINERU_VLLM_DISABLE_CUSTOM_ALL_REDUCE",
            }
            for key, env_name in optional_bools.items():
                value = env_bool(env_name)
                if value is not None:
                    llm_kwargs[key] = value
            extra_kwargs = env_json("MINERU_VLLM_KWARGS")
            if isinstance(extra_kwargs, dict):
                llm_kwargs.update(extra_kwargs)
            self.llm = LLM(**llm_kwargs)
            client_kwargs["vllm_llm"] = self.llm
        else:
            raise ValueError(f"unsupported MINERU_BACKEND={backend!r}; expected transformers or vllm-engine")
        self.client = MinerUClient(
            **client_kwargs,
        )

    def _format_result(self, content_list: Any) -> str:
        from mineru_vl_utils.post_process import json2md

        output_format = os.environ.get("MINERU_OUTPUT_FORMAT", "markdown").strip().lower()
        if output_format == "json":
            return json.dumps(content_list, ensure_ascii=False)
        return str(json2md(content_list)).strip()

    def run_one(self, image_path: str, prompt: str) -> str:
        del prompt
        image = load_image(image_path)
        try:
            content_list = self.client.two_step_extract(image)
            return self._format_result(content_list)
        finally:
            image.close()

    def run_batch(self, rows: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
        del prompt
        images = [load_image(str(row["image_path"])) for row in rows]
        try:
            extracted = self.client.batch_two_step_extract(images)
            if len(extracted) != len(rows):
                raise RuntimeError(f"MinerU returned {len(extracted)} results for {len(rows)} images")
            return [
                {"sample_id": str(row["sample_id"]), "status": "ok", "raw_response": self._format_result(content)}
                for row, content in zip(rows, extracted)
            ]
        finally:
            for image in images:
                image.close()


class UnlimitedOCRRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.args = args
        if args.device.startswith("cuda") and torch.cuda.is_available():
            index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
            torch.cuda.set_device(index)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            use_safetensors=True,
            dtype=torch_dtype(args.dtype),
        ).eval()
        if args.device.startswith("cuda") and torch.cuda.is_available():
            self.model = self.model.cuda()
        elif args.device:
            self.model = self.model.to(args.device)

    def run_one(self, image_path: str, prompt: str) -> str:
        request_prompt = prompt if "<image>" in prompt else "<image>\n" + prompt
        crop_mode = self.args.unlimited_image_mode == "gundam"
        image_size = 640 if crop_mode else 1024
        with tempfile.TemporaryDirectory(prefix="formtsr_unlimited_ocr_") as tmp_dir:
            result = self.model.infer(
                self.tokenizer,
                prompt=request_prompt,
                image_file=image_path,
                output_path=tmp_dir,
                base_size=1024,
                image_size=image_size,
                crop_mode=crop_mode,
                max_length=self.args.max_tokens,
                no_repeat_ngram_size=35,
                ngram_window=self.args.unlimited_ngram_window,
                temperature=self.args.temperature,
                save_results=False,
                eval_mode=True,
            )
        if isinstance(result, tuple):
            return str(result[0]).strip()
        return str(result).strip()


def make_runner(args: argparse.Namespace) -> Any:
    if args.backend in {"paddleocr_vl", "generic_image_text"}:
        return GenericImageTextRunner(args)
    if args.backend == "mineru_qwen2vl":
        return MinerUQwen2VLRunner(args)
    if args.backend == "mineru_client":
        return MinerUClientRunner(args)
    if args.backend == "unlimited_ocr":
        return UnlimitedOCRRunner(args)
    raise ValueError(f"unsupported backend: {args.backend}")


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
    rows_out: list[dict[str, Any]] = []
    try:
        runner = make_runner(args)
    except Exception as exc:
        rows_out = [
            {"sample_id": str(row.get("sample_id", "")), "status": "error", "raw_response": "", "error": f"load failed: {exc}"}
            for row in batch
        ]
        output_path = os.environ.get("FORMTSR_BATCH_OUTPUT")
        if output_path:
            Path(output_path).write_text(json.dumps(rows_out, ensure_ascii=False), encoding="utf-8")
        print(f"HF document VLM load failed: {exc}", file=sys.stderr)
        return 1

    chunk_size = max(1, int(args.batch_size))
    if hasattr(runner, "run_batch"):
        for start_index in range(0, len(batch), chunk_size):
            chunk = batch[start_index : start_index + chunk_size]
            try:
                chunk_results = runner.run_batch(chunk, prompt)
            except Exception as exc:
                chunk_results = [
                    {
                        "sample_id": str(row.get("sample_id", "")),
                        "status": "error",
                        "raw_response": "",
                        "error": str(exc),
                    }
                    for row in chunk
                ]
            for result in chunk_results:
                rows_out.append(result)
                emit_jsonl(jsonl_path, result)
            elapsed = time.monotonic() - started
            print(f"{args.backend} progress: {len(rows_out)}/{len(batch)} finished in {elapsed:.1f}s", file=sys.stderr)
    else:
        for index, row in enumerate(batch, start=1):
            try:
                raw = runner.run_one(str(row["image_path"]), prompt)
                result = {"sample_id": str(row["sample_id"]), "status": "ok", "raw_response": raw}
            except Exception as exc:
                result = {"sample_id": str(row.get("sample_id", "")), "status": "error", "raw_response": "", "error": str(exc)}
            rows_out.append(result)
            emit_jsonl(jsonl_path, result)
            if index % chunk_size == 0 or index == len(batch):
                elapsed = time.monotonic() - started
                print(f"{args.backend} progress: {index}/{len(batch)} finished in {elapsed:.1f}s", file=sys.stderr)

    output_path = os.environ.get("FORMTSR_BATCH_OUTPUT")
    if output_path:
        Path(output_path).write_text(json.dumps(rows_out, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(rows_out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
