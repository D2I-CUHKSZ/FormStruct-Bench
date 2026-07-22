from __future__ import annotations

import base64
import asyncio
import json
import mimetypes
import os
import shlex
import signal
import subprocess
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class AdapterResult:
    status: str
    raw_response: str = ""
    error: str | None = None


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _loads_json_from_stdout(stdout: str) -> Any:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        start = stdout.find("[")
        end = stdout.rfind("]")
        if start >= 0 and end > start:
            return json.loads(stdout[start : end + 1])
        raise


def _json_or_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] in "[{":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _append_cli_args(args: list[str], value: Any) -> None:
    value = _json_or_value(value)
    if value in (None, ""):
        return
    if isinstance(value, str):
        args.extend(shlex.split(value))
        return
    if isinstance(value, list):
        args.extend(str(item) for item in value)
        return
    if not isinstance(value, dict):
        raise ValueError(f"unsupported vLLM extra args value: {type(value).__name__}")
    for key, item in value.items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(item, bool):
            if item:
                args.append(flag)
            continue
        if isinstance(item, list):
            for child in item:
                args.extend([flag, str(child)])
            continue
        if item is not None:
            args.extend([flag, str(item)])


def _formtsr_vlm_response_schema(
    *,
    compact: bool = False,
    hierarchical: bool = False,
) -> dict[str, Any]:
    text_schema: dict[str, Any] = {"type": "string", "maxLength": 60 if compact else 160}
    short_text_schema: dict[str, Any] = {"type": "string", "maxLength": 32 if compact else 80}
    id_schema: dict[str, Any] = {"type": "string", "maxLength": 32}
    bbox_schema: dict[str, Any] = {
        "type": "array",
        "items": {"type": "number", "minimum": 0, "maximum": 1000},
        "minItems": 4,
        "maxItems": 4,
    }
    region_type_schema: dict[str, Any] = {
        "type": "string",
        "enum": ["title", "section", "field", "value", "text", "widget", "table", "other"],
    }
    widget_type_schema: dict[str, Any] = {
        "type": "string",
        "enum": ["checkbox", "radio", "input", "signature", "other"],
    }
    widget_state_schema: dict[str, Any] = {
        "type": "string",
        "enum": ["selected", "unselected", "unknown", "filled", "blank"],
    }
    scalar_schema: dict[str, Any] = {
        "anyOf": [
            text_schema,
            {"type": "number"},
            {"type": "boolean"},
            {"type": "array", "maxItems": 5 if compact else 20, "items": text_schema},
        ]
    }
    # Dataset-derived guardrails: GT scan over 7000 labels found max string
    # length 469, max object width 28, max array length 18, and max depth 7.
    # These wider bounds prevent runaway duplicate generation without clipping GT.
    answer_text_schema: dict[str, Any] = {"type": "string", "maxLength": 512}
    answer_scalar_schema: dict[str, Any] = {
        "anyOf": [
            answer_text_schema,
            {"type": "number"},
            {"type": "boolean"},
            {"type": "null"},
        ]
    }

    def bounded_answer_value_schema(depth: int = 8) -> dict[str, Any]:
        if depth <= 0:
            return answer_scalar_schema
        child_schema = bounded_answer_value_schema(depth - 1)
        return {
            "anyOf": [
                answer_text_schema,
                {"type": "number"},
                {"type": "boolean"},
                {"type": "null"},
                {"type": "array", "maxItems": 32, "items": child_schema},
                {
                    "type": "object",
                    "maxProperties": 32,
                    "additionalProperties": child_schema,
                },
            ]
        }

    answer_value_schema: dict[str, Any] = bounded_answer_value_schema()
    cell_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "maxProperties": 7,
        "properties": {
            "id": id_schema,
            "row": {"type": "integer"},
            "col": {"type": "integer"},
            "rowspan": {"type": "integer"},
            "colspan": {"type": "integer"},
            "bbox": bbox_schema,
            "text": text_schema,
        },
    }
    if hierarchical:
        cell_schema["required"] = ["id", "row", "col"]
    max_regions = 10 if compact else 80 if hierarchical else 60
    max_widgets = 8 if compact else 80 if hierarchical else 40
    max_grids = 1 if compact else 10
    max_cells = 12 if compact else 160 if hierarchical else 100
    max_ligs = 4 if compact else 20
    max_relations = 12 if compact else 220 if hierarchical else 60
    return {
        "title": "FormTSRPrediction",
        "type": "object",
        "additionalProperties": False,
        "required": ["regions", "widgets", "local_grids", "cells", "line_item_groups", "relations", "answer"],
        "properties": {
            "regions": {
                "type": "array",
                "maxItems": max_regions,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "maxProperties": 4,
                    "properties": {
                        "id": id_schema,
                        "type": region_type_schema,
                        "bbox": bbox_schema,
                        "text": text_schema,
                    },
                    **({"required": ["id", "type", "bbox", "text"]} if hierarchical else {}),
                },
            },
            "widgets": {
                "type": "array",
                "maxItems": max_widgets,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "maxProperties": 8 if hierarchical else 5,
                    "properties": {
                        "id": id_schema,
                        "type": widget_type_schema,
                        "bbox": bbox_schema,
                        "label": text_schema,
                        "selected": {"type": "boolean"},
                        **(
                            {
                                "state": widget_state_schema,
                                "group_id": id_schema,
                                "group_type": text_schema,
                            }
                            if hierarchical
                            else {}
                        ),
                    },
                    **(
                        {
                            "required": [
                                "id",
                                "type",
                                "bbox",
                                "label",
                                "state",
                                "group_id",
                                "group_type",
                            ]
                        }
                        if hierarchical
                        else {}
                    ),
                },
            },
            "local_grids": {
                "type": "array",
                "maxItems": max_grids,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "maxProperties": 3,
                    "properties": {
                        "id": id_schema,
                        "region_id": id_schema,
                        "cells": {"type": "array", "maxItems": max_cells, "items": cell_schema},
                    },
                    **({"required": ["id", "region_id", "cells"]} if hierarchical else {}),
                },
            },
            "cells": {"type": "array", "maxItems": max_cells, "items": cell_schema},
            "line_item_groups": {
                "type": "array",
                "maxItems": max_ligs,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "maxProperties": 3,
                    "properties": {
                        "id": id_schema,
                        "bbox": bbox_schema,
                        "text": text_schema,
                    },
                    **({"required": ["id", "bbox"]} if hierarchical else {}),
                },
            },
            "relations": {
                "type": "array",
                "maxItems": max_relations,
                "items": (
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["u", "r", "v"],
                        "properties": {
                            "u": id_schema,
                            "r": {
                                "type": "string",
                                "enum": [
                                    "key-value",
                                    "parent-child",
                                    "field-widget",
                                    "key-to-cell",
                                ],
                            },
                            "v": id_schema,
                        },
                    }
                    if hierarchical
                    else {"type": "object", "additionalProperties": scalar_schema, "maxProperties": 8}
                ),
            },
            "answer": {
                "type": "object",
                "maxProperties": 32,
                "additionalProperties": answer_value_schema,
            },
        },
    }


class BaseAdapter:
    env_var: str | None = None

    def __init__(self, model_config: dict[str, Any]) -> None:
        self.model_config = model_config

    @property
    def name(self) -> str:
        return str(self.model_config.get("name") or self.model_config.get("model") or "model")

    def missing_credentials(self) -> bool:
        return bool(self.env_var and not os.environ.get(self.env_var))

    def run(self, sample: dict[str, Any], prompt: str) -> AdapterResult:
        raise NotImplementedError

    def supports_batch(self) -> bool:
        return False

    def run_batch(
        self,
        samples: list[dict[str, Any]],
        prompt: str,
        on_result: Callable[[dict[str, Any], AdapterResult], None] | None = None,
    ) -> list[AdapterResult]:
        return [self.run(sample, prompt) for sample in samples]


class MissingSDKAdapter(BaseAdapter):
    sdk_name = ""

    def run(self, sample: dict[str, Any], prompt: str) -> AdapterResult:
        if self.missing_credentials():
            return AdapterResult("missing_credentials", error=f"missing {self.env_var}")
        return AdapterResult("provider_unavailable", error=f"{self.sdk_name} SDK is not installed")


class OpenAIAdapter(MissingSDKAdapter):
    env_var = "OPENAI_API_KEY"
    sdk_name = "openai"

    def _codex_config_paths(self) -> list[Path]:
        configured = self.model_config.get("codex_config")
        paths: list[Path] = []
        if configured:
            paths.append(Path(str(configured)).expanduser())
        paths.extend(
            [
                Path.home() / ".codex" / "config.toml",
                Path("/path/to/data/tools/codex/dot-codex/config.toml"),
            ]
        )
        return paths

    def _codex_auth_paths(self) -> list[Path]:
        configured = self.model_config.get("codex_auth")
        paths: list[Path] = []
        if configured:
            paths.append(Path(str(configured)).expanduser())
        paths.extend(
            [
                Path.home() / ".codex" / "auth.json",
                Path("/path/to/data/tools/codex/dot-codex/auth.json"),
            ]
        )
        return paths

    def _find_first(self, value: Any, names: set[str]) -> str | None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_norm = str(key).lower().replace("-", "_")
                if key_norm in names and isinstance(child, str) and child.strip():
                    return child.strip()
            for child in value.values():
                found = self._find_first(child, names)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = self._find_first(child, names)
                if found:
                    return found
        return None

    def _load_codex_openai_credentials(self) -> tuple[str | None, str | None]:
        api_key: str | None = None
        for path in self._codex_auth_paths():
            if not path.exists():
                continue
            try:
                auth_data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            api_key = self._find_first(auth_data, {"openai_api_key", "api_key", "key", "token", "auth_token"})
            if api_key:
                break

        base_url: str | None = None
        for path in self._codex_config_paths():
            if not path.exists():
                continue
            try:
                data = tomllib.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            base_url = self._find_first(data, {"base_url", "api_base", "openai_base_url", "endpoint"})
            api_key = api_key or self._find_first(data, {"openai_api_key", "api_key", "key", "token", "auth_token"})
            if api_key or base_url:
                return api_key, base_url
        return api_key, base_url

    def _credentials(self) -> tuple[str | None, str | None]:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        if api_key and base_url:
            return api_key, base_url
        cfg_key, cfg_base_url = self._load_codex_openai_credentials()
        return api_key or cfg_key, base_url or cfg_base_url

    def missing_credentials(self) -> bool:
        api_key, _base_url = self._credentials()
        return not bool(api_key)

    def run(self, sample: dict[str, Any], prompt: str) -> AdapterResult:
        api_key, base_url = self._credentials()
        if not api_key:
            return AdapterResult("missing_credentials", error=f"missing {self.env_var}")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            return AdapterResult("provider_unavailable", error=f"openai SDK unavailable: {exc}")
        try:
            image_path = Path(sample["image_path"])
            image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            timeout_seconds = float(self.model_config.get("timeout_seconds", 60))
            client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_seconds}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = OpenAI(**client_kwargs)
            last_error: str | None = None
            retries = int(self.model_config.get("retries", 0))
            retry_backoff_seconds = float(self.model_config.get("retry_backoff_seconds", 120))
            for attempt in range(retries + 1):
                try:
                    response = client.responses.create(
                        model=str(self.model_config.get("model")),
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": prompt},
                                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"},
                                ],
                            }
                        ],
                        temperature=float(self.model_config.get("temperature", 0)),
                        max_output_tokens=int(self.model_config.get("max_tokens", 4096)),
                        timeout=timeout_seconds,
                    )
                    raw = getattr(response, "output_text", None) or str(response)
                    return AdapterResult("ok", raw_response=raw)
                except Exception as exc:
                    last_error = str(exc)
                    retryable = "524" in last_error or "timeout" in last_error.lower()
                    if attempt < retries and retryable:
                        time.sleep(retry_backoff_seconds)
                        continue
                    return AdapterResult("error", error=last_error)
            return AdapterResult("error", error=last_error or "unknown OpenAI adapter error")
        except Exception as exc:
            return AdapterResult("error", error=str(exc))


class AnthropicAdapter(MissingSDKAdapter):
    env_var = "ANTHROPIC_API_KEY"
    sdk_name = "anthropic"

    def run(self, sample: dict[str, Any], prompt: str) -> AdapterResult:
        if self.missing_credentials():
            return AdapterResult("missing_credentials", error=f"missing {self.env_var}")
        try:
            import anthropic  # type: ignore
        except Exception as exc:
            return AdapterResult("provider_unavailable", error=f"anthropic SDK unavailable: {exc}")
        try:
            image_b64 = base64.b64encode(Path(sample["image_path"]).read_bytes()).decode("ascii")
            client = anthropic.Anthropic()
            message = client.messages.create(
                model=str(self.model_config.get("model")),
                max_tokens=int(self.model_config.get("max_tokens", 4096)),
                temperature=float(self.model_config.get("temperature", 0)),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
                            },
                        ],
                    }
                ],
            )
            raw = "\n".join(block.text for block in message.content if getattr(block, "type", "") == "text")
            return AdapterResult("ok", raw_response=raw)
        except Exception as exc:
            return AdapterResult("error", error=str(exc))


class GeminiAdapter(MissingSDKAdapter):
    env_var = "GOOGLE_API_KEY"
    sdk_name = "google-generativeai"

    def run(self, sample: dict[str, Any], prompt: str) -> AdapterResult:
        if self.missing_credentials():
            return AdapterResult("missing_credentials", error=f"missing {self.env_var}")
        try:
            import google.generativeai as genai  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as exc:
            return AdapterResult("provider_unavailable", error=f"gemini dependencies unavailable: {exc}")
        try:
            genai.configure(api_key=os.environ[self.env_var or "GOOGLE_API_KEY"])
            model = genai.GenerativeModel(str(self.model_config.get("model")))
            response = model.generate_content(
                [prompt, Image.open(sample["image_path"])],
                generation_config={
                    "temperature": float(self.model_config.get("temperature", 0)),
                    "max_output_tokens": int(self.model_config.get("max_tokens", 4096)),
                },
            )
            return AdapterResult("ok", raw_response=getattr(response, "text", str(response)))
        except Exception as exc:
            return AdapterResult("error", error=str(exc))


class LocalHFAdapter(BaseAdapter):
    def _env(self, sample: dict[str, Any] | None, prompt: str) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "FORMTSR_PROMPT": prompt,
                "FORMTSR_MODEL": str(self.model_config.get("model", "")),
            }
        )
        if sample is not None:
            env.update(
                {
                    "FORMTSR_IMAGE_PATH": str(sample["image_path"]),
                    "FORMTSR_SAMPLE_ID": str(sample["sample_id"]),
                }
        )
        for key, value in dict(self.model_config.get("env", {})).items():
            env[str(key)] = str(value)
        cuda_home = env.get("CUDA_HOME")
        if cuda_home:
            cuda_bin = str(Path(cuda_home) / "bin")
            env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")
        return env

    def supports_batch(self) -> bool:
        return bool(self.model_config.get("batch_command"))

    def run(self, sample: dict[str, Any], prompt: str) -> AdapterResult:
        command = self.model_config.get("command")
        if not command:
            return AdapterResult("provider_unavailable", error="local_hf_vlm requires a command in config")
        env = self._env(sample, prompt)
        try:
            proc = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                timeout=int(self.model_config.get("timeout_seconds", 600)),
                env=env,
                check=False,
            )
            if proc.returncode != 0:
                return AdapterResult("error", raw_response=proc.stdout, error=proc.stderr.strip())
            return AdapterResult("ok", raw_response=proc.stdout)
        except Exception as exc:
            return AdapterResult("error", error=str(exc))

    def _decode_result_item(self, item: dict[str, Any]) -> AdapterResult:
        return AdapterResult(
            str(item.get("status") or "ok"),
            raw_response=str(item.get("raw_response") or ""),
            error=str(item["error"]) if item.get("error") is not None else None,
        )

    def run_batch(
        self,
        samples: list[dict[str, Any]],
        prompt: str,
        on_result: Callable[[dict[str, Any], AdapterResult], None] | None = None,
    ) -> list[AdapterResult]:
        command = self.model_config.get("batch_command")
        if not command:
            return super().run_batch(samples, prompt, on_result=on_result)
        max_samples_per_engine = int(self.model_config.get("max_samples_per_engine", 0) or 0)
        if max_samples_per_engine > 0 and len(samples) > max_samples_per_engine:
            results: list[AdapterResult] = []
            for start in range(0, len(samples), max_samples_per_engine):
                chunk = samples[start : start + max_samples_per_engine]
                results.extend(self._run_batch_once(command, chunk, prompt, on_result=on_result))
            return results
        return self._run_batch_once(command, samples, prompt, on_result=on_result)

    def _run_batch_once(
        self,
        command: str,
        samples: list[dict[str, Any]],
        prompt: str,
        on_result: Callable[[dict[str, Any], AdapterResult], None] | None = None,
    ) -> list[AdapterResult]:
        env = self._env(None, prompt)
        sample_by_id = {str(sample["sample_id"]): sample for sample in samples}
        batch_input = tempfile.NamedTemporaryFile(
            prefix="formtsr_local_hf_batch_",
            suffix=".samples.json",
            mode="w",
            encoding="utf-8",
            delete=False,
        )
        batch_input_path = Path(batch_input.name)
        json.dump(
            [
                {
                    "sample_id": str(sample["sample_id"]),
                    "image_path": str(sample["image_path"]),
                }
                for sample in samples
            ],
            batch_input,
            ensure_ascii=False,
        )
        batch_input.close()
        env.pop("FORMTSR_BATCH", None)
        env["FORMTSR_BATCH_FILE"] = str(batch_input_path)
        batch_output = tempfile.NamedTemporaryFile(prefix="formtsr_sglang_batch_", suffix=".json", delete=False)
        batch_output_path = Path(batch_output.name)
        batch_output.close()
        batch_output_jsonl = tempfile.NamedTemporaryFile(prefix="formtsr_sglang_batch_", suffix=".jsonl", delete=False)
        batch_output_jsonl_path = Path(batch_output_jsonl.name)
        batch_output_jsonl.close()
        stdout_log = tempfile.NamedTemporaryFile(prefix="formtsr_sglang_batch_", suffix=".stdout", mode="w+", encoding="utf-8", delete=False)
        stderr_log = tempfile.NamedTemporaryFile(prefix="formtsr_sglang_batch_", suffix=".stderr", mode="w+", encoding="utf-8", delete=False)
        stdout_log_path = Path(stdout_log.name)
        stderr_log_path = Path(stderr_log.name)
        env["FORMTSR_BATCH_OUTPUT"] = str(batch_output_path)
        env["FORMTSR_BATCH_OUTPUT_JSONL"] = str(batch_output_jsonl_path)
        env["FORMTSR_BATCH_SIZE"] = str(max(1, int(self.model_config.get("batch_size", 1))))
        emitted: dict[str, AdapterResult] = {}
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                text=True,
                stdout=stdout_log,
                stderr=stderr_log,
                env=env,
            )
            stdout, stderr = "", ""
            timeout_seconds = int(self.model_config.get("timeout_seconds", 600))
            started = time.monotonic()
            jsonl_offset = 0
            while proc.poll() is None:
                if time.monotonic() - started > timeout_seconds:
                    proc.kill()
                    proc.wait()
                    stdout_log.flush()
                    stderr_log.flush()
                    stdout = stdout_log_path.read_text(encoding="utf-8", errors="replace")
                    stderr = stderr_log_path.read_text(encoding="utf-8", errors="replace")
                    raise RuntimeError(f"batch command timed out after {timeout_seconds}s\n{stderr.strip()}")
                if on_result and batch_output_jsonl_path.exists():
                    with batch_output_jsonl_path.open("r", encoding="utf-8") as fh:
                        fh.seek(jsonl_offset)
                        while True:
                            line = fh.readline()
                            if not line:
                                break
                            if not line.endswith("\n"):
                                break
                            next_offset = fh.tell()
                            if not line.strip():
                                jsonl_offset = next_offset
                                continue
                            try:
                                item = json.loads(line)
                            except json.JSONDecodeError:
                                break
                            jsonl_offset = next_offset
                            if not isinstance(item, dict) or not isinstance(item.get("sample_id"), str):
                                continue
                            sample = sample_by_id.get(item["sample_id"])
                            if sample is None or item["sample_id"] in emitted:
                                continue
                            result = self._decode_result_item(item)
                            emitted[item["sample_id"]] = result
                            on_result(sample, result)
                time.sleep(1)
            proc.wait()
            stdout_log.flush()
            stderr_log.flush()
            stdout = stdout_log_path.read_text(encoding="utf-8", errors="replace")
            stderr = stderr_log_path.read_text(encoding="utf-8", errors="replace")
            if on_result and batch_output_jsonl_path.exists():
                with batch_output_jsonl_path.open("r", encoding="utf-8") as fh:
                    fh.seek(jsonl_offset)
                    while True:
                        line = fh.readline()
                        if not line:
                            break
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(item, dict) or not isinstance(item.get("sample_id"), str):
                            continue
                        sample = sample_by_id.get(item["sample_id"])
                        if sample is None or item["sample_id"] in emitted:
                            continue
                        result = self._decode_result_item(item)
                        emitted[item["sample_id"]] = result
                        on_result(sample, result)
            if proc.returncode != 0:
                log_dir = Path(str(self.model_config.get("log_dir", "outputs/main_exp/logs")))
                log_dir.mkdir(parents=True, exist_ok=True)
                failed_stdout = log_dir / f"{self.name}_failed.stdout"
                failed_stderr = log_dir / f"{self.name}_failed.stderr"
                failed_stdout.write_text(stdout, encoding="utf-8")
                failed_stderr.write_text(stderr, encoding="utf-8")
                raise RuntimeError(
                    f"batch command failed with code {proc.returncode}; stderr saved to {failed_stderr}"
                )
            decoded = json.loads(batch_output_path.read_text(encoding="utf-8")) if batch_output_path.exists() and batch_output_path.stat().st_size else _loads_json_from_stdout(stdout)
            if not isinstance(decoded, list):
                raise ValueError("batch command did not return a JSON list")
            by_id: dict[str, dict[str, Any]] = {}
            for item in decoded:
                if isinstance(item, dict) and isinstance(item.get("sample_id"), str):
                    by_id[item["sample_id"]] = item
            results: list[AdapterResult] = []
            for sample in samples:
                item = by_id.get(str(sample["sample_id"]))
                if item is None:
                    results.append(AdapterResult("error", error="missing batch output for sample"))
                    continue
                result = self._decode_result_item(item)
                results.append(result)
                if on_result and str(sample["sample_id"]) not in emitted:
                    emitted[str(sample["sample_id"])] = result
                    on_result(sample, result)
            return results
        except Exception:
            raise
        finally:
            try:
                batch_input_path.unlink()
            except OSError:
                pass
            try:
                batch_output_path.unlink()
            except OSError:
                pass
            try:
                batch_output_jsonl_path.unlink()
            except OSError:
                pass
            try:
                stdout_log.close()
                stdout_log_path.unlink()
            except OSError:
                pass
            try:
                stderr_log.close()
                stderr_log_path.unlink()
            except OSError:
                pass


class LocalSGLangServerAdapter(BaseAdapter):
    def _env(self, prompt: str) -> dict[str, str]:
        env = os.environ.copy()
        env["FORMTSR_PROMPT"] = prompt
        env["FORMTSR_MODEL"] = str(self.model_config.get("model", ""))
        for key, value in dict(self.model_config.get("env", {})).items():
            env[str(key)] = str(value)
        cuda_home = env.get("CUDA_HOME")
        if cuda_home:
            env["PATH"] = str(Path(cuda_home) / "bin") + os.pathsep + env.get("PATH", "")
        py_path = env.get("PYTHONPATH", "")
        stub_path = str(Path("scripts/sglang_stubs").resolve())
        env["PYTHONPATH"] = stub_path + (os.pathsep + py_path if py_path else "")
        return env

    def supports_batch(self) -> bool:
        return True

    def _base_url(self) -> str:
        return str(self.model_config.get("base_url") or dict(self.model_config.get("env", {})).get("SGLANG_BASE_URL") or "http://127.0.0.1:30000").rstrip("/")

    def _served_model(self) -> str:
        env_cfg = dict(self.model_config.get("env", {}))
        return str(self.model_config.get("served_model") or env_cfg.get("FORMTSR_SERVED_MODEL") or self.model_config.get("model") or "default")

    def _server_command(self) -> str:
        configured = self.model_config.get("server_command")
        if configured:
            return str(configured)
        env_cfg = dict(self.model_config.get("env", {}))
        model_path = str(env_cfg.get("SGLANG_MODEL_PATH") or self.model_config.get("model_path") or self.model_config.get("model") or "")
        host = str(self.model_config.get("host") or env_cfg.get("SGLANG_HOST") or "127.0.0.1")
        port = int(self.model_config.get("port") or env_cfg.get("SGLANG_PORT") or 30000)
        python = str(self.model_config.get("python") or "./.venv/bin/python")
        args = [
            python,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--served-model-name",
            self._served_model(),
            "--trust-remote-code",
            "--enable-multimodal",
            "--dp-size",
            str(env_cfg.get("SGLANG_DP_SIZE", self.model_config.get("dp_size", "1"))),
            "--mem-fraction-static",
            str(env_cfg.get("SGLANG_MEM_FRACTION_STATIC", "0.90")),
            "--kv-cache-dtype",
            str(env_cfg.get("SGLANG_KV_CACHE_DTYPE", "auto")),
            "--context-length",
            str(env_cfg.get("SGLANG_CONTEXT_LENGTH", "24576")),
            "--max-total-tokens",
            str(env_cfg.get("SGLANG_MAX_TOTAL_TOKENS", "32768")),
            "--max-running-requests",
            str(env_cfg.get("SGLANG_MAX_RUNNING_REQUESTS", "32")),
            "--max-queued-requests",
            str(env_cfg.get("SGLANG_MAX_QUEUED_REQUESTS", "128")),
            "--chunked-prefill-size",
            str(env_cfg.get("SGLANG_CHUNKED_PREFILL_SIZE", "8192")),
            "--max-prefill-tokens",
            str(env_cfg.get("SGLANG_MAX_PREFILL_TOKENS", "24576")),
            "--schedule-conservativeness",
            str(env_cfg.get("SGLANG_SCHEDULE_CONSERVATIVENESS", "0.8")),
            "--cuda-graph-max-bs",
            str(env_cfg.get("SGLANG_CUDA_GRAPH_MAX_BS", "64")),
            "--attention-backend",
            str(env_cfg.get("SGLANG_ATTENTION_BACKEND", "triton")),
            "--sampling-backend",
            str(env_cfg.get("SGLANG_SAMPLING_BACKEND", "pytorch")),
            "--grammar-backend",
            str(env_cfg.get("SGLANG_GRAMMAR_BACKEND", "xgrammar")),
            "--log-level",
            str(env_cfg.get("SGLANG_LOG_LEVEL", "info")),
        ]
        chat_template = env_cfg.get("SGLANG_CHAT_TEMPLATE", self.model_config.get("chat_template"))
        if chat_template:
            args.extend(["--chat-template", str(chat_template)])
        if str(env_cfg.get("SGLANG_DISABLE_CUDA_GRAPH", "false")).strip().lower() in {"1", "true", "yes", "on"}:
            args.append("--disable-cuda-graph")
        if str(env_cfg.get("SGLANG_DISABLE_OVERLAP_SCHEDULE", "false")).strip().lower() in {"1", "true", "yes", "on"}:
            args.append("--disable-overlap-schedule")
        _append_cli_args(args, self.model_config.get("sglang_extra_args", env_cfg.get("SGLANG_EXTRA_ARGS")))
        return subprocess.list2cmdline(args)

    def _request_json(self, path: str, timeout: float = 5.0) -> dict[str, Any]:
        req = urllib.request.Request(self._base_url() + path, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"unexpected response from {path}")
        return data

    def _wait_ready(self, proc: subprocess.Popen[str], stdout_path: Path, stderr_path: Path | None = None) -> None:
        if stderr_path is None:
            stderr_path = stdout_path
        timeout_seconds = int(self.model_config.get("server_start_timeout_seconds", 900))
        deadline = time.monotonic() + timeout_seconds
        last_error = ""
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
                raise RuntimeError(f"SGLang server exited before ready with code {proc.returncode}\n{stderr[-4000:]}")
            try:
                self._request_json("/v1/models", timeout=5)
                return
            except Exception as exc:
                last_error = str(exc)
                time.sleep(2)
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
        raise RuntimeError(f"SGLang server did not become ready after {timeout_seconds}s: {last_error}\n{stderr[-4000:]}")

    def _stop_server(self, proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                pass

    def _image_data_url(self, path: Path) -> str:
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _payload(self, sample: dict[str, Any], prompt: str) -> dict[str, Any]:
        env_cfg = dict(self.model_config.get("env", {}))
        prompt_prefix = str(self.model_config.get("prompt_prefix") or env_cfg.get("FORMTSR_PROMPT_PREFIX") or "")
        prompt_suffix = str(self.model_config.get("prompt_suffix") or env_cfg.get("FORMTSR_PROMPT_SUFFIX") or "")
        request_prompt = prompt_prefix + prompt + prompt_suffix
        extra_body: dict[str, Any] = {}
        if _truthy(
            self.model_config.get(
                "include_chat_template_kwargs",
                env_cfg.get("FORMTSR_INCLUDE_CHAT_TEMPLATE_KWARGS"),
            ),
            default=True,
        ):
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": _truthy(env_cfg.get("SGLANG_ENABLE_THINKING"), default=False)
            }
        configured_extra_body = self.model_config.get("extra_body", env_cfg.get("FORMTSR_EXTRA_BODY"))
        configured_extra_body = _json_or_value(configured_extra_body)
        if isinstance(configured_extra_body, dict):
            extra_body.update(configured_extra_body)
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request_prompt},
                    {"type": "image_url", "image_url": {"url": self._image_data_url(Path(sample["image_path"]))}},
                ],
            }
        ]
        assistant_prefill = str(
            self.model_config.get("assistant_prefill") or env_cfg.get("FORMTSR_ASSISTANT_PREFILL") or ""
        ).replace("\\n", "\n")
        if assistant_prefill:
            messages.append({"role": "assistant", "content": assistant_prefill})

        request_extra_body: dict[str, Any] = {}
        payload: dict[str, Any] = {
            "model": self._served_model(),
            "messages": messages,
            "temperature": float(self.model_config.get("temperature", 0)),
            "max_tokens": int(self.model_config.get("max_tokens", 4096)),
        }
        if assistant_prefill:
            request_extra_body["add_generation_prompt"] = False
            request_extra_body["continue_final_message"] = _truthy(
                self.model_config.get("continue_final_message", env_cfg.get("FORMTSR_CONTINUE_FINAL_MESSAGE")),
                default=True,
            )
        elif "add_generation_prompt" in self.model_config or "FORMTSR_ADD_GENERATION_PROMPT" in env_cfg:
            request_extra_body["add_generation_prompt"] = _truthy(
                self.model_config.get("add_generation_prompt", env_cfg.get("FORMTSR_ADD_GENERATION_PROMPT")),
                default=True,
            )
        if "continue_final_message" in self.model_config or "FORMTSR_CONTINUE_FINAL_MESSAGE" in env_cfg:
            request_extra_body["continue_final_message"] = _truthy(
                self.model_config.get("continue_final_message", env_cfg.get("FORMTSR_CONTINUE_FINAL_MESSAGE")),
                default=False,
            )
        response_format = str(self.model_config.get("response_format") or env_cfg.get("SGLANG_RESPONSE_FORMAT") or "").strip()
        if response_format:
            if response_format == "json_object":
                payload["response_format"] = {"type": "json_object"}
            elif response_format == "json_schema":
                schema = self.model_config.get("response_json_schema")
                schema_name = str(self.model_config.get("response_json_schema_name") or env_cfg.get("SGLANG_RESPONSE_JSON_SCHEMA") or "").strip()
                known_schema_names = {
                    "formtsr_vlm_v1",
                    "formtsr_vlm_compact_v1",
                    "formtsr_hierarchical_v2",
                }
                if not isinstance(schema, dict) and schema_name in known_schema_names:
                    schema = _formtsr_vlm_response_schema(
                        compact=schema_name == "formtsr_vlm_compact_v1",
                        hierarchical=schema_name == "formtsr_hierarchical_v2",
                    )
                if isinstance(schema, dict):
                    response_format_style = str(
                        self.model_config.get("response_format_style")
                        or env_cfg.get("FORMTSR_RESPONSE_FORMAT_STYLE")
                        or "sglang"
                    ).strip().lower()
                    if response_format_style in {"structured_outputs", "vllm_structured_outputs"}:
                        request_extra_body["structured_outputs"] = {"json": schema}
                    elif response_format_style in {"guided_json", "vllm_guided_json"}:
                        request_extra_body["guided_json"] = schema
                    elif response_format_style in {"openai", "vllm"}:
                        payload["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema_name or "formtsr_vlm",
                                "schema": schema,
                                "strict": True,
                            },
                        }
                    else:
                        payload["response_format"] = {"type": "json_schema", "schema": schema}
            elif response_format == "text":
                payload["response_format"] = {"type": "text"}
        if request_extra_body:
            extra_body.update(request_extra_body)
        if extra_body:
            payload["extra_body"] = extra_body
        debug_payload_path = str(env_cfg.get("FORMTSR_DEBUG_PAYLOAD_PATH", "")).strip()
        if debug_payload_path and not Path(debug_payload_path).exists():
            debug_payload = {
                "model": payload.get("model"),
                "temperature": payload.get("temperature"),
                "max_tokens": payload.get("max_tokens"),
                "message_roles": [message.get("role") for message in messages],
                "assistant_prefill": assistant_prefill,
                "add_generation_prompt": request_extra_body.get("add_generation_prompt"),
                "continue_final_message": request_extra_body.get("continue_final_message"),
                "response_format": payload.get("response_format"),
                "extra_body": payload.get("extra_body"),
            }
            try:
                Path(debug_payload_path).parent.mkdir(parents=True, exist_ok=True)
                Path(debug_payload_path).write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                pass
        stop = str(env_cfg.get("SGLANG_STOP", env_cfg.get("VLLM_STOP", ""))).replace("\\n", "\n")
        if stop:
            payload["stop"] = [item for item in stop.split("|") if item]
        return payload

    def _extract_text(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0]
            message = getattr(first, "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    chunks = [str(getattr(item, "text", "")) for item in content if getattr(item, "text", "")]
                    if chunks:
                        return "\n".join(chunks)
                reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
                if isinstance(reasoning, str):
                    return reasoning
            text = getattr(first, "text", None)
            if isinstance(text, str):
                return text
        if hasattr(response, "model_dump_json"):
            try:
                return str(response.model_dump_json(indent=2))
            except Exception:
                pass
        return str(response)

    def _is_fail_fast_error(self, error: str) -> bool:
        if not _truthy(self.model_config.get("fail_fast_server_errors"), default=False):
            return False
        lower = error.lower()
        return any(
            marker in lower
            for marker in (
                "connection error",
                "connection refused",
                "server disconnected",
                "enginedead",
                "enginecore",
            )
        )

    async def _run_requests(
        self,
        samples: list[dict[str, Any]],
        prompt: str,
        on_result: Callable[[dict[str, Any], AdapterResult], None] | None,
        server_proc: subprocess.Popen[str] | None = None,
    ) -> list[AdapterResult]:
        from openai import AsyncOpenAI  # type: ignore

        client = AsyncOpenAI(api_key="EMPTY", base_url=self._base_url() + "/v1", timeout=float(self.model_config.get("request_timeout_seconds", 1800)))
        concurrency = max(1, int(self.model_config.get("concurrency", self.model_config.get("batch_size", 16))))
        retries = int(self.model_config.get("retries", 1))
        retry_sleep = float(self.model_config.get("retry_backoff_seconds", 10))
        semaphore = asyncio.Semaphore(concurrency)
        results: list[AdapterResult | None] = [None] * len(samples)
        fatal_event = asyncio.Event()
        fatal_error: dict[str, str] = {}

        async def run_one(index: int) -> None:
            if fatal_event.is_set():
                return
            sample = samples[index]
            async with semaphore:
                if fatal_event.is_set():
                    return
                last_error: str | None = None
                for attempt in range(retries + 1):
                    try:
                        response = await client.chat.completions.create(**self._payload(sample, prompt))
                        result = AdapterResult("ok", raw_response=self._extract_text(response))
                        results[index] = result
                        if on_result:
                            on_result(sample, result)
                        return
                    except Exception as exc:
                        last_error = str(exc)
                        if self._is_fail_fast_error(last_error):
                            fatal_error["error"] = last_error
                            fatal_event.set()
                            result = AdapterResult("error", error=last_error)
                            results[index] = result
                            if on_result:
                                on_result(sample, result)
                            return
                        if attempt < retries:
                            await asyncio.sleep(retry_sleep)
                result = AdapterResult("error", error=last_error or "unknown SGLang server request error")
                results[index] = result
                if on_result:
                    on_result(sample, result)

        tasks = [asyncio.create_task(run_one(index)) for index in range(len(samples))]
        pending: set[asyncio.Task[None]] = set(tasks)
        try:
            while pending:
                if server_proc is not None and server_proc.poll() is not None:
                    fatal_error["error"] = f"local server exited with code {server_proc.returncode}"
                    fatal_event.set()
                if fatal_event.is_set():
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    break
                done, pending = await asyncio.wait(pending, timeout=1, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task.cancelled():
                        continue
                    exc = task.exception()
                    if exc is not None:
                        last_error = str(exc)
                        if self._is_fail_fast_error(last_error):
                            fatal_error["error"] = last_error
                            fatal_event.set()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        finally:
            await client.close()
        if fatal_event.is_set():
            raise RuntimeError(f"local server connection failed; stopped pending requests: {fatal_error.get('error')}")
        return [result if result is not None else AdapterResult("error", error="missing request result") for result in results]

    def run(self, sample: dict[str, Any], prompt: str) -> AdapterResult:
        return self.run_batch([sample], prompt)[0]

    def run_batch(
        self,
        samples: list[dict[str, Any]],
        prompt: str,
        on_result: Callable[[dict[str, Any], AdapterResult], None] | None = None,
    ) -> list[AdapterResult]:
        log_dir = Path(str(self.model_config.get("log_dir", "outputs/main_exp/logs")))
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{self.name}_server.stdout"
        stderr_path = log_dir / f"{self.name}_server.stderr"
        env = self._env(prompt)
        command = self._server_command()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.Popen(command, shell=True, text=True, stdout=stdout, stderr=stderr, env=env, preexec_fn=os.setsid)
            try:
                self._wait_ready(proc, stdout_path, stderr_path)
                results = asyncio.run(self._run_requests(samples, prompt, on_result, server_proc=proc))
                return results
            finally:
                self._stop_server(proc)


class LocalVLLMServerAdapter(LocalSGLangServerAdapter):
    def _server_served_model(self) -> str:
        """Allow requests to select a statically loaded LoRA by another name."""
        env_cfg = dict(self.model_config.get("env", {}))
        return str(
            self.model_config.get("server_served_model")
            or env_cfg.get("VLLM_BASE_SERVED_MODEL")
            or self._served_model()
        )

    def _env(self, prompt: str) -> dict[str, str]:
        env = os.environ.copy()
        env["FORMTSR_PROMPT"] = prompt
        env["FORMTSR_MODEL"] = str(self.model_config.get("model", ""))
        for key, value in dict(self.model_config.get("env", {})).items():
            env[str(key)] = str(value)
        site_packages = env.get("VLLM_SITE_PACKAGES")
        if site_packages:
            lib_dirs = [
                Path(env["VLLM_PYTHON_LIB_DIR"]) if env.get("VLLM_PYTHON_LIB_DIR") else None,
                Path(site_packages) / "torch" / "lib",
                Path(site_packages) / "nvidia" / "cu13" / "lib",
                Path(site_packages) / "nvidia" / "cuda_runtime" / "lib",
                Path(site_packages) / "nvidia" / "cuda_nvrtc" / "lib",
                Path(site_packages) / "nvidia" / "cuda_cupti" / "lib",
                Path(site_packages) / "nvidia" / "cublas" / "lib",
                Path(site_packages) / "nvidia" / "cudnn" / "lib",
                Path(site_packages) / "nvidia" / "cufft" / "lib",
                Path(site_packages) / "nvidia" / "curand" / "lib",
                Path(site_packages) / "nvidia" / "cusparse" / "lib",
                Path(site_packages) / "nvidia" / "cusolver" / "lib",
                Path(site_packages) / "nvidia" / "nccl" / "lib",
                Path(site_packages) / "nvidia" / "nvjitlink" / "lib",
                Path(site_packages) / "nvidia" / "nvtx" / "lib",
            ]
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                [str(path) for path in lib_dirs if path is not None and path.exists()] + ([existing] if existing else [])
            )
        return env

    def _append_internal_dp_args(self, args: list[str], env_cfg: dict[str, Any]) -> None:
        option_specs = [
            ("data_parallel_size", "VLLM_DATA_PARALLEL_SIZE", "--data-parallel-size"),
            ("data_parallel_size_local", "VLLM_DATA_PARALLEL_SIZE_LOCAL", "--data-parallel-size-local"),
            ("data_parallel_address", "VLLM_DATA_PARALLEL_ADDRESS", "--data-parallel-address"),
            ("data_parallel_rpc_port", "VLLM_DATA_PARALLEL_RPC_PORT", "--data-parallel-rpc-port"),
            ("data_parallel_backend", "VLLM_DATA_PARALLEL_BACKEND", "--data-parallel-backend"),
            ("data_parallel_start_rank", "VLLM_DATA_PARALLEL_START_RANK", "--data-parallel-start-rank"),
        ]
        for config_key, env_key, flag in option_specs:
            value = self.model_config.get(config_key) or env_cfg.get(env_key)
            if value not in (None, ""):
                args.extend([flag, str(value)])

        if _truthy(self.model_config.get("data_parallel_hybrid_lb") or env_cfg.get("VLLM_DATA_PARALLEL_HYBRID_LB"), default=False):
            args.append("--data-parallel-hybrid-lb")
        if _truthy(self.model_config.get("data_parallel_external_lb") or env_cfg.get("VLLM_DATA_PARALLEL_EXTERNAL_LB"), default=False):
            args.append("--data-parallel-external-lb")

    def _server_command(self) -> str:
        configured = self.model_config.get("server_command")
        if configured:
            return str(configured)
        env_cfg = dict(self.model_config.get("env", {}))
        model_path = str(env_cfg.get("VLLM_MODEL_PATH") or self.model_config.get("model_path") or "")
        if not model_path:
            raise ValueError("local_vllm_server_vlm requires server_command or VLLM_MODEL_PATH")
        host = str(self.model_config.get("host") or env_cfg.get("VLLM_HOST") or "127.0.0.1")
        port = int(self.model_config.get("port") or env_cfg.get("VLLM_PORT") or 8000)
        vllm_bin = str(env_cfg.get("VLLM_BIN") or "./.venv-vllm/bin/vllm")
        args = [
            vllm_bin,
            "serve",
            model_path,
            "--trust-remote-code",
            "--host",
            host,
            "--port",
            str(port),
            "--served-model-name",
            self._server_served_model(),
            "--tensor-parallel-size",
            str(self.model_config.get("tensor_parallel_size") or env_cfg.get("VLLM_TENSOR_PARALLEL_SIZE") or 1),
            "--gpu-memory-utilization",
            str(self.model_config.get("gpu_memory_utilization") or env_cfg.get("VLLM_GPU_MEMORY_UTILIZATION") or "0.90"),
        ]
        max_model_len = self.model_config.get("max_model_len") or env_cfg.get("VLLM_MAX_MODEL_LEN")
        if max_model_len:
            args.extend(["--max-model-len", str(max_model_len)])
        if _truthy(self.model_config.get("enforce_eager") or env_cfg.get("VLLM_ENFORCE_EAGER"), default=False):
            args.append("--enforce-eager")
        max_num_seqs = self.model_config.get("max_num_seqs") or env_cfg.get("VLLM_MAX_NUM_SEQS")
        if max_num_seqs:
            args.extend(["--max-num-seqs", str(max_num_seqs)])
        max_num_batched_tokens = self.model_config.get("max_num_batched_tokens") or env_cfg.get("VLLM_MAX_NUM_BATCHED_TOKENS")
        if max_num_batched_tokens:
            args.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])
        safetensors_load_strategy = self.model_config.get("safetensors_load_strategy") or env_cfg.get("VLLM_SAFETENSORS_LOAD_STRATEGY")
        if safetensors_load_strategy:
            args.extend(["--safetensors-load-strategy", str(safetensors_load_strategy)])
        hf_overrides = self.model_config.get("hf_overrides") or env_cfg.get("VLLM_HF_OVERRIDES")
        if hf_overrides:
            if isinstance(hf_overrides, (dict, list)):
                hf_overrides_arg = json.dumps(hf_overrides, ensure_ascii=False)
            else:
                hf_overrides_arg = str(hf_overrides)
            args.extend(["--hf-overrides", hf_overrides_arg])
        mm_processor_cache_gb = (
            self.model_config["mm_processor_cache_gb"]
            if "mm_processor_cache_gb" in self.model_config
            else env_cfg.get("VLLM_MM_PROCESSOR_CACHE_GB")
        )
        if mm_processor_cache_gb is not None:
            args.extend(["--mm-processor-cache-gb", str(mm_processor_cache_gb)])
        mm_processor_cache_type = self.model_config.get("mm_processor_cache_type") or env_cfg.get("VLLM_MM_PROCESSOR_CACHE_TYPE")
        if mm_processor_cache_type:
            args.extend(["--mm-processor-cache-type", str(mm_processor_cache_type)])
        limit_mm_per_prompt = self.model_config.get("limit_mm_per_prompt") or env_cfg.get("VLLM_LIMIT_MM_PER_PROMPT")
        if limit_mm_per_prompt:
            if isinstance(limit_mm_per_prompt, dict):
                limit_mm_arg = ",".join(f"{key}={value}" for key, value in limit_mm_per_prompt.items())
            else:
                limit_mm_arg = str(limit_mm_per_prompt)
            args.extend(["--limit-mm-per-prompt", limit_mm_arg])
        self._append_internal_dp_args(args, env_cfg)
        api_server_count = self.model_config.get("api_server_count") or env_cfg.get("VLLM_API_SERVER_COUNT")
        if api_server_count:
            args.extend(["--api-server-count", str(api_server_count)])
        _append_cli_args(args, self.model_config.get("vllm_extra_args", env_cfg.get("VLLM_EXTRA_ARGS")))
        return subprocess.list2cmdline(args)

    def _is_fail_fast_error(self, error: str) -> bool:
        if not _truthy(self.model_config.get("fail_fast_server_errors"), default=True):
            return False
        lower = error.lower()
        return any(
            marker in lower
            for marker in (
                "connection error",
                "connection refused",
                "server disconnected",
                "enginedead",
                "enginecore",
            )
        )

    def _vllm_log_tail(self, stdout_path: Path, stderr_path: Path, tail_chars: int = 8000) -> str:
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
        return (
            f"stdout log: {stdout_path}\n"
            f"{stdout[-tail_chars:]}\n"
            f"stderr log: {stderr_path}\n"
            f"{stderr[-tail_chars:]}"
        )

    def _wait_ready(self, proc: subprocess.Popen[str], stdout_path: Path, stderr_path: Path) -> None:
        timeout_seconds = int(self.model_config.get("server_start_timeout_seconds", 900))
        deadline = time.monotonic() + timeout_seconds
        last_error = ""
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                logs = self._vllm_log_tail(stdout_path, stderr_path)
                raise RuntimeError(f"vLLM server exited before ready with code {proc.returncode}\n{logs}")
            try:
                self._request_json("/v1/models", timeout=5)
                return
            except Exception as exc:
                last_error = str(exc)
                time.sleep(2)
        logs = self._vllm_log_tail(stdout_path, stderr_path)
        raise RuntimeError(f"vLLM server did not become ready after {timeout_seconds}s: {last_error}\n{logs}")


class ExternalVLLMServerAdapter(LocalVLLMServerAdapter):
    """Use an already-running vLLM endpoint without owning its lifecycle."""

    def run_batch(
        self,
        samples: list[dict[str, Any]],
        prompt: str,
        on_result: Callable[[dict[str, Any], AdapterResult], None] | None = None,
    ) -> list[AdapterResult]:
        return asyncio.run(self._run_requests(samples, prompt, on_result))


class LocalVLLMMultiServerAdapter(LocalVLLMServerAdapter):
    def _replicas(self) -> list[dict[str, Any]]:
        configured = self.model_config.get("replicas")
        if isinstance(configured, list) and configured:
            return [dict(item) for item in configured if isinstance(item, dict)]
        env_cfg = dict(self.model_config.get("env", {}))
        devices = str(env_cfg.get("CUDA_VISIBLE_DEVICES", "0,1")).split(",")
        base_port = int(self.model_config.get("port") or env_cfg.get("VLLM_PORT") or 8000)
        return [
            {"cuda_visible_devices": device.strip(), "port": base_port + index}
            for index, device in enumerate(devices)
            if device.strip()
        ]

    def _replica_env(self, prompt: str, replica: dict[str, Any]) -> dict[str, str]:
        env = self._env(prompt)
        if replica.get("cuda_visible_devices") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(replica["cuda_visible_devices"])
        for key in ("VLLM_DATA_PARALLEL_SIZE", "VLLM_DATA_PARALLEL_SIZE_LOCAL", "VLLM_API_SERVER_COUNT"):
            env.pop(key, None)
        for key, value in dict(replica.get("env", {})).items():
            env[str(key)] = str(value)
        return env

    def _replica_base_url(self, replica: dict[str, Any]) -> str:
        env_cfg = dict(self.model_config.get("env", {}))
        host = str(replica.get("host") or self.model_config.get("host") or env_cfg.get("VLLM_HOST") or "127.0.0.1")
        port = int(replica.get("port") or self.model_config.get("port") or env_cfg.get("VLLM_PORT") or 8000)
        return f"http://{host}:{port}"

    def _request_json_url(self, base_url: str, path: str, timeout: float = 5.0) -> dict[str, Any]:
        req = urllib.request.Request(base_url.rstrip("/") + path, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"unexpected response from {path}")
        return data

    def _server_command_for_replica(self, replica: dict[str, Any]) -> str:
        configured = replica.get("server_command") or self.model_config.get("server_command")
        if configured:
            return str(configured)
        env_cfg = dict(self.model_config.get("env", {}))
        model_path = str(replica.get("model_path") or env_cfg.get("VLLM_MODEL_PATH") or self.model_config.get("model_path") or "")
        if not model_path:
            raise ValueError("local_vllm_multi_server_vlm requires VLLM_MODEL_PATH")
        host = str(replica.get("host") or self.model_config.get("host") or env_cfg.get("VLLM_HOST") or "127.0.0.1")
        port = int(replica.get("port") or self.model_config.get("port") or env_cfg.get("VLLM_PORT") or 8000)
        vllm_bin = str(env_cfg.get("VLLM_BIN") or "./.venv-vllm/bin/vllm")
        args = [
            vllm_bin,
            "serve",
            model_path,
            "--trust-remote-code",
            "--host",
            host,
            "--port",
            str(port),
            "--served-model-name",
            self._server_served_model(),
            "--tensor-parallel-size",
            str(self.model_config.get("tensor_parallel_size") or env_cfg.get("VLLM_TENSOR_PARALLEL_SIZE") or 1),
            "--gpu-memory-utilization",
            str(self.model_config.get("gpu_memory_utilization") or env_cfg.get("VLLM_GPU_MEMORY_UTILIZATION") or "0.90"),
        ]
        max_model_len = self.model_config.get("max_model_len") or env_cfg.get("VLLM_MAX_MODEL_LEN")
        if max_model_len:
            args.extend(["--max-model-len", str(max_model_len)])
        if _truthy(self.model_config.get("enforce_eager") or env_cfg.get("VLLM_ENFORCE_EAGER"), default=False):
            args.append("--enforce-eager")
        max_num_seqs = self.model_config.get("max_num_seqs") or env_cfg.get("VLLM_MAX_NUM_SEQS")
        if max_num_seqs:
            args.extend(["--max-num-seqs", str(max_num_seqs)])
        max_num_batched_tokens = self.model_config.get("max_num_batched_tokens") or env_cfg.get("VLLM_MAX_NUM_BATCHED_TOKENS")
        if max_num_batched_tokens:
            args.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])
        safetensors_load_strategy = self.model_config.get("safetensors_load_strategy") or env_cfg.get("VLLM_SAFETENSORS_LOAD_STRATEGY")
        if safetensors_load_strategy:
            args.extend(["--safetensors-load-strategy", str(safetensors_load_strategy)])
        mm_processor_cache_gb = (
            self.model_config["mm_processor_cache_gb"]
            if "mm_processor_cache_gb" in self.model_config
            else env_cfg.get("VLLM_MM_PROCESSOR_CACHE_GB")
        )
        if mm_processor_cache_gb is not None:
            args.extend(["--mm-processor-cache-gb", str(mm_processor_cache_gb)])
        mm_processor_cache_type = self.model_config.get("mm_processor_cache_type") or env_cfg.get("VLLM_MM_PROCESSOR_CACHE_TYPE")
        if mm_processor_cache_type:
            args.extend(["--mm-processor-cache-type", str(mm_processor_cache_type)])
        _append_cli_args(args, replica.get("vllm_extra_args") or self.model_config.get("vllm_extra_args", env_cfg.get("VLLM_EXTRA_ARGS")))
        return subprocess.list2cmdline(args)

    def _wait_ready_url(self, base_url: str, proc: subprocess.Popen[str], stderr_path: Path) -> None:
        timeout_seconds = int(self.model_config.get("server_start_timeout_seconds", 900))
        deadline = time.monotonic() + timeout_seconds
        last_error = ""
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
                raise RuntimeError(f"vLLM replica exited before ready with code {proc.returncode}\n{stderr[-4000:]}")
            try:
                self._request_json_url(base_url, "/v1/models", timeout=5)
                return
            except Exception as exc:
                last_error = str(exc)
                time.sleep(2)
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
        raise RuntimeError(f"vLLM replica did not become ready after {timeout_seconds}s: {last_error}\n{stderr[-4000:]}")

    async def _run_requests_multi(
        self,
        samples: list[dict[str, Any]],
        prompt: str,
        on_result: Callable[[dict[str, Any], AdapterResult], None] | None,
        base_urls: list[str],
        server_procs: list[subprocess.Popen[str]],
    ) -> list[AdapterResult]:
        from openai import AsyncOpenAI  # type: ignore

        clients = [
            AsyncOpenAI(api_key="EMPTY", base_url=base_url.rstrip("/") + "/v1", timeout=float(self.model_config.get("request_timeout_seconds", 1800)))
            for base_url in base_urls
        ]
        concurrency = max(1, int(self.model_config.get("concurrency", self.model_config.get("batch_size", len(base_urls)))))
        per_replica_concurrency = max(
            1,
            int(
                self.model_config.get(
                    "per_replica_concurrency",
                    (concurrency + len(clients) - 1) // len(clients),
                )
            ),
        )
        retries = int(self.model_config.get("retries", 1))
        retry_sleep = float(self.model_config.get("retry_backoff_seconds", 10))
        results: list[AdapterResult | None] = [None] * len(samples)
        fatal_event = asyncio.Event()
        fatal_error: dict[str, str] = {}
        queue: asyncio.Queue[int] = asyncio.Queue()
        for index in range(len(samples)):
            queue.put_nowait(index)

        async def worker(client_index: int) -> None:
            client = clients[client_index]
            while not fatal_event.is_set():
                try:
                    index = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                sample = samples[index]
                if fatal_event.is_set():
                    queue.task_done()
                    return
                last_error: str | None = None
                try:
                    for attempt in range(retries + 1):
                        try:
                            response = await client.chat.completions.create(**self._payload(sample, prompt))
                            result = AdapterResult("ok", raw_response=self._extract_text(response))
                            results[index] = result
                            if on_result:
                                on_result(sample, result)
                            break
                        except Exception as exc:
                            last_error = str(exc)
                            if self._is_fail_fast_error(last_error):
                                fatal_error["error"] = last_error
                                fatal_event.set()
                                result = AdapterResult("error", error=last_error)
                                results[index] = result
                                if on_result:
                                    on_result(sample, result)
                                break
                            if attempt < retries:
                                await asyncio.sleep(retry_sleep)
                    else:
                        result = AdapterResult("error", error=last_error or "unknown vLLM replica request error")
                        results[index] = result
                        if on_result:
                            on_result(sample, result)
                finally:
                    queue.task_done()

        tasks = [
            asyncio.create_task(worker(client_index))
            for client_index in range(len(clients))
            for _ in range(per_replica_concurrency)
        ]
        pending: set[asyncio.Task[None]] = set(tasks)
        try:
            while pending:
                for proc in server_procs:
                    if proc.poll() is not None:
                        fatal_error["error"] = f"local vLLM replica exited with code {proc.returncode}"
                        fatal_event.set()
                        break
                if fatal_event.is_set():
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    break
                done, pending = await asyncio.wait(pending, timeout=1, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task.cancelled():
                        continue
                    exc = task.exception()
                    if exc is not None:
                        last_error = str(exc)
                        if self._is_fail_fast_error(last_error):
                            fatal_error["error"] = last_error
                            fatal_event.set()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        finally:
            for client in clients:
                await client.close()
        if fatal_event.is_set():
            raise RuntimeError(f"local vLLM replica connection failed; stopped pending requests: {fatal_error.get('error')}")
        return [result if result is not None else AdapterResult("error", error="missing request result") for result in results]

    def run_batch(
        self,
        samples: list[dict[str, Any]],
        prompt: str,
        on_result: Callable[[dict[str, Any], AdapterResult], None] | None = None,
    ) -> list[AdapterResult]:
        log_dir = Path(str(self.model_config.get("log_dir", "outputs/main_exp/logs")))
        log_dir.mkdir(parents=True, exist_ok=True)
        procs: list[subprocess.Popen[str]] = []
        handles: list[Any] = []
        base_urls: list[str] = []
        try:
            for index, replica in enumerate(self._replicas()):
                stdout_path = log_dir / f"{self.name}_server_replica{index}.stdout"
                stderr_path = log_dir / f"{self.name}_server_replica{index}.stderr"
                stdout = stdout_path.open("w", encoding="utf-8")
                stderr = stderr_path.open("w", encoding="utf-8")
                handles.extend([stdout, stderr])
                env = self._replica_env(prompt, replica)
                command = self._server_command_for_replica(replica)
                proc = subprocess.Popen(command, shell=True, text=True, stdout=stdout, stderr=stderr, env=env, preexec_fn=os.setsid)
                procs.append(proc)
                base_urls.append(self._replica_base_url(replica))
            for index, (base_url, proc) in enumerate(zip(base_urls, procs)):
                stderr_path = log_dir / f"{self.name}_server_replica{index}.stderr"
                self._wait_ready_url(base_url, proc, stderr_path)
            return asyncio.run(self._run_requests_multi(samples, prompt, on_result, base_urls, procs))
        finally:
            for proc in procs:
                self._stop_server(proc)
            for handle in handles:
                try:
                    handle.close()
                except OSError:
                    pass


class TraditionalTSRAdapter(BaseAdapter):
    def run(self, sample: dict[str, Any], prompt: str) -> AdapterResult:
        command = self.model_config.get("command")
        if not command:
            return AdapterResult(
                "provider_unavailable",
                error="traditional_tsr requires a command; form-specific metrics will be NA when not run",
            )
        return LocalHFAdapter(self.model_config).run(sample, prompt)


def make_adapter(model_config: dict[str, Any]) -> BaseAdapter:
    provider = str(model_config.get("provider", "")).strip()
    if provider == "openai_vlm":
        return OpenAIAdapter(model_config)
    if provider == "anthropic_vlm":
        return AnthropicAdapter(model_config)
    if provider == "gemini_vlm":
        return GeminiAdapter(model_config)
    if provider == "local_hf_vlm":
        return LocalHFAdapter(model_config)
    if provider == "local_sglang_server_vlm":
        return LocalSGLangServerAdapter(model_config)
    if provider == "local_vllm_server_vlm":
        return LocalVLLMServerAdapter(model_config)
    if provider == "external_vllm_server_vlm":
        return ExternalVLLMServerAdapter(model_config)
    if provider == "local_vllm_multi_server_vlm":
        return LocalVLLMMultiServerAdapter(model_config)
    if provider == "traditional_tsr":
        return TraditionalTSRAdapter(model_config)
    return MissingSDKAdapter(model_config)
