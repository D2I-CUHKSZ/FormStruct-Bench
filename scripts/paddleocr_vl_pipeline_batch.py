from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def truthy(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaddleOCR-VL pipeline batch inference for FormTSR.")
    parser.add_argument("--pipeline-version", default=os.environ.get("PADDLEOCR_PIPELINE_VERSION", "v1.6"))
    parser.add_argument("--vl-rec-model-name", default=os.environ.get("PADDLEOCR_VL_REC_MODEL_NAME", "PaddleOCR-VL-1.6-0.9B"))
    parser.add_argument("--vl-rec-model-dir", default=os.environ.get("PADDLEOCR_VL_REC_MODEL_DIR", ""))
    parser.add_argument("--vl-rec-backend", default=os.environ.get("PADDLEOCR_VL_REC_BACKEND", "sglang-server"))
    parser.add_argument("--vl-rec-server-url", default=os.environ.get("PADDLEOCR_VL_REC_SERVER_URL", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--vl-rec-api-model-name", default=os.environ.get("PADDLEOCR_VL_REC_API_MODEL_NAME", "PaddleOCR-VL-1.6-0.9B"))
    parser.add_argument("--vl-rec-api-key", default=os.environ.get("PADDLEOCR_VL_REC_API_KEY", "EMPTY"))
    parser.add_argument("--vl-rec-max-concurrency", type=int, default=int(os.environ.get("PADDLEOCR_VL_REC_MAX_CONCURRENCY", "8")))
    parser.add_argument("--layout-detection-model-dir", default=os.environ.get("PADDLEOCR_LAYOUT_DETECTION_MODEL_DIR", ""))
    parser.add_argument(
        "--use-ocr-for-image-block",
        default=os.environ.get("PADDLEOCR_USE_OCR_FOR_IMAGE_BLOCK", ""),
        help="Whether PaddleOCR-VL should OCR layout blocks classified as image. Use true/false or leave empty for pipeline default.",
    )
    parser.add_argument(
        "--use-layout-detection",
        default=os.environ.get("PADDLEOCR_USE_LAYOUT_DETECTION", ""),
        help="Whether to run PaddleOCR-VL layout detection before recognition. Use true/false or leave empty for pipeline default.",
    )
    parser.add_argument(
        "--merge-layout-blocks",
        default=os.environ.get("PADDLEOCR_MERGE_LAYOUT_BLOCKS", ""),
        help="Whether to merge detected layout blocks. Use true/false or leave empty for pipeline default.",
    )
    parser.add_argument(
        "--format-block-content",
        default=os.environ.get("PADDLEOCR_FORMAT_BLOCK_CONTENT", ""),
        help="Whether to ask PaddleOCR-VL to format recognized block content. Use true/false or leave empty for pipeline default.",
    )
    parser.add_argument("--prompt-label", default=os.environ.get("PADDLEOCR_PROMPT_LABEL", ""))
    parser.add_argument("--layout-threshold", default=os.environ.get("PADDLEOCR_LAYOUT_THRESHOLD", ""))
    parser.add_argument("--layout-nms", default=os.environ.get("PADDLEOCR_LAYOUT_NMS", ""))
    parser.add_argument("--layout-unclip-ratio", default=os.environ.get("PADDLEOCR_LAYOUT_UNCLIP_RATIO", ""))
    parser.add_argument("--layout-merge-bboxes-mode", default=os.environ.get("PADDLEOCR_LAYOUT_MERGE_BBOXES_MODE", ""))
    parser.add_argument("--top-p", default=os.environ.get("PADDLEOCR_TOP_P", ""))
    parser.add_argument("--repetition-penalty", default=os.environ.get("PADDLEOCR_REPETITION_PENALTY", ""))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("PADDLEOCR_MAX_NEW_TOKENS", os.environ.get("HF_MAX_TOKENS", "4096"))))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("PADDLEOCR_TEMPERATURE", os.environ.get("HF_TEMPERATURE", "0"))))
    parser.add_argument("--server-host", default=os.environ.get("PADDLEOCR_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--server-port", type=int, default=int(os.environ.get("PADDLEOCR_SERVER_PORT", "8080")))
    parser.add_argument("--server-timeout", type=int, default=int(os.environ.get("PADDLEOCR_SERVER_START_TIMEOUT", "1200")))
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
            raise ValueError("FORMTSR_BATCH must contain a JSON list")
        return [row for row in rows if isinstance(row, dict)]
    image_path = os.environ.get("FORMTSR_IMAGE_PATH", "")
    sample_id = os.environ.get("FORMTSR_SAMPLE_ID", "")
    if image_path and sample_id:
        return [{"sample_id": sample_id, "image_path": image_path}]
    raise ValueError("missing FORMTSR_BATCH_FILE/FORMTSR_BATCH or FORMTSR_IMAGE_PATH/FORMTSR_SAMPLE_ID")


def emit_jsonl(path_value: str, row: dict[str, Any]) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        if truthy(os.environ.get("FORMTSR_RESULT_FSYNC")):
            os.fsync(fh.fileno())


def server_models_url(args: argparse.Namespace) -> str:
    return f"http://{args.server_host}:{args.server_port}/v1/models"


def server_ready(args: argparse.Namespace) -> bool:
    try:
        with urllib.request.urlopen(server_models_url(args), timeout=2) as response:
            return 200 <= int(response.status) < 500
    except (OSError, urllib.error.URLError):
        return False


def wait_for_server(args: argparse.Namespace, proc: subprocess.Popen[str] | None = None) -> None:
    deadline = time.monotonic() + max(1, args.server_timeout)
    while time.monotonic() < deadline:
        if server_ready(args):
            return
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"PaddleOCR SGLang server exited early with code {proc.returncode}")
        time.sleep(2)
    raise TimeoutError(f"PaddleOCR SGLang server did not become ready at {server_models_url(args)}")


def default_server_command(args: argparse.Namespace) -> str:
    model_dir_arg = f" --model_dir {shell_quote(args.vl_rec_model_dir)}" if args.vl_rec_model_dir else ""
    return (
        f"{shell_quote(sys.executable)} -m paddlex.inference.genai.server"
        f" --model_name {shell_quote(args.vl_rec_model_name)}"
        f"{model_dir_arg}"
        " --backend sglang"
        f" --host {shell_quote(args.server_host)}"
        f" --port {args.server_port}"
    )


def shell_quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))


def optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def start_server_if_requested(args: argparse.Namespace) -> subprocess.Popen[str] | None:
    if server_ready(args):
        print(f"PaddleOCR SGLang server already ready at {server_models_url(args)}", file=sys.stderr)
        return None
    if not truthy(os.environ.get("PADDLEOCR_START_SERVER")):
        wait_for_server(args)
        return None

    log_dir = Path(os.environ.get("PADDLEOCR_SERVER_LOG_DIR", "outputs/main_exp/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "paddleocr_vl_sglang_server.stdout"
    stderr_path = log_dir / "paddleocr_vl_sglang_server.stderr"
    command = os.environ.get("PADDLEOCR_SERVER_COMMAND") or default_server_command(args)
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    stdout_fh = stdout_path.open("a", encoding="utf-8")
    stderr_fh = stderr_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        shell=True,
        text=True,
        stdout=stdout_fh,
        stderr=stderr_fh,
        env=env,
        start_new_session=True,
    )
    wait_for_server(args, proc=proc)
    print(f"PaddleOCR SGLang server ready at {server_models_url(args)}", file=sys.stderr)
    return proc


def stop_server(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None or truthy(os.environ.get("PADDLEOCR_KEEP_SERVER")):
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait(timeout=10)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "mode") and hasattr(value, "size"):
        return {
            "_type": value.__class__.__name__,
            "mode": str(getattr(value, "mode", "")),
            "size": json_safe(getattr(value, "size", "")),
        }
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def result_json(result: Any) -> dict[str, Any]:
    data = getattr(result, "json", None)
    if callable(data):
        data = data()
    if isinstance(data, dict):
        return json_safe(data)
    if isinstance(result, dict):
        return json_safe(result)
    return {"text": str(result)}


def result_markdown(result: Any) -> Any:
    data = getattr(result, "markdown", None)
    if callable(data):
        data = data()
    return json_safe(data)


class PaddleOCRVLPipelineRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        from paddleocr import PaddleOCRVL

        self.optional_predict_kwargs: dict[str, Any] = {}
        kwargs: dict[str, Any] = {
            "pipeline_version": args.pipeline_version,
            "vl_rec_backend": args.vl_rec_backend,
            "vl_rec_server_url": args.vl_rec_server_url,
            "vl_rec_max_concurrency": args.vl_rec_max_concurrency,
            "vl_rec_api_model_name": args.vl_rec_api_model_name,
            "vl_rec_api_key": args.vl_rec_api_key,
        }
        if args.vl_rec_model_name:
            kwargs["vl_rec_model_name"] = args.vl_rec_model_name
        if args.vl_rec_model_dir:
            kwargs["vl_rec_model_dir"] = args.vl_rec_model_dir
        if args.layout_detection_model_dir:
            kwargs["layout_detection_model_dir"] = args.layout_detection_model_dir
        for option_name, cli_value in (
            ("use_ocr_for_image_block", args.use_ocr_for_image_block),
            ("use_layout_detection", args.use_layout_detection),
            ("merge_layout_blocks", args.merge_layout_blocks),
            ("format_block_content", args.format_block_content),
        ):
            parsed = optional_bool(cli_value)
            if parsed is not None:
                kwargs[option_name] = parsed
                self.optional_predict_kwargs[option_name] = parsed
        layout_nms = optional_bool(args.layout_nms)
        if layout_nms is not None:
            self.optional_predict_kwargs["layout_nms"] = layout_nms
        for option_name, cli_value in (
            ("layout_threshold", args.layout_threshold),
            ("layout_unclip_ratio", args.layout_unclip_ratio),
            ("top_p", args.top_p),
            ("repetition_penalty", args.repetition_penalty),
        ):
            parsed_float = optional_float(cli_value)
            if parsed_float is not None:
                self.optional_predict_kwargs[option_name] = parsed_float
        if args.prompt_label.strip():
            self.optional_predict_kwargs["prompt_label"] = args.prompt_label.strip()
        if args.layout_merge_bboxes_mode.strip():
            self.optional_predict_kwargs["layout_merge_bboxes_mode"] = args.layout_merge_bboxes_mode.strip()
        self.args = args
        self.pipeline = PaddleOCRVL(**kwargs)

    def run_one(self, image_path: str) -> str:
        predict_kwargs: dict[str, Any] = {
            "max_new_tokens": self.args.max_new_tokens,
            "temperature": self.args.temperature,
        }
        predict_kwargs.update(self.optional_predict_kwargs)
        results = self.pipeline.predict(image_path, **predict_kwargs)
        pages = [result_json(result) for result in results]
        markdown = [result_markdown(result) for result in results]
        payload = {
            "backend": "paddleocr_vl_pipeline",
            "pipeline_version": self.args.pipeline_version,
            "vl_rec_backend": self.args.vl_rec_backend,
            "pages": pages,
            "markdown": markdown,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def write_final_output(rows_out: list[dict[str, Any]]) -> None:
    output_path = os.environ.get("FORMTSR_BATCH_OUTPUT")
    if output_path:
        Path(output_path).write_text(json.dumps(rows_out, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(rows_out, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    batch = load_batch()
    jsonl_path = os.environ.get("FORMTSR_BATCH_OUTPUT_JSONL", "")
    started = time.monotonic()
    rows_out: list[dict[str, Any]] = []
    server_proc: subprocess.Popen[str] | None = None

    try:
        server_proc = start_server_if_requested(args)
        runner = PaddleOCRVLPipelineRunner(args)
    except Exception as exc:
        rows_out = [
            {"sample_id": str(row.get("sample_id", "")), "status": "error", "raw_response": "", "error": f"load failed: {exc}"}
            for row in batch
        ]
        write_final_output(rows_out)
        print(f"PaddleOCR-VL pipeline load failed: {exc}", file=sys.stderr)
        stop_server(server_proc)
        return 1

    try:
        for index, row in enumerate(batch, start=1):
            sample_id = str(row.get("sample_id", ""))
            try:
                raw = runner.run_one(str(row["image_path"]))
                result = {"sample_id": sample_id, "status": "ok", "raw_response": raw}
            except Exception as exc:
                result = {"sample_id": sample_id, "status": "error", "raw_response": "", "error": str(exc)}
            rows_out.append(result)
            emit_jsonl(jsonl_path, result)
            if index % max(1, int(os.environ.get("FORMTSR_BATCH_SIZE", "1"))) == 0 or index == len(batch):
                elapsed = time.monotonic() - started
                print(f"paddleocr_vl_pipeline progress: {index}/{len(batch)} finished in {elapsed:.1f}s", file=sys.stderr)
        write_final_output(rows_out)
        return 0
    finally:
        stop_server(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
