from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_ALIASES = {
    "local_sglang_vlm": "Qwen3.6-35B-A3B",
    "local_sglang_vlm_aligned_metadata": "Qwen3.6-35B-A3B_aligned_metadata",
}


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _minimal_yaml(text: str) -> dict[str, Any]:
    # Supports the simple config/main_experiment.yaml shape shipped with this package.
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    last_key_at_indent: dict[int, tuple[Any, str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            item_text = stripped[2:].strip()
            if not isinstance(parent, list):
                container_parent, key = last_key_at_indent[indent]
                new_list: list[Any] = []
                container_parent[key] = new_list
                parent = new_list
                stack.append((indent, parent))
            if item_text:
                if ":" in item_text:
                    key, value = item_text.split(":", 1)
                    item: dict[str, Any] = {key.strip(): _parse_scalar(value)}
                    parent.append(item)
                    stack.append((indent + 2, item))
                else:
                    parent.append(_parse_scalar(item_text))
            else:
                item = {}
                parent.append(item)
                stack.append((indent + 2, item))
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
        else:
            parent[key] = {}
            last_key_at_indent[indent + 2] = (parent, key)
            stack.append((indent, parent[key]))
    return root


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
        raise ValueError(f"config root must be an object: {path}")
    except ModuleNotFoundError:
        return _minimal_yaml(text)


def enabled_models(config: dict[str, Any], requested: set[str] | None = None) -> list[dict[str, Any]]:
    models = config.get("models", [])
    if isinstance(models, dict):
        models = list(models.values())
    normalized_requested = None
    if requested:
        normalized_requested = {MODEL_ALIASES.get(name, name) for name in requested}
    result = []
    for model in models:
        if not isinstance(model, dict):
            continue
        name = str(model.get("name", ""))
        if normalized_requested and name not in normalized_requested:
            continue
        if normalized_requested or model.get("enabled", True):
            result.append(model)
    return result
