from __future__ import annotations

import json
import re
from typing import Any


FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    text = THINK_RE.sub("", text)
    text = text.replace("<think>", "").replace("</think>", "")
    match = FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_balanced_json(text: str) -> str:
    text = _strip_fences(text)
    start_positions = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
    if not start_positions:
        return text
    start = min(start_positions)
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text[start:]


def _repair_common(text: str) -> str:
    text = text.strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def parse_json_response(raw: str) -> tuple[Any | None, dict[str, Any]]:
    candidate = _repair_common(_extract_balanced_json(raw))
    try:
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            return None, {
                "valid": False,
                "error": f"top-level JSON must be an object, got {type(parsed).__name__}",
                "candidate": candidate[:4000],
            }
        return parsed, {"valid": True, "error": None, "candidate": candidate}
    except json.JSONDecodeError as exc:
        return None, {
            "valid": False,
            "error": f"{exc.msg} at line {exc.lineno} column {exc.colno}",
            "candidate": candidate[:4000],
        }
