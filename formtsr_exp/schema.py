from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _walk(value: Any, path: str, paths: dict[str, Counter[str]], keys: Counter[str], depth: int = 0) -> None:
    paths[path][_type_name(value)] += 1
    if isinstance(value, dict):
        for key, child in value.items():
            keys[str(key)] += 1
            child_path = f"{path}.{key}" if path else str(key)
            _walk(child, child_path, paths, keys, depth + 1)
    elif isinstance(value, list):
        for child in value[:20]:
            _walk(child, f"{path}[]", paths, keys, depth + 1)


def summarize_schema(index_rows: list[dict[str, Any]], out_path: Path, *, max_files: int = 25) -> dict[str, Any]:
    paths: dict[str, Counter[str]] = defaultdict(Counter)
    keys: Counter[str] = Counter()
    files: list[str] = []
    top_level_types: Counter[str] = Counter()
    parse_errors: list[dict[str, str]] = []

    for row in index_rows[:max_files]:
        label_path = Path(row["label_path"])
        files.append(str(label_path))
        try:
            payload = read_json(label_path)
        except Exception as exc:
            parse_errors.append({"label_path": str(label_path), "error": str(exc)})
            continue
        top_level_types[_type_name(payload)] += 1
        _walk(payload, "$", paths, keys)

    summary = {
        "source": "FormTSR/datasets/*/*/answer.json",
        "inspected_files": files,
        "n_inspected": len(files),
        "top_level_types": dict(top_level_types),
        "frequent_keys": [{"key": key, "count": count} for key, count in keys.most_common(100)],
        "paths": [
            {"path": path, "types": dict(counter)}
            for path, counter in sorted(paths.items(), key=lambda item: item[0])[:500]
        ],
        "parse_errors": parse_errors,
        "observed_label_format": (
            "The sampled answer.json files are semantic key-value trees. They do not contain explicit "
            "region boxes, cell grids, widget groups, or relation edge annotations unless such fields "
            "appear in a particular sample."
        ),
        "metric_mapping": {
            "TSR-path": "Strict field accuracy on answer.json leaf values with the full GT path preserved.",
            "VAcc": "Path-independent value accuracy over non-empty answer.json leaf values, using normalized multiset matching.",
            "R-F1": "Formal reporting uses structure_metrics_report: run-level bbox normalization, canonical type compatibility, and full-scope page-macro F1.",
            "LIG-F1": "Formal reporting uses corrected bbox normalization and page-macro F1 over all GT-applicable pages.",
            "WAcc": "Uses metadata-selected widget fields and compares corresponding answer values between prediction and GT.",
        },
    }
    write_json(out_path, summary)
    return summary
