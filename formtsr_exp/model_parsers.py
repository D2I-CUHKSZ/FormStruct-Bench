from __future__ import annotations

import html
import json
import re
from html.parser import HTMLParser
from typing import Any


_DEEPSEEK_BBOX_RE = re.compile(r"^\s*([A-Za-z_][\w-]*)\s*\[\[([^\]]+)\]\]\s*$")
_UNLIMITED_DET_RE = re.compile(
    r"^\s*<\|det\|>\s*([A-Za-z_][\w-]*)\s*\[([^\]]+)\]\s*<\|/det\|>\s*(.*?)\s*$"
)
_UNLIMITED_REF_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.DOTALL)
_UNLIMITED_DET_BLOCK_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)


def _parse_bbox(raw: str) -> list[float] | None:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4:
        return None
    values: list[float] = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            return None
    return values


def _clean_ocr_text(text: str) -> str:
    cleaned = html.unescape(text).replace("\xa0", " ").strip()
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned)
    return cleaned.strip()


def clean_unlimited_ocr_output(raw: str) -> str:
    """Drop UnlimitedOCR grounding boxes while keeping OCR/markdown text."""

    refs = [_clean_ocr_text(match.group(1)) for match in _UNLIMITED_REF_RE.finditer(raw)]
    refs = [item for item in refs if item]
    if refs:
        return "\n".join(refs)
    cleaned = _UNLIMITED_DET_BLOCK_RE.sub("", raw)
    for token in ("<|ref|>", "<|/ref|>", "<|det|>", "<|/det|>"):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def parse_unlimited_ocr_markdown(raw: str) -> dict[str, Any]:
    return parse_deepseek_ocr_markdown(clean_unlimited_ocr_output(raw))


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _region_type(kind: str, text: str) -> str:
    kind_norm = kind.strip().lower()
    if text.startswith("# "):
        return "title"
    if text.startswith("## ") or kind_norm in {"sub_title", "subtitle", "heading", "header"}:
        return "section"
    cleaned = _clean_ocr_text(text)
    if ":" in cleaned or "：" in cleaned:
        return "field"
    if kind_norm in {"title"}:
        return "title"
    return "text"


def parse_deepseek_ocr_markdown(raw: str) -> dict[str, Any]:
    """Convert DeepSeek-OCR-2 markdown-with-bboxes output into benchmark JSON.

    DeepSeek-OCR-2's official prompt emits OCR text plus layout markers such as
    ``text[[x1, y1, x2, y2]]``. This parser does not infer missing semantic
    fields; it only preserves the observed OCR lines and detected boxes.
    """

    lines = raw.splitlines()
    regions: list[dict[str, Any]] = []
    ocr_lines: list[str] = []
    seen_ocr_lines: set[str] = set()

    index = 0
    region_index = 1
    while index < len(lines):
        line = lines[index].strip()
        det_match = _UNLIMITED_DET_RE.match(line)
        if det_match:
            kind = det_match.group(1)
            bbox = _parse_bbox(det_match.group(2))
            cleaned_text = _clean_ocr_text(det_match.group(3))
            if cleaned_text:
                if cleaned_text not in seen_ocr_lines:
                    ocr_lines.append(cleaned_text)
                    seen_ocr_lines.add(cleaned_text)
                if bbox is not None:
                    regions.append(
                        {
                            "id": f"r{region_index}",
                            "type": _region_type(kind, cleaned_text),
                            "bbox": bbox,
                            "text": cleaned_text,
                        }
                    )
                    region_index += 1
            index += 1
            continue

        match = _DEEPSEEK_BBOX_RE.match(line)
        if not match:
            cleaned = _clean_ocr_text(line)
            if cleaned and cleaned not in seen_ocr_lines:
                ocr_lines.append(cleaned)
                seen_ocr_lines.add(cleaned)
            index += 1
            continue

        kind = match.group(1)
        bbox = _parse_bbox(match.group(2))
        text = ""
        next_index = index + 1
        while next_index < len(lines):
            candidate = lines[next_index].strip()
            if not candidate:
                next_index += 1
                continue
            if _DEEPSEEK_BBOX_RE.match(candidate):
                break
            text = candidate
            break

        cleaned_text = _clean_ocr_text(text)
        if cleaned_text:
            if cleaned_text not in seen_ocr_lines:
                ocr_lines.append(cleaned_text)
                seen_ocr_lines.add(cleaned_text)
            if bbox is not None:
                regions.append(
                    {
                        "id": f"r{region_index}",
                        "type": _region_type(kind, text),
                        "bbox": bbox,
                        "text": cleaned_text,
                    }
                )
                region_index += 1
        index = max(next_index + 1, index + 1)

    return {
        "regions": regions,
        "widgets": [],
        "local_grids": [],
        "cells": [],
        "line_item_groups": [],
        "relations": [],
        "answer": {"ocr_lines": ocr_lines},
    }


def _bbox_from_paddle(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        for key in ("block_bbox", "bbox", "box", "coordinate", "poly", "polygon", "points"):
            if key in value:
                parsed = _bbox_from_paddle(value[key])
                if parsed is not None:
                    return parsed
        return None
    if not isinstance(value, list):
        return None
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        x1, y1, x2, y2 = [float(item) for item in value]
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    points: list[tuple[float, float]] = []
    if all(isinstance(item, list) and len(item) >= 2 for item in value):
        for item in value:
            try:
                points.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
    elif len(value) >= 8 and len(value) % 2 == 0:
        try:
            flat = [float(item) for item in value]
        except (TypeError, ValueError):
            flat = []
        points = list(zip(flat[0::2], flat[1::2]))
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _paddle_region_type(label: str, text: str) -> str:
    label_norm = label.strip().lower()
    if label_norm in {"doc_title", "figure_title", "title"}:
        return "title"
    if label_norm in {"paragraph_title", "header", "heading"}:
        return "section"
    if label_norm in {"table"}:
        return "table"
    if label_norm in {"image", "figure", "chart", "seal"}:
        return label_norm
    if label_norm in {"formula", "inline_formula", "display_formula"}:
        return "formula"
    if ":" in text or "：" in text:
        return "field"
    return "text"


class _TableTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.cells: list[str] = []
        self._current_row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._finish_cell()
            self._finish_row()
            self._current_row = []
        elif tag in {"td", "th"}:
            self._finish_cell()
            if self._current_row is None:
                self._current_row = []
            self._cell_parts = []
        elif tag == "br" and self._cell_parts is not None:
            self._cell_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"}:
            self._finish_cell()
        elif tag == "tr":
            self._finish_cell()
            self._finish_row()

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def close(self) -> None:
        super().close()
        self._finish_cell()
        self._finish_row()

    def _finish_cell(self) -> None:
        if self._cell_parts is None:
            return
        text = _clean_extracted_text("".join(self._cell_parts))
        if text:
            self.cells.append(text)
            if self._current_row is None:
                self._current_row = []
            self._current_row.append(text)
        self._cell_parts = None

    def _finish_row(self) -> None:
        if self._current_row:
            self.rows.append(self._current_row)
        self._current_row = None


_HTML_TABLE_RE = re.compile(r"<\s*/?\s*(table|tr|td|th)\b", re.IGNORECASE)


def _looks_like_html_table(text: str) -> bool:
    return bool(_HTML_TABLE_RE.search(text))


def _clean_extracted_text(text: str) -> str:
    cleaned = _clean_ocr_text(text)
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_table_text(text: str) -> tuple[list[str], list[list[str]]]:
    parser = _TableTextParser()
    parser.feed(text)
    parser.close()
    return parser.cells, parser.rows


def _add_unique_line(lines: list[str], seen: set[str], text: str) -> None:
    cleaned = _clean_extracted_text(text)
    if cleaned and cleaned not in seen:
        lines.append(cleaned)
        seen.add(cleaned)


def _bbox_iou_from_lists(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _has_duplicate_region(regions: list[dict[str, Any]], bbox: list[float], label: str) -> bool:
    label_norm = label.strip().lower()
    for region in regions:
        existing_bbox = _bbox_from_paddle(region.get("bbox"))
        if existing_bbox is None:
            continue
        existing_label = str(region.get("source_label") or "").strip().lower()
        if existing_label == label_norm and _bbox_iou_from_lists(existing_bbox, bbox) >= 0.98:
            return True
    return False


def _paddle_page_payload(page: Any) -> dict[str, Any]:
    if isinstance(page, dict) and isinstance(page.get("res"), dict):
        return page["res"]
    return page if isinstance(page, dict) else {}


def _paddle_pages_from_raw(raw: str) -> list[dict[str, Any]]:
    loaded = json.loads(raw)
    if isinstance(loaded, dict):
        pages = loaded.get("pages") or loaded.get("paddleocr_pages")
        if isinstance(pages, list):
            return [_paddle_page_payload(page) for page in pages]
        return [_paddle_page_payload(loaded)]
    if isinstance(loaded, list):
        return [_paddle_page_payload(page) for page in loaded]
    return []


def parse_paddleocr_vl_pipeline(raw: str) -> dict[str, Any]:
    """Convert official PaddleOCR-VL pipeline JSON into benchmark JSON.

    This parser preserves observed PaddleOCR-VL blocks. It maps
    ``parsing_res_list`` and non-duplicate ``layout_det_res.boxes`` items to
    regions, and exposes recognized text plus HTML table cells under
    ``answer``. It does not infer FormTSR field paths.
    """

    pages = _paddle_pages_from_raw(raw)
    regions: list[dict[str, Any]] = []
    ocr_lines: list[str] = []
    table_cells: list[str] = []
    table_rows: list[str] = []
    seen_ocr_lines: set[str] = set()

    region_index = 1
    for page_index, page in enumerate(pages):
        for block_index, block in enumerate(as_list(page.get("parsing_res_list"))):
            if not isinstance(block, dict):
                continue
            raw_text = str(block.get("block_content") or block.get("content") or "")
            text = _clean_extracted_text(raw_text)
            if _looks_like_html_table(raw_text):
                cells, rows = _extract_table_text(raw_text)
                table_cells.extend(cells)
                table_rows.extend(" | ".join(row) for row in rows if row)
                for row in rows:
                    _add_unique_line(ocr_lines, seen_ocr_lines, " | ".join(row))
            else:
                for line in text.splitlines() or [text]:
                    _add_unique_line(ocr_lines, seen_ocr_lines, line)
            bbox = _bbox_from_paddle(block)
            if bbox is None:
                continue
            label = str(block.get("block_label") or block.get("label") or "text")
            region = {
                "id": str(block.get("global_block_id") or block.get("block_id") or f"r{region_index}"),
                "type": _paddle_region_type(label, text),
                "bbox": bbox,
                "text": text,
            }
            region["source_label"] = label
            region["page_index"] = page.get("page_index", page_index)
            region["source_block_index"] = block_index
            regions.append(region)
            region_index += 1

        layout_det_res = page.get("layout_det_res") if isinstance(page.get("layout_det_res"), dict) else {}
        for det_index, det_box in enumerate(as_list(layout_det_res.get("boxes"))):
            if not isinstance(det_box, dict):
                continue
            bbox = _bbox_from_paddle(det_box)
            if bbox is None:
                continue
            label = str(det_box.get("label") or det_box.get("block_label") or "text")
            if _has_duplicate_region(regions, bbox, label):
                continue
            region = {
                "id": f"det{page_index + 1}_{det_index + 1}",
                "type": _paddle_region_type(label, ""),
                "bbox": bbox,
                "text": "",
                "source_label": label,
                "page_index": page.get("page_index", page_index),
                "source": "layout_det_res",
                "source_block_index": det_index,
            }
            if "score" in det_box:
                region["score"] = det_box["score"]
            regions.append(region)
            region_index += 1

    return {
        "regions": regions,
        "widgets": [],
        "local_grids": [],
        "cells": [],
        "line_item_groups": [],
        "relations": [],
        "answer": {
            "ocr_lines": ocr_lines,
            "table_cells": table_cells,
            "table_rows": table_rows,
        },
    }
