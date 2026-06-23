"""Helpers for adapting PyMuPDF4LLM layout JSON to ParseBench layout output."""

from __future__ import annotations

import html
import json
from typing import Any

from parse_bench.schemas.layout_detection_output import (
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
    LayoutTableContent,
    LayoutTextContent,
)


def layout_json_to_layout_output(
    layout_json: Any,
    *,
    example_id: str,
    pipeline_name: str,
    markdown: str = "",
) -> LayoutOutput:
    parsed = _parse_layout_json(layout_json)
    pages = parsed.get("pages") if isinstance(parsed, dict) else []
    if not isinstance(pages, list):
        pages = []

    page_items = [(_page_number(page, index), page) for index, page in enumerate(pages) if isinstance(page, dict)]
    predictions = _layout_predictions(page_items)
    effective_markdown = markdown if markdown.strip() else _pages_markdown(page_items)

    return LayoutOutput(
        task_type="layout_detection",
        example_id=example_id,
        pipeline_name=pipeline_name,
        model=LayoutDetectionModel.PYMUPDF4LLM,
        image_width=max(int(round(_first_dimension(page_items, "width"))), 1),
        image_height=max(int(round(_first_dimension(page_items, "height"))), 1),
        predictions=predictions,
        markdown=effective_markdown,
    )


def _parse_layout_json(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return value if isinstance(value, dict) else {}


def _page_number(page: dict[str, Any], page_index: int) -> int:
    value = page.get("page_number")
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0:
        return int(value)
    return page_index + 1


def _first_dimension(pages: list[tuple[int, dict[str, Any]]], key: str) -> float:
    for _page_number, page in pages:
        value = page.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return 1.0


def _layout_predictions(pages: list[tuple[int, dict[str, Any]]]) -> list[LayoutPrediction]:
    predictions: list[LayoutPrediction] = []
    for page_number_value, page in pages:
        boxes = page.get("boxes") or []
        if not isinstance(boxes, list):
            continue
        for box in boxes:
            if not isinstance(box, dict):
                continue
            label = str(box.get("boxclass") or "").strip()
            bbox = _box_bbox(box)
            if not label or bbox is None:
                continue
            predictions.append(
                LayoutPrediction(
                    bbox=bbox,
                    score=1.0,
                    label=label,
                    page=page_number_value,
                    content=_box_content(label, box),
                    provider_metadata={"order_index": len(predictions), "source": "pymupdf4llm.to_json"},
                )
            )
    return predictions


def _box_bbox(box: dict[str, Any]) -> list[float] | None:
    try:
        x0 = float(box["x0"])
        y0 = float(box["y0"])
        x1 = float(box["x1"])
        y1 = float(box["y1"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (x0 < x1 and y0 < y1):
        return None
    return [x0, y0, x1, y1]


def _box_content(label: str, box: dict[str, Any]) -> LayoutTextContent | LayoutTableContent | None:
    table = box.get("table")
    if label == "table" and isinstance(table, dict):
        table_html = _table_to_html(table)
        if table_html:
            return LayoutTableContent(html=table_html)
        markdown = table.get("markdown")
        if isinstance(markdown, str) and markdown.strip():
            return LayoutTextContent(text=markdown)
    text = _box_text(box)
    if text:
        return LayoutTextContent(text=text)
    return None


def _box_text(box: dict[str, Any]) -> str:
    parts: list[str] = []
    textlines = box.get("textlines") or []
    if not isinstance(textlines, list):
        return ""
    for line in textlines:
        if not isinstance(line, dict):
            continue
        spans = line.get("spans") or []
        if not isinstance(spans, list):
            continue
        for span in spans:
            if isinstance(span, dict) and span.get("text"):
                parts.append(str(span["text"]))
    return " ".join(parts).strip()


def _pages_markdown(pages: list[tuple[int, dict[str, Any]]]) -> str:
    page_parts: list[str] = []
    for _page_number, page in pages:
        boxes = page.get("boxes") or []
        if not isinstance(boxes, list):
            continue
        block_parts = [_box_markdown(box) for box in boxes if isinstance(box, dict)]
        block_parts = [block for block in block_parts if block]
        if block_parts:
            page_parts.append("\n\n".join(block_parts))
    return "\n\n".join(page_parts)


def _box_markdown(box: dict[str, Any]) -> str:
    table = box.get("table")
    if str(box.get("boxclass") or "").strip() == "table" and isinstance(table, dict):
        markdown = table.get("markdown")
        if isinstance(markdown, str) and markdown.strip():
            return markdown.strip()
        rows = table.get("extract")
        if isinstance(rows, list):
            rendered_rows = [
                "| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |"
                for row in rows
                if isinstance(row, list)
            ]
            if rendered_rows:
                return "\n".join(rendered_rows)
    return _box_text(box)


def _table_to_html(table: dict[str, Any]) -> str:
    rows = table.get("extract")
    if not isinstance(rows, list) or not rows:
        return ""
    html_rows: list[str] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        cells = "".join(f"<td>{html.escape('' if cell is None else str(cell))}</td>" for cell in row)
        html_rows.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(html_rows) + "</table>" if html_rows else ""
