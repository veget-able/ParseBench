"""Provider for PyMuPDF4LLM PARSE."""

import json
from datetime import datetime
import os
from pathlib import Path
from typing import Any

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    PageIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType


def _box_text(box: dict[str, Any]) -> str:
    """Aggregate text from a PyMuPDF4LLM layout box's textlines/spans."""
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


@register_provider("pymupdf4llm")
class PyMuPDF4LLMProvider(Provider):
    """Provider for PyMuPDF4LLM (markdown). AGPL — runtime dep only."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._use_tgif = self.base_config.get("use_tgif")
        self._activate_layout = bool(self.base_config.get("activate_layout", self._use_tgif is not None))
        if self._use_tgif is not None:
            os.environ["USE_TGIF"] = str(self._use_tgif)

    def _import_pymupdf4llm(self) -> Any:
        try:
            if self._activate_layout:
                import pymupdf.layout  # noqa: F401
            import pymupdf4llm
        except ImportError as e:
            raise ProviderConfigError("pymupdf4llm not installed. Run: pip install pymupdf4llm") from e
        return pymupdf4llm

    def _extract(self, pdf_path: str) -> dict[str, Any]:
        pymupdf4llm = self._import_pymupdf4llm()

        try:
            table_output = self.base_config.get("table_output")
            if table_output is None and os.environ.get("PYMUPDF_TABLE_HTML"):
                table_output = "html"
            table_kwargs = {"table_output": table_output} if table_output is not None else {}
            page_chunks = pymupdf4llm.to_markdown(
                pdf_path, page_chunks=True, show_progress=False, use_ocr=False, **table_kwargs
            )
        except Exception as e:
            raise ProviderPermanentError(f"PyMuPDF4LLM error: {e}") from e

        layout_pages: list[dict[str, Any]] = []
        try:
            try:
                layout_json = pymupdf4llm.to_json(pdf_path, use_ocr=False)
            except TypeError:
                layout_json = pymupdf4llm.to_json(pdf_path)
            parsed_layout = layout_json if isinstance(layout_json, dict) else json.loads(layout_json)
            for page in parsed_layout.get("pages", []):
                if not isinstance(page, dict):
                    continue
                boxes = []
                for box in page.get("boxes", []) or []:
                    if not isinstance(box, dict):
                        continue
                    boxes.append(
                        {
                            "bbox": [box.get("x0"), box.get("y0"), box.get("x1"), box.get("y1")],
                            "boxclass": box.get("boxclass") or "text",
                            "text": _box_text(box),
                        }
                    )
                layout_pages.append(
                    {
                        "page_number": page.get("page_number", 1),
                        "width": page.get("width"),
                        "height": page.get("height"),
                        "boxes": boxes,
                    }
                )
        except Exception:
            layout_pages = []

        pages = []
        for i, chunk in enumerate(page_chunks):
            text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            pages.append({"page_index": i, "text": text})

        return {
            "pages": pages,
            "num_pages": len(pages),
            "layout_pages": layout_pages,
            "layout_source": "pymupdf4llm.to_json",
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"PyMuPDF4LLMProvider only supports PARSE, got {request.product_type}")

        pdf_path = Path(request.source_file_path)
        if not pdf_path.exists():
            raise ProviderPermanentError(f"File not found: {pdf_path}")

        started_at = datetime.now()
        try:
            raw_output = self._extract(str(pdf_path))
            completed_at = datetime.now()
            return RawInferenceResult(
                request=request,
                pipeline=pipeline,
                pipeline_name=pipeline.pipeline_name,
                product_type=request.product_type,
                raw_output=raw_output,
                started_at=started_at,
                completed_at=completed_at,
                latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
            )
        except (ProviderPermanentError, ProviderConfigError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error: {e}") from e

    @staticmethod
    def _convert_md_tables_to_html(content: str) -> str:
        import markdown2

        lines = content.split("\n")
        result_parts: list[str] = []
        table_lines: list[str] = []
        in_table = False

        def _flush() -> None:
            nonlocal table_lines
            if len(table_lines) >= 2:
                html = markdown2.markdown("\n".join(table_lines), extras=["tables"]).strip()
                if "<table>" in html.lower():
                    result_parts.append(html)
                else:
                    result_parts.extend(table_lines)
            else:
                result_parts.extend(table_lines)
            table_lines = []

        for line in lines:
            if "|" in line and line.strip().startswith("|"):
                in_table = True
                table_lines.append(line)
            else:
                if in_table:
                    _flush()
                    in_table = False
                result_parts.append(line)
        if in_table:
            _flush()
        return "\n".join(result_parts)

    @staticmethod
    def _build_layout_pages(raw_layout: list[dict[str, Any]]) -> list[ParseLayoutPageIR]:
        """Map raw PyMuPDF4LLM layout boxes into ParseOutput.layout_pages."""
        output: list[ParseLayoutPageIR] = []
        for page in raw_layout:
            try:
                page_width = float(page.get("width") or 0) or None
                page_height = float(page.get("height") or 0) or None
            except (TypeError, ValueError):
                page_width = page_height = None
            if not page_width or not page_height:
                continue

            items: list[LayoutItemIR] = []
            for box in page.get("boxes", []) or []:
                bbox = box.get("bbox") or []
                if len(bbox) != 4 or any(coord is None for coord in bbox):
                    continue
                x0, y0, x1, y1 = (float(coord) for coord in bbox)
                if not (x0 < x1 and y0 < y1):
                    continue
                label = str(box.get("boxclass") or "text")
                text = str(box.get("text") or "")
                items.append(
                    LayoutItemIR(
                        type=label,
                        value=text,
                        layout_segments=[
                            LayoutSegmentIR(
                                x=x0 / page_width,
                                y=y0 / page_height,
                                w=(x1 - x0) / page_width,
                                h=(y1 - y0) / page_height,
                                label=label,
                            )
                        ],
                    )
                )

            output.append(
                ParseLayoutPageIR(
                    page_number=int(page.get("page_number", 1)),
                    width=page_width,
                    height=page_height,
                    items=items,
                )
            )
        return output

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        pages: list[PageIR] = []
        page_texts: list[str] = []
        for page_data in raw_result.raw_output.get("pages", []):
            page_index = page_data.get("page_index", 0)
            text = self._convert_md_tables_to_html(page_data.get("text", "") or "")
            pages.append(PageIR(page_index=page_index, markdown=text))
            page_texts.append(text)

        full_text = "\n\n".join(page_texts)
        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=self._build_layout_pages(raw_result.raw_output.get("layout_pages", [])),
            markdown=full_text,
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
