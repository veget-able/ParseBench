"""Provider for PyMuPDF4LLM layout detection via to_json()."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Any

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
)
from parse_bench.inference.providers.layoutdet._pymupdf4llm import layout_json_to_layout_output
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType


@register_provider("pymupdf4llm_layout")
class PyMuPDF4LLMLayoutProvider(Provider):
    """Provider for PyMuPDF4LLM to_json layout output."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._use_tgif = self.base_config.get("use_tgif")
        if self._use_tgif is not None:
            os.environ["USE_TGIF"] = str(self._use_tgif)

    def _import_pymupdf4llm(self) -> Any:
        try:
            import pymupdf.layout  # noqa: F401
            import pymupdf4llm
        except ImportError as e:
            raise ProviderConfigError("pymupdf4llm not installed. Run: pip install pymupdf4llm") from e
        return pymupdf4llm

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.LAYOUT_DETECTION:
            raise ProviderPermanentError(
                f"PyMuPDF4LLMLayoutProvider only supports LAYOUT_DETECTION, got {request.product_type}"
            )

        pdf_path = Path(request.source_file_path)
        if not pdf_path.exists():
            raise ProviderPermanentError(f"File not found: {pdf_path}")

        pymupdf4llm = self._import_pymupdf4llm()
        started_at = datetime.now()
        try:
            raw_output = {
                "layout_json": pymupdf4llm.to_json(str(pdf_path)),
                "markdown": "",
            }
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
            raise ProviderPermanentError(f"PyMuPDF4LLM layout error: {e}") from e

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.LAYOUT_DETECTION:
            raise ProviderPermanentError(
                f"PyMuPDF4LLMLayoutProvider only supports LAYOUT_DETECTION, got {raw_result.product_type}"
            )

        output = layout_json_to_layout_output(
            raw_result.raw_output.get("layout_json"),
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            markdown=str(raw_result.raw_output.get("markdown", "")),
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
