"""Provider for Extend AI EXTRACT using the official Python SDK.

Based on Extend AI documentation: https://docs.extend.ai/developers/sd-ks
SDK: pip install extend-ai
"""

import hashlib
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.extract.citations import extract_extend_field_citations
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

_Extend: Any = None
_ApiError: Any = Exception
try:
    from extend_ai import Extend as _ImportedExtend
    from extend_ai.core.api_error import ApiError as _ImportedApiError

    _Extend = _ImportedExtend
    _ApiError = _ImportedApiError
    _HAS_EXTEND_AI = True
except ImportError:
    _HAS_EXTEND_AI = False

Extend: Any = _Extend
ApiError: Any = _ApiError

# JSON Schema properties not supported by Extend AI
UNSUPPORTED_SCHEMA_PROPERTIES = {
    "pattern",
    "not",
    "allOf",
    "anyOf",
    "oneOf",
    "if",
    "then",
    "else",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "patternProperties",
    "format",
    "const",
    "contentMediaType",
    "contentEncoding",
}


def _is_extract_product_type(value: Any) -> bool:
    extract_type = getattr(ProductType, "EXTRACT", None)
    if extract_type is not None and value == extract_type:
        return True
    return bool(getattr(value, "value", value) == "extract")


def _extract_output_cls() -> type[Any]:
    from parse_bench.schemas.extract_output import ExtractOutput

    return ExtractOutput


def _adapt_schema_for_extend(schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """
    Adapt a JSON schema for Extend AI compatibility.

    Extend AI has limited JSON Schema support:
    1. Array items must have type "object" (no primitive arrays like string[])
    2. Many advanced keywords (pattern, not, allOf, etc.) are not supported

    This adapter:
    - Wraps primitive array items in objects with a "value" property
    - Strips unsupported schema properties

    Returns:
        tuple: (adapted_schema, primitive_array_paths) where primitive_array_paths
               maps JSON paths to the primitive types that were wrapped
    """
    primitive_array_paths: dict[str, list[str]] = {}

    def adapt_node(node: dict[str, Any], path: str = "") -> dict[str, Any]:
        if not isinstance(node, dict):
            return node

        result = {}
        node_type = node.get("type")

        for key, value in node.items():
            # Skip unsupported properties
            if key in UNSUPPORTED_SCHEMA_PROPERTIES:
                continue

            if key == "properties" and isinstance(value, dict):
                # Recurse into properties
                result["properties"] = {
                    prop_name: adapt_node(prop_schema, f"{path}.{prop_name}" if path else prop_name)
                    for prop_name, prop_schema in value.items()
                }
            elif key == "items" and node_type == "array":
                # Handle array items
                if isinstance(value, dict):
                    items_type = value.get("type")
                    # Check if items is a primitive type
                    if items_type in ("string", "number", "integer", "boolean"):
                        # Wrap primitive in object with "value" property
                        primitive_array_paths[path] = [items_type]
                        result["items"] = {
                            "type": "object",  # type: ignore
                            "properties": {"value": adapt_node(value, f"{path}[items].value")},
                        }
                    else:
                        # Recurse into object items
                        result["items"] = adapt_node(value, f"{path}[items]")
                else:
                    result["items"] = value
            else:
                result[key] = value

        return result

    adapted = adapt_node(schema)
    return adapted, primitive_array_paths


def _adapt_result_from_extend(data: Any, primitive_array_paths: dict[str, list[str]], path: str = "") -> Any:
    """
    Adapt extraction results back to match the original schema.

    Unwraps primitive values that were wrapped in objects for Extend AI compatibility.
    """
    if data is None:
        return None

    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            result[key] = _adapt_result_from_extend(value, primitive_array_paths, current_path)
        return result

    if isinstance(data, list):
        # Check if this array path had primitive items that were wrapped
        if path in primitive_array_paths:
            # Unwrap the "value" from each object
            return [item.get("value") if isinstance(item, dict) else item for item in data]
        else:
            # Recurse into array items
            return [_adapt_result_from_extend(item, primitive_array_paths, f"{path}[items]") for item in data]

    return data


@register_provider("extend")
class ExtendProvider(Provider):
    """
    Provider for Extend AI document extraction using the official SDK.

    This provider uses the extend-ai Python SDK for extraction tasks.
    SDK Documentation: https://docs.extend.ai/developers/sd-ks

    Workflow:
    1. Upload file via client.file.upload()
    2. Create processor with schema via client.processor.create() (cached per schema hash)
    3. Run processor via client.processor_run.create() with sync=True

    Note: This provider adapts schemas to handle Extend AI's limited JSON Schema support:
    - Primitive arrays (string[], number[]) are wrapped in objects
    - Unsupported properties (pattern, not, allOf, etc.) are stripped
    """

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `api_key`: Extend AI API key (defaults to EXTEND_API_KEY env var)
            - `base_url`: Optional base URL for different deployments
              (default: https://api.extend.ai, alternatives: https://api.us2.extend.app,
               https://api.eu1.extend.ai)
            - `processor_name_prefix`: Prefix for processor names (default: "bench_")
            - `timeout`: Request timeout in seconds (default: 300)
        """
        super().__init__(provider_name, base_config)

        if not _HAS_EXTEND_AI or Extend is None:
            raise ProviderConfigError("ExtendProvider requires extend-ai. Install it with: pip install extend-ai")

        # Get API key
        api_key = self.base_config.get("api_key") or os.getenv("EXTEND_API_KEY")
        if not api_key:
            raise ProviderConfigError(
                "Extend AI API key is required. Set EXTEND_API_KEY environment variable or pass api_key in base_config."
            )

        # Configuration
        self._processor_name_prefix = self.base_config.get("processor_name_prefix", "bench_")
        timeout = self.base_config.get("timeout", 300)

        # Initialize the Extend client
        client_kwargs: dict[str, Any] = {
            "token": api_key,
            "timeout": float(timeout),
        }

        # Optional base URL for different deployments (US2, EU1, etc.)
        base_url = self.base_config.get("base_url")
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = Extend(**client_kwargs)

        # Cache for processor IDs by schema hash (thread-safe)
        self._processor_cache: dict[str, str] = {}
        self._processor_cache_lock = threading.Lock()

    def _get_config_hash(self, config: dict[str, Any]) -> str:
        """Get a deterministic hash of a config for caching processors."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _handle_api_error(self, e: ApiError, context: str) -> None:
        """Convert SDK ApiError to appropriate ProviderError."""
        status_code = getattr(e, "status_code", None)
        error_body = getattr(e, "body", str(e))

        if status_code == 429:
            raise ProviderRateLimitError(f"Rate limit exceeded during {context}: {error_body}")
        elif status_code in (502, 503, 504):
            raise ProviderTransientError(f"Transient error during {context}: {status_code} - {error_body}")
        elif status_code and status_code >= 400:
            raise ProviderPermanentError(f"Error during {context}: {status_code} - {error_body}")
        else:
            raise ProviderPermanentError(f"API error during {context}: {error_body}")

    def _upload_file(self, file_path: str) -> str:
        """
        Upload a file to Extend AI.

        :param file_path: Path to the file to upload
        :return: File ID from Extend AI
        :raises ProviderError: For any upload errors
        """
        try:
            with open(file_path, "rb") as f:
                upload_response = self._client.files.upload(file=f)

            # Extract file ID from response
            if hasattr(upload_response, "id"):
                return str(upload_response.id)
            elif hasattr(upload_response, "file") and hasattr(upload_response.file, "id"):
                return str(upload_response.file.id)
            elif isinstance(upload_response, dict):
                file_data = upload_response.get("file", upload_response)
                file_id = file_data.get("id") or file_data.get("fileId")
                if file_id:
                    return str(file_id)

            raise ProviderPermanentError(f"No file ID in upload response: {upload_response}")

        except ApiError as e:
            self._handle_api_error(e, "file upload")
            raise  # Should not reach here, but satisfies type checker
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["timeout", "timed out", "connection", "network", "readtimeout"]):
                raise ProviderTransientError(f"Transient error during file upload: {e}") from e
            raise ProviderPermanentError(f"Unexpected error during file upload: {e}") from e

    def _build_processor_config(self, schema: dict[str, Any], pipeline_config: dict[str, Any]) -> dict[str, Any]:
        """
        Build the processor config by merging schema with pipeline config options.

        :param schema: JSON schema for extraction
        :param pipeline_config: Pipeline configuration options
        :return: Complete processor config
        """
        config: dict[str, Any] = {
            "type": "EXTRACT",
            "schema": schema,
        }

        # Add baseProcessor if specified (e.g., "extraction_performance")
        if "baseProcessor" in pipeline_config:
            config["baseProcessor"] = pipeline_config["baseProcessor"]

        # Add baseVersion if specified (e.g., "4.1.1")
        if "baseVersion" in pipeline_config:
            config["baseVersion"] = pipeline_config["baseVersion"]

        # Add advancedOptions if specified
        if "advancedOptions" in pipeline_config:
            config["advancedOptions"] = pipeline_config["advancedOptions"]

        return config

    def _find_processor_by_name(self, name: str) -> str | None:
        """
        Find an existing processor by name.

        Handles pagination to search through all processors.

        :param name: Name of the processor to find
        :return: Processor ID if found, None otherwise
        """
        try:
            next_page_token: str | None = None

            while True:
                # List processors with pagination
                if next_page_token:
                    list_response = self._client.processor.list(next_page_token=next_page_token)
                else:
                    list_response = self._client.processor.list()

                # Extract processors from response
                processors: list[Any] = []
                if hasattr(list_response, "processors"):
                    processors = list_response.processors or []
                elif hasattr(list_response, "data"):
                    processors = list_response.data or []
                elif isinstance(list_response, list):
                    processors = list_response

                # Search for processor by name
                for processor in processors:
                    proc_name = getattr(processor, "name", None)
                    if proc_name == name:
                        proc_id = getattr(processor, "id", None)
                        if proc_id:
                            return str(proc_id)

                # Check for next page
                next_page_token = getattr(list_response, "next_page_token", None)
                if not next_page_token:
                    break

            return None

        except Exception:
            # If listing fails, return None and let creation handle it
            return None

    def _create_processor(self, processor_config: dict[str, Any], config_hash: str) -> str:
        """
        Create an extraction processor with the given config.

        :param processor_config: Full processor configuration including schema
        :param config_hash: Hash of the config for naming
        :return: Processor ID
        :raises ProviderError: For any creation errors
        """
        processor_name = f"{self._processor_name_prefix}{config_hash}"

        try:
            processor_response = self._client.processor.create(
                name=processor_name,
                type="EXTRACT",  # type: ignore[arg-type]
                config=processor_config,  # type: ignore[arg-type]
            )

            # Extract processor ID from response
            # Response is ProcessorCreateResponse with a 'processor' attribute
            if hasattr(processor_response, "processor"):
                processor = processor_response.processor
                if hasattr(processor, "id"):
                    return str(processor.id)
            elif hasattr(processor_response, "id"):
                return str(processor_response.id)
            elif isinstance(processor_response, dict):
                # Handle dict response
                if "processor" in processor_response:
                    processor_id = processor_response["processor"].get("id")
                else:
                    processor_id = processor_response.get("id") or processor_response.get("processorId")
                if processor_id:
                    return str(processor_id)

            raise ProviderPermanentError(f"No processor ID in creation response: {processor_response}")

        except ApiError as e:
            # Check if processor already exists
            error_body = getattr(e, "body", {})
            error_msg = ""
            if isinstance(error_body, dict):
                error_msg = error_body.get("error", "")
            else:
                error_msg = str(error_body)

            if "already exists" in error_msg.lower():
                # Try to find the existing processor
                existing_id = self._find_processor_by_name(processor_name)
                if existing_id:
                    return existing_id

            self._handle_api_error(e, "processor creation")
            raise  # Should not reach here, but satisfies type checker
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["timeout", "timed out", "connection", "network", "readtimeout"]):
                raise ProviderTransientError(f"Transient error during processor creation: {e}") from e
            raise ProviderPermanentError(f"Unexpected error during processor creation: {e}") from e

    def _get_or_create_processor(self, processor_config: dict[str, Any]) -> str:
        """
        Get existing processor ID or create a new one for the given config.

        Thread-safe: uses locking to prevent concurrent creation of same processor.

        :param processor_config: Full processor configuration including schema
        :return: Processor ID
        """
        config_hash = self._get_config_hash(processor_config)
        processor_name = f"{self._processor_name_prefix}{config_hash}"

        # Fast path: check cache without lock
        if config_hash in self._processor_cache:
            return self._processor_cache[config_hash]

        # Slow path: acquire lock to prevent concurrent creation
        with self._processor_cache_lock:
            # Double-check after acquiring lock
            if config_hash in self._processor_cache:
                return self._processor_cache[config_hash]

            # Check if processor already exists in Extend before creating
            existing_id = self._find_processor_by_name(processor_name)
            if existing_id:
                self._processor_cache[config_hash] = existing_id
                return existing_id

            # Create new processor
            processor_id = self._create_processor(processor_config, config_hash)
            self._processor_cache[config_hash] = processor_id
            return processor_id

    def _run_processor(self, processor_id: str, file_id: str) -> dict[str, Any]:
        """
        Run a processor on a file synchronously.

        :param processor_id: ID of the processor to run
        :param file_id: ID of the uploaded file
        :return: Raw response from the processor run
        :raises ProviderError: For any run errors
        """
        try:
            run_response = self._client.processor_run.create(
                processor_id=processor_id,
                file={"fileId": file_id},  # type: ignore[arg-type]
                sync=True,  # Synchronous processing - waits for completion
            )

            # Convert response to dict for storage
            if hasattr(run_response, "model_dump"):
                return cast(dict[str, Any], run_response.model_dump())
            elif hasattr(run_response, "dict"):
                return cast(dict[str, Any], run_response.dict())
            elif isinstance(run_response, dict):
                return run_response
            else:
                # Try to extract attributes manually
                result: dict[str, Any] = {}
                for attr in [
                    "id",
                    "status",
                    "output",
                    "extracted_data",
                    "extractedData",
                    "data",
                    "result",
                    "error",
                    "processorId",
                    "fileId",
                ]:
                    if hasattr(run_response, attr):
                        value = getattr(run_response, attr)
                        if not callable(value):
                            result[attr] = value
                return result

        except ApiError as e:
            self._handle_api_error(e, "processor run")
            raise  # Should not reach here, but satisfies type checker
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["timeout", "timed out", "connection", "network", "readtimeout"]):
                raise ProviderTransientError(f"Transient error during processor run: {e}") from e
            raise ProviderPermanentError(f"Unexpected error during processor run: {e}") from e

    def _extract_document(
        self,
        file_path: str,
        schema: dict[str, Any],
        pipeline_config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extract data from a document using Extend AI.

        :param file_path: Path to the document file
        :param schema: JSON schema for extraction
        :param pipeline_config: Pipeline configuration options
        :return: Raw API response with extracted data
        :raises ProviderError: For any extraction errors
        """
        # Step 0: Adapt schema for Extend AI compatibility
        adapted_schema, primitive_array_paths = _adapt_schema_for_extend(schema)

        # Step 1: Upload file
        file_id = self._upload_file(file_path)

        # Step 2: Build processor config with adapted schema and pipeline options
        processor_config = self._build_processor_config(adapted_schema, pipeline_config)

        # Step 3: Get or create processor for this config
        processor_id = self._get_or_create_processor(processor_config)

        # Step 4: Run processor synchronously
        result = self._run_processor(processor_id, file_id)

        # Add metadata (including schema adaptation info for normalization)
        result["_extend_metadata"] = {
            "file_id": file_id,
            "processor_id": processor_id,
            "primitive_array_paths": primitive_array_paths,
        }

        return result

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run inference and return raw results.

        :param pipeline: Pipeline specification
        :param request: Inference request (must include schema_override for EXTRACT)
        :return: Raw inference result
        :raises ProviderError: For any provider-related failures
        """
        if not _is_extract_product_type(request.product_type):
            raise ProviderPermanentError(
                f"ExtendProvider only supports EXTRACT product type, got {request.product_type}"
            )

        # Schema is required for extraction
        if not request.schema_override:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override"
            )

        started_at = datetime.now()

        # Check if file exists
        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        try:
            # Run extraction with pipeline config options
            raw_output = self._extract_document(
                file_path=str(file_path),
                schema=request.schema_override,
                pipeline_config=pipeline.config,
            )

            completed_at = datetime.now()
            latency_ms = int((completed_at - started_at).total_seconds() * 1000)

            return RawInferenceResult(
                request=request,
                pipeline=pipeline,
                pipeline_name=pipeline.pipeline_name,
                product_type=request.product_type,
                raw_output=raw_output,
                started_at=started_at,
                completed_at=completed_at,
                latency_in_ms=latency_ms,
            )

        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce ExtractOutput.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        :raises ProviderError: For any normalization failures
        """
        if not _is_extract_product_type(raw_result.product_type):
            raise ProviderPermanentError(
                f"ExtendProvider only supports EXTRACT product type, got {raw_result.product_type}"
            )

        # Extract the structured data from processor_run.output.value
        extracted_data = raw_result.raw_output.get("processor_run", {}).get("output", {}).get("value", {})

        # Adapt the result back to match the original schema
        # (unwrap primitive arrays that were wrapped for Extend AI)
        primitive_array_paths = raw_result.raw_output.get("_extend_metadata", {}).get("primitive_array_paths", {})
        if primitive_array_paths:
            extracted_data = _adapt_result_from_extend(extracted_data, primitive_array_paths)

        output = _extract_output_cls()(
            task_type="extract",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted_data,
            field_citations=extract_extend_field_citations(raw_result.raw_output),
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
