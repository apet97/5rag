"""Structured error codes and taxonomy for Clockify RAG system.

Provides operator-friendly error codes for troubleshooting and monitoring.
Each error code has:
- Unique identifier (RAG_E###)
- Category (config, retrieval, llm, index, validation)
- Human-readable description
- Troubleshooting hints

Error Code Format: RAG_E### where ### is a 3-digit number
- 0xx: Configuration errors
- 1xx: Index/Build errors
- 2xx: Retrieval errors
- 3xx: LLM errors
- 4xx: Validation errors
- 5xx: Internal errors
"""

from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ErrorInfo:
    """Error information with code, message, and troubleshooting hints."""

    code: str
    category: str
    message: str
    hints: list[str]
    doc_url: Optional[str] = None


class ErrorCode(Enum):
    """Structured error codes for the RAG system."""

    # ========================================================================
    # Configuration Errors (0xx)
    # ========================================================================
    INVALID_CONFIG = "RAG_E001"
    INVALID_ENDPOINT_URL = "RAG_E002"
    INVALID_MODEL_NAME = "RAG_E003"
    INVALID_RETRIEVAL_PARAMS = "RAG_E004"
    MISSING_API_KEY = "RAG_E005"
    INVALID_API_KEY = "RAG_E006"

    # ========================================================================
    # Index/Build Errors (1xx)
    # ========================================================================
    INDEX_NOT_READY = "RAG_E101"
    INDEX_BUILD_FAILED = "RAG_E102"
    INDEX_CORRUPT = "RAG_E103"
    FAISS_TRAINING_FAILED = "RAG_E104"
    EMBEDDING_DIMENSION_MISMATCH = "RAG_E105"
    CHUNK_PARSING_FAILED = "RAG_E106"

    # ========================================================================
    # Retrieval Errors (2xx)
    # ========================================================================
    RETRIEVAL_FAILED = "RAG_E201"
    EMBEDDING_FAILED = "RAG_E202"
    BM25_SEARCH_FAILED = "RAG_E203"
    FAISS_SEARCH_FAILED = "RAG_E204"
    INSUFFICIENT_COVERAGE = "RAG_E205"
    QUERY_TOO_LONG = "RAG_E206"

    # ========================================================================
    # LLM Errors (3xx)
    # ========================================================================
    LLM_UNAVAILABLE = "RAG_E301"
    LLM_TIMEOUT = "RAG_E302"
    LLM_RESPONSE_INVALID = "RAG_E303"
    LLM_CONTEXT_OVERFLOW = "RAG_E304"
    LLM_GENERATION_FAILED = "RAG_E305"
    FALLBACK_FAILED = "RAG_E306"

    # ========================================================================
    # Validation Errors (4xx)
    # ========================================================================
    INVALID_QUESTION = "RAG_E401"
    INVALID_PARAMETERS = "RAG_E402"
    CITATION_VALIDATION_FAILED = "RAG_E403"
    HALLUCINATION_DETECTED = "RAG_E404"

    # ========================================================================
    # Internal Errors (5xx)
    # ========================================================================
    INTERNAL_ERROR = "RAG_E501"
    CACHE_ERROR = "RAG_E502"
    METRICS_ERROR = "RAG_E503"
    RATE_LIMIT_EXCEEDED = "RAG_E504"


# Error information database
ERROR_INFO: Dict[str, ErrorInfo] = {
    # Configuration Errors
    "RAG_E001": ErrorInfo(
        code="RAG_E001",
        category="Configuration",
        message="Invalid configuration",
        hints=[
            "Check your config file syntax (YAML)",
            "Validate all required fields are present",
            "Run 'ragctl config-show' to view effective configuration",
        ],
    ),
    "RAG_E002": ErrorInfo(
        code="RAG_E002",
        category="Configuration",
        message="Invalid Ollama endpoint URL",
        hints=[
            "Set RAG_OLLAMA_URL environment variable (e.g., http://127.0.0.1:11434)",
            "Check URL format: must be http:// or https://",
            "Verify endpoint is reachable: curl $RAG_OLLAMA_URL/api/tags",
            "If using VPN, ensure you're connected",
        ],
    ),
    "RAG_E003": ErrorInfo(
        code="RAG_E003",
        category="Configuration",
        message="Invalid model name",
        hints=[
            "Check model name spelling and format",
            "List available models: curl $RAG_OLLAMA_URL/api/tags",
            "Ensure model is pulled: ollama pull <model-name>",
        ],
    ),
    "RAG_E004": ErrorInfo(
        code="RAG_E004",
        category="Configuration",
        message="Invalid retrieval parameters",
        hints=[
            "Check top_k (1-100), pack_top (1-50), threshold (0.0-1.0)",
            "Ensure top_k >= pack_top",
            "See docs/RETRIEVAL_TUNING_GUIDE.md for parameter guidance",
        ],
    ),
    "RAG_E005": ErrorInfo(
        code="RAG_E005",
        category="Configuration",
        message="Missing API key",
        hints=[
            "Set API_KEY header in request",
            "Ensure RAG_API_ALLOWED_KEYS environment variable is set",
            "Disable auth if not needed: set API_AUTH_MODE=none",
        ],
    ),
    "RAG_E006": ErrorInfo(
        code="RAG_E006",
        category="Configuration",
        message="Invalid API key",
        hints=[
            "Verify API key is in RAG_API_ALLOWED_KEYS",
            "Check for typos or whitespace in API key",
            "Regenerate API key if compromised",
        ],
    ),
    # Index/Build Errors
    "RAG_E101": ErrorInfo(
        code="RAG_E101",
        category="Index",
        message="Index not ready",
        hints=[
            "Build index: ragctl ingest",
            "Check index files exist: ls -lh chunks.jsonl vecs_n.npy bm25.json",
            "Verify index metadata: cat index.meta.json",
            "Wait for startup index loading to complete",
        ],
    ),
    "RAG_E102": ErrorInfo(
        code="RAG_E102",
        category="Index",
        message="Index build failed",
        hints=[
            "Check knowledge base file exists and is readable",
            "Verify sufficient disk space: df -h",
            "Check logs for detailed error: tail -f logs/rag.log",
            "Try rebuilding with --force flag",
        ],
    ),
    "RAG_E103": ErrorInfo(
        code="RAG_E103",
        category="Index",
        message="Index files corrupted",
        hints=[
            "Rebuild index from scratch: ragctl ingest --force",
            "Check file integrity: file chunks.jsonl vecs_n.npy",
            "Verify disk health: smartctl -a /dev/sda",
        ],
    ),
    "RAG_E104": ErrorInfo(
        code="RAG_E104",
        category="Index",
        message="FAISS training failed",
        hints=[
            "Check vector count >= nlist parameter (default 64)",
            "System will automatically fall back to FlatIP index",
            "For large datasets, increase nlist: export FAISS_NLIST=256",
        ],
    ),
    "RAG_E105": ErrorInfo(
        code="RAG_E105",
        category="Index",
        message="Embedding dimension mismatch",
        hints=[
            "Cannot mix local embeddings (384-dim) with Ollama (768-dim)",
            "Rebuild index with consistent embedding backend",
            "Check EMB_BACKEND setting (local or ollama)",
            "Clear embedding cache: rm emb_cache.jsonl",
        ],
    ),
    "RAG_E106": ErrorInfo(
        code="RAG_E106",
        category="Index",
        message="Chunk parsing failed",
        hints=[
            "Check knowledge base Markdown syntax",
            "Verify article headers follow '# [ARTICLE]' convention",
            "Check for malformed Unicode or special characters",
        ],
    ),
    # Retrieval Errors
    "RAG_E201": ErrorInfo(
        code="RAG_E201",
        category="Retrieval",
        message="Retrieval failed",
        hints=[
            "Check index is ready: ragctl doctor",
            "Verify query is not empty",
            "Check logs for underlying cause",
        ],
    ),
    "RAG_E202": ErrorInfo(
        code="RAG_E202",
        category="Retrieval",
        message="Query embedding failed",
        hints=[
            "Check Ollama connectivity: curl $RAG_OLLAMA_URL/api/tags",
            "Verify embedding model is available: ollama list",
            "Check embedding backend: echo $EMB_BACKEND (local or ollama)",
        ],
    ),
    "RAG_E203": ErrorInfo(
        code="RAG_E203",
        category="Retrieval",
        message="BM25 search failed",
        hints=[
            "Check BM25 index exists: ls -lh bm25.json",
            "Rebuild index if corrupted: ragctl ingest --force",
            "Check query tokenization (empty query after tokenization?)",
        ],
    ),
    "RAG_E204": ErrorInfo(
        code="RAG_E204",
        category="Retrieval",
        message="FAISS search failed",
        hints=[
            "Check FAISS index exists: ls -lh faiss.index",
            "System will automatically fall back to HNSW or BM25",
            "Rebuild FAISS index: ragctl ingest --force",
        ],
    ),
    "RAG_E205": ErrorInfo(
        code="RAG_E205",
        category="Retrieval",
        message="Insufficient coverage (less than 2 relevant chunks found)",
        hints=[
            "Lower threshold: export RAG_THRESHOLD=0.1",
            "Increase top_k: export RAG_TOP_K=20",
            "Check if query matches knowledge base domain",
            "Review retrieval tuning: docs/RETRIEVAL_TUNING_GUIDE.md",
        ],
    ),
    "RAG_E206": ErrorInfo(
        code="RAG_E206",
        category="Retrieval",
        message="Query exceeds maximum length",
        hints=[
            "Query length limit: 1,000,000 characters (DoS protection)",
            "Shorten query to be more concise",
            "If legitimate use case, increase MAX_QUERY_LENGTH",
        ],
    ),
    # LLM Errors
    "RAG_E301": ErrorInfo(
        code="RAG_E301",
        category="LLM",
        message="LLM endpoint unavailable",
        hints=[
            "Check Ollama is running: curl $RAG_OLLAMA_URL/api/tags",
            "Verify VPN connection if using internal endpoint",
            "Check firewall/network: ping <ollama-host>",
            "Automatic fallback should trigger if enabled (RAG_FALLBACK_ENABLED=true)",
        ],
    ),
    "RAG_E302": ErrorInfo(
        code="RAG_E302",
        category="LLM",
        message="LLM request timeout",
        hints=[
            "Increase timeout: export RAG_CHAT_READ_TIMEOUT=180",
            "Check model size (larger models take longer)",
            "Verify server load: curl $RAG_OLLAMA_URL/api/ps",
            "Consider using faster model or reducing num_ctx",
        ],
    ),
    "RAG_E303": ErrorInfo(
        code="RAG_E303",
        category="LLM",
        message="LLM response invalid or unparseable",
        hints=[
            "Check LLM returned valid JSON format",
            "Verify prompt template in retrieval.py",
            "Check model supports JSON output",
            "Try with different model: export RAG_CHAT_MODEL=<model>",
        ],
    ),
    "RAG_E304": ErrorInfo(
        code="RAG_E304",
        category="LLM",
        message="Context exceeds LLM window",
        hints=[
            "Reduce pack_top: export RAG_PACK_TOP=5",
            "Use model with larger context: qwen2.5:32b (32k) or gpt-oss:20b (128k)",
            "Reduce CTX_BUDGET: export CTX_BUDGET=8000",
        ],
    ),
    "RAG_E305": ErrorInfo(
        code="RAG_E305",
        category="LLM",
        message="LLM generation failed",
        hints=[
            "Check LLM error message in logs",
            "Verify model is loaded: ollama list",
            "Try regenerating with different seed",
            "Check for CUDA/memory errors if using GPU",
        ],
    ),
    "RAG_E306": ErrorInfo(
        code="RAG_E306",
        category="LLM",
        message="Both primary and fallback LLMs failed",
        hints=[
            "Check both endpoints are accessible",
            "Verify fallback model is different from primary",
            "Check fallback is enabled: echo $RAG_FALLBACK_ENABLED",
            "See operator runbook: docs/internals/RUNBOOK.md",
        ],
    ),
    # Validation Errors
    "RAG_E401": ErrorInfo(
        code="RAG_E401",
        category="Validation",
        message="Invalid question format",
        hints=[
            "Question must be non-empty string",
            "Check for malicious input (XSS, injection)",
            "Maximum length: 10,000 characters",
        ],
    ),
    "RAG_E402": ErrorInfo(
        code="RAG_E402",
        category="Validation",
        message="Invalid parameters",
        hints=[
            "Check parameter types and ranges",
            "top_k: 1-100, pack_top: 1-50, threshold: 0.0-1.0",
            "Ensure top_k >= pack_top",
        ],
    ),
    "RAG_E403": ErrorInfo(
        code="RAG_E403",
        category="Validation",
        message="Citation validation failed",
        hints=[
            "Answer references chunk IDs not in context",
            "This may indicate hallucination or citation parsing error",
            "Check answer format includes [id_123, id_456] citations",
        ],
    ),
    "RAG_E404": ErrorInfo(
        code="RAG_E404",
        category="Validation",
        message="Potential hallucination detected",
        hints=[
            "Answer claims contradict source documents",
            "Low semantic entailment score from NLI model",
            "Review answer and flag for human verification",
        ],
    ),
    # Internal Errors
    "RAG_E501": ErrorInfo(
        code="RAG_E501",
        category="Internal",
        message="Internal server error",
        hints=[
            "Check application logs: tail -f logs/rag.log",
            "Report bug with stack trace and request ID",
            "Try restarting service: systemctl restart rag-api",
        ],
    ),
    "RAG_E502": ErrorInfo(
        code="RAG_E502",
        category="Internal",
        message="Cache error",
        hints=[
            "Clear query cache: rm -f rag_queries.jsonl",
            "Clear embedding cache: rm -f emb_cache.jsonl",
            "Check disk space: df -h",
        ],
    ),
    "RAG_E503": ErrorInfo(
        code="RAG_E503",
        category="Internal",
        message="Metrics collection error",
        hints=[
            "Non-critical: metrics collection failed but query succeeded",
            "Check metrics endpoint: curl http://localhost:8000/v1/metrics",
            "Verify Prometheus scraping configuration",
        ],
    ),
    "RAG_E504": ErrorInfo(
        code="RAG_E504",
        category="Internal",
        message="Rate limit exceeded",
        hints=[
            "Too many requests from this client",
            f"Wait before retrying (see Retry-After header)",
            "Increase rate limit: export RAG_RATE_LIMIT_RPM=120",
            "Disable rate limiting: export RAG_RATE_LIMIT_ENABLED=false",
        ],
    ),
}


def get_error_info(error_code: str) -> Optional[ErrorInfo]:
    """Get error information for a given error code.

    Args:
        error_code: Error code string (e.g., "RAG_E001")

    Returns:
        ErrorInfo object with code, message, and hints, or None if not found

    Example:
        >>> info = get_error_info("RAG_E001")
        >>> print(info.message)
        'Invalid configuration'
        >>> print("\\n".join(info.hints))
        Check your config file syntax (YAML)
        ...
    """
    return ERROR_INFO.get(error_code)


def format_error_message(
    error_code: str,
    details: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Format a structured error message for API responses.

    Args:
        error_code: Error code from ErrorCode enum
        details: Optional additional details about the error
        request_id: Optional request ID for tracing

    Returns:
        Dict with structured error information

    Example:
        >>> format_error_message("RAG_E002", "Invalid URL: not-a-url", "req-123")
        {
            "error": {
                "code": "RAG_E002",
                "category": "Configuration",
                "message": "Invalid Ollama endpoint URL",
                "details": "Invalid URL: not-a-url",
                "hints": [...],
                "request_id": "req-123"
            }
        }
    """
    info = get_error_info(error_code)

    if not info:
        # Unknown error code
        return {
            "error": {
                "code": error_code,
                "category": "Unknown",
                "message": "Unknown error code",
                "details": details,
                "request_id": request_id,
            }
        }

    error_dict = {
        "code": info.code,
        "category": info.category,
        "message": info.message,
        "hints": info.hints,
    }

    if details:
        error_dict["details"] = details

    if request_id:
        error_dict["request_id"] = request_id

    if info.doc_url:
        error_dict["doc_url"] = info.doc_url

    return {"error": error_dict}
