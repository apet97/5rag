"""Configuration constants for Clockify RAG system."""

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Optional

# FIX (Error #13): Helper functions for safe environment variable parsing
_logger = logging.getLogger(__name__)


def _get_env_value(
    primary: str,
    default: Optional[str] = None,
    legacy_keys: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Read environment variables with optional legacy fallbacks.

    Args:
        primary: Preferred environment variable name (new `RAG_*` namespace)
        default: Default value if nothing is set
        legacy_keys: Older env var names to support for backwards compatibility

    Returns:
        The first non-empty environment value, or the provided default.
    """
    keys = [primary]
    if legacy_keys:
        keys.extend(legacy_keys)
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def _parse_env_float(key: str, default: float, min_val: float = None, max_val: float = None) -> float:
    """Parse float from environment with validation.

    FIX (Error #13): Prevents crashes from invalid env var values.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)

    Returns:
        Parsed and validated float value
    """
    value = os.environ.get(key)
    if value is None:
        return default

    try:
        parsed = float(value)
    except ValueError as e:
        _logger.error(f"Invalid float for {key}='{value}': {e}. " f"Using default: {default}")
        return default

    if min_val is not None and parsed < min_val:
        _logger.warning(f"{key}={parsed} below minimum {min_val}, clamping")
        return min_val
    if max_val is not None and parsed > max_val:
        _logger.warning(f"{key}={parsed} above maximum {max_val}, clamping")
        return max_val

    return parsed


def _parse_env_int(key: str, default: int, min_val: int = None, max_val: int = None) -> int:
    """Parse int from environment with validation.

    FIX (Error #13): Prevents crashes from invalid env var values.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)

    Returns:
        Parsed and validated int value
    """
    value = os.environ.get(key)
    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError as e:
        _logger.error(f"Invalid integer for {key}='{value}': {e}. " f"Using default: {default}")
        return default

    if min_val is not None and parsed < min_val:
        _logger.warning(f"{key}={parsed} below minimum {min_val}, clamping")
        return min_val
    if max_val is not None and parsed > max_val:
        _logger.warning(f"{key}={parsed} above maximum {max_val}, clamping")
        return max_val

    return parsed


def _get_bool_env(var_name: str, default: str = "1", legacy_keys: Optional[Iterable[str]] = None) -> bool:
    """Read a boolean environment variable with optional legacy aliases."""

    keys = [var_name]
    if legacy_keys:
        keys.extend(legacy_keys)

    for key in keys:
        if key in os.environ:
            value = os.environ[key]
            break
    else:
        value = default

    return value.lower() not in {"0", "false", "no", "off", ""}


def allow_proxies_enabled() -> bool:
    """Return whether proxy usage is enabled (includes legacy USE_PROXY alias)."""

    return _get_bool_env("ALLOW_PROXIES", "0", legacy_keys=("USE_PROXY",))


def get_query_expansions_path() -> Optional[str]:
    """Return the currently configured query expansion file path."""

    value = _get_env_value("CLOCKIFY_QUERY_EXPANSIONS", None)
    return value or None


# ====== OLLAMA CONFIG ======
# 🔧 ENVIRONMENT PROFILES
# The default endpoint below is for internal company VPN use only.
# For other environments, set RAG_OLLAMA_URL environment variable.
#
# Common profiles:
#   - Local Ollama:    export RAG_OLLAMA_URL="http://127.0.0.1:11434"
#   - Company VPN:     export RAG_OLLAMA_URL="http://10.127.0.192:11434"
#   - Custom endpoint: export RAG_OLLAMA_URL="http://your-host:port"
#
# ⚠️  WARNING: The default below is ENVIRONMENT-SPECIFIC!
# It only works from company VPN. Change it for your deployment environment.

# Environment profile presets
ENV_PROFILE_LOCAL = "http://127.0.0.1:11434"
ENV_PROFILE_VPN_INTERNAL = "http://10.127.0.192:11434"  # Company VPN only

# Backwards compatibility constants (deprecated - use RAG_OLLAMA_URL env var instead)
DEFAULT_LOCAL_OLLAMA_URL = ENV_PROFILE_LOCAL
DEFAULT_RAG_OLLAMA_URL = ENV_PROFILE_VPN_INTERNAL  # DEPRECATED: VPN-specific default

# Default endpoint (VPN-specific - override with RAG_OLLAMA_URL for other environments)
_DEFAULT_RAG_OLLAMA_URL = ENV_PROFILE_VPN_INTERNAL

RAG_OLLAMA_URL = _get_env_value(
    "RAG_OLLAMA_URL",
    default=_DEFAULT_RAG_OLLAMA_URL,
    legacy_keys=("OLLAMA_URL",),
)
RAG_CHAT_MODEL = _get_env_value(
    "RAG_CHAT_MODEL",
    default="qwen2.5:32b",
    legacy_keys=("GEN_MODEL", "CHAT_MODEL"),
)
RAG_EMBED_MODEL = _get_env_value(
    "RAG_EMBED_MODEL",
    default="nomic-embed-text:latest",
    legacy_keys=("EMB_MODEL", "EMBED_MODEL"),
)

# ====== PROVIDER SELECTION (GPT-OSS-20B Integration) ======
# RAG_PROVIDER: Select LLM backend ("ollama" or "gpt-oss")
# - "ollama": Use standard Ollama with qwen2.5:32b (default)
# - "gpt-oss": Use OpenAI's gpt-oss-20b reasoning model (128k context)
RAG_PROVIDER = _get_env_value("RAG_PROVIDER", default="ollama").lower()

# ====== GPT-OSS-20B CONFIG ======
# GPT-OSS-20B: OpenAI's open-weight 20B reasoning model
# - 128k context window (vs qwen2.5:32b's 32k)
# - ~21B total params, ~3.6B active per token (MoE architecture)
# - Optimized for reasoning and coding tasks
# - Served via Ollama-compatible API at same endpoint

# Model name for GPT-OSS (when RAG_PROVIDER=gpt-oss)
RAG_GPT_OSS_MODEL = _get_env_value("RAG_GPT_OSS_MODEL", default="gpt-oss:20b")

# Sampling parameters for GPT-OSS
# OpenAI recommends temperature=1.0, top_p=1.0 for optimal reasoning
# For RAG QA, you may prefer slightly lower temperature (e.g., 0.7) for more deterministic answers
RAG_GPT_OSS_TEMPERATURE = _parse_env_float("RAG_GPT_OSS_TEMPERATURE", 1.0, min_val=0.0, max_val=2.0)
RAG_GPT_OSS_TOP_P = _parse_env_float("RAG_GPT_OSS_TOP_P", 1.0, min_val=0.0, max_val=1.0)

# Context window and budget for GPT-OSS
# GPT-OSS has 128k token context window (vs qwen's 32k)
# We can use more context for retrieval while still leaving room for generation
RAG_GPT_OSS_CTX_WINDOW = _parse_env_int("RAG_GPT_OSS_CTX_WINDOW", 128000, min_val=4096, max_val=200000)  # 128k tokens

# Context budget for RAG snippets when using GPT-OSS
# Allocate ~12.5% of 128k context (16k tokens) for snippets, leaving room for:
# - System instructions (~500 tokens)
# - User question (~100 tokens)
# - Response generation (~2k tokens)
# - Safety margin for tokenization variance
RAG_GPT_OSS_CTX_BUDGET = _parse_env_int(
    "RAG_GPT_OSS_CTX_BUDGET", 16000, min_val=1000, max_val=100000
)  # 16k tokens for snippets

# Timeout settings for GPT-OSS (reasoning models may take longer)
# Chat timeout increased to 180s (vs 120s for qwen) to allow for reasoning traces
RAG_GPT_OSS_CHAT_TIMEOUT = _parse_env_float("RAG_GPT_OSS_CHAT_TIMEOUT", 180.0, min_val=10.0, max_val=600.0)

# ====== AUTOMATIC MODEL FALLBACK ======
# Enable automatic fallback to secondary model when primary model is unavailable
# This prevents total service failure when the primary model (qwen2.5:32b) is down
RAG_FALLBACK_ENABLED = _get_env_value("RAG_FALLBACK_ENABLED", default="true").lower() in ("true", "1", "yes")

# Fallback provider and model to use when primary is unavailable
# Default: Fall back to gpt-oss:20b (128k context, good reasoning ability)
RAG_FALLBACK_PROVIDER = _get_env_value("RAG_FALLBACK_PROVIDER", default="gpt-oss").lower()
RAG_FALLBACK_MODEL = _get_env_value("RAG_FALLBACK_MODEL", default="gpt-oss:20b")

_INITIAL_RAG_LLM_CLIENT = _get_env_value("RAG_LLM_CLIENT", default="")


@dataclass(frozen=True)
class LLMSettings:
    """Typed snapshot of the current LLM configuration."""

    base_url: str
    chat_model: str
    embed_model: str
    client_mode: str
    provider: str  # "ollama" or "gpt-oss"


def get_llm_client_mode(default: str = "") -> str:
    """Return the preferred LLM client mode (`mock`, `ollama`, etc.)."""

    raw_value = os.environ.get("RAG_LLM_CLIENT")
    if raw_value is None:
        raw_value = _INITIAL_RAG_LLM_CLIENT
    normalized = (raw_value or "").strip().lower()
    if not normalized:
        return default.strip().lower()
    return normalized


def current_llm_settings(default_client_mode: str = "") -> LLMSettings:
    """Return a dataclass capturing the current LLM configuration.

    When RAG_PROVIDER=gpt-oss, returns gpt-oss-specific settings.
    Otherwise, returns standard Ollama settings (default).
    """
    provider = RAG_PROVIDER

    # Select model based on provider
    if provider == "gpt-oss":
        chat_model = RAG_GPT_OSS_MODEL
    else:
        chat_model = RAG_CHAT_MODEL

    return LLMSettings(
        base_url=RAG_OLLAMA_URL,
        chat_model=chat_model,
        embed_model=RAG_EMBED_MODEL,
        client_mode=get_llm_client_mode(default_client_mode),
        provider=provider,
    )


def get_context_budget() -> int:
    """Get the appropriate context budget based on active provider.

    Returns:
        Context budget in tokens:
        - GPT-OSS: 16000 tokens (~12.5% of 128k context)
        - Ollama/default: 12000 tokens (~36% of 32k context)
    """
    if RAG_PROVIDER == "gpt-oss":
        return RAG_GPT_OSS_CTX_BUDGET
    return CTX_TOKEN_BUDGET


def get_context_window() -> int:
    """Get the appropriate context window size based on active provider.

    Returns:
        Context window size in tokens:
        - GPT-OSS: 128000 tokens (128k)
        - Ollama/default: 32768 tokens (32k)
    """
    if RAG_PROVIDER == "gpt-oss":
        return RAG_GPT_OSS_CTX_WINDOW
    return DEFAULT_NUM_CTX


# Backwards-compatible aliases (legacy code/tests expect these names)
OLLAMA_URL = RAG_OLLAMA_URL
GEN_MODEL = RAG_CHAT_MODEL
EMB_MODEL = RAG_EMBED_MODEL

# ====== CHUNKING CONFIG ======
CHUNK_CHARS = _parse_env_int("CHUNK_CHARS", 1600, min_val=100, max_val=8000)
CHUNK_OVERLAP = _parse_env_int("CHUNK_OVERLAP", 200, min_val=0, max_val=4000)

# ====== RETRIEVAL CONFIG ======
# OPTIMIZATION: Increase retrieval parameters for better recall on internal deployment
DEFAULT_TOP_K = _parse_env_int("DEFAULT_TOP_K", 15, min_val=1, max_val=100)  # Was 12, now 15 (more candidates)
DEFAULT_PACK_TOP = _parse_env_int(
    "DEFAULT_PACK_TOP", 8, min_val=1, max_val=50
)  # Was 6, now 8 (more snippets in context)
DEFAULT_THRESHOLD = _parse_env_float(
    "DEFAULT_THRESHOLD", 0.25, min_val=0.0, max_val=1.0
)  # Was 0.30, now 0.25 (lower bar)
DEFAULT_SEED = 42

# OPTIMIZATION: Increase max query length for internal use (no DoS risk)
MAX_QUERY_LENGTH = _parse_env_int("MAX_QUERY_LENGTH", 1000000, min_val=100, max_val=10000000)  # Was 10K, now 1M

# ====== BM25 CONFIG ======
# BM25 parameters (tuned for technical documentation)
# OPTIMIZATION: Increase k1 from 1.0 to 1.2 for slightly better term frequency saturation
# OPTIMIZATION: Keep b at 0.65 for technical docs (reduces length penalty)
# FIX (Error #13): Use safe env var parsing
BM25_K1 = _parse_env_float("BM25_K1", 1.2, min_val=0.1, max_val=10.0)  # Was 1.0, now 1.2
BM25_B = _parse_env_float("BM25_B", 0.65, min_val=0.0, max_val=1.0)

# ====== LLM CONFIG ======
# OPTIMIZATION: Increase DEFAULT_NUM_CTX to 32768 to match Qwen 32B's full context window
# This allows us to use more context for better retrieval quality
# pack_snippets enforces effective_budget = min(CTX_TOKEN_BUDGET, num_ctx * 0.6)
# With value of 32768: effective = min(12000, 19660) = 12000 ✅
# Fully utilizes Qwen 32B's 32K context window capacity
# FIX (Error #13): Use safe env var parsing
DEFAULT_NUM_CTX = _parse_env_int("DEFAULT_NUM_CTX", 32768, min_val=512, max_val=128000)  # Was 16384, now 32768
# Allow overriding generation length via env for ops tuning
DEFAULT_NUM_PREDICT = _parse_env_int("DEFAULT_NUM_PREDICT", 512, min_val=32, max_val=4096)
# FIX: Increase default retries from 0 to 2 for remote Ollama resilience
# Remote endpoints (especially over VPN) benefit from transient error retry
# Can be overridden via DEFAULT_RETRIES env var or --retries CLI flag
DEFAULT_RETRIES = _parse_env_int("DEFAULT_RETRIES", 2, min_val=0, max_val=10)  # Was 0, now 2

# ====== MMR & CONTEXT BUDGET ======
# OPTIMIZATION: Increase MMR_LAMBDA to 0.75 to favor relevance slightly over diversity
MMR_LAMBDA = _parse_env_float("MMR_LAMBDA", 0.75, min_val=0.0, max_val=1.0)  # Was 0.7, now 0.75
# OPTIMIZATION: Increase context budget from 6000 to 12000 tokens to better utilize Qwen 32B's capacity
# Qwen 32B has 32K context window; we reserve 60% for snippets (pack_snippets enforces this)
# Old: 6000 tokens (~24K chars) was still conservative
# New: 12000 tokens (~48K chars) allows 2x more context while leaving room for Q+A
# Can be overridden via CTX_BUDGET env var
# FIX (Error #13): Use safe env var parsing
CTX_TOKEN_BUDGET = _parse_env_int("CTX_BUDGET", 12000, min_val=100, max_val=100000)  # Was 6000, now 12000

# ====== EMBEDDINGS BACKEND (v4.1) ======
EMB_BACKEND = (_get_env_value("EMB_BACKEND", "local") or "local").lower()  # "local" or "ollama"

# Embedding dimensions:
# - local (SentenceTransformer all-MiniLM-L6-v2): 384-dim
# - ollama (nomic-embed-text): 768-dim
EMB_DIM_LOCAL = 384
EMB_DIM_OLLAMA = 768
EMB_DIM = EMB_DIM_LOCAL if EMB_BACKEND == "local" else EMB_DIM_OLLAMA

# ====== ANN (Approximate Nearest Neighbors) (v4.1) ======
USE_ANN = (_get_env_value("ANN", "faiss") or "faiss").lower()  # "faiss" or "none"
# Note: nlist reduced from 256→64 for arm64 macOS stability (avoid IVF training segfault)
# FIX (Error #13): Use safe env var parsing
ANN_NLIST = _parse_env_int("ANN_NLIST", 64, min_val=8, max_val=1024)  # IVF clusters (reduced for stability)
ANN_NPROBE = _parse_env_int("ANN_NPROBE", 16, min_val=1, max_val=256)  # clusters to search
FAISS_IVF_MIN_ROWS = _parse_env_int("FAISS_IVF_MIN_ROWS", 20000, min_val=0, max_val=1_000_000)

# ====== HYBRID SCORING (v4.1) ======
# FIX (Error #13): Use safe env var parsing
ALPHA_HYBRID = _parse_env_float(
    "ALPHA", 0.5, min_val=0.0, max_val=1.0
)  # 0.5 = BM25 and dense equally weighted (fallback)

# ====== INTENT CLASSIFICATION (v5.9) ======
# OPTIMIZATION: Enable intent-based retrieval for +8-12% accuracy improvement
# When enabled, alpha_hybrid is dynamically adjusted based on query intent:
# - Procedural (how-to): 0.65 (favor BM25 for keyword matching)
# - Factual (what/define): 0.35 (favor dense for semantic understanding)
# - Pricing: 0.70 (high BM25 for exact pricing terms)
# - Troubleshooting: 0.60 (favor BM25 for error messages)
# - General: 0.50 (balanced, same as ALPHA_HYBRID)
USE_INTENT_CLASSIFICATION = _get_bool_env("USE_INTENT_CLASSIFICATION", "1")


# ====== KPI TIMINGS (v4.1) ======
class KPI:
    """Global KPI tracking for performance metrics."""

    retrieve_ms = 0
    ann_ms = 0
    rerank_ms = 0
    ask_ms = 0


# ====== TIMEOUT CONFIG ======
# Task G: Deterministic timeouts (environment-configurable for ops)
# FIX (Error #13): Use safe env var parsing
EMB_CONNECT_T = _parse_env_float("EMB_CONNECT_TIMEOUT", 3.0, min_val=0.1, max_val=60.0)
EMB_READ_T = _parse_env_float("EMB_READ_TIMEOUT", 60.0, min_val=1.0, max_val=600.0)
CHAT_CONNECT_T = _parse_env_float("CHAT_CONNECT_TIMEOUT", 3.0, min_val=0.1, max_val=60.0)
CHAT_READ_T = _parse_env_float("CHAT_READ_TIMEOUT", 120.0, min_val=1.0, max_val=600.0)
RERANK_READ_T = _parse_env_float("RERANK_READ_TIMEOUT", 180.0, min_val=1.0, max_val=600.0)

# ====== EMBEDDING BATCHING CONFIG (Rank 10) ======
# Parallel embedding generation for faster KB builds (3-5x speedup)
# FIX (Error #13): Use safe env var parsing
EMB_MAX_WORKERS = _parse_env_int("EMB_MAX_WORKERS", 8, min_val=1, max_val=64)  # Concurrent requests
EMB_BATCH_SIZE = _parse_env_int("EMB_BATCH_SIZE", 32, min_val=1, max_val=1000)  # Texts per batch

# ====== REFUSAL STRING ======
# Exact refusal string (ASCII quotes only)
REFUSAL_STR = "I don't know based on the MD."

# ====== LOGGING CONFIG ======


# Query logging configuration
QUERY_LOG_FILE = _get_env_value("RAG_LOG_FILE", "rag_queries.jsonl") or "rag_queries.jsonl"
LOG_QUERY_INCLUDE_ANSWER = _get_bool_env("RAG_LOG_INCLUDE_ANSWER", "1")
LOG_QUERY_ANSWER_PLACEHOLDER = _get_env_value("RAG_LOG_ANSWER_PLACEHOLDER", "[REDACTED]") or "[REDACTED]"
LOG_QUERY_INCLUDE_CHUNKS = _get_bool_env(
    "RAG_LOG_INCLUDE_CHUNKS", "0"
)  # Redact chunk text by default for security/privacy

# Citation validation configuration
STRICT_CITATIONS = _get_bool_env(
    "RAG_STRICT_CITATIONS", "0"
)  # Refuse answers without citations (improves trust in regulated environments)

# ====== CACHING & RATE LIMITING CONFIG ======
# Query cache size
CACHE_MAXSIZE = _parse_env_int("CACHE_MAXSIZE", 100, min_val=1, max_val=10000)
# Cache TTL in seconds
CACHE_TTL = _parse_env_int("CACHE_TTL", 3600, min_val=60, max_val=86400)
# Rate limiting: max requests per window
RATE_LIMIT_REQUESTS = _parse_env_int("RATE_LIMIT_REQUESTS", 10, min_val=1, max_val=1000)
# Rate limiting window in seconds
RATE_LIMIT_WINDOW = _parse_env_int("RATE_LIMIT_WINDOW", 60, min_val=1, max_val=3600)

# ====== API AUTH CONFIG ======
# API Configuration
API_HOST = _get_env_value("API_HOST", "127.0.0.1") or "127.0.0.1"
API_PORT = _parse_env_int("API_PORT", 8000, min_val=1, max_val=65535)
API_WORKERS = _parse_env_int("API_WORKERS", 4, min_val=1, max_val=64)
CORS_ENABLED = _get_bool_env("CORS_ENABLED", "1") # Enabled by default

API_AUTH_MODE = (_get_env_value("API_AUTH_MODE", "none") or "none").strip().lower()
_api_keys_raw = _get_env_value("API_ALLOWED_KEYS", "")
if _api_keys_raw.strip():
    API_ALLOWED_KEYS = frozenset(key.strip() for key in _api_keys_raw.split(",") if key.strip())
else:
    API_ALLOWED_KEYS = frozenset()
API_KEY_HEADER = (_get_env_value("API_KEY_HEADER", "x-api-key") or "x-api-key").strip() or "x-api-key"

# ====== WARMUP CONFIG ======
# Warm-up on startup
WARMUP_ENABLED = _get_bool_env("WARMUP", "1")

# ====== NLTK DOWNLOAD CONFIG ======
# Auto-download NLTK data
NLTK_AUTO_DOWNLOAD = _get_bool_env("NLTK_AUTO_DOWNLOAD", "1")

# ====== QUERY EXPANSION CONFIG ======
# Query expansion file path
CLOCKIFY_QUERY_EXPANSIONS = get_query_expansions_path()

# Maximum query expansion file size (in bytes)
MAX_QUERY_EXPANSION_FILE_SIZE = _parse_env_int(
    "MAX_QUERY_EXPANSION_FILE_SIZE", 10485760, min_val=1024, max_val=104857600
)  # 10MB default, 100MB max

# ====== PROXY CONFIGURATION ======
# Optional HTTP proxy support (disabled by default for security)
ALLOW_PROXIES = allow_proxies_enabled()  # Enable proxy usage when set to 1/true/yes
HTTP_PROXY = _get_env_value("HTTP_PROXY", "") or ""
HTTPS_PROXY = _get_env_value("HTTPS_PROXY", "") or ""

# Set proxy environment variables if allowed and configured
if ALLOW_PROXIES:
    if HTTP_PROXY:
        os.environ["HTTP_PROXY"] = HTTP_PROXY
    if HTTPS_PROXY:
        os.environ["HTTPS_PROXY"] = HTTPS_PROXY

# ====== FILE PATHS ======
FILES = {
    "chunks": "chunks.jsonl",
    "emb": "vecs_n.npy",  # Pre-normalized embeddings (float32)
    "emb_f16": "vecs_f16.memmap",  # float16 memory-mapped (optional)
    "emb_cache": "emb_cache.jsonl",  # Per-chunk embedding cache
    "meta": "meta.jsonl",
    "bm25": "bm25.json",
    "faiss_index": "faiss.index",  # FAISS IVFFlat index (v4.1)
    "hnsw": "hnsw_cosine.bin",  # Optional HNSW index (if USE_HNSWLIB=1)
    "index_meta": "index.meta.json",  # Artifact versioning
}

# ====== BUILD LOCK CONFIG ======
BUILD_LOCK = ".build.lock"
# FIX (Error #13): Use safe env var parsing
BUILD_LOCK_TTL_SEC = _parse_env_int("BUILD_LOCK_TTL_SEC", 900, min_val=60, max_val=7200)  # Task D: 15 minutes default

# ====== RETRIEVAL CONFIG (CONTINUED) ======
# FAISS/HNSW candidate generation (Quick Win #6)
# Expose FAISS candidate knobs through env for prod-level tuning
FAISS_CANDIDATE_MULTIPLIER = _parse_env_int(
    "FAISS_CANDIDATE_MULTIPLIER", 3, min_val=1, max_val=10
)  # Retrieve top_k * N
ANN_CANDIDATE_MIN = _parse_env_int(
    "ANN_CANDIDATE_MIN", 200, min_val=1, max_val=2000
)  # Minimum candidates even if top_k is small

# Reranking (Quick Win #6)
RERANK_SNIPPET_MAX_CHARS = 500  # Truncate chunk text for reranking prompt
RERANK_MAX_CHUNKS = 12  # Maximum chunks to send to reranking

# Retrieval thresholds (Quick Win #6)
COVERAGE_MIN_CHUNKS = 2  # Minimum chunks above threshold to proceed

# ====== PRECOMPUTED FAQ CACHE (Analysis Section 9.1 #3) ======
# OPTIMIZATION: Pre-generate answers for top FAQs for 100% cache hit on common queries
FAQ_CACHE_ENABLED = _get_bool_env("FAQ_CACHE_ENABLED", "0")  # Disabled by default (requires build step)
FAQ_CACHE_PATH = _get_env_value("FAQ_CACHE_PATH", "faq_cache.json") or "faq_cache.json"


# ====== CONFIGURATION VALIDATION ======


def validate_config() -> dict:
    """Validate the current RAG configuration for common issues.

    Checks for:
    - URL format and reachability (basic validation)
    - Fallback configuration sanity
    - Timeout settings reasonableness
    - Model name consistency

    Returns:
        Dictionary with validation results:
        {
            "valid": bool,  # True if all checks pass
            "warnings": List[str],  # List of warning messages
            "errors": List[str],  # List of error messages
            "config_snapshot": dict,  # Current config values
        }

    Note:
        This does NOT check if the server is actually reachable.
        Use api_client.validate_models() to check server connectivity
        and model availability.
    """
    warnings = []
    errors = []

    # Check URL format
    if not RAG_OLLAMA_URL.startswith(("http://", "https://")):
        errors.append(f"RAG_OLLAMA_URL must start with http:// or https://, got: {RAG_OLLAMA_URL}")

    # Check fallback configuration
    if RAG_FALLBACK_ENABLED:
        if RAG_PROVIDER == RAG_FALLBACK_PROVIDER:
            warnings.append(
                f"Fallback enabled but primary provider ({RAG_PROVIDER}) "
                f"is same as fallback provider ({RAG_FALLBACK_PROVIDER}). "
                "Fallback will have no effect."
            )

        if RAG_PROVIDER == "gpt-oss" and RAG_FALLBACK_PROVIDER == "gpt-oss":
            warnings.append(
                "Both primary and fallback are set to gpt-oss. " "Consider setting fallback to 'ollama' for redundancy."
            )

    # Check timeout settings
    if CHAT_READ_T < 30:
        warnings.append(
            f"CHAT_READ_T ({CHAT_READ_T}s) is very short. "
            "LLM responses may timeout. Consider increasing to at least 60s."
        )

    if EMB_READ_T < 10:
        warnings.append(
            f"EMB_READ_T ({EMB_READ_T}s) is very short. "
            "Embedding generation may timeout. Consider increasing to at least 30s."
        )

    # Check retrieval parameters
    if DEFAULT_TOP_K < DEFAULT_PACK_TOP:
        errors.append(f"DEFAULT_TOP_K ({DEFAULT_TOP_K}) must be >= DEFAULT_PACK_TOP ({DEFAULT_PACK_TOP})")

    if DEFAULT_THRESHOLD < 0.0 or DEFAULT_THRESHOLD > 1.0:
        errors.append(f"DEFAULT_THRESHOLD ({DEFAULT_THRESHOLD}) must be between 0.0 and 1.0")

    # Check context budget
    if CTX_TOKEN_BUDGET <= 0:
        errors.append(f"CTX_TOKEN_BUDGET ({CTX_TOKEN_BUDGET}) must be > 0")

    if RAG_PROVIDER == "gpt-oss" and RAG_GPT_OSS_CTX_BUDGET > RAG_GPT_OSS_CTX_WINDOW:
        warnings.append(
            f"GPT-OSS context budget ({RAG_GPT_OSS_CTX_BUDGET}) "
            f"exceeds context window ({RAG_GPT_OSS_CTX_WINDOW}). "
            "Responses may be truncated."
        )

    # Build config snapshot
    config_snapshot = {
        "llm_endpoint": RAG_OLLAMA_URL,
        "provider": RAG_PROVIDER,
        "chat_model": RAG_CHAT_MODEL,
        "embed_model": RAG_EMBED_MODEL,
        "fallback_enabled": RAG_FALLBACK_ENABLED,
        "fallback_provider": RAG_FALLBACK_PROVIDER if RAG_FALLBACK_ENABLED else None,
        "fallback_model": RAG_FALLBACK_MODEL if RAG_FALLBACK_ENABLED else None,
        "chat_timeout": CHAT_READ_T,
        "embed_timeout": EMB_READ_T,
        "top_k": DEFAULT_TOP_K,
        "pack_top": DEFAULT_PACK_TOP,
        "threshold": DEFAULT_THRESHOLD,
        "ctx_budget": get_context_budget(),
        "ctx_window": get_context_window(),
    }

    return {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
        "config_snapshot": config_snapshot,
    }
