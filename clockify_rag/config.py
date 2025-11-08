"""Configuration constants for Clockify RAG system."""

import logging
import os

# FIX (Error #13): Helper functions for safe environment variable parsing
_logger = logging.getLogger(__name__)


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
        _logger.error(
            f"Invalid float for {key}='{value}': {e}. "
            f"Using default: {default}"
        )
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
        _logger.error(
            f"Invalid integer for {key}='{value}': {e}. "
            f"Using default: {default}"
        )
        return default

    if min_val is not None and parsed < min_val:
        _logger.warning(f"{key}={parsed} below minimum {min_val}, clamping")
        return min_val
    if max_val is not None and parsed > max_val:
        _logger.warning(f"{key}={parsed} above maximum {max_val}, clamping")
        return max_val

    return parsed


# ====== OLLAMA CONFIG ======
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen2.5:32b")
EMB_MODEL = os.environ.get("EMB_MODEL", "nomic-embed-text")

# ====== CHUNKING CONFIG ======
CHUNK_CHARS = 1600
CHUNK_OVERLAP = 200

# ====== RETRIEVAL CONFIG ======
DEFAULT_TOP_K = 12
DEFAULT_PACK_TOP = 6
DEFAULT_THRESHOLD = 0.30
DEFAULT_SEED = 42

# FIX (Error #5): Input validation to prevent DoS attacks
MAX_QUERY_LENGTH = _parse_env_int("MAX_QUERY_LENGTH", 10000, min_val=100, max_val=100000)  # 10K chars max

# ====== BM25 CONFIG ======
# BM25 parameters (tuned for technical documentation)
# Lower k1 (1.2→1.0): Reduces term frequency saturation for repeated technical terms
# Lower b (0.75→0.65): Reduces length normalization penalty for longer docs
# FIX (Error #13): Use safe env var parsing
BM25_K1 = _parse_env_float("BM25_K1", 1.0, min_val=0.1, max_val=10.0)
BM25_B = _parse_env_float("BM25_B", 0.65, min_val=0.0, max_val=1.0)

# ====== LLM CONFIG ======
# FIX: Increase DEFAULT_NUM_CTX from 8192 to 16384 to support CTX_TOKEN_BUDGET of 6000
# pack_snippets enforces effective_budget = min(CTX_TOKEN_BUDGET, num_ctx * 0.6)
# With old value of 8192: effective = min(6000, 4915) = 4915 ❌
# With new value of 16384: effective = min(6000, 9830) = 6000 ✅
# Still well within Qwen 32B's 32K context window capacity
# FIX (Error #13): Use safe env var parsing
DEFAULT_NUM_CTX = _parse_env_int("DEFAULT_NUM_CTX", 16384, min_val=512, max_val=128000)  # Was 8192, now 16384
DEFAULT_NUM_PREDICT = 512
# FIX: Increase default retries from 0 to 2 for remote Ollama resilience
# Remote endpoints (especially over VPN) benefit from transient error retry
# Can be overridden via DEFAULT_RETRIES env var or --retries CLI flag
DEFAULT_RETRIES = _parse_env_int("DEFAULT_RETRIES", 2, min_val=0, max_val=10)  # Was 0, now 2

# ====== MMR & CONTEXT BUDGET ======
MMR_LAMBDA = 0.7
# FIX: Increase context budget from 2800 to 6000 tokens to better utilize Qwen 32B's capacity
# Qwen 32B has 32K context window; we reserve 60% for snippets (pack_snippets enforces this)
# Old: 2800 tokens (~11K chars) was too conservative, causing unnecessary truncation
# New: 6000 tokens (~24K chars) allows more context while leaving room for Q+A
# Can be overridden via CTX_BUDGET env var
# FIX (Error #13): Use safe env var parsing
CTX_TOKEN_BUDGET = _parse_env_int("CTX_BUDGET", 6000, min_val=100, max_val=100000)  # Was 2800, now 6000

# ====== EMBEDDINGS BACKEND (v4.1) ======
EMB_BACKEND = os.environ.get("EMB_BACKEND", "local")  # "local" or "ollama"

# Embedding dimensions:
# - local (SentenceTransformer all-MiniLM-L6-v2): 384-dim
# - ollama (nomic-embed-text): 768-dim
EMB_DIM_LOCAL = 384
EMB_DIM_OLLAMA = 768
EMB_DIM = EMB_DIM_LOCAL if EMB_BACKEND == "local" else EMB_DIM_OLLAMA

# ====== ANN (Approximate Nearest Neighbors) (v4.1) ======
USE_ANN = os.environ.get("ANN", "faiss")  # "faiss" or "none"
# Note: nlist reduced from 256→64 for arm64 macOS stability (avoid IVF training segfault)
# FIX (Error #13): Use safe env var parsing
ANN_NLIST = _parse_env_int("ANN_NLIST", 64, min_val=8, max_val=1024)  # IVF clusters (reduced for stability)
ANN_NPROBE = _parse_env_int("ANN_NPROBE", 16, min_val=1, max_val=256)  # clusters to search

# ====== HYBRID SCORING (v4.1) ======
# FIX (Error #13): Use safe env var parsing
ALPHA_HYBRID = _parse_env_float("ALPHA", 0.5, min_val=0.0, max_val=1.0)  # 0.5 = BM25 and dense equally weighted

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


def _get_bool_env(var_name: str, default: str = "1") -> bool:
    """Read a boolean environment variable."""

    value = os.environ.get(var_name, default)
    return value.lower() not in {"0", "false", "no", "off", ""}


# Query logging configuration
QUERY_LOG_FILE = os.environ.get("RAG_LOG_FILE", "rag_queries.jsonl")
LOG_QUERY_INCLUDE_ANSWER = _get_bool_env("RAG_LOG_INCLUDE_ANSWER", "1")
LOG_QUERY_ANSWER_PLACEHOLDER = os.environ.get("RAG_LOG_ANSWER_PLACEHOLDER", "[REDACTED]")
LOG_QUERY_INCLUDE_CHUNKS = _get_bool_env("RAG_LOG_INCLUDE_CHUNKS", "0")  # Redact chunk text by default for security/privacy

# Citation validation configuration
STRICT_CITATIONS = _get_bool_env("RAG_STRICT_CITATIONS", "0")  # Refuse answers without citations (improves trust in regulated environments)

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
FAISS_CANDIDATE_MULTIPLIER = 3  # Retrieve top_k * 3 candidates for reranking
ANN_CANDIDATE_MIN = 200  # Minimum candidates even if top_k is small

# Reranking (Quick Win #6)
RERANK_SNIPPET_MAX_CHARS = 500  # Truncate chunk text for reranking prompt
RERANK_MAX_CHUNKS = 12  # Maximum chunks to send to reranking

# Retrieval thresholds (Quick Win #6)
COVERAGE_MIN_CHUNKS = 2  # Minimum chunks above threshold to proceed
