"""Query caching and rate limiting for RAG system."""

import hashlib
import logging
import os
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

# FIX (Error #2): Declare globals at module level for safe initialization
_RATE_LIMITER = None
_QUERY_CACHE = None


class RateLimiter:
    """Sliding window rate limiter with configurable enable/disable.

    Can be disabled for internal deployment (adds ~5-10ms overhead when enabled).
    When disabled, all methods return permissive values (backward compatible).

    Uses a sliding window algorithm to track requests per time window.
    Thread-safe for concurrent API access.
    """

    def __init__(self, max_requests=10, window_seconds=60, enabled=True):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            enabled: Enable rate limiting (default: True, set False for internal deployment)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.enabled = enabled

        if self.enabled:
            self.requests = deque()  # (timestamp,) tuples
            self._lock = threading.Lock()
            logger.info(f"RateLimiter enabled: {max_requests} requests per {window_seconds}s window")
        else:
            logger.info("RateLimiter disabled (no-op for internal deployment)")

    def _clean_old_requests(self, now: float):
        """Remove requests older than window (internal helper, assumes lock held)."""
        cutoff = now - self.window_seconds
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

    def allow_request(self) -> bool:
        """Check if request is allowed under rate limit.

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        if not self.enabled:
            return True  # Always allow when disabled

        with self._lock:
            now = time.time()
            self._clean_old_requests(now)

            # Check if within limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True

            return False

    def wait_time(self) -> float:
        """Calculate seconds to wait before next request is allowed.

        Returns:
            Seconds to wait (0.0 if request would be allowed now)
        """
        if not self.enabled:
            return 0.0  # Never wait when disabled

        with self._lock:
            now = time.time()
            self._clean_old_requests(now)

            # If under limit, no wait needed
            if len(self.requests) < self.max_requests:
                return 0.0

            # Calculate when oldest request will expire
            oldest = self.requests[0]
            wait = (oldest + self.window_seconds) - now
            return max(0.0, wait)


# Global rate limiter (10 queries per minute by default, disabled for internal use)
def get_rate_limiter():
    """Get global rate limiter instance.

    FIX (Error #2): Use proper `is None` check instead of fragile globals() check.

    Rate limiting is disabled by default for internal deployment.
    Set RATE_LIMIT_ENABLED=true in environment to enable for public APIs.
    """
    from . import config  # Import here to avoid circular import

    global _RATE_LIMITER
    if _RATE_LIMITER is None:
        _RATE_LIMITER = RateLimiter(
            max_requests=config.RATE_LIMIT_REQUESTS,
            window_seconds=config.RATE_LIMIT_WINDOW,
            enabled=config.RATE_LIMIT_ENABLED,
        )
    return _RATE_LIMITER


class QueryCache:
    """TTL-based cache for repeated queries to eliminate redundant computation.

    Features:
    - LRU eviction with configurable maxsize
    - TTL-based expiration
    - Thread-safe for concurrent access
    - Optional automatic persistence with background thread
    """

    def __init__(
        self,
        maxsize=100,
        ttl_seconds=3600,
        auto_save_enabled=False,
        auto_save_interval=300,
        save_path="query_cache.json",
    ):
        """Initialize query cache.

        Args:
            maxsize: Maximum number of cached queries (LRU eviction)
            ttl_seconds: Time-to-live for cache entries in seconds
            auto_save_enabled: Enable automatic background persistence (default: False)
            auto_save_interval: Auto-save interval in seconds (default: 300 = 5 minutes)
            save_path: File path for cache persistence (default: query_cache.json)
        """
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.cache: dict = {}  # {question_hash: (answer, metadata_with_timestamp, timestamp)}
        # FIX (Error #4): Add maxlen as defense-in-depth safety net
        # maxlen = maxsize * 2 provides safety buffer if cleanup fails
        self.access_order: deque = deque(maxlen=maxsize * 2)  # For LRU eviction
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()  # Thread safety lock

        # Auto-save configuration
        self.auto_save_enabled = auto_save_enabled
        self.auto_save_interval = auto_save_interval
        self.save_path = save_path
        self._save_thread = None
        self._stop_save_thread = threading.Event()
        self._dirty = False  # Track if cache has unsaved changes

        # Start auto-save thread if enabled
        if self.auto_save_enabled:
            self._start_auto_save_thread()
            logger.info(f"Cache auto-save enabled: interval={auto_save_interval}s, path={save_path}")

    def _start_auto_save_thread(self):
        """Start background thread for automatic cache persistence."""
        if self._save_thread is not None:
            logger.warning("Auto-save thread already running")
            return

        self._stop_save_thread.clear()
        self._save_thread = threading.Thread(target=self._auto_save_loop, daemon=True, name="QueryCache-AutoSave")
        self._save_thread.start()
        logger.debug("Auto-save thread started")

    def _auto_save_loop(self):
        """Background thread loop for periodic cache saving."""
        while not self._stop_save_thread.wait(timeout=self.auto_save_interval):
            try:
                # Only save if there are unsaved changes
                with self._lock:
                    if self._dirty and len(self.cache) > 0:
                        logger.debug("Auto-save: saving cache (dirty flag set)")
                        self.save(self.save_path)
                        self._dirty = False
                    else:
                        logger.debug("Auto-save: skipping (no changes or empty cache)")
            except Exception as e:
                logger.warning(f"Auto-save failed: {e}")

        # Final save on shutdown if dirty
        try:
            with self._lock:
                if self._dirty and len(self.cache) > 0:
                    logger.info("Auto-save: final save on shutdown")
                    self.save(self.save_path)
                    self._dirty = False
        except Exception as e:
            logger.warning(f"Auto-save final save failed: {e}")

    def stop_auto_save(self):
        """Stop the auto-save background thread gracefully."""
        if self._save_thread is None:
            return

        logger.info("Stopping auto-save thread...")
        self._stop_save_thread.set()

        # Wait for thread to finish (with timeout)
        self._save_thread.join(timeout=5.0)
        if self._save_thread.is_alive():
            logger.warning("Auto-save thread did not stop within timeout")
        else:
            logger.info("Auto-save thread stopped")

        self._save_thread = None

    def _hash_question(self, question: str, params: dict = None) -> str:
        """Generate cache key from question and retrieval parameters.

        Args:
            question: User question
            params: Retrieval parameters (top_k, pack_top, use_rerank, threshold)
        """
        if params is None:
            cache_input = question
        else:
            # Sort params for consistent hashing
            sorted_params = sorted(params.items())
            cache_input = question + str(sorted_params)
        return hashlib.md5(cache_input.encode("utf-8")).hexdigest()

    def get(self, question: str, params: dict = None):
        """Retrieve cached answer if available and not expired.

        Args:
            question: User question
            params: Retrieval parameters (optional, for cache key)

        Returns:
            (answer, metadata) tuple if cache hit, None if cache miss
        """
        with self._lock:
            key = self._hash_question(question, params)

            if key not in self.cache:
                self.misses += 1
                return None

            answer, metadata, timestamp = self.cache[key]
            # Ensure metadata exposes cache timestamp for downstream logging
            metadata_timestamp = metadata.get("timestamp")
            if metadata_timestamp is None:
                metadata_timestamp = timestamp
                metadata["timestamp"] = metadata_timestamp

            age = time.time() - metadata_timestamp

            # Check if expired
            if age > self.ttl_seconds:
                del self.cache[key]
                self.access_order.remove(key)
                self.misses += 1
                return None

            # Cache hit - update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            self.hits += 1
            logger.debug(f"[cache] HIT question_hash={key[:8]} age={age:.1f}s")
            return answer, metadata

    def put(self, question: str, answer: str, metadata: dict, params: dict = None):
        """Store answer in cache.

        Args:
            question: User question
            answer: Generated answer
            metadata: Answer metadata (selected chunks, scores, etc.)
            params: Retrieval parameters (optional, for cache key)
        """
        with self._lock:
            key = self._hash_question(question, params)

            # Evict oldest entry if cache full
            if len(self.cache) >= self.maxsize and key not in self.cache:
                oldest = self.access_order.popleft()
                del self.cache[oldest]
                logger.debug(f"[cache] EVICT question_hash={oldest[:8]} (LRU)")

            # Store entry with timestamp
            # OPTIMIZATION: Use shallow copy instead of deepcopy for performance.
            # The metadata dict is created fresh in answer_once and not mutated in place by callers.
            timestamp = time.time()
            metadata_copy = metadata.copy() if metadata is not None else {}
            metadata_copy["timestamp"] = timestamp
            self.cache[key] = (answer, metadata_copy, timestamp)

            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

            # Mark cache as dirty for auto-save
            self._dirty = True

            logger.debug(f"[cache] PUT question_hash={key[:8]}")

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0
            self._dirty = False  # Nothing to save after clear
            logger.info("[cache] CLEAR")

    def __del__(self):
        """Destructor to ensure auto-save thread is stopped."""
        try:
            self.stop_auto_save()
        except Exception:
            pass  # Best-effort cleanup

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, size, hit_rate
        """
        with self._lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "size": len(self.cache),
                "maxsize": self.maxsize,
                "hit_rate": hit_rate,
            }

    def save(self, path: str = "query_cache.json"):
        """Save cache to disk for persistence across sessions.

        OPTIMIZATION: Enables 100% cache hit rate on repeated queries after restart.

        Args:
            path: File path to save cache (default: query_cache.json)
        """
        import json

        with self._lock:
            try:
                cache_data = {
                    "version": "1.0",
                    "maxsize": self.maxsize,
                    "ttl_seconds": self.ttl_seconds,
                    "entries": [
                        {"key": key, "answer": answer, "metadata": metadata, "timestamp": timestamp}
                        for key, (answer, metadata, timestamp) in self.cache.items()
                    ],
                    "access_order": list(self.access_order),
                    "hits": self.hits,
                    "misses": self.misses,
                }
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                logger.info(f"[cache] SAVE {len(self.cache)} entries to {path}")
            except Exception as e:
                logger.warning(f"[cache] Failed to save cache: {e}")

    def load(self, path: str = "query_cache.json"):
        """Load cache from disk to restore across sessions.

        OPTIMIZATION: Restores previous session's cache for instant hits on repeated queries.

        Args:
            path: File path to load cache from (default: query_cache.json)

        Returns:
            Number of entries loaded (0 if file doesn't exist or load fails)
        """
        import json

        with self._lock:
            if not os.path.exists(path):
                logger.debug(f"[cache] No cache file found at {path}")
                return 0

            try:
                with open(path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                # Validate version
                version = cache_data.get("version", "1.0")
                if version != "1.0":
                    logger.warning(f"[cache] Incompatible cache version {version}, skipping load")
                    return 0

                # Restore entries, filtering out expired ones
                now = time.time()
                loaded_count = 0
                for entry in cache_data.get("entries", []):
                    key = entry["key"]
                    answer = entry["answer"]
                    metadata = entry["metadata"]
                    timestamp = entry["timestamp"]

                    # Skip expired entries
                    age = now - timestamp
                    if age > self.ttl_seconds:
                        continue

                    self.cache[key] = (answer, metadata, timestamp)
                    loaded_count += 1

                # Restore access order (only for non-expired keys)
                self.access_order = deque(
                    [k for k in cache_data.get("access_order", []) if k in self.cache], maxlen=self.maxsize * 2
                )

                # Restore stats (reset to avoid inflated numbers from old sessions)
                # self.hits = cache_data.get("hits", 0)
                # self.misses = cache_data.get("misses", 0)

                logger.info(
                    f"[cache] LOAD {loaded_count} entries from {path} (skipped {len(cache_data.get('entries', [])) - loaded_count} expired)"
                )
                return loaded_count

            except Exception as e:
                logger.warning(f"[cache] Failed to load cache: {e}")
                return 0


# Global query cache (100 entries, 1 hour TTL by default)
def get_query_cache():
    """Get global query cache instance.

    FIX (Error #2): Use proper `is None` check instead of fragile globals() check.

    Auto-save is disabled by default. Set CACHE_AUTO_SAVE_ENABLED=true to enable
    automatic background persistence (prevents cache loss on crashes).
    """
    from . import config  # Import here to avoid circular import

    global _QUERY_CACHE
    if _QUERY_CACHE is None:
        _QUERY_CACHE = QueryCache(
            maxsize=config.CACHE_MAXSIZE,
            ttl_seconds=config.CACHE_TTL,
            auto_save_enabled=config.CACHE_AUTO_SAVE_ENABLED,
            auto_save_interval=config.CACHE_AUTO_SAVE_INTERVAL,
            save_path=config.FILES.get("query_cache", "query_cache.json"),
        )
    return _QUERY_CACHE


def log_query(
    query: str, answer: str, retrieved_chunks: list, latency_ms: float, refused: bool = False, metadata: dict = None
):
    """Log query with structured JSON format for monitoring and analytics.

    FIX (Error #6): Sanitizes user input to prevent log injection attacks.
    """
    import json
    from .config import (
        LOG_QUERY_ANSWER_PLACEHOLDER,
        LOG_QUERY_INCLUDE_ANSWER,
        LOG_QUERY_INCLUDE_CHUNKS,
        QUERY_LOG_FILE,
    )
    from .utils import sanitize_for_log

    normalized_chunks = []
    for chunk in retrieved_chunks:
        if isinstance(chunk, dict):
            normalized = chunk.copy()
            chunk_id = normalized.get("id") or normalized.get("chunk_id")
            normalized["id"] = chunk_id
            normalized["dense"] = float(normalized.get("dense", normalized.get("score", 0.0)))
            normalized["bm25"] = float(normalized.get("bm25", 0.0))
            normalized["hybrid"] = float(normalized.get("hybrid", normalized["dense"]))
            # Redact chunk text for security/privacy unless explicitly enabled
            if not LOG_QUERY_INCLUDE_CHUNKS:
                normalized.pop("chunk", None)  # Remove full chunk text
                normalized.pop("text", None)  # Remove text field if present
        else:
            normalized = {
                "id": chunk,
                "dense": 0.0,
                "bm25": 0.0,
                "hybrid": 0.0,
            }
        normalized_chunks.append(normalized)

    chunk_ids = [c.get("id") for c in normalized_chunks]
    dense_scores = [c.get("dense", 0.0) for c in normalized_chunks]
    bm25_scores = [c.get("bm25", 0.0) for c in normalized_chunks]
    hybrid_scores = [c.get("hybrid", 0.0) for c in normalized_chunks]
    primary_scores = hybrid_scores if hybrid_scores else []
    avg_chunk_score = (sum(primary_scores) / len(primary_scores)) if primary_scores else 0.0
    max_chunk_score = max(primary_scores) if primary_scores else 0.0

    # FIX: Sanitize metadata to prevent chunk text leaks
    # Deep copy and remove any 'text'/'chunk' fields from nested structures
    import copy

    sanitized_metadata = copy.deepcopy(metadata) if metadata else {}
    if not LOG_QUERY_INCLUDE_CHUNKS and isinstance(sanitized_metadata, dict):
        # Remove chunk text from any nested chunk dicts in metadata
        for key in list(sanitized_metadata.keys()):
            val = sanitized_metadata[key]
            if isinstance(val, dict):
                val.pop("text", None)
                val.pop("chunk", None)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        item.pop("text", None)
                        item.pop("chunk", None)

    # FIX (Error #6): Sanitize query and answer to prevent log injection
    log_entry = {
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": sanitize_for_log(query, max_length=2000),
        "refused": refused,
        "latency_ms": latency_ms,
        "num_chunks_retrieved": len(chunk_ids),
        "chunk_ids": chunk_ids,
        "chunk_scores": {
            "dense": dense_scores,
            "bm25": bm25_scores,
            "hybrid": hybrid_scores,
        },
        "retrieved_chunks": normalized_chunks,
        "avg_chunk_score": avg_chunk_score,
        "max_chunk_score": max_chunk_score,
        "metadata": sanitized_metadata,
    }

    if LOG_QUERY_INCLUDE_ANSWER:
        log_entry["answer"] = sanitize_for_log(answer, max_length=5000)
    elif LOG_QUERY_ANSWER_PLACEHOLDER:
        log_entry["answer"] = LOG_QUERY_ANSWER_PLACEHOLDER

    try:
        with open(QUERY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log query: {e}")
