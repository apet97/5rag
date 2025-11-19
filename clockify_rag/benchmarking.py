"""Benchmark suite for Clockify RAG.

Measures:
- Latency: Time to complete operations
- Throughput: Operations per second
- Memory: Memory usage during operations
"""

import gc
import json
import os
import time
import tracemalloc
import logging
from statistics import mean, median, stdev
from typing import Callable, List, Dict, Any, Optional

import numpy as np

from . import config
from .retrieval import retrieve, embed_query, RETRIEVE_PROFILE_LAST
from .embedding import embed_texts
from .answer import answer_once
from .chunking import build_chunks
from .indexing import load_index

logger = logging.getLogger(__name__)

# Allow CI smoke tests to bypass external services
if os.environ.get("BENCHMARK_FAKE_REMOTE") == "1":

    def _fake_embed_query(question: str, retries: int = 0) -> np.ndarray:
        """Deterministic unit-vector embedding based on question hash."""
        seed = abs(hash(question)) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=config.EMB_DIM).astype("float32")
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    def _fake_embed_texts(texts, retries: int = 0):
        if not texts:
            return np.zeros((0, config.EMB_DIM), dtype="float32")
        vecs = [_fake_embed_query(t, retries) for t in texts]
        return np.vstack(vecs).astype("float32")

    def _fake_answer_once(
        question,
        chunks,
        vecs_n,
        bm,
        top_k=12,
        pack_top=6,
        threshold=0.30,
        use_rerank=False,
        debug=False,
        hnsw=None,
        seed=0,
        num_ctx=0,
        num_predict=0,
        retries=0,
        faiss_index_path=None,
    ):
        """Offline-friendly answer stub using hybrid retrieval only."""
        selected, scores = retrieve(
            question, chunks, vecs_n, bm, top_k=top_k, hnsw=hnsw, retries=retries, faiss_index_path=faiss_index_path
        )
        summary_chunks = [chunks[i]["text"] for i in selected[:1]]
        answer_text = summary_chunks[0] if summary_chunks else "No answer available."
        metadata = {
            "selected": [chunks[i]["id"] for i in selected],
            "scores": scores,
            "timings": {},
            "cached": False,
            "cache_hit": False,
        }
        return {"answer": answer_text, "metadata": metadata}

    # Monkey-patching is handled by the caller or we can use dependency injection
    # For now, we'll just expose these fakes if needed, but the module functions
    # use the imports. To make this work inside the module without monkey-patching
    # global imports which might affect other threads, we might need a context or
    # just rely on the environment variable check in the actual functions if we were
    # rewriting them. But `benchmark.py` did monkey-patching.
    #
    # A better approach for the library is to allow passing the functions to benchmark.


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str):
        self.name = name
        self.latencies = []  # milliseconds
        self.memory_peak = 0  # bytes
        self.memory_current = 0  # bytes
        self.metadata = {}

    def add_latency(self, latency_ms: float):
        self.latencies.append(latency_ms)

    def set_memory(self, peak_bytes: int, current_bytes: int):
        self.memory_peak = peak_bytes
        self.memory_current = current_bytes

    def set_metadata(self, **kwargs):
        self.metadata.update(kwargs)

    def summary(self) -> dict:
        """Get summary statistics."""
        if not self.latencies:
            return {"name": self.name, "error": "No measurements"}

        summary = {
            "name": self.name,
            "latency_ms": {
                "mean": round(mean(self.latencies), 2),
                "median": round(median(self.latencies), 2),
                "stdev": round(stdev(self.latencies), 2) if len(self.latencies) > 1 else 0,
                "min": round(min(self.latencies), 2),
                "max": round(max(self.latencies), 2),
                "p95": round(sorted(self.latencies)[int(len(self.latencies) * 0.95)], 2),
            },
            "throughput": {
                "ops_per_sec": round(1000 / mean(self.latencies), 2),
            },
            "memory_mb": {
                "peak": round(self.memory_peak / 1024 / 1024, 2),
                "current": round(self.memory_current / 1024 / 1024, 2),
            },
            "iterations": len(self.latencies),
        }

        if self.metadata:
            summary["metadata"] = self.metadata

        return summary


def run_benchmark(func: Callable, iterations: int = 10, warmup: int = 2) -> BenchmarkResult:
    """Benchmark a function with latency and memory tracking."""
    result = BenchmarkResult(func.__name__)

    # Warmup
    for _ in range(warmup):
        func()
        gc.collect()

    # Benchmark with memory tracking
    tracemalloc.start()
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        result.add_latency((end - start) * 1000)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result.set_memory(peak, current)

    return result


def aggregate_retrieval_profiles(profiles: List[Dict]) -> Dict:
    """Aggregate retrieval profiling samples into summary stats."""
    if not profiles:
        return {}

    dot_mean = mean(p.get("dense_dot_time_ms", 0.0) for p in profiles)
    computed_mean = mean(p.get("dense_computed", 0) for p in profiles)
    saved_mean = mean(p.get("dense_saved", 0) for p in profiles)
    candidates_mean = mean(p.get("candidates", 0) for p in profiles)
    total_saved = sum(p.get("dense_saved", 0) for p in profiles)
    total = sum(p.get("dense_total", 0) for p in profiles)
    saved_ratio = round(total_saved / total, 3) if total else 0.0

    if any(p.get("used_faiss") for p in profiles):
        ann_mode = "faiss"
    elif any(p.get("used_hnsw") for p in profiles):
        ann_mode = "hnsw"
    else:
        ann_mode = "linear"

    return {
        "ann_mode": ann_mode,
        "dense_dot_ms_mean": round(dot_mean, 3),
        "dense_computed_mean": round(computed_mean, 2),
        "dense_saved_mean": round(saved_mean, 2),
        "dense_saved_ratio": saved_ratio,
        "candidates_mean": round(candidates_mean, 2),
    }


def benchmark_embedding_single(chunks, iterations=10):
    """Benchmark single text embedding."""
    text = chunks[0]["text"] if chunks else "How do I track time in Clockify?"

    # Use fake if env var set (handled by module level check if we were monkey patching,
    # but here we just use the imported function. If we want to support the fake remote
    # we need to swap the function being called)
    fn_to_call = _fake_embed_query if os.environ.get("BENCHMARK_FAKE_REMOTE") == "1" else embed_query

    def run():
        fn_to_call(text)

    result = run_benchmark(run, iterations=iterations, warmup=2)
    result.name = "embed_single"
    return result


def benchmark_embedding_batch(chunks, iterations=5):
    """Benchmark batch embedding (10 chunks)."""
    batch = chunks[:10] if len(chunks) >= 10 else chunks
    texts = [c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in batch]

    fn_to_call = _fake_embed_texts if os.environ.get("BENCHMARK_FAKE_REMOTE") == "1" else embed_texts

    def run():
        fn_to_call(texts)

    result = run_benchmark(run, iterations=iterations, warmup=1)
    result.name = "embed_batch_10"
    return result


def benchmark_embedding_large_batch(chunks, iterations=3):
    """Benchmark large batch embedding (100 chunks)."""
    batch = chunks[:100] if len(chunks) >= 100 else chunks
    texts = [c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in batch]

    fn_to_call = _fake_embed_texts if os.environ.get("BENCHMARK_FAKE_REMOTE") == "1" else embed_texts

    def run():
        fn_to_call(texts)

    result = run_benchmark(run, iterations=iterations, warmup=1)
    result.name = "embed_batch_100"
    return result


def benchmark_retrieval_hybrid(chunks, vecs_n, bm, hnsw=None, faiss_index_path=None, iterations=20):
    """Benchmark hybrid (BM25 + dense) retrieval."""
    question = "How do I track time in Clockify?"
    profiles = []

    def run():
        retrieve(question, chunks, vecs_n, bm, top_k=12, hnsw=hnsw, faiss_index_path=faiss_index_path)
        if RETRIEVE_PROFILE_LAST:
            profiles.append(dict(RETRIEVE_PROFILE_LAST))

    result = run_benchmark(run, iterations=iterations, warmup=3)
    result.name = "retrieve_hybrid"
    if profiles:
        result.set_metadata(**aggregate_retrieval_profiles(profiles))
    return result


def benchmark_retrieval_with_mmr(chunks, vecs_n, bm, hnsw=None, faiss_index_path=None, iterations=20):
    """Benchmark retrieval + MMR diversification."""
    question = "How do I track time in Clockify?"
    profiles = []

    def run():
        selected, scores = retrieve(
            question, chunks, vecs_n, bm, top_k=12, hnsw=hnsw, faiss_index_path=faiss_index_path
        )
        # Simulate MMR (already included in answer_once, but measure separately)
        _ = selected[:6]  # Pack top 6
        if RETRIEVE_PROFILE_LAST:
            profiles.append(dict(RETRIEVE_PROFILE_LAST))

    result = run_benchmark(run, iterations=iterations, warmup=3)
    result.name = "retrieve_with_mmr"
    if profiles:
        result.set_metadata(**aggregate_retrieval_profiles(profiles))
    return result


def benchmark_e2e_simple(chunks, vecs_n, bm, hnsw=None, faiss_index_path=None, iterations=10):
    """Benchmark end-to-end answer generation (simple query)."""
    question = "How do I track time?"

    fn_to_call = _fake_answer_once if os.environ.get("BENCHMARK_FAKE_REMOTE") == "1" else answer_once

    def run():
        try:
            fn_to_call(
                question,
                chunks,
                vecs_n,
                bm,
                top_k=12,
                pack_top=6,
                threshold=0.30,
                hnsw=hnsw,
                faiss_index_path=faiss_index_path,
            )
        except Exception as e:
            print(f"Warning: E2E benchmark failed: {e}")

    result = run_benchmark(run, iterations=iterations, warmup=2)
    result.name = "e2e_simple_query"
    return result


def benchmark_e2e_complex(chunks, vecs_n, bm, hnsw=None, faiss_index_path=None, iterations=5):
    """Benchmark end-to-end answer generation (complex query)."""
    question = "What are the differences between the pricing plans and which features are included in each tier?"

    fn_to_call = _fake_answer_once if os.environ.get("BENCHMARK_FAKE_REMOTE") == "1" else answer_once

    def run():
        try:
            fn_to_call(
                question,
                chunks,
                vecs_n,
                bm,
                top_k=12,
                pack_top=6,
                threshold=0.30,
                hnsw=hnsw,
                faiss_index_path=faiss_index_path,
            )
        except Exception as e:
            print(f"Warning: E2E complex benchmark failed: {e}")

    result = run_benchmark(run, iterations=iterations, warmup=1)
    result.name = "e2e_complex_query"
    return result


def benchmark_chunking(md_path, iterations=5):
    """Benchmark chunking performance."""

    def run():
        build_chunks(md_path)

    result = run_benchmark(run, iterations=iterations, warmup=1)
    result.name = "chunking"
    return result
