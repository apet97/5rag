#!/usr/bin/env python3
"""
Benchmark suite for Clockify RAG CLI.

DEPRECATED: Use `ragctl benchmark` instead.
"""

import argparse
import json
import os
import sys
import time
import warnings

from clockify_rag.config import EMB_BACKEND
from clockify_rag.indexing import load_index
from clockify_rag.benchmarking import (
    benchmark_embedding_single,
    benchmark_embedding_batch,
    benchmark_embedding_large_batch,
    benchmark_retrieval_hybrid,
    benchmark_retrieval_with_mmr,
    benchmark_e2e_simple,
    benchmark_e2e_complex,
    benchmark_chunking,
)

def main():
    warnings.warn(
        "This script is deprecated and will be removed in v7.0. Use 'ragctl benchmark' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    print("⚠️  WARNING: This script is deprecated. Please use 'ragctl benchmark' instead.\n")

    parser = argparse.ArgumentParser(description="Benchmark Clockify RAG CLI")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (fewer iterations)")
    parser.add_argument("--embedding", action="store_true", help="Only embedding benchmarks")
    parser.add_argument("--retrieval", action="store_true", help="Only retrieval benchmarks")
    parser.add_argument("--e2e", action="store_true", help="Only end-to-end benchmarks")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON file")
    args = parser.parse_args()

    # Adjust iterations for quick mode
    iter_multiplier = 0.5 if args.quick else 1.0

    print("=" * 70)
    print("CLOCKIFY RAG BENCHMARK SUITE")
    print("=" * 70)
    print(f"Embedding backend: {EMB_BACKEND}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print()

    # Load index
    print("[1/2] Loading index...")
    result = load_index()
    if result is None:
        print("❌ Failed to load index. Run 'make build' first.")
        sys.exit(1)

    # Handle new dict return format from load_index()
    if isinstance(result, dict):
        chunks = result["chunks"]
        vecs_n = result["vecs_n"]
        bm = result["bm"]
        hnsw = result.get("faiss_index")  # May be None
        faiss_index_path = result.get("faiss_index_path")
    else:
        # Legacy tuple format (backward compatibility)
        chunks, vecs_n, bm, hnsw = result
        faiss_index_path = None
    print(f"✅ Loaded {len(chunks)} chunks")
    print()

    # Run benchmarks
    print("[2/2] Running benchmarks...")
    print()

    results = []

    # Embedding benchmarks
    if not args.retrieval and not args.e2e:
        print("--- Embedding Benchmarks ---")
        if not args.quick:
            results.append(benchmark_embedding_single(chunks, iterations=int(10 * iter_multiplier)))
            print(f"✅ {results[-1].name}: {results[-1].summary()['latency_ms']['mean']:.2f}ms")

        results.append(benchmark_embedding_batch(chunks, iterations=int(5 * iter_multiplier)))
        print(f"✅ {results[-1].name}: {results[-1].summary()['latency_ms']['mean']:.2f}ms")

        if not args.quick:
            results.append(benchmark_embedding_large_batch(chunks, iterations=int(3 * iter_multiplier)))
            print(f"✅ {results[-1].name}: {results[-1].summary()['latency_ms']['mean']:.2f}ms")
        print()

    # Retrieval benchmarks
    if not args.embedding and not args.e2e:
        print("--- Retrieval Benchmarks ---")
        results.append(benchmark_retrieval_hybrid(chunks, vecs_n, bm, hnsw=hnsw, faiss_index_path=faiss_index_path, iterations=int(20 * iter_multiplier)))
        print(f"✅ {results[-1].name}: {results[-1].summary()['latency_ms']['mean']:.2f}ms")

        results.append(benchmark_retrieval_with_mmr(chunks, vecs_n, bm, hnsw=hnsw, faiss_index_path=faiss_index_path, iterations=int(20 * iter_multiplier)))
        print(f"✅ {results[-1].name}: {results[-1].summary()['latency_ms']['mean']:.2f}ms")
        print()

    # End-to-end benchmarks
    if not args.embedding and not args.retrieval:
        print("--- End-to-End Benchmarks ---")
        results.append(benchmark_e2e_simple(chunks, vecs_n, bm, hnsw=hnsw, faiss_index_path=faiss_index_path, iterations=int(10 * iter_multiplier)))
        print(f"✅ {results[-1].name}: {results[-1].summary()['latency_ms']['mean']:.2f}ms")

        if not args.quick:
            results.append(benchmark_e2e_complex(chunks, vecs_n, bm, hnsw=hnsw, faiss_index_path=faiss_index_path, iterations=int(5 * iter_multiplier)))
            print(f"✅ {results[-1].name}: {results[-1].summary()['latency_ms']['mean']:.2f}ms")
        print()

    # Chunking benchmark
    if not args.embedding and not args.retrieval and not args.e2e and not args.quick:
        if os.path.exists("knowledge_full.md"):
            print("--- Chunking Benchmark ---")
            results.append(benchmark_chunking("knowledge_full.md", iterations=5))
            print(f"✅ {results[-1].name}: {results[-1].summary()['latency_ms']['mean']:.2f}ms")
            print()

    # Summary
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    summaries = [r.summary() for r in results]
    for s in summaries:
        print(f"\n{s['name']}:")
        print(f"  Latency:    {s['latency_ms']['mean']:.2f}ms ± {s['latency_ms']['stdev']:.2f}ms")
        print(f"  Throughput: {s['throughput']['ops_per_sec']:.2f} ops/sec")
        print(f"  Memory:     {s['memory_mb']['peak']:.2f} MB peak")
        metadata = s.get("metadata")
        if metadata:
            print("  Profiling:")
            for key, value in metadata.items():
                print(f"    {key}: {value}")

    # Save to JSON
    output_data = {
        "timestamp": time.time(),
        "backend": EMB_BACKEND,
        "quick_mode": args.quick,
        "results": summaries,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print()
    print(f"✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
