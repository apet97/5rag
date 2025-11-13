#!/usr/bin/env python3
"""End-to-end smoke test for the Clockify RAG stack.

The script loads the local index, runs `answer_once` with the configured
LLM client (mock by default), and prints a compact summary.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from clockify_rag.indexing import load_index
from clockify_rag.answer import answer_once
from clockify_rag import config


def run_smoke_test(question: str, top_k: int, pack_top: int, threshold: float, retries: int) -> int:
    """Execute the smoke test and return an exit code."""
    index = load_index()
    if index is None:
        print("❌ Index artifacts not found. Run `make build` or `ragctl ingest` first.")
        return 1

    if isinstance(index, dict):
        chunks = index["chunks"]
        vecs_n = index["vecs_n"]
        bm = index["bm"]
        hnsw = index.get("faiss_index")
    else:
        chunks, vecs_n, bm, hnsw = index
    result = answer_once(
        question,
        chunks,
        vecs_n,
        bm,
        top_k=top_k,
        pack_top=pack_top,
        threshold=threshold,
        hnsw=hnsw,
        retries=retries,
    )

    answer = result["answer"]
    refused = result.get("refused", False)
    confidence = result.get("confidence")
    selected = result.get("selected_chunks", [])
    metadata = result.get("metadata", {})

    endpoint = config.RAG_OLLAMA_URL
    client_mode = os.environ.get("RAG_LLM_CLIENT", "ollama")

    print("============== SMOKE TEST ==============")
    print(f"Question:      {question}")
    print(f"Ollama URL:    {endpoint}")
    print(f"LLM client:    {client_mode}")
    print(f"Top-k / Pack:  {top_k} / {pack_top}")
    print("----------------------------------------")
    print(f"Answer:        {answer}")
    print(f"Confidence:    {confidence}")
    print(f"Selected IDs:  {selected[:3]}")
    if metadata.get("llm_error"):
        print(f"LLM Error:     {metadata['llm_error']} - {metadata.get('llm_error_msg')}")
    timing = result.get("timing") or {}
    if timing:
        print(
            "Timing (ms):    total={total:.1f} retrieve={ret:.1f} llm={llm:.1f}".format(
                total=timing.get("total_ms", 0.0),
                ret=timing.get("retrieve_ms", 0.0),
                llm=timing.get("llm_ms", 0.0),
            )
        )
    routing = result.get("routing") or {}
    if routing:
        print(
            f"Routing:       action={routing.get('action')} level={routing.get('level')} escalated={routing.get('escalated')}"
        )
    print("========================================")

    if refused or metadata.get("llm_error"):
        print("⚠️  Smoke test detected a refusal or error.")
        return 2

    print("✅ Smoke test succeeded.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight RAG smoke test.")
    parser.add_argument(
        "--question",
        default="How do I track time in Clockify?",
        help="Question to send through the pipeline",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Candidates to retrieve")
    parser.add_argument("--pack-top", type=int, default=4, help="Snippets to pack")
    parser.add_argument("--threshold", type=float, default=0.25, help="Similarity threshold")
    parser.add_argument("--retries", type=int, default=config.DEFAULT_RETRIES, help="LLM retry count")

    args = parser.parse_args(argv)
    return run_smoke_test(args.question, args.top_k, args.pack_top, args.threshold, args.retries)


if __name__ == "__main__":  # pragma: no cover - manual utility
    sys.exit(main())
