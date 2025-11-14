#!/usr/bin/env python3
"""
Clockify Internal Support CLI – Stateless RAG with Hybrid Retrieval

HOW TO RUN
==========
  # Build knowledge base (one-time)
  python3 clockify_support_cli_final.py build knowledge_full.md

  # Start interactive REPL
  python3 clockify_support_cli_final.py chat [--debug] [--rerank] [--topk 12] [--pack 6] [--threshold 0.30]

  # Or auto-start REPL with no args
  python3 clockify_support_cli_final.py

DESIGN
======
- Fully offline: uses only http://127.0.0.1:11434 (local Ollama)
- Stateless REPL: each turn forgets prior context
- Hybrid retrieval: BM25 (sparse) + dense (semantic) + MMR diversification
- Closed-book: refuses low-confidence answers
- Artifact versioning: auto-rebuild if KB drifts
- No external APIs or web calls
"""

# Standard library imports
import atexit
import hashlib
import json
import logging
import os
import sys
import threading
import time
from typing import Any

# Third-party imports
import numpy as np
# Package imports
import clockify_rag.config as config
from clockify_rag.caching import get_query_cache, get_rate_limiter, QueryCache, RateLimiter, log_query
from clockify_rag.utils import _release_lock_if_owner, _log_config_summary
from clockify_rag.exceptions import EmbeddingError, LLMError, IndexLoadError, BuildError
from clockify_rag.cli import (
    setup_cli_args,
    configure_logging_and_config,
    handle_build_command,
    handle_ask_command,
    handle_chat_command,
    chat_repl,
    warmup_on_startup
)
from clockify_rag.api_client import get_llm_client


def run_selftest() -> bool:
    """
    Lightweight production self-test.

    Checks:
    1. Model endpoint reachable at RAG_OLLAMA_URL / OLLAMA_URL.
    2. Required models visible (best-effort).
    3. Basic index presence.
    4. Retrieval path does not crash.
    """
    ok = True

    # 1) Check model endpoint
    try:
        client = get_llm_client()
        models = client.list_models()
        model_names = {m.get("name") or m.get("model") for m in models}
        required = {config.RAG_CHAT_MODEL, config.RAG_EMBED_MODEL}
        missing = {m for m in required if m and m not in model_names}
        print(f"[SELFTEST] Model endpoint OK at {config.RAG_OLLAMA_URL}")
        if missing:
            print(f"[SELFTEST] Warning: missing models (not fatal for selftest): {sorted(missing)}")
    except Exception as e:
        print(f"[SELFTEST] ERROR: model endpoint check failed: {e}")
        ok = False

    # 2) Check index artifacts (best-effort; use config paths if defined)
    index_ok = True
    index_candidates = []

    for name in ("CHUNKS_PATH", "BM25_PATH", "FAISS_INDEX_PATH"):
        path = getattr(config, name, None)
        if path:
            index_candidates.append(path)

    # Fallback: legacy default filenames
    if not index_candidates:
        index_candidates = ["chunks.jsonl", "bm25.json", "faiss.index"]

    missing_any = False
    for path in index_candidates:
        if not os.path.exists(path):
            missing_any = True

    if missing_any:
        print("[SELFTEST] WARNING: one or more index files are missing.")
        print("          Run the build command for your KB before using in production.")
        index_ok = False
    else:
        print("[SELFTEST] Index files found.")

    # 3) Try a tiny retrieval if index seems present
    if index_ok and ok:
        try:
            from clockify_rag.retrieval import retrieve
            results = retrieve("healthcheck", top_k=1)
            if not results:
                print("[SELFTEST] WARNING: retrieval returned no results for 'healthcheck'.")
            else:
                print("[SELFTEST] Retrieval path OK.")
        except Exception as e:
            print(f"[SELFTEST] ERROR: retrieval check failed: {e}")
            ok = False

    return ok and index_ok
from clockify_rag.error_handlers import print_system_health

# Re-export config constants and functions for backward compatibility with tests
from clockify_rag.config import (
    LOG_QUERY_INCLUDE_CHUNKS,
    QUERY_LOG_FILE,
)

# Re-export functions used by tests
from clockify_rag.answer import answer_once
from clockify_rag.retrieval import retrieve, coverage_ok
from clockify_rag.indexing import build
from clockify_rag.answer import (
    apply_mmr_diversification,
    apply_reranking,
    pack_snippets,
    generate_llm_answer,
)
from clockify_rag.utils import inject_policy_preamble

# ====== MODULE GLOBALS ======
logger = logging.getLogger(__name__)
QUERY_LOG_DISABLED = False  # Can be set to True via --no-log flag
atexit.register(_release_lock_if_owner)

# Global instances
RATE_LIMITER = get_rate_limiter()
QUERY_CACHE = get_query_cache()


# ====== MAIN ENTRY POINT ======

def main():
    """Main entry point - delegates to CLI module for all functionality."""
    global QUERY_LOG_DISABLED

    try:
        # Parse command line arguments
        args = setup_cli_args()

        # Configure logging and global config
        QUERY_LOG_DISABLED = configure_logging_and_config(args)

        # Handle selftest if requested
        if getattr(args, "selftest", False):
            success = run_selftest()
            sys.exit(0 if success else 1)

        # Auto-start REPL if no command given
        if args.cmd is None:
            chat_repl()
            return

        # Route to appropriate command handler
        if args.cmd == "build":
            handle_build_command(args)
        elif args.cmd == "ask":
            handle_ask_command(args)
        elif args.cmd == "chat":
            handle_chat_command(args)
    
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Rank 29: cProfile profiling support
    # Check for --profile flag early (before parsing to avoid double-parse)
    if "--profile" in sys.argv:
        import cProfile
        import pstats
        import io

        print("=" * 60)
        print("cProfile: Performance profiling enabled")
        print("=" * 60)

        profiler = cProfile.Profile()
        profiler.enable()

        try:
            main()
        finally:
            profiler.disable()

            # Print stats to stdout
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.strip_dirs()
            ps.sort_stats(pstats.SortKey.CUMULATIVE)  # Sort by cumulative time

            print("\n" + "=" * 60)
            print("cProfile: Top 30 functions by cumulative time")
            print("=" * 60)
            ps.print_stats(30)

            print(s.getvalue())

            # Optionally save to file
            profile_file = "clockify_rag_profile.stats"
            profiler.dump_stats(profile_file)
            print(f"\nFull profile saved to: {profile_file}")
            print(f"View with: python -m pstats {profile_file}")
# The run_selftest function is now defined at the top of the file


if __name__ == "__main__":
    main()
