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
- Fully offline: uses Ollama-compatible endpoint (default: http://10.127.0.192:11434)
- Stateless REPL: each turn forgets prior context
- Hybrid retrieval: BM25 (sparse) + dense (semantic) + MMR diversification
- Closed-book: refuses low-confidence answers
- Artifact versioning: auto-rebuild if KB drifts
- Automatic fallback: Falls back to gpt-oss:20b when primary model unavailable
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
    warmup_on_startup,
)
from clockify_rag.api_client import get_llm_client


def run_selftest() -> bool:
    """
    Comprehensive production self-test and configuration doctor.

    Checks:
    1. Configuration validation (URLs, timeouts, retrieval params)
    2. Model endpoint connectivity (optional, gated by RAG_REAL_OLLAMA_TESTS=1)
    3. Required models availability
    4. Index artifacts presence
    5. Retrieval path smoke test (in strict mode)

    Environment variables:
    - RAG_REAL_OLLAMA_TESTS=1: Enable real network connectivity checks (requires VPN)
    - SELFTEST_STRICT=1: Enable strict mode (fails on warnings)
    - RAG_LLM_CLIENT=mock: Force mock client for offline testing
    """
    ok = True
    strict_mode = os.environ.get("SELFTEST_STRICT", "").lower() not in {"", "0", "false", "no"}
    real_ollama_tests = os.environ.get("RAG_REAL_OLLAMA_TESTS", "").lower() in {"1", "true", "yes"}
    client_mode = (os.environ.get("RAG_LLM_CLIENT") or "").strip().lower()

    print("=" * 70)
    print("RAG CONFIGURATION DOCTOR & SELF-TEST")
    print("=" * 70)
    print()

    # === SECTION 1: Configuration Validation ===
    print("📋 CONFIGURATION VALIDATION")
    print("-" * 70)

    from clockify_rag.config import validate_config

    config_result = validate_config()

    # Print config snapshot
    snapshot = config_result["config_snapshot"]
    print(f"  LLM Endpoint:     {snapshot['llm_endpoint']}")
    print(f"  Provider:         {snapshot['provider']}")
    print(f"  Chat Model:       {snapshot['chat_model']}")
    print(f"  Embed Model:      {snapshot['embed_model']}")
    print(f"  Fallback Enabled: {snapshot['fallback_enabled']}")
    if snapshot["fallback_enabled"]:
        print(f"  Fallback Provider: {snapshot['fallback_provider']}")
        print(f"  Fallback Model:    {snapshot['fallback_model']}")
    print(f"  Chat Timeout:     {snapshot['chat_timeout']}s")
    print(f"  Embed Timeout:    {snapshot['embed_timeout']}s")
    print(f"  Top-K:            {snapshot['top_k']}")
    print(f"  Pack-Top:         {snapshot['pack_top']}")
    print(f"  Threshold:        {snapshot['threshold']}")
    print(f"  Context Budget:   {snapshot['ctx_budget']} tokens")
    print(f"  Context Window:   {snapshot['ctx_window']} tokens")
    print()

    # Print warnings and errors
    if config_result["warnings"]:
        print("  ⚠️  WARNINGS:")
        for warning in config_result["warnings"]:
            print(f"    • {warning}")
        print()

    if config_result["errors"]:
        print("  ❌ ERRORS:")
        for error in config_result["errors"]:
            print(f"    • {error}")
        print()
        ok = False

    if config_result["valid"]:
        print("  ✅ Configuration validation passed")
    else:
        print("  ❌ Configuration validation failed")
        ok = False
    print()

    # === SECTION 2: Model Endpoint & Availability ===
    print("🔌 MODEL ENDPOINT & AVAILABILITY")
    print("-" * 70)

    if not client_mode:
        os.environ["RAG_LLM_CLIENT"] = "mock"
        client_mode = "mock"

    if client_mode == "mock":
        print("  ℹ️  Using mock LLM client (offline mode)")
        print("  💡 Set RAG_REAL_OLLAMA_TESTS=1 to test real endpoint connectivity")
    elif not real_ollama_tests:
        print("  ℹ️  Skipping real network connectivity checks (CI-safe mode)")
        print("  💡 Set RAG_REAL_OLLAMA_TESTS=1 to enable real endpoint tests")
    else:
        print(f"  🌐 Testing connectivity to {config.RAG_OLLAMA_URL}...")

        # Check server connectivity and model availability
        from clockify_rag.api_client import validate_models

        try:
            model_result = validate_models(log_warnings=False)

            if model_result["server_reachable"]:
                print(f"  ✅ Server reachable at {config.RAG_OLLAMA_URL}")

                # Check required models
                chat_available = model_result["chat_model"]["available"]
                embed_available = model_result["embed_model"]["available"]
                fallback_available = model_result["fallback_model"]["available"]

                print(f"  📦 Available models: {len(model_result['models_available'])}")
                print(f"     • Chat model ({config.RAG_CHAT_MODEL}): {'✅' if chat_available else '❌ MISSING'}")
                print(f"     • Embed model ({config.RAG_EMBED_MODEL}): {'✅' if embed_available else '❌ MISSING'}")

                if config.RAG_FALLBACK_ENABLED:
                    print(
                        f"     • Fallback model ({config.RAG_FALLBACK_MODEL}): {'✅' if fallback_available else '⚠️  MISSING'}"
                    )

                if not model_result["all_required_available"]:
                    print()
                    print("  ⚠️  Some required models are missing!")
                    print("  💡 Pull missing models:")
                    if not chat_available:
                        print(f"     ollama pull {config.RAG_CHAT_MODEL}")
                    if not embed_available:
                        print(f"     ollama pull {config.RAG_EMBED_MODEL}")
                    if config.RAG_FALLBACK_ENABLED and not fallback_available:
                        print(f"     ollama pull {config.RAG_FALLBACK_MODEL}")

                    if strict_mode:
                        ok = False
            else:
                print(f"  ❌ Server unreachable at {config.RAG_OLLAMA_URL}")
                print("  💡 Check if Ollama is running: curl {}/api/version".format(config.RAG_OLLAMA_URL))
                if strict_mode:
                    ok = False
        except Exception as e:
            print(f"  ⚠️  Model endpoint check failed: {e}")
            if strict_mode:
                ok = False
    print()

    # === SECTION 3: Index Artifacts ===
    print("📂 INDEX ARTIFACTS")
    print("-" * 70)

    index_ok = True
    index_candidates = []

    for name in ("CHUNKS_PATH", "BM25_PATH", "FAISS_INDEX_PATH"):
        path = getattr(config, name, None)
        if path:
            index_candidates.append(path)

    # Fallback: legacy default filenames
    if not index_candidates:
        index_candidates = ["chunks.jsonl", "bm25.json", "faiss.index"]

    missing_files = [path for path in index_candidates if not os.path.exists(path)]

    if missing_files:
        print("  ⚠️  Missing index files:")
        for path in missing_files:
            print(f"     • {path}")
        print()
        print("  💡 Build the knowledge base:")
        print("     python3 clockify_support_cli_final.py build knowledge_full.md")
        if strict_mode:
            index_ok = False
            ok = False
    else:
        print("  ✅ All index files present")
        for path in index_candidates:
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"     • {path} ({size_mb:.2f} MB)")
    print()

    # === SECTION 4: Retrieval Path Smoke Test (Strict Mode Only) ===
    if strict_mode and index_ok:
        print("🔍 RETRIEVAL PATH SMOKE TEST")
        print("-" * 70)
        try:
            from clockify_rag.cli import ensure_index_ready

            ensure_index_ready(retries=0)
            print("  ✅ Retrieval path OK")
        except Exception as e:
            print(f"  ❌ Retrieval check failed: {e}")
            ok = False
        print()

    # === FINAL SUMMARY ===
    print("=" * 70)
    if ok:
        print("✅ SELF-TEST PASSED")
        if not real_ollama_tests:
            print("💡 Run with RAG_REAL_OLLAMA_TESTS=1 for full connectivity tests")
    else:
        print("❌ SELF-TEST FAILED")
    print("=" * 70)

    return ok


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
