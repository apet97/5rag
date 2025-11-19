"""Modern Typer-based CLI for Clockify RAG system.

Provides commands:
- ragctl doctor: System diagnostics and configuration check
- ragctl ingest: Build index from knowledge base
- ragctl query: Single query (non-interactive)
- ragctl chat: Interactive REPL
- ragctl eval: Run RAGAS evaluation
"""

import json
import logging
import os
import platform
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import config
from .answer import answer_once, answer_to_json
from .cli import ensure_index_ready, chat_repl
from .indexing import build
from .utils import check_ollama_connectivity
from .api_client import validate_models

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="Clockify RAG Command-Line Interface")


# ============================================================================
# Doctor Command: System Diagnostics
# ============================================================================


def get_device_info() -> dict:
    """Detect and return device information."""
    try:
        import torch

        device = "cpu"
        reason = "default"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            reason = "Metal Performance Shaders (Apple Silicon)"
        elif torch.cuda.is_available():
            device = "cuda"
            reason = f"CUDA ({torch.cuda.get_device_name(0)})"

        return {
            "device": device,
            "reason": reason,
            "torch_version": torch.__version__,
            "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        }
    except ImportError:
        return {
            "device": "unknown",
            "reason": "torch not installed",
            "torch_version": "N/A",
            "mps_available": False,
        }
    except Exception as e:
        return {
            "device": "error",
            "reason": str(e),
            "torch_version": "N/A",
            "mps_available": False,
        }


def get_dependency_info() -> dict:
    """Check for key dependencies."""
    deps = {}

    packages = [
        "numpy",
        "torch",
        "sentence_transformers",
        "faiss",
        "hnswlib",
        "rank_bm25",
        "fastapi",
        "uvicorn",
        "requests",
        "typer",
    ]

    for pkg in packages:
        try:
            mod = __import__(pkg)
            deps[pkg] = {
                "installed": True,
                "version": getattr(mod, "__version__", "unknown"),
            }
        except ImportError:
            deps[pkg] = {
                "installed": False,
                "version": "N/A",
            }

    return deps


def get_index_info() -> dict:
    """Check index files and their status."""
    info = {}
    required_files = [
        config.FILES["chunks"],
        config.FILES["emb"],
        config.FILES["meta"],
        config.FILES["bm25"],
        config.FILES["index_meta"],
    ]

    for key, fname in config.FILES.items():
        exists = os.path.exists(fname)
        size = os.path.getsize(fname) if exists else 0
        info[key] = {
            "file": fname,
            "exists": exists,
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2) if size > 0 else 0,
        }

    all_required = all(os.path.exists(fname) for fname in required_files)
    return {
        "files": info,
        "index_ready": all_required,
    }


@app.command()
def doctor(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Detailed output"),
    json_output: bool = typer.Option(False, "--json", help="JSON output for scripting"),
) -> None:
    """Run diagnostics on system and configuration.

    Checks:
    - Python version and platform
    - Device detection (CPU/MPS/CUDA)
    - Key dependencies
    - Index files and status
    - Configuration validation
    - Ollama connectivity
    """
    if json_output:
        # JSON mode for scripting
        output = {
            "system": {
                "platform": platform.system(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
                "python_executable": sys.executable,
            },
            "device": get_device_info(),
            "dependencies": get_dependency_info(),
            "index": get_index_info(),
            "config": {
                "provider": config.RAG_PROVIDER,
                "ollama_url": config.RAG_OLLAMA_URL,
                "gen_model": config.RAG_CHAT_MODEL,
                "emb_model": config.RAG_EMBED_MODEL,
                "chunk_size": config.CHUNK_CHARS,
                "top_k": config.DEFAULT_TOP_K,
                "pack_top": config.DEFAULT_PACK_TOP,
                "context_budget": config.get_context_budget(),
                "context_window": config.get_context_window(),
            },
        }

        # Check Ollama connectivity
        try:
            normalized = check_ollama_connectivity(config.RAG_OLLAMA_URL, timeout=3)
            output["ollama"] = {"connected": True, "endpoint": normalized, "error": None}
        except Exception as e:
            output["ollama"] = {"connected": False, "endpoint": config.RAG_OLLAMA_URL, "error": str(e)}

        console.print(json.dumps(output, indent=2))
        return

    # Rich console output
    console.print(Panel("🔍 Clockify RAG System Diagnostics", style="bold blue"))
    console.print()

    # System Info
    table = Table(title="System Information", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Platform", platform.system())
    table.add_row("Architecture", platform.machine())
    table.add_row("Python Version", platform.python_version())
    table.add_row("Python Executable", sys.executable)
    console.print(table)
    console.print()

    # Device Info
    device_info = get_device_info()
    device_emoji = "🚀" if device_info["device"] != "cpu" else "📱"
    device_table = Table(title=f"{device_emoji} Device Detection", show_header=False)
    device_table.add_column("Key", style="cyan")
    device_table.add_column("Value", style="white")
    device_table.add_row("Device", device_info["device"].upper())
    device_table.add_row("Reason", device_info["reason"])
    device_table.add_row("PyTorch Version", device_info["torch_version"])
    device_table.add_row("MPS Available", "✅ Yes" if device_info["mps_available"] else "❌ No")
    console.print(device_table)
    console.print()

    # Dependencies
    deps = get_dependency_info()
    deps_table = Table(title="📦 Key Dependencies", show_header=True)
    deps_table.add_column("Package", style="cyan")
    deps_table.add_column("Status", style="white")
    deps_table.add_column("Version", style="green")
    for pkg, info in sorted(deps.items()):
        status = "✅" if info["installed"] else "❌"
        version = info["version"] if info["installed"] else "—"
        deps_table.add_row(pkg, status, version)
    console.print(deps_table)
    console.print()

    # Index Status
    index_info = get_index_info()
    index_ready = index_info["index_ready"]
    index_emoji = "✅" if index_ready else "❌"
    console.print(f"{index_emoji} Index Status: {'READY' if index_ready else 'NOT READY (run: ragctl ingest)'}")
    console.print()

    # Ollama Connectivity & Model Validation
    try:
        model_result = validate_models(log_warnings=False)
        
        if model_result["server_reachable"]:
            console.print(f"✅ Ollama: Connected to {config.RAG_OLLAMA_URL}")
            
            # Check required models
            chat_available = model_result["chat_model"]["available"]
            embed_available = model_result["embed_model"]["available"]
            fallback_available = model_result["fallback_model"]["available"]
            
            console.print(f"  📦 Available models: {len(model_result['models_available'])}")
            console.print(f"     • Chat model ({config.RAG_CHAT_MODEL}): {'✅' if chat_available else '❌ MISSING'}")
            console.print(f"     • Embed model ({config.RAG_EMBED_MODEL}): {'✅' if embed_available else '❌ MISSING'}")
            
            if config.RAG_FALLBACK_ENABLED:
                console.print(f"     • Fallback model ({config.RAG_FALLBACK_MODEL}): {'✅' if fallback_available else '⚠️  MISSING'}")
            
            if not model_result["all_required_available"]:
                console.print("\n  ⚠️  Some required models are missing! Run 'ollama pull <model>' to fix.")
        else:
            console.print(f"❌ Ollama: Connection failed to {config.RAG_OLLAMA_URL}")
            console.print("  💡 Check if Ollama is running and accessible.")
            
    except Exception as e:
        console.print(f"❌ Ollama: Validation error - {e}")
    console.print()

    # Configuration Summary
    provider = config.RAG_PROVIDER
    provider_emoji = "🧠" if provider == "gpt-oss" else "🦙"
    config_table = Table(title=f"{provider_emoji} Configuration ({provider.upper()})", show_header=False)
    config_table.add_column("Key", style="cyan")
    config_table.add_column("Value", style="white")
    config_table.add_row("Provider", provider.upper())
    config_table.add_row("Ollama URL", config.RAG_OLLAMA_URL)

    # Provider-specific model display
    if provider == "gpt-oss":
        config_table.add_row("Generation Model", f"{config.RAG_GPT_OSS_MODEL} (GPT-OSS-20B)")
        config_table.add_row("Temperature", f"{config.RAG_GPT_OSS_TEMPERATURE}")
        config_table.add_row("Top-P", f"{config.RAG_GPT_OSS_TOP_P}")
        config_table.add_row("Context Window", f"{config.RAG_GPT_OSS_CTX_WINDOW:,} tokens (128k)")
        config_table.add_row("Context Budget", f"{config.RAG_GPT_OSS_CTX_BUDGET:,} tokens")
    else:
        config_table.add_row("Generation Model", config.RAG_CHAT_MODEL)
        config_table.add_row("Context Window", f"{config.DEFAULT_NUM_CTX:,} tokens")
        config_table.add_row("Context Budget", f"{config.CTX_TOKEN_BUDGET:,} tokens")

    config_table.add_row("Embedding Model", config.RAG_EMBED_MODEL)
    config_table.add_row("Chunk Size", str(config.CHUNK_CHARS))
    config_table.add_row("Top-K Retrieval", str(config.DEFAULT_TOP_K))
    config_table.add_row("Pack Top", str(config.DEFAULT_PACK_TOP))
    console.print(config_table)
    console.print()

    if verbose:
        # Detailed index file listing
        console.print("[bold]📁 Index Files (Detailed):[/bold]")
        for key, file_info in index_info["files"].items():
            status = "✅" if file_info["exists"] else "❌"
            size_str = f"{file_info['size_mb']} MB" if file_info["size_mb"] > 0 else "—"
            console.print(f"  {status} {key:20} {file_info['file']:30} {size_str}")
        console.print()

    console.print("✨ Diagnostics complete!")


# ============================================================================
# Config Command: Show Configuration
# ============================================================================


@app.command()
def config_show(
    config_file: Optional[str] = typer.Option(None, "--config-file", help="Config file path"),
    format: str = typer.Option("yaml", "--format", help="Output format: yaml or json"),
) -> None:
    """Show effective configuration with all overrides applied.

    Displays the merged configuration from:
    1. Default config (config/default.yaml)
    2. Custom config file (if specified)
    3. Environment variable overrides (RAG_* prefix)

    Example:
        ragctl config-show
        ragctl config-show --config-file config/production.yaml
        ragctl config-show --format json
    """
    from .config_loader import export_effective_config

    try:
        effective_config = export_effective_config(config_file=config_file)

        if format == "json":
            console.print(json.dumps(effective_config, indent=2))
        else:
            # YAML output
            try:
                import yaml
                console.print(yaml.dump(effective_config, default_flow_style=False, sort_keys=False))
            except ImportError:
                console.print("⚠️  PyYAML not installed. Showing JSON format instead:")
                console.print("    Install with: pip install PyYAML")
                console.print()
                console.print(json.dumps(effective_config, indent=2))

    except Exception as e:
        console.print(f"❌ Failed to load config: {e}")
        logger.error(f"Config error: {e}", exc_info=True)
        raise typer.Exit(1)


# ============================================================================
# Ingest Command: Build Index
# ============================================================================


@app.command()
def ingest(
    input: Optional[str] = typer.Option(
        None, "--input", "-i", help="Input markdown file or directory (default: knowledge_full.md)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for index (default: current directory)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force rebuild even if index exists"),
) -> None:
    """Build or rebuild the index from knowledge base.

    Performs:
    1. Chunking: Split markdown into semantic chunks
    2. Embedding: Generate vector embeddings
    3. Indexing: Build FAISS/HNSW indexes and BM25 index
    4. Validation: Verify all artifacts

    Example:
        ragctl ingest --input ./docs --output ./var/index
    """
    input_file = input or "knowledge_full.md"
    output_dir = output or "."

    if not os.path.exists(input_file):
        console.print(f"❌ Input file not found: {input_file}")
        raise typer.Exit(1)

    console.print(f"📥 Ingesting: {input_file}")
    console.print(f"📤 Output directory: {output_dir}")

    try:
        build(input_file, retries=2)
        console.print("✅ Index built successfully!")

        # Verify
        idx_info = get_index_info()
        if idx_info["index_ready"]:
            console.print("✅ All artifacts verified")
        else:
            console.print("⚠️ Some artifacts missing")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"❌ Build failed: {e}")
        logger.error(f"Build error: {e}", exc_info=True)
        raise typer.Exit(1)


# ============================================================================
# Query Command: Single Query
# ============================================================================


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(15, "--top-k", help="Number of chunks to retrieve"),
    pack_top: int = typer.Option(8, "--pack-top", help="Number of chunks to include in context"),
    threshold: float = typer.Option(0.25, "--threshold", help="Minimum similarity threshold"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
) -> None:
    """Ask a single question and get an answer.

    Example:
        ragctl query "How do I track time in Clockify?"
    """
    console.print(f"❓ Question: {question}")

    try:
        chunks, vecs_n, bm, hnsw = ensure_index_ready(retries=2)

        result = answer_once(
            question,
            chunks,
            vecs_n,
            bm,
            top_k=top_k,
            pack_top=pack_top,
            threshold=threshold,
            hnsw=hnsw,
        )

        answer = result["answer"]
        meta = result.get("metadata", {})
        timing = result.get("timing")
        routing = result.get("routing")
        selected_chunks = result.get("selected_chunks", [])
        refused = result.get("refused", False)

        if json_output:
            used_tokens = meta.get("used_tokens") or len(selected_chunks)
            payload = answer_to_json(
                answer,
                selected_chunks,
                used_tokens,
                top_k,
                pack_top,
                result.get("confidence"),
                metadata=meta,
                routing=routing,
                timing=timing,
                refused=refused,
            )
            chunk_ids = result.get("selected_chunk_ids")
            sources = []

            if chunk_ids:
                for identifier in chunk_ids:
                    sources.append(identifier if isinstance(identifier, str) else str(identifier))
            elif selected_chunks:
                for identifier in selected_chunks:
                    if isinstance(identifier, dict) and identifier.get("id"):
                        sources.append(str(identifier["id"]))
                    elif isinstance(identifier, int) and 0 <= identifier < len(chunks):
                        sources.append(str(chunks[identifier]["id"]))
                    else:
                        sources.append(str(identifier))

            payload["question"] = question
            payload["sources"] = sources
            payload["num_sources"] = len(sources)
            console.print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            console.print()
            console.print(answer)
            if debug:
                console.print()
                debug_line = (
                    f"[dim]confidence={result.get('confidence', 'n/a')} "
                    f"refused={refused} sources={len(selected_chunks)} "
                    f"total_ms={(timing or {}).get('total_ms', 'n/a')}[/dim]"
                )
                console.print(debug_line)
                if selected_chunks:
                    console.print(f"[dim]Sources: {selected_chunks[:3]}[/dim]")

    except Exception as e:
        console.print(f"❌ Error: {e}")
        logger.error(f"Query error: {e}", exc_info=True)
        raise typer.Exit(1)


# ============================================================================
# Chat Command: Interactive REPL
# ============================================================================


@app.command()
def chat(
    top_k: int = typer.Option(15, "--top-k", help="Number of chunks to retrieve"),
    pack_top: int = typer.Option(8, "--pack-top", help="Number of chunks to include in context"),
    threshold: float = typer.Option(0.25, "--threshold", help="Minimum similarity threshold"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
) -> None:
    """Start interactive chat REPL.

    Commands:
        :exit    - Quit
        :debug   - Toggle debug output
        :config  - Show configuration
        :help    - Show help

    Example:
        ragctl chat
        > What is Clockify?
        > How do I set up SSO?
        > :exit
    """
    console.print(Panel("💬 Clockify RAG Chat", style="bold green"))
    console.print("Type ':exit' to quit, ':debug' to toggle debug, ':help' for help")
    console.print()

    try:
        chat_repl(
            top_k=top_k,
            pack_top=pack_top,
            threshold=threshold,
            debug=debug,
        )
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"❌ Error: {e}")
        logger.error(f"Chat error: {e}", exc_info=True)
        raise typer.Exit(1)


# ============================================================================
# Eval Command: RAGAS Evaluation
# ============================================================================


@app.command()
def eval(
    questions_file: str = typer.Option("data/eval_questions.jsonl", "--questions", "-q", help="Questions JSONL file"),
    output_dir: str = typer.Option("var/reports", "--output", "-o", help="Output directory for reports"),
    sample_size: Optional[int] = typer.Option(None, "--sample", "-s", help="Sample size (default: all)"),
    metrics: Optional[str] = typer.Option(
        "faithfulness,answer_relevancy",
        "--metrics",
        "-m",
        help="Comma-separated metrics to compute",
    ),
) -> None:
    """Run RAGAS evaluation on a set of questions.

    Computes metrics:
    - Faithfulness: Is the answer faithful to the context?
    - Answer Relevancy: Is the answer relevant to the question?
    - Context Precision: Is the context relevant to the question?
    - Context Recall: Does the context contain all relevant information?

    Example:
        ragctl eval --questions data/questions.jsonl --metrics faithfulness,answer_relevancy
    """
    console.print(f"📊 Running evaluation on {questions_file}")

    try:
        import ragas
        console.print(f"✅ RAGAS {ragas.__version__} loaded")
    except ImportError:
        # RAGAS is optional for internal metrics (MRR/NDCG) but required for LLM-based metrics
        # For now, we focus on the retrieval metrics implemented in evaluation.py
        pass

    try:
        from .evaluation import evaluate_dataset, SUCCESS_THRESHOLDS

        results = evaluate_dataset(
            dataset_path=questions_file,
            verbose=True,  # Always verbose for CLI to show progress
            llm_report=False, # TODO: Add flag for this
            llm_output=os.path.join(output_dir, "llm_answers.jsonl")
        )

        # Display Results
        console.print()
        console.print(Panel("📈 Evaluation Results", style="bold green"))
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Threshold", justify="right")
        table.add_column("Status", justify="center")

        metrics_map = {
            "mrr_at_10": "MRR@10",
            "precision_at_5": "Precision@5",
            "ndcg_at_10": "NDCG@10"
        }

        success = True
        for key, label in metrics_map.items():
            score = results.get(key, 0.0)
            threshold = SUCCESS_THRESHOLDS.get(key, 0.0)
            passed = score >= threshold
            if not passed:
                success = False
            status = "✅" if passed else "❌"
            table.add_row(label, f"{score:.3f}", f"{threshold:.3f}", status)

        console.print(table)
        console.print()

        if success:
            console.print("✨ All thresholds passed!")
        else:
            console.print("⚠️  Some thresholds were not met.")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"❌ Evaluation failed: {e}")
        logger.error(f"Eval error: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def benchmark(
    quick: bool = typer.Option(False, "--quick", help="Quick benchmark (fewer iterations)"),
    embedding: bool = typer.Option(False, "--embedding", help="Only embedding benchmarks"),
    retrieval: bool = typer.Option(False, "--retrieval", help="Only retrieval benchmarks"),
    e2e: bool = typer.Option(False, "--e2e", help="Only end-to-end benchmarks"),
    output: str = typer.Option("benchmark_results.json", "--output", help="Output JSON file"),
):
    """Run performance benchmarks."""
    import time
    import json
    from .config import EMB_BACKEND
    from .benchmarking import (
        benchmark_embedding_single,
        benchmark_embedding_batch,
        benchmark_embedding_large_batch,
        benchmark_retrieval_hybrid,
        benchmark_retrieval_with_mmr,
        benchmark_e2e_simple,
        benchmark_e2e_complex,
        benchmark_chunking,
    )

    # Adjust iterations for quick mode
    iter_multiplier = 0.5 if quick else 1.0

    console.print(Panel("⏱️  Clockify RAG Benchmark Suite", style="bold blue"))
    console.print(f"Embedding backend: {EMB_BACKEND}")
    console.print(f"Mode: {'Quick' if quick else 'Full'}")
    console.print()

    # Load index
    with console.status("[bold green]Loading index..."):
        chunks, vecs_n, bm, hnsw = ensure_index_ready(retries=2)
        # We need faiss_index_path if hnsw is None but file exists, 
        # but ensure_index_ready returns loaded objects.
        # For benchmarking, we might want to pass the path if we want to test loading?
        # But the benchmark functions take loaded objects mostly.
        # Let's check if we can get the path.
        from .config import FILES
        faiss_index_path = FILES["faiss_index"] if os.path.exists(FILES["faiss_index"]) else None

    console.print(f"✅ Loaded {len(chunks)} chunks")
    console.print()

    results = []

    # Embedding benchmarks
    if not retrieval and not e2e:
        console.print("[bold cyan]--- Embedding Benchmarks ---[/bold cyan]")
        if not quick:
            with console.status("Benchmarking single embedding..."):
                res = benchmark_embedding_single(chunks, iterations=int(10 * iter_multiplier))
                results.append(res)
                console.print(f"✅ {res.name}: {res.summary()['latency_ms']['mean']:.2f}ms")

        with console.status("Benchmarking batch embedding (10)..."):
            res = benchmark_embedding_batch(chunks, iterations=int(5 * iter_multiplier))
            results.append(res)
            console.print(f"✅ {res.name}: {res.summary()['latency_ms']['mean']:.2f}ms")

        if not quick:
            with console.status("Benchmarking large batch embedding (100)..."):
                res = benchmark_embedding_large_batch(chunks, iterations=int(3 * iter_multiplier))
                results.append(res)
                console.print(f"✅ {res.name}: {res.summary()['latency_ms']['mean']:.2f}ms")
        console.print()

    # Retrieval benchmarks
    if not embedding and not e2e:
        console.print("[bold cyan]--- Retrieval Benchmarks ---[/bold cyan]")
        with console.status("Benchmarking hybrid retrieval..."):
            res = benchmark_retrieval_hybrid(chunks, vecs_n, bm, hnsw=hnsw, faiss_index_path=faiss_index_path, iterations=int(20 * iter_multiplier))
            results.append(res)
            console.print(f"✅ {res.name}: {res.summary()['latency_ms']['mean']:.2f}ms")

        with console.status("Benchmarking retrieval + MMR..."):
            res = benchmark_retrieval_with_mmr(chunks, vecs_n, bm, hnsw=hnsw, faiss_index_path=faiss_index_path, iterations=int(20 * iter_multiplier))
            results.append(res)
            console.print(f"✅ {res.name}: {res.summary()['latency_ms']['mean']:.2f}ms")
        console.print()

    # End-to-end benchmarks
    if not embedding and not retrieval:
        console.print("[bold cyan]--- End-to-End Benchmarks ---[/bold cyan]")
        with console.status("Benchmarking simple query..."):
            res = benchmark_e2e_simple(chunks, vecs_n, bm, hnsw=hnsw, faiss_index_path=faiss_index_path, iterations=int(10 * iter_multiplier))
            results.append(res)
            console.print(f"✅ {res.name}: {res.summary()['latency_ms']['mean']:.2f}ms")

        if not quick:
            with console.status("Benchmarking complex query..."):
                res = benchmark_e2e_complex(chunks, vecs_n, bm, hnsw=hnsw, faiss_index_path=faiss_index_path, iterations=int(5 * iter_multiplier))
                results.append(res)
                console.print(f"✅ {res.name}: {res.summary()['latency_ms']['mean']:.2f}ms")
        console.print()

    # Chunking benchmark
    if not embedding and not retrieval and not e2e and not quick:
        if os.path.exists("knowledge_full.md"):
            console.print("[bold cyan]--- Chunking Benchmark ---[/bold cyan]")
            with console.status("Benchmarking chunking..."):
                res = benchmark_chunking("knowledge_full.md", iterations=5)
                results.append(res)
                console.print(f"✅ {res.name}: {res.summary()['latency_ms']['mean']:.2f}ms")
            console.print()

    # Summary
    console.print()
    console.print(Panel("📊 Benchmark Results", style="bold green"))
    
    summaries = [r.summary() for r in results]
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Throughput (ops/s)", justify="right")
    table.add_column("Memory (MB)", justify="right")

    for s in summaries:
        latency = f"{s['latency_ms']['mean']:.2f} ± {s['latency_ms']['stdev']:.2f}"
        throughput = f"{s['throughput']['ops_per_sec']:.2f}"
        memory = f"{s['memory_mb']['peak']:.2f}"
        table.add_row(s['name'], latency, throughput, memory)

    console.print(table)

    # Save to JSON
    output_data = {
        "timestamp": time.time(),
        "backend": EMB_BACKEND,
        "quick_mode": quick,
        "results": summaries,
    }

    with open(output, "w") as f:
        json.dump(output_data, f, indent=2)

    console.print()
    console.print(f"✅ Results saved to {output}")


# ============================================================================
# Entry Point
# ============================================================================


if __name__ == "__main__":
    app()
