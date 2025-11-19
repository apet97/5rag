# Clockify RAG – Production-Ready Retrieval System

**Status**: ✅ Production Ready | **Version**: 6.0 | **Platform**: macOS (M1/Intel), Linux, Docker

A production-grade Retrieval-Augmented Generation (RAG) system for Clockify documentation with automatic LLM fallback, hybrid retrieval, and Apple Silicon optimization.

## Key Features

- ✅ **Flexible Configuration**: Supports local Ollama, VPN endpoints, or custom deployments
- ✅ **Automatic Fallback**: Qwen 2.5 32B → GPT-OSS-20B on connection/timeout/5xx errors
- ✅ **Hybrid Retrieval**: BM25 (keyword) + Dense (semantic) + MMR (diversity)
- ✅ **Apple Silicon**: Optimized for M1/M2/M3 Macs with MPS acceleration
- ✅ **Fully Offline**: All processing local (no external API calls)
- ✅ **Production-Ready**: Thread-safe, well-tested, comprehensive monitoring

## 🚀 Quick Start (90 Seconds)

### Quick Start Path

**⚠️  Configure your endpoint first:**
```bash
# Choose your environment profile:
export RAG_OLLAMA_URL="http://127.0.0.1:11434"       # Local Ollama
# OR
export RAG_OLLAMA_URL="http://10.127.0.192:11434"    # Company VPN
# OR
export RAG_OLLAMA_URL="http://your-host:port"        # Custom endpoint
```

Then run:
```bash
git clone https://github.com/apet97/1rag.git
cd 1rag
make dev      # Setup venv + install deps (2-3 min)
make build    # Build index (5-10 min)
make chat     # Start interactive CLI
```

**Default Configuration** (if RAG_OLLAMA_URL not set):
- LLM endpoint: `http://10.127.0.192:11434` (⚠️  Company VPN only - change for your environment)
- Chat model: `qwen2.5:32b`
- Embed model: `nomic-embed-text:latest`
- Fallback: `gpt-oss:20b` (automatic on primary failure)

### Platform-Specific Installation

**macOS (Apple Silicon M1/M2/M3):**
```bash
# Use conda for best FAISS compatibility
conda create -n rag_env python=3.11
conda activate rag_env
conda install -c conda-forge faiss-cpu=1.8.0
pip install -e '.[dev]'
```

**Linux / macOS Intel:**
```bash
python3 -m venv rag_env
source rag_env/bin/activate
pip install -e '.[dev]'
```

For detailed installation guides, see:
- **Mac M1**: [docs/platform/M1_COMPATIBILITY.md](docs/platform/M1_COMPATIBILITY.md)
- **Linux**: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- **Docker**: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

## 🔧 Optional Configuration

**Only needed if you want to override defaults.** See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for complete reference.

### Option 1: YAML Config File (Recommended for Complex Setups)

Create a custom config file based on `config/default.yaml`:

```bash
# View current effective configuration
ragctl config-show

# Use custom config file
cp config/default.yaml config/my-config.yaml
# Edit config/my-config.yaml with your settings
export RAG_CONFIG_FILE=config/my-config.yaml

# Or specify per-command
ragctl --config-file config/my-config.yaml query "How do I track time?"
```

**Config Precedence** (highest to lowest):
1. Environment variables (`RAG_*` prefix)
2. Custom config file (`--config-file` or `RAG_CONFIG_FILE`)
3. Default config (`config/default.yaml`)
4. Hardcoded defaults in `config.py`

### Option 2: Environment Variables (Quick Override)

```bash
# Example: Use local Ollama instead of company VPN endpoint
export RAG_OLLAMA_URL=http://127.0.0.1:11434

# Example: Adjust retrieval parameters
export RAG_TOP_K=20
export RAG_HYBRID_ALPHA=0.3  # More semantic, less keyword matching

# Example: Disable fallback (fail fast)
export RAG_FALLBACK_ENABLED=false
```

## 🧪 Validation & Health Check

**Configuration Doctor** (validates config, connectivity, models):
```bash
# Quick check (CI-safe, no real network calls)
python3 clockify_support_cli_final.py --selftest

# Full check with real network connectivity (requires VPN)
RAG_REAL_OLLAMA_TESTS=1 python3 clockify_support_cli_final.py --selftest
```

**Test Suite**:
```bash
make test                 # Full test suite with coverage
make smoke                # Offline smoke test (mock client)
make eval-gate            # Retrieval quality thresholds (MRR/NDCG)
```

## 📚 Documentation

**Getting Started:**
- [docs/README.md](docs/README.md) – Documentation index
- [docs/README.md](docs/README.md) – Documentation index

**Core Guides:**
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) – System design and data flow
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) – Complete config reference
- [docs/internals/RUNBOOK.md](docs/internals/RUNBOOK.md) – Operations guide (build, deploy, troubleshoot)

**Development:**
- [docs/internals/TESTING.md](docs/internals/TESTING.md) – Testing guide and best practices
- [docs/internals/API.md](docs/internals/API.md) – HTTP API documentation

**Platform-Specific:**
- [docs/platform/M1_COMPATIBILITY.md](docs/platform/M1_COMPATIBILITY.md) – Apple Silicon guide
- [docs/platform/INSTALL_macOS_ARM64.md](docs/platform/INSTALL_macOS_ARM64.md) – M1 installation details

**Reference:**
- [docs/reference/PROJECT_STRUCTURE.md](docs/reference/PROJECT_STRUCTURE.md) – Codebase structure
- [docs/reference/PROJECT_STRUCTURE.md](docs/reference/PROJECT_STRUCTURE.md) – Codebase structure
- [docs/reference/FAQ_CACHE_USAGE.md](docs/reference/FAQ_CACHE_USAGE.md) – Cache usage FAQ

## 🛡️ Automatic LLM Fallback

**Default Behavior** (enabled out-of-box):
- ✅ Primary succeeds → uses `qwen2.5:32b`
- ⚠️ Primary unavailable → logs warning, falls back to `gpt-oss:20b`
- ❌ Both unavailable → returns error to user

**Triggers:**
- Connection errors (VPN disconnected, server down)
- Timeout errors (slow network, overloaded server)
- 5xx server errors (503, 502, 500)

**Customization:** See [docs/CONFIGURATION.md](docs/CONFIGURATION.md)

## Platform Compatibility

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux** | ✅ Full Support | Recommended for production |
| **macOS Intel** | ✅ Full Support | Uses IVFFlat FAISS index |
| **macOS Apple Silicon (M1/M2/M3)** | ✅ **Full Support** | See [M1_COMPATIBILITY.md](docs/platform/M1_COMPATIBILITY.md) |
| **Windows** | ⚠️ WSL2 Recommended | Native support via WSL2 |
| **Docker** | ✅ Multi-Arch | linux/amd64, linux/arm64 |

## Usage Examples

### Interactive Chat
```bash
# Start chat REPL
make chat

# Or with debug mode
python3 clockify_support_cli_final.py chat --debug

# With custom retrieval params
python3 clockify_support_cli_final.py chat --topk 15 --pack 8 --threshold 0.25
```

### Single Query
```bash
# Basic query
python3 clockify_support_cli_final.py ask "How do I track time in Clockify?"

# With reranking and JSON output
python3 clockify_support_cli_final.py ask "How do I set up SSO?" --rerank --json
```

### Build Index
```bash
# Build from knowledge base
python3 clockify_support_cli_final.py build knowledge_full.md

# Or via make
make build
```

## Statistics

| Metric | Value |
|--------|-------|
| **Version** | 6.0 |
| **Python version** | 3.11+ (3.9 minimum) |
| **Platform support** | Linux, macOS (Intel + M1), Windows (WSL2), Docker |
| **Knowledge base size** | 6.9 MB (~155K lines) |
| **Embedding dimension** | 384 (all-MiniLM-L6-v2) |
| **Default chunks** | ~380-400 chunks @ 1600 chars each |
| **Build time (M1)** | ~30 seconds (local embeddings) |
| **Query latency** | ~6-11 seconds (LLM dominates) |
| **M1 performance gain** | 30-70% faster than Intel |

## Requirements

**Core Dependencies:**
- Python 3.11+ (3.9 minimum)
- numpy, requests, sentence-transformers, torch, rank-bm25
- Optional: faiss-cpu (recommended for performance)

**External Services:**
- Ollama-compatible endpoint (default: `http://10.127.0.192:11434`)
- Models: `qwen2.5:32b`, `nomic-embed-text`, `gpt-oss:20b` (fallback)

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for complete deployment requirements.

## Production Deployment

**Thread-Safe Deployment** (v5.1+, **RECOMMENDED**):
```bash
# Multi-threaded with gunicorn
gunicorn -w 4 --threads 4 app:app

# Or with uvicorn
uvicorn app:app --workers 4
```

**Monitoring:**
- Fallback events logged with clear warnings
- Query cache hit rates tracked
- Latency metrics (retrieval, LLM, total)
- Retrieval quality metrics (MRR, NDCG, Precision)

See [docs/internals/OPERATIONS.md](docs/internals/OPERATIONS.md) for operational guidance.

## Support & Troubleshooting

**Common Issues:**

**FAISS import error on M1:**
```bash
conda install -c conda-forge faiss-cpu=1.8.0
```

**Slow performance:**
```bash
# Check if running native ARM (not Rosetta)
python3 -c "import platform; print(platform.machine())"
# Expected on M1: arm64
```

**Configuration issues:**
```bash
# Run configuration doctor
python3 clockify_support_cli_final.py --selftest
```

For more help, see:
- [docs/internals/RUNBOOK.md](docs/internals/RUNBOOK.md) – Operations and troubleshooting
- [docs/platform/M1_COMPATIBILITY.md](docs/platform/M1_COMPATIBILITY.md) – M1-specific issues
- [docs/reference/FAQ_CACHE_USAGE.md](docs/reference/FAQ_CACHE_USAGE.md) – Caching FAQ

---

**Version**: 6.0 (Zero-Config Release)
**Date**: 2025-11-18
**Status**: 🚀 **PRODUCTION-READY**
**Platform**: Linux | macOS (Intel + Apple Silicon) | Windows (WSL2) | Docker
