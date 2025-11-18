# 🚀 Quickstart: Mac M1 Pro from Scratch

**Target**: Get 2rag running on a fresh Mac M1 Pro in 5-10 minutes
**Status**: ✅ Production Ready | **Version**: 6.0 (Zero-Config Release)

---

## Prerequisites

Before you begin, ensure you have:

1. **Xcode Command Line Tools** (for compiling Python packages)
   ```bash
   xcode-select --install
   ```

2. **Python 3.11+** (comes with macOS, or install via Homebrew)
   ```bash
   # Check version
   python3 --version  # Should be 3.11 or higher

   # If needed, install via Homebrew
   brew install python@3.11
   ```

3. **Company VPN Access** (for default LLM endpoint at `http://10.127.0.192:11434`)
   - Or run local Ollama (see "Local Ollama Setup" below)

---

## ✅ Zero-Config Installation (4 Commands)

**No environment variables needed!** Works out-of-box.

```bash
# 1. Clone the repository
git clone https://github.com/apet97/2rag.git
cd 2rag

# 2. Create virtual environment and install dependencies
make dev
# This creates rag_env/, activates it, and installs all packages
# Takes 2-3 minutes on M1 Pro

# 3. Build the search index
make build
# Builds chunks, embeddings, FAISS/BM25 indexes
# Uses local embeddings (all-MiniLM-L6-v2) - no LLM needed
# Takes 5-10 minutes for ~150 pages of docs

# 4. Start chatting!
make chat
# Interactive CLI with command history and debug mode
```

**That's it!** The system automatically uses:
- LLM endpoint: `http://10.127.0.192:11434` (company VPN)
- Chat model: `qwen2.5:32b` (32k context, 0.0 temperature)
- Embed model (for retrieval): `nomic-embed-text:latest` (768-dim)
- Indexing embed model: `all-MiniLM-L6-v2` (384-dim, local, no API calls)

---

## 🎯 Usage Examples

### Interactive Chat

```bash
make chat
```

```
Welcome to Clockify RAG CLI v6.0
Type ':help' for commands, ':exit' to quit
Using provider: ollama (qwen2.5:32b)
Connected to: http://10.127.0.192:11434

> How do I track time in Clockify?

Retrieving context...
Generating answer...

You can track time in Clockify using several methods:
1. Timer: Click the "Start Timer" button...
[Citations: chunk_042, chunk_089]

> :debug
Debug mode: ON

> What are the pricing plans?
[Shows retrieved chunks, scores, and ranking]
...
```

### Single Query (Non-Interactive)

```bash
source rag_env/bin/activate
ragctl query "How do I export my time entries?"
```

### API Server

```bash
source rag_env/bin/activate
uvicorn clockify_rag.api:app --host 0.0.0.0 --port 8000

# Test it
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I track time?"}'
```

---

## 🔧 Optional: Use GPT-OSS-20B Instead of Qwen

Want to use OpenAI's reasoning model (128k context vs qwen's 32k)?

```bash
# Set provider before running
export RAG_PROVIDER=gpt-oss
make chat
```

Or permanently in your shell profile:
```bash
echo 'export RAG_PROVIDER=gpt-oss' >> ~/.zshrc
source ~/.zshrc
```

---

## 🏠 Optional: Local Ollama Setup

If you don't have VPN access or prefer to run Ollama locally:

### 1. Install Ollama
```bash
brew install ollama
```

### 2. Start Ollama Server
```bash
ollama serve
# Runs on http://127.0.0.1:11434
```

### 3. Pull Models
```bash
ollama pull nomic-embed-text:latest  # For embeddings (768-dim)
ollama pull qwen2.5:32b               # For answer generation (32k context)

# Optional: GPT-OSS-20B (128k context)
ollama pull gpt-oss-20b
```

### 4. Override Default Endpoint
```bash
export RAG_OLLAMA_URL=http://127.0.0.1:11434
make chat
```

---

## 🧪 Testing & Validation

### System Health Check
```bash
source rag_env/bin/activate
ragctl doctor --verbose
```

Shows:
- Python version and platform (arm64 detection)
- Installed packages and versions
- Index status (chunks, FAISS, BM25)
- LLM connectivity (if endpoint reachable)
- Configuration summary

### Offline Smoke Test
```bash
make smoke
```
Uses mock LLM client (no network) to verify:
- Index loading
- Retrieval pipeline
- Answer generation logic
- Citation extraction

### Full Test Suite
```bash
make test
```
Runs pytest with coverage report.

---

## 📊 System Requirements

| Component | Requirement | Mac M1 Pro |
|-----------|-------------|------------|
| **Python** | 3.11+ | ✅ Native ARM64 |
| **Memory** | 4GB+ | ✅ 16-32GB typical |
| **Disk** | 2GB free | ✅ |
| **FAISS** | CPU version | ✅ via conda or pip |
| **PyTorch** | 2.0+ | ✅ MPS acceleration |
| **Network** | VPN or local | ✅ |

---

## 🛠️ Troubleshooting

### Issue: `make dev` fails with compiler errors

**Solution**: Install Xcode CLT
```bash
xcode-select --install
sudo xcode-select --reset
```

### Issue: FAISS import error

**Solution**: Use conda for FAISS (recommended for M1)
```bash
conda create -n rag_env python=3.11
conda activate rag_env
conda install -c conda-forge faiss-cpu=1.8.0
pip install -e '.[dev]'
```

### Issue: Can't connect to LLM endpoint

**Check 1**: VPN connected?
```bash
curl http://10.127.0.192:11434/api/version
```

**Check 2**: Use local Ollama instead
```bash
export RAG_OLLAMA_URL=http://127.0.0.1:11434
ollama serve  # In separate terminal
```

**Check 3**: Use mock client (offline)
```bash
export RAG_LLM_CLIENT=mock
make chat
```

### Issue: Slow index build

**Expected**: 5-10 min on M1 Pro for ~150 pages
- Chunking: 10-20s
- Local embeddings: 4-8 min (CPU-based, uses all cores)
- FAISS index: 5-10s
- BM25 index: 2-5s

**Speedup**: Use GPU-accelerated embeddings (experimental)
```bash
export EMB_BACKEND=ollama  # Uses Ollama API instead of local
make build  # Faster if Ollama server is on fast hardware
```

### Issue: Import errors after `make dev`

**Solution**: Reinstall in editable mode
```bash
source rag_env/bin/activate
pip install --force-reinstall -e '.[dev]'
```

---

## 📚 Next Steps

- **Production deployment**: See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- **Mac M1 technical details**: See [docs/M1_COMPATIBILITY.md](docs/M1_COMPATIBILITY.md)
- **Configuration reference**: See [docs/CONFIGURATION.md](docs/CONFIGURATION.md)
- **API documentation**: See [docs/API.md](docs/API.md)
- **Testing guide**: See [docs/TESTING.md](docs/TESTING.md)

---

## 🆘 Getting Help

1. **Check existing docs**: `ls docs/`
2. **Run diagnostics**: `ragctl doctor --verbose`
3. **Check logs**: `tail -f logs/combined.log` (if exists)
4. **GitHub Issues**: [github.com/apet97/2rag/issues](https://github.com/apet97/2rag/issues)

---

**Last Updated**: 2025-11-18 | **Platform**: macOS M1 Pro | **Python**: 3.11+
