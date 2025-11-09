# macOS ARM64 Installation Guide

Complete step-by-step installation guide for Apple Silicon (M1/M2/M3) Macs.

## Why This Guide?

Installing RAG systems on Apple Silicon can be tricky due to:
- ARM64 vs x86-64 architecture differences
- PyTorch MPS (Metal Performance Shaders) setup
- FAISS binary compatibility
- Rosetta 2 emulation pitfalls

This guide ensures native ARM64 installation with GPU acceleration.

## Prerequisites

- **macOS**: 12.3+ (for MPS support)
- **Disk**: 5GB free (index + models)
- **Memory**: 8GB+ recommended
- **Network**: For initial setup and Ollama model downloads

## Quick Installation (5 minutes)

### 1. Automated Bootstrap

The easiest way: run the bootstrap script.

```bash
# Clone repository
git clone https://github.com/apet97/1rag.git
cd 1rag

# Run bootstrap
bash scripts/bootstrap_macos_arm64.sh
```

This script:
- ✅ Detects ARM64 architecture
- ✅ Installs Homebrew (if needed)
- ✅ Installs pyenv (Python version manager)
- ✅ Installs Python 3.11 native ARM64
- ✅ Installs uv (fast dependency manager)
- ✅ Installs all dependencies
- ✅ Verifies MPS support
- ✅ Shows next steps

**Done! Skip to [Building the Index](#building-the-index)**

---

## Manual Installation (if bootstrap fails)

### Step 1: Verify You're Running Native ARM64

```bash
# Check architecture
uname -m
# Output should be: arm64

# Verify NOT running under Rosetta
arch
# Output should be: arm64, NOT i386
```

If you see `x86_64` or `i386`:

```bash
# You're in Rosetta emulation. Fix:
# 1. Reinstall Homebrew for ARM64
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Reinstall Python
brew uninstall python@3.11
brew install python@3.11

# 3. Verify
which python3  # Should be /opt/homebrew/bin/python3
python3 -c "import platform; print(platform.machine())"  # Should show arm64
```

### Step 2: Install Homebrew

```bash
# Check if installed
brew --version

# If not, install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add to PATH
export PATH="/opt/homebrew/bin:$PATH"
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zprofile
```

### Step 3: Install pyenv (Python Version Manager)

```bash
# Install pyenv
brew install pyenv

# Add to shell profile (~/.zprofile for zsh, ~/.bashrc for bash)
echo 'eval "$(pyenv init --path)"' >> ~/.zprofile
echo 'eval "$(pyenv init -)"' >> ~/.zprofile

# Reload shell
source ~/.zprofile
```

### Step 4: Install Python 3.11

```bash
# List available versions
pyenv install --list | grep "^[[:space:]]*3.11"

# Install latest 3.11
pyenv install 3.11.14  # (or latest version)

# Set as default (optional)
pyenv global 3.11.14

# Verify
python3 --version
# Output: Python 3.11.14
```

### Step 5: Install uv (Fast Dependency Manager)

```bash
# Install uv via Homebrew
brew install uv

# Verify
uv --version
# Output: uv 0.x.x
```

### Step 6: Clone Repository and Install Dependencies

```bash
# Clone
git clone https://github.com/apet97/1rag.git
cd 1rag

# Set local Python version (if using pyenv)
pyenv local 3.11.14

# Install project dependencies
uv sync

# Install development dependencies (optional)
uv sync --extra dev

# Verify installation
python3 -c "import clockify_rag; print('✅ clockify_rag imported successfully')"
```

### Step 7: Verify Installation

```bash
# Run doctor command
python3 -m clockify_rag.cli_modern doctor

# Expected output should show:
# - Platform: Darwin arm64
# - Device: mps (Metal Performance Shaders)
# - MPS Available: ✅ Yes
# - Index Status: ❌ NOT READY (run: ragctl ingest)
```

---

## Setting Up Ollama (Required for LLM)

### Option A: Homebrew (Easiest)

```bash
# Install Ollama
brew install ollama

# Start service
ollama serve

# In another terminal, download models
ollama pull nomic-embed-text      # ~275 MB (embedding)
ollama pull qwen2.5:32b          # ~20 GB (generation, takes time)

# (Or use smaller model for testing)
ollama pull mistral               # ~4 GB (faster, lower quality)
```

### Option B: Download Binary

```bash
# Go to https://ollama.ai/download/mac
# Download and install macOS app
# Run from Applications

# In terminal, verify
ollama serve

# Download models
ollama pull nomic-embed-text
ollama pull qwen2.5:32b
```

### Option C: Docker

```bash
# Install Docker Desktop from https://docker.com/products/docker-desktop

# Run Ollama in Docker
docker run -d -p 11434:11434 -v ollama:/root/.ollama ollama/ollama

# Pull models
docker exec $(docker ps -q) ollama pull nomic-embed-text
docker exec $(docker ps -q) ollama pull qwen2.5:32b
```

### Verify Ollama Installation

```bash
# Check version
ollama --version

# Check models
ollama list
# Output should include:
# - nomic-embed-text
# - qwen2.5:32b (or alternative)

# Test API
curl http://127.0.0.1:11434/api/version
# Output: {"version":"x.x.x"}
```

---

## Building the Index

### Quick Build (with test KB)

```bash
# Build index from default knowledge_full.md
python3 -m clockify_rag.cli_modern ingest

# Expected: ✅ Index built successfully!
```

### Verify Index

```bash
# List index files
ls -lh chunks.jsonl vecs_n.npy meta.jsonl bm25.json

# Run diagnostics
python3 -m clockify_rag.cli_modern doctor
# Expected: Index Status: ✅ READY
```

---

## Testing the Installation

### 1. Interactive Chat

```bash
# Start interactive session
python3 -m clockify_rag.cli_modern chat

# Try asking
> What is Clockify?
> How do I track time offline?
> :exit
```

### 2. Single Query

```bash
# Ask a question
python3 -m clockify_rag.cli_modern query "How do I export data?"

# With JSON output
python3 -m clockify_rag.cli_modern query "How do I export?" --json | jq .confidence
```

### 3. API Server

```bash
# Start server
python3 -m uvicorn clockify_rag.api:app --host 127.0.0.1 --port 8000

# In another terminal
curl http://localhost:8000/v1/health
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Clockify?"}'
```

---

## Troubleshooting

### Problem: "Wrong architecture (x86_64, need arm64)"

**Cause**: Running under Rosetta emulation (x86 translation layer)

**Fix**:
```bash
# Verify
python3 -c "import platform; print(platform.machine())"

# Should show: arm64
# If shows: x86_64, you're in Rosetta!

# Solution: Reinstall Python natively
brew uninstall python@3.11
brew install python@3.11

# Verify installation location
which python3
# Should be: /opt/homebrew/bin/python3 (NOT /usr/local/bin)
```

### Problem: "MPS not available" or "torch using CPU only"

**Cause**: PyTorch not compiled with MPS, or macOS too old

**Fix**:
```bash
# Check macOS version (need 12.3+)
sw_vers

# If too old, upgrade macOS

# Reinstall PyTorch with MPS support
pip uninstall torch
pip install torch==2.4.2  # Should auto-detect MPS

# Verify
python3 -c "import torch; print(torch.backends.mps.is_available())"
# Output: True
```

### Problem: FAISS "Segmentation fault"

**Cause**: FAISS IVF index incompatibility on M1

**Fix**:
```bash
# Application should auto-fallback to BM25
# But to explicitly disable FAISS:
USE_ANN=none python3 -m clockify_rag.cli_modern ingest

# Or use HNSW instead
USE_ANN=hnsw python3 -m clockify_rag.cli_modern ingest
```

### Problem: Slow embeddings (>5 seconds per query)

**Cause**: Running under Rosetta or no MPS acceleration

**Fix**:
```bash
# Verify architecture
python3 -c "import platform; print(f'Machine: {platform.machine()}')"

# Verify MPS enabled
python3 << 'EOF'
import torch
if torch.backends.mps.is_available():
    print("✅ MPS enabled")
else:
    print("❌ MPS disabled - reinstall PyTorch")
EOF

# Check if using CPU
python3 << 'EOF'
import torch
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/multilingual-e5-base")
sentences = ["Hello world"]
embeddings = model.encode(sentences)
print(f"Device used: {embeddings.device if hasattr(embeddings, 'device') else 'CPU'}")
EOF
```

### Problem: "Ollama connection refused"

**Cause**: Ollama not running or wrong URL

**Fix**:
```bash
# Check if running
ps aux | grep ollama

# If not, start it
ollama serve

# Check URL
curl http://127.0.0.1:11434/api/version

# If using Docker
docker ps | grep ollama
docker logs <container_id>

# If using Docker Desktop on macOS, use:
export OLLAMA_URL=http://host.docker.internal:11434
```

### Problem: Out of memory during build

**Cause**: Large knowledge base or low system memory

**Fix**:
```bash
# Reduce batch size
EMB_BATCH_SIZE=8 python3 -m clockify_rag.cli_modern ingest

# Use fewer workers
EMB_MAX_WORKERS=2 python3 -m clockify_rag.cli_modern ingest

# Try smaller embedding model
EMB_MODEL=all-MiniLM-L6-v2 python3 -m clockify_rag.cli_modern ingest
```

---

## Performance Optimization

### Enable MPS Acceleration

Ensure MPS is enabled:

```bash
python3 << 'EOF'
import torch
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"✅ MPS available: {device}")
else:
    print("❌ MPS not available")
EOF
```

### Benchmark Your Setup

```bash
# Time index build
time python3 -m clockify_rag.cli_modern ingest

# Time query
time python3 -m clockify_rag.cli_modern query "What is Clockify?"

# Profile with debug output
LOG_LEVEL=debug python3 -m clockify_rag.cli_modern query "q" --debug
```

### Expected Performance (M1 Pro, 16GB)

| Operation | Time |
|-----------|------|
| Index Build (384 chunks) | ~30 seconds |
| First Query (warmup) | ~2 seconds |
| Subsequent Queries | ~1 second |
| Embedding 100 texts | ~60 seconds |

---

## Next Steps

1. **[Build the Index](../docs/README.md#building-the-index)** from your knowledge base
2. **[Start chatting](../docs/README.md#commands)**
3. **[Configure performance](../docs/CONFIG.md)** for your use case
4. **[Deploy to production](../docs/OPERATIONS.md)** via Docker or API
5. **[Read architecture](../docs/ARCHITECTURE.md)** to understand the system

---

## Additional Resources

- **PyTorch MPS**: https://pytorch.org/docs/stable/notes/mps.html
- **Homebrew**: https://brew.sh
- **Ollama**: https://ollama.ai
- **Apple Silicon**: https://developer.apple.com/

---

**Stuck?** Check [README.md troubleshooting section](../docs/README.md#troubleshooting) or open an issue on GitHub.
