#!/bin/bash
# Clockify RAG CLI - Automated Setup Script
# One-command setup: ./setup.sh
#
# This script:
# 1. Checks system requirements (Python 3.9+, curl)
# 2. Creates Python virtual environment
# 3. Installs dependencies
# 4. Checks Ollama connectivity
# 5. Optionally installs pre-commit hooks

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
echo "======================================================================"
echo "  Clockify RAG CLI - Automated Setup"
echo "======================================================================"
echo ""

# Step 1: Check Python version
info "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    error "python3 not found. Please install Python 3.9 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    error "Python 3.9+ required. Found: $PYTHON_VERSION"
    exit 1
fi

success "Python $PYTHON_VERSION detected"

# Step 2: Check curl
info "Checking curl availability..."
if ! command -v curl &> /dev/null; then
    warning "curl not found. Ollama check will be skipped."
    SKIP_OLLAMA_CHECK=1
else
    success "curl found"
fi

# Step 3: Create virtual environment
info "Creating virtual environment (rag_env)..."
if [ -d "rag_env" ]; then
    warning "Virtual environment already exists. Skipping creation."
else
    python3 -m venv rag_env
    success "Virtual environment created"
fi

# Step 4: Activate virtual environment
info "Activating virtual environment..."
source rag_env/bin/activate || {
    error "Failed to activate virtual environment"
    exit 1
}
success "Virtual environment activated"

# Step 5: Upgrade pip
info "Upgrading pip..."
python -m pip install --upgrade pip -q
success "pip upgraded"

# Step 5.5: Check for M1 and recommend conda for FAISS compatibility
MACHINE_ARCH=$(uname -m 2>/dev/null || echo "unknown")
SYSTEM_OS=$(uname -s 2>/dev/null || echo "unknown")

if [ "$SYSTEM_OS" = "Darwin" ] && [ "$MACHINE_ARCH" = "arm64" ]; then
    echo ""
    warning "Apple Silicon (M1/M2/M3) detected!"
    echo ""
    echo "  For best FAISS compatibility on M1 Macs, we recommend using conda instead of pip."
    echo ""
    echo "  Why conda?"
    echo "    • FAISS ARM64 builds available via conda-forge"
    echo "    • PyTorch with MPS acceleration"
    echo "    • Better package compatibility on Apple Silicon"
    echo ""
    echo "  Quick conda setup:"
    echo "    1. Install Miniforge: brew install miniforge"
    echo "    2. Create environment: conda create -n rag_env python=3.11"
    echo "    3. Activate: conda activate rag_env"
    echo "    4. Install: see requirements-m1.txt for one-line command"
    echo ""
    echo "  See M1_COMPATIBILITY.md for detailed instructions."
    echo ""
    read -p "Continue with pip installation anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Setup cancelled. Please use conda for M1 installation."
        info "See requirements-m1.txt or M1_COMPATIBILITY.md for instructions."
        exit 0
    fi
    warning "Continuing with pip... FAISS may fail to install on M1."
    echo ""
fi

# Step 6: Install dependencies
info "Installing dependencies (this may take a few minutes)..."
if [ -f "requirements.lock" ]; then
    info "Installing from requirements.lock (pinned versions)..."
    pip install -r requirements.lock -q
else
    info "Installing from requirements.txt..."
    pip install -r requirements.txt -q
fi
success "Dependencies installed"

# Step 7: Check Ollama connectivity
if [ -z "$SKIP_OLLAMA_CHECK" ]; then
    info "Checking Ollama connectivity..."
    OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
    if curl -sf "$OLLAMA_URL/api/version" > /dev/null 2>&1; then
        OLLAMA_VERSION=$(curl -s "$OLLAMA_URL/api/version" | grep -o '"version":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
        success "Ollama is running (version: $OLLAMA_VERSION)"

        # Check for required models
        info "Checking for required Ollama models..."

        # Check embedding model
        if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
            success "Embedding model (nomic-embed-text) found"
        else
            warning "Embedding model (nomic-embed-text) not found"
            echo "  To install: ollama pull nomic-embed-text"
        fi

        # Check generation model
        if ollama list 2>/dev/null | grep -q "qwen2.5:32b"; then
            success "Generation model (qwen2.5:32b) found"
        else
            warning "Generation model (qwen2.5:32b) not found"
            echo "  To install: ollama pull qwen2.5:32b"
            echo "  Or use smaller model: ollama pull qwen2.5:7b"
        fi
    else
        warning "Ollama is not running or not reachable at $OLLAMA_URL"
        echo "  Please start Ollama: ollama serve"
        echo "  Or install from: https://ollama.com"
    fi
fi

# Step 8: Optional pre-commit hooks
echo ""
read -p "Install pre-commit git hooks? (recommended for contributors) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    info "Installing pre-commit hooks..."
    pre-commit install
    success "Pre-commit hooks installed"
else
    info "Skipping pre-commit hooks"
fi

# Step 9: Summary
echo ""
echo "======================================================================"
success "Setup complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     $ source rag_env/bin/activate"
echo ""
echo "  2. Build knowledge base (first time only):"
echo "     $ make build"
echo ""
echo "  3. Start interactive chat:"
echo "     $ make chat"
echo ""
echo "  4. Run tests:"
echo "     $ make test"
echo ""
echo "For more commands, run:"
echo "  $ make help"
echo ""
echo "Documentation:"
echo "  - Quick start: SUPPORT_CLI_QUICKSTART.md"
echo "  - Full guide:  CLOCKIFY_SUPPORT_CLI_README.md"
echo ""
