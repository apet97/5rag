#!/usr/bin/env bash
#
# Install FAISS on Apple Silicon (M1/M2/M3) Macs
#
# FAISS pip wheels are not available for macOS ARM64, so we must use conda.
# This script automates the installation and verification.
#
# Usage:
#   ./scripts/install_faiss_m1.sh
#
# Requirements:
#   - macOS on Apple Silicon (arm64)
#   - Conda installed (Miniconda or Anaconda)
#   - Python 3.11 or 3.12

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}ℹ${NC}  $1"
}

echo_success() {
    echo -e "${GREEN}✅${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}⚠️${NC}  $1"
}

echo_error() {
    echo -e "${RED}❌${NC} $1"
}

# Check if we're on macOS ARM64
check_platform() {
    if [[ "$(uname -s)" != "Darwin" ]]; then
        echo_error "This script is only for macOS. Detected: $(uname -s)"
        exit 1
    fi

    if [[ "$(uname -m)" != "arm64" ]]; then
        echo_error "This script is for Apple Silicon (M1/M2/M3) only. Detected: $(uname -m)"
        echo_info "For Intel Macs, use: pip install faiss-cpu"
        exit 1
    fi

    echo_success "Platform check: macOS ARM64"
}

# Check if conda is installed
check_conda() {
    if ! command -v conda &> /dev/null; then
        echo_error "Conda not found. Please install Miniconda or Anaconda first:"
        echo ""
        echo "  Download Miniconda for macOS ARM64:"
        echo "  https://docs.conda.io/en/latest/miniconda.html"
        echo ""
        echo "  Or use Homebrew:"
        echo "  brew install --cask miniconda"
        exit 1
    fi

    echo_success "Conda found: $(conda --version)"
}

# Check if we're in a conda environment
check_conda_env() {
    if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" == "base" ]]; then
        echo_warning "Not in a conda environment (or in 'base')"
        echo_info "It's recommended to use a dedicated environment:"
        echo ""
        echo "  conda create -n rag_env python=3.11"
        echo "  conda activate rag_env"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo_success "Using conda environment: $CONDA_DEFAULT_ENV"
    fi
}

# Check Python version
check_python_version() {
    python_version=$(python --version 2>&1 | awk '{print $2}')
    python_major=$(echo "$python_version" | cut -d. -f1)
    python_minor=$(echo "$python_version" | cut -d. -f2)

    if [[ "$python_major" -lt 3 ]] || [[ "$python_major" -eq 3 && "$python_minor" -lt 11 ]]; then
        echo_error "Python $python_version detected. Requires Python ≥3.11"
        exit 1
    fi

    echo_success "Python version: $python_version"
}

# Check if FAISS is already installed
check_existing_faiss() {
    if python -c "import faiss" 2>/dev/null; then
        existing_version=$(python -c "import faiss; print(faiss.__version__)" 2>/dev/null || echo "unknown")
        echo_warning "FAISS $existing_version is already installed"
        read -p "Reinstall? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo_info "Skipping installation"
            return 1
        fi
        echo_info "Uninstalling existing FAISS..."
        conda remove -y faiss-cpu || pip uninstall -y faiss-cpu || true
    fi
    return 0
}

# Install FAISS via conda
install_faiss() {
    echo_info "Installing FAISS 1.8.0 from conda-forge..."
    echo_info "This may take 2-3 minutes..."

    if conda install -y -c conda-forge faiss-cpu=1.8.0; then
        echo_success "FAISS installed successfully"
    else
        echo_error "FAISS installation failed"
        echo_info "Try manual installation:"
        echo "  conda install -c conda-forge faiss-cpu=1.8.0"
        exit 1
    fi
}

# Verify FAISS installation
verify_faiss() {
    echo_info "Verifying FAISS installation..."

    if ! python -c "import faiss" 2>/dev/null; then
        echo_error "FAISS import failed"
        exit 1
    fi

    faiss_version=$(python -c "import faiss; print(faiss.__version__)")
    echo_success "FAISS $faiss_version imported successfully"

    # Test basic functionality
    echo_info "Running FAISS smoke test..."
    if python -c "
import numpy as np
import faiss

# Create small test index
d = 64
vectors = np.random.rand(10, d).astype('float32')
faiss.normalize_L2(vectors)

# Test IndexFlatIP (used by RAG system)
index = faiss.IndexFlatIP(d)
index.add(vectors)

# Test search
query = np.random.rand(1, d).astype('float32')
faiss.normalize_L2(query)
distances, indices = index.search(query, 5)

print(f'Search successful: found {len(indices[0])} results')
assert len(indices[0]) == 5, 'Expected 5 results'
" 2>/dev/null; then
        echo_success "FAISS smoke test passed"
    else
        echo_error "FAISS smoke test failed"
        exit 1
    fi
}

# Main installation flow
main() {
    echo ""
    echo "========================================"
    echo "  FAISS Installation for Apple Silicon"
    echo "========================================"
    echo ""

    check_platform
    check_conda
    check_conda_env
    check_python_version
    echo ""

    if check_existing_faiss; then
        install_faiss
    fi

    echo ""
    verify_faiss
    echo ""

    echo_success "Installation complete!"
    echo ""
    echo_info "Next steps:"
    echo "  1. Build the RAG index: ragctl ingest"
    echo "  2. Verify FAISS is used: ragctl doctor"
    echo "  3. Run queries: ragctl query 'How do I track time?'"
    echo ""
}

main
