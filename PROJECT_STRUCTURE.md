# Clockify RAG CLI - Project Structure

## Directory Layout

```
/Users/15x/Downloads/KBDOC/
│
├── 📄 clockify_rag.py              ← Main application (11 KB)
│   └── Three commands: chunk, embed, ask
│
├── 📚 Documentation/
│   ├── INSTALLATION_SUMMARY.txt     ← Start here! (Setup guide)
│   ├── QUICKSTART.md                ← Fast onboarding (10 min read)
│   ├── README_RAG.md                ← Comprehensive guide (30 min read)
│   ├── TEST_GUIDE.md                ← Testing & validation
│   ├── FILES_MANIFEST.md            ← File inventory & dependencies
│   └── PROJECT_STRUCTURE.md         ← This file
│
├── ⚙️  Configuration/
│   ├── requirements.txt             ← Python dependencies
│   ├── config_example.py            ← Configuration reference
│   └── setup.sh                     ← Automated setup script
│
├── 🐍 Virtual Environment/
│   └── rag_env/                     ← Python venv with dependencies
│       ├── bin/                     ← Executable scripts
│       │   └── activate             ← Source to activate environment
│       ├── lib/                     ← Python packages (requests, numpy)
│       └── pyvenv.cfg               ← Environment config
│
├── 📖 Knowledge Base (Source)/
│   └── knowledge_full.md            ← Clockify docs (6.9 MB)
│
└── 🔄 Generated Files (After Running)/
    ├── chunks.jsonl                 ← Chunked documentation
    ├── vecs.npy                     ← Embedding vectors
    └── meta.jsonl                   ← Chunk metadata
```

## File Descriptions

### Core Application

#### `clockify_rag.py`
- **Type**: Python script (executable)
- **Size**: ~11 KB
- **Dependencies**: requests, numpy, Python 3.7+
- **Purpose**: Complete RAG CLI implementation
- **Functionality**:
  ```bash
  python3 clockify_rag.py chunk          # Split markdown into chunks
  python3 clockify_rag.py embed          # Generate embeddings
  python3 clockify_rag.py ask "..."      # Query knowledge base
  ```
- **External Services**: Ollama (local instance)

### Documentation Files

| File | Size | Purpose | Audience | Read Time |
|------|------|---------|----------|-----------|
| INSTALLATION_SUMMARY.txt | 6 KB | Overview & quick setup | Everyone | 5 min |
| QUICKSTART.md | 4.7 KB | Fast onboarding guide | New users | 10 min |
| README_RAG.md | 7.5 KB | Complete technical guide | Technical users | 30 min |
| TEST_GUIDE.md | ~12 KB | Testing & validation suite | QA/Testers | 20 min |
| FILES_MANIFEST.md | 8.9 KB | File inventory & workflow | Developers | 15 min |
| config_example.py | 6.6 KB | Configuration reference | Advanced users | 10 min |
| PROJECT_STRUCTURE.md | This file | Directory structure | Everyone | 5 min |

### Configuration & Setup

#### `requirements.txt`
- **Size**: 30 bytes
- **Purpose**: Python package dependencies
- **Contents**:
  ```
  requests==2.32.5
  numpy==2.3.4
  ```
- **Usage**: `pip install -r requirements.txt`

#### `config_example.py`
- **Size**: 6.6 KB
- **Purpose**: Reference guide for all configurable parameters
- **Sections**:
  - Chunking configuration
  - Embedding settings
  - Retrieval parameters
  - LLM configuration
  - System prompts
  - File paths
  - Performance tuning
  - Advanced options
- **Usage**: Reference only; shows how to customize `clockify_rag.py`

#### `setup.sh`
- **Size**: 2.6 KB
- **Type**: Bash script (executable)
- **Purpose**: Automated first-time setup
- **Steps**:
  1. Verify Python installation
  2. Create virtual environment
  3. Install dependencies
  4. Validate knowledge base file
  5. Run chunking (optional)
  6. Run embedding (optional)
- **Usage**: `./setup.sh`
- **Platform**: macOS/Linux (use manual setup on Windows)

### Virtual Environment

#### `rag_env/`
- **Type**: Python virtual environment
- **Size**: 50-100 MB
- **Created by**: `setup.sh` or `python3 -m venv rag_env`
- **Contents**:
  - Python 3 interpreter
  - Package manager (pip)
  - Installed packages:
    - `requests` (HTTP client)
    - `numpy` (numerical computing)
- **Activation**:
  ```bash
  # macOS/Linux
  source rag_env/bin/activate
  
  # Windows
  rag_env\Scripts\activate
  ```
- **Cleanup**: `rm -rf rag_env/` (can be recreated from requirements.txt)

### Knowledge Base Files

#### `knowledge_full.md` (Input)
- **Type**: Markdown file
- **Size**: 6.9 MB
- **Content**: ~150 pages of Clockify documentation
- **Format**: Markdown with H2 headers (`##`) as primary split points
- **Status**: Pre-provided; do not modify
- **Used by**: `python3 clockify_rag.py chunk`

#### `chunks.jsonl` (Generated)
- **Type**: JSON Lines format
- **Format**: One JSON object per line
- **Structure**: `{"id": <int>, "text": <string>}`
- **Size**: Typically 50 KB - 500 KB
- **Example**:
  ```json
  {"id": 0, "text": "## Getting Started\nThis section..."}
  {"id": 1, "text": "## Time Tracking\nTime tracking enables..."}
  ```
- **Generated by**: `python3 clockify_rag.py chunk`
- **Used by**: `python3 clockify_rag.py embed`
- **Regenerable**: Yes (from knowledge_full.md)
- **Cleanup**: Safe to delete; run `chunk` command to regenerate

#### `vecs.npy` (Generated)
- **Type**: NumPy binary array
- **Format**: `.npy` binary format (not human-readable)
- **Shape**: [num_chunks, 768]
  - `num_chunks`: Number of document chunks (typically 500-2000)
  - `768`: Embedding dimension (for nomic-embed-text model)
- **Data Type**: float32
- **Size**: Typically 2-5 MB
- **Generated by**: `python3 clockify_rag.py embed`
- **Used by**: `python3 clockify_rag.py ask` (similarity search)
- **Regenerable**: Yes (from chunks.jsonl)
- **Cleanup**: Safe to delete; run `embed` command to regenerate

#### `meta.jsonl` (Generated)
- **Type**: JSON Lines format
- **Format**: One JSON object per line
- **Structure**: `{"id": <int>, "text": <string>}`
- **Size**: Typically 2-10 MB
- **Purpose**: Metadata parallel to vecs.npy (one entry per vector)
- **Index Alignment**: Line N of meta.jsonl ↔ Row N of vecs.npy
- **Generated by**: `python3 clockify_rag.py embed`
- **Used by**: `python3 clockify_rag.py ask` (chunk retrieval & display)
- **Regenerable**: Yes (from chunks.jsonl)
- **Cleanup**: Safe to delete; run `embed` command to regenerate

## Workflow Diagram

```
Setup Phase (One-Time):
┌─────────────────┐
│ knowledge_full  │  (6.9 MB markdown file)
│      .md        │
└────────┬────────┘
         │
         ├─► [python3 clockify_rag.py chunk]
         │
         ▼
┌─────────────────┐
│  chunks.jsonl   │  (Split by ## sections)
└────────┬────────┘
         │
         ├─► [python3 clockify_rag.py embed]
         │   (Calls Ollama: nomic-embed-text)
         │
    ┌────┴─────┬──────────────────┐
    │           │                  │
    ▼           ▼                  ▼
┌────────┐ ┌────────┐       ┌──────────┐
│ vecs   │ │ meta   │  +    │  Ollama  │
│ .npy   │ │ .jsonl │       │ Instance │
└────────┘ └────────┘       └──────────┘

Usage Phase (Repeatable):
┌──────────────┐
│   User       │
│  Question    │
└────────┬─────┘
         │
         ├─► [python3 clockify_rag.py ask "..."]
         │
    ┌────┴──────┬──────────────────┐
    │            │                  │
    ▼            ▼                  ▼
┌────────┐ ┌──────────┐      ┌──────────┐
│ vecs   │ │ meta     │  +   │  Ollama  │
│ .npy   │ │ .jsonl   │      │          │
└────────┘ └──────────┘      └─────┬────┘
    │            │                  │
    └────┬────────┘                 │
         │ (Cosine Similarity)      │
         │                          │
    ┌────▼──────────────────────────▼──┐
    │  Top 6 Relevant Chunks        QA │
    │  + System Prompt              LLM│
    └────┬───────────────────────────┬──┘
         │                           │
         └────────────┬──────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  Answer with  │
              │  Citations    │
              └───────────────┘
```

## Storage & Memory Requirements

### Disk Space

| Component | Size | Optional | Regenerable |
|-----------|------|----------|---|
| clockify_rag.py | <1 MB | No | Yes (source control) |
| Documentation | ~50 KB | Yes | Yes |
| rag_env/ | 50-100 MB | No | Yes |
| knowledge_full.md | 6.9 MB | No | No (source data) |
| chunks.jsonl | 50-500 KB | Yes | Yes |
| vecs.npy | 2-5 MB | Yes | Yes |
| meta.jsonl | 2-10 MB | Yes | Yes |
| **Total** | **~70-120 MB** | | |

### Runtime Memory

- Chunking: ~100 MB (reading markdown)
- Embedding: ~2-4 GB peak (processing chunks)
- Query: ~500 MB (loading vecs + meta)

## Dependencies Tree

```
clockify_rag.py
├── Python Standard Library
│   ├── argparse (CLI argument parsing)
│   ├── json (JSON I/O)
│   ├── os (file operations)
│   └── requests (HTTP calls)
│
└── Third-Party Libraries
    ├── requests==2.32.5
    │   ├── urllib3
    │   ├── certifi
    │   ├── charset_normalizer
    │   └── idna
    │
    └── numpy==2.3.4

External Services:
├── Ollama (http://10.127.0.192:11434)
│   ├── nomic-embed-text (embedding model)
│   └── qwen2.5:32b (LLM model)
│
└── File System
    ├── Input: knowledge_full.md
    ├── Intermediate: chunks.jsonl
    └── Output: vecs.npy, meta.jsonl
```

## Configuration Parameters

Located in `clockify_rag.py` (lines 6-15):

```python
CHUNK_SIZE = 1600          # Max chunk size (characters)
CHUNK_OVERLAP = 200        # Overlap between sub-chunks
OLLAMA_URL = "http://10.127.0.192:11434"  # Ollama endpoint
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:32b"
```

Advanced configuration options available in `config_example.py`.

## Development & Customization

### To Modify Chunking Strategy
- Edit `chunk_file()` function in `clockify_rag.py`
- Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` constants
- Re-run: `python3 clockify_rag.py chunk`

### To Use Different Models
- Edit embedding model: Change `EMBED_MODEL` constant
- Edit LLM model: Change `CHAT_MODEL` constant
- Ensure models are available in Ollama: `ollama pull <model>`
- Re-run embedding: `python3 clockify_rag.py embed`

### To Adjust Retrieval Behavior
- Edit `ask_question()` function
- Modify `SIMILARITY_THRESHOLD` for stricter/looser matching
- Change `top_indices = ... -6` to retrieve more/fewer chunks
- Edit `MIN_RELEVANT_CHUNKS` threshold

### To Change System Prompt
- Edit `system_message` in `ask_question()` function
- Control how LLM uses retrieved snippets

## Quick Reference Commands

```bash
# Setup (one-time)
./setup.sh
source rag_env/bin/activate
pip install -r requirements.txt

# Build knowledge base (one-time)
python3 clockify_rag.py chunk
python3 clockify_rag.py embed

# Usage (repeatable)
source rag_env/bin/activate
python3 clockify_rag.py ask "Your question here"

# Cleanup (if needed)
rm -rf chunks.jsonl vecs.npy meta.jsonl
python3 clockify_rag.py chunk && python3 clockify_rag.py embed

# Full reset
rm -rf rag_env/ chunks.jsonl vecs.npy meta.jsonl
./setup.sh
```

## File Permissions

```bash
# Executable scripts
chmod +x setup.sh

# Readable/writable data files
chmod 644 clockify_rag.py
chmod 644 requirements.txt
chmod 644 config_example.py
chmod 644 knowledge_full.md
```

---

**Version**: 1.0  
**Last Updated**: 2025-11-05  
**Platform**: macOS/Linux primary, Windows with manual setup  
**Python**: 3.7+  
**Status**: Production-Ready
