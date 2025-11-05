# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Clockify RAG CLI** – A local, offline retrieval-augmented generation (RAG) system that answers questions about Clockify's documentation using a local Ollama instance.

- **Two implementations**: v1.0 (simple, educational) and v2.0 (production-ready, recommended)
- **Fully offline**: No external APIs; uses local Ollama at `http://127.0.0.1:11434` (configurable)
- **Knowledge source**: 6.9 MB Clockify markdown documentation (~150 pages)
- **User interface**: CLI with command-line interface (v1.0) or interactive REPL (v2.0)

## Architecture & Components

### High-Level Pipeline

```
Knowledge Base (knowledge_full.md)
    ↓
Chunking (split by ## headings, max 1600 chars)
    ↓
Embedding (nomic-embed-text model via Ollama)
    ↓
Storage (vecs.npy vectors + meta.jsonl metadata + BM25 index)
    ↓
Query (user question)
    ↓
Retrieval (v1.0: cosine similarity; v2.0: hybrid BM25+dense+MMR)
    ↓
LLM (qwen2.5:32b via Ollama to generate answer)
    ↓
Response (answer with citations or "I don't know based on the MD.")
```

### Core Files

| File | Version | Lines | Purpose |
|------|---------|-------|---------|
| `clockify_rag.py` | v1.0 | ~350 | Simple three-step CLI: chunk → embed → ask |
| `clockify_support_cli.py` | v2.0 | ~1400+ | Production-grade with hybrid retrieval, REPL, debug mode |
| `knowledge_full.md` | N/A | ~7.2 MB | Merged Clockify docs (input to chunking) |
| `chunks.jsonl` | Generated | Varies | JSONL format: `{"id": int, "text": str}` per line |
| `vecs.npy` | Generated | Binary | NumPy array [num_chunks, 768] (normalized embeddings) |
| `meta.jsonl` | Generated | Varies | Metadata parallel to vecs.npy (chunk ID, text) |
| `bm25.json` | Generated (v2.0) | Varies | BM25 index for keyword search |
| `rag_env/` | N/A | N/A | Python virtual environment (pre-configured) |

### Dependencies

**Python packages** (in `requirements.txt`):
- `requests==2.32.5` – HTTP client for Ollama API calls
- `numpy==2.3.4` – Numerical arrays (embeddings, vectors)

**External services**:
- Ollama (default: http://127.0.0.1:11434, configurable via OLLAMA_URL) with models:
  - `nomic-embed-text` – 768-dim semantic embeddings
  - `qwen2.5:32b` – LLM for answer generation

**Configuration** (hardcoded in scripts, can be overridden via environment variables):
- `CHUNK_SIZE = 1600` – Characters per chunk
- `CHUNK_OVERLAP = 200` – Character overlap for oversized chunks
- `DEFAULT_TOP_K = 12` (v2.0) – Chunks to retrieve before reranking
- `DEFAULT_PACK_TOP = 6` (v2.0) – Final chunks to include in context
- `DEFAULT_THRESHOLD = 0.30` (v2.0) – Minimum similarity score for acceptance

## Common Development Tasks

### Build the Knowledge Base (One-Time)

**v2.0 (Recommended)**:
```bash
source rag_env/bin/activate
python3 clockify_support_cli.py build knowledge_full.md
```
Creates: `chunks.jsonl`, `vecs_n.npy`, `meta.jsonl`, `bm25.json`, `index.meta.json`

**v1.0**:
```bash
source rag_env/bin/activate
python3 clockify_rag.py chunk
python3 clockify_rag.py embed
```
Creates: `chunks.jsonl`, `vecs.npy`, `meta.jsonl`

### Run the Application

**v2.0 (Interactive REPL)**:
```bash
source rag_env/bin/activate
python3 clockify_support_cli.py chat [--debug]
```
Then type questions at the prompt. Type `:exit` to quit, `:debug` to toggle diagnostics.

**v1.0 (Single Query)**:
```bash
source rag_env/bin/activate
python3 clockify_rag.py ask "Your question here"
```

### Debug/Test an Index

**v2.0 with debug output**:
```bash
python3 clockify_support_cli.py chat --debug
> :debug
> Your question
[Shows retrieved chunks, scores, ranking]
```

**Rebuild index** (if corrupted):
```bash
rm -f chunks.jsonl vecs_n.npy meta.jsonl bm25.json index.meta.json
python3 clockify_support_cli.py build knowledge_full.md
```

## Key Implementation Details

### Chunking Strategy (Both Versions)

- Splits by second-level markdown headings (`##`)
- Enforces max `CHUNK_SIZE` (1600 chars)
- For oversized chunks, creates sub-chunks with `CHUNK_OVERLAP` (200 chars) for context preservation
- Assigns unique monotonic integer IDs

### Retrieval Pipeline

**v1.0**:
1. Embed question with nomic-embed-text
2. Cosine similarity against all chunk vectors
3. Retrieve top 6 chunks
4. Check if ≥2 chunks have similarity ≥0.3
5. Pass to LLM with system prompt

**v2.0 (Hybrid)**:
1. Embed question with nomic-embed-text
2. Retrieve top `DEFAULT_TOP_K` (12) via:
   - **BM25**: Exact keyword matching (sparse)
   - **Dense**: Cosine similarity (semantic)
3. Merge results, apply **MMR** (Maximal Marginal Relevance, lambda=0.7) to diversify
4. Pack top `DEFAULT_PACK_TOP` (6) chunks if similarity ≥ `DEFAULT_THRESHOLD` (0.30)
5. Format snippets with ID, title, context
6. Pass to LLM with system prompt requiring closed-book answers

### LLM Prompting

Both versions instruct the model to:
- Use only provided snippets
- Refuse if information not in snippets
- Return exact phrase: `"I don't know based on the MD."`
- Cite source IDs when answering

### Offline Operation

- All embeddings computed locally (nomic-embed-text)
- All LLM inference local (qwen2.5:32b)
- No external API calls
- No internet required
- Deterministic timeouts (configurable via env vars)

## Testing & Validation

**Example queries** to validate the system:
```bash
python3 clockify_support_cli.py ask "How do I track time in Clockify?"
python3 clockify_support_cli.py ask "What are the pricing plans?"
python3 clockify_support_cli.py ask "Can I track time offline?"
python3 clockify_support_cli.py ask "How do I set up SSO?"
```

**Expected output**:
- Answer with citations: `[id_123, id_456]`
- Or refusal: `"I don't know based on the MD."`

## Configuration & Customization

### Environment Variables (v2.0)

```bash
# Ollama endpoint (default: http://127.0.0.1:11434)
# Override only if Ollama runs on a different machine
export OLLAMA_URL="http://127.0.0.1:11434"

# Model names
export GEN_MODEL="qwen2.5:32b"
export EMB_MODEL="nomic-embed-text"

# Context budget in tokens (~11,200 chars at 4 tokens/word)
export CTX_BUDGET="2800"

# HTTP timeout control
export EMB_CONNECT_TIMEOUT="3"
export EMB_READ_TIMEOUT="120"
export CHAT_CONNECT_TIMEOUT="3"
export CHAT_READ_TIMEOUT="180"
```

### Tuning Parameters (Edit in scripts)

**v1.0** (`clockify_rag.py`, lines 7-15):
```python
CHUNK_SIZE = 1600          # Increase for larger chunks
CHUNK_OVERLAP = 200        # More overlap = better context at boundaries
SIMILARITY_THRESHOLD = 0.3 # Stricter = fewer false positives
```

**v2.0** (`clockify_support_cli.py`, lines 41-51):
```python
CHUNK_CHARS = 1600         # Character limit per chunk
CHUNK_OVERLAP = 200        # Overlap between sub-chunks
DEFAULT_TOP_K = 12         # Retrieve before reranking
DEFAULT_PACK_TOP = 6       # Final chunks in context
DEFAULT_THRESHOLD = 0.30   # Minimum similarity
MMR_LAMBDA = 0.7           # Diversity vs. relevance (0-1)
CTX_TOKEN_BUDGET = 2800    # Context window budget
```

## File Workflows

### Artifact Versioning (v2.0 Only)

`index.meta.json` tracks:
- MD5 hash of knowledge_full.md (detects source changes)
- Last build timestamp
- Index version

If source KB changes, rebuild is automatically triggered.

### Storage Layout

```
/Users/15x/Downloads/KBDOC/
├── clockify_rag.py              (v1.0)
├── clockify_support_cli.py       (v2.0) ← USE THIS
├── knowledge_full.md            (6.9 MB input)
├── chunks.jsonl                 (generated)
├── vecs_n.npy / vecs.npy        (generated)
├── meta.jsonl                   (generated)
├── bm25.json                    (v2.0 only)
├── index.meta.json              (v2.0 only)
├── rag_env/                     (venv)
└── [documentation files]
```

## Troubleshooting Notes for Future Development

### Connection Issues
- Verify Ollama is running: `ollama serve`
- Default endpoint is `http://127.0.0.1:11434` (localhost)
- Override with `OLLAMA_URL` env var if Ollama runs on different machine
- Check connectivity: `curl http://127.0.0.1:11434/api/version`

### Model Issues
- Pull missing models: `ollama pull nomic-embed-text`, `ollama pull qwen2.5:32b`
- Check available: `ollama list`

### Performance Tuning
- **Slow embeddings**: Increase `EMB_READ_TIMEOUT` or reduce `CHUNK_SIZE`
- **Low accuracy**: Adjust `DEFAULT_THRESHOLD` (lower = more lenient), increase `DEFAULT_TOP_K`
- **Memory spikes**: Reduce `DEFAULT_PACK_TOP` or `CTX_TOKEN_BUDGET`

### Build Failures
- **Lock file stale**: Remove `.build.lock` if process crashed
- **Index corrupted**: Delete generated files and rebuild
- **KB changed**: Edit `knowledge_full.md`, trigger rebuild (v2.0 detects automatically)

## Version Comparison

| Aspect | v1.0 | v2.0 |
|--------|------|------|
| Accuracy | ~70% | ~85% |
| Retrieval | Cosine similarity only | Hybrid (BM25 + dense + MMR) |
| UI | Command-line (separate steps) | Interactive REPL |
| Debug capability | None | Yes (`:debug` toggle) |
| Stateful | No | No (each query fresh) |
| Lock mechanism | None | Yes (atomic file creation) |
| File count | 1 script | 1 script |
| Complexity | Low | Moderate |
| **Recommended** | **No** | **YES** ✅ |

## Documentation Map

- **START_HERE.md** – Entry point with quick choice (v1 vs v2)
- **SUPPORT_CLI_QUICKSTART.md** – v2.0 5-minute quick start
- **CLOCKIFY_SUPPORT_CLI_README.md** – v2.0 full technical guide
- **QUICKSTART.md** – v1.0 quick start
- **README_RAG.md** – v1.0 full guide (30 min read)
- **VERSION_COMPARISON.md** – Detailed v1 vs v2 analysis
- **PROJECT_STRUCTURE.md** – Directory layout and file purposes

## Notes for Future Work

1. **Extend knowledge base**: Add new .md sections to `knowledge_full.md`, rebuild index
2. **Switch models**: Change `EMB_MODEL` or `GEN_MODEL` constants, ensure available in Ollama
3. **Deploy to team**: v2.0 is single-file, easier to distribute; include `rag_env/` or require users to install numpy + requests
4. **Optimize retrieval**: v2.0's MMR+BM25 is near optimal; consider cross-encoder reranking for marginal gains
5. **Add feedback loop**: v2.0 doesn't log interactions; could add optional JSON logging for model fine-tuning
6. **Scale to multiple KBs**: Both support single KB only; would need multi-index wrapper

---

**Version**: 2.0 Recommended (1.0 also available)
**Status**: ✅ Production Ready
**Date**: 2025-11-05
**Platform**: macOS/Linux (v1.0 and v2.0); Windows requires manual venv setup
**Python**: 3.7+
**Apple Silicon**: ✅ M1/M2/M3 Compatible - See [M1_COMPATIBILITY.md](M1_COMPATIBILITY.md) for installation guide
