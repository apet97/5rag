# Clockify RAG CLI - Quick Start Guide

## One-Time Setup

### 1. Install Dependencies

The tool requires Python 3.7+ and a few packages. A virtual environment has already been created in `rag_env/`.

Activate the environment (MacOS/Linux):
```bash
source rag_env/bin/activate
```

Windows:
```bash
rag_env\Scripts\activate
```

Dependencies are already installed: `requests`, `numpy`.

### 2. Start Ollama Server

Ensure Ollama is running locally (default: `http://10.127.0.192:11434`):

```bash
ollama serve
```

Verify the required models are available:
```bash
ollama list
```

If missing, pull them:
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:32b
```

---

## Building the Knowledge Base (One-Time)

### Step 1: Build the Knowledge Base

```bash
source rag_env/bin/activate
python3 clockify_support_cli_final.py build knowledge_full.md
```

The `build` command chunks the Markdown, generates dense embeddings, constructs the BM25 index, and writes out FAISS artifacts in one pass.

**Expected output:**
```
======================================================================
BUILDING KNOWLEDGE BASE
======================================================================
[1/4] Parsing and chunking...
[2/4] Embedding with Ollama...
[3/4] Building BM25 index...
[4/4] Done.
======================================================================
```

This may take a few minutes depending on the size of your knowledge base and hardware.

---

## Using the Tool (Repeatable)

### Ask Questions

Once chunks and embeddings are generated, query the knowledge base:

```bash
source rag_env/bin/activate
python3 clockify_support_cli_final.py ask "How do I track time in Clockify?" --rerank --json
```

The `ask` command accepts all retrieval knobs (`--topk`, `--pack`, `--threshold`, `--rerank`) and `--json` for structured output, so it fits neatly into scripts or automation.

**Example Response:**
```
You can track time in Clockify by [15, 23]:
- Using the Timer button to start/stop tracking in real-time
- Manually entering time entries with custom dates and durations
- Integrating with third-party apps like Google Calendar, Slack, or Jira
```

Or if information isn't available:
```
I don't know based on the MD.
```

---

## Example Queries

Try these questions to test the system:

```bash
python3 clockify_support_cli_final.py ask "What is time rounding in Clockify?"
python3 clockify_support_cli_final.py ask "How do I set up projects?" --topk 16 --pack 8
python3 clockify_support_cli_final.py ask "Can I track time offline?" --rerank
python3 clockify_support_cli_final.py ask "How do I export reports?" --json
python3 clockify_support_cli_final.py ask "What billing modes does Clockify support?"
```

---

## File Structure

After setup, your directory contains:

```
/Users/15x/Downloads/KBDOC/
├── clockify_support_cli_final.py   # Main CLI tool
├── rag_env/                        # Virtual environment (activate with: source rag_env/bin/activate)
├── knowledge_full.md               # Source documentation (6.9 MB)
├── chunks.jsonl                    # Generated: documentation chunks
├── vecs_n.npy                      # Generated: normalized embedding vectors
├── meta.jsonl                      # Generated: chunk metadata
├── bm25.json                       # Generated: sparse index
├── faiss.index                     # Generated: ANN index (if enabled)
├── README_RAG.md                   # Full documentation
└── QUICKSTART.md                   # This file
```

---

## Troubleshooting

### Connection Error: "Cannot connect to Ollama"

```
Embedding request failed for chunk 0: Connection refused
```

**Solution**: Ensure Ollama is running:
```bash
ollama serve
```

Check the URL in `clockify_support_cli_final.py` (line 11) matches your Ollama instance.

### Model Not Found

```
Embedding API returned error 404
```

**Solution**: Pull the missing model:
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:32b
```

### Memory Issues

If you run out of memory during embedding:
- Reduce `CHUNK_SIZE` in `clockify_support_cli_final.py` (currently 1600)
- Ensure you have at least 8GB RAM available
- Close other applications

### Slow Responses

First-time queries may be slow as models load. Subsequent queries are faster.

---

## How It Works

1. **Build**: `clockify_support_cli_final.py build` chunks `knowledge_full.md` (1600 char max with 200-char overlap)
2. **Embed**: Dense vectors are generated via Ollama (`nomic-embed-text`) or the local backend
3. **Index**: BM25 and optional FAISS indices are stored for fast retrieval
4. **Retrieve**: Each `ask`/`chat` query scores BM25 + dense similarities with hybrid weighting
5. **Rerank (optional)**: `--rerank` performs an LLM rerank pass for higher precision
6. **Answer**: Top snippets are packed and sent to `qwen2.5:32b`; the CLI refuses when coverage is insufficient

---

## More Information

See `README_RAG.md` for:
- Detailed architecture explanation
- Advanced configuration options
- Customization examples
- Performance optimization tips
- Known limitations & future improvements

---

## Quick Command Reference

```bash
# Activate environment
source rag_env/bin/activate

# Run setup (one-time)
python3 clockify_support_cli_final.py build knowledge_full.md   # Build retrieval artifacts

# Ask questions (repeatable)
python3 clockify_support_cli_final.py ask "Your question here" --rerank --json

# Interactive mode
python3 clockify_support_cli_final.py chat --debug

# Help
python3 clockify_support_cli_final.py --help
```

---

**Happy querying!** 🚀
