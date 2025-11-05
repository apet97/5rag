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

### Step 1: Chunk the Documentation

```bash
source rag_env/bin/activate
python3 clockify_rag.py chunk
```

This reads `knowledge_full.md` and creates `chunks.jsonl` with ~150 pages of Clockify docs split into manageable chunks.

**Expected output:**
```
Chunking complete: X chunks written to chunks.jsonl
```

### Step 2: Generate Embeddings

```bash
python3 clockify_rag.py embed
```

This generates vector embeddings for all chunks using the local `nomic-embed-text` model.

**Expected output:**
```
Embedding complete: X vectors saved to vecs.npy, metadata to meta.jsonl
```

This may take a few minutes depending on the size of your knowledge base and hardware.

---

## Using the Tool (Repeatable)

### Ask Questions

Once chunks and embeddings are generated, query the knowledge base:

```bash
source rag_env/bin/activate
python3 clockify_rag.py ask "How do I track time in Clockify?"
```

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
python3 clockify_rag.py ask "What is time rounding in Clockify?"
python3 clockify_rag.py ask "How do I set up projects?"
python3 clockify_rag.py ask "Can I track time offline?"
python3 clockify_rag.py ask "How do I export reports?"
python3 clockify_rag.py ask "What billing modes does Clockify support?"
```

---

## File Structure

After setup, your directory contains:

```
/Users/15x/Downloads/KBDOC/
├── clockify_rag.py           # Main CLI tool
├── rag_env/                  # Virtual environment (activate with: source rag_env/bin/activate)
├── knowledge_full.md         # Source documentation (6.9 MB)
├── chunks.jsonl              # Generated: documentation chunks
├── vecs.npy                  # Generated: embedding vectors
├── meta.jsonl                # Generated: chunk metadata
├── README_RAG.md             # Full documentation
└── QUICKSTART.md             # This file
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

Check the URL in `clockify_rag.py` (line 11) matches your Ollama instance.

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
- Reduce `CHUNK_SIZE` in `clockify_rag.py` (currently 1600)
- Ensure you have at least 8GB RAM available
- Close other applications

### Slow Responses

First-time queries may be slow as models load. Subsequent queries are faster.

---

## How It Works

1. **Chunking**: Splits `knowledge_full.md` by `##` sections with 1600-char max per chunk
2. **Embedding**: Converts each chunk to a semantic vector using `nomic-embed-text`
3. **Retrieval**: Computes cosine similarity between your question and all chunks
4. **Ranking**: Returns top 6 most relevant chunks
5. **QA**: Passes retrieved chunks + question to `qwen2.5:32b` LLM for answer generation
6. **Safety**: Rejects answers if fewer than 2 chunks are highly relevant (similarity ≥ 0.3)

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
python3 clockify_rag.py chunk    # Create chunks
python3 clockify_rag.py embed    # Generate embeddings

# Ask questions (repeatable)
python3 clockify_rag.py ask "Your question here"

# Help
python3 clockify_rag.py --help
```

---

**Happy querying!** 🚀
