# Configuration Guide

> **Note:** The canonical, up-to-date environment variable reference now lives in [docs/CONFIGURATION.md](CONFIGURATION.md). This document remains for historical context and deeper parameter explanations.

Comprehensive documentation of all Clockify RAG configuration options.

## Configuration Methods

There are three ways to configure the system:

1. **Environment Variables** (highest priority): `PARAM_NAME=value command`
2. **`.env` File**: Create `.env` file in project root
3. **config/defaults.yaml**: Default values (included in repo)

### Priority Order

```
Environment Variables > .env file > config/defaults.yaml > hardcoded defaults
```

### Example

```bash
# Method 1: Environment variable
OLLAMA_URL=http://remote.ollama:11434 ragctl query "question"

# Method 2: .env file
echo "OLLAMA_URL=http://remote.ollama:11434" >> .env
ragctl query "question"

# Method 3: defaults.yaml
# Edit config/defaults.yaml and set ollama.url
```

## All Configuration Parameters

### Ollama / LLM Configuration

#### `OLLAMA_URL`
- **Default**: `http://127.0.0.1:11434`
- **Type**: URL
- **Description**: Base URL for Ollama service
- **Notes**:
  - Use `http://host.docker.internal:11434` in Docker on macOS
  - Can point to remote Ollama server
  - Must include protocol (http/https)

Example:
```bash
OLLAMA_URL=http://ollama.example.com:11434 ragctl chat
```

#### `GEN_MODEL`
- **Default**: `qwen2.5:32b`
- **Type**: string
- **Description**: Model for text generation
- **Alternatives**:
  - `mistral`: Smaller, faster (7B)
  - `neural-chat`: Medium (7B)
  - `tinyllama`: Tiny (1.1B)
  - `llama2`: Classic (7B or 70B)

Example:
```bash
GEN_MODEL=mistral ragctl chat
```

#### `EMB_MODEL`
- **Default**: `nomic-embed-text` (Ollama) or `intfloat/multilingual-e5-base` (local)
- **Type**: string
- **Description**: Model for embedding texts
- **Alternatives**:
  - `all-MiniLM-L6-v2`: Smaller (384-dim)
  - `all-mpnet-base-v2`: Better quality (768-dim)
  - `multilingual-e5-base`: Multilingual (768-dim)

#### `CHAT_READ_TIMEOUT`
- **Default**: `120`
- **Type**: float (seconds)
- **Range**: 1.0 - 600.0
- **Description**: Timeout for LLM generation requests
- **Notes**: Increase if model is slow

Example:
```bash
CHAT_READ_TIMEOUT=300 ragctl chat
```

#### `EMB_READ_TIMEOUT`
- **Default**: `60`
- **Type**: float (seconds)
- **Range**: 1.0 - 600.0
- **Description**: Timeout for embedding requests

#### `DEFAULT_RETRIES`
- **Default**: `2`
- **Type**: int
- **Range**: 0 - 10
- **Description**: Retry failed Ollama requests
- **Notes**: Useful for remote/unreliable connections

### Retrieval Configuration

#### `DEFAULT_TOP_K`
- **Default**: `15`
- **Type**: int
- **Range**: 1 - 100
- **Description**: Number of chunks to retrieve before filtering
- **Notes**:
  - Higher = better recall, slower retrieval
  - Lower = faster, may miss relevant chunks
  - Typical range: 10-30

Example:
```bash
DEFAULT_TOP_K=25 ragctl query "complex question"
```

#### `DEFAULT_PACK_TOP`
- **Default**: `8`
- **Type**: int
- **Range**: 1 - 50
- **Description**: Final number of chunks to include in context
- **Notes**:
  - Must be ≤ DEFAULT_TOP_K
  - Higher = more context, longer inference time
  - Typical range: 5-12

#### `DEFAULT_THRESHOLD`
- **Default**: `0.25`
- **Type**: float
- **Range**: 0.0 - 1.0
- **Description**: Minimum similarity score for chunk inclusion
- **Notes**:
  - Lower = more lenient, more false positives
  - Higher = stricter, fewer chunks included
  - Typical range: 0.2-0.4

Example:
```bash
DEFAULT_THRESHOLD=0.35 ragctl query "specific question"
```

#### `CTX_BUDGET` (Context Token Budget)
- **Default**: `12000`
- **Type**: int
- **Range**: 100 - 100000
- **Description**: Maximum tokens for context in LLM
- **Notes**:
  - Qwen 32B: 32K context window → use 12K
  - Smaller models: reduce to 6K-8K
  - Larger models: can use 20K+

Example:
```bash
CTX_BUDGET=8000 ragctl chat  # For smaller models
```

#### `MMR_LAMBDA` (Maximal Marginal Relevance)
- **Default**: `0.75`
- **Type**: float
- **Range**: 0.0 - 1.0
- **Description**: Relevance vs. diversity trade-off
- **Notes**:
  - 0.0 = maximize diversity (avoid redundancy)
  - 1.0 = maximize relevance (pure similarity ranking)
  - 0.7-0.8 = recommended balanced

### Chunking Configuration

#### `CHUNK_CHARS`
- **Default**: `1600`
- **Type**: int
- **Description**: Maximum characters per chunk
- **Notes**:
  - Larger chunks = more context, slower
  - Smaller chunks = finer granularity
  - Typical: 1000-2000

Example:
```bash
CHUNK_CHARS=2000 ragctl ingest --input knowledge_full.md
```

#### `CHUNK_OVERLAP`
- **Default**: `200`
- **Type**: int
- **Description**: Character overlap between chunks
- **Notes**:
  - Prevents losing context at chunk boundaries
  - 10-20% of CHUNK_CHARS recommended

### BM25 Configuration

#### `BM25_K1`
- **Default**: `1.2`
- **Type**: float
- **Range**: 0.1 - 10.0
- **Description**: Term frequency saturation parameter
- **Notes**:
  - Higher = more emphasis on term frequency
  - Typical: 1.0-1.5

#### `BM25_B`
- **Default**: `0.65`
- **Type**: float
- **Range**: 0.0 - 1.0
- **Description**: Length normalization
- **Notes**:
  - 0.0 = no length normalization
  - 1.0 = full normalization
  - 0.6-0.7 = good for technical docs

### Hybrid Search

#### `ALPHA_HYBRID`
- **Default**: `0.5`
- **Type**: float
- **Range**: 0.0 - 1.0
- **Description**: BM25 weight in hybrid search
- **Formula**: `score = alpha * bm25 + (1 - alpha) * dense`
- **Notes**:
  - 0.0 = dense search only (semantic)
  - 0.5 = equal weight (balanced)
  - 1.0 = BM25 only (keyword)

Example:
```bash
# More semantic search
ALPHA_HYBRID=0.3 ragctl query "What is the meaning of..."

# More keyword-based
ALPHA_HYBRID=0.7 ragctl query "error: connection timeout"
```

### Index Configuration

#### `USE_ANN`
- **Default**: `faiss`
- **Type**: string (faiss | none)
- **Description**: Approximate Nearest Neighbors backend
- **Notes**:
  - `faiss`: Fast vector search (requires conda on M1)
  - `none`: Skip ANN, use BM25 only (slower but reliable)

Example:
```bash
# Fallback to BM25 if FAISS unavailable
USE_ANN=none ragctl ingest
```

#### `ANN_NLIST`
- **Default**: `64` (M1 stability), `256` (x86)
- **Type**: int
- **Description**: FAISS IVF clusters
- **Notes**:
  - Lower on M1 (64) to avoid segfaults
  - Higher on x86 (256) for better accuracy
  - Must rebuild index to change

#### `ANN_NPROBE`
- **Default**: `16`
- **Type**: int
- **Description**: Clusters to search per query
- **Notes**:
  - Higher = more accurate, slower
  - Typical: 8-32

### Caching Configuration

#### `QUERY_CACHE_ENABLED`
- **Default**: `true`
- **Type**: bool
- **Description**: Enable query result caching
- **Notes**: Dramatically speeds up repeated questions

#### `QUERY_CACHE_MAX`
- **Default**: `1000`
- **Type**: int
- **Description**: Maximum cached queries
- **Notes**: LRU eviction when limit reached

### Logging Configuration

#### `LOG_LEVEL`
- **Default**: `info`
- **Type**: string (debug | info | warning | error)
- **Description**: Console log verbosity

Example:
```bash
LOG_LEVEL=debug ragctl chat  # Verbose output
```

#### `RAG_LOG_FILE`
- **Default**: `rag_queries.jsonl`
- **Type**: path
- **Description**: Query log file (JSONL format)
- **Notes**:
  - Records all queries and answers
  - Disabled if not set
  - Useful for analytics

Example:
```bash
RAG_LOG_FILE=./var/logs/queries.jsonl ragctl chat
```

#### `RAG_LOG_INCLUDE_ANSWER`
- **Default**: `1` (enabled)
- **Type**: bool
- **Description**: Include answers in logs
- **Notes**: Set to `0` to redact for privacy

#### `RAG_LOG_INCLUDE_CHUNKS`
- **Default**: `0` (disabled)
- **Type**: bool
- **Description**: Include retrieved chunks in logs
- **Notes**: Disabled by default for privacy

### Advanced Configuration

#### `DEFAULT_NUM_CTX`
- **Default**: `32768`
- **Type**: int
- **Description**: Context window size in LLM
- **Notes**:
  - Qwen 32B: 32768
  - Smaller models: 4096-8192
  - Don't change unless using different model

#### `DEFAULT_NUM_PREDICT`
- **Default**: `512`
- **Type**: int
- **Description**: Maximum output tokens
- **Notes**: Limit LLM output length

#### `MAX_QUERY_LENGTH`
- **Default**: `10000`
- **Type**: int
- **Description**: Maximum question length
- **Notes**: Protection against DoS/resource exhaustion

#### `EMB_MAX_WORKERS`
- **Default**: `8`
- **Type**: int
- **Range**: 1 - 64
- **Description**: Parallel threads for embedding
- **Notes**: Increase for faster bulk embedding

#### `EMB_BATCH_SIZE`
- **Default**: `32`
- **Type**: int
- **Range**: 1 - 1000
- **Description**: Texts per embedding batch
- **Notes**: Higher = faster but more memory

### Confidence Routing (v5.9)

#### `USE_INTENT_CLASSIFICATION`
- **Default**: `1` (enabled)
- **Type**: bool
- **Description**: Dynamic alpha adjustment based on query intent
- **Notes**: +8-12% accuracy improvement

### FAQ Caching (v5.9)

#### `FAQ_CACHE_ENABLED`
- **Default**: `0` (disabled)
- **Type**: bool
- **Description**: Enable precomputed FAQ cache
- **Notes**: Requires precomputation during build

#### `FAQ_CACHE_PATH`
- **Default**: `faq_cache.json`
- **Type**: path
- **Description**: Path to FAQ cache file

## Example Configurations

### M1/M2/M3 Mac Optimization

```bash
# Leverage GPU acceleration
OLLAMA_URL=http://127.0.0.1:11434
GEN_MODEL=qwen2.5:32b

# Balanced retrieval
DEFAULT_TOP_K=20
DEFAULT_PACK_TOP=10
DEFAULT_THRESHOLD=0.2

# Memory efficient
EMB_BATCH_SIZE=16
EMB_MAX_WORKERS=4
```

### Fast/Lightweight Setup

```bash
# Use smaller models
GEN_MODEL=mistral
EMB_MODEL=all-MiniLM-L6-v2

# Reduced context
CTX_BUDGET=6000
DEFAULT_TOP_K=10
DEFAULT_PACK_TOP=5

# Faster retrieval
ALPHA_HYBRID=1.0  # BM25 only

# Disable ANN if memory constrained
USE_ANN=none
```

### High-Quality/Accurate Setup

```bash
# Use largest models
GEN_MODEL=qwen2.5:32b
EMB_MODEL=multilingual-e5-base

# More context
CTX_BUDGET=20000
DEFAULT_TOP_K=30
DEFAULT_PACK_TOP=15

# Lower threshold for more results
DEFAULT_THRESHOLD=0.15

# Balanced hybrid search
ALPHA_HYBRID=0.5
MMR_LAMBDA=0.7

# More workers for embedding
EMB_MAX_WORKERS=16
```

### High-Volume API Setup

```bash
# Production settings
OLLAMA_URL=http://ollama-cluster.internal:11434
LOG_LEVEL=warning

# Caching enabled
QUERY_CACHE_MAX=5000

# Query logging
RAG_LOG_FILE=/var/log/rag_queries.jsonl
RAG_LOG_INCLUDE_ANSWER=1
RAG_LOG_INCLUDE_CHUNKS=0  # Privacy

# Faster retrieval
DEFAULT_TOP_K=12
DEFAULT_PACK_TOP=6

# Parallel embedding
EMB_MAX_WORKERS=16
EMB_BATCH_SIZE=64
```

## Environment File Example

Create `.env` file:

```bash
# .env file (local development)

# Ollama
OLLAMA_URL=http://127.0.0.1:11434
GEN_MODEL=qwen2.5:32b
EMB_MODEL=nomic-embed-text

# Retrieval
DEFAULT_TOP_K=15
DEFAULT_PACK_TOP=8
DEFAULT_THRESHOLD=0.25

# Context
CTX_BUDGET=12000

# Logging
LOG_LEVEL=info
RAG_LOG_FILE=./var/logs/queries.jsonl

# Performance
EMB_MAX_WORKERS=8
EMB_BATCH_SIZE=32
```

## Validation & Constraints

- **OLLAMA_URL**: Must be valid URL with http/https
- **Timeouts**: Must be positive floats (seconds)
- **TOP_K**: Must be ≥ 1 and ≤ 100
- **PACK_TOP**: Must be ≤ TOP_K
- **THRESHOLD**: Must be 0.0-1.0
- **ALPHA**: Must be 0.0-1.0
- **MMR_LAMBDA**: Must be 0.0-1.0

Invalid values will:
1. Log a warning
2. Use safe default
3. Continue execution

## Performance Tuning

### Slow Queries?

1. **Reduce TOP_K**: Retrieve fewer candidates
2. **Increase ALPHA_HYBRID**: Use BM25 (faster)
3. **Disable logging**: Remove RAG_LOG_FILE
4. **Increase PACK_TOP threshold**: Filter more aggressively

### Low Quality Answers?

1. **Increase TOP_K**: Get more candidates
2. **Lower THRESHOLD**: Include marginal chunks
3. **Increase MMR_LAMBDA**: Favor relevance
4. **Enable debug**: See retrieval scores

### High Memory Usage?

1. **Reduce EMB_MAX_WORKERS**: Fewer threads
2. **Reduce EMB_BATCH_SIZE**: Smaller batches
3. **Reduce CTX_BUDGET**: Less context
4. **Use smaller models**: all-MiniLM instead of e5

## Debugging Configuration

```bash
# Show resolved config
LOG_LEVEL=debug ragctl doctor

# Trace retrieval
ragctl query "q" --debug

# Verify environment variables
env | grep -i "^(ollama|rag|emb|bm25|ctx|default)"
```

---

For more information, see:
- [INSTALL_macOS_ARM64.md](INSTALL_macOS_ARM64.md) - M1 specific setup
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [README.md](README.md) - Quick start
