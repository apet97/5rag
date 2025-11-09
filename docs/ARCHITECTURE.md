# System Architecture

High-level overview of the Clockify RAG system design, components, and data flow.

## High-Level Pipeline

```
Knowledge Base (Markdown)
        ↓
   Ingestion
        ↓
   Chunking (semantic splits)
        ↓
   Embedding (local SentenceTransformer)
        ↓
   ┌──────────────────────────────────────┐
   │      Multi-Backend Indexing          │
   ├──────────────────────────────────────┤
   │ • FAISS (vector similarity, primary) │
   │ • HNSW (ANN fallback)                │
   │ • BM25 (keyword/lexical fallback)    │
   └──────────────────────────────────────┘
        ↓
   Hybrid Retrieval (BM25 + Dense + MMR)
        ↓
   Query Caching (optional)
        ↓
   LLM Inference (Ollama or external API)
        ↓
   Answer Generation + Citation
        ↓
        User
```

## Component Overview

### 1. Ingestion & Chunking

**File**: `clockify_rag/chunking.py`

**Purpose**: Parse markdown documents and split into semantic chunks.

**Process**:
```
Markdown → Parse by ## headings → Enforce max size → Add overlap → Unique IDs
```

**Key Parameters**:
- `CHUNK_CHARS` (1600): Max characters per chunk
- `CHUNK_OVERLAP` (200): Overlap between sub-chunks

**Output**: `chunks.jsonl` (one JSON per line: `{id, text, source}`)

### 2. Embedding

**File**: `clockify_rag/embedding.py`

**Purpose**: Convert text to dense vector representations.

**Backends**:
- **Local** (default): SentenceTransformer
  - Model: `intfloat/multilingual-e5-base` (768-dim)
  - Device: CPU, MPS (macOS), or CUDA
  - Speed: 100 texts/min (CPU), 500+/min (MPS)

- **Ollama**: Remote API
  - Model: `nomic-embed-text` (768-dim)
  - Speed: Depends on Ollama host

**Output**: `vecs_n.npy` (numpy array [chunks, 768])

### 3. Indexing

**Files**: `clockify_rag/indexing.py`

**Three-Layer Fallback Strategy**:

```python
try:
    use FAISS (primary, fast ANN)
except:
    try:
        use HNSW (fallback ANN)
    except:
        use BM25 (reliable keyword search)
```

#### FAISS (Primary)
- **Index Type**: IVFFlat
- **Parameters**:
  - `nlist` = 64 (M1 for stability) or 256 (x86)
  - `nprobe` = 16 (clusters to search)
- **Speed**: <10ms per query
- **Memory**: ~100MB for 1000 chunks
- **Files**: `faiss.index`

#### HNSW (Fallback)
- **Type**: Hierarchical Navigable Small World
- **Parameters**:
  - `m` = 16 (neighbors per node)
  - `ef_construction` = 200
- **Speed**: 20-50ms per query
- **Memory**: More than FAISS
- **Files**: `hnsw_cosine.bin`

#### BM25 (Last Resort)
- **Type**: Okapi BM25 (classic TF-IDF variant)
- **Parameters**:
  - `k1` = 1.2 (term saturation)
  - `b` = 0.65 (length normalization)
- **Speed**: 50-200ms per query
- **Memory**: <10MB
- **Files**: `bm25.json`

### 4. Query Processing

**File**: `clockify_rag/retrieval.py`

**Steps**:

1. **Query Embedding**
   - Embed user question with same model as chunks
   - Output: 768-dim vector

2. **Dual Retrieval**
   - BM25: Keyword matching → top_k results
   - Dense: Vector similarity → top_k results

3. **Hybrid Fusion** (Reciprocal Rank Fusion)
   ```
   score = 1/(1 + bm25_rank) + 1/(1 + dense_rank)
   weighted = alpha * bm25_score + (1 - alpha) * dense_score
   ```

4. **MMR Reranking** (Maximal Marginal Relevance)
   - Avoid redundant/similar chunks
   - Balance relevance vs diversity
   - Formula: `mmr_score = lambda * relevance - (1 - lambda) * similarity_to_selected`

5. **Filtering**
   - Keep chunks with score ≥ threshold
   - Limit to pack_top chunks
   - Check coverage (min 2 chunks)

### 5. Context Packing

**File**: `clockify_rag/retrieval.py` (`pack_snippets`)

**Purpose**: Format retrieved chunks for LLM context window.

**Process**:
```
Selected chunks → Format with metadata → Estimate tokens →
Truncate if needed → Create context string
```

**Token Budget**:
- `CTX_TOKEN_BUDGET` (12000 tokens)
- Formula: `text_length ≈ tokens × 4 chars/token`
- Reserve: 60% for snippets, 40% for question/answer

### 6. LLM Inference

**File**: `clockify_rag/answer.py`

**Backends**:
- **Ollama**: Local inference (default)
  - HTTP API to `OLLAMA_URL`
  - Model: `GEN_MODEL` (default: qwen2.5:32b)
  - Streaming support

- **OpenAI**: External API (requires key)
- **Anthropic**: External API (requires key)

**Prompt Format**:
```
[System]: You are a helpful assistant. Answer only from provided context.
If not in context, say: "I don't know based on the MD."

[Context]:
[Retrieved chunks with formatting]

[User]: [Question]

[Assistant]: [Generated answer]
```

### 7. Caching

**Files**: `clockify_rag/caching.py`

**Query Cache**:
- In-memory LRU cache
- Stores: question → answer + metadata
- TTL: 1 hour
- Max entries: 1000
- Persistence: JSON disk file

**Rate Limiter** (optional):
- Per-IP request limiting
- Configurable: queries/minute
- Token bucket algorithm

### 8. CLI & API

**Files**:
- `clockify_rag/cli_modern.py` - Typer CLI
- `clockify_rag/api.py` - FastAPI REST server

**CLI Commands**:
- `ragctl doctor` - System diagnostics
- `ragctl ingest` - Build index
- `ragctl query` - Single query
- `ragctl chat` - Interactive REPL
- `ragctl eval` - RAGAS evaluation

**API Endpoints**:
- `GET /health` - Health check
- `POST /v1/query` - Submit question
- `POST /v1/ingest` - Rebuild index (background)
- `GET /v1/config` - Current config
- `GET /v1/metrics` - Metrics

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                    User Query (CLI/API)                    │
│                                                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Query Caching Layer   │◄──Cache Hit? Return
            │  (LRU + Redis)         │
            └────────┬───────────────┘
                     │
                     ▼
          ┌──────────────────────────┐
          │   Embed Question         │
          │   (SentenceTransformer)  │
          └────────┬─────────────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
       ▼                       ▼
┌─────────────────┐   ┌──────────────────┐
│  BM25 Retrieval │   │ Dense Retrieval  │
│  (Keyword)      │   │ (Vector ANN)     │
└────────┬────────┘   └─────────┬────────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Hybrid Fusion (RRF)   │
         │ alpha * bm25 +        │
         │ (1-alpha) * dense     │
         └──────────┬────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ MMR Reranking        │
         │ (Avoid Redundancy)   │
         └──────────┬────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Pack Snippets        │
         │ Format for LLM       │
         │ Token Budget Check   │
         └──────────┬────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ LLM Inference        │
         │ (Ollama/OpenAI)      │
         └──────────┬────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Extract Citations    │
         │ Confidence Scoring   │
         └──────────┬────────────┘
                    │
                    ▼
            ┌─────────────────┐
            │  Log Query      │
            │  (JSONL file)   │
            └────────┬────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Return Answer       │
          │  + Citations         │
          │  + Confidence        │
          └──────────────────────┘
```

## Module Dependencies

```
clockify_rag/
├── __init__.py                  (package exports)
├── config.py                    (settings + env vars)
├── exceptions.py                (custom exceptions)
├── utils.py                     (helpers)
├── http_utils.py                (HTTP sessions)
│
├── chunking.py                  (→ utils)
│   Parse & split markdown
│
├── embedding.py                 (→ config, utils)
│   Local SentenceTransformer or Ollama
│
├── indexing.py                  (→ config, chunking, embedding)
│   FAISS + HNSW + BM25
│
├── retrieval.py                 (→ config, indexing, embedding)
│   Hybrid search (BM25 + dense + MMR)
│
├── caching.py                   (→ config, utils)
│   Query cache + rate limiting
│
├── answer.py                    (→ retrieval, config)
│   LLM inference + citations
│
├── metrics.py                   (→ utils)
│   Performance tracking
│
├── cli_modern.py                (→ all above)
│   Typer CLI (doctor, ingest, query, chat)
│
├── api.py                       (→ all above)
│   FastAPI REST server
│
└── plugins/                     (extensibility)
    Custom retrievers, rerankers
```

## Execution Flow (Single Query)

```
1. User enters question
   ↓
2. Check query cache → hit → return cached answer
   ↓
3. Embed question with SentenceTransformer
   ↓
4. Parallel:
   a) BM25 retrieval (top 15 chunks by keywords)
   b) FAISS/HNSW retrieval (top 15 chunks by vector similarity)
   ↓
5. Fuse results (reciprocal rank fusion with alpha weighting)
   ↓
6. Apply MMR (remove redundant chunks)
   ↓
7. Filter by threshold (≥0.25 similarity)
   ↓
8. Pack snippets (format + estimate tokens)
   ↓
9. Call LLM with context window ≤12000 tokens
   ↓
10. Extract citations from generated answer
    ↓
11. Score confidence
    ↓
12. Cache result
    ↓
13. Log query (optional)
    ↓
14. Return answer + sources + confidence
```

## Performance Characteristics

### Latency (M1 Pro, 16GB)

| Phase | Duration |
|-------|----------|
| Query embedding | 50-100ms |
| BM25 retrieval | 10-20ms |
| FAISS retrieval | 5-15ms |
| Fusion + reranking | 20-50ms |
| Context packing | 10-20ms |
| LLM inference | 500-1500ms |
| **Total** | **600-1700ms** |

### Memory Usage

| Component | Size |
|-----------|------|
| SentenceTransformer model | 300-600 MB |
| FAISS index (1000 chunks) | 100-200 MB |
| BM25 index | 10-50 MB |
| Cached queries (1000) | 50-100 MB |
| **Total** | **500-1000 MB** |

### Index Size

| Artifact | Size (per 100 chunks) |
|----------|---------------------|
| chunks.jsonl | 1-2 MB |
| vecs_n.npy | 300 KB |
| bm25.json | 100-200 KB |
| faiss.index | 1-2 MB |
| **Total** | **3-5 MB** |

## Scalability

### Maximum Practical Index Sizes

- **Small** (0-1000 chunks): All backends ✅
- **Medium** (1000-10000 chunks): FAISS + HNSW ✅, BM25 ✅
- **Large** (10000-100000 chunks): FAISS ✅, HNSW ⚠️, BM25 ⚠️
- **XLarge** (>100000 chunks): FAISS only (with sharding)

### Optimization for Scale

1. **Shard index**: Split by category/domain
2. **Use cloud FAISS**: Vespa, Elasticsearch, Pinecone
3. **Async embeddings**: Process in batches
4. **Distributed LLM**: vLLM, Ray Serve
5. **Caching**: Redis for multi-instance

## Extension Points

### Custom Retrievers

Implement `RetrieverPlugin` interface:

```python
from clockify_rag.plugins import RetrieverPlugin, register_plugin

class MyRetriever(RetrieverPlugin):
    def retrieve(self, question: str, top_k: int):
        # Custom logic
        return results

    def get_name(self) -> str:
        return "my_retriever"

register_plugin(MyRetriever())
```

### Custom Rerankers

```python
class MyReranker(RerankerPlugin):
    def rerank(self, question: str, candidates: list):
        # Custom scoring
        return sorted_candidates
```

### Custom Embedders

```python
class MyEmbedder(EmbedderPlugin):
    def embed(self, texts: list[str]):
        # Custom embeddings
        return vectors
```

## Testing Strategy

### Unit Tests (tests/)

- Chunking logic
- Embedding interfaces
- BM25 ranking
- Cache operations
- Answer extraction

### Integration Tests

- End-to-end ingestion
- Retrieval pipeline
- LLM inference
- API endpoints

### Smoke Tests (CI)

- Quick index build
- Single query
- Doctor diagnostics

## Future Enhancements

1. **Cross-Encoder Reranking**: 15% accuracy boost (bge-reranker)
2. **Hybrid Search v2**: Learning-to-rank instead of RRF
3. **Knowledge Graph**: Entity linking + relationship retrieval
4. **Long Context**: Support 100K+ token windows
5. **Adaptive Routing**: Confidence-based escalation
6. **Multi-Language**: Language detection + routing
7. **Persistent Cache**: Redis backend for distributed systems

---

For implementation details, see:
- [README.md](README.md) - Quick start
- [CONFIG.md](CONFIG.md) - Configuration options
- [Source code](../clockify_rag/) - Actual implementation
