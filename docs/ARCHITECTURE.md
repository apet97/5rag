# System Architecture

High-level overview of the Clockify RAG system design, components, and data flow. Use this as the single source of truth for understanding how data moves through the system, component responsibilities, and external dependencies.

## Table of Contents

- [Architecture Snapshot](#architecture-snapshot)
- [High-Level Responsibilities](#high-level-responsibilities)
- [Data Flow](#data-flow)
- [Component Details](#component-details)
- [External Services](#external-services)
- [Artifacts & Storage](#artifacts--storage)
- [Performance Characteristics](#performance-characteristics)
- [Extension Points](#extension-points)
- [Testing Strategy](#testing-strategy)

## Architecture Snapshot

**Entry Points:**
- `ragctl` Typer CLI (doctor/ingest/query/chat)
- FastAPI server (`clockify_rag.api:app`)
- Automation scripts (`scripts/smoke_rag.py`, `eval.py`)
- Legacy: `clockify_support_cli_final.py` (wraps same modules)

**Central Configuration:**
- `clockify_rag/config.py` loads defaults, `.env`, and runtime overrides
- All URLs, model names, timeouts, and artifact paths flow through this module
- See [docs/CONFIGURATION.md](CONFIGURATION.md) for complete reference

**Artifacts:**
- `chunks.jsonl`, `vecs_n.npy`, `bm25.json`, `faiss.index`, `index.meta.json`
- Live beside the repository, rebuilt deterministically via `ragctl ingest`

**External Services:**
- Ollama-compatible endpoint at `http://10.127.0.192:11434` (default, overridable)
- Local filesystem for corpus Markdown and index artifacts
- Optional FAISS/HNSW libraries for ANN retrieval (falls back to BM25-only when unavailable)

## High-Level Responsibilities

| Stage | Responsibility | Key Modules / Files |
|-------|----------------|---------------------|
| **Ingestion & Normalization** | Convert Markdown/HTML/PDF/txt/docx sources into normalized Markdown following `# [ARTICLE]` convention | `clockify_rag/ingestion.py`, `knowledge_full.md`, [docs/internals/INGESTION.md](internals/INGESTION.md) |
| **Chunking** | Parse Markdown articles, split by headings/sentences with overlap, emit normalized chunks with stable IDs | `clockify_rag/chunking.py`, [docs/internals/CHUNKING.md](internals/CHUNKING.md) |
| **Embedding Layer** | Produce semantic vectors using local SentenceTransformers (default) or Ollama endpoint. Handles batching, retries, caching | `clockify_rag/embedding.py`, `emb_cache.jsonl` |
| **Vector & Lexical Indexes** | Maintain FAISS IVFFlat (primary), HNSW (fallback), and BM25 sparse indexes | `clockify_rag/indexing.py`, index artifacts |
| **Retriever & Reranker** | Execute BM25 + dense dual retrieval, reciprocal-rank fusion, intent-aware weighting, optional LLM reranking, MMR diversification | `clockify_rag/retrieval.py`, `clockify_rag/intent_classification.py` |
| **Answer Orchestration** | Drive end-to-end `answer_once` flow: validation, caching, retrieval, reranking, prompt construction, LLM call, citation validation | `clockify_rag/answer.py`, `clockify_rag/caching.py`, `clockify_rag/error_handlers.py` |
| **LLM Client Abstraction** | Unified Ollama client with retries/timeouts plus deterministic mock for tests/CI (`RAG_LLM_CLIENT=mock`) | `clockify_rag/api_client.py` |
| **APIs & CLIs** | Expose FastAPI endpoints, Typer-powered CLI (`ragctl`), and legacy wrappers | `clockify_rag/api.py`, `clockify_rag/cli_modern.py`, `clockify_support_cli_final.py`, `Makefile` |
| **Observability** | Structured logging plus in-process metrics (counters, gauges, histograms) | `clockify_rag/logging_config.py`, `clockify_rag/metrics.py`, [docs/internals/LOGGING_CONFIG.md](internals/LOGGING_CONFIG.md) |
| **Evaluation** | Offline retrieval evaluation via `eval.py`, RAGAS integration, datasets under `eval_datasets/` | `eval.py`, [docs/EVALUATION.md](EVALUATION.md) |

## Data Flow

### High-Level Pipeline

```
Source Docs (Markdown/HTML/PDF)
        ↓
   Ingestion (normalize to Markdown)
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

### Detailed Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query (CLI/API)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Query Caching Layer   │◄──Cache Hit? Return
            │  (LRU + Disk)          │
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
         │ Hybrid Fusion (RRF)  │
         │ alpha * bm25 +       │
         │ (1-alpha) * dense    │
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

### Execution Flow (Single Query)

1. User enters question
2. Check query cache → hit → return cached answer
3. Embed question with SentenceTransformer
4. **Parallel:**
   - BM25 retrieval (top 15 chunks by keywords)
   - FAISS/HNSW retrieval (top 15 chunks by vector similarity)
5. Fuse results (reciprocal rank fusion with alpha weighting)
6. Apply MMR (remove redundant chunks)
7. Filter by threshold (≥0.25 similarity)
8. Pack snippets (format + estimate tokens)
9. Call LLM with context window ≤12000 tokens
10. Extract citations from generated answer
11. Score confidence
12. Cache result
13. Log query (optional)
14. Return answer + sources + confidence

## Component Details

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
- **Parameters**: `nlist=64` (M1 stability) or `256` (x86), `nprobe=16`
- **Speed**: <10ms per query
- **Memory**: ~100MB for 1000 chunks
- **Files**: `faiss.index`

#### HNSW (Fallback)
- **Type**: Hierarchical Navigable Small World
- **Parameters**: `m=16`, `ef_construction=200`
- **Speed**: 20-50ms per query
- **Files**: `hnsw_cosine.bin`

#### BM25 (Last Resort)
- **Type**: Okapi BM25 (classic TF-IDF variant)
- **Parameters**: `k1=1.2`, `b=0.65`
- **Speed**: 50-200ms per query
- **Memory**: <10MB
- **Files**: `bm25.json`

### 4. Query Processing

**File**: `clockify_rag/retrieval.py`

**Steps**:

1. **Query Embedding**: Embed question with same model as chunks → 768-dim vector
2. **Dual Retrieval**:
   - BM25: Keyword matching → top_k results
   - Dense: Vector similarity → top_k results
3. **Hybrid Fusion** (Reciprocal Rank Fusion):
   ```
   score = 1/(1 + bm25_rank) + 1/(1 + dense_rank)
   weighted = alpha * bm25_score + (1 - alpha) * dense_score
   ```
4. **MMR Reranking** (Maximal Marginal Relevance):
   - Balance relevance vs diversity
   - Formula: `mmr_score = lambda * relevance - (1 - lambda) * similarity_to_selected`
5. **Filtering**:
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
  - **Automatic Fallback**: `gpt-oss:20b` on connection/timeout/5xx errors

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

## External Services

**Ollama-compatible LLM host** (default: `http://10.127.0.192:11434`):
- Chat/generation model: `qwen2.5:32b`
- Embedding model: `nomic-embed-text:latest`
- Fallback model: `gpt-oss:20b` (automatic on primary failure)
- Accessible from company VPN (required for remote endpoint)
- All network clients handle timeouts, retries, and are mockable for offline testing

**Local storage**:
- Vector artifacts (FAISS/HNSW/BM25)
- Chunk metadata (`chunks.jsonl`, `meta.jsonl`)
- Logs (`logs/`)
- Caches (`emb_cache.jsonl`, `rag_queries.jsonl`)

**Python runtime** (3.11+):
- Optional Apple Silicon acceleration (PyTorch MPS)
- FAISS wheels (conda for M1)
- Docker support: linux/amd64 and linux/arm64

## Environment Assumptions

**Development**: macOS M1 Pro laptops with VPN access to remote Ollama host. Local runs default to mock LLM clients; setting `RAG_OLLAMA_URL` switches to real host.

**Production**: Linux containers (amd64) managed by platform team. Containers mount persistent storage for indexes and expose FastAPI on port 8000.

**Offline/CI**: Must succeed without network access. All tests and evaluation scripts default to mock LLM client and deterministic embeddings.

## Artifacts & Storage

| Artifact | Purpose | Generated by | Size (per 100 chunks) |
|----------|---------|--------------|---------------------|
| `chunks.jsonl` / `meta.jsonl` | Chunk metadata and helper fields | `clockify_rag.chunking.build_chunks` | 1-2 MB |
| `vecs_n.npy` (`float32`) | Normalized dense embeddings | `clockify_rag.embedding.embed_texts` | 300 KB |
| `bm25.json` | Sparse keyword index | `clockify_rag.indexing.build_bm25` | 100-200 KB |
| `faiss.index` / `hnsw_cosine.bin` | ANN indexes for dense retrieval | `clockify_rag.indexing.build_faiss_index` | 1-2 MB |
| `index.meta.json` | Versioning and checksum info | `clockify_rag.indexing.save_index_meta` | <1 KB |
| `rag_queries.jsonl` | Structured query logs (redaction-aware) | `clockify_rag.caching.log_query` | Variable |
| `logs/` | General application logs (JSON/text) | `clockify_rag.logging_config` | Variable |

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

### Scalability

**Maximum Practical Index Sizes:**
- **Small** (0-1000 chunks): All backends ✅
- **Medium** (1000-10000 chunks): FAISS + HNSW ✅, BM25 ✅
- **Large** (10000-100000 chunks): FAISS ✅, HNSW ⚠️, BM25 ⚠️
- **XLarge** (>100000 chunks): FAISS only (with sharding)

**Optimization for Scale:**
1. Shard index by category/domain
2. Use cloud FAISS (Vespa, Elasticsearch, Pinecone)
3. Async embeddings in batches
4. Distributed LLM (vLLM, Ray Serve)
5. Redis for multi-instance caching

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

See [docs/PLUGIN_GUIDE.md](PLUGIN_GUIDE.md) for complete plugin documentation.

## Testing Strategy

### Unit Tests (`tests/`)
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

See [docs/TESTING.md](TESTING.md) for complete testing guide.

## Deployment & Operations

**Docker Compose**: `docker-compose.yml` launches FastAPI server plus optional local Ollama profile

**Dockerfile**: Multi-stage production image (installs package, copies artifacts, exposes uvicorn)

**Runbooks**: [docs/RUNBOOK.md](RUNBOOK.md) covers health checks, log paths, rebuild steps, connectivity verification

**Platform Notes**: [docs/DEPLOYMENT.md](DEPLOYMENT.md) captures Apple Silicon vs linux/amd64 specifics, environment variables, mock vs real LLM clients

---

**Related Documentation:**
- [README.md](../README.md) - Quick start and overview
- [CONFIGURATION.md](CONFIGURATION.md) - Complete config reference
- [RUNBOOK.md](RUNBOOK.md) - Operations guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment

**Keep this document updated** when architecture evolves (new ingestion sources, retrievers, or evaluation tooling). Changes that add new dependencies or external touchpoints must be reflected here before merging to `main`.
