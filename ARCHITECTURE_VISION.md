# Architecture Vision - RAG Tool Roadmap

**Document Purpose**: Long-term architectural improvements and strategic roadmap for evolving from monolithic prototype to production-grade modular system.

**Timeline**: 6-12 months

**Goal**: Transform Clockify RAG CLI from functional prototype to enterprise-grade retrieval system.

---

## Table of Contents

1. [Current State Assessment](#current-state-assessment)
2. [Target Architecture](#target-architecture)
3. [Modularization Plan](#modularization-plan)
4. [Plugin Architecture](#plugin-architecture)
5. [API Design](#api-design)
6. [Scaling Strategy](#scaling-strategy)
7. [Advanced Features Roadmap](#advanced-features-roadmap)
8. [Migration Path](#migration-path)

---

## Current State Assessment

### Architecture Overview (v4.1)

```
┌──────────────────────────────────────────────────────────┐
│ clockify_support_cli_final.py (2000+ lines)              │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ CLI Entry Point (main, argparse)                   │ │
│  └────────────────────────────────────────────────────┘ │
│                         ↓                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Build Pipeline                                     │ │
│  │ - Chunking                                         │ │
│  │ - Embedding (local or Ollama)                      │ │
│  │ - BM25 indexing                                    │ │
│  │ - FAISS indexing (optional)                        │ │
│  └────────────────────────────────────────────────────┘ │
│                         ↓                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Query Pipeline (answer_once)                       │ │
│  │ - Embed query                                      │ │
│  │ - Hybrid retrieval (BM25 + dense + MMR)            │ │
│  │ - Optional reranking                               │ │
│  │ - Coverage check                                   │ │
│  │ - Snippet packing                                  │ │
│  │ - LLM generation                                   │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  Global State: EMB_BACKEND, USE_ANN, ALPHA_HYBRID       │
│  File I/O: Atomic writes, build lock                    │
└──────────────────────────────────────────────────────────┘
```

### Strengths
- ✅ Single-file simplicity (easy to deploy)
- ✅ Functional hybrid retrieval pipeline
- ✅ Good file I/O safety (atomic operations)
- ✅ ARM64/M1 compatibility

### Limitations
- ❌ Monolithic (2000+ LOC, hard to test/modify)
- ❌ Tight coupling (retrieval + packing + generation in one function)
- ❌ Global state mutation
- ❌ No plugin system
- ❌ CLI-only (no programmatic API)
- ❌ Single-process (cannot scale to multiple workers)

---

## Target Architecture

### Vision: Modular, Plugin-Based, API-First

```
┌─────────────────────────────────────────────────────────────────────┐
│                          API Layer                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ CLI (Click)  │  │ REST (FastAPI│  │ gRPC         │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       Core Library                                  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Pipeline Orchestrator                                        │  │
│  │  - ConfigManager                                             │  │
│  │  - PluginRegistry                                            │  │
│  │  - EventBus (for telemetry)                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                               ↓                                     │
│  ┌──────────────┬────────────┬────────────┬────────────────────┐  │
│  │  Chunkers    │ Embedders  │ Retrievers │ Generators         │  │
│  │              │            │            │                    │  │
│  │ - Markdown   │ - Local    │ - BM25     │ - Ollama           │  │
│  │ - Sentence   │ - Ollama   │ - Dense    │ - OpenAI           │  │
│  │ - Semantic   │ - OpenAI   │ - Hybrid   │ - Local (GGUF)     │  │
│  │              │            │ - DPR      │                    │  │
│  └──────────────┴────────────┴────────────┴────────────────────┘  │
│                               ↓                                     │
│  ┌──────────────┬────────────┬────────────┬────────────────────┐  │
│  │  Indexes     │ Rerankers  │ Packers    │ Evaluators         │  │
│  │              │            │            │                    │  │
│  │ - FAISS      │ - MMR      │ - Token    │ - MRR              │  │
│  │ - HNSW       │ - CrossEnc │ - Semantic │ - NDCG             │  │
│  │ - Pinecone   │ - LLM      │ - Diversity│ - BERTScore        │  │
│  └──────────────┴────────────┴────────────┴────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       Storage Layer                                 │
│  ┌──────────────┬────────────┬────────────┬────────────────────┐  │
│  │ Vector Store │ BM25 Index │ Metadata   │ Cache              │  │
│  │ (FAISS/local)│ (JSON/ES)  │ (SQLite)   │ (Redis/Local)      │  │
│  └──────────────┴────────────┴────────────┴────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Separation of Concerns**: Each component has single responsibility
2. **Plugin-Based**: Easy to swap implementations (embedder, retriever, etc.)
3. **Configuration-Driven**: No hardcoded parameters
4. **API-First**: Programmatic access + CLI + REST
5. **Testable**: Unit tests for every component
6. **Scalable**: Horizontal scaling via workers
7. **Observable**: Metrics, traces, logs

---

## Modularization Plan

### Phase 1: Extract Core Modules (Month 1-2)

#### Proposed Directory Structure

```
clockify_rag/
├── __init__.py
├── __version__.py
│
├── core/
│   ├── __init__.py
│   ├── config.py          # RAGConfig dataclass
│   ├── types.py           # Type definitions (Chunk, Document, Retrieval Result)
│   └── pipeline.py        # Pipeline orchestrator
│
├── chunkers/
│   ├── __init__.py
│   ├── base.py            # ChunkerBase abstract class
│   ├── markdown.py        # MarkdownChunker (current implementation)
│   ├── sentence.py        # SentenceChunker (NLTK-based)
│   └── semantic.py        # SemanticChunker (future: LLM-based)
│
├── embedders/
│   ├── __init__.py
│   ├── base.py            # EmbedderBase abstract class
│   ├── local.py           # LocalEmbedder (SentenceTransformers)
│   ├── ollama.py          # OllamaEmbedder
│   └── openai.py          # OpenAIEmbedder (future)
│
├── indexes/
│   ├── __init__.py
│   ├── base.py            # IndexBase abstract class
│   ├── bm25.py            # BM25Index
│   ├── faiss_index.py     # FAISSIndex
│   ├── hnsw.py            # HNSWIndex
│   └── pinecone.py        # PineconeIndex (future: cloud vector DB)
│
├── retrievers/
│   ├── __init__.py
│   ├── base.py            # RetrieverBase abstract class
│   ├── dense.py           # DenseRetriever (cosine similarity)
│   ├── sparse.py          # SparseRetriever (BM25)
│   ├── hybrid.py          # HybridRetriever (current implementation)
│   └── dpr.py             # DPRRetriever (future: Dense Passage Retrieval)
│
├── rerankers/
│   ├── __init__.py
│   ├── base.py            # RerankerBase abstract class
│   ├── mmr.py             # MMRReranker (current implementation)
│   ├── crossencoder.py    # CrossEncoderReranker
│   └── llm.py             # LLMReranker (current optional implementation)
│
├── packers/
│   ├── __init__.py
│   ├── base.py            # PackerBase abstract class
│   ├── token.py           # TokenBudgetPacker (current implementation)
│   ├── semantic.py        # SemanticPacker (future: cluster similar chunks)
│   └── diversity.py       # DiversityPacker (future: ensure diverse viewpoints)
│
├── generators/
│   ├── __init__.py
│   ├── base.py            # GeneratorBase abstract class
│   ├── ollama.py          # OllamaGenerator (current implementation)
│   ├── openai.py          # OpenAIGenerator (future)
│   └── local.py           # LocalGenerator (future: llama.cpp, GGUF)
│
├── evaluators/
│   ├── __init__.py
│   ├── retrieval.py       # MRR, NDCG, precision@k, recall@k
│   ├── generation.py      # BERTScore, ROUGE, BLEU
│   └── end_to_end.py      # Human evaluation, A/B testing
│
├── utils/
│   ├── __init__.py
│   ├── io.py              # File I/O (atomic writes, locks)
│   ├── logging.py         # Structured logging
│   ├── metrics.py         # Prometheus metrics
│   └── caching.py         # LRU cache, TTL cache
│
├── cli/
│   ├── __init__.py
│   ├── main.py            # Click CLI entry point
│   ├── build.py           # Build commands
│   ├── query.py           # Query commands
│   └── eval.py            # Evaluation commands
│
├── api/
│   ├── __init__.py
│   ├── rest.py            # FastAPI REST server
│   ├── grpc/              # gRPC service (future)
│   │   ├── __init__.py
│   │   ├── server.py
│   │   └── rag.proto
│   └── models.py          # Pydantic request/response models
│
└── tests/
    ├── __init__.py
    ├── unit/              # Unit tests per module
    │   ├── test_chunker.py
    │   ├── test_embedder.py
    │   ├── test_retriever.py
    │   └── ...
    ├── integration/       # Integration tests
    │   ├── test_pipeline.py
    │   └── test_api.py
    └── fixtures/          # Test data
        ├── test_kb.md
        └── ground_truth.jsonl
```

#### Migration Strategy

1. **Create new package structure** (Week 1)
   ```bash
   mkdir -p clockify_rag/{core,chunkers,embedders,indexes,retrievers,rerankers,packers,generators,evaluators,utils,cli,api,tests}
   ```

2. **Extract base classes** (Week 2)
   - Define abstract interfaces for each component
   - Document expected inputs/outputs
   - Add type hints

3. **Move current implementation** (Weeks 3-4)
   - Copy functions from `clockify_support_cli_final.py`
   - Adapt to new interfaces
   - Add unit tests as you go

4. **Create new CLI wrapper** (Week 5)
   - Use Click instead of argparse (better UX)
   - Import from `clockify_rag` package
   - Maintain backward compatibility

5. **Deprecate old file** (Week 6)
   - Add deprecation warning to `clockify_support_cli_final.py`
   - Redirect to new CLI
   - Update documentation

---

## Plugin Architecture

### Design: Registry Pattern + ABC

#### Plugin Registry

```python
# clockify_rag/core/registry.py
from typing import Dict, Type, TypeVar
from abc import ABC

T = TypeVar('T')

class PluginRegistry:
    """Registry for RAG components."""

    def __init__(self):
        self._embedders: Dict[str, Type] = {}
        self._retrievers: Dict[str, Type] = {}
        self._rerankers: Dict[str, Type] = {}
        self._generators: Dict[str, Type] = {}

    def register_embedder(self, name: str, cls: Type):
        """Register an embedder plugin."""
        self._embedders[name] = cls

    def get_embedder(self, name: str, **kwargs):
        """Instantiate embedder by name."""
        if name not in self._embedders:
            raise ValueError(f"Unknown embedder: {name}. Available: {list(self._embedders.keys())}")
        return self._embedders[name](**kwargs)

    # Similar for retriever, reranker, generator...

# Global registry
registry = PluginRegistry()
```

#### Plugin Example: Custom Embedder

```python
# my_custom_embedder.py
from clockify_rag.embedders.base import EmbedderBase
from clockify_rag.core.registry import registry
import numpy as np

class MyCustomEmbedder(EmbedderBase):
    """Custom embedder using BERT."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using BERT [CLS] token."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        # Use [CLS] token embedding
        embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
        return embeddings.astype("float32")

# Register plugin
registry.register_embedder("my_bert", MyCustomEmbedder)
```

#### Usage

```python
from clockify_rag.core.registry import registry
from clockify_rag.core.config import RAGConfig

# Option 1: Use built-in embedder
config = RAGConfig(embedder="local")
embedder = registry.get_embedder(config.embedder)

# Option 2: Use custom embedder
registry.register_embedder("my_bert", MyCustomEmbedder)
config = RAGConfig(embedder="my_bert")
embedder = registry.get_embedder(config.embedder, model_name="bert-large-uncased")

# Embed texts
embeddings = embedder.embed(["How do I track time?", "What are pricing plans?"])
```

---

## API Design

### REST API (FastAPI)

#### Endpoints

```python
# clockify_rag/api/rest.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from clockify_rag.core.pipeline import RAGPipeline

app = FastAPI(title="Clockify RAG API", version="2.0")
pipeline = None  # Lazy-loaded on startup

class QueryRequest(BaseModel):
    question: str
    top_k: int = 12
    pack_top: int = 6
    threshold: float = 0.30
    use_rerank: bool = False
    debug: bool = False

class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    metadata: dict

class BuildRequest(BaseModel):
    kb_path: str
    force_rebuild: bool = False

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Answer a question using RAG pipeline."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Call /build first.")

    try:
        answer, metadata = pipeline.answer(
            question=request.question,
            top_k=request.top_k,
            pack_top=request.pack_top,
            threshold=request.threshold,
            use_rerank=request.use_rerank
        )
        return QueryResponse(
            answer=answer,
            citations=metadata.get("selected", []),
            metadata=metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/build")
async def build(request: BuildRequest):
    """Build knowledge base from markdown file."""
    global pipeline
    try:
        pipeline = RAGPipeline.from_kb(request.kb_path, force_rebuild=request.force_rebuild)
        return {"status": "success", "chunks": len(pipeline.chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None
    }

@app.get("/api/v1/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest
    return generate_latest()
```

#### Client SDK

```python
# clockify_rag/client.py
import requests
from typing import Optional

class RAGClient:
    """Client SDK for Clockify RAG API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def query(self, question: str, top_k: int = 12, pack_top: int = 6,
              threshold: float = 0.30, use_rerank: bool = False) -> dict:
        """Ask a question."""
        response = requests.post(
            f"{self.base_url}/api/v1/query",
            json={
                "question": question,
                "top_k": top_k,
                "pack_top": pack_top,
                "threshold": threshold,
                "use_rerank": use_rerank
            }
        )
        response.raise_for_status()
        return response.json()

    def build(self, kb_path: str, force_rebuild: bool = False):
        """Build knowledge base."""
        response = requests.post(
            f"{self.base_url}/api/v1/build",
            json={"kb_path": kb_path, "force_rebuild": force_rebuild}
        )
        response.raise_for_status()
        return response.json()

# Usage
client = RAGClient("http://localhost:8000")
result = client.query("How do I track time?")
print(result["answer"])
```

---

## Scaling Strategy

### Horizontal Scaling Architecture

```
                          ┌─────────────┐
                          │ Load Balancer│
                          │  (nginx)    │
                          └──────┬──────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
                ↓                ↓                ↓
        ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
        │ API Server 1  │ │ API Server 2  │ │ API Server N  │
        │ (FastAPI)     │ │ (FastAPI)     │ │ (FastAPI)     │
        └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
                │                 │                 │
                └─────────────────┼─────────────────┘
                                  │
                          ┌───────┴────────┐
                          │                │
                          ↓                ↓
                  ┌───────────────┐ ┌───────────────┐
                  │ Redis Cache   │ │ Vector Store  │
                  │ (query cache) │ │ (FAISS/Pinecone)│
                  └───────────────┘ └───────────────┘
                          │
                          ↓
                  ┌───────────────┐
                  │ Metadata DB   │
                  │ (PostgreSQL)  │
                  └───────────────┘
```

### Distributed Indexing (for multi-GB KBs)

```python
# cloudify_rag/distributed/indexer.py
from celery import Celery
from clockify_rag.core.pipeline import RAGPipeline

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task
def build_chunk_batch(chunk_ids: list, kb_path: str):
    """Build embeddings for a batch of chunks."""
    pipeline = RAGPipeline.from_kb(kb_path)
    chunks = [pipeline.chunks[i] for i in chunk_ids]
    embeddings = pipeline.embedder.embed([c["text"] for c in chunks])
    return chunk_ids, embeddings.tolist()

# Usage: distribute across workers
from celery import group

# Partition chunks into batches
batch_size = 100
num_chunks = 10000
batches = [list(range(i, min(i + batch_size, num_chunks)))
           for i in range(0, num_chunks, batch_size)]

# Execute in parallel
job = group(build_chunk_batch.s(batch, "knowledge_full.md") for batch in batches)
result = job.apply_async()

# Collect results
embeddings_dict = {}
for chunk_ids, embs in result.get():
    for cid, emb in zip(chunk_ids, embs):
        embeddings_dict[cid] = emb
```

### Query Caching Strategy

```python
# clockify_rag/utils/caching.py
import redis
import hashlib
import json

class QueryCache:
    """Redis-based query cache with TTL."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl: int = 3600):
        self.client = redis.from_url(redis_url)
        self.ttl = ttl  # seconds

    def _key(self, question: str, params: dict) -> str:
        """Generate cache key from question and parameters."""
        canonical = json.dumps({"q": question, **params}, sort_keys=True)
        return f"rag:query:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"

    def get(self, question: str, params: dict) -> Optional[dict]:
        """Get cached result."""
        key = self._key(question, params)
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None

    def set(self, question: str, params: dict, result: dict):
        """Cache result with TTL."""
        key = self._key(question, params)
        self.client.setex(key, self.ttl, json.dumps(result))

# Usage in API
cache = QueryCache(ttl=3600)  # 1 hour

@app.post("/api/v1/query")
async def query(request: QueryRequest):
    params = {"top_k": request.top_k, "pack_top": request.pack_top, "threshold": request.threshold}

    # Check cache
    cached = cache.get(request.question, params)
    if cached:
        return QueryResponse(**cached)

    # Compute
    answer, metadata = pipeline.answer(...)

    # Cache result
    result = {"answer": answer, "citations": metadata.get("selected", []), "metadata": metadata}
    cache.set(request.question, params, result)

    return QueryResponse(**result)
```

---

## Advanced Features Roadmap

### Phase 2: Advanced RAG (Month 3-4)

#### 1. Multi-Modal Support (Images, Tables, Code)

**Vision**: Handle images, tables, code snippets in documentation.

**Implementation**:
- **Image embedding**: CLIP for image-text matching
- **Table parsing**: Extract tables to JSON, embed structured data
- **Code embedding**: CodeBERT for code snippets

```python
# clockify_rag/chunkers/multimodal.py
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

class MultiModalChunker(ChunkerBase):
    """Chunker that extracts text, images, tables, code."""

    def chunk(self, document: str) -> List[Chunk]:
        chunks = []

        # Extract markdown sections
        text_chunks = self._extract_text(document)
        chunks.extend(text_chunks)

        # Extract images
        image_chunks = self._extract_images(document)
        chunks.extend(image_chunks)

        # Extract tables
        table_chunks = self._extract_tables(document)
        chunks.extend(table_chunks)

        # Extract code blocks
        code_chunks = self._extract_code(document)
        chunks.extend(code_chunks)

        return chunks

    def _extract_images(self, document: str) -> List[Chunk]:
        """Extract images from markdown ![alt](path)."""
        import re
        pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
        matches = re.findall(pattern, document)

        chunks = []
        for alt_text, image_path in matches:
            chunks.append(Chunk(
                id=str(uuid.uuid4()),
                type="image",
                text=alt_text,  # Alt text for text search
                metadata={"image_path": image_path}
            ))
        return chunks
```

#### 2. Conversational RAG (Multi-Turn)

**Vision**: Support follow-up questions with conversation history.

**Implementation**:
- Store conversation context (last N turns)
- Rephrase follow-up questions with context
- Attribute answers to conversation turn

```python
# clockify_rag/generators/conversational.py
class ConversationalGenerator(GeneratorBase):
    """Generator with conversation history."""

    def __init__(self, base_generator: GeneratorBase, max_history: int = 3):
        self.base_generator = base_generator
        self.history = []
        self.max_history = max_history

    def generate(self, question: str, context: str, **kwargs) -> str:
        """Generate answer with conversation history."""
        # Build prompt with history
        history_text = self._format_history()
        full_prompt = f"{history_text}\n\nCurrent question: {question}\n\nContext:\n{context}"

        # Generate
        answer = self.base_generator.generate(full_prompt, **kwargs)

        # Update history
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return answer

    def _format_history(self) -> str:
        """Format conversation history."""
        if not self.history:
            return ""
        lines = ["Conversation history:"]
        for i, turn in enumerate(self.history, 1):
            lines.append(f"Q{i}: {turn['question']}")
            lines.append(f"A{i}: {turn['answer']}")
        return "\n".join(lines)
```

#### 3. Query Decomposition (Multi-Hop)

**Vision**: Break complex questions into sub-questions, answer each, synthesize.

**Implementation**:
- LLM decomposes question → [sub-question-1, sub-question-2, ...]
- Retrieve for each sub-question
- Synthesize final answer from sub-answers

```python
# clockify_rag/retrievers/multihop.py
class MultiHopRetriever(RetrieverBase):
    """Retriever that decomposes complex questions."""

    def __init__(self, base_retriever: RetrieverBase, decomposer_llm):
        self.base_retriever = base_retriever
        self.decomposer_llm = decomposer_llm

    def retrieve(self, question: str, top_k: int = 12) -> List[Chunk]:
        """Decompose question, retrieve for each sub-question."""
        # Decompose
        sub_questions = self._decompose(question)

        # Retrieve for each
        all_chunks = []
        for sub_q in sub_questions:
            chunks = self.base_retriever.retrieve(sub_q, top_k=top_k // len(sub_questions))
            all_chunks.extend(chunks)

        # Deduplicate and rank
        unique_chunks = {c.id: c for c in all_chunks}.values()
        return sorted(unique_chunks, key=lambda c: c.score, reverse=True)[:top_k]

    def _decompose(self, question: str) -> List[str]:
        """Decompose complex question into sub-questions."""
        prompt = f"Break this question into 2-3 simpler sub-questions:\\n{question}\\n\\nSub-questions (one per line):"
        response = self.decomposer_llm.generate(prompt, max_tokens=200)
        return [line.strip() for line in response.split("\\n") if line.strip()]
```

### Phase 3: Enterprise Features (Month 5-6)

#### 1. Access Control & Multi-Tenancy

```python
# clockify_rag/core/auth.py
from functools import wraps
from fastapi import Header, HTTPException

class AccessControl:
    """Role-based access control for RAG API."""

    def __init__(self):
        self.user_permissions = {}  # user_id → [permissions]

    def require_permission(self, permission: str):
        """Decorator to require permission for endpoint."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                user_id = kwargs.get("user_id")  # From auth token
                if not self._has_permission(user_id, permission):
                    raise HTTPException(status_code=403, detail="Forbidden")
                return await func(*args, **kwargs)
            return wrapper
        return decorator

    def _has_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has permission."""
        return permission in self.user_permissions.get(user_id, [])

# Usage
ac = AccessControl()

@app.post("/api/v1/build")
@ac.require_permission("build:write")
async def build(request: BuildRequest, user_id: str = Header(None)):
    ...
```

#### 2. Audit Logging & Compliance

```python
# clockify_rag/utils/audit.py
import logging
import json
from datetime import datetime

class AuditLogger:
    """Structured audit logger for compliance (GDPR, SOC2, HIPAA)."""

    def __init__(self, log_file: str = "audit.jsonl"):
        self.log_file = log_file
        self.logger = logging.getLogger("audit")

    def log_query(self, user_id: str, question: str, answer: str, citations: List[str]):
        """Log query event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "query",
            "user_id": user_id,
            "question_hash": hashlib.sha256(question.encode()).hexdigest()[:16],
            "answer_hash": hashlib.sha256(answer.encode()).hexdigest()[:16],
            "citations": citations,
            "ip_address": self._get_client_ip()
        }
        self._write_event(event)

    def log_build(self, user_id: str, kb_path: str, chunks_count: int):
        """Log build event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "build",
            "user_id": user_id,
            "kb_path": kb_path,
            "chunks_count": chunks_count
        }
        self._write_event(event)

    def _write_event(self, event: dict):
        """Append event to audit log (JSONL format)."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\\n")
```

---

## Migration Path

### Incremental Migration Strategy (6 months)

| Month | Phase | Deliverables | Backward Compat |
|-------|-------|--------------|-----------------|
| **1** | **Foundation** | - Create package structure<br>- Extract base classes<br>- Setup CI/CD | ✅ Old CLI still works |
| **2** | **Core Migration** | - Migrate chunker, embedder, retriever<br>- Add unit tests (50% coverage)<br>- New CLI (Click) | ✅ Old CLI deprecated but functional |
| **3** | **Advanced RAG** | - Add cross-encoder reranking<br>- Add evaluation framework<br>- Tune parameters | ✅ Old CLI redirects to new |
| **4** | **API Layer** | - FastAPI REST API<br>- Client SDK<br>- OpenAPI docs | 🔄 Breaking change (new API) |
| **5** | **Scaling** | - Redis caching<br>- Celery workers<br>- Prometheus metrics | 🔄 Requires infrastructure |
| **6** | **Enterprise** | - Access control<br>- Audit logging<br>- Multi-modal support | 🔄 Enterprise features |

### Backward Compatibility Plan

1. **Months 1-2**: Old CLI (`clockify_support_cli_final.py`) still works
2. **Month 3**: Old CLI shows deprecation warning, redirects to new CLI
3. **Month 4**: Old CLI removed, only package API available

**Migration Script**:
```bash
#!/bin/bash
# migrate.sh - Helper script for users

echo "Migrating to Clockify RAG v2.0..."

# Backup old config
cp clockify_support_cli_final.py clockify_support_cli_final.py.bak

# Install new package
pip install --upgrade clockify-rag

# Migrate config
python -m clockify_rag.migrate --from-cli --config config.yaml

# Test
python -m clockify_rag build knowledge_full.md
python -m clockify_rag query "How do I track time?"

echo "Migration complete! Old CLI backed up to clockify_support_cli_final.py.bak"
```

---

## Success Metrics

### KPIs for Architectural Improvements

| Metric | Current (v4.1) | Target (v5.0) | How to Measure |
|--------|----------------|---------------|----------------|
| **Code Coverage** | 0% (no unit tests) | 80% | pytest --cov |
| **API Latency** | N/A (no API) | <500ms p95 | Prometheus |
| **Build Time** | ~30s (384 chunks) | <15s (with cache) | time build |
| **Retrieval Accuracy** | Unknown (no eval) | MRR@10 > 0.75 | Evaluation framework |
| **Deployment Time** | Manual (10 steps) | <5 min (automated) | CI/CD pipeline |
| **Horizontal Scale** | 1 process | 10+ workers | Load testing |

---

## Conclusion

This architecture vision transforms the Clockify RAG CLI from a **monolithic prototype** to an **enterprise-grade modular system** over 6 months. Key improvements:

1. **Modularization**: Single-responsibility components, testable, maintainable
2. **Plugin Architecture**: Easy to swap implementations, extensible
3. **API-First**: Programmatic access enables integration with other systems
4. **Scalable**: Horizontal scaling with Redis cache and Celery workers
5. **Observable**: Metrics, traces, audit logs for production monitoring
6. **Enterprise-Ready**: Access control, multi-tenancy, compliance

**Next Steps**:
1. Review and approve architecture vision
2. Create detailed implementation plan (sprints, tickets)
3. Start with Phase 1: Modularization (Month 1-2)
4. Iteratively deliver value while maintaining backward compatibility

---

**End of Architecture Vision**
