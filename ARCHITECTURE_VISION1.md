# Architecture Vision

## Modularization Roadmap
1. **Core package split**  
   - Create `clockify_rag/core/` with submodules `retrieval`, `context`, `evaluation`, `io`.  
   - Move shared logic from `clockify_support_cli_final.py` into core modules.  
   - Provide stable typed interfaces (dataclasses for chunks, retrieval scores).
2. **Thin CLI layer**  
   - Replace monolithic CLI with lightweight wrapper importing from `clockify_rag`.  
   - Implement Click/Typer-based CLI for better UX while preserving current argparse flags.  
   - Ensure CLI only handles argument parsing, environment config, and output formatting.
3. **Service-ready API**  
   - Expose `clockify_rag.api` with `build_index`, `load_index`, `answer_query`, supporting synchronous and streaming responses.  
   - Add FastAPI reference implementation using same core modules.

## Plugin & Extension Architecture
- **Plugin discovery:** Use `importlib.metadata.entry_points(group="clockify_rag.plugins")` to auto-load retrievers/rerankers.  
- **Validation hooks:** Each plugin exposes `validate()` returning diagnostics aggregated in startup logs.  
- **Configuration:** Allow YAML/JSON config mapping plugin names to pipeline stages.  
- **Fallback ordering:** Provide default hybrid retrieval, but allow overriding dense retriever, rerank model, or citation formatter individually.

## API Design
- **Data models:**
  ```python
  @dataclass
  class Chunk:
      id: str
      title: str
      section: str
      url: str
      text: str
  
  @dataclass
  class RetrievalResult:
      chunk: Chunk
      scores: dict[str, float]
  
  @dataclass
  class Answer:
      text: str
      citations: list[str]
      confidence: int
      diagnostics: dict[str, Any]
  ```
- **Primary functions:**
  - `build_index(source: Path, *, cache: bool = True) -> IndexArtifacts`
  - `retrieve(question: str, artifacts: IndexArtifacts, *, top_k: int) -> list[RetrievalResult]`
  - `generate_answer(question: str, context: list[Chunk], *, model_cfg: ModelConfig) -> Answer`

## Scaling Strategy
1. **Distributed indexing**  
   - Partition knowledge base by namespace, build FAISS indexes per shard, store metadata in SQLite.  
   - Use multiprocessing to parallelize BM25 tokenization.
2. **Query caching & serving**  
   - Introduce Redis-backed cache for `answer_once` results and embedding vectors.  
   - Implement TTL eviction aligned with RateLimiter to prevent hammering.
3. **Streaming responses**  
   - Support event-stream output from LLMs; propagate partial answers via generator interface.
4. **Observability**  
   - Emit OpenTelemetry spans for build/retrieve/ask.  
   - Aggregate KPI metrics and retrieval profile stats in Prometheus.

## Long-Term Enhancements
- **Cross-encoder reranking:** Add optional ColBERT / bge-reranker integration with batching and caching.  
- **Multilingual support:** Normalize Unicode, add language detection, route to language-specific embeddings.  
- **Metadata enrichment:** Attach creation timestamps, provenance, and snippet confidence to each chunk.  
- **Evaluation suite:** Automate nightly run producing leaderboard of metrics stored in JSON for regression detection.  
- **Knowledge lifecycle:** Track `kb_sha` and diff knowledge base snapshots, triggering incremental rebuilds.
## Goals
- Deliver a modular, testable Clockify RAG platform with clear separation between reusable library code and thin CLIs.
- Guarantee deterministic builds and reproducible retrieval behaviour across environments.
- Provide hooks for experimentation (plugins, rerankers, evaluation pipelines) without destabilizing the production path.
- Support future scaling requirements (sharded indexes, distributed evaluation, multi-model backends).

## Target State Overview
```
clockify_rag/
  config.py          # Typed configuration objects & env parsing
  io/                # Atomic file helpers, hashing, locking
  chunking.py        # Sentence-aware parsing (library only)
  embeddings/
    local.py         # SentenceTransformer backend
    remote.py        # Ollama/HTTP backend
  retrieval/
    hybrid.py        # Shared hybrid retrieval logic
    ann.py           # FAISS/HNSW adapters
    packing.py       # Context packing utilities
    prompts.py       # System & rerank prompts with templating
  caching.py         # Query cache & rate limiter
  http.py            # Session factory & retry policies
  plugins/           # Registry + discovery (entry points)
  evaluation/        # Offline metrics, dataset loaders
cli/
  support.py         # Thin wrapper around library (answer_once, chat loop)
  build.py           # Build command delegating to clockify_rag.indexing
  benchmark.py       # CLI entry point for benchmark harness
```
- All modules expose dataclasses or typed protocol interfaces to simplify testing and dependency injection.
- Plugin registration uses Python entry points or YAML manifests so first-party and third-party retrievers share the same contract.

## Modularization Plan
1. **Phase 1 – Decompose monolithic CLI (week 1)**
   - Move reusable classes (QueryCache, RateLimiter, HTTP helpers, DenseScoreStore, packing) into `clockify_rag` modules.
   - Introduce `cli/` package containing minimal glue code that imports from the library.
   - Add integration tests ensuring CLI and library share behaviour.
2. **Phase 2 – Typed configuration & dependency injection (week 2)**
   - Replace global constants with `Config` dataclass loaded via `pydantic` or `attrs` and injected where needed.
   - Centralize logging setup and KPI instrumentation in `clockify_rag.telemetry`.
3. **Phase 3 – Retrieval pipeline hardening (weeks 3-4)**
   - Rebuild hybrid retrieval using shared components: query expansion, dense search, sparse search, reranking pipeline.
   - Expose pipeline as `HybridRetriever` class with overridable strategies; wire plugin registry to allow drop-in replacements.
4. **Phase 4 – Evaluation & benchmarking (week 5)**
   - Finalize `clockify_rag.evaluation` with offline metrics, golden datasets, and CLI commands (`rag eval`, `rag bench`).
   - Integrate evaluation into CI and nightly runs with S3 artifact upload.
5. **Phase 5 – Scaling & multi-tenancy (week 6+)**
   - Introduce index manifest format supporting shards/partitions.
   - Provide read-only gRPC/REST service for retrieval to serve multiple clients.

## Plugin & Experimentation Strategy
- Define plugin entry points for `RetrieverPlugin`, `RerankPlugin`, `EmbeddingPlugin`, and `IndexPlugin` (already scaffolded).
- Add `plugins.yml` manifest allowing ops teams to enable/disable plugins per deployment without code changes.
- Provide sandbox harness (`rag plugins test`) that loads plugins in isolation with sample questions.
- Version prompts separately (`prompts/system_v1.json`, etc.) and allow plugin-provided prompt modifiers.

## API & Surface Design
- Primary interface: `clockify_rag.retrieval.answer(question: str, *, config: Config) -> AnswerResult` returning dataclass with answer, citations, debug metadata.
- CLI wrappers (`clockify-support chat`, `clockify-support build`) call into library functions and stream structured JSON events.
- Future API service exposes `/retrieve` and `/answer` endpoints (FastAPI) with optional streaming responses.
- Provide Python SDK for automation partners (thin wrapper around HTTP API).

## Scaling Strategy
- **Index build**: parallelize embedding generation using process pool; persist embeddings via memory-mapped float16 arrays with checksum manifest.
- **Query serving**:
  - Warm FAISS/HNSW indexes at startup and expose async retrieval function.
  - Add Redis-backed query cache for multi-instance deployments.
  - Support configurable ANN backends (FAISS, ScaNN, managed vector DB) behind common interface.
- **Monitoring**: emit structured metrics (Prometheus/OpenTelemetry) for retrieval latency, cache hit rate, refusal rate.

## Roadmap Milestones
| Milestone | Description | Owner | Target |
|-----------|-------------|-------|--------|
| M1 | CLI/library separation complete, tests green | RAG Core | 2 weeks |
| M2 | Evaluation harness + CI gate live | RAG Quality | 4 weeks |
| M3 | Plugin manifest + external retriever demo | Platform | 6 weeks |
| M4 | Hosted retrieval API alpha (single node) | Infra | 8 weeks |
| M5 | Sharded index + Redis cache support | Infra | 12 weeks |

## Risk Mitigation
- Introduce static analysis (ruff, mypy) in CI to catch syntax regressions like the FAISS bug.
- Enforce code owners for shared modules to avoid ad-hoc duplication.
- Add pre-commit hook running lightweight integration tests before merging.

## Open Questions
- Should query expansion remain static JSON or move to learned expansion models? Evaluate once evaluation harness is stable.
- Decide between FastAPI and gRPC for service layer after benchmarking expected QPS.
- Determine storage location for large index artifacts (S3 vs internal object store) and how to version them.

## Next Steps
1. Approve modularization plan and assign owners.
2. Kick off Phase 1 branch focusing solely on CLI decomposition + bug fixes.
3. Stand up weekly architecture sync to track progress and unblock dependencies.
