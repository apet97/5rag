# HTTP API Reference

The FastAPI server (default `uvicorn clockify_rag.api:app --port 8000`) exposes versioned endpoints that mirror the CLI behaviour. All responses are JSON and include structured metadata and routing hints, so clients can automate escalation workflows without parsing logs.

## Authentication
None. Run the service behind your own network controls (VPN, ingress proxy, etc.).

## Endpoints

### `GET /health` and `GET /v1/health`
Quick readiness probe.
```json
{
  "status": "healthy",
  "timestamp": "2025-11-10T12:34:45.512345",
  "version": "5.9.1",
  "platform": "Linux x86_64",
  "index_ready": true,
  "ollama_connected": true
}
```
`status` can be `healthy`, `degraded` (index loaded but Ollama unreachable), or `unavailable`.

### `GET /v1/config`
Returns the resolved runtime configuration (mainly for debugging):
```json
{
  "ollama_url": "http://10.127.0.192:11434",
  "gen_model": "qwen2.5:32b",
  "emb_model": "nomic-embed-text:latest",
  "chunk_size": 1600,
  "top_k": 15,
  "pack_top": 8,
  "threshold": 0.25
}
```

### `POST /v1/query`
Primary query endpoint. Body schema:
```json
{
  "question": "How do I track time in Clockify?",
  "top_k": 15,
  "pack_top": 8,
  "threshold": 0.25,
  "debug": false
}
```
Response schema (all fields documented so CLI + API stay in sync):
```json
{
  "question": "How do I track time in Clockify?",
  "answer": "Step-by-step ... [chunk ids]",
  "confidence": 82,
  "sources": [12, 48, 109],
  "timestamp": "2025-11-10T13:05:22.301245",
  "processing_time_ms": 912.4,
  "refused": false,
  "metadata": {
    "retrieval_count": 15,
    "packed_count": 8,
    "used_tokens": 4875,
    "rerank_applied": false
  },
  "routing": {
    "action": "auto_approve",
    "level": "high",
    "confidence": 82,
    "escalated": false,
    "reason": "confidence >= 70"
  },
  "timing": {
    "total_ms": 912.4,
    "retrieve_ms": 210.2,
    "mmr_ms": 4.8,
    "rerank_ms": 0.0,
    "llm_ms": 674.3
  }
}
```
On coverage failures or LLM errors the `answer` will contain the refusal string (`"I don't know based on the MD."`) and `metadata.llm_error` / `metadata.coverage_check` describe the reason.

### `POST /v1/ingest`
Starts an asynchronous index rebuild. The body accepts `input_file` and `force` flags. Response:
```json
{
  "status": "processing",
  "message": "Index build started in background from knowledge_full.md",
  "timestamp": "2025-11-10T13:12:00.113423",
  "index_ready": false
}
```
Monitor `/health` for completion.

### `GET /v1/metrics`
Lightweight JSON metrics dump – useful until an external scraper is attached.
```json
{
  "timestamp": "2025-11-10T13:14:11.551",
  "index_ready": true,
  "chunks_loaded": 3821
}
```
For richer insights, consume the structured logs (`rag.query.start`, `rag.query.complete`, `rag.query.failure`) written to `$RAG_LOG_FILE`.

## CLI parity
`ragctl query --json` and `clockify_support_cli_final.py ask --json` return the same payload as `/v1/query`, so scripts can switch between the CLI and HTTP API without adapting downstream tooling.
