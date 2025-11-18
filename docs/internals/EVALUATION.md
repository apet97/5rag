# Evaluation Guide

This repository ships with a self-contained harness for measuring retrieval quality and (optionally) end-to-end LLM answers. The harness is entirely offline by default so CI and contributors without VPN access can run it safely.

## Datasets

- Default dataset: `eval_datasets/clockify_v1.jsonl`
  - Fields: `query`, `difficulty`, `tags`, `notes`, `relevant_chunks`
  - Relevant chunks are resolved by title/section, so you do not need stable chunk IDs.
- Add additional datasets to `eval_datasets/` (JSONL with one example per line).

## Offline Retrieval Evaluation

```bash
python eval.py --dataset eval_datasets/clockify_v1.jsonl
```

What it does:
- Loads the hybrid index if available (`chunks.jsonl`, `vecs_n.npy`, `bm25.json`, optional `faiss.index`).
- Falls back to a lexical BM25 retriever if the hybrid artifacts are missing (useful for CI).
- Computes MRR@10, Precision@5, and NDCG@10.
- Fails the build if metrics drop below the thresholds specified via `--min-*` flags.

This command requires **no LLM connectivity** and is what CI runs by default.

## LLM Answer Report (optional)

When you want to inspect actual answers, enable the answer pass:

```bash
# Offline / CI-safe (uses the deterministic mock client)
RAG_LLM_CLIENT=mock python eval.py --dataset eval_datasets/clockify_v1.jsonl --llm-report

# Real Ollama host (VPN)
RAG_OLLAMA_URL=http://10.127.0.192:11434 python eval.py \
    --dataset eval_datasets/clockify_v1.jsonl \
    --llm-report \
    --llm-output eval_reports/qwen_answers.jsonl
```

Notes:
- `--llm-report` reuses the regular evaluation dataset but also runs `answer_once` for each query.
- Results are saved (JSONL) to `eval_reports/llm_answers.jsonl` by default. Each line contains the query, answer, confidence, refusal flag, and metadata/citations.
- The mock client is used automatically when `RAG_LLM_CLIENT=mock` (recommended for CI / smoke tests).
- Point to the production Ollama host by setting `RAG_OLLAMA_URL` / `RAG_CHAT_MODEL` / `RAG_EMBED_MODEL`.

## One-Command Summary

- **Retrieval only (CI):** `python eval.py --dataset eval_datasets/clockify_v1.jsonl`
- **Retrieval + LLM (manual review):** `python eval.py --dataset eval_datasets/clockify_v1.jsonl --llm-report`

See the console output for metric interpretations and the generated report path (when `--llm-report` is enabled).
