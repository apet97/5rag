# Retrieval Tuning Guide

This guide explains how to validate, tune, and optimize retrieval quality in the RAG system.

## Table of Contents

1. [Understanding Retrieval](#understanding-retrieval)
2. [Evaluation Metrics](#evaluation-metrics)
3. [Key Configuration Parameters](#key-configuration-parameters)
4. [Running Retrieval Evaluation](#running-retrieval-evaluation)
5. [Parameter Tuning](#parameter-tuning)
6. [Troubleshooting Poor Retrieval](#troubleshooting-poor-retrieval)

---

## Understanding Retrieval

The RAG system uses **hybrid retrieval** combining:

1. **BM25 (Lexical)**: Token-based matching, good for exact keyword queries
2. **Dense (Semantic)**: Embedding-based similarity, good for conceptual queries
3. **Fusion**: Weighted combination of both scores

### Retrieval Pipeline

```
Query → Embed → [BM25 Score] + [FAISS/Dense Score] → Hybrid Fusion → MMR Diversification → Top-K
```

### Fusion Formula

```python
hybrid_score = alpha * bm25_score + (1 - alpha) * dense_score
```

- `alpha = 0.0`: Pure dense (semantic only)
- `alpha = 0.5`: Balanced hybrid (default)
- `alpha = 1.0`: Pure BM25 (lexical only)

---

## Evaluation Metrics

The system uses three metrics from `evaluation.py`:

### 1. MRR@10 (Mean Reciprocal Rank)
**What it measures**: Rank of first relevant result in top 10
**Interpretation**:
- `MRR = 1.0`: First result is always relevant
- `MRR = 0.5`: Relevant result at position 2 on average
- `MRR < 0.5`: Relevant results often outside top 3

**Success Threshold**: ≥ 0.70

### 2. Precision@5
**What it measures**: Fraction of top 5 results that are relevant
**Interpretation**:
- `P@5 = 0.8`: 4 out of 5 top results are relevant
- `P@5 = 0.4`: 2 out of 5 top results are relevant

**Success Threshold**: ≥ 0.35

### 3. NDCG@10 (Normalized Discounted Cumulative Gain)
**What it measures**: Position-aware ranking quality (earlier relevant results score higher)
**Interpretation**:
- `NDCG = 1.0`: Perfect ranking
- `NDCG = 0.6`: Reasonable ranking, some relevant results lower in list
- `NDCG < 0.5`: Poor ranking

**Success Threshold**: ≥ 0.60

---

## Key Configuration Parameters

### Environment Variables (config.py)

| Parameter | Default | Description | Impact on Retrieval |
|-----------|---------|-------------|---------------------|
| `RAG_TOP_K` | `15` | Candidates retrieved before reranking | Higher = more recall, slower |
| `RAG_PACK_TOP` | `8` | Final chunks sent to LLM | Higher = more context, slower LLM |
| `RAG_THRESHOLD` | `0.25` | Min similarity score to include | Higher = stricter, fewer results |
| `RAG_HYBRID_ALPHA` | `0.5` | BM25 vs dense weight (0-1) | `0.5` = balanced, `0.3` = more semantic |
| `RAG_MMR_LAMBDA` | `0.75` | Relevance vs diversity trade-off | Higher = more relevance, less diversity |
| `RAG_FAISS_MULTIPLIER` | `3` | FAISS candidate multiplier | `top_k * multiplier` candidates before rerank |

### Intent-Based Alpha Adjustment

The system automatically adjusts `alpha` based on query intent (see `retrieval.py:classify_intent`):

- **Procedural queries** ("How do I..."): `alpha = 0.65` (favor BM25 for step-by-step)
- **Factual queries** ("What is..."): `alpha = 0.35` (favor semantic for concepts)
- **Pricing queries** ("How much..."): `alpha = 0.70` (favor BM25 for specific info)
- **General queries**: `alpha = 0.50` (balanced)

---

## Running Retrieval Evaluation

### 1. Run Full Evaluation

```bash
ragctl eval --questions eval_datasets/clockify_v1.jsonl
```

**Output**:
```
📈 Evaluation Results
┌──────────────┬────────┬───────────┬────────┐
│ Metric       │  Score │ Threshold │ Status │
├──────────────┼────────┼───────────┼────────┤
│ MRR@10       │  0.723 │     0.700 │   ✅   │
│ Precision@5  │  0.412 │     0.350 │   ✅   │
│ NDCG@10      │  0.651 │     0.600 │   ✅   │
└──────────────┴────────┴───────────┴────────┘
```

### 2. Run with Verbose Output

```bash
ragctl eval --questions eval_datasets/clockify_v1.jsonl --verbose
```

Shows per-query metrics:
```
Query: How do I track time in Clockify? | MRR: 0.500 | P@5: 0.400 | NDCG@10: 0.630
Query: What are the pricing tiers? | MRR: 1.000 | P@5: 0.600 | NDCG@10: 0.811
...
```

### 3. Inspect Retrieval for Specific Query

```bash
ragctl query "How do I track time?" --debug
```

**Debug output** shows:
- Top-K retrieved chunks with scores
- BM25 vs dense score breakdown
- Which chunks passed threshold
- MMR diversification impact

---

## Parameter Tuning

### Scenario 1: Low MRR (First Relevant Result Too Low)

**Symptoms**: `MRR < 0.50`, users must scroll to find relevant answers

**Diagnosis**:
- Run: `ragctl query "your problem query" --debug`
- Check: Is the first relevant chunk ranked low?

**Solutions**:
1. **Increase dense weight**: `export RAG_HYBRID_ALPHA=0.3` (favor semantic)
2. **Lower threshold**: `export RAG_THRESHOLD=0.20` (include more candidates)
3. **Check embeddings**: Ensure `RAG_EMBED_MODEL=nomic-embed-text:latest` is active
4. **Verify index**: Run `ragctl ingest` to rebuild if switching embedding backends

### Scenario 2: Low Precision@5 (Too Many Irrelevant Results)

**Symptoms**: `P@5 < 0.30`, top results contain off-topic chunks

**Solutions**:
1. **Increase threshold**: `export RAG_THRESHOLD=0.35` (stricter filtering)
2. **Adjust alpha based on query type**:
   - For keyword-heavy queries: `export RAG_HYBRID_ALPHA=0.7` (favor BM25)
   - For conceptual queries: `export RAG_HYBRID_ALPHA=0.3` (favor dense)
3. **Check query expansion**: Review `query_expansions.json` for false synonyms

### Scenario 3: Low NDCG@10 (Relevant Results in Wrong Order)

**Symptoms**: `NDCG < 0.50`, relevant chunks present but poorly ranked

**Solutions**:
1. **Enable FAISS**: Ensure index built with FAISS (faster, better ranking)
   ```bash
   ragctl doctor  # Check FAISS status
   ragctl ingest  # Rebuild if needed
   ```
2. **Tune MMR lambda**: Lower for more diversity
   ```bash
   export RAG_MMR_LAMBDA=0.6  # More diversity, less strict relevance
   ```
3. **Increase candidate pool**:
   ```bash
   export RAG_TOP_K=20  # Retrieve more candidates
   export RAG_FAISS_MULTIPLIER=4  # Cast wider net for FAISS
   ```

### Scenario 4: Queries Timing Out or Too Slow

**Symptoms**: `ragctl query` takes >15s, users report slow responses

**Solutions**:
1. **Reduce top_k**:
   ```bash
   export RAG_TOP_K=10  # Fewer candidates
   export RAG_PACK_TOP=5  # Fewer chunks to LLM
   ```
2. **Use FAISS**: Faster than linear scan
   ```bash
   ragctl doctor  # Check "FAISS index: ✅"
   ```
3. **Optimize FAISS** (M1 specific):
   ```bash
   export RAG_FAISS_NLIST=64  # Already optimized for M1
   export RAG_FAISS_NPROBE=16  # Balance speed vs accuracy
   ```
4. **Benchmark retrieval**:
   ```bash
   ragctl benchmark --retrieval
   ```

---

## Troubleshooting Poor Retrieval

### Problem: "No results returned" or "Coverage check failed"

**Diagnosis**:
```bash
ragctl query "your query" --debug
```

Check log for: `Coverage check failed: only N chunks above threshold`

**Solutions**:
1. **Lower threshold**:
   ```bash
   export RAG_THRESHOLD=0.15  # Very permissive
   ```
2. **Check if query is in-domain**:
   - Query: "How do I use Clockify?" → Should work
   - Query: "Quantum physics equations" → Won't work (out of domain)
3. **Verify index exists**:
   ```bash
   ls -lh var/chunks.jsonl var/faiss.index
   ragctl ingest  # Rebuild if missing
   ```

### Problem: "Dimension mismatch" error

**Symptoms**:
```
❌ Embedding dimension mismatch: stored=384, expected=768
```

**Cause**: Switched embedding backend (local ↔ Ollama) without rebuilding index

**Solution**:
```bash
# Check current backend
echo $EMB_BACKEND  # "local" or "ollama"

# Rebuild index with current backend
ragctl ingest

# Verify
ragctl doctor
```

### Problem: Retrieval metrics dropped after code change

**Solution**: Run regression tests
```bash
# 1. Benchmark before change
ragctl benchmark --retrieval --output baseline.json

# 2. Make code change

# 3. Benchmark after change
ragctl benchmark --retrieval --output after_change.json

# 4. Compare
python -c "
import json
baseline = json.load(open('baseline.json'))
after = json.load(open('after_change.json'))
print(f\"Latency change: {after['retrieve_hybrid']['latency_ms']['mean'] / baseline['retrieve_hybrid']['latency_ms']['mean']:.2%}\")
"
```

### Problem: FAISS not available on M1

**Symptoms**:
```
⚠️  FAISS not available, falling back to linear search
```

**Solution**: Install FAISS for M1
```bash
# Option 1: Conda (recommended for M1)
conda install -c conda-forge faiss-cpu=1.8.0

# Option 2: Use provided script
./scripts/install_faiss_m1.sh

# Verify
python -c "import faiss; print(f'FAISS {faiss.__version__} loaded')"

# Rebuild index
ragctl ingest
```

---

## Advanced Tuning: Alpha Sweep

To find optimal `alpha` for your dataset:

```bash
#!/bin/bash
# alpha_sweep.sh

for alpha in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  echo "Testing alpha=$alpha"
  export RAG_HYBRID_ALPHA=$alpha
  ragctl eval --questions eval_datasets/clockify_v1.jsonl | grep "MRR@10" >> alpha_sweep_results.txt
done

# Find best alpha
grep "MRR@10" alpha_sweep_results.txt | sort -k2 -rn | head -1
```

**Expected output**:
```
alpha=0.5 → MRR@10: 0.723
alpha=0.4 → MRR@10: 0.745  ← Best
alpha=0.3 → MRR@10: 0.712
```

---

## Recommended Configuration for Production

Based on testing with `eval_datasets/clockify_v1.jsonl`:

```bash
# ~/.bashrc or config file
export RAG_TOP_K=15
export RAG_PACK_TOP=8
export RAG_THRESHOLD=0.25
export RAG_HYBRID_ALPHA=0.5
export RAG_MMR_LAMBDA=0.75
export RAG_FAISS_MULTIPLIER=3
export EMB_BACKEND=ollama  # Use nomic-embed-text for best quality
```

**Why these values**:
- `top_k=15`: Good recall without excessive latency
- `threshold=0.25`: Balanced precision/recall
- `alpha=0.5`: Works well for mixed query types
- `mmr_lambda=0.75`: Prioritizes relevance over diversity
- `faiss_multiplier=3`: 3x oversampling for reranking

---

## Next Steps

1. **Establish baseline**: Run `ragctl eval` to get current metrics
2. **Identify issues**: Use `--verbose` to find problematic queries
3. **Tune parameters**: Adjust based on scenarios above
4. **Measure impact**: Re-run `ragctl eval` after changes
5. **Document changes**: Update this guide with your findings

For questions or issues, see:
- `docs/ARCHITECTURE.md` - Retrieval pipeline details
- `docs/OPERATIONS.md` - Operational runbook
- `clockify_rag/retrieval.py` - Source code
