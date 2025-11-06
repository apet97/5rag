# Quick Wins (Top 10)

All items require <30 minutes, offer high impact, and are low risk.

1. **Import fixes for clockify_rag.http_utils**  
   Add missing imports at top of module:  
   ```python
   import logging
   import os
   import requests
   ```  
   Impact: prevents immediate NameError when module imported.【F:clockify_rag/http_utils.py†L1-L40】

2. **Raise local embedding batch size constant**  
   Match CLI tuning by bumping `_ST_BATCH_SIZE` to 96.  
   ```python
   _ST_BATCH_SIZE = int(os.getenv("EMB_BATCH", "96"))
   ```  
   Impact: ~2x faster SentenceTransformer batches.【F:clockify_rag/embedding.py†L9-L32】

3. **Honor query logging privacy flags**  
   Before writing `answer` in `log_query`, gate on `LOG_QUERY_INCLUDE_ANSWER`.  
   ```python
   from clockify_rag.config import LOG_QUERY_INCLUDE_ANSWER, LOG_QUERY_ANSWER_PLACEHOLDER
   if LOG_QUERY_INCLUDE_ANSWER:
       log_entry["answer"] = answer
   elif LOG_QUERY_ANSWER_PLACEHOLDER:
       log_entry["answer"] = LOG_QUERY_ANSWER_PLACEHOLDER
   ```  
   Impact: avoids leaking sensitive answers in logs.【F:clockify_support_cli_final.py†L2388-L2458】

4. **Expose retrieval profiling summary in debug logs**  
   After `log_kpi`, add:  
   ```python
   if logger.isEnabledFor(logging.DEBUG) and RETRIEVE_PROFILE_LAST:
       logger.debug(json.dumps({"event": "retrieve_profile", **RETRIEVE_PROFILE_LAST}))
   ```  
   Impact: immediate visibility into ANN reuse.【F:clockify_support_cli_final.py†L1508-L1692】

5. **Document query expansion override**  
   Add docstring snippet to README showing `CLOCKIFY_QUERY_EXPANSIONS` usage.  
   Impact: encourages tuning without code changes.【F:clockify_support_cli_final.py†L167-L238】

6. **Remove legacy CLI stub**  
   Delete `clockify_support_cli.py` (17 LOC) or replace with import to main entrypoint.  
   Impact: prevents accidental import of outdated module.【F:clockify_support_cli.py†L1-L17】

7. **Update Makefile default target**  
   Point `run` target to `python -m clockify_support_cli_final chat`.  
   Impact: ensures contributors run maintained CLI.【F:Makefile†L1-L123】

8. **Add `set -euo pipefail` to shell scripts**  
   Prepend safety flags to scripts under `scripts/`.  
   ```bash
   set -euo pipefail
   ```  
   Impact: avoids silent failures in automation.【F:scripts/acceptance_test.sh†L1-L80】

9. **Surface JSON confidence in tests**  
   Extend `tests/test_json_output.py` to assert `confidence` field exists.  
   Impact: locks in Rank 28 change.【F:tests/test_json_output.py†L1-L36】

10. **Add README link to benchmark suite**  
    Insert quick command snippet referencing `python benchmark.py --quick`.  
    Impact: makes performance tooling discoverable.【F:README.md†L1-L525】
Each item below can be delivered in <30 minutes, carries low risk, and has visible impact.

1. **Fix FAISS branch syntax regression**  
   _Impact:_ Restores ANN speedups.  
   _Snippet:_
   ```python
   # clockify_support_cli_final.py
   if remaining_idx.size:
       dot_start = time.perf_counter()
       dense_scores_full[remaining_idx] = vecs_n[remaining_idx].dot(qv_n)
       dot_elapsed = time.perf_counter() - dot_start
       dense_computed = int(remaining_idx.size)
   # remove stray max(...) and closing paren
   distances = np.asarray(D[0], dtype="float32")
   ```

2. **Reuse packaged QueryCache**  
   _Impact:_ Eliminates race conditions and duplication.  
   _Snippet:_
   ```python
   # clockify_support_cli_final.py
   from clockify_rag.caching import QueryCache, RateLimiter
   # delete the local QueryCache/RateLimiter class definitions
   CACHE = QueryCache(maxsize=200, ttl_seconds=3600)
   ```

3. **Initialize dense_scores in HNSW branch**  
   _Impact:_ Prevents NameError when FAISS disabled.  
   _Snippet:_
   ```python
   elif hnsw:
       _, cand = hnsw.knn_query(qv_n, k=max(...))
       candidate_idx = cand[0].tolist()
       dense_scores_full = vecs_n.dot(qv_n)
       dense_scores = dense_scores_full[candidate_idx]
   ```

4. **Remove duplicate candidate assignments in dense fallback**  
   _Impact:_ Avoids redundant dot product.  
   _Snippet:_
   ```python
   else:
       dense_scores_full = vecs_n.dot(qv_n)
       dense_scores = dense_scores_full
       candidate_idx = np.arange(n_chunks).tolist()
   ```

5. **Deduplicate FAISS candidate ids**  
   _Impact:_ Improves diversity for packing.  
   _Snippet:_
   ```python
   candidate_idx = sorted(set(int(i) for i in candidate_idx if 0 <= i < n_chunks))
   candidate_idx_array = np.array(candidate_idx, dtype=np.int32)
   ```

6. **Provide safe default when hybrid_full falls back**  
   _Impact:_ Accurate KPI logs.  
   _Snippet:_
   ```python
   else:
       hybrid_full = np.full(len(chunks), -1.0, dtype="float32")
       for idx, score in zip(candidate_idx, hybrid):
           hybrid_full[idx] = score
   ```

7. **Short-circuit cache metadata enrichment**  
   _Impact:_ Stops AttributeError when metadata is None.  
   _Snippet:_
   ```python
   answer, metadata, timestamp = self.cache[key]
   metadata = dict(metadata or {})
   metadata.setdefault("timestamp", timestamp)
   ```

8. **Guard sanitize_question against double spaces from strip**  
   _Impact:_ Eliminates false negatives when question becomes empty.  
   _Snippet:_
   ```python
   q = q.strip()
   if not q:
       raise ValueError("Question cannot be empty. Hint: Provide a meaningful question about Clockify.")
   ```

9. **Default benchmark harness to fake remote**  
   _Impact:_ Allows CI to run benchmarks offline.  
   _Snippet:_
   ```python
   use_real = os.environ.get("BENCHMARK_REAL_REMOTE") == "1"
   if not use_real:
       _patch_fake_remote()
   ```

10. **Require auth token in DeepSeek shim**  
    _Impact:_ Prevents accidental public exposure.  
    _Snippet:_
    ```python
    if not AUTH_TOKEN:
        raise SystemExit("Set SHIM_AUTH_TOKEN to secure the proxy")
    ```
