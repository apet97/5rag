"""Evaluation module for RAG system performance metrics.

Computes MRR, Precision, and NDCG for retrieval evaluation.
Supports both hybrid (production) and lexical (fallback) retrieval modes.
"""

import json
import os
import re
import sys
import logging
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Any, Union

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from . import config
from .chunking import build_chunks
from .retrieval import retrieve
from .indexing import load_index
from .answer import answer_once

logger = logging.getLogger(__name__)

SUCCESS_THRESHOLDS = {
    "mrr_at_10": 0.70,
    "precision_at_5": 0.35,
    "ndcg_at_10": 0.60,
}

TOP_K = 12


def compute_mrr(retrieved_ids: List[int], relevant_ids: Set[int]) -> float:
    """Mean Reciprocal Rank - measures rank of first relevant result."""
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def compute_precision_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int = 5) -> float:
    """Precision@K - fraction of top K results that are relevant."""
    if k == 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    hits = len(set(retrieved_k) & relevant_ids)
    return hits / k


def compute_ndcg_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain@K - position-aware metric."""
    dcg = sum(1.0 / np.log2(i + 2) if doc_id in relevant_ids else 0.0 for i, doc_id in enumerate(retrieved_ids[:k]))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0


def _normalize_text(value: str) -> str:
    """Lowercase + collapse whitespace for robust key matching."""
    return re.sub(r"\s+", " ", value.strip().lower())


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25 fallback."""
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return [tok for tok in cleaned.split() if tok]


class LexicalRetriever:
    """BM25-based lexical retriever used when the hybrid index is unavailable."""

    def __init__(self, chunks: List[Dict]) -> None:
        if BM25Okapi is None:
            raise RuntimeError("rank-bm25 is required for lexical evaluation.")

        corpus_tokens = []
        for chunk in chunks:
            fields = " ".join(
                [
                    chunk.get("title", ""),
                    chunk.get("section", ""),
                    chunk.get("text", ""),
                ]
            )
            corpus_tokens.append(_tokenize(fields))
        self._bm25 = BM25Okapi(corpus_tokens)

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[int]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        return order.tolist()


def _load_chunks_from_disk_or_build() -> List[Dict]:
    """Load chunk metadata from disk or rebuild from the knowledge base."""
    chunks_path = config.FILES["chunks"]
    if os.path.exists(chunks_path):
        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks

    knowledge_path = "knowledge_full.md"  # Default location
    if not os.path.exists(knowledge_path):
        raise FileNotFoundError(f"{knowledge_path} is missing. Run 'ragctl ingest' first.")

    return build_chunks(knowledge_path)


def _build_chunk_lookup(chunks: List[Dict]) -> Tuple[Dict[str, int], Dict[str, Set[int]], Dict[str, Set[int]]]:
    """Create helper mappings for resolving dataset references."""
    id_map: Dict[str, int] = {}
    title_section_map: Dict[str, Set[int]] = defaultdict(set)
    title_map: Dict[str, Set[int]] = defaultdict(set)

    for idx, chunk in enumerate(chunks):
        cid = chunk.get("id")
        if cid is not None:
            id_map[str(cid)] = idx

        title = _normalize_text(chunk.get("title", ""))
        section = _normalize_text(chunk.get("section", ""))
        title_map[title].add(idx)
        title_section_map[f"{title}|{section}"].add(idx)

    return id_map, title_section_map, title_map


def _resolve_relevant_indices(
    example: Dict,
    id_map: Dict[str, int],
    title_section_map: Dict[str, Set[int]],
    title_map: Dict[str, Set[int]],
) -> Set[int]:
    """Resolve dataset annotations into chunk indices."""
    relevant: Set[int] = set()

    for cid in example.get("relevant_chunk_ids", []) or []:
        idx = id_map.get(str(cid))
        if idx is not None:
            relevant.add(idx)

    for chunk_ref in example.get("relevant_chunks", []) or []:
        title = _normalize_text(chunk_ref.get("title", ""))
        section = _normalize_text(chunk_ref.get("section", ""))
        key = f"{title}|{section}"
        if section and key in title_section_map:
            relevant.update(title_section_map[key])
            continue
        if title in title_map:
            relevant.update(title_map[title])

    return relevant


def evaluate_dataset(
    dataset_path: str,
    verbose: bool = False,
    llm_report: bool = False,
    llm_output: Optional[str] = None,
) -> Dict[str, Any]:
    """Run evaluation on dataset.

    Args:
        dataset_path: Path to evaluation dataset JSONL file
        verbose: Print per-query results if True
        llm_report: Generate LLM answers for qualitative analysis
        llm_output: Path to save LLM answers

    Returns:
        Dict with evaluation metrics
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    # Load chunks
    chunks = _load_chunks_from_disk_or_build()

    # Check for hybrid artifacts
    hybrid_artifacts_exist = (
        os.path.exists(config.FILES["emb"])
        and os.path.exists(config.FILES["bm25"])
        and os.path.exists(config.FILES["chunks"])
    )
    faiss_available = os.path.exists(config.FILES["faiss_index"])
    faiss_index_path = config.FILES["faiss_index"] if faiss_available else None

    rag_available = False
    retrieval_chunks = chunks
    retrieval_fn = None
    vecs_n = None
    bm = None
    hnsw = None
    retrieval_mode = "unknown"

    if hybrid_artifacts_exist:
        logger.info("Hybrid artifacts detected - using hybrid retrieval path")
        try:
            result = load_index()
            if result is None:
                raise RuntimeError("load_index() returned None")

            if isinstance(result, dict):
                retrieval_chunks = result["chunks"]
                vecs_n = result["vecs_n"]
                bm = result["bm"]
                hnsw = result.get("faiss_index")
            else:
                retrieval_chunks, vecs_n, bm, hnsw = result

            def _hybrid_retrieve(q):
                selected, _ = retrieve(
                    q,
                    retrieval_chunks,
                    vecs_n,
                    bm,
                    top_k=TOP_K,
                    hnsw=hnsw,
                    faiss_index_path=faiss_index_path,
                )
                return selected

            retrieval_fn = _hybrid_retrieve
            rag_available = True
            retrieval_mode = f"Hybrid (FAISS={'enabled' if faiss_available else 'disabled'})"
        except Exception as exc:
            logger.error(f"Failed to load hybrid index: {exc}")
            raise
    else:
        logger.warning("No hybrid artifacts found - using lexical BM25 fallback")
        lexical_retriever = LexicalRetriever(retrieval_chunks)
        retrieval_fn = lexical_retriever.retrieve
        retrieval_mode = "Lexical BM25 (fallback)"

    # Load dataset
    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    id_map, title_section_map, title_map = _build_chunk_lookup(retrieval_chunks)

    mrr_scores = []
    precision_at_5_scores = []
    ndcg_at_10_scores = []
    skipped = 0
    llm_outputs = []

    for i, example in enumerate(dataset):
        query = example["query"]
        relevant_ids = _resolve_relevant_indices(example, id_map, title_section_map, title_map)

        if not relevant_ids:
            if verbose:
                logger.warning(f"Skipping query {i+1} ('{query}') - no matching relevant chunks found.")
            skipped += 1
            continue

        try:
            retrieved_ids = list(retrieval_fn(query)) if retrieval_fn else []

            mrr = compute_mrr(retrieved_ids, relevant_ids)
            precision_at_5 = compute_precision_at_k(retrieved_ids, relevant_ids, k=5)
            ndcg_at_10 = compute_ndcg_at_k(retrieved_ids, relevant_ids, k=10)

            mrr_scores.append(mrr)
            precision_at_5_scores.append(precision_at_5)
            ndcg_at_10_scores.append(ndcg_at_10)

            if verbose:
                print(f"Query: {query} | MRR: {mrr:.3f} | P@5: {precision_at_5:.3f} | NDCG@10: {ndcg_at_10:.3f}")

            if llm_report and rag_available:
                answer_payload = answer_once(
                    query,
                    retrieval_chunks,
                    vecs_n,
                    bm,
                    hnsw=hnsw,
                    faiss_index_path=faiss_index_path,
                )
                llm_outputs.append(
                    {
                        "query": query,
                        "answer": answer_payload["answer"],
                        "confidence": answer_payload.get("confidence"),
                        "refused": answer_payload.get("refused"),
                        "metadata": answer_payload.get("metadata", {}),
                    }
                )

        except Exception as e:
            logger.error(f"Error evaluating query '{query}': {e}")
            continue

    if not mrr_scores:
        raise RuntimeError("No queries processed")

    results = {
        "dataset_size": len(dataset),
        "mrr_at_10": float(np.mean(mrr_scores)),
        "precision_at_5": float(np.mean(precision_at_5_scores)),
        "ndcg_at_10": float(np.mean(ndcg_at_10_scores)),
        "mrr_std": float(np.std(mrr_scores)),
        "precision_std": float(np.std(precision_at_5_scores)),
        "ndcg_std": float(np.std(ndcg_at_10_scores)),
        "retrieval_mode": retrieval_mode,
        "hybrid_available": rag_available,
        "faiss_enabled": faiss_available,
        "queries_processed": len(mrr_scores),
        "queries_skipped": skipped,
    }

    if llm_report and llm_outputs:
        output_path = llm_output or "eval_reports/llm_answers.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in llm_outputs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        results["llm_report_path"] = output_path

    return results
