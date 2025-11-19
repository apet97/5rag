#!/usr/bin/env python3
"""Offline evaluation runner for the Clockify RAG system.

DEPRECATED: Use `ragctl eval` instead.
"""

import argparse
import sys
import warnings
from clockify_rag.evaluation import evaluate_dataset, SUCCESS_THRESHOLDS

def main():
    warnings.warn(
        "This script is deprecated and will be removed in v7.0. Use 'ragctl eval' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    print("⚠️  WARNING: This script is deprecated. Please use 'ragctl eval' instead.\n")

    parser = argparse.ArgumentParser(description="Evaluate RAG system on ground truth dataset")
    parser.add_argument(
        "--dataset",
        default="eval_datasets/clockify_v1.jsonl",
        help="Path to evaluation dataset",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-query results")
    parser.add_argument(
        "--min-mrr",
        type=float,
        default=SUCCESS_THRESHOLDS["mrr_at_10"],
        help="Minimum acceptable MRR@10 (default: %(default).2f)",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=SUCCESS_THRESHOLDS["precision_at_5"],
        help="Minimum acceptable Precision@5 (default: %(default).2f)",
    )
    parser.add_argument(
        "--min-ndcg",
        type=float,
        default=SUCCESS_THRESHOLDS["ndcg_at_10"],
        help="Minimum acceptable NDCG@10 (default: %(default).2f)",
    )
    parser.add_argument(
        "--llm-report",
        action="store_true",
        help="Generate LLM answers for each query (uses answer_once and respects RAG_LLM_CLIENT)",
    )
    parser.add_argument(
        "--llm-output",
        default="eval_reports/llm_answers.jsonl",
        help="Path to save LLM answer report when --llm-report is used",
    )
    args = parser.parse_args()

    try:
        results = evaluate_dataset(
            dataset_path=args.dataset,
            verbose=args.verbose,
            llm_report=args.llm_report,
            llm_output=args.llm_output,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Print results (similar to original script for backward compat)
    print("\n" + "="*70)
    print(f"RAG EVALUATION RESULTS")
    print("="*70)
    print(f"Retrieval Mode:  {results['retrieval_mode']}")
    if results['hybrid_available']:
        ann_status = "✅ FAISS enabled" if results['faiss_enabled'] else "⚠️  FAISS disabled (full-scan)"
        print(f"ANN Index:       {ann_status}")
    print(f"Dataset size:    {results['dataset_size']}")
    print(f"Queries eval:    {results['queries_processed']}")
    if results['queries_skipped']:
        print(f"Skipped queries: {results['queries_skipped']} (missing relevance annotations)")
    print("-"*70)
    print(f"MRR@10:          {results['mrr_at_10']:.3f} (±{results['mrr_std']:.3f})")
    print(f"Precision@5:     {results['precision_at_5']:.3f} (±{results['precision_std']:.3f})")
    print(f"NDCG@10:         {results['ndcg_at_10']:.3f} (±{results['ndcg_std']:.3f})")
    print("="*70)

    thresholds = {
        "mrr_at_10": args.min_mrr,
        "precision_at_5": args.min_precision,
        "ndcg_at_10": args.min_ndcg,
    }

    success = all(results[key] >= threshold for key, threshold in thresholds.items())
    if not success:
        print(
            "\nEvaluation thresholds not met:" +
            " ".join(
                f"{key}={results[key]:.3f} < {threshold:.2f}"
                for key, threshold in thresholds.items()
                if results[key] < threshold
            )
        )
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
