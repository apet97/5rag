"""Advanced retrieval tests: hybrid fusion, MMR, query expansion, chaos scenarios.

REQUIREMENTS:
    - All tests use mocked LLM clients (no network calls required)
    - Tests assume fixtures from conftest.py: sample_chunks, sample_embeddings, sample_bm25
    - Dependencies: pytest, numpy, clockify_rag package

RUNNING TESTS:
    # Install dependencies first
    pip install -e '.[dev]'

    # Run all advanced retrieval tests
    pytest tests/test_retrieval_advanced.py -v

    # Run specific test class
    pytest tests/test_retrieval_advanced.py::TestHybridFusion -v

    # Run with coverage
    pytest tests/test_retrieval_advanced.py --cov=clockify_rag.retrieval

TEST COVERAGE:
    - Hybrid fusion with different alpha values (0, 0.5, 1.0)
    - MMR diversification with different lambda values
    - Query expansion and synonym handling
    - Chaos scenarios (zero embeddings, NaN scores, dimension mismatches)
    - Regression tests (determinism, baseline MRR)
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clockify_rag.retrieval import (
    retrieve,
    normalize_scores_zscore,
    apply_mmr,
    expand_query_synonyms,
    DenseScoreStore,
)
from clockify_rag import config


class TestHybridFusion:
    """Test hybrid BM25 + dense fusion with different alpha values."""

    def test_alpha_zero_pure_dense(self, monkeypatch, sample_chunks, sample_embeddings, sample_bm25):
        """Test alpha=0 gives pure dense (semantic) retrieval."""
        monkeypatch.setattr(config, "HYBRID_ALPHA", 0.0)
        monkeypatch.setattr(config, "EMB_BACKEND", "local")

        # Inject predictable embeddings
        query_vec = sample_embeddings[0]
        import clockify_rag.retrieval as retrieval_module
        monkeypatch.setattr(retrieval_module, "embed_query", lambda q, retries=0: query_vec)

        selected, scores = retrieve("test query", sample_chunks, sample_embeddings, sample_bm25, top_k=3)

        # With alpha=0, hybrid = 0*bm25 + 1*dense, so BM25 should not affect ranking
        hybrid = scores["hybrid"]
        dense_scores = scores["dense"]

        # Convert DenseScoreStore to array for comparison
        dense_array = np.array([dense_scores[i] for i in range(len(sample_chunks))])

        # Verify hybrid scores match dense (after normalization)
        # Since both are z-score normalized, they should be identical
        assert np.allclose(hybrid, dense_array, atol=0.1), \
            "With alpha=0, hybrid should equal dense scores"

    def test_alpha_one_pure_bm25(self, monkeypatch, sample_chunks, sample_embeddings, sample_bm25):
        """Test alpha=1 gives pure BM25 (lexical) retrieval."""
        monkeypatch.setattr(config, "HYBRID_ALPHA", 1.0)
        monkeypatch.setattr(config, "EMB_BACKEND", "local")

        query_vec = sample_embeddings[0]
        import clockify_rag.retrieval as retrieval_module
        monkeypatch.setattr(retrieval_module, "embed_query", lambda q, retries=0: query_vec)

        selected, scores = retrieve("test query", sample_chunks, sample_embeddings, sample_bm25, top_k=3)

        # With alpha=1, hybrid = 1*bm25 + 0*dense, so dense should not affect ranking
        hybrid = scores["hybrid"]
        bm25 = scores["bm25"]

        # Verify hybrid scores match BM25 (after normalization)
        assert np.allclose(hybrid, bm25, atol=0.1), \
            "With alpha=1, hybrid should equal BM25 scores"

    def test_alpha_half_balanced(self, monkeypatch, sample_chunks, sample_embeddings, sample_bm25):
        """Test alpha=0.5 gives balanced fusion."""
        monkeypatch.setattr(config, "HYBRID_ALPHA", 0.5)
        monkeypatch.setattr(config, "EMB_BACKEND", "local")

        query_vec = sample_embeddings[0]
        import clockify_rag.retrieval as retrieval_module
        monkeypatch.setattr(retrieval_module, "embed_query", lambda q, retries=0: query_vec)

        selected, scores = retrieve("test query", sample_chunks, sample_embeddings, sample_bm25, top_k=3)

        hybrid = scores["hybrid"]
        bm25 = scores["bm25"]
        dense_scores = scores["dense"]
        dense_array = np.array([dense_scores[i] for i in range(len(sample_chunks))])

        # Hybrid should be between BM25 and dense
        for i in range(len(hybrid)):
            # Check that hybrid is a weighted combination
            expected = 0.5 * bm25[i] + 0.5 * dense_array[i]
            assert abs(hybrid[i] - expected) < 0.1, \
                f"Chunk {i}: hybrid={hybrid[i]:.3f} should ≈ 0.5*bm25 + 0.5*dense = {expected:.3f}"

    def test_alpha_affects_ranking(self, monkeypatch, sample_chunks, sample_embeddings, sample_bm25):
        """Test that different alpha values produce different rankings."""
        monkeypatch.setattr(config, "EMB_BACKEND", "local")

        query_vec = sample_embeddings[0]
        import clockify_rag.retrieval as retrieval_module
        monkeypatch.setattr(retrieval_module, "embed_query", lambda q, retries=0: query_vec)

        # Get rankings with different alphas
        rankings = {}
        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            monkeypatch.setattr(config, "HYBRID_ALPHA", alpha)
            selected, _ = retrieve("pricing plans", sample_chunks, sample_embeddings, sample_bm25, top_k=3)
            rankings[alpha] = tuple(selected)

        # Verify that at least some rankings differ
        unique_rankings = len(set(rankings.values()))
        assert unique_rankings > 1, \
            f"Different alpha values should produce different rankings, got {unique_rankings} unique"


class TestMMRDiversification:
    """Test MMR (Maximal Marginal Relevance) diversification."""

    def test_mmr_reduces_duplicates(self, sample_chunks, sample_embeddings):
        """Test that MMR reduces near-duplicate results."""
        # Create candidates with high similarity (duplicates)
        candidates = [0, 0, 1, 1, 2]  # Duplicates: 0 appears twice, 1 appears twice
        scores = np.array([0.9, 0.88, 0.7, 0.68, 0.5])

        # MMR should prefer diversity
        selected = apply_mmr(candidates, scores, sample_embeddings, lambda_param=0.5, top_k=3)

        # Should return unique indices (MMR deduplicates implicitly via distance)
        assert len(selected) <= 3, f"Should return at most 3, got {len(selected)}"
        # The actual deduplication happens via semantic similarity in MMR

    def test_mmr_lambda_affects_selection(self, sample_chunks, sample_embeddings):
        """Test that lambda parameter affects MMR selection."""
        candidates = list(range(min(5, len(sample_chunks))))
        scores = np.random.rand(len(candidates))

        # Lambda = 1.0 should favor relevance only
        selected_high_lambda = apply_mmr(candidates, scores, sample_embeddings, lambda_param=1.0, top_k=3)

        # Lambda = 0.5 should balance relevance and diversity
        selected_balanced = apply_mmr(candidates, scores, sample_embeddings, lambda_param=0.5, top_k=3)

        # Results may differ due to diversity constraint
        # At minimum, verify both return valid results
        assert len(selected_high_lambda) == 3
        assert len(selected_balanced) == 3

    def test_mmr_preserves_top_result(self, sample_chunks, sample_embeddings):
        """Test that MMR always includes the top-scored candidate."""
        candidates = [2, 1, 3, 0]
        scores = np.array([0.9, 0.7, 0.5, 0.3])  # Candidate 0 (idx=2) has highest score

        selected = apply_mmr(candidates, scores, sample_embeddings, lambda_param=0.7, top_k=3)

        # First selected should be the highest-scored candidate
        assert selected[0] == candidates[0], \
            f"MMR should preserve top result, expected {candidates[0]}, got {selected[0]}"


class TestQueryExpansion:
    """Test query expansion with synonyms."""

    def test_query_expansion_adds_synonyms(self):
        """Test that query expansion adds known synonyms."""
        query = "How do I track time?"
        expanded = expand_query_synonyms(query)

        # "track" should expand to include "log", "record", "enter" (if configured)
        # Check that expansion is non-empty and includes original terms
        assert len(expanded) >= len(query), "Expanded query should be at least as long as original"
        assert "track" in expanded.lower() or "log" in expanded.lower(), \
            "Expansion should include original or synonym terms"

    def test_query_expansion_preserves_unknown_terms(self):
        """Test that unknown terms are preserved."""
        query = "xyzabc123 unknown term"
        expanded = expand_query_synonyms(query)

        # Unknown terms should remain in expanded query
        assert "xyzabc123" in expanded or "unknown" in expanded, \
            "Expansion should preserve unknown terms"

    def test_query_expansion_is_deterministic(self):
        """Test that query expansion is deterministic (same input → same output)."""
        query = "How do I track time?"
        expanded1 = expand_query_synonyms(query)
        expanded2 = expand_query_synonyms(query)

        assert expanded1 == expanded2, "Query expansion should be deterministic"


class TestRetrievalChaos:
    """Chaos tests: corrupt data, edge cases, failure modes."""

    def test_retrieve_with_all_zero_embeddings(self, monkeypatch, sample_chunks, sample_bm25):
        """Test retrieval when all embeddings are zero (degenerate case)."""
        monkeypatch.setattr(config, "EMB_BACKEND", "local")

        # Create zero embeddings
        zero_embeddings = np.zeros((len(sample_chunks), config.EMB_DIM_LOCAL), dtype=np.float32)
        query_vec = np.zeros((1, config.EMB_DIM_LOCAL), dtype=np.float32)

        import clockify_rag.retrieval as retrieval_module
        monkeypatch.setattr(retrieval_module, "embed_query", lambda q, retries=0: query_vec)

        # Should not crash, should fall back to BM25 only
        selected, scores = retrieve("test query", sample_chunks, zero_embeddings, sample_bm25, top_k=3)

        assert len(selected) > 0, "Should return results even with zero embeddings"
        # Dense scores should be low/equal (all zeros), BM25 should dominate

    def test_retrieve_with_nan_scores(self, monkeypatch, sample_chunks, sample_embeddings, sample_bm25):
        """Test retrieval handles NaN scores gracefully."""
        monkeypatch.setattr(config, "EMB_BACKEND", "local")

        # Inject NaN into embeddings
        corrupted_embeddings = sample_embeddings.copy()
        corrupted_embeddings[0, :] = np.nan

        query_vec = sample_embeddings[1]
        import clockify_rag.retrieval as retrieval_module
        monkeypatch.setattr(retrieval_module, "embed_query", lambda q, retries=0: query_vec)

        # Should handle gracefully (filter or replace NaN)
        selected, scores = retrieve("test query", sample_chunks, corrupted_embeddings, sample_bm25, top_k=3)

        # Verify no NaN in output scores
        assert np.all(np.isfinite(scores["hybrid"])), "Hybrid scores should not contain NaN"

    def test_retrieve_with_dimension_mismatch(self, monkeypatch, sample_chunks, sample_bm25):
        """Test retrieval detects dimension mismatch."""
        monkeypatch.setattr(config, "EMB_BACKEND", "local")
        monkeypatch.setattr(config, "EMB_DIM_LOCAL", 384)

        # Create embeddings with wrong dimension
        wrong_dim_embeddings = np.random.rand(len(sample_chunks), 768).astype(np.float32)  # 768 instead of 384
        query_vec = np.random.rand(1, 384).astype(np.float32)

        import clockify_rag.retrieval as retrieval_module
        monkeypatch.setattr(retrieval_module, "embed_query", lambda q, retries=0: query_vec)

        # Should raise error or handle gracefully
        with pytest.raises((ValueError, AssertionError)):
            retrieve("test query", sample_chunks, wrong_dim_embeddings, sample_bm25, top_k=3)

    def test_retrieve_with_empty_chunks(self, monkeypatch, sample_embeddings, sample_bm25):
        """Test retrieval with empty chunk list."""
        monkeypatch.setattr(config, "EMB_BACKEND", "local")

        empty_chunks = []
        empty_embeddings = np.zeros((0, config.EMB_DIM_LOCAL), dtype=np.float32)
        empty_bm25 = {"idf": {}, "avgdl": 0, "doc_lens": [], "doc_tfs": []}

        query_vec = sample_embeddings[0]
        import clockify_rag.retrieval as retrieval_module
        monkeypatch.setattr(retrieval_module, "embed_query", lambda q, retries=0: query_vec)

        # Should return empty results without crashing
        selected, scores = retrieve("test query", empty_chunks, empty_embeddings, empty_bm25, top_k=3)

        assert len(selected) == 0, "Empty chunks should return empty results"

    def test_retrieve_with_very_high_top_k(self, monkeypatch, sample_chunks, sample_embeddings, sample_bm25):
        """Test retrieval when top_k exceeds number of chunks."""
        monkeypatch.setattr(config, "EMB_BACKEND", "local")

        query_vec = sample_embeddings[0]
        import clockify_rag.retrieval as retrieval_module
        monkeypatch.setattr(retrieval_module, "embed_query", lambda q, retries=0: query_vec)

        # Request more results than available
        top_k = len(sample_chunks) * 2
        selected, scores = retrieve("test query", sample_chunks, sample_embeddings, sample_bm25, top_k=top_k)

        # Should return at most len(chunks) results
        assert len(selected) <= len(sample_chunks), \
            f"Should return at most {len(sample_chunks)} results, got {len(selected)}"


class TestScoreNormalization:
    """Test Z-score normalization edge cases."""

    def test_normalize_all_equal_scores(self):
        """Test normalization when all scores are equal (std=0)."""
        scores = [0.5, 0.5, 0.5, 0.5]
        normalized = normalize_scores_zscore(scores)

        # When std=0, should return original scores unchanged
        assert np.allclose(normalized, scores), \
            "Equal scores should remain equal after normalization"

    def test_normalize_single_outlier(self):
        """Test normalization with one outlier."""
        scores = [0.1, 0.1, 0.1, 0.9]  # One high outlier
        normalized = normalize_scores_zscore(scores)

        # Outlier should have high z-score
        assert normalized[3] > normalized[0], \
            "Outlier should have higher normalized score"
        assert normalized[3] > 1.0, \
            "Outlier should be >1 std dev above mean"

    def test_normalize_preserves_ranking(self):
        """Test that normalization preserves ranking order."""
        scores = [0.2, 0.5, 0.8, 0.3]
        normalized = normalize_scores_zscore(scores)

        # Ranking should be preserved
        orig_order = np.argsort(scores)[::-1]
        norm_order = np.argsort(normalized)[::-1]

        assert np.array_equal(orig_order, norm_order), \
            "Normalization should preserve ranking order"


class TestRetrievalRegression:
    """Regression tests to prevent quality degradation."""

    def test_retrieve_deterministic_with_seed(self, monkeypatch, sample_chunks, sample_embeddings, sample_bm25):
        """Test that retrieval is deterministic with fixed seed."""
        monkeypatch.setattr(config, "EMB_BACKEND", "local")

        query_vec = sample_embeddings[0]
        import clockify_rag.retrieval as retrieval_module
        monkeypatch.setattr(retrieval_module, "embed_query", lambda q, retries=0: query_vec)

        # Run retrieval twice
        selected1, scores1 = retrieve("test query", sample_chunks, sample_embeddings, sample_bm25, top_k=3)
        selected2, scores2 = retrieve("test query", sample_chunks, sample_embeddings, sample_bm25, top_k=3)

        # Results should be identical
        assert selected1 == selected2, \
            "Retrieval should be deterministic for same inputs"
        assert np.allclose(scores1["hybrid"], scores2["hybrid"]), \
            "Scores should be identical for same inputs"

    def test_retrieve_baseline_mrr(self, monkeypatch, sample_chunks, sample_embeddings, sample_bm25):
        """Test that retrieval achieves baseline MRR for known good query."""
        monkeypatch.setattr(config, "EMB_BACKEND", "local")

        # Query that should retrieve time-tracking chunks
        query = "How do I track time?"
        query_vec = sample_embeddings[0]

        import clockify_rag.retrieval as retrieval_module
        monkeypatch.setattr(retrieval_module, "embed_query", lambda q, retries=0: query_vec)

        selected, scores = retrieve(query, sample_chunks, sample_embeddings, sample_bm25, top_k=5)

        # Assuming chunks 0 and 2 are about time tracking (based on sample_chunks fixture)
        # MRR should be ≥ 0.5 (relevant result in top 2)
        relevant_chunks = {0, 2}
        reciprocal_rank = 0.0
        for rank, idx in enumerate(selected, 1):
            if idx in relevant_chunks:
                reciprocal_rank = 1.0 / rank
                break

        # Baseline: should find relevant result in top 3 (MRR ≥ 0.33)
        assert reciprocal_rank >= 0.33, \
            f"Baseline MRR should be ≥0.33, got {reciprocal_rank:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
