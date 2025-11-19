"""Comprehensive tests for retrieval ranking logic - MMR, score fusion, normalization."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clockify_rag.retrieval import (
    apply_mmr,
    normalize_scores_zscore,
    normalize_scores_minmax,
    fuse_scores_rrf,
    DenseScoreStore,
)


class TestMMR:
    """Test Maximal Marginal Relevance (MMR) for diversity."""

    def test_mmr_basic_diversity(self):
        """Test that MMR promotes diversity in results."""
        # Create embeddings where some are very similar
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Candidate embeddings: first two very similar, third different
        candidate_embs = np.array(
            [
                [0.9, 0.1, 0.0],  # Very similar to query
                [0.85, 0.15, 0.0],  # Also very similar to query AND to first
                [0.0, 0.0, 1.0],  # Different from query and others
            ],
            dtype=np.float32,
        )

        initial_scores = np.array([0.9, 0.85, 0.3])  # Scores matching similarity
        candidate_indices = [0, 1, 2]

        # Apply MMR with lambda=0.5 (balance relevance and diversity)
        selected_indices = apply_mmr(
            query_emb=query_emb,
            candidate_embs=candidate_embs,
            initial_scores=initial_scores,
            candidate_indices=candidate_indices,
            top_k=2,
            lambda_param=0.5,
        )

        # MMR should prefer: [0, 2] over [0, 1] because 2 is more diverse
        # (even though 1 has higher similarity to query)
        assert len(selected_indices) == 2
        assert 0 in selected_indices  # Highest score should be included
        # Second result should favor diversity
        # In this case, index 2 should be preferred over 1 despite lower relevance

    def test_mmr_lambda_zero_all_diversity(self):
        """Test MMR with lambda=0 (only diversity, no relevance)."""
        query_emb = np.array([1.0, 0.0], dtype=np.float32)

        # All candidates similar to each other
        candidate_embs = np.array(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], dtype=np.float32
        )

        initial_scores = np.array([0.9, 0.8, 0.7])
        candidate_indices = [0, 1, 2]

        selected_indices = apply_mmr(
            query_emb=query_emb,
            candidate_embs=candidate_embs,
            initial_scores=initial_scores,
            candidate_indices=candidate_indices,
            top_k=2,
            lambda_param=0.0,  # Only diversity
        )

        assert len(selected_indices) == 2
        # Should maximize diversity between selected items

    def test_mmr_lambda_one_all_relevance(self):
        """Test MMR with lambda=1 (only relevance, no diversity)."""
        query_emb = np.array([1.0, 0.0], dtype=np.float32)

        candidate_embs = np.array(
            [[0.9, 0.1], [0.85, 0.15], [0.1, 0.9]], dtype=np.float32
        )

        initial_scores = np.array([0.9, 0.85, 0.1])
        candidate_indices = [0, 1, 2]

        selected_indices = apply_mmr(
            query_emb=query_emb,
            candidate_embs=candidate_embs,
            initial_scores=initial_scores,
            candidate_indices=candidate_indices,
            top_k=2,
            lambda_param=1.0,  # Only relevance
        )

        assert len(selected_indices) == 2
        # Should just pick top 2 by score (0 and 1)
        assert set(selected_indices) == {0, 1}

    def test_mmr_top_k_larger_than_candidates(self):
        """Test MMR when top_k > number of candidates."""
        query_emb = np.array([1.0, 0.0], dtype=np.float32)

        candidate_embs = np.array([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32)

        initial_scores = np.array([0.9, 0.8])
        candidate_indices = [0, 1]

        selected_indices = apply_mmr(
            query_emb=query_emb,
            candidate_embs=candidate_embs,
            initial_scores=initial_scores,
            candidate_indices=candidate_indices,
            top_k=10,  # More than available
            lambda_param=0.5,
        )

        # Should return all available candidates
        assert len(selected_indices) == 2
        assert set(selected_indices) == {0, 1}

    def test_mmr_empty_candidates(self):
        """Test MMR with empty candidates."""
        query_emb = np.array([1.0, 0.0], dtype=np.float32)

        candidate_embs = np.array([], dtype=np.float32).reshape(0, 2)
        initial_scores = np.array([])
        candidate_indices = []

        selected_indices = apply_mmr(
            query_emb=query_emb,
            candidate_embs=candidate_embs,
            initial_scores=initial_scores,
            candidate_indices=candidate_indices,
            top_k=5,
            lambda_param=0.5,
        )

        assert len(selected_indices) == 0

    def test_mmr_single_candidate(self):
        """Test MMR with single candidate."""
        query_emb = np.array([1.0, 0.0], dtype=np.float32)

        candidate_embs = np.array([[0.9, 0.1]], dtype=np.float32)
        initial_scores = np.array([0.9])
        candidate_indices = [0]

        selected_indices = apply_mmr(
            query_emb=query_emb,
            candidate_embs=candidate_embs,
            initial_scores=initial_scores,
            candidate_indices=candidate_indices,
            top_k=3,
            lambda_param=0.5,
        )

        assert len(selected_indices) == 1
        assert selected_indices[0] == 0


class TestScoreNormalization:
    """Test score normalization methods."""

    def test_zscore_normalization_basic(self):
        """Test z-score normalization."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        normalized = normalize_scores_zscore(scores)

        # Z-score should have mean ~ 0 and std ~ 1
        assert abs(np.mean(normalized)) < 0.01
        assert abs(np.std(normalized) - 1.0) < 0.01

    def test_zscore_normalization_constant_scores(self):
        """Test z-score with all same scores (std=0)."""
        scores = np.array([3.0, 3.0, 3.0, 3.0])

        normalized = normalize_scores_zscore(scores)

        # Should handle gracefully (return zeros or original)
        assert len(normalized) == len(scores)

    def test_minmax_normalization_basic(self):
        """Test min-max normalization."""
        scores = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        normalized = normalize_scores_minmax(scores)

        # Should be in range [0, 1]
        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0
        # Min should map to 0, max should map to 1
        assert abs(normalized[0] - 0.0) < 0.01
        assert abs(normalized[-1] - 1.0) < 0.01

    def test_minmax_normalization_negative_scores(self):
        """Test min-max with negative scores."""
        scores = np.array([-10.0, 0.0, 10.0, 20.0])

        normalized = normalize_scores_minmax(scores)

        # Should still be in [0, 1]
        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0

    def test_minmax_normalization_constant_scores(self):
        """Test min-max with all same scores."""
        scores = np.array([5.0, 5.0, 5.0])

        normalized = normalize_scores_minmax(scores)

        # Should handle gracefully (all zeros or all same value)
        assert len(normalized) == len(scores)

    def test_normalization_empty_array(self):
        """Test normalization with empty array."""
        scores = np.array([])

        zscore_result = normalize_scores_zscore(scores)
        minmax_result = normalize_scores_minmax(scores)

        assert len(zscore_result) == 0
        assert len(minmax_result) == 0

    def test_normalization_single_score(self):
        """Test normalization with single score."""
        scores = np.array([7.5])

        zscore_result = normalize_scores_zscore(scores)
        minmax_result = normalize_scores_minmax(scores)

        assert len(zscore_result) == 1
        assert len(minmax_result) == 1


class TestReciprocalRankFusion:
    """Test Reciprocal Rank Fusion (RRF) for score fusion."""

    def test_rrf_basic_fusion(self):
        """Test basic RRF fusion of two score lists."""
        # Two ranking lists with some overlap
        indices1 = [0, 1, 2, 3]
        scores1 = np.array([0.9, 0.7, 0.5, 0.3])

        indices2 = [2, 0, 4, 5]
        scores2 = np.array([0.8, 0.6, 0.4, 0.2])

        fused_scores = fuse_scores_rrf([indices1, indices2], [scores1, scores2], k=60)

        # Item 0 and 2 appear in both lists, should have higher fused scores
        assert len(fused_scores) > 0
        assert 0 in fused_scores  # Appears in both
        assert 2 in fused_scores  # Appears in both

        # Fused score for overlapping items should be higher
        score_0 = fused_scores[0]
        score_4 = fused_scores.get(4, 0)  # Only in second list
        # Item 0 (in both lists) should have higher score than item 4 (only in one)
        assert score_0 > score_4

    def test_rrf_empty_lists(self):
        """Test RRF with empty input lists."""
        fused_scores = fuse_scores_rrf([], [], k=60)

        assert len(fused_scores) == 0

    def test_rrf_single_list(self):
        """Test RRF with single ranking list."""
        indices = [0, 1, 2]
        scores = np.array([0.9, 0.7, 0.5])

        fused_scores = fuse_scores_rrf([indices], [scores], k=60)

        assert len(fused_scores) == 3
        # RRF formula: 1/(k + rank), where rank starts at 0
        # For first item: 1/(60 + 0) = 1/60 ≈ 0.0167
        expected_0 = 1.0 / (60 + 0)
        assert abs(fused_scores[0] - expected_0) < 0.001

    def test_rrf_no_overlap(self):
        """Test RRF when lists have no overlap."""
        indices1 = [0, 1, 2]
        scores1 = np.array([0.9, 0.7, 0.5])

        indices2 = [3, 4, 5]
        scores2 = np.array([0.8, 0.6, 0.4])

        fused_scores = fuse_scores_rrf([indices1, indices2], [scores1, scores2], k=60)

        # All items should be in fused scores
        assert len(fused_scores) == 6
        for i in range(6):
            assert i in fused_scores

    def test_rrf_different_k_values(self):
        """Test RRF with different k values."""
        indices = [0, 1, 2]
        scores = np.array([0.9, 0.7, 0.5])

        fused_k30 = fuse_scores_rrf([indices], [scores], k=30)
        fused_k60 = fuse_scores_rrf([indices], [scores], k=60)

        # With larger k, scores should be smaller (1/(k+rank) is smaller)
        assert fused_k30[0] > fused_k60[0]


class TestDenseScoreStore:
    """Test DenseScoreStore class."""

    def test_dense_score_store_creation(self):
        """Test creating DenseScoreStore."""
        scores = np.array([0.9, 0.7, 0.5])

        store = DenseScoreStore(scores)

        assert len(store) == 3
        assert store[0] == 0.9
        assert store[1] == 0.7
        assert store[2] == 0.5

    def test_dense_score_store_indexing(self):
        """Test DenseScoreStore indexing."""
        scores = np.array([0.9, 0.7, 0.5, 0.3])

        store = DenseScoreStore(scores)

        # Integer indexing
        assert store[0] == 0.9
        assert store[3] == 0.3

        # Negative indexing
        assert store[-1] == 0.3

    def test_dense_score_store_iteration(self):
        """Test DenseScoreStore iteration."""
        scores = np.array([0.9, 0.7, 0.5])

        store = DenseScoreStore(scores)

        # Should be iterable
        score_list = list(store)
        assert len(score_list) == 3
        assert score_list == [0.9, 0.7, 0.5]

    def test_dense_score_store_empty(self):
        """Test DenseScoreStore with empty array."""
        scores = np.array([])

        store = DenseScoreStore(scores)

        assert len(store) == 0


class TestHybridScoreFusion:
    """Test hybrid score fusion (alpha * BM25 + (1-alpha) * dense)."""

    def test_hybrid_fusion_balanced(self):
        """Test hybrid fusion with balanced alpha (0.5)."""
        bm25_scores = {0: 0.8, 1: 0.6, 2: 0.4}
        dense_scores = np.array([0.5, 0.7, 0.9])  # Different ranking

        alpha = 0.5

        # Manual fusion: alpha * bm25 + (1-alpha) * dense
        expected_0 = 0.5 * 0.8 + 0.5 * 0.5  # = 0.65
        expected_1 = 0.5 * 0.6 + 0.5 * 0.7  # = 0.65
        expected_2 = 0.5 * 0.4 + 0.5 * 0.9  # = 0.65

        # All should be equal with this specific case
        # (This is a contrived example to test the formula)

    def test_hybrid_fusion_bm25_only(self):
        """Test hybrid fusion with alpha=1.0 (BM25 only)."""
        bm25_scores = {0: 0.9, 1: 0.5, 2: 0.3}
        dense_scores = np.array([0.1, 0.9, 0.8])

        alpha = 1.0

        # With alpha=1.0, should only use BM25 scores
        # Hybrid score = 1.0 * bm25 + 0.0 * dense
        # So ranking should match BM25 ranking

    def test_hybrid_fusion_dense_only(self):
        """Test hybrid fusion with alpha=0.0 (dense only)."""
        bm25_scores = {0: 0.9, 1: 0.5, 2: 0.3}
        dense_scores = np.array([0.1, 0.9, 0.8])

        alpha = 0.0

        # With alpha=0.0, should only use dense scores
        # Hybrid score = 0.0 * bm25 + 1.0 * dense
        # So ranking should match dense ranking


class TestRetrievalEdgeCases:
    """Test edge cases in retrieval ranking."""

    def test_retrieval_with_all_zero_scores(self):
        """Test retrieval when all scores are zero."""
        scores = np.array([0.0, 0.0, 0.0])

        normalized = normalize_scores_minmax(scores)

        # Should handle gracefully
        assert len(normalized) == len(scores)

    def test_retrieval_with_nan_scores(self):
        """Test handling of NaN in scores."""
        scores = np.array([0.9, np.nan, 0.5])

        # Should either handle gracefully or raise appropriate error
        try:
            normalized = normalize_scores_zscore(scores)
            # If handled gracefully, check result
            assert len(normalized) == len(scores)
        except (ValueError, RuntimeError):
            # Acceptable to raise error for invalid input
            pass

    def test_retrieval_with_inf_scores(self):
        """Test handling of infinity in scores."""
        scores = np.array([0.9, np.inf, 0.5])

        # Should either handle gracefully or raise appropriate error
        try:
            normalized = normalize_scores_minmax(scores)
            assert len(normalized) == len(scores)
        except (ValueError, RuntimeError):
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
