"""Citation validation module for verifying answer provenance.

Provides multiple citation extraction strategies and semantic relevance checking
to ensure LLM answers are properly grounded in source documents.

Features:
- Multiple citation format parsers (fallback strategy)
- Semantic relevance checking with cosine similarity
- Citation confidence scoring
- Support for various citation formats: [id1, id2], (id1), {id1}, etc.
"""

import logging
import re
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


def extract_citations_regex(answer: str) -> List[str]:
    """Extract citations using regex patterns (primary method).

    Supports formats:
    - [id_123, id_456]
    - [id_123]
    - id_123, id_456 (without brackets)

    Args:
        answer: LLM answer text

    Returns:
        List of extracted citation IDs

    Example:
        >>> extract_citations_regex("Answer based on [id_123, id_456].")
        ['id_123', 'id_456']
    """
    # Pattern 1: Standard bracket format [id1, id2, ...]
    bracket_pattern = r"\[([^\]]+)\]"
    matches = re.findall(bracket_pattern, answer)

    citations = []
    for match in matches:
        # Split by comma and clean
        parts = match.split(",")
        for part in parts:
            cleaned = part.strip()
            # Check if it looks like an ID (alphanumeric with underscores/hyphens)
            if re.match(r"^[a-zA-Z0-9_-]+$", cleaned):
                citations.append(cleaned)

    return citations


def extract_citations_parentheses(answer: str) -> List[str]:
    """Extract citations from parentheses format (fallback method 1).

    Supports formats:
    - (id_123)
    - (id_123, id_456)

    Args:
        answer: LLM answer text

    Returns:
        List of extracted citation IDs
    """
    pattern = r"\(([^\)]+)\)"
    matches = re.findall(pattern, answer)

    citations = []
    for match in matches:
        parts = match.split(",")
        for part in parts:
            cleaned = part.strip()
            if re.match(r"^[a-zA-Z0-9_-]+$", cleaned) and ("id" in cleaned.lower() or "_" in cleaned):
                citations.append(cleaned)

    return citations


def extract_citations_curly_braces(answer: str) -> List[str]:
    """Extract citations from curly braces format (fallback method 2).

    Supports formats:
    - {id_123}
    - {id_123, id_456}

    Args:
        answer: LLM answer text

    Returns:
        List of extracted citation IDs
    """
    pattern = r"\{([^\}]+)\}"
    matches = re.findall(pattern, answer)

    citations = []
    for match in matches:
        parts = match.split(",")
        for part in parts:
            cleaned = part.strip()
            if re.match(r"^[a-zA-Z0-9_-]+$", cleaned) and ("id" in cleaned.lower() or "_" in cleaned):
                citations.append(cleaned)

    return citations


def extract_citations_inline(answer: str) -> List[str]:
    """Extract inline citations without brackets (fallback method 3).

    Looks for patterns like: "According to id_123, ..." or "Source: id_456"

    Args:
        answer: LLM answer text

    Returns:
        List of extracted citation IDs
    """
    # Pattern: word boundary, id pattern, word boundary
    pattern = r"\b((?:id_|chunk_|doc_)[a-zA-Z0-9_-]+)\b"
    matches = re.findall(pattern, answer, re.IGNORECASE)

    return list(set(matches))  # Remove duplicates


def extract_citations_multi_strategy(answer: str) -> List[str]:
    """Extract citations using multiple strategies with fallback.

    Tries strategies in order:
    1. Regex bracket format (primary)
    2. Parentheses format
    3. Curly braces format
    4. Inline format (no delimiters)

    Args:
        answer: LLM answer text

    Returns:
        List of extracted citation IDs (deduplicated)

    Example:
        >>> extract_citations_multi_strategy("Based on [id_123] and (id_456)")
        ['id_123', 'id_456']
    """
    all_citations = []

    # Strategy 1: Regex brackets (primary)
    citations_brackets = extract_citations_regex(answer)
    if citations_brackets:
        all_citations.extend(citations_brackets)
        logger.debug(f"Found {len(citations_brackets)} citations using bracket format")

    # Strategy 2: Parentheses
    citations_parens = extract_citations_parentheses(answer)
    if citations_parens:
        all_citations.extend(citations_parens)
        logger.debug(f"Found {len(citations_parens)} citations using parentheses format")

    # Strategy 3: Curly braces
    citations_curly = extract_citations_curly_braces(answer)
    if citations_curly:
        all_citations.extend(citations_curly)
        logger.debug(f"Found {len(citations_curly)} citations using curly braces format")

    # Strategy 4: Inline (only if no other citations found)
    if not all_citations:
        citations_inline = extract_citations_inline(answer)
        if citations_inline:
            all_citations.extend(citations_inline)
            logger.debug(f"Found {len(citations_inline)} citations using inline format (fallback)")

    # Deduplicate while preserving order
    seen = set()
    unique_citations = []
    for cit in all_citations:
        if cit not in seen:
            seen.add(cit)
            unique_citations.append(cit)

    logger.info(f"Extracted {len(unique_citations)} unique citations from answer")
    return unique_citations


def validate_citation_ids(
    citations: List[str],
    valid_chunk_ids: List[str],
) -> Tuple[bool, List[str], List[str]]:
    """Validate that citations reference actual chunks in context.

    Args:
        citations: List of citation IDs from answer
        valid_chunk_ids: List of chunk IDs that were in the context

    Returns:
        Tuple of (is_valid, valid_citations, invalid_citations)

    Example:
        >>> validate_citation_ids(['id_1', 'id_2', 'id_99'], ['id_1', 'id_2'])
        (False, ['id_1', 'id_2'], ['id_99'])
    """
    # Normalize to strings for comparison
    valid_set = set(str(cid) for cid in valid_chunk_ids)

    valid_citations = [cid for cid in citations if cid in valid_set]
    invalid_citations = [cid for cid in citations if cid not in valid_set]

    is_valid = len(invalid_citations) == 0

    return is_valid, valid_citations, invalid_citations


def compute_semantic_relevance(
    answer: str,
    cited_chunk_texts: List[str],
    answer_embedding: Optional[np.ndarray] = None,
    chunk_embeddings: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """Compute semantic relevance between answer and cited chunks.

    Uses cosine similarity to check if answer is semantically related to
    the chunks it cites (helps detect hallucination).

    Args:
        answer: LLM answer text
        cited_chunk_texts: Text of chunks that were cited
        answer_embedding: Pre-computed embedding of answer (optional)
        chunk_embeddings: Pre-computed embeddings of chunks (optional)

    Returns:
        Dict with relevance metrics:
        - mean_similarity: Average cosine similarity
        - max_similarity: Maximum cosine similarity
        - min_similarity: Minimum cosine similarity
        - all_above_threshold: Boolean, True if all > 0.3

    Note:
        If embeddings not provided, returns None (semantic check skipped)
    """
    if answer_embedding is None or chunk_embeddings is None or not chunk_embeddings:
        logger.debug("Semantic relevance check skipped (embeddings not provided)")
        return {
            "mean_similarity": None,
            "max_similarity": None,
            "min_similarity": None,
            "all_above_threshold": None,
        }

    # Normalize embeddings
    answer_norm = answer_embedding / (np.linalg.norm(answer_embedding) + 1e-9)

    similarities = []
    for chunk_emb in chunk_embeddings:
        chunk_norm = chunk_emb / (np.linalg.norm(chunk_emb) + 1e-9)
        sim = float(np.dot(answer_norm, chunk_norm))
        similarities.append(sim)

    if not similarities:
        return {
            "mean_similarity": None,
            "max_similarity": None,
            "min_similarity": None,
            "all_above_threshold": None,
        }

    mean_sim = float(np.mean(similarities))
    max_sim = float(np.max(similarities))
    min_sim = float(np.min(similarities))

    # Threshold for "relevant" citation: cosine similarity > 0.3
    threshold = 0.3
    all_above_threshold = all(s > threshold for s in similarities)

    logger.debug(
        f"Semantic relevance: mean={mean_sim:.3f}, max={max_sim:.3f}, "
        f"min={min_sim:.3f}, all_above_{threshold}={all_above_threshold}"
    )

    return {
        "mean_similarity": mean_sim,
        "max_similarity": max_sim,
        "min_similarity": min_sim,
        "all_above_threshold": all_above_threshold,
    }


def compute_citation_confidence(
    num_citations: int,
    num_valid_citations: int,
    semantic_relevance: Optional[Dict[str, float]] = None,
) -> float:
    """Compute overall confidence score for citations (0-100).

    Factors:
    - Coverage: percentage of valid citations
    - Semantic relevance: if available, boost/penalize based on similarity

    Args:
        num_citations: Total citations found
        num_valid_citations: Number of valid citations (in context)
        semantic_relevance: Optional semantic relevance metrics

    Returns:
        Confidence score (0-100)

    Example:
        >>> compute_citation_confidence(3, 3, {"mean_similarity": 0.7})
        85.0
    """
    if num_citations == 0:
        return 0.0

    # Base score: percentage of valid citations
    validity_ratio = num_valid_citations / num_citations
    base_score = validity_ratio * 100

    # Adjust based on semantic relevance
    if semantic_relevance and semantic_relevance.get("mean_similarity") is not None:
        mean_sim = semantic_relevance["mean_similarity"]

        # Boost if high similarity (>0.5), penalize if low (<0.3)
        if mean_sim > 0.5:
            boost = min(15, (mean_sim - 0.5) * 30)  # Up to +15 points
            base_score = min(100, base_score + boost)
        elif mean_sim < 0.3:
            penalty = (0.3 - mean_sim) * 30  # Up to -9 points
            base_score = max(0, base_score - penalty)

    return round(base_score, 1)


def validate_citations_comprehensive(
    answer: str,
    context_chunk_ids: List[str],
    context_chunk_texts: Optional[List[str]] = None,
    answer_embedding: Optional[np.ndarray] = None,
    chunk_embeddings: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """Comprehensive citation validation with multiple strategies.

    Args:
        answer: LLM answer text
        context_chunk_ids: IDs of chunks in context
        context_chunk_texts: Text of chunks in context (for semantic check)
        answer_embedding: Pre-computed answer embedding (optional)
        chunk_embeddings: Pre-computed chunk embeddings (optional)

    Returns:
        Dict with validation results:
        - citations: List of extracted citation IDs
        - valid_citations: List of valid citations
        - invalid_citations: List of invalid citations
        - is_valid: Boolean, True if all citations are valid
        - confidence: Citation confidence score (0-100)
        - semantic_relevance: Semantic relevance metrics (if embeddings provided)

    Example:
        >>> result = validate_citations_comprehensive(
        ...     "Based on [id_1, id_2].",
        ...     context_chunk_ids=['id_1', 'id_2', 'id_3']
        ... )
        >>> result['is_valid']
        True
        >>> result['confidence']
        100.0
    """
    # Extract citations using multi-strategy approach
    citations = extract_citations_multi_strategy(answer)

    # Validate citation IDs
    is_valid, valid_citations, invalid_citations = validate_citation_ids(citations, context_chunk_ids)

    # Compute semantic relevance (if embeddings provided)
    semantic_relevance = None
    if context_chunk_texts and answer_embedding is not None and chunk_embeddings:
        # Filter to only cited chunks
        cited_indices = [i for i, cid in enumerate(context_chunk_ids) if str(cid) in valid_citations]
        cited_chunk_texts = [context_chunk_texts[i] for i in cited_indices]
        cited_embeddings = [chunk_embeddings[i] for i in cited_indices]

        if cited_embeddings:
            semantic_relevance = compute_semantic_relevance(
                answer, cited_chunk_texts, answer_embedding, cited_embeddings
            )

    # Compute citation confidence
    confidence = compute_citation_confidence(len(citations), len(valid_citations), semantic_relevance)

    return {
        "citations": citations,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "is_valid": is_valid,
        "confidence": confidence,
        "semantic_relevance": semantic_relevance,
    }
