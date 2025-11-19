"""Hallucination detection using Natural Language Inference (NLI).

Provides lightweight hallucination detection by checking if LLM answers
are entailed by (logically follow from) the source documents.

This is OPTIONAL and requires the sentence-transformers library with NLI models.
If not available, the system gracefully degrades.

Features:
- Lightweight NLI model (DeBERTa-v3-small, ~140MB)
- Batch processing for efficiency
- Configurable thresholds
- Graceful degradation if model not available

Usage:
    from clockify_rag.hallucination_detector import detect_hallucination

    result = detect_hallucination(
        answer="Time tracking is done via the timer.",
        source_texts=["Click the timer icon to start tracking time."],
        threshold=0.5
    )

    if result["likely_hallucination"]:
        print(f"Warning: Low entailment score {result['score']}")
"""

import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# Global state for lazy-loaded NLI model
_NLI_MODEL = None
_NLI_AVAILABLE = None


def _check_nli_availability() -> bool:
    """Check if NLI models are available (sentence-transformers library).

    Returns:
        True if NLI models can be loaded, False otherwise
    """
    global _NLI_AVAILABLE

    if _NLI_AVAILABLE is not None:
        return _NLI_AVAILABLE

    try:
        from sentence_transformers import CrossEncoder

        _NLI_AVAILABLE = True
        return True
    except ImportError:
        logger.info(
            "Hallucination detection unavailable: sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )
        _NLI_AVAILABLE = False
        return False


def _load_nli_model():
    """Lazy-load NLI model for hallucination detection.

    Uses cross-encoder/nli-deberta-v3-small (~140MB):
    - Lightweight and fast
    - Good accuracy for entailment detection
    - Predicts: entailment, contradiction, neutral

    Returns:
        CrossEncoder model or None if unavailable
    """
    global _NLI_MODEL

    if _NLI_MODEL is not None:
        return _NLI_MODEL

    if not _check_nli_availability():
        return None

    try:
        from sentence_transformers import CrossEncoder

        # Use small NLI model for fast inference
        _NLI_MODEL = CrossEncoder("cross-encoder/nli-deberta-v3-small")
        logger.info("Loaded NLI model: cross-encoder/nli-deberta-v3-small (~140MB)")
        return _NLI_MODEL

    except Exception as e:
        logger.warning(f"Failed to load NLI model: {e}")
        _NLI_MODEL = None
        return None


def compute_entailment_score(
    answer: str,
    source_texts: List[str],
    model=None,
) -> Optional[float]:
    """Compute entailment score between answer and source texts.

    Uses NLI model to predict if answer is entailed by (logically follows from)
    the source texts. Higher score = more likely the answer is grounded in sources.

    Args:
        answer: LLM answer to check
        source_texts: Source document texts
        model: Pre-loaded NLI model (optional, will load if None)

    Returns:
        Entailment score (0-1) or None if model unavailable
        - 0.0-0.3: Likely hallucination (contradiction/not entailed)
        - 0.3-0.6: Uncertain (neutral)
        - 0.6-1.0: Likely grounded (entailed)

    Example:
        >>> score = compute_entailment_score(
        ...     "Time tracking requires a timer.",
        ...     ["Use the timer to track time."]
        ... )
        >>> score > 0.6  # High entailment
        True
    """
    if model is None:
        model = _load_nli_model()

    if model is None:
        logger.debug("Entailment check skipped (NLI model not available)")
        return None

    if not source_texts:
        logger.warning("No source texts provided for entailment check")
        return None

    try:
        # Create pairs: (source, answer) for each source
        # Format: premise (source) → hypothesis (answer)
        pairs = [[source, answer] for source in source_texts]

        # Predict entailment for all pairs (batch processing)
        # Returns logits: [contradiction, entailment, neutral]
        scores = model.predict(pairs)

        # Extract entailment scores (index 1)
        # scores is array of shape (num_pairs, 3) with logits
        # We want the maximum entailment score across all sources
        import numpy as np

        scores_array = np.array(scores)

        # Apply softmax to convert logits to probabilities
        exp_scores = np.exp(scores_array)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        # Get entailment probabilities (column 1)
        entailment_probs = probs[:, 1]

        # Return maximum entailment probability across all sources
        # (if any source strongly entails answer, it's not hallucination)
        max_entailment = float(np.max(entailment_probs))

        logger.debug(
            f"Entailment scores: mean={np.mean(entailment_probs):.3f}, "
            f"max={max_entailment:.3f}, "
            f"min={np.min(entailment_probs):.3f}"
        )

        return max_entailment

    except Exception as e:
        logger.warning(f"Entailment computation failed: {e}")
        return None


def detect_hallucination(
    answer: str,
    source_texts: List[str],
    threshold: float = 0.5,
    enable_detection: bool = True,
) -> Dict[str, Any]:
    """Detect potential hallucination in LLM answer.

    Checks if answer is entailed by (logically follows from) source texts
    using NLI model.

    Args:
        answer: LLM answer to check
        source_texts: Source document texts that should support the answer
        threshold: Entailment threshold (default: 0.5)
                  Below this = likely hallucination
        enable_detection: Enable/disable detection (default: True)

    Returns:
        Dict with detection results:
        - enabled: Whether detection was enabled
        - available: Whether NLI model is available
        - score: Entailment score (0-1) or None
        - threshold: Threshold used
        - likely_hallucination: Boolean, True if score < threshold
        - confidence: How confident the detector is (based on score distance from threshold)

    Example:
        >>> result = detect_hallucination(
        ...     "The product costs $1000.",
        ...     ["Our pricing starts at $10."],
        ...     threshold=0.5
        ... )
        >>> result['likely_hallucination']
        True
        >>> result['score']
        0.15  # Low entailment = likely fabricated
    """
    if not enable_detection:
        return {
            "enabled": False,
            "available": None,
            "score": None,
            "threshold": threshold,
            "likely_hallucination": None,
            "confidence": None,
        }

    # Check if NLI model is available
    if not _check_nli_availability():
        return {
            "enabled": True,
            "available": False,
            "score": None,
            "threshold": threshold,
            "likely_hallucination": None,
            "confidence": None,
        }

    # Compute entailment score
    score = compute_entailment_score(answer, source_texts)

    if score is None:
        return {
            "enabled": True,
            "available": True,
            "score": None,
            "threshold": threshold,
            "likely_hallucination": None,
            "confidence": None,
        }

    # Detect hallucination based on threshold
    likely_hallucination = score < threshold

    # Compute confidence: distance from threshold
    # Far from threshold = high confidence, close = low confidence
    distance_from_threshold = abs(score - threshold)
    confidence = min(100, distance_from_threshold * 200)  # Scale to 0-100

    logger.info(
        f"Hallucination detection: score={score:.3f}, threshold={threshold}, "
        f"likely_hallucination={likely_hallucination}, confidence={confidence:.1f}"
    )

    return {
        "enabled": True,
        "available": True,
        "score": score,
        "threshold": threshold,
        "likely_hallucination": likely_hallucination,
        "confidence": confidence,
    }


def check_answer_consistency(
    answer: str,
    source_chunks: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Check if answer is consistent with source chunks (convenience wrapper).

    Args:
        answer: LLM answer text
        source_chunks: List of chunk dicts with 'text' field
        threshold: Entailment threshold (default: 0.5)

    Returns:
        Detection results from detect_hallucination()

    Example:
        >>> chunks = [{"text": "Use timer to track time."}]
        >>> result = check_answer_consistency(
        ...     "Time tracking requires a timer.",
        ...     chunks
        ... )
        >>> result['likely_hallucination']
        False
    """
    # Extract text from chunks
    source_texts = [chunk.get("text", "") for chunk in source_chunks if chunk.get("text")]

    return detect_hallucination(answer, source_texts, threshold=threshold)
