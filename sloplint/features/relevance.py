"""
Relevance feature extraction for AI slop analysis.

Measures how well text content aligns with its topic/context.
Based on the paper's finding that Relevance is the strongest predictor of slop (β=0.06).
"""

import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Global model instance for efficiency
_relevance_model = None


def get_relevance_model() -> Any:
    """Get or initialize the sentence transformer model for relevance calculation."""
    global _relevance_model
    if _relevance_model is None:
        try:
            # Suppress the loss_type warning during model loading
            import warnings
            from contextlib import redirect_stderr
            from io import StringIO
            
            # Capture stderr during model loading to suppress the warning
            stderr_capture = StringIO()
            with redirect_stderr(stderr_capture):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Use a lightweight model for efficiency
                    _relevance_model = SentenceTransformer("all-MiniLM-L6-v2")

            # Try to use GPU if available
            try:
                import torch

                if torch.cuda.is_available():
                    _relevance_model = _relevance_model.to("cuda")
                    logger.info("Using CUDA GPU for sentence transformers")
                elif torch.backends.mps.is_available():
                    _relevance_model = _relevance_model.to("mps")
                    logger.info("Using Apple Metal GPU for sentence transformers")
                else:
                    logger.info("Using CPU for sentence transformers")
            except Exception as e:
                logger.warning(f"Could not move model to GPU: {e}")

        except Exception as e:
            logger.warning(f"Could not load sentence transformer model: {e}")
            _relevance_model = None
    return _relevance_model


def calculate_sentence_relevance(
    sentences: list[str], topic_context: str | None = None
) -> dict[str, Any]:
    """
    Calculate relevance scores for each sentence.

    Args:
        sentences: List of sentences to analyze
        topic_context: Optional topic/context for comparison

    Returns:
        Dictionary with relevance metrics
    """
    if not sentences:
        return {
            "mean_similarity": 0.5,
            "min_similarity": 0.5,
            "low_relevance_ratio": 0.0,
            "relevance_variance": 0.0,
            "relevance_score": 0.5,
        }

    model = get_relevance_model()
    if model is None:
        # Fallback to simple heuristics
        return _calculate_relevance_heuristics(sentences, topic_context)

    try:
        # Encode sentences
        sentence_embeddings = model.encode(sentences)

        if topic_context:
            # Compare against topic context
            topic_embedding = model.encode([topic_context])
            similarities = cosine_similarity(
                sentence_embeddings, topic_embedding
            ).flatten()
        else:
            # Compare sentences to each other (coherence-based relevance)
            similarities = []
            for i, sent_emb in enumerate(sentence_embeddings):
                # Compare to other sentences
                other_embeddings = (
                    np.vstack([sentence_embeddings[:i], sentence_embeddings[i + 1 :]])
                    if len(sentence_embeddings) > 1
                    else sentence_embeddings
                )

                if len(other_embeddings) > 0:
                    sim_scores = cosine_similarity(
                        [sent_emb], other_embeddings
                    ).flatten()
                    similarities.append(np.mean(sim_scores))
                else:
                    similarities.append(0.5)

            similarities = np.array(similarities)

        # Calculate relevance metrics
        mean_sim = float(np.mean(similarities))
        min_sim = float(np.min(similarities))
        relevance_variance = float(np.var(similarities))

        # Count sentences with low relevance (below threshold)
        low_relevance_threshold = 0.3
        low_relevance_count = np.sum(similarities < low_relevance_threshold)
        low_relevance_ratio = float(low_relevance_count / len(similarities))

        # Overall relevance score (higher = more relevant)
        relevance_score = mean_sim * (1 - low_relevance_ratio)

        return {
            "mean_similarity": mean_sim,
            "min_similarity": min_sim,
            "low_relevance_ratio": low_relevance_ratio,
            "relevance_variance": relevance_variance,
            "relevance_score": relevance_score,
            "sentence_similarities": similarities.tolist(),
        }

    except Exception as e:
        logger.error(f"Error in sentence relevance calculation: {e}")
        return _calculate_relevance_heuristics(sentences, topic_context)


def _calculate_relevance_heuristics(
    sentences: list[str], topic_context: str | None = None
) -> dict[str, Any]:
    """
    Fallback relevance calculation using simple heuristics.

    Args:
        sentences: List of sentences
        topic_context: Optional topic context

    Returns:
        Dictionary with heuristic relevance metrics
    """
    if not sentences:
        return {
            "mean_similarity": 0.5,
            "min_similarity": 0.5,
            "low_relevance_ratio": 0.0,
            "relevance_variance": 0.0,
            "relevance_score": 0.5,
        }

    # Simple heuristics for relevance
    relevance_scores = []

    for sentence in sentences:
        score = 0.5  # Base score

        # Check for topic-related keywords if context provided
        if topic_context:
            topic_words = set(topic_context.lower().split())
            sentence_words = set(sentence.lower().split())
            overlap = len(topic_words.intersection(sentence_words))
            if len(topic_words) > 0:
                score += min(0.3, overlap / len(topic_words))

        # Check for off-topic indicators
        off_topic_phrases = [
            "i don't know",
            "i'm not sure",
            "i can't help",
            "i'm sorry",
            "unfortunately",
            "i apologize",
            "i cannot",
            "i'm unable",
        ]

        sentence_lower = sentence.lower()
        for phrase in off_topic_phrases:
            if phrase in sentence_lower:
                score -= 0.2

        # Check for generic responses
        generic_phrases = [
            "let me help",
            "i understand",
            "i can assist",
            "i'm here to help",
            "i hope this helps",
            "please let me know",
            "feel free to ask",
        ]

        for phrase in generic_phrases:
            if phrase in sentence_lower:
                score -= 0.1

        # Normalize score
        score = max(0.0, min(1.0, score))
        relevance_scores.append(score)

    relevance_scores = np.array(relevance_scores)

    return {
        "mean_similarity": float(np.mean(relevance_scores)),
        "min_similarity": float(np.min(relevance_scores)),
        "low_relevance_ratio": float(
            np.sum(relevance_scores < 0.3) / len(relevance_scores)
        ),
        "relevance_variance": float(np.var(relevance_scores)),
        "relevance_score": float(np.mean(relevance_scores)),
        "sentence_similarities": relevance_scores.tolist(),
    }


def detect_relevance_spans(
    text: str, sentences: list[str], threshold: float = 0.3
) -> list[dict[str, Any]]:
    """
    Detect spans of text with low relevance.

    Args:
        text: Full text
        sentences: List of sentences
        threshold: Relevance threshold for flagging

    Returns:
        List of relevance spans
    """
    relevance_data = calculate_sentence_relevance(sentences)
    similarities = relevance_data.get("sentence_similarities", [])

    spans = []
    current_pos = 0

    for sentence, similarity in zip(sentences, similarities, strict=False):
        if similarity < threshold:
            # Find sentence position in text
            start_pos = text.find(sentence, current_pos)
            if start_pos != -1:
                end_pos = start_pos + len(sentence)
                spans.append(
                    {
                        "start": start_pos,
                        "end": end_pos,
                        "text": sentence,
                        "relevance_score": similarity,
                        "type": "low_relevance",
                    }
                )
                current_pos = end_pos

    return spans


def extract_features(
    text: str, sentences: list[str], tokens: list[str], topic_context: str | None = None
) -> dict[str, Any]:
    """
    Extract all relevance-related features.

    Args:
        text: Full text to analyze
        sentences: List of sentences
        tokens: List of tokens
        topic_context: Optional topic/context for comparison

    Returns:
        Dictionary with relevance features
    """
    try:
        # Calculate sentence-level relevance
        relevance_data = calculate_sentence_relevance(sentences, topic_context)

        # Detect low-relevance spans
        relevance_spans = detect_relevance_spans(text, sentences)

        # Calculate overall relevance score
        # Lower scores indicate more slop (less relevant content)
        overall_relevance = relevance_data["relevance_score"]

        return {
            "mean_similarity": relevance_data["mean_similarity"],
            "min_similarity": relevance_data["min_similarity"],
            "low_relevance_ratio": relevance_data["low_relevance_ratio"],
            "relevance_variance": relevance_data["relevance_variance"],
            "relevance_score": overall_relevance,
            "relevance_spans": relevance_spans,
            "value": overall_relevance,  # For compatibility with combine.py
        }

    except Exception as e:
        logger.error(f"Error in relevance feature extraction: {e}")
        return {
            "mean_similarity": 0.5,
            "min_similarity": 0.5,
            "low_relevance_ratio": 0.0,
            "relevance_variance": 0.0,
            "relevance_score": 0.5,
            "relevance_spans": [],
            "value": 0.5,
        }
