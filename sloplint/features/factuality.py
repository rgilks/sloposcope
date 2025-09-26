"""
Factuality feature extraction for AI slop analysis.

Measures the accuracy and factual correctness of text content.
Based on the paper's finding that Factuality has the highest agreement (AC₁=0.76).
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def detect_factual_claims(text: str) -> list[dict[str, Any]]:
    """
    Detect potential factual claims in text.

    Args:
        text: Text to analyze

    Returns:
        List of detected factual claims
    """
    claims = []

    # Patterns that often indicate factual claims
    factual_patterns = [
        r"\b(?:is|are|was|were|has|have|had|will|would|can|could|should|must)\s+(?:a|an|the)?\s*\w+",
        r"\b(?:according to|studies show|research indicates|data shows|evidence suggests)",
        r"\b(?:percent|%|\d+%|\d+\.\d+%)\b",
        r"\b(?:million|billion|thousand)\b",
        r"\b(?:in \d{4}|since \d{4}|during \d{4})\b",
        r"\b(?:proven|confirmed|verified|established|demonstrated)\b",
    ]

    for pattern in factual_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            claims.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "type": "factual_claim",
                    "pattern": pattern,
                }
            )

    return claims


def detect_hedging_language(text: str) -> list[dict[str, Any]]:
    """
    Detect hedging language that reduces factuality confidence.

    Args:
        text: Text to analyze

    Returns:
        List of hedging phrases
    """
    hedging_patterns = [
        r"\b(?:might|may|could|possibly|perhaps|maybe|likely|probably|seems|appears)\b",
        r"\b(?:i think|i believe|i feel|i assume|i guess|i suppose)\b",
        r"\b(?:some|many|most|often|usually|generally|typically)\b",
        r"\b(?:according to some|some studies|research suggests|it appears that)\b",
    ]

    hedging_spans = []
    for pattern in hedging_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            hedging_spans.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "type": "hedging",
                }
            )

    return hedging_spans


def detect_unsupported_claims(text: str) -> list[dict[str, Any]]:
    """
    Detect claims that lack supporting evidence or sources.

    Args:
        text: Text to analyze

    Returns:
        List of unsupported claims
    """
    unsupported_patterns = [
        r"\b(?:everyone knows|it is well known|obviously|clearly|undoubtedly|certainly)\b",
        r"\b(?:studies show|research proves|data confirms|evidence shows)\b(?!\s+(?:that|this|the))\b",
        r"\b(?:experts agree|scientists say|doctors recommend)\b(?!\s+(?:that|this|the))\b",
        r"\b(?:it has been proven|it is established|it is confirmed)\b",
    ]

    unsupported_spans = []
    for pattern in unsupported_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            unsupported_spans.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "type": "unsupported_claim",
                }
            )

    return unsupported_spans


def detect_contradictory_statements(text: str) -> list[dict[str, Any]]:
    """
    Detect potentially contradictory statements.

    Args:
        text: Text to analyze

    Returns:
        List of contradictory statements
    """
    contradiction_patterns = [
        r"\b(?:however|but|although|despite|nevertheless|on the other hand)\b.*\b(?:however|but|although|despite|nevertheless|on the other hand)\b",
        r"\b(?:always|never)\b.*\b(?:sometimes|occasionally|rarely)\b",
        r"\b(?:all|every|none)\b.*\b(?:some|many|most)\b",
    ]

    contradictions = []
    for pattern in contradiction_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            contradictions.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "type": "contradiction",
                }
            )

    return contradictions


def calculate_factuality_score(
    text: str,
    factual_claims: list[dict],
    hedging_spans: list[dict],
    unsupported_spans: list[dict],
    contradictions: list[dict],
) -> float:
    """
    Calculate overall factuality score.

    Args:
        text: Full text
        factual_claims: List of detected factual claims
        hedging_spans: List of hedging language
        unsupported_spans: List of unsupported claims
        contradictions: List of contradictions

    Returns:
        Factuality score (0-1, higher = more factual)
    """
    if not text.strip():
        return 0.5

    # Base score
    score = 0.7  # Assume text is somewhat factual by default

    # Penalize hedging language (reduces confidence)
    hedging_penalty = len(hedging_spans) * 0.05
    score -= min(hedging_penalty, 0.3)

    # Penalize unsupported claims
    unsupported_penalty = len(unsupported_spans) * 0.1
    score -= min(unsupported_penalty, 0.4)

    # Penalize contradictions
    contradiction_penalty = len(contradictions) * 0.15
    score -= min(contradiction_penalty, 0.5)

    # Reward presence of factual claims (but not too much)
    factual_bonus = min(len(factual_claims) * 0.02, 0.1)
    score += factual_bonus

    # Check for source citations or references
    citation_patterns = [
        r"\[[\d,\s]+\]",  # [1, 2, 3]
        r"\([A-Za-z\s]+,\s*\d{4}\)",  # (Author, 2024)
        r"\b(?:according to|as stated in|as reported by)\b",
    ]

    has_citations = any(
        re.search(pattern, text, re.IGNORECASE) for pattern in citation_patterns
    )
    if has_citations:
        score += 0.1

    # Normalize score
    return max(0.0, min(1.0, score))


def extract_features(
    text: str, sentences: list[str], tokens: list[str]
) -> dict[str, Any]:
    """
    Extract all factuality-related features.

    Args:
        text: Full text to analyze
        sentences: List of sentences
        tokens: List of tokens

    Returns:
        Dictionary with factuality features
    """
    try:
        # Detect various types of factual issues
        factual_claims = detect_factual_claims(text)
        hedging_spans = detect_hedging_language(text)
        unsupported_spans = detect_unsupported_claims(text)
        contradictions = detect_contradictory_statements(text)

        # Calculate overall factuality score
        factuality_score = calculate_factuality_score(
            text, factual_claims, hedging_spans, unsupported_spans, contradictions
        )

        # Calculate additional metrics
        total_claims = len(factual_claims)
        hedging_ratio = len(hedging_spans) / max(len(sentences), 1)
        unsupported_ratio = len(unsupported_spans) / max(len(sentences), 1)

        return {
            "factual_claims_count": total_claims,
            "hedging_count": len(hedging_spans),
            "unsupported_claims_count": len(unsupported_spans),
            "contradictions_count": len(contradictions),
            "hedging_ratio": hedging_ratio,
            "unsupported_ratio": unsupported_ratio,
            "factuality_score": factuality_score,
            "factual_claims": factual_claims,
            "hedging_spans": hedging_spans,
            "unsupported_spans": unsupported_spans,
            "contradiction_spans": contradictions,
            "value": factuality_score,  # For compatibility with combine.py
        }

    except Exception as e:
        logger.error(f"Error in factuality feature extraction: {e}")
        return {
            "factual_claims_count": 0,
            "hedging_count": 0,
            "unsupported_claims_count": 0,
            "contradictions_count": 0,
            "hedging_ratio": 0.0,
            "unsupported_ratio": 0.0,
            "factuality_score": 0.5,
            "factual_claims": [],
            "hedging_spans": [],
            "unsupported_spans": [],
            "contradiction_spans": [],
            "value": 0.5,
        }
