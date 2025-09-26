"""
Fluency feature extraction for AI slop analysis.

Measures the naturalness and fluency of language.
Based on the paper's taxonomy where Fluency (SQ4) measures language naturalness.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def detect_grammar_errors(text: str) -> list[dict[str, Any]]:
    """
    Detect basic grammar errors and unnatural constructions.

    Args:
        text: Text to analyze

    Returns:
        List of grammar error spans
    """
    errors = []

    # Common grammar error patterns
    error_patterns = [
        # Subject-verb disagreement
        (r"\b(?:he|she|it)\s+(?:are|were)\b", "subject_verb_disagreement"),
        (r"\b(?:they|we|you)\s+(?:is|was)\b", "subject_verb_disagreement"),
        # Double negatives
        (
            r"\b(?:don\'t|doesn\'t|didn\'t|won\'t|can\'t|shouldn\'t|wouldn\'t)\s+\w*\s+(?:no|not|never|nothing|nobody|nowhere)\b",
            "double_negative",
        ),
        # Missing articles
        (
            r"\b(?:is|are|was|were)\s+(?:good|bad|important|necessary|possible|difficult|easy)\s+(?:idea|thing|way|time|place)\b",
            "missing_article",
        ),
        # Unnatural word order
        (
            r"\b(?:very|really|quite|rather|somewhat|fairly)\s+(?:very|really|quite|rather|somewhat|fairly)\b",
            "redundant_intensifier",
        ),
        # Awkward constructions
        (
            r"\b(?:in order to|so as to)\s+(?:in order to|so as to)\b",
            "redundant_purpose",
        ),
        (
            r"\b(?:due to the fact that|because of the fact that)\b",
            "wordy_construction",
        ),
        # Repetitive constructions
        (r"\b(\w+)\s+\1\s+\1\b", "word_repetition"),
    ]

    for pattern, error_type in error_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            errors.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "type": error_type,
                }
            )

    return errors


def detect_unnatural_phrases(text: str) -> list[dict[str, Any]]:
    """
    Detect phrases that sound unnatural or AI-generated.

    Args:
        text: Text to analyze

    Returns:
        List of unnatural phrase spans
    """
    unnatural_patterns = [
        # Overly formal or stilted language
        r"\b(?:it is important to note that|it should be noted that|it is worth noting that)\b",
        r"\b(?:in the context of|within the framework of|in terms of the fact that)\b",
        r"\b(?:with regard to|in relation to|pertaining to)\b",
        # Redundant phrases
        r"\b(?:each and every|first and foremost|various different|completely finished|totally complete)\b",
        r"\b(?:free gift|past history|future plans|new innovation|unexpected surprise)\b",
        # Awkward transitions
        r"\b(?:moving forward|going forward|at this point in time|in today\'s society)\b",
        r"\b(?:it goes without saying|needless to say|it is clear that)\b",
        # Overused AI phrases
        r"\b(?:i understand|i can help|i\'m here to assist|let me help you|i hope this helps)\b",
        r"\b(?:please let me know|feel free to ask|don\'t hesitate to|if you have any questions)\b",
    ]

    unnatural_spans = []
    for pattern in unnatural_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            unnatural_spans.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "type": "unnatural_phrase",
                }
            )

    return unnatural_spans


def detect_sentence_fragments(sentences: list[str]) -> list[dict[str, Any]]:
    """
    Detect sentence fragments and incomplete sentences.

    Args:
        sentences: List of sentences

    Returns:
        List of sentence fragment information
    """
    fragments = []

    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check for incomplete sentences
        fragment_indicators = [
            # Starting with lowercase (unless proper noun)
            sentence[0].islower() and not sentence.split()[0].istitle(),
            # Missing subject or verb
            not re.search(
                r"\b(?:i|you|he|she|it|we|they|this|that|these|those)\b",
                sentence,
                re.IGNORECASE,
            )
            and not re.search(
                r"\b(?:is|are|was|were|has|have|had|will|would|can|could|should|must)\b",
                sentence,
                re.IGNORECASE,
            ),
            # Ending with conjunction
            sentence.rstrip(".,!?").endswith(
                ("and", "but", "or", "so", "yet", "for", "nor")
            ),
            # Very short sentences without clear structure
            len(sentence.split()) < 3 and not sentence.endswith(("!", "?")),
        ]

        if any(fragment_indicators):
            fragments.append(
                {
                    "sentence_index": i,
                    "text": sentence,
                    "type": "sentence_fragment",
                }
            )

    return fragments


def calculate_fluency_score(
    text: str,
    sentences: list[str],
    grammar_errors: list[dict],
    unnatural_phrases: list[dict],
    fragments: list[dict],
) -> float:
    """
    Calculate overall fluency score.

    Args:
        text: Full text
        sentences: List of sentences
        grammar_errors: List of grammar errors
        unnatural_phrases: List of unnatural phrases
        fragments: List of sentence fragments

    Returns:
        Fluency score (0-1, higher = more fluent)
    """
    if not text.strip() or not sentences:
        return 0.5

    # Base score
    score = 0.8  # Assume text is somewhat fluent by default

    # Penalize grammar errors
    grammar_penalty = len(grammar_errors) * 0.1
    score -= min(grammar_penalty, 0.4)

    # Penalize unnatural phrases
    unnatural_penalty = len(unnatural_phrases) * 0.05
    score -= min(unnatural_penalty, 0.3)

    # Penalize sentence fragments
    fragment_penalty = len(fragments) * 0.08
    score -= min(fragment_penalty, 0.3)

    # Check for good sentence variety
    sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
    if sentence_lengths:
        length_variance = sum(
            (length - sum(sentence_lengths) / len(sentence_lengths)) ** 2
            for length in sentence_lengths
        ) / len(sentence_lengths)
        # Moderate variance is good for fluency
        if 5 < length_variance < 50:
            score += 0.05
        elif length_variance > 100:  # Too much variance
            score -= 0.05

    # Check for proper punctuation
    punctuation_score = 0
    for sentence in sentences:
        if sentence.strip() and sentence.strip()[-1] in ".!?":
            punctuation_score += 1

    if sentences:
        punctuation_ratio = punctuation_score / len(sentences)
        if punctuation_ratio > 0.8:
            score += 0.05
        elif punctuation_ratio < 0.5:
            score -= 0.1

    # Normalize score
    return max(0.0, min(1.0, score))


def extract_features(
    text: str, sentences: list[str], tokens: list[str]
) -> dict[str, Any]:
    """
    Extract all fluency-related features.

    Args:
        text: Full text to analyze
        sentences: List of sentences
        tokens: List of tokens

    Returns:
        Dictionary with fluency features
    """
    try:
        # Detect fluency issues
        grammar_errors = detect_grammar_errors(text)
        unnatural_phrases = detect_unnatural_phrases(text)
        fragments = detect_sentence_fragments(sentences)

        # Calculate overall fluency score
        fluency_score = calculate_fluency_score(
            text, sentences, grammar_errors, unnatural_phrases, fragments
        )

        # Calculate additional metrics
        grammar_error_ratio = len(grammar_errors) / max(len(sentences), 1)
        unnatural_phrase_ratio = len(unnatural_phrases) / max(len(sentences), 1)
        fragment_ratio = len(fragments) / max(len(sentences), 1)

        return {
            "grammar_errors_count": len(grammar_errors),
            "unnatural_phrases_count": len(unnatural_phrases),
            "fragments_count": len(fragments),
            "grammar_error_ratio": grammar_error_ratio,
            "unnatural_phrase_ratio": unnatural_phrase_ratio,
            "fragment_ratio": fragment_ratio,
            "fluency_score": fluency_score,
            "grammar_errors": grammar_errors,
            "unnatural_phrases": unnatural_phrases,
            "fragments": fragments,
            "value": fluency_score,  # For compatibility with combine.py
        }

    except Exception as e:
        logger.error(f"Error in fluency feature extraction: {e}")
        return {
            "grammar_errors_count": 0,
            "unnatural_phrases_count": 0,
            "fragments_count": 0,
            "grammar_error_ratio": 0.0,
            "unnatural_phrase_ratio": 0.0,
            "fragment_ratio": 0.0,
            "fluency_score": 0.5,
            "grammar_errors": [],
            "unnatural_phrases": [],
            "fragments": [],
            "value": 0.5,
        }
