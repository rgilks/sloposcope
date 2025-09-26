"""
Verbosity analysis for AI slop detection.

Calculates wordiness, sentence length, filler words, and structural complexity.
"""

import logging
import re
import statistics
from typing import Any

logger = logging.getLogger(__name__)


def calculate_words_per_sentence(sentences: list[str]) -> float:
    """Calculate average words per sentence."""
    if not sentences:
        return 0.0

    word_counts = []
    for sentence in sentences:
        words = sentence.split()
        word_counts.append(len(words))

    return statistics.mean(word_counts) if word_counts else 0.0


def calculate_filler_words_ratio(text: str, tokens: list[str]) -> float:
    """Calculate ratio of filler/discourse words."""
    # Common filler and discourse words
    filler_words = {
        "actually",
        "basically",
        "essentially",
        "literally",
        "really",
        "very",
        "quite",
        "rather",
        "somewhat",
        "kind of",
        "sort of",
        "you know",
        "like",
        "well",
        "so",
        "then",
        "now",
        "here",
        "there",
        "just",
        "even",
        "also",
        "and",
        "but",
        "or",
        "however",
        "moreover",
        "furthermore",
        "therefore",
        "thus",
        "hence",
        "consequently",
    }

    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0.0

    filler_count = sum(1 for token in tokens if token.lower() in filler_words)
    return filler_count / total_tokens


def calculate_listiness(text: str) -> float:
    """Calculate how much of the text consists of lists."""
    lines = text.split("\n")
    total_lines = len(lines)

    if total_lines == 0:
        return 0.0

    # Count lines that look like list items
    list_lines = 0
    for line in lines:
        line = line.strip()
        if line:
            # Check for bullet points, numbered lists, etc.
            if (
                line.startswith(
                    (
                        "-",
                        "*",
                        "•",
                        "1.",
                        "2.",
                        "3.",
                        "4.",
                        "5.",
                        "6.",
                        "7.",
                        "8.",
                        "9.",
                        "10.",
                    )
                )
                or re.match(r"^\d+\.", line)
                or re.match(r"^[a-zA-Z]\.", line)
            ):
                list_lines += 1

    return list_lines / total_lines


def calculate_sentence_length_variance(sentences: list[str]) -> float:
    """Calculate variance in sentence lengths."""
    if not sentences:
        return 0.0

    word_counts = []
    for sentence in sentences:
        words = sentence.split()
        word_counts.append(len(words))

    if len(word_counts) < 2:
        return 0.0

    return statistics.variance(word_counts)


def detect_verbosity_spans(text: str, tokens: list[str]) -> list[dict[str, Any]]:
    """Detect spans of verbose or wordy content."""
    spans = []

    # Look for long sentences (more than 25 words)
    sentences = text.split(".")
    current_pos = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) > 25:
            span_start = current_pos
            span_end = current_pos + len(sentence)
            spans.append(
                {
                    "start": span_start,
                    "end": span_end,
                    "type": "long_sentence",
                    "note": f"Very long sentence ({len(sentence.split())} words)",
                }
            )

        current_pos += len(sentence) + 1  # +1 for the period

    return spans


def extract_features(
    text: str, sentences: list[str], tokens: list[str]
) -> dict[str, Any]:
    """Extract all verbosity-related features."""
    try:
        # Calculate words per sentence
        words_per_sentence = calculate_words_per_sentence(sentences)

        # Calculate filler words ratio
        filler_ratio = calculate_filler_words_ratio(text, tokens)

        # Calculate listiness
        listiness = calculate_listiness(text)

        # Calculate sentence length variance
        sentence_variance = calculate_sentence_length_variance(sentences)

        # Detect verbosity spans
        verbosity_spans = detect_verbosity_spans(text, tokens)

        # Calculate overall verbosity score
        # Higher scores indicate more verbosity (worse)
        overall_verbosity = (
            min(1.0, words_per_sentence / 30.0) * 0.4  # Words per sentence (normalized)
            + filler_ratio * 0.3  # Filler words
            + listiness * 0.2  # Listiness
            + min(1.0, sentence_variance / 50.0) * 0.1  # Sentence length variance
        )

        return {
            "words_per_sentence": words_per_sentence,
            "filler_ratio": filler_ratio,
            "listiness": listiness,
            "sentence_variance": sentence_variance,
            "overall_verbosity": overall_verbosity,
            "verbosity_spans": verbosity_spans,
        }

    except Exception as e:
        logger.error(f"Error in verbosity feature extraction: {e}")
        return {
            "words_per_sentence": 0.0,
            "filler_ratio": 0.0,
            "listiness": 0.0,
            "sentence_variance": 0.0,
            "overall_verbosity": 0.0,
            "verbosity_spans": [],
        }
