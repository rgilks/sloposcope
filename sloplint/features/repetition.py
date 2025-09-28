"""
Repetition detection for AI slop analysis.

Calculates various types of repetition in text including n-gram repetition,
sentence repetition, and compression ratio.
"""

import gzip
import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


def calculate_ngram_repetition(tokens: list[str], n: int) -> float:
    """Calculate n-gram repetition rate."""
    if len(tokens) < n:
        return 0.0

    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        ngrams.append(ngram)

    if not ngrams:
        return 0.0

    # Count n-gram frequencies
    ngram_counts = Counter(ngrams)
    total_ngrams = len(ngrams)

    # Calculate repetition rate
    repeated_ngrams = sum(count for count in ngram_counts.values() if count > 1)
    repetition_rate = repeated_ngrams / total_ngrams

    return repetition_rate


def calculate_sentence_repetition(sentences: list[str]) -> float:
    """Calculate sentence-level repetition using multiple strategies."""
    if len(sentences) < 2:
        return 0.0

    total_sentences = len(sentences)
    repetition_score = 0.0

    # Strategy 1: Exact sentence matches
    sentence_counts = Counter(sentences)
    exact_repeats = sum(count for count in sentence_counts.values() if count > 1)
    repetition_score += (exact_repeats / total_sentences) * 0.5

    # Strategy 2: Word-level repetition within sentences
    all_words = []
    for sentence in sentences:
        words = sentence.lower().split()
        all_words.extend(words)

    if all_words:
        word_counts = Counter(all_words)
        # Calculate how many words appear more than once
        repeated_words = sum(count for count in word_counts.values() if count > 1)
        word_repetition_rate = repeated_words / len(all_words)
        repetition_score += word_repetition_rate * 0.3

    # Strategy 3: N-gram repetition (bigrams)
    if len(sentences) >= 2:
        bigrams = []
        for sentence in sentences:
            words = sentence.lower().split()
            if len(words) >= 2:
                for i in range(len(words) - 1):
                    bigrams.append(f"{words[i]}_{words[i + 1]}")

        if bigrams:
            bigram_counts = Counter(bigrams)
            repeated_bigrams = sum(
                count for count in bigram_counts.values() if count > 1
            )
            bigram_repetition_rate = repeated_bigrams / len(bigrams)
            repetition_score += bigram_repetition_rate * 0.2

    return min(repetition_score, 1.0)


def calculate_compression_ratio(text: str) -> float:
    """Calculate gzip compression ratio as a measure of repetition."""
    if not text:
        return 1.0

    # Convert to bytes
    text_bytes = text.encode("utf-8")

    # Compress with gzip
    compressed_bytes = gzip.compress(text_bytes)

    # Calculate compression ratio
    compression_ratio = len(compressed_bytes) / len(text_bytes)

    # Lower ratio = more repetition (better compression)
    # Invert so higher score = more repetition
    repetition_score = 1.0 - compression_ratio

    return max(0.0, min(1.0, repetition_score))


def detect_repetition_spans(
    tokens: list[str], threshold: float = 0.1
) -> list[dict[str, Any]]:
    """Detect spans of repetitive content."""
    spans = []

    # Look for repeated 2-grams and 3-grams
    for n in [2, 3, 4]:
        if len(tokens) < n:
            continue

        seen_ngrams: dict[tuple[str, ...], int] = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])

            if ngram in seen_ngrams:
                # Found repetition
                prev_start = seen_ngrams[ngram]
                current_start = i

                # Create span for the repeated sequence
                span_start = prev_start * len(tokens[0])  # Rough character position
                span_end = (current_start + n) * len(tokens[0])

                spans.append(
                    {
                        "start": span_start,
                        "end": span_end,
                        "type": f"repeated_{n}gram",
                        "note": f"Repeated {n}-gram: {' '.join(tokens[i : i + n])}",
                    }
                )
            else:
                seen_ngrams[ngram] = i

    return spans


def extract_features(
    text: str, sentences: list[str], tokens: list[str]
) -> dict[str, Any]:
    """Extract all repetition-related features."""
    try:
        # Calculate n-gram repetition rates
        ngram_repetition = {}
        for n in [1, 2, 3, 4]:
            rate = calculate_ngram_repetition(tokens, n)
            ngram_repetition[n] = rate

        # Calculate sentence repetition
        sentence_repetition = calculate_sentence_repetition(sentences)

        # Calculate compression ratio
        compression_ratio = calculate_compression_ratio(text)

        # Detect repetition spans
        repetition_spans = detect_repetition_spans(tokens)

        # Calculate overall repetition score
        # Weighted combination of different repetition measures
        overall_repetition = (
            ngram_repetition[1] * 0.3  # Unigram repetition
            + ngram_repetition[2] * 0.25  # Bigram repetition
            + ngram_repetition[3] * 0.25  # Trigram repetition
            + ngram_repetition[4] * 0.2  # 4-gram repetition
            + sentence_repetition * 0.5  # Sentence repetition
            + compression_ratio * 0.3  # Compression ratio
        )

        return {
            "ngram_repetition": ngram_repetition,
            "sentence_repetition": sentence_repetition,
            "compression_ratio": compression_ratio,
            "overall_repetition": overall_repetition,
            "repetition_spans": repetition_spans,
        }

    except Exception as e:
        logger.error(f"Error in repetition feature extraction: {e}")
        return {
            "ngram_repetition": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
            "sentence_repetition": 0.0,
            "compression_ratio": 1.0,
            "overall_repetition": 0.0,
            "repetition_spans": [],
        }
