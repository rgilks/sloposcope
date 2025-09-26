"""
Complexity feature extraction for AI slop analysis.

Measures unnecessarily complex vocabulary and readability.
Based on the paper's taxonomy where Word Complexity (SQ6) measures use of unnecessarily complex vocabulary.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Common complex words that have simpler alternatives
COMPLEXITY_MAPPINGS = {
    # Overly formal alternatives
    "utilize": "use",
    "facilitate": "help",
    "implement": "do",
    "commence": "start",
    "terminate": "end",
    "ascertain": "find out",
    "endeavor": "try",
    "procure": "get",
    "expedite": "speed up",
    "elucidate": "explain",
    "comprehend": "understand",
    "perceive": "see",
    "demonstrate": "show",
    "indicate": "show",
    "substantiate": "prove",
    "constitute": "are",
    "encompass": "include",
    "optimize": "improve",
    "leverage": "use",
    "synergy": "working together",
    "paradigm": "model",
    "methodology": "method",
    "infrastructure": "system",
    "implementation": "putting into practice",
    "functionality": "features",
    "capabilities": "abilities",
    "parameters": "settings",
    "criteria": "standards",
    "prerequisites": "requirements",
    "prerequisite": "requirement",
    "subsequently": "later",
    "consequently": "so",
    "furthermore": "also",
    "moreover": "also",
    "nevertheless": "but",
    "notwithstanding": "despite",
    "aforementioned": "mentioned above",
    "herein": "here",
    "therein": "there",
    "whereby": "by which",
    "wherein": "in which",
    "whereof": "of which",
    "whereto": "to which",
    "whereupon": "upon which",
    "wherewith": "with which",
    "wherewithal": "means",
    "whereabouts": "location",
    "wherefore": "why",
    "whereas": "while",
}

# Academic jargon that's often unnecessarily complex
ACADEMIC_JARGON = {
    "paradigm",
    "methodology",
    "infrastructure",
    "implementation",
    "functionality",
    "capabilities",
    "parameters",
    "criteria",
    "prerequisites",
    "subsequently",
    "consequently",
    "furthermore",
    "moreover",
    "nevertheless",
    "notwithstanding",
    "aforementioned",
    "herein",
    "therein",
    "whereby",
    "wherein",
    "whereof",
    "whereto",
    "whereupon",
    "wherewith",
    "wherewithal",
    "whereabouts",
    "wherefore",
    "whereas",
    "leverage",
    "synergy",
    "optimize",
    "facilitate",
    "utilize",
    "expedite",
    "elucidate",
    "comprehend",
    "perceive",
    "demonstrate",
    "indicate",
    "substantiate",
    "constitute",
    "encompass",
    "ascertain",
    "endeavor",
    "procure",
    "commence",
    "terminate",
    "implement",
}

# Business buzzwords that add complexity without value
BUZZWORDS = {
    "synergy",
    "leverage",
    "paradigm",
    "methodology",
    "infrastructure",
    "implementation",
    "functionality",
    "capabilities",
    "parameters",
    "criteria",
    "prerequisites",
    "optimize",
    "facilitate",
    "utilize",
    "expedite",
    "elucidate",
    "comprehend",
    "perceive",
    "demonstrate",
    "indicate",
    "substantiate",
    "constitute",
    "encompass",
    "ascertain",
    "endeavor",
    "procure",
    "commence",
    "terminate",
    "implement",
    "streamline",
    "revolutionize",
    "innovate",
    "disrupt",
    "transform",
    "empower",
    "enable",
    "enhance",
    "maximize",
    "minimize",
    "collaboration",
    "partnership",
    "engagement",
    "alignment",
    "integration",
}


def calculate_flesch_kincaid_grade_level(text: str) -> float:
    """
    Calculate Flesch-Kincaid Grade Level for readability.

    Args:
        text: Text to analyze

    Returns:
        Grade level score
    """
    if not text.strip():
        return 0.0

    # Count sentences
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    if sentence_count == 0:
        return 0.0

    # Count words
    words = re.findall(r"\b\w+\b", text.lower())
    word_count = len(words)

    if word_count == 0:
        return 0.0

    # Count syllables (approximate)
    syllable_count = 0
    for word in words:
        # Simple syllable counting
        vowels = "aeiouy"
        syllable_count += sum(1 for char in word if char in vowels)

        # Adjust for silent 'e'
        if word.endswith("e"):
            syllable_count -= 1

        # Every word has at least one syllable
        syllable_count = max(1, syllable_count)

    # Flesch-Kincaid Grade Level formula
    grade_level = (
        0.39 * (word_count / sentence_count)
        + 11.8 * (syllable_count / word_count)
        - 15.59
    )

    return max(0.0, grade_level)


def detect_complex_words(text: str) -> list[dict[str, Any]]:
    """
    Detect unnecessarily complex words and phrases.

    Args:
        text: Text to analyze

    Returns:
        List of complex word spans
    """
    complex_spans = []
    words = re.findall(r"\b\w+\b", text)

    for word in words:
        word_lower = word.lower()

        # Check for complex alternatives
        if word_lower in COMPLEXITY_MAPPINGS:
            complex_spans.append(
                {
                    "word": word,
                    "simple_alternative": COMPLEXITY_MAPPINGS[word_lower],
                    "type": "complex_word",
                    "category": "formal_alternative",
                }
            )

        # Check for academic jargon
        elif word_lower in ACADEMIC_JARGON:
            complex_spans.append(
                {
                    "word": word,
                    "simple_alternative": "simpler word",
                    "type": "complex_word",
                    "category": "academic_jargon",
                }
            )

        # Check for business buzzwords
        elif word_lower in BUZZWORDS:
            complex_spans.append(
                {
                    "word": word,
                    "simple_alternative": "clearer word",
                    "type": "complex_word",
                    "category": "buzzword",
                }
            )

        # Check for very long words (>10 characters)
        elif len(word) > 10:
            complex_spans.append(
                {
                    "word": word,
                    "simple_alternative": "shorter word",
                    "type": "complex_word",
                    "category": "long_word",
                }
            )

    return complex_spans


def detect_complex_phrases(text: str) -> list[dict[str, Any]]:
    """
    Detect unnecessarily complex phrases and constructions.

    Args:
        text: Text to analyze

    Returns:
        List of complex phrase spans
    """
    complex_phrases = []

    # Complex phrase patterns
    complex_patterns = [
        # Wordy constructions
        (r"\b(?:in order to|so as to)\b", "wordy_purpose"),
        (r"\b(?:due to the fact that|because of the fact that)\b", "wordy_cause"),
        (r"\b(?:in the event that|in case that)\b", "wordy_condition"),
        (r"\b(?:at this point in time|at the present time)\b", "wordy_time"),
        (r"\b(?:in the near future|in the not too distant future)\b", "wordy_future"),
        # Redundant phrases
        (r"\b(?:each and every|first and foremost|various different)\b", "redundant"),
        (
            r"\b(?:completely finished|totally complete|entirely complete)\b",
            "redundant",
        ),
        (r"\b(?:free gift|past history|future plans|new innovation)\b", "redundant"),
        # Overly formal constructions
        (
            r"\b(?:it is important to note that|it should be noted that)\b",
            "formal_construction",
        ),
        (
            r"\b(?:it is worth noting that|it is interesting to note that)\b",
            "formal_construction",
        ),
        (r"\b(?:in the context of|within the framework of)\b", "formal_construction"),
        # Passive voice constructions
        (
            r"\b(?:it was decided that|it was determined that|it was found that)\b",
            "passive_construction",
        ),
        (
            r"\b(?:it has been shown that|it has been demonstrated that)\b",
            "passive_construction",
        ),
    ]

    for pattern, phrase_type in complex_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            complex_phrases.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "type": "complex_phrase",
                    "category": phrase_type,
                }
            )

    return complex_phrases


def calculate_complexity_score(
    text: str,
    complex_words: list[dict],
    complex_phrases: list[dict],
    grade_level: float,
) -> float:
    """
    Calculate overall complexity score.

    Args:
        text: Full text
        complex_words: List of complex words
        complex_phrases: List of complex phrases
        grade_level: Flesch-Kincaid grade level

    Returns:
        Complexity score (0-1, higher = more complex/unnecessarily complex)
    """
    if not text.strip():
        return 0.5

    # Base score
    score = 0.3  # Start with moderate complexity

    # Add complexity based on grade level
    if grade_level > 12:  # College level
        score += 0.3
    elif grade_level > 10:  # High school level
        score += 0.2
    elif grade_level > 8:  # Middle school level
        score += 0.1

    # Add complexity based on complex words
    word_complexity = len(complex_words) * 0.05
    score += min(word_complexity, 0.3)

    # Add complexity based on complex phrases
    phrase_complexity = len(complex_phrases) * 0.08
    score += min(phrase_complexity, 0.3)

    # Check for excessive use of complex words
    words = re.findall(r"\b\w+\b", text.lower())
    if words:
        complex_word_ratio = len(complex_words) / len(words)
        if complex_word_ratio > 0.1:  # More than 10% complex words
            score += 0.2
        elif complex_word_ratio > 0.05:  # More than 5% complex words
            score += 0.1

    # Normalize score
    return max(0.0, min(1.0, score))


def extract_features(
    text: str, sentences: list[str], tokens: list[str]
) -> dict[str, Any]:
    """
    Extract all complexity-related features.

    Args:
        text: Full text to analyze
        sentences: List of sentences
        tokens: List of tokens

    Returns:
        Dictionary with complexity features
    """
    try:
        # Calculate readability metrics
        grade_level = calculate_flesch_kincaid_grade_level(text)

        # Detect complex elements
        complex_words = detect_complex_words(text)
        complex_phrases = detect_complex_phrases(text)

        # Calculate overall complexity score
        complexity_score = calculate_complexity_score(
            text, complex_words, complex_phrases, grade_level
        )

        # Calculate additional metrics
        word_count = len(re.findall(r"\b\w+\b", text))
        complex_word_ratio = len(complex_words) / max(word_count, 1)
        complex_phrase_ratio = len(complex_phrases) / max(len(sentences), 1)

        return {
            "flesch_kincaid_grade": grade_level,
            "complex_words_count": len(complex_words),
            "complex_phrases_count": len(complex_phrases),
            "complex_word_ratio": complex_word_ratio,
            "complex_phrase_ratio": complex_phrase_ratio,
            "complexity_score": complexity_score,
            "complex_words": complex_words,
            "complex_phrases": complex_phrases,
            "value": complexity_score,  # For compatibility with combine.py
        }

    except Exception as e:
        logger.error(f"Error in complexity feature extraction: {e}")
        return {
            "flesch_kincaid_grade": 0.0,
            "complex_words_count": 0,
            "complex_phrases_count": 0,
            "complex_word_ratio": 0.0,
            "complex_phrase_ratio": 0.0,
            "complexity_score": 0.5,
            "complex_words": [],
            "complex_phrases": [],
            "value": 0.5,
        }
