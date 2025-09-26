"""
Subjectivity/Bias feature extraction for AI slop analysis.

Measures the presence of subjective language and bias.
Based on the paper's taxonomy where Bias (IQ2) assesses subjectivity/bias presence.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Subjective language patterns
SUBJECTIVE_PATTERNS = {
    # Opinion markers
    "opinion": [
        r"\b(?:i think|i believe|i feel|i assume|i guess|i suppose|i reckon)\b",
        r"\b(?:in my opinion|in my view|from my perspective|as i see it)\b",
        r"\b(?:personally|frankly|honestly|truthfully|admittedly)\b",
    ],
    # Emotional language
    "emotional": [
        r"\b(?:amazing|incredible|fantastic|terrible|awful|horrible|disgusting)\b",
        r"\b(?:love|hate|adore|despise|fear|worry|concern|excited|thrilled)\b",
        r"\b(?:shocking|surprising|disappointing|frustrating|annoying)\b",
    ],
    # Evaluative language
    "evaluative": [
        r"\b(?:good|bad|great|excellent|poor|terrible|wonderful|awful)\b",
        r"\b(?:best|worst|better|worse|superior|inferior|outstanding|mediocre)\b",
        r"\b(?:right|wrong|correct|incorrect|proper|improper|appropriate|inappropriate)\b",
    ],
    # Certainty markers
    "certainty": [
        r"\b(?:definitely|certainly|absolutely|surely|undoubtedly|clearly)\b",
        r"\b(?:obviously|evidently|apparently|seemingly|presumably)\b",
        r"\b(?:without doubt|no question|beyond question|for certain)\b",
    ],
    # Uncertainty markers
    "uncertainty": [
        r"\b(?:maybe|perhaps|possibly|probably|likely|unlikely|might|could)\b",
        r"\b(?:seems|appears|looks like|suggests|indicates|implies)\b",
        r"\b(?:i\'m not sure|i don\'t know|it\'s unclear|hard to say)\b",
    ],
}

# Bias indicators
BIAS_PATTERNS = {
    # Generalization
    "generalization": [
        r"\b(?:all|every|everyone|everybody|everything|always|never|none|no one)\b",
        r"\b(?:most|many|some|few|several|various|different)\b",
        r"\b(?:typically|usually|generally|commonly|frequently|often)\b",
    ],
    # Stereotyping
    "stereotyping": [
        r"\b(?:people like|those who|individuals who|persons who)\b",
        r"\b(?:the typical|the average|the usual|the common)\b",
        r"\b(?:as expected|as usual|as always|as typical)\b",
    ],
    # Loaded language
    "loaded": [
        r"\b(?:so-called|alleged|supposed|claimed|purported)\b",
        r"\b(?:merely|simply|just|only|barely|hardly)\b",
        r"\b(?:even|still|yet|already|finally|at last)\b",
    ],
    # Emotional bias
    "emotional_bias": [
        r"\b(?:unfortunately|sadly|regrettably|disappointingly)\b",
        r"\b(?:fortunately|thankfully|luckily|happily)\b",
        r"\b(?:shockingly|surprisingly|amazingly|incredibly)\b",
    ],
}

# Neutral language patterns (reduce subjectivity)
NEUTRAL_PATTERNS = [
    r"\b(?:according to|based on|research shows|studies indicate)\b",
    r"\b(?:data suggests|evidence shows|findings indicate)\b",
    r"\b(?:it is reported|it has been found|it was observed)\b",
]


def detect_subjective_language(text: str) -> list[dict[str, Any]]:
    """
    Detect subjective language patterns.

    Args:
        text: Text to analyze

    Returns:
        List of subjective language spans
    """
    subjective_spans = []

    for category, patterns in SUBJECTIVE_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                subjective_spans.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "text": match.group(),
                        "type": "subjective_language",
                        "category": category,
                    }
                )

    return subjective_spans


def detect_bias_language(text: str) -> list[dict[str, Any]]:
    """
    Detect bias language patterns.

    Args:
        text: Text to analyze

    Returns:
        List of bias language spans
    """
    bias_spans = []

    for category, patterns in BIAS_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                bias_spans.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "text": match.group(),
                        "type": "bias_language",
                        "category": category,
                    }
                )

    return bias_spans


def detect_neutral_language(text: str) -> list[dict[str, Any]]:
    """
    Detect neutral, objective language patterns.

    Args:
        text: Text to analyze

    Returns:
        List of neutral language spans
    """
    neutral_spans = []

    for pattern in NEUTRAL_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            neutral_spans.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "type": "neutral_language",
                }
            )

    return neutral_spans


def calculate_subjectivity_score(
    text: str,
    subjective_spans: list[dict],
    bias_spans: list[dict],
    neutral_spans: list[dict],
) -> float:
    """
    Calculate overall subjectivity score.

    Args:
        text: Full text
        subjective_spans: List of subjective language spans
        bias_spans: List of bias language spans
        neutral_spans: List of neutral language spans

    Returns:
        Subjectivity score (0-1, higher = more subjective/biased)
    """
    if not text.strip():
        return 0.5

    # Base score
    score = 0.3  # Start with moderate subjectivity

    # Add subjectivity based on subjective language
    subjective_penalty = len(subjective_spans) * 0.05
    score += min(subjective_penalty, 0.3)

    # Add bias based on bias language
    bias_penalty = len(bias_spans) * 0.08
    score += min(bias_penalty, 0.4)

    # Reduce subjectivity based on neutral language
    neutral_bonus = len(neutral_spans) * 0.03
    score -= min(neutral_bonus, 0.2)

    # Check for excessive use of subjective language
    words = re.findall(r"\b\w+\b", text.lower())
    if words:
        subjective_ratio = len(subjective_spans) / len(words)
        if subjective_ratio > 0.05:  # More than 5% subjective language
            score += 0.2
        elif subjective_ratio > 0.02:  # More than 2% subjective language
            score += 0.1

    # Check for emotional language density
    emotional_words = [
        span for span in subjective_spans if span["category"] == "emotional"
    ]
    if words:
        emotional_ratio = len(emotional_words) / len(words)
        if emotional_ratio > 0.03:  # More than 3% emotional language
            score += 0.15

    # Normalize score
    return max(0.0, min(1.0, score))


def extract_features(
    text: str, sentences: list[str], tokens: list[str]
) -> dict[str, Any]:
    """
    Extract all subjectivity/bias-related features.

    Args:
        text: Full text to analyze
        sentences: List of sentences
        tokens: List of tokens

    Returns:
        Dictionary with subjectivity features
    """
    try:
        # Detect subjective and bias language
        subjective_spans = detect_subjective_language(text)
        bias_spans = detect_bias_language(text)
        neutral_spans = detect_neutral_language(text)

        # Calculate overall subjectivity score
        subjectivity_score = calculate_subjectivity_score(
            text, subjective_spans, bias_spans, neutral_spans
        )

        # Calculate additional metrics
        word_count = len(re.findall(r"\b\w+\b", text))
        subjective_ratio = len(subjective_spans) / max(word_count, 1)
        bias_ratio = len(bias_spans) / max(word_count, 1)
        neutral_ratio = len(neutral_spans) / max(word_count, 1)

        # Categorize subjective language
        subjective_by_category = {}
        for span in subjective_spans:
            category = span["category"]
            if category not in subjective_by_category:
                subjective_by_category[category] = 0
            subjective_by_category[category] += 1

        # Categorize bias language
        bias_by_category = {}
        for span in bias_spans:
            category = span["category"]
            if category not in bias_by_category:
                bias_by_category[category] = 0
            bias_by_category[category] += 1

        return {
            "subjective_count": len(subjective_spans),
            "bias_count": len(bias_spans),
            "neutral_count": len(neutral_spans),
            "subjective_ratio": subjective_ratio,
            "bias_ratio": bias_ratio,
            "neutral_ratio": neutral_ratio,
            "subjectivity_score": subjectivity_score,
            "subjective_by_category": subjective_by_category,
            "bias_by_category": bias_by_category,
            "subjective_spans": subjective_spans,
            "bias_spans": bias_spans,
            "neutral_spans": neutral_spans,
            "value": subjectivity_score,  # For compatibility with combine.py
        }

    except Exception as e:
        logger.error(f"Error in subjectivity feature extraction: {e}")
        return {
            "subjective_count": 0,
            "bias_count": 0,
            "neutral_count": 0,
            "subjective_ratio": 0.0,
            "bias_ratio": 0.0,
            "neutral_ratio": 0.0,
            "subjectivity_score": 0.5,
            "subjective_by_category": {},
            "bias_by_category": {},
            "subjective_spans": [],
            "bias_spans": [],
            "neutral_spans": [],
            "value": 0.5,
        }
