#!/usr/bin/env python3
"""
Improve tone detection for better AI slop identification.
"""

import re
from typing import Any


def enhanced_hedging_detection(text: str) -> dict[str, Any]:
    """Enhanced hedging detection with more patterns."""

    # Expanded hedging patterns
    hedging_patterns = [
        # Uncertainty markers
        r"\b(perhaps|maybe|might|could|may|possibly|potentially|likely|probably)\b",
        r"\b(somewhat|rather|tends to|appears to|seems to|allegedly|arguably)\b",
        r"\b(supposedly|reportedly|understandably|reasonably|comparatively)\b",
        r"\b(relatively|fairly|quite|pretty|almost|nearly|approximately|roughly)\b",
        # Corporate hedging
        r"\b(it is important to note|it should be noted|it is worth noting)\b",
        r"\b(it appears that|it seems that|it would seem that)\b",
        r"\b(one might argue|it could be argued|it is arguable)\b",
        r"\b(to some extent|in some cases|under certain circumstances)\b",
        # AI-specific hedging
        r"\b(based on my analysis|according to my understanding)\b",
        r"\b(I would suggest|I would recommend|I believe)\b",
        r"\b(it is my understanding|from my perspective)\b",
    ]

    hedging_count = 0
    hedging_spans = []

    for pattern in hedging_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            hedging_count += 1
            hedging_spans.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "type": "hedging",
                }
            )

    return {
        "hedging_count": hedging_count,
        "hedging_spans": hedging_spans,
        "hedging_density": hedging_count / max(len(text.split()), 1) * 100,
    }


def enhanced_sycophancy_detection(text: str) -> dict[str, Any]:
    """Enhanced sycophancy detection."""

    sycophancy_patterns = [
        # Excessive agreement
        r"\b(absolutely|definitely|certainly|obviously|clearly|undoubtedly)\b",
        r"\b(unquestionably|beyond doubt|of course|naturally|exactly|precisely)\b",
        # Overly positive language
        r"\b(perfect|excellent|wonderful|amazing|fantastic|brilliant|outstanding)\b",
        r"\b(superb|impressive|remarkable|exceptional|incredible|phenomenal)\b",
        # Corporate sycophancy
        r"\b(revolutionary|groundbreaking|cutting-edge|state-of-the-art)\b",
        r"\b(industry-leading|best-in-class|world-class|premium|elite)\b",
        # AI sycophancy patterns
        r"\b(I completely agree|you are absolutely right|that is exactly correct)\b",
        r"\b(you have made an excellent point|that is a brilliant observation)\b",
    ]

    sycophancy_count = 0
    sycophancy_spans = []

    for pattern in sycophancy_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            sycophancy_count += 1
            sycophancy_spans.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "type": "sycophancy",
                }
            )

    return {
        "sycophancy_count": sycophancy_count,
        "sycophancy_spans": sycophancy_spans,
        "sycophancy_density": sycophancy_count / max(len(text.split()), 1) * 100,
    }


def enhanced_formality_detection(text: str) -> dict[str, Any]:
    """Enhanced formality detection."""

    # Formal language patterns
    formal_patterns = [
        r"\b(utilize|facilitate|implement|establish|commence|terminate)\b",
        r"\b(commence|initiate|conclude|finalize|optimize|maximize)\b",
        r"\b(endeavor|undertaking|initiative|comprehensive|systematic)\b",
        r"\b(methodology|framework|paradigm|infrastructure|mechanism)\b",
    ]

    # Passive voice detection (simplified)
    passive_patterns = [
        r"\b(is|are|was|were|be|been|being)\s+\w+ed\b",
        r"\b(has|have|had)\s+been\s+\w+ed\b",
    ]

    formal_count = 0
    passive_count = 0

    for pattern in formal_patterns:
        formal_count += len(re.findall(pattern, text, re.IGNORECASE))

    for pattern in passive_patterns:
        passive_count += len(re.findall(pattern, text, re.IGNORECASE))

    return {
        "formal_count": formal_count,
        "passive_count": passive_count,
        "formality_density": formal_count / max(len(text.split()), 1) * 100,
        "passive_density": passive_count / max(len(text.split()), 1) * 100,
    }


def calculate_enhanced_tone_score(text: str) -> dict[str, Any]:
    """Calculate enhanced tone score."""

    hedging_data = enhanced_hedging_detection(text)
    sycophancy_data = enhanced_sycophancy_detection(text)
    formality_data = enhanced_formality_detection(text)

    # Calculate tone score (higher = more sloppy)
    hedging_score = min(
        hedging_data["hedging_density"] / 5.0, 1.0
    )  # Normalize to 5% threshold
    sycophancy_score = min(
        sycophancy_data["sycophancy_density"] / 3.0, 1.0
    )  # Normalize to 3% threshold
    formality_score = min(
        formality_data["formality_density"] / 8.0, 1.0
    )  # Normalize to 8% threshold

    # Combined tone score
    tone_score = hedging_score * 0.4 + sycophancy_score * 0.4 + formality_score * 0.2

    return {
        "hedging_data": hedging_data,
        "sycophancy_data": sycophancy_data,
        "formality_data": formality_data,
        "hedging_score": hedging_score,
        "sycophancy_score": sycophancy_score,
        "formality_score": formality_score,
        "tone_score": tone_score,
    }


def test_enhanced_tone_detection():
    """Test the enhanced tone detection."""

    test_cases = [
        (
            "Corporate speak",
            "Our team of experts is dedicated to providing you with the highest quality service and support. We believe in the power of collaboration and are committed to helping you succeed in all your endeavors.",
        ),
        (
            "Hedging heavy",
            "Perhaps it might be somewhat beneficial to consider the possibility that this approach could potentially yield somewhat positive results, though it is arguable whether this would be the optimal solution.",
        ),
        (
            "Sycophantic",
            "This is absolutely brilliant! You are completely right and this is definitely the best approach. This is exactly what we need and it is unquestionably the perfect solution.",
        ),
        (
            "Overly formal",
            "We shall utilize our comprehensive methodology to facilitate the implementation of this revolutionary framework. This systematic approach will optimize our endeavors and maximize our outcomes.",
        ),
        (
            "Clean text",
            "I went to the store yesterday and bought some milk. The weather was nice, so I walked home instead of taking the bus.",
        ),
    ]

    print("🧪 Testing Enhanced Tone Detection")
    print("=" * 50)

    for name, text in test_cases:
        print(f"\n{name}:")
        print(f"Text: {text[:60]}...")

        result = calculate_enhanced_tone_score(text)

        print(
            f"  Hedging Score: {result['hedging_score']:.3f} ({result['hedging_data']['hedging_count']} instances)"
        )
        print(
            f"  Sycophancy Score: {result['sycophancy_score']:.3f} ({result['sycophancy_data']['sycophancy_count']} instances)"
        )
        print(
            f"  Formality Score: {result['formality_score']:.3f} ({result['formality_data']['formal_count']} instances)"
        )
        print(f"  Overall Tone Score: {result['tone_score']:.3f}")


if __name__ == "__main__":
    test_enhanced_tone_detection()
