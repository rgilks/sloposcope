#!/usr/bin/env python3
"""
Improve template detection for better AI slop identification.
"""

import re
from typing import Any


def enhanced_boilerplate_detection(text: str) -> dict[str, Any]:
    """Enhanced boilerplate phrase detection."""

    # Comprehensive boilerplate patterns
    boilerplate_patterns = [
        # Corporate boilerplate
        r"\b(our team of experts|dedicated to providing|highest quality service)\b",
        r"\b(we believe in|power of collaboration|committed to helping)\b",
        r"\b(comprehensive solution|strategic approach|core competencies)\b",
        r"\b(leverage our|maximize efficiency|optimize performance)\b",
        # AI response boilerplate
        r"\b(I would be happy to assist|based on my analysis|I recommend)\b",
        r"\b(I understand your concern|let me provide|comprehensive solution)\b",
        r"\b(this approach has been proven|effective in similar situations)\b",
        # Generic business boilerplate
        r"\b(in today\'s fast-paced|competitive landscape|ever-evolving)\b",
        r"\b(cutting-edge technology|innovative solutions|transformative impact)\b",
        r"\b(seamless integration|end-to-end|holistic approach)\b",
        # Academic/formal boilerplate
        r"\b(it is important to note|it should be noted|it is worth noting)\b",
        r"\b(this analysis reveals|important patterns|warrant further investigation)\b",
        r"\b(the implications of these findings|extend beyond|immediate scope)\b",
        # List/structured content patterns
        r"\b(here are \d+ ways|the following steps|key points to consider)\b",
        r"\b(first and foremost|last but not least|in conclusion)\b",
        r"\b(to summarize|in summary|in other words)\b",
    ]

    boilerplate_count = 0
    boilerplate_spans = []

    for pattern in boilerplate_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            boilerplate_count += 1
            boilerplate_spans.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "type": "boilerplate",
                }
            )

    return {
        "boilerplate_count": boilerplate_count,
        "boilerplate_spans": boilerplate_spans,
        "boilerplate_density": boilerplate_count / max(len(text.split()), 1) * 100,
    }


def detect_repetitive_phrases(text: str) -> dict[str, Any]:
    """Detect repetitive phrase patterns."""

    # Split into sentences
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Look for repeated sentence starts
    sentence_starts = []
    for sentence in sentences:
        words = sentence.split()[:3]  # First 3 words
        if len(words) >= 2:
            sentence_starts.append(" ".join(words).lower())

    # Count repetitions
    start_counts = {}
    for start in sentence_starts:
        start_counts[start] = start_counts.get(start, 0) + 1

    # Find repeated starts
    repeated_starts = {
        start: count for start, count in start_counts.items() if count > 1
    }

    return {
        "repeated_starts": repeated_starts,
        "repetition_score": len(repeated_starts) / max(len(sentences), 1),
    }


def detect_list_patterns(text: str) -> dict[str, Any]:
    """Detect list-like patterns that indicate templated content."""

    # Numbered lists
    numbered_pattern = r"\b\d+\.\s+\w+"
    numbered_matches = len(re.findall(numbered_pattern, text))

    # Bullet points
    bullet_pattern = r"[•·▪▫]\s+\w+"
    bullet_matches = len(re.findall(bullet_pattern, text))

    # Dash lists
    dash_pattern = r"-\s+\w+"
    dash_matches = len(re.findall(dash_pattern, text))

    # "Here are X ways/things/steps" patterns
    ways_pattern = r"\bhere are \d+ (ways|things|steps|points|reasons|benefits)\b"
    ways_matches = len(re.findall(ways_pattern, text, re.IGNORECASE))

    total_list_items = numbered_matches + bullet_matches + dash_matches

    return {
        "numbered_items": numbered_matches,
        "bullet_items": bullet_matches,
        "dash_items": dash_matches,
        "ways_patterns": ways_matches,
        "total_list_items": total_list_items,
        "list_density": total_list_items / max(len(text.split()), 1) * 100,
    }


def calculate_enhanced_template_score(text: str) -> dict[str, Any]:
    """Calculate enhanced template score."""

    boilerplate_data = enhanced_boilerplate_detection(text)
    repetition_data = detect_repetitive_phrases(text)
    list_data = detect_list_patterns(text)

    # Calculate template score components
    boilerplate_score = min(
        boilerplate_data["boilerplate_density"] / 2.0, 1.0
    )  # Normalize to 2% threshold
    repetition_score = min(
        repetition_data["repetition_score"] * 2.0, 1.0
    )  # Scale repetition
    list_score = min(list_data["list_density"] / 5.0, 1.0)  # Normalize to 5% threshold

    # Combined template score
    template_score = boilerplate_score * 0.5 + repetition_score * 0.3 + list_score * 0.2

    return {
        "boilerplate_data": boilerplate_data,
        "repetition_data": repetition_data,
        "list_data": list_data,
        "boilerplate_score": boilerplate_score,
        "repetition_score": repetition_score,
        "list_score": list_score,
        "template_score": template_score,
    }


def test_enhanced_template_detection():
    """Test the enhanced template detection."""

    test_cases = [
        (
            "Corporate boilerplate",
            "Our team of experts is dedicated to providing you with the highest quality service and support. We believe in the power of collaboration and are committed to helping you succeed in all your endeavors.",
        ),
        (
            "AI response boilerplate",
            "I would be happy to assist you with this request. Based on my analysis, I recommend implementing a strategic approach that leverages our core competencies and maximizes efficiency.",
        ),
        (
            "List-heavy content",
            "Here are 5 ways to improve your productivity: 1. Set clear goals. 2. Prioritize tasks. 3. Eliminate distractions. 4. Take breaks. 5. Review progress.",
        ),
        (
            "Repetitive structure",
            "The solution is effective. The solution works well. The solution provides results. The solution is the best approach.",
        ),
        (
            "Clean text",
            "I went to the store yesterday and bought some milk. The weather was nice, so I walked home instead of taking the bus.",
        ),
    ]

    print("🧪 Testing Enhanced Template Detection")
    print("=" * 50)

    for name, text in test_cases:
        print(f"\n{name}:")
        print(f"Text: {text[:60]}...")

        result = calculate_enhanced_template_score(text)

        print(
            f"  Boilerplate Score: {result['boilerplate_score']:.3f} ({result['boilerplate_data']['boilerplate_count']} instances)"
        )
        print(
            f"  Repetition Score: {result['repetition_score']:.3f} ({len(result['repetition_data']['repeated_starts'])} repeated starts)"
        )
        print(
            f"  List Score: {result['list_score']:.3f} ({result['list_data']['total_list_items']} list items)"
        )
        print(f"  Overall Template Score: {result['template_score']:.3f}")


if __name__ == "__main__":
    test_enhanced_template_detection()
