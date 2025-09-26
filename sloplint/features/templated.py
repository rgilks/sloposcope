"""
Templated content detection for AI slop analysis.

Detects boilerplate phrases, repetitive patterns, and templated writing.
"""

import re
import math
from collections import Counter
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class TemplateDetector:
    """Detects templated and boilerplate content."""

    def __init__(self):
        """Initialize with known boilerplate patterns."""
        # Common AI-generated boilerplate phrases
        self.boilerplate_patterns = [
            # Generic introductions
            r"In conclusion",
            r"As an AI",
            r"I am an AI",
            r"Here are [0-9]+ ways",
            r"It is important to note",
            r"As previously mentioned",
            r"This is because",
            r"In other words",
            r"It should be noted that",
            r"As we can see",
            r"This suggests that",
            r"Based on the above",
            r"To summarize",
            r"All in all",
            r"In summary",
            r"As a result",
            r"Therefore",
            r"Consequently",
            r"Moreover",
            r"Furthermore",
            r"However",
            r"Nevertheless",
            r"Despite this",
            r"On the other hand",
            r"First(ly)?",
            r"Second(ly)?",
            r"Third(ly)?",
            r"Next",
            r"Finally",
            r"Last but not least",
            r"Let me explain",
            r"Allow me to elaborate",
            r"I will now",
            r"I'm going to",
            r"It's worth noting",
            r"Please note that",
            r"Keep in mind",
            r"Don't forget",
            r"Remember that",
        ]

        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.boilerplate_patterns]

    def detect_boilerplate_spans(self, text: str) -> List[Dict[str, Any]]:
        """Detect spans containing boilerplate phrases."""
        spans = []

        for pattern in self.compiled_patterns:
            matches = list(pattern.finditer(text))
            for match in matches:
                spans.append({
                    "start": match.start(),
                    "end": match.end(),
                    "type": "boilerplate",
                    "note": f"Boilerplate phrase: '{match.group()}'",
                })

        return spans

    def calculate_pos_diversity(self, pos_tags: List[str], window_size: int = 5) -> float:
        """Calculate diversity of POS tag 5-grams."""
        if len(pos_tags) < window_size:
            return 0.0

        # Create 5-grams of POS tags
        pos_ngrams = []
        for i in range(len(pos_tags) - window_size + 1):
            ngram = tuple(pos_tags[i:i+window_size])
            pos_ngrams.append(ngram)

        if not pos_ngrams:
            return 0.0

        # Calculate entropy of POS n-gram distribution
        ngram_counts = Counter(pos_ngrams)
        total_ngrams = len(pos_ngrams)

        entropy = 0.0
        for count in ngram_counts.values():
            prob = count / total_ngrams
            entropy -= prob * math.log2(prob)

        # Normalize to [0,1] (higher entropy = more diverse)
        max_entropy = math.log2(len(ngram_counts)) if ngram_counts else 1.0
        diversity_score = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        return diversity_score

    def calculate_pattern_repetition(self, sentences: List[str]) -> float:
        """Calculate repetition of sentence patterns."""
        if not sentences:
            return 0.0

        # Simple pattern matching - look for common sentence starters
        starters = Counter()
        for sentence in sentences:
            words = sentence.strip().split()
            if words:
                starter = words[0].lower()
                starters[starter] += 1

        # Calculate repetition rate
        total_sentences = len(sentences)
        repeated_starters = sum(count for count in starters.values() if count > 1)
        repetition_rate = repeated_starters / total_sentences

        return repetition_rate


def extract_features(text: str, sentences: List[str], tokens: List[str], pos_tags: List[str] = None) -> Dict[str, Any]:
    """Extract all templated content features."""
    try:
        detector = TemplateDetector()

        # Detect boilerplate spans
        boilerplate_spans = detector.detect_boilerplate_spans(text)

        # Calculate POS diversity (if POS tags available)
        pos_diversity = 0.0
        if pos_tags:
            pos_diversity = detector.calculate_pos_diversity(pos_tags, window_size=5)

        # Calculate pattern repetition
        pattern_repetition = detector.calculate_pattern_repetition(sentences)

        # Count boilerplate hits
        boilerplate_hits = len(boilerplate_spans)

        # Calculate overall templated score
        # Lower diversity and higher repetition = more templated
        templated_score = (
            (1.0 - pos_diversity) * 0.5 +    # Low diversity = templated
            pattern_repetition * 0.3 +       # High repetition = templated
            min(1.0, boilerplate_hits / 10.0) * 0.2  # Boilerplate phrases
        )

        return {
            "boilerplate_hits": boilerplate_hits,
            "pos_diversity": pos_diversity,
            "pattern_repetition": pattern_repetition,
            "templated_score": templated_score,
            "templated_spans": boilerplate_spans,
        }

    except Exception as e:
        logger.error(f"Error in templated feature extraction: {e}")
        return {
            "boilerplate_hits": 0,
            "pos_diversity": 0.0,
            "pattern_repetition": 0.0,
            "templated_score": 0.0,
            "templated_spans": [],
        }
