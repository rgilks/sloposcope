"""
Tone analysis for AI slop detection.

Detects hedging, sycophancy, formality, and other tone-related issues.
"""

import re
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ToneAnalyzer:
    """Analyzes tone issues in text."""

    def __init__(self):
        """Initialize with tone pattern lists."""
        # Hedging patterns (words that indicate uncertainty)
        self.hedging_patterns = [
            r"perhaps", r"maybe", r"might", r"could", r"may", r"possibly",
            r"potentially", r"likely", r"probably", r"somewhat", r"rather",
            r"tends to", r"appears to", r"seems to", r"allegedly", r"arguably",
            r"supposedly", r"reportedly", r"understandably", r"reasonably",
            r"comparatively", r"relatively", r"fairly", r"quite", r"pretty",
            r"almost", r"nearly", r"approximately", r"roughly", r"about",
        ]

        # Sycophancy patterns (overly agreeable or flattering language)
        self.sycophancy_patterns = [
            r"absolutely", r"definitely", r"certainly", r"obviously",
            r"clearly", r"undoubtedly", r"unquestionably", r"beyond doubt",
            r"of course", r"naturally", r"exactly", r"precisely", r"perfect",
            r"excellent", r"wonderful", r"amazing", r"fantastic", r"brilliant",
            r"outstanding", r"superb", r"impressive", r"remarkable", r"exceptional",
        ]

        # Formality markers (excessively formal language)
        self.formality_patterns = [
            r"moreover", r"furthermore", r"nevertheless", r"notwithstanding",
            r"albeit", r"henceforth", r"herein", r"therein", r"hereby",
            r"thereby", r"whereas", r"whilst", r"utilize", r"facilitate",
            r"implement", r"comprehensive", r"extensive", r"significant",
            r"substantial", r"considerable", r"numerous", r"various",
        ]

        # Compile all patterns
        self.compiled_hedging = [re.compile(p, re.IGNORECASE) for p in self.hedging_patterns]
        self.compiled_sycophancy = [re.compile(p, re.IGNORECASE) for p in self.sycophancy_patterns]
        self.compiled_formality = [re.compile(p, re.IGNORECASE) for p in self.formality_patterns]

    def count_hedging(self, text: str) -> int:
        """Count hedging expressions in text."""
        count = 0
        for pattern in self.compiled_hedging:
            count += len(pattern.findall(text))
        return count

    def count_sycophancy(self, text: str) -> int:
        """Count sycophantic expressions in text."""
        count = 0
        for pattern in self.compiled_sycophancy:
            count += len(pattern.findall(text))
        return count

    def count_formality(self, text: str) -> int:
        """Count formality markers in text."""
        count = 0
        for pattern in self.compiled_formality:
            count += len(pattern.findall(text))
        return count

    def detect_tone_spans(self, text: str) -> List[Dict[str, Any]]:
        """Detect spans containing tone issues."""
        spans = []

        # Find hedging spans
        for pattern in self.compiled_hedging:
            for match in pattern.finditer(text):
                spans.append({
                    "start": match.start(),
                    "end": match.end(),
                    "type": "hedging",
                    "note": f"Hedging expression: '{match.group()}'",
                })

        # Find sycophancy spans
        for pattern in self.compiled_sycophancy:
            for match in pattern.finditer(text):
                spans.append({
                    "start": match.start(),
                    "end": match.end(),
                    "type": "sycophancy",
                    "note": f"Sycophantic expression: '{match.group()}'",
                })

        return spans


def calculate_passive_voice_ratio(sentences: List[str]) -> float:
    """Calculate ratio of passive voice constructions."""
    # This is a simplified implementation
    # In practice would use dependency parsing

    passive_indicators = ['was', 'were', 'been', 'being', 'is', 'are', 'am']

    passive_sentences = 0
    total_sentences = len(sentences)

    if total_sentences == 0:
        return 0.0

    for sentence in sentences:
        words = sentence.lower().split()
        if any(indicator in words for indicator in passive_indicators):
            # Look for past participle after passive indicator
            for i, word in enumerate(words):
                if word in passive_indicators:
                    # Check if followed by past participle (ends with ed)
                    if i + 1 < len(words) and words[i + 1].endswith('ed'):
                        passive_sentences += 1
                        break

    return passive_sentences / total_sentences


def extract_features(text: str, sentences: List[str], tokens: List[str]) -> Dict[str, Any]:
    """Extract all tone-related features."""
    try:
        analyzer = ToneAnalyzer()

        # Count different tone markers
        hedging_count = analyzer.count_hedging(text)
        sycophancy_count = analyzer.count_sycophancy(text)
        formality_count = analyzer.count_formality(text)

        # Calculate ratios per 100 words
        total_words = len(tokens)
        if total_words == 0:
            hedging_ratio = 0.0
            sycophancy_ratio = 0.0
            formality_ratio = 0.0
        else:
            hedging_ratio = (hedging_count / total_words) * 100
            sycophancy_ratio = (sycophancy_count / total_words) * 100
            formality_ratio = (formality_count / total_words) * 100

        # Calculate passive voice ratio
        passive_ratio = calculate_passive_voice_ratio(sentences)

        # Detect tone spans
        tone_spans = analyzer.detect_tone_spans(text)

        # Calculate overall tone score
        # Higher hedging, sycophancy, and formality = more problematic tone
        tone_score = (
            min(1.0, hedging_ratio / 10.0) * 0.4 +      # Hedging (normalized)
            min(1.0, sycophancy_ratio / 5.0) * 0.4 +    # Sycophancy (normalized)
            min(1.0, formality_ratio / 8.0) * 0.2       # Formality (normalized)
        )

        return {
            "hedging_count": hedging_count,
            "sycophancy_count": sycophancy_count,
            "formality_count": formality_count,
            "hedging_ratio": hedging_ratio,
            "sycophancy_ratio": sycophancy_ratio,
            "formality_ratio": formality_ratio,
            "passive_ratio": passive_ratio,
            "tone_score": tone_score,
            "tone_spans": tone_spans,
        }

    except Exception as e:
        logger.error(f"Error in tone feature extraction: {e}")
        return {
            "hedging_count": 0,
            "sycophancy_count": 0,
            "formality_count": 0,
            "hedging_ratio": 0.0,
            "sycophancy_ratio": 0.0,
            "formality_ratio": 0.0,
            "passive_ratio": 0.0,
            "tone_score": 0.0,
            "tone_spans": [],
        }
