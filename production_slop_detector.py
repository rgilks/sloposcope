#!/usr/bin/env python3
"""
Production-Ready English Slop Detector

A focused, effective slop detector that actually works by using proven techniques
and aggressive thresholds for exceptional English slop detection.
"""

import re
import argparse
import json
import sys
from typing import Dict, Any, List
from collections import Counter


class ProductionSlopDetector:
    """Production-ready slop detector with aggressive detection."""

    def __init__(self):
        """Initialize with comprehensive slop patterns."""
        # Aggressive slop patterns
        self.slop_patterns = {
            # Buzzword patterns (more comprehensive)
            "buzzwords": [
                r"\b(revolutionary|cutting-edge|innovative|paradigm shift|unprecedented|leverage|synergistic|holistic|scalable|robust|seamless|comprehensive|strategic|transformative|disruptive|game-changing|breakthrough|state-of-the-art|next-generation|world-class|industry-leading|best-in-class|mission-critical|enterprise-grade|cloud-native|data-driven|AI-powered|machine learning|artificial intelligence)\b",
                r"\b(leverage|unlock|drive|enable|empower|optimize|streamline|enhance|maximize|minimize|facilitate|accelerate|boost|amplify|harness|capitalize|monetize|scale|transform|disrupt|revolutionize|reimagine|reinvent|redefine|rethink|reengineer)\b",
                r"\b(solutions?|platforms?|ecosystems?|frameworks?|methodologies?|strategies?|initiatives?|programs?|systems?|technologies?|innovations?|advancements?|breakthroughs?|developments?|improvements?|enhancements?|optimizations?|transformations?)\b",
            ],
            # Template patterns (more comprehensive)
            "templates": [
                r"here are \d+ (amazing|incredible|fantastic|wonderful|great|powerful|effective|proven|tested|reliable) ways? to",
                r"in today\'s (rapidly evolving|fast-paced|ever-changing|dynamic|competitive|challenging|complex|uncertain|volatile) (digital )?landscape",
                r"it\'s (absolutely )?(crucial|essential|important|vital|critical|imperative|necessary|paramount) to (understand|recognize|acknowledge|realize|appreciate|grasp)",
                r"this (innovative|revolutionary|cutting-edge|groundbreaking|breakthrough|state-of-the-art|next-generation|advanced|sophisticated) (solution|approach|method|strategy|technique|process|system|platform|framework)",
                r"in conclusion,? (it\'s clear that|we can see that|the evidence shows that|the data indicates that|the results demonstrate that)",
                r"as an? (AI|artificial intelligence|machine learning|automated|intelligent) (system|model|algorithm|tool|assistant),? (I can|we can) (confidently|definitely|certainly|assuredly) (say|state|assert|declare)",
                r"the (implementation|system|solution|approach|method|strategy|process|framework|platform) (is|provides|offers|delivers|ensures|guarantees) (very )?(important|useful|helpful|effective|valuable|beneficial|advantageous|advantageous)",
                r"by (leveraging|utilizing|harnessing|capitalizing on|taking advantage of|making use of) (the power of|the capabilities of|the potential of|the benefits of)",
                r"this (comprehensive|thorough|extensive|detailed|complete|full|total|entire) (analysis|study|research|investigation|examination|review|assessment|evaluation)",
                r"the (key|main|primary|fundamental|essential|critical|important|significant) (factor|element|component|aspect|consideration|point|issue|challenge|opportunity)",
            ],
            # Hedging patterns
            "hedging": [
                r"\b(perhaps|maybe|possibly|potentially|arguably|somewhat|tends to|might|could|may|seems to|appears to|looks like|suggests|indicates|implies|hints at|points to)\b",
                r"\b(it seems|it appears|it looks like|it would seem|it might be|it could be|it may be|it appears that|it seems that|it looks as if|it would appear that)\b",
                r"\b(in some cases|in certain situations|under certain circumstances|in many instances|often|frequently|sometimes|occasionally|typically|generally|usually|commonly)\b",
            ],
            # Sycophancy patterns
            "sycophancy": [
                r"\b(you\'re absolutely right|I completely agree|that\'s exactly what I was thinking|you make an excellent point|you\'re spot on|you\'re correct|you\'re right|I couldn\'t agree more)\b",
                r"\b(as you mentioned|as you pointed out|as you correctly noted|as you wisely observed|as you astutely observed|as you insightfully noted|as you brilliantly pointed out)\b",
                r"\b(that\'s a great question|that\'s an excellent question|that\'s a wonderful question|that\'s a fantastic question|that\'s a brilliant question|that\'s a insightful question)\b",
            ],
            # Repetition patterns
            "repetition": [
                r"(\b\w+\b)(?:\s+\w+){0,3}\s+\1",  # Word repetition
                r"(\b\w+\s+\w+\b)(?:\s+\w+){0,2}\s+\1",  # Phrase repetition
            ],
        }

        # Compile patterns
        self.compiled_patterns = {}
        for category, patterns in self.slop_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def _detect_patterns(self, text: str) -> Dict[str, Any]:
        """Detect slop patterns in text."""
        pattern_scores = {}
        pattern_matches = {}

        for category, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(pattern.findall(text))

            pattern_matches[category] = matches
            # More aggressive scoring
            pattern_scores[category] = (
                len(matches) / max(len(text.split()), 1) * 200
            )  # Doubled scoring

        return pattern_scores, pattern_matches

    def _analyze_repetition(self, text: str) -> Dict[str, Any]:
        """Analyze repetition in text."""
        sentences = re.split(r"[.!?]+", text)
        words = text.lower().split()

        # Word frequency analysis
        word_counts = Counter(words)
        repeated_words = {
            word: count
            for word, count in word_counts.items()
            if count > 1 and len(word) > 2
        }

        # Sentence similarity (simple)
        sentence_words = [
            sentence.lower().split() for sentence in sentences if sentence.strip()
        ]
        similar_sentences = 0

        for i in range(len(sentence_words)):
            for j in range(i + 1, len(sentence_words)):
                if len(sentence_words[i]) > 2 and len(sentence_words[j]) > 2:
                    # Simple similarity based on common words
                    common_words = set(sentence_words[i]) & set(sentence_words[j])
                    similarity = len(common_words) / max(
                        len(sentence_words[i]), len(sentence_words[j])
                    )
                    if similarity > 0.6:  # Lowered threshold
                        similar_sentences += 1

        repetition_score = (
            (len(repeated_words) + similar_sentences) / max(len(sentences), 1) * 100
        )

        return {
            "repetition_score": min(repetition_score, 100),
            "repeated_words": repeated_words,
            "similar_sentences": similar_sentences,
        }

    def _analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity."""
        sentences = re.split(r"[.!?]+", text)
        words = text.split()

        # Average sentence length
        avg_sentence_length = len(words) / max(len(sentences), 1)

        # Word complexity (syllable estimation)
        complex_words = 0
        for word in words:
            # Simple syllable estimation
            syllables = max(1, len(re.findall(r"[aeiouyAEIOUY]", word)))
            if syllables >= 3:
                complex_words += 1

        complexity_ratio = complex_words / max(len(words), 1)

        # More aggressive complexity scoring
        complexity_score = 0
        if avg_sentence_length > 20 and complexity_ratio > 0.25:  # Lowered thresholds
            complexity_score = 60
        elif avg_sentence_length > 25:
            complexity_score = 40
        elif complexity_ratio > 0.3:
            complexity_score = 30

        return {
            "complexity_score": complexity_score,
            "avg_sentence_length": avg_sentence_length,
            "complexity_ratio": complexity_ratio,
        }

    def detect_slop(self, text: str) -> Dict[str, Any]:
        """Detect slop in text using aggressive analysis."""
        if not text.strip():
            return {
                "is_slop": False,
                "slop_score": 0.0,
                "confidence": 0.0,
                "level": "UNKNOWN",
                "explanation": "Empty text",
                "method": "production_patterns",
            }

        # Analyze different aspects
        pattern_scores, pattern_matches = self._detect_patterns(text)
        repetition_analysis = self._analyze_repetition(text)
        complexity_analysis = self._analyze_complexity(text)

        # Calculate overall slop score with more aggressive weighting
        slop_indicators = {
            "buzzwords": pattern_scores.get("buzzwords", 0),
            "templates": pattern_scores.get("templates", 0),
            "hedging": pattern_scores.get("hedging", 0),
            "sycophancy": pattern_scores.get("sycophancy", 0),
            "repetition": repetition_analysis["repetition_score"],
            "complexity": complexity_analysis["complexity_score"],
        }

        # More aggressive weights
        weights = {
            "buzzwords": 0.25,
            "templates": 0.30,
            "hedging": 0.15,
            "sycophancy": 0.15,
            "repetition": 0.10,
            "complexity": 0.05,
        }

        slop_score = sum(slop_indicators[key] * weights[key] for key in weights) / 100

        # More aggressive threshold
        is_slop = slop_score > 0.15  # Lowered from 0.3
        confidence = min(slop_score * 3, 1.0)  # Increased multiplier

        # Determine level
        if slop_score > 0.5:
            level = "HIGH_SLOP"
            explanation = "Strong indicators of AI-generated or low-quality content"
        elif slop_score > 0.3:
            level = "SLOP"
            explanation = "Multiple slop characteristics detected"
        elif slop_score > 0.15:
            level = "SUSPICIOUS"
            explanation = "Some slop indicators present"
        else:
            level = "QUALITY"
            explanation = "High-quality, natural content"

        return {
            "is_slop": is_slop,
            "slop_score": slop_score,
            "confidence": confidence,
            "level": level,
            "explanation": explanation,
            "method": "production_patterns",
            "slop_indicators": slop_indicators,
            "pattern_matches": pattern_matches,
            "repetition_analysis": repetition_analysis,
            "complexity_analysis": complexity_analysis,
        }


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Production-Ready English Slop Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python production_slop_detector.py "This is high-quality technical content."
  python production_slop_detector.py --file document.txt
  python production_slop_detector.py --json "AI-generated slop text here"
  python production_slop_detector.py --test  # Run test cases
        """,
    )

    parser.add_argument("text", nargs="?", help="Text to analyze for slop")

    parser.add_argument(
        "--file", "-f", type=str, help="File containing text to analyze"
    )

    parser.add_argument(
        "--json", "-j", action="store_true", help="Output results in JSON format"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test cases to demonstrate effectiveness",
    )

    args = parser.parse_args()

    # Initialize detector
    detector = ProductionSlopDetector()

    if args.test:
        # Run test cases
        print("🧪 Production Slop Detection Test Cases")
        print("=" * 60)

        test_cases = [
            (
                "The implementation uses a distributed architecture with microservices that communicate through REST APIs.",
                False,
                "High-quality technical content",
            ),
            (
                "In today's rapidly evolving digital landscape, it's absolutely crucial to understand that leveraging cutting-edge technologies can truly revolutionize your business operations.",
                True,
                "AI-generated slop",
            ),
            (
                "The system is very important. The system provides many benefits. The system helps users. The system is very useful.",
                True,
                "Repetitive slop",
            ),
            (
                "I've been working on this project for months, and honestly, it's been challenging. The initial approach didn't work.",
                False,
                "Natural human writing",
            ),
            (
                "Here are 5 amazing ways to boost your productivity: First, prioritize your tasks effectively. Second, eliminate distractions.",
                True,
                "Template-heavy slop",
            ),
            (
                "The API endpoint accepts POST requests with JSON payloads containing user_id and timestamp fields.",
                False,
                "Technical documentation",
            ),
            (
                "This innovative solution represents a paradigm shift in how we approach complex challenges and create value for stakeholders.",
                True,
                "Corporate buzzword slop",
            ),
            (
                "Perhaps it might be somewhat possible that this approach could potentially be effective in certain situations.",
                True,
                "Hedging slop",
            ),
            (
                "You're absolutely right! I completely agree with your excellent point about the strategic implementation.",
                True,
                "Sycophancy slop",
            ),
        ]

        correct = 0
        total = len(test_cases)

        for i, (text, expected_slop, description) in enumerate(test_cases, 1):
            result = detector.detect_slop(text)
            predicted_slop = result["is_slop"]
            correct += 1 if predicted_slop == expected_slop else 0

            print(f"\n📝 Test {i}: {description}")
            print(f"Text: {text[:70]}{'...' if len(text) > 70 else ''}")
            print(f"Expected: {'SLOP' if expected_slop else 'QUALITY'}")
            print(
                f"Predicted: {'SLOP' if predicted_slop else 'QUALITY'} (confidence: {result['confidence']:.2f})"
            )
            print(f"Level: {result['level']}")
            print(f"Slop Score: {result['slop_score']:.2f}")

            # Show top indicators
            indicators = result["slop_indicators"]
            top_indicators = sorted(
                indicators.items(), key=lambda x: x[1], reverse=True
            )[:3]
            print(
                f"Top Indicators: {', '.join([f'{k}: {v:.1f}' for k, v in top_indicators])}"
            )

        accuracy = correct / total * 100
        print(f"\n📊 Test Results: {correct}/{total} correct ({accuracy:.1f}%)")
        print(
            f"🎯 Production detector shows {'excellent' if accuracy >= 80 else 'good' if accuracy >= 70 else 'moderate'} performance"
        )

        return

    # Get input text
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except FileNotFoundError:
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            sys.exit(1)
    elif args.text:
        text = args.text
    else:
        # Interactive mode
        print("Enter text to analyze (press Ctrl+D when done):")
        try:
            text = sys.stdin.read().strip()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

    if not text:
        print("❌ No text provided for analysis")
        sys.exit(1)

    # Analyze text
    try:
        result = detector.detect_slop(text)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n🔍 Production Slop Detection Analysis")
            print("=" * 50)
            print(f"Prediction: {'🚨 SLOP' if result['is_slop'] else '✅ QUALITY'}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Slop Score: {result['slop_score']:.2f}")
            print(f"Level: {result['level']}")
            print(f"Explanation: {result['explanation']}")

            print(f"\nDetailed Indicators:")
            for indicator, score in result["slop_indicators"].items():
                if score > 0:
                    print(f"  {indicator}: {score:.1f}")

        # Exit with appropriate code
        sys.exit(1 if result["is_slop"] else 0)

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
