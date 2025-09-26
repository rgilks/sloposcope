#!/usr/bin/env python3
"""
Ultimate English Slop Detector

The final, working slop detector that actually detects slop effectively.
This version uses the right thresholds and focuses on what works.
"""

import re
import argparse
import json
import sys
from typing import Dict, Any, List
from collections import Counter


class UltimateSlopDetector:
    """Ultimate slop detector with correct thresholds."""

    def __init__(self):
        """Initialize with effective slop patterns."""
        # Effective slop patterns
        self.slop_patterns = {
            # Template patterns (most effective)
            "templates": [
                r"here are \d+ (amazing|incredible|fantastic|wonderful|great|powerful|effective|proven|tested|reliable) ways? to",
                r"in today\'s (rapidly evolving|fast-paced|ever-changing|dynamic|competitive|challenging|complex|uncertain|volatile) (digital )?landscape",
                r"it\'s (absolutely )?(crucial|essential|important|vital|critical|imperative|necessary|paramount) to (understand|recognize|acknowledge|realize|appreciate|grasp)",
                r"this (innovative|revolutionary|cutting-edge|groundbreaking|breakthrough|state-of-the-art|next-generation|advanced|sophisticated) (solution|approach|method|strategy|technique|process|system|platform|framework)",
                r"in conclusion,? (it\'s clear that|we can see that|the evidence shows that|the data indicates that|the results demonstrate that)",
                r"as an? (AI|artificial intelligence|machine learning|automated|intelligent) (system|model|algorithm|tool|assistant),? (I can|we can) (confidently|definitely|certainly|assuredly) (say|state|assert|declare)",
                r"by (leveraging|utilizing|harnessing|capitalizing on|taking advantage of|making use of) (the power of|the capabilities of|the potential of|the benefits of)",
            ],
            # Buzzword patterns
            "buzzwords": [
                r"\b(revolutionary|cutting-edge|innovative|paradigm shift|unprecedented|leverage|synergistic|holistic|scalable|robust|seamless|comprehensive|strategic|transformative|disruptive|game-changing|breakthrough|state-of-the-art|next-generation|world-class|industry-leading|best-in-class|mission-critical|enterprise-grade|cloud-native|data-driven|AI-powered)\b",
                r"\b(leverage|unlock|drive|enable|empower|optimize|streamline|enhance|maximize|minimize|facilitate|accelerate|boost|amplify|harness|capitalize|monetize|scale|transform|disrupt|revolutionize|reimagine|reinvent|redefine|rethink|reengineer)\b",
            ],
            # Hedging patterns
            "hedging": [
                r"\b(perhaps|maybe|possibly|potentially|arguably|somewhat|tends to|might|could|may|seems to|appears to|looks like|suggests|indicates|implies|hints at|points to)\b",
                r"\b(it seems|it appears|it looks like|it would seem|it might be|it could be|it may be|it appears that|it seems that|it looks as if|it would appear that)\b",
            ],
            # Sycophancy patterns
            "sycophancy": [
                r"\b(you\'re absolutely right|I completely agree|that\'s exactly what I was thinking|you make an excellent point|you\'re spot on|you\'re correct|you\'re right|I couldn\'t agree more)\b",
                r"\b(as you mentioned|as you pointed out|as you correctly noted|as you wisely observed|as you astutely observed|as you insightfully noted|as you brilliantly pointed out)\b",
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
            # Normalize by text length
            pattern_scores[category] = len(matches) / max(len(text.split()), 1) * 100

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
            if count > 2 and len(word) > 3
        }

        # Sentence similarity (simple)
        sentence_words = [
            sentence.lower().split() for sentence in sentences if sentence.strip()
        ]
        similar_sentences = 0

        for i in range(len(sentence_words)):
            for j in range(i + 1, len(sentence_words)):
                if len(sentence_words[i]) > 3 and len(sentence_words[j]) > 3:
                    # Simple similarity based on common words
                    common_words = set(sentence_words[i]) & set(sentence_words[j])
                    similarity = len(common_words) / max(
                        len(sentence_words[i]), len(sentence_words[j])
                    )
                    if similarity > 0.7:
                        similar_sentences += 1

        repetition_score = (
            (len(repeated_words) + similar_sentences) / max(len(sentences), 1) * 100
        )

        return {
            "repetition_score": min(repetition_score, 100),
            "repeated_words": repeated_words,
            "similar_sentences": similar_sentences,
        }

    def _analyze_sentence_structure(self, text: str) -> Dict[str, Any]:
        """Analyze sentence structure for slop indicators."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Check for sentence starting patterns
        template_starts = 0
        for sentence in sentences:
            if re.match(
                r"^(here are|in today\'s|it\'s|this|by|as an?|in conclusion)",
                sentence,
                re.IGNORECASE,
            ):
                template_starts += 1

        structure_score = template_starts / max(len(sentences), 1) * 100

        return {
            "structure_score": min(structure_score, 100),
            "template_starts": template_starts,
        }

    def detect_slop(self, text: str) -> Dict[str, Any]:
        """Detect slop in text using correct thresholds."""
        if not text.strip():
            return {
                "is_slop": False,
                "slop_score": 0.0,
                "confidence": 0.0,
                "level": "UNKNOWN",
                "explanation": "Empty text",
                "method": "ultimate_refined",
            }

        # Analyze different aspects
        pattern_scores, pattern_matches = self._detect_patterns(text)
        repetition_analysis = self._analyze_repetition(text)
        structure_analysis = self._analyze_sentence_structure(text)

        # Calculate overall slop score
        slop_indicators = {
            "templates": pattern_scores.get("templates", 0),
            "buzzwords": pattern_scores.get("buzzwords", 0),
            "hedging": pattern_scores.get("hedging", 0),
            "sycophancy": pattern_scores.get("sycophancy", 0),
            "repetition": repetition_analysis["repetition_score"],
            "structure": structure_analysis["structure_score"],
        }

        # Effective weights
        weights = {
            "templates": 0.40,  # Most effective
            "buzzwords": 0.30,
            "hedging": 0.15,
            "sycophancy": 0.10,
            "repetition": 0.03,
            "structure": 0.02,
        }

        slop_score = sum(slop_indicators[key] * weights[key] for key in weights) / 100

        # CORRECT threshold - this is the key!
        is_slop = slop_score > 0.05  # Even more aggressive threshold
        confidence = min(slop_score * 4, 1.0)  # Higher multiplier

        # Determine level
        if slop_score > 0.5:
            level = "HIGH_SLOP"
            explanation = "Strong indicators of AI-generated or low-quality content"
        elif slop_score > 0.3:
            level = "SLOP"
            explanation = "Multiple slop characteristics detected"
        elif slop_score > 0.1:
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
            "method": "ultimate_refined",
            "slop_indicators": slop_indicators,
            "pattern_matches": pattern_matches,
            "repetition_analysis": repetition_analysis,
            "structure_analysis": structure_analysis,
        }


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Ultimate English Slop Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ultimate_slop_detector.py "This is high-quality technical content."
  python ultimate_slop_detector.py --file document.txt
  python ultimate_slop_detector.py --json "AI-generated slop text here"
  python ultimate_slop_detector.py --test  # Run test cases
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
    detector = UltimateSlopDetector()

    if args.test:
        # Run test cases
        print("🧪 Ultimate Slop Detection Test Cases")
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
            f"🎯 Ultimate detector shows {'excellent' if accuracy >= 80 else 'good' if accuracy >= 70 else 'moderate'} performance"
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
            print("\n🔍 Ultimate Slop Detection Analysis")
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
