#!/usr/bin/env python3
"""
Ultimate Slop Detector v3 - Final Refinement for Maximum Accuracy

This version specifically addresses repetitive slop detection while maintaining
excellent performance on other slop types.
"""

import re
import json
import sys
import argparse
from collections import Counter
from typing import Dict, List, Any, Tuple


class UltimateSlopDetectorV3:
    """Ultimate slop detector v3 with refined repetitive slop detection."""

    def __init__(self):
        """Initialize with the most effective slop patterns."""
        # Most effective slop patterns
        self.slop_patterns = {
            # Template patterns (highest effectiveness)
            "templates": [
                r"here are \d+ (amazing|incredible|fantastic|wonderful|great|powerful|effective|proven|tested|reliable) ways? to",
                r"in today\'s (rapidly evolving|fast-paced|ever-changing|dynamic|competitive|challenging|complex|uncertain|volatile) (digital )?landscape",
                r"it\'s (absolutely )?(crucial|essential|important|vital|critical|imperative|necessary|paramount) to (understand|recognize|acknowledge|realize|appreciate|grasp)",
                r"this (innovative|revolutionary|cutting-edge|groundbreaking|breakthrough|state-of-the-art|next-generation|advanced|sophisticated) (solution|approach|method|strategy|technique|process|system|platform|framework)",
                r"in conclusion,? (it\'s clear that|we can see that|the evidence shows that|the data indicates that|the results demonstrate that)",
                r"as an? (AI|artificial intelligence|machine learning|automated|intelligent) (system|model|algorithm|tool|assistant),? (I can|we can) (confidently|definitely|certainly|assuredly) (say|state|assert|declare)",
                r"by (leveraging|utilizing|harnessing|capitalizing on|taking advantage of|making use of) (the power of|the capabilities of|the potential of|the benefits of)",
                r"first,? (and foremost|of all|and most importantly)",
                r"second,? (and equally important|of all|and most importantly)",
                r"third,? (and finally|of all|and most importantly)",
                r"last but not least",
                r"moving forward",
                r"going forward",
                r"at the end of the day",
                r"it\'s worth noting that",
                r"it\'s important to (understand|recognize|acknowledge|realize|appreciate|grasp)",
            ],
            # Buzzword patterns (high effectiveness)
            "buzzwords": [
                r"\b(revolutionary|cutting-edge|innovative|paradigm shift|unprecedented|leverage|synergistic|holistic|scalable|robust|seamless|comprehensive|strategic|transformative|disruptive|game-changing|breakthrough|state-of-the-art|next-generation|world-class|industry-leading|best-in-class|mission-critical|enterprise-grade|cloud-native|data-driven|AI-powered)\b",
                r"\b(leverage|unlock|drive|enable|empower|optimize|streamline|enhance|maximize|minimize|facilitate|accelerate|boost|amplify|harness|capitalize|monetize|scale|transform|disrupt|revolutionize|reimagine|reinvent|redefine|rethink|reengineer)\b",
            ],
            # Hedging patterns (moderate effectiveness)
            "hedging": [
                r"\b(perhaps|maybe|possibly|potentially|arguably|somewhat|tends to|might|could|may|seems to|appears to|looks like|suggests|indicates|implies|hints at|points to)\b",
                r"\b(it seems|it appears|it looks like|it would seem|it might be|it could be|it may be|it appears that|it seems that|it looks as if|it would appear that)\b",
                r"\b(to some extent|in some ways|in certain cases|under certain circumstances|depending on|subject to|contingent upon|based on|according to|in accordance with)\b",
            ],
            # Sycophancy patterns (moderate effectiveness)
            "sycophancy": [
                r"\b(you\'re absolutely right|I completely agree|that\'s exactly what I was thinking|you make an excellent point|you\'re spot on|you\'re correct|you\'re right|I couldn\'t agree more)\b",
                r"\b(as you mentioned|as you pointed out|as you correctly noted|as you wisely observed|as you astutely observed|as you insightfully noted|as you brilliantly pointed out)\b",
                r"\b(that\'s a great question|that\'s an excellent point|that\'s a wonderful observation|that\'s a brilliant insight|that\'s a fantastic idea)\b",
            ],
            # Corporate speak patterns (high effectiveness)
            "corporate_speak": [
                r"\b(circle back|touch base|deep dive|drill down|level set|move the needle|low-hanging fruit|win-win|best of breed|out of the box|think outside the box)\b",
                r"\b(take it offline|put a pin in it|run it up the flagpole|throw it against the wall|ballpark figure|back of the envelope)\b",
                r"\b(synergy|bandwidth|deliverables|action items|key takeaways|next steps|moving forward|going forward)\b",
            ],
            # AI-generated patterns (high effectiveness)
            "ai_patterns": [
                r"\b(as an AI|as a language model|I don\'t have personal|I cannot|I am unable to|I don\'t have access to|I cannot provide|I cannot give)\b",
                r"\b(I can help you|I can assist you|I can provide|I can offer|I can suggest|I can recommend|I can guide you)\b",
                r"\b(here\'s how|here\'s what|here\'s why|here\'s when|here\'s where|here\'s who|here\'s which)\b",
            ],
        }

        # Compile patterns
        self.compiled_patterns = {}
        for category, patterns in self.slop_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def _detect_patterns(
        self, text: str
    ) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Detect slop patterns with optimized scoring."""
        pattern_scores = {}
        pattern_matches = {}

        for category, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(pattern.findall(text))

            pattern_matches[category] = matches
            # Optimized scoring - use word count for normalization
            words = text.split()
            pattern_scores[category] = len(matches) / max(len(words), 1) * 100

        return pattern_scores, pattern_matches

    def _analyze_repetition(self, text: str) -> Dict[str, Any]:
        """Enhanced repetition analysis with aggressive scoring for repetitive slop."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.lower().split()

        # Aggressive word frequency analysis
        word_counts = Counter(words)
        repeated_words = {
            word: count
            for word, count in word_counts.items()
            if count > 1 and len(word) > 2
        }

        # Enhanced sentence similarity with lower threshold
        sentence_words = [
            sentence.lower().split() for sentence in sentences if sentence.strip()
        ]
        similar_sentences = 0

        for i in range(len(sentence_words)):
            for j in range(i + 1, len(sentence_words)):
                if len(sentence_words[i]) > 2 and len(sentence_words[j]) > 2:
                    common_words = set(sentence_words[i]) & set(sentence_words[j])
                    similarity = len(common_words) / max(
                        len(sentence_words[i]), len(sentence_words[j])
                    )
                    if similarity > 0.4:  # Much lower threshold for similarity
                        similar_sentences += 1

        # Check for repetitive sentence structures (like "The system is...")
        repetitive_structures = 0
        for sentence in sentences:
            if re.match(
                r"^(the \w+ is|the \w+ provides|the \w+ helps|the \w+ can)",
                sentence.lower(),
            ):
                repetitive_structures += 1

        # Aggressive scoring with multiple factors
        repetition_score = (
            (
                len(repeated_words) * 4
                + similar_sentences * 6
                + repetitive_structures * 8
            )
            / max(len(sentences), 1)
            * 100
        )

        return {
            "repetition_score": min(repetition_score, 100),
            "repeated_words": repeated_words,
            "similar_sentences": similar_sentences,
            "repetitive_structures": repetitive_structures,
        }

    def _analyze_sentence_structure(self, text: str) -> Dict[str, Any]:
        """Analyze sentence structure for slop indicators."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Template detection
        template_starts = 0
        for sentence in sentences:
            if re.match(
                r"^(here are|in today\'s|it\'s|this|by|as an?|in conclusion|first|second|third|last but not least|moving forward|going forward)",
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
        """Detect slop using optimized thresholds and scoring."""
        if not text or not text.strip():
            return {
                "is_slop": False,
                "confidence": 0.0,
                "slop_score": 0.0,
                "level": "UNKNOWN",
                "explanation": "Empty text",
                "method": "ultimate_v3",
            }

        # Analyze different aspects
        pattern_scores, pattern_matches = self._detect_patterns(text)
        repetition_analysis = self._analyze_repetition(text)
        structure_analysis = self._analyze_sentence_structure(text)

        # Combine all indicators
        slop_indicators = {
            **pattern_scores,
            "repetition": repetition_analysis["repetition_score"],
            "structure": structure_analysis["structure_score"],
        }

        # Optimized weights with higher weight for repetition
        weights = {
            "templates": 0.30,  # High effectiveness
            "buzzwords": 0.25,
            "corporate_speak": 0.15,
            "ai_patterns": 0.10,
            "repetition": 0.10,  # Increased weight for repetition
            "hedging": 0.05,
            "sycophancy": 0.03,
            "structure": 0.02,
        }

        # Calculate weighted slop score
        slop_score = (
            sum(slop_indicators.get(key, 0) * weights.get(key, 0) for key in weights)
            / 100
        )

        # OPTIMIZED THRESHOLD - aggressive but balanced
        is_slop = slop_score > 0.015  # Slightly more aggressive threshold
        confidence = min(slop_score * 10, 1.0)  # Higher multiplier

        # Determine level
        if slop_score > 0.3:
            level = "HIGH_SLOP"
            explanation = "Extremely high slop content"
        elif slop_score > 0.1:
            level = "MEDIUM_SLOP"
            explanation = "Moderate slop content"
        elif is_slop:
            level = "LOW_SLOP"
            explanation = "Low-level slop detected"
        else:
            level = "QUALITY"
            explanation = "High-quality, natural content"

        return {
            "is_slop": is_slop,
            "confidence": confidence,
            "slop_score": slop_score,
            "level": level,
            "explanation": explanation,
            "slop_indicators": slop_indicators,
            "pattern_matches": pattern_matches,
            "method": "ultimate_v3",
        }


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Ultimate AI Slop Detection Tool v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ultimate_slop_detector_v3.py "This is high-quality technical content."
  python ultimate_slop_detector_v3.py --file document.txt
  python ultimate_slop_detector_v3.py --json "AI-generated slop text here"
  python ultimate_slop_detector_v3.py --test  # Run test cases
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
    detector = UltimateSlopDetectorV3()

    if args.test:
        # Run test cases
        print("🧪 Ultimate Slop Detection v3 Test Cases")
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
            (
                "Let's circle back on this and touch base next week. We need to drill down into the deliverables and move the needle on our KPIs.",
                True,
                "Corporate speak slop",
            ),
            (
                "As an AI, I can help you with that. Here's how you can approach this problem step by step.",
                True,
                "AI-generated pattern slop",
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
            f"🎯 Ultimate v3 detector shows {'excellent' if accuracy >= 95 else 'good' if accuracy >= 90 else 'moderate'} performance"
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
    elif args.text:
        text = args.text
    else:
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
            print("\n🔍 Ultimate Slop Detection v3 Analysis")
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
