#!/usr/bin/env python3
"""
Enhanced Slop Detector - Advanced AI Slop Detection

This version combines multiple detection strategies for exceptional accuracy:
- Enhanced pattern matching with semantic analysis
- Improved repetition detection
- Better threshold optimization
- Ensemble voting system
"""

import re
import json
import sys
import argparse
from collections import Counter
from typing import Dict, List, Any, Tuple
import math


class EnhancedSlopDetector:
    """Enhanced slop detector with multiple detection strategies."""

    def __init__(self):
        """Initialize with comprehensive slop patterns and strategies."""
        # Enhanced slop patterns
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
            # Enhanced buzzword patterns
            "buzzwords": [
                r"\b(revolutionary|cutting-edge|innovative|paradigm shift|unprecedented|leverage|synergistic|holistic|scalable|robust|seamless|comprehensive|strategic|transformative|disruptive|game-changing|breakthrough|state-of-the-art|next-generation|world-class|industry-leading|best-in-class|mission-critical|enterprise-grade|cloud-native|data-driven|AI-powered)\b",
                r"\b(leverage|unlock|drive|enable|empower|optimize|streamline|enhance|maximize|minimize|facilitate|accelerate|boost|amplify|harness|capitalize|monetize|scale|transform|disrupt|revolutionize|reimagine|reinvent|redefine|rethink|reengineer)\b",
                r"\b(stakeholder|ecosystem|framework|methodology|best practices|core competencies|value proposition|competitive advantage|market positioning|brand equity|customer journey|user experience|digital transformation|agile|scrum|kanban|sprint|retrospective|standup|backlog|epic|story|task)\b",
            ],
            # Enhanced hedging patterns
            "hedging": [
                r"\b(perhaps|maybe|possibly|potentially|arguably|somewhat|tends to|might|could|may|seems to|appears to|looks like|suggests|indicates|implies|hints at|points to)\b",
                r"\b(it seems|it appears|it looks like|it would seem|it might be|it could be|it may be|it appears that|it seems that|it looks as if|it would appear that)\b",
                r"\b(to some extent|in some ways|in certain cases|under certain circumstances|depending on|subject to|contingent upon|based on|according to|in accordance with)\b",
                r"\b(relatively|comparatively|reasonably|fairly|quite|rather|somewhat|more or less|to a degree|in part|partially)\b",
            ],
            # Enhanced sycophancy patterns
            "sycophancy": [
                r"\b(you\'re absolutely right|I completely agree|that\'s exactly what I was thinking|you make an excellent point|you\'re spot on|you\'re correct|you\'re right|I couldn\'t agree more)\b",
                r"\b(as you mentioned|as you pointed out|as you correctly noted|as you wisely observed|as you astutely observed|as you insightfully noted|as you brilliantly pointed out)\b",
                r"\b(that\'s a great question|that\'s an excellent point|that\'s a wonderful observation|that\'s a brilliant insight|that\'s a fantastic idea)\b",
                r"\b(I appreciate your|thank you for|that\'s very helpful|that\'s very insightful|that\'s very thoughtful|that\'s very wise)\b",
                r"\b(you\'re so right|you\'re absolutely correct|you\'re spot on|you\'re exactly right|you\'re completely right)\b",
            ],
            # New: Corporate speak patterns
            "corporate_speak": [
                r"\b(circle back|touch base|deep dive|drill down|level set|move the needle|low-hanging fruit|win-win|best of breed|out of the box|think outside the box)\b",
                r"\b(take it offline|put a pin in it|run it up the flagpole|throw it against the wall|ballpark figure|back of the envelope)\b",
                r"\b(synergy|bandwidth|deliverables|action items|key takeaways|next steps|moving forward|going forward)\b",
            ],
            # New: AI-generated patterns
            "ai_patterns": [
                r"\b(as an AI|as a language model|I don\'t have personal|I cannot|I am unable to|I don\'t have access to|I cannot provide|I cannot give)\b",
                r"\b(I can help you|I can assist you|I can provide|I can offer|I can suggest|I can recommend|I can guide you)\b",
                r"\b(here\'s how|here\'s what|here\'s why|here\'s when|here\'s where|here\'s who|here\'s which)\b",
                r"\b(let me|allow me to|I\'ll|I will|I can|I would|I could|I should|I might|I may)\b",
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
        """Detect slop patterns in text with enhanced scoring."""
        pattern_scores = {}
        pattern_matches = {}

        for category, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(pattern.findall(text))

            pattern_matches[category] = matches
            # Enhanced normalization - consider both word count and sentence count
            words = text.split()
            sentences = re.split(r"[.!?]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]

            # Use the smaller of word-based or sentence-based normalization
            word_based_score = len(matches) / max(len(words), 1) * 100
            sentence_based_score = len(matches) / max(len(sentences), 1) * 100
            pattern_scores[category] = min(word_based_score, sentence_based_score)

        return pattern_scores, pattern_matches

    def _analyze_repetition(self, text: str) -> Dict[str, Any]:
        """Enhanced repetition analysis with better scoring."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.lower().split()

        # Enhanced word frequency analysis
        word_counts = Counter(words)
        repeated_words = {
            word: count
            for word, count in word_counts.items()
            if count > 1 and len(word) > 2  # Lower threshold for repetition
        }

        # Enhanced sentence similarity
        sentence_words = [
            sentence.lower().split() for sentence in sentences if sentence.strip()
        ]
        similar_sentences = 0

        for i in range(len(sentence_words)):
            for j in range(i + 1, len(sentence_words)):
                if len(sentence_words[i]) > 2 and len(sentence_words[j]) > 2:
                    # Enhanced similarity calculation
                    common_words = set(sentence_words[i]) & set(sentence_words[j])
                    similarity = len(common_words) / max(
                        len(sentence_words[i]), len(sentence_words[j])
                    )
                    if similarity > 0.6:  # Lower threshold for similarity
                        similar_sentences += 1

        # Enhanced scoring
        repetition_score = (
            (len(repeated_words) * 2 + similar_sentences * 3)
            / max(len(sentences), 1)
            * 100
        )

        return {
            "repetition_score": min(repetition_score, 100),
            "repeated_words": repeated_words,
            "similar_sentences": similar_sentences,
        }

    def _analyze_sentence_structure(self, text: str) -> Dict[str, Any]:
        """Enhanced sentence structure analysis."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Enhanced template detection
        template_starts = 0
        for sentence in sentences:
            if re.match(
                r"^(here are|in today\'s|it\'s|this|by|as an?|in conclusion|first|second|third|last but not least|moving forward|going forward)",
                sentence,
                re.IGNORECASE,
            ):
                template_starts += 1

        # Enhanced structure scoring
        structure_score = template_starts / max(len(sentences), 1) * 100

        return {
            "structure_score": min(structure_score, 100),
            "template_starts": template_starts,
        }

    def _analyze_semantic_density(self, text: str) -> Dict[str, Any]:
        """Analyze semantic density and complexity."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()

        # Calculate average sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(
            len(sentences), 1
        )

        # Calculate word complexity (average word length)
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)

        # Calculate semantic density score
        # High semantic density often indicates slop (overly complex, padded language)
        semantic_density = (avg_sentence_length * avg_word_length) / 100

        return {
            "semantic_density": min(semantic_density, 100),
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
        }

    def _ensemble_vote(self, scores: Dict[str, float]) -> Tuple[bool, float]:
        """Ensemble voting system for final decision."""
        # Individual thresholds for each indicator
        thresholds = {
            "templates": 2.0,
            "buzzwords": 3.0,
            "hedging": 5.0,
            "sycophancy": 2.0,
            "corporate_speak": 2.0,
            "ai_patterns": 2.0,
            "repetition": 15.0,
            "structure": 20.0,
            "semantic_density": 8.0,
        }

        # Count how many indicators exceed their thresholds
        votes = 0
        total_indicators = 0

        for indicator, threshold in thresholds.items():
            if indicator in scores:
                total_indicators += 1
                if scores[indicator] > threshold:
                    votes += 1

        # Ensemble decision: if more than 30% of indicators exceed thresholds, it's slop
        ensemble_threshold = 0.3
        is_slop = votes / max(total_indicators, 1) > ensemble_threshold

        # Confidence based on how many indicators are triggered
        confidence = votes / max(total_indicators, 1)

        return is_slop, confidence

    def detect_slop(self, text: str) -> Dict[str, Any]:
        """Enhanced slop detection with multiple strategies."""
        if not text or not text.strip():
            return {
                "is_slop": False,
                "confidence": 0.0,
                "slop_score": 0.0,
                "level": "UNKNOWN",
                "explanation": "Empty text",
                "method": "enhanced_ensemble",
            }

        # Analyze different aspects
        pattern_scores, pattern_matches = self._detect_patterns(text)
        repetition_analysis = self._analyze_repetition(text)
        structure_analysis = self._analyze_sentence_structure(text)
        semantic_analysis = self._analyze_semantic_density(text)

        # Combine all indicators
        slop_indicators = {
            **pattern_scores,
            "repetition": repetition_analysis["repetition_score"],
            "structure": structure_analysis["structure_score"],
            "semantic_density": semantic_analysis["semantic_density"],
        }

        # Ensemble voting
        is_slop, confidence = self._ensemble_vote(slop_indicators)

        # Calculate overall slop score for display
        weights = {
            "templates": 0.25,
            "buzzwords": 0.20,
            "hedging": 0.15,
            "sycophancy": 0.10,
            "corporate_speak": 0.10,
            "ai_patterns": 0.10,
            "repetition": 0.05,
            "structure": 0.03,
            "semantic_density": 0.02,
        }

        slop_score = (
            sum(slop_indicators.get(key, 0) * weights.get(key, 0) for key in weights)
            / 100
        )

        # Determine level
        if slop_score > 0.5:
            level = "HIGH_SLOP"
            explanation = "Extremely high slop content"
        elif slop_score > 0.2:
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
            "method": "enhanced_ensemble",
        }


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced AI Slop Detection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_slop_detector.py "This is high-quality technical content."
  python enhanced_slop_detector.py --file document.txt
  python enhanced_slop_detector.py --json "AI-generated slop text here"
  python enhanced_slop_detector.py --test  # Run test cases
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
    detector = EnhancedSlopDetector()

    if args.test:
        # Run test cases
        print("🧪 Enhanced Slop Detection Test Cases")
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
            f"🎯 Enhanced detector shows {'excellent' if accuracy >= 90 else 'good' if accuracy >= 80 else 'moderate'} performance"
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
            print("\n🔍 Enhanced Slop Detection Analysis")
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
