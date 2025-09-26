#!/usr/bin/env python3
"""
Aggressive Slop Detector - Right Balance

This version is more aggressive with reliable slop indicators while maintaining
natural writing protection. Based on the research paper findings.
"""

import re
import json
import sys
import argparse
from collections import Counter
from typing import Dict, List, Any, Tuple
import math


class AggressiveSlopDetector:
    """Aggressive slop detector with right balance."""

    def __init__(self):
        """Initialize with effective slop patterns."""
        # Core slop patterns from research
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

    def _analyze_information_density(self, text: str) -> Dict[str, Any]:
        """
        Analyze information density - the core slop dimension from research.
        Based on: "Many words, little information; filler or fluff"
        """
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 1. Filler word detection
        filler_words = [
            "very", "really", "quite", "rather", "somewhat", "fairly", "pretty",
            "absolutely", "completely", "totally", "entirely", "extremely",
            "incredibly", "amazingly", "wonderfully", "fantastically",
            "indeed", "certainly", "definitely", "surely", "obviously",
            "clearly", "naturally", "of course", "undoubtedly",
            "various", "numerous", "multiple", "several", "many", "different",
            "kind of", "sort of", "type of", "way of", "manner of",
            "in order to", "so as to", "with the aim of", "with the purpose of",
            "it is important to", "it is crucial to", "it is essential to",
            "it should be noted", "it is worth noting", "it is worth mentioning",
            "in today's", "in the modern", "in the current", "in the present",
            "in the context of", "in terms of", "with regard to", "in relation to",
            "as a result", "as a consequence", "therefore", "thus", "hence",
            "furthermore", "moreover", "additionally", "in addition",
            "however", "nevertheless", "nonetheless", "on the other hand",
            "in conclusion", "to conclude", "in summary", "to summarize",
        ]
        
        filler_count = 0
        for word in words:
            if word.lower() in filler_words:
                filler_count += 1
        
        # 2. Redundant phrase detection
        redundant_phrases = [
            r"\b(in today's rapidly evolving|in today's fast-paced|in today's ever-changing)",
            r"\b(it is important to note that|it is worth noting that|it should be noted that)",
            r"\b(in order to|so as to|with the aim of|with the purpose of)",
            r"\b(it is clear that|it is obvious that|it is evident that)",
            r"\b(as a matter of fact|in fact|actually|indeed)",
            r"\b(needless to say|it goes without saying|obviously|clearly)",
            r"\b(in the final analysis|at the end of the day|when all is said and done)",
            r"\b(it is worth mentioning|it is important to mention|it should be mentioned)",
            r"\b(in this regard|in this context|in this respect|in this connection)",
            r"\b(as we have seen|as we can see|as we know|as we know it)",
        ]
        
        redundant_count = 0
        for pattern in redundant_phrases:
            matches = re.findall(pattern, text, re.IGNORECASE)
            redundant_count += len(matches)
        
        # 3. Verbosity analysis
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        
        # 4. Information-to-word ratio (heuristic)
        # Count meaningful words vs total words
        meaningful_words = 0
        for word in words:
            word_lower = word.lower().strip(".,!?;:")
            if (len(word_lower) > 3 and 
                word_lower not in filler_words and 
                not re.match(r"^(the|and|or|but|in|on|at|to|for|of|with|by)$", word_lower)):
                meaningful_words += 1
        
        information_ratio = meaningful_words / max(len(words), 1)
        
        # 5. Repetitive content detection
        word_counts = Counter(word.lower() for word in words if len(word) > 3)
        repetitive_words = sum(1 for count in word_counts.values() if count > 2)
        
        # 6. Calculate information density score
        filler_score = (filler_count / max(len(words), 1)) * 100
        redundant_score = (redundant_count / max(len(sentences), 1)) * 100
        verbosity_score = min((avg_sentence_length - 15) / 10 * 100, 100) if avg_sentence_length > 15 else 0
        repetition_score = (repetitive_words / max(len(words), 1)) * 100
        
        # Combined information density score
        density_score = (filler_score * 0.3 + redundant_score * 0.3 + 
                        verbosity_score * 0.2 + repetition_score * 0.2)
        
        return {
            "information_density_score": min(density_score, 100),
            "filler_words": filler_count,
            "redundant_phrases": redundant_count,
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "information_ratio": information_ratio,
            "repetitive_words": repetitive_words,
            "filler_score": filler_score,
            "redundant_score": redundant_score,
            "verbosity_score": verbosity_score,
            "repetition_score": repetition_score,
        }

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

    def _analyze_repetition_smart(self, text: str) -> Dict[str, Any]:
        """Smart repetition analysis that distinguishes natural vs slop repetition."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.lower().split()

        # Word frequency analysis
        word_counts = Counter(words)
        repeated_words = {
            word: count
            for word, count in word_counts.items()
            if count > 2 and len(word) > 3
        }

        # Check for slop-style repetitive structures
        slop_repetitive_structures = 0
        natural_repetitive_structures = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Slop patterns: "The X is Y. The X provides Z. The X helps W."
            if re.match(
                r"^(the \w+ is|the \w+ provides|the \w+ helps|the \w+ can|the \w+ will|the \w+ should)",
                sentence_lower,
            ):
                slop_repetitive_structures += 1
            
            # Natural patterns: "I love X. X is amazing. X teaches me."
            elif re.match(r"^(i \w+|we \w+|you \w+|they \w+)", sentence_lower):
                natural_repetitive_structures += 1

        # Sentence similarity with context awareness
        sentence_words = [
            sentence.lower().split() for sentence in sentences if sentence.strip()
        ]
        similar_sentences = 0
        natural_similar_sentences = 0

        for i in range(len(sentence_words)):
            for j in range(i + 1, len(sentence_words)):
                if len(sentence_words[i]) > 3 and len(sentence_words[j]) > 3:
                    common_words = set(sentence_words[i]) & set(sentence_words[j])
                    similarity = len(common_words) / max(
                        len(sentence_words[i]), len(sentence_words[j])
                    )
                    
                    if similarity > 0.7:  # High similarity threshold
                        # Check if it's natural or slop-style repetition
                        if re.match(
                            r"^(the \w+|this \w+|that \w+)", sentences[i].lower()
                        ) and re.match(
                            r"^(the \w+|this \w+|that \w+)", sentences[j].lower()
                        ):
                            similar_sentences += 1  # Slop-style
                        else:
                            natural_similar_sentences += 1  # Natural-style

        # Smart scoring: penalize slop repetition more than natural repetition
        slop_repetition_score = (
            (
                len(repeated_words) * 2
                + similar_sentences * 4
                + slop_repetitive_structures * 6
            )
            / max(len(sentences), 1)
            * 100
        )
        
        natural_repetition_score = (
            (natural_similar_sentences * 1 + natural_repetitive_structures * 1)
            / max(len(sentences), 1)
            * 100
        )

        # Final score: slop repetition minus natural repetition (with cap)
        final_repetition_score = max(
            0, slop_repetition_score - natural_repetition_score
        )

        return {
            "repetition_score": min(final_repetition_score, 100),
            "repeated_words": repeated_words,
            "similar_sentences": similar_sentences,
            "slop_repetitive_structures": slop_repetitive_structures,
            "natural_repetitive_structures": natural_repetitive_structures,
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

    def _is_natural_human_writing(
        self, text: str, slop_indicators: Dict[str, float]
    ) -> bool:
        """Enhanced natural human writing detection."""
        # Natural human writing indicators
        natural_indicators = [
            r"\b(I\'ve|I have|I was|I am|I\'m|I can|I will|I would|I should|I might|I may|I love|I like|I enjoy|I hate|I dislike)\b",
            r"\b(honestly|frankly|actually|really|truly|genuinely|personally|obviously|clearly|definitely|certainly)\b",
            r"\b(months|years|days|weeks|hours|minutes|ago|yesterday|today|tomorrow|yesterday|last week|next week)\b",
            r"\b(working on|been working|started|finished|completed|failed|succeeded|tried|attempted|managed)\b",
            r"\b(challenging|difficult|hard|easy|simple|complex|complicated|frustrating|annoying|amazing|wonderful|terrible|awful)\b",
            r"\b(weather|coffee|book|books|walk|park|birds|peaceful|relaxing|nice|good|hot|cold|warm|cool)\b",
            r"\b(thinking about|not sure|maybe|perhaps|probably|definitely|certainly|absolutely|definitely not)\b",
        ]
        
        natural_score = 0
        for pattern in natural_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            natural_score += len(matches)
        
        # Check for conversational patterns
        conversational_patterns = [
            r"\b(you know|I mean|like|um|uh|well|so|anyway|actually|basically)\b",
            r"\b(what do you think|do you know|have you seen|did you hear|can you believe)\b",
        ]
        
        conversational_score = 0
        for pattern in conversational_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            conversational_score += len(matches)
        
        # Enhanced natural writing detection
        if (
            (natural_score > 1 or conversational_score > 0)
            and slop_indicators.get("templates", 0) < 3
            and slop_indicators.get("buzzwords", 0) < 3
        ):
            return True
        
        return False

    def detect_slop(self, text: str) -> Dict[str, Any]:
        """Detect slop using aggressive but balanced approach."""
        if not text or not text.strip():
            return {
                "is_slop": False,
                "confidence": 0.0,
                "slop_score": 0.0,
                "level": "UNKNOWN",
                "explanation": "Empty text",
                "method": "aggressive",
            }

        # Analyze different aspects
        pattern_scores, pattern_matches = self._detect_patterns(text)
        repetition_analysis = self._analyze_repetition_smart(text)
        structure_analysis = self._analyze_sentence_structure(text)
        density_analysis = self._analyze_information_density(text)

        # Combine all indicators
        slop_indicators = {
            **pattern_scores,
            "repetition": repetition_analysis["repetition_score"],
            "structure": structure_analysis["structure_score"],
            "information_density": density_analysis["information_density_score"],
        }

        # Check for natural human writing
        is_natural = self._is_natural_human_writing(text, slop_indicators)

        # Aggressive weights - focus on reliable indicators
        weights = {
            "information_density": 0.25,  # Research-based dimension
            "templates": 0.30,  # Highest effectiveness - more aggressive
            "buzzwords": 0.25,  # High effectiveness - more aggressive
            "ai_patterns": 0.15,  # Important for AI detection - more aggressive
            "corporate_speak": 0.03,
            "hedging": 0.01,
            "sycophancy": 0.01,
            "repetition": 0.00,  # Disabled - too prone to false positives
            "structure": 0.00,  # Disabled - too prone to false positives
        }

        # Calculate weighted slop score
        slop_score = (
            sum(slop_indicators.get(key, 0) * weights.get(key, 0) for key in weights)
            / 100
        )

        # Aggressive threshold - lower threshold for better detection
        base_threshold = 0.025
        threshold = base_threshold * 2 if is_natural else base_threshold
        
        is_slop = slop_score > threshold
        confidence = min(slop_score * 6, 1.0)

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
            "is_natural_writing": is_natural,
            "density_analysis": density_analysis,
            "method": "aggressive",
        }


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Aggressive AI Slop Detection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aggressive_slop_detector.py "This is high-quality technical content."
  python aggressive_slop_detector.py --file document.txt
  python aggressive_slop_detector.py --json "AI-generated slop text here"
  python aggressive_slop_detector.py --test  # Run test cases
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
    detector = AggressiveSlopDetector()

    if args.test:
        # Run test cases
        print("🧪 Aggressive Slop Detection Test Cases")
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
            (
                "The weather is nice today. I went for a walk in the park and saw some birds. It was peaceful and relaxing.",
                False,
                "Natural repetitive writing",
            ),
            (
                "I love reading books. Books are amazing. Books teach me things. Books are wonderful.",
                False,
                "Natural repetitive writing",
            ),
            (
                "The coffee was good. The coffee was hot. The coffee was perfect. I enjoyed the coffee very much.",
                False,
                "Natural repetitive writing",
            ),
            (
                "In today's rapidly evolving digital landscape, it is absolutely crucial to understand that leveraging cutting-edge technologies can truly revolutionize your business operations and drive unprecedented value creation.",
                True,
                "High information density slop",
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
            print(f"Natural Writing: {result.get('is_natural_writing', False)}")
            print(f"Information Density: {result.get('density_analysis', {}).get('information_density_score', 0):.1f}")

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
            f"🎯 Aggressive detector shows {'excellent' if accuracy >= 95 else 'good' if accuracy >= 90 else 'moderate'} performance"
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
            print("\n🔍 Aggressive Slop Detection Analysis")
            print("=" * 50)
            print(f"Prediction: {'🚨 SLOP' if result['is_slop'] else '✅ QUALITY'}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Slop Score: {result['slop_score']:.2f}")
            print(f"Level: {result['level']}")
            print(f"Explanation: {result['explanation']}")
            print(f"Natural Writing: {result.get('is_natural_writing', False)}")
            print(f"Information Density: {result.get('density_analysis', {}).get('information_density_score', 0):.1f}")

            print(f"\nDetailed Indicators:")
            for indicator, score in result["slop_indicators"].items():
                if score > 0:
                    print(f"  {indicator}: {score:.1f}")

            # Show density analysis details
            density = result.get("density_analysis", {})
            if density:
                print(f"\nInformation Density Analysis:")
                print(f"  Filler Words: {density.get('filler_words', 0)}")
                print(f"  Redundant Phrases: {density.get('redundant_phrases', 0)}")
                print(f"  Avg Sentence Length: {density.get('avg_sentence_length', 0):.1f}")
                print(f"  Information Ratio: {density.get('information_ratio', 0):.2f}")

        # Exit with appropriate code
        sys.exit(1 if result["is_slop"] else 0)

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
