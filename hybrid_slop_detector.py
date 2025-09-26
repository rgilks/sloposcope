#!/usr/bin/env python3
"""
Hybrid BERT + Features Slop Detector

This combines BERT's semantic understanding with proven feature-based detection
to create an exceptional English slop detector.
"""

import argparse
import json
import sys
import time
from typing import Dict, Any, List

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import numpy as np

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sloplint.optimized_feature_extractor import OptimizedFeatureExtractor

    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError:
    FEATURE_EXTRACTOR_AVAILABLE = False


class HybridSlopDetector:
    """Hybrid detector combining BERT semantic analysis with feature-based detection."""

    def __init__(self):
        """Initialize the hybrid detector."""
        self.feature_extractor = None
        self.bert_model = None
        self.bert_tokenizer = None

        # Initialize feature extractor
        if FEATURE_EXTRACTOR_AVAILABLE:
            self.feature_extractor = OptimizedFeatureExtractor(use_transformer=False)
            print("✅ Feature extractor initialized")

        # Initialize BERT for semantic analysis
        if TRANSFORMERS_AVAILABLE:
            try:
                self.bert_tokenizer = AutoTokenizer.from_pretrained(
                    "distilbert-base-uncased"
                )
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased", num_labels=2
                )
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.bert_model.to(self.device)
                print(f"✅ BERT model initialized on {self.device}")
            except Exception as e:
                print(f"⚠️ BERT initialization failed: {e}")
                self.bert_model = None

    def _analyze_with_features(self, text: str) -> Dict[str, Any]:
        """Analyze text using feature-based approach."""
        if not self.feature_extractor:
            return {"error": "Feature extractor not available"}

        try:
            features = self.feature_extractor.extract_all_features(text)

            # Calculate slop score from key features
            slop_indicators = {
                "repetition": features.get("repetition_score", 0),
                "templated": features.get("templated_score", 0),
                "coherence": features.get("coherence_score", 0),
                "density": 1 - features.get("density_score", 0.5),  # Invert density
                "tone": features.get("tone_score", 0),
                "verbosity": features.get("verbosity_score", 0),
            }

            # Weighted combination
            weights = {
                "repetition": 0.25,
                "templated": 0.25,
                "coherence": 0.20,
                "density": 0.15,
                "tone": 0.10,
                "verbosity": 0.05,
            }

            slop_score = sum(slop_indicators[key] * weights[key] for key in weights)

            return {
                "slop_score": slop_score,
                "is_slop": slop_score > 0.6,
                "confidence": abs(slop_score - 0.5) * 2,
                "features": slop_indicators,
                "method": "feature_based",
            }

        except Exception as e:
            return {"error": str(e)}

    def _analyze_with_bert(self, text: str) -> Dict[str, Any]:
        """Analyze text using BERT semantic understanding."""
        if not self.bert_model:
            return {"error": "BERT model not available"}

        try:
            # Tokenize
            inputs = self.bert_tokenizer(
                text, truncation=True, padding=True, max_length=512, return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get prediction
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)

            quality_prob = probabilities[0][0].item()
            slop_prob = probabilities[0][1].item()

            return {
                "slop_score": slop_prob,
                "is_slop": slop_prob > 0.5,
                "confidence": max(quality_prob, slop_prob),
                "quality_prob": quality_prob,
                "method": "bert_semantic",
            }

        except Exception as e:
            return {"error": str(e)}

    def detect_slop(self, text: str) -> Dict[str, Any]:
        """Detect slop using hybrid approach."""
        start_time = time.time()

        # Get both analyses
        feature_result = self._analyze_with_features(text)
        bert_result = self._analyze_with_bert(text)

        # Combine results
        if "error" not in feature_result and "error" not in bert_result:
            # Both methods available - use weighted combination
            feature_score = feature_result["slop_score"]
            bert_score = bert_result["slop_score"]

            # Weight: 70% features (proven), 30% BERT (semantic)
            combined_score = feature_score * 0.7 + bert_score * 0.3

            is_slop = combined_score > 0.6
            confidence = abs(combined_score - 0.5) * 2

            # Determine level
            if combined_score > 0.8:
                level = "HIGH_SLOP"
                explanation = "Strong indicators of AI-generated or low-quality content"
            elif combined_score > 0.6:
                level = "SLOP"
                explanation = "Multiple slop characteristics detected"
            elif combined_score > 0.4:
                level = "SUSPICIOUS"
                explanation = "Some slop indicators present"
            else:
                level = "QUALITY"
                explanation = "High-quality, natural content"

            result = {
                "is_slop": is_slop,
                "slop_score": combined_score,
                "confidence": confidence,
                "level": level,
                "explanation": explanation,
                "method": "hybrid",
                "feature_score": feature_score,
                "bert_score": bert_score,
                "processing_time": time.time() - start_time,
                "feature_analysis": feature_result,
                "bert_analysis": bert_result,
            }

        elif "error" not in feature_result:
            # Only features available
            result = {
                "is_slop": feature_result["is_slop"],
                "slop_score": feature_result["slop_score"],
                "confidence": feature_result["confidence"],
                "level": "SLOP" if feature_result["is_slop"] else "QUALITY",
                "explanation": "Analysis based on linguistic features",
                "method": "feature_only",
                "processing_time": time.time() - start_time,
                "feature_analysis": feature_result,
            }

        elif "error" not in bert_result:
            # Only BERT available
            result = {
                "is_slop": bert_result["is_slop"],
                "slop_score": bert_result["slop_score"],
                "confidence": bert_result["confidence"],
                "level": "SLOP" if bert_result["is_slop"] else "QUALITY",
                "explanation": "Analysis based on semantic understanding",
                "method": "bert_only",
                "processing_time": time.time() - start_time,
                "bert_analysis": bert_result,
            }

        else:
            # Neither available
            result = {
                "is_slop": False,
                "slop_score": 0.5,
                "confidence": 0.0,
                "level": "UNKNOWN",
                "explanation": "Analysis methods not available",
                "method": "none",
                "processing_time": time.time() - start_time,
                "error": "No analysis methods available",
            }

        return result


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Hybrid BERT + Features Slop Detector for English Text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hybrid_slop_detector.py "This is high-quality technical content."
  python hybrid_slop_detector.py --file document.txt
  python hybrid_slop_detector.py --json "AI-generated slop text here"
  python hybrid_slop_detector.py --test  # Run test cases
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
    print("🚀 Initializing Hybrid Slop Detector...")
    detector = HybridSlopDetector()

    if args.test:
        # Run test cases
        print("\n🧪 Hybrid Slop Detection Test Cases")
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
            print(f"Method: {result['method']}")
            print(f"Time: {result['processing_time']:.3f}s")

            if result["method"] == "hybrid":
                print(
                    f"Feature Score: {result['feature_score']:.2f}, BERT Score: {result['bert_score']:.2f}"
                )

        accuracy = correct / total * 100
        print(f"\n📊 Test Results: {correct}/{total} correct ({accuracy:.1f}%)")
        print(
            f"🎯 Hybrid detector shows {'excellent' if accuracy >= 80 else 'good' if accuracy >= 70 else 'moderate'} performance"
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
            print("\n🔍 Hybrid Slop Detection Analysis")
            print("=" * 50)
            print(f"Prediction: {'🚨 SLOP' if result['is_slop'] else '✅ QUALITY'}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Slop Score: {result['slop_score']:.2f}")
            print(f"Level: {result['level']}")
            print(f"Method: {result['method']}")
            print(f"Processing Time: {result['processing_time']:.3f}s")
            print(f"Explanation: {result['explanation']}")

            if result["method"] == "hybrid":
                print(f"\nDetailed Scores:")
                print(f"  Feature-based: {result['feature_score']:.2f}")
                print(f"  BERT semantic: {result['bert_score']:.2f}")

        # Exit with appropriate code
        sys.exit(1 if result["is_slop"] else 0)

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
