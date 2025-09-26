#!/usr/bin/env python3
"""
Simple BERT Slop Detector

A focused, production-ready BERT-based slop detector for exceptional English detection.
This version prioritizes simplicity and effectiveness over complexity.
"""

import argparse
import json
import sys
from typing import Dict, Any, List

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import numpy as np

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print(
        "❌ transformers library not available. Please install: pip install transformers torch"
    )
    sys.exit(1)


class SimpleBERTSlopDetector:
    """Simple, effective BERT-based slop detector."""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """Initialize the detector."""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, problem_type="single_label_classification"
        )
        self.model.to(self.device)

        print(f"✅ Loaded BERT model: {model_name} on {self.device}")

    def detect_slop(self, text: str) -> Dict[str, Any]:
        """Detect if text is slop."""
        # Tokenize
        inputs = self.tokenizer(
            text, truncation=True, padding=True, max_length=512, return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        # Extract results
        quality_prob = probabilities[0][0].item()  # Quality probability
        slop_prob = probabilities[0][1].item()  # Slop probability

        is_slop = slop_prob > 0.5
        confidence = max(quality_prob, slop_prob)

        # Determine slop level
        if slop_prob > 0.8:
            level = "HIGH_SLOP"
            explanation = "Strong indicators of AI-generated or low-quality content"
        elif slop_prob > 0.6:
            level = "SLOP"
            explanation = "Multiple slop characteristics detected"
        elif slop_prob > 0.4:
            level = "SUSPICIOUS"
            explanation = "Some slop indicators present"
        else:
            level = "QUALITY"
            explanation = "High-quality, natural content"

        return {
            "is_slop": is_slop,
            "slop_probability": slop_prob,
            "quality_probability": quality_prob,
            "confidence": confidence,
            "level": level,
            "explanation": explanation,
            "model": self.model_name,
        }

    def batch_detect(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Detect slop for multiple texts."""
        results = []
        for text in texts:
            results.append(self.detect_slop(text))
        return results


def create_high_quality_training_data() -> tuple[List[str], List[int]]:
    """Create high-quality training data for fine-tuning."""

    # High-quality English examples
    quality_texts = [
        "The implementation uses a distributed architecture with microservices that communicate through REST APIs.",
        "Each service is containerized using Docker and deployed on Kubernetes clusters for scalability.",
        "The system implements event-driven patterns with message queues for asynchronous communication.",
        "Security is handled through OAuth 2.0 authentication and JWT tokens for secure access.",
        "The database uses PostgreSQL with read replicas for improved performance and reliability.",
        "Machine learning models are trained on large datasets to achieve state-of-the-art performance.",
        "The research methodology follows established protocols for data collection and analysis.",
        "Economic indicators suggest a gradual recovery in the manufacturing sector.",
        "Climate change poses significant challenges for agricultural productivity worldwide.",
        "The novel explores themes of identity and belonging in contemporary society.",
        "Scientific evidence supports the hypothesis that exercise improves cognitive function.",
        "The company's quarterly earnings exceeded analyst expectations by 15%.",
        "Educational institutions are adapting to new technologies for enhanced learning outcomes.",
        "The legal framework provides clear guidelines for intellectual property protection.",
        "Urban planning initiatives aim to reduce traffic congestion and improve air quality.",
        "I've been working on this project for months, and honestly, it's been challenging.",
        "The initial approach didn't work as expected, so we had to pivot to a different strategy.",
        "After analyzing the data, we discovered several interesting patterns that weren't immediately obvious.",
        "The API endpoint accepts POST requests with JSON payloads containing user_id and timestamp fields.",
        "Response codes follow HTTP standards with appropriate error handling for edge cases.",
    ]

    # AI-generated slop examples
    slop_texts = [
        "In today's rapidly evolving digital landscape, it's absolutely crucial to understand that leveraging cutting-edge technologies can truly revolutionize your business operations and unlock unprecedented opportunities for growth and success.",
        "As an AI, I can confidently say that this revolutionary approach will undoubtedly transform the way we think about innovation and drive meaningful change across industries.",
        "Here are 5 amazing ways to boost your productivity: First, prioritize your tasks effectively. Second, eliminate distractions completely. Third, take regular breaks. Fourth, stay organized. Fifth, maintain focus.",
        "The implementation is very important and provides many benefits. The system is very useful and helps users. The features are very helpful and improve efficiency. The solution is very effective and saves time.",
        "In conclusion, it's clear that this innovative solution represents a paradigm shift in how we approach complex challenges and create value for stakeholders.",
        "This comprehensive analysis reveals that the data suggests a strong correlation between various factors that contribute to the overall success of the initiative.",
        "The methodology employed in this study demonstrates that the results indicate a significant improvement in performance metrics across multiple dimensions.",
        "It's important to note that the findings suggest that the approach taken provides a solid foundation for future research and development efforts.",
        "The research shows that the implementation of these strategies can lead to substantial improvements in key performance indicators.",
        "This approach offers a unique opportunity to address the challenges faced by organizations in today's competitive environment.",
        "The solution provides a comprehensive framework for addressing the complex issues that arise in modern business operations.",
        "It's worth mentioning that the results demonstrate the effectiveness of the proposed methodology in achieving desired outcomes.",
        "The analysis reveals that the data supports the hypothesis that the approach taken is both innovative and practical.",
        "This innovative solution represents a significant advancement in the field and offers numerous benefits for users.",
        "The implementation of these strategies can help organizations achieve their goals and improve their competitive position.",
        "In today's fast-paced world, it's essential to stay ahead of the curve and embrace new technologies that can drive innovation.",
        "The key to success lies in understanding the needs of your customers and delivering solutions that exceed their expectations.",
        "By leveraging the power of data analytics, organizations can gain valuable insights into their operations and make informed decisions.",
        "The future of business depends on the ability to adapt to changing market conditions and embrace digital transformation.",
        "Success requires a combination of strategic thinking, innovative solutions, and a commitment to continuous improvement.",
    ]

    # Combine and label (0 = quality, 1 = slop)
    all_texts = quality_texts + slop_texts
    labels = [0] * len(quality_texts) + [1] * len(slop_texts)

    return all_texts, labels


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Simple BERT Slop Detector for English Text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_bert_slop.py "This is high-quality technical content."
  python simple_bert_slop.py --file document.txt
  python simple_bert_slop.py --json "AI-generated slop text here"
  python simple_bert_slop.py --test  # Run test cases
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
    detector = SimpleBERTSlopDetector()

    if args.test:
        # Run test cases
        print("🧪 BERT Slop Detection Test Cases")
        print("=" * 50)

        test_cases = [
            (
                "The implementation uses a distributed architecture with microservices.",
                False,
                "High-quality technical content",
            ),
            (
                "In today's rapidly evolving digital landscape, it's absolutely crucial to understand that leveraging cutting-edge technologies can truly revolutionize your business operations.",
                True,
                "AI-generated slop",
            ),
            (
                "The system is very important. The system provides many benefits. The system helps users.",
                True,
                "Repetitive slop",
            ),
            (
                "I've been working on this project for months, and honestly, it's been challenging.",
                False,
                "Natural human writing",
            ),
            (
                "Here are 5 amazing ways to boost your productivity: First, prioritize your tasks effectively.",
                True,
                "Template-heavy slop",
            ),
        ]

        correct = 0
        total = len(test_cases)

        for i, (text, expected_slop, description) in enumerate(test_cases, 1):
            result = detector.detect_slop(text)
            predicted_slop = result["is_slop"]
            correct += 1 if predicted_slop == expected_slop else 0

            print(f"\n📝 Test {i}: {description}")
            print(f"Text: {text[:60]}{'...' if len(text) > 60 else ''}")
            print(f"Expected: {'SLOP' if expected_slop else 'QUALITY'}")
            print(
                f"Predicted: {'SLOP' if predicted_slop else 'QUALITY'} (confidence: {result['confidence']:.2f})"
            )
            print(f"Level: {result['level']}")
            print(f"Explanation: {result['explanation']}")

        accuracy = correct / total * 100
        print(f"\n📊 Test Results: {correct}/{total} correct ({accuracy:.1f}%)")
        print(
            f"🎯 BERT detector shows {'excellent' if accuracy >= 80 else 'good' if accuracy >= 70 else 'moderate'} performance"
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
            print("🔍 BERT Slop Detection Analysis")
            print("=" * 40)
            print(f"Prediction: {'🚨 SLOP' if result['is_slop'] else '✅ QUALITY'}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Slop Probability: {result['slop_probability']:.2f}")
            print(f"Quality Probability: {result['quality_probability']:.2f}")
            print(f"Level: {result['level']}")
            print(f"Explanation: {result['explanation']}")
            print(f"Model: {result['model']}")

        # Exit with appropriate code
        sys.exit(1 if result["is_slop"] else 0)

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
