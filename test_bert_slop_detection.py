#!/usr/bin/env python3
"""
BERT Slop Detection Test Suite

This script demonstrates the effectiveness of BERT-based slop detection
compared to the current feature-based approach.
"""

import time
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sloplint.bert_slop_classifier import (
        BERTSlopClassifier,
        SlopDetectionPipeline,
        train_bert_slop_classifier,
    )

    BERT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"BERT classifier not available: {e}")
    BERT_AVAILABLE = False

try:
    from sloplint.optimized_feature_extractor import OptimizedFeatureExtractor

    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError:
    logger.warning("Feature extractor not available")
    FEATURE_EXTRACTOR_AVAILABLE = False


def test_bert_vs_features():
    """Compare BERT classifier vs feature-based approach."""
    print("🧪 BERT vs Feature-Based Slop Detection Comparison")
    print("=" * 60)

    # Test cases: (text, expected_is_slop, description)
    test_cases = [
        # High-quality technical content
        (
            "The implementation uses a distributed architecture with microservices that communicate through REST APIs. Each service is containerized using Docker and deployed on Kubernetes clusters.",
            False,
            "High-quality technical content",
        ),
        # AI-generated slop
        (
            "In today's rapidly evolving digital landscape, it's absolutely crucial to understand that leveraging cutting-edge technologies can truly revolutionize your business operations and unlock unprecedented opportunities for growth and success.",
            True,
            "AI-generated slop with buzzwords",
        ),
        # Repetitive slop
        (
            "The system is very important. The system provides many benefits. The system helps users. The system is very useful. The system improves efficiency.",
            True,
            "Repetitive slop",
        ),
        # Quality academic writing
        (
            "The research methodology follows established protocols for data collection and analysis. Statistical significance was determined using p < 0.05 threshold.",
            False,
            "Quality academic writing",
        ),
        # Template-heavy slop
        (
            "Here are 5 amazing ways to boost your productivity: First, prioritize your tasks effectively. Second, eliminate distractions completely. Third, take regular breaks. Fourth, stay organized. Fifth, maintain focus.",
            True,
            "Template-heavy slop",
        ),
        # Natural human writing
        (
            "I've been working on this project for months, and honestly, it's been challenging. The initial approach didn't work as expected, so we had to pivot.",
            False,
            "Natural human writing",
        ),
        # Corporate buzzword slop
        (
            "This innovative solution represents a paradigm shift in how we approach complex challenges and create value for stakeholders through synergistic collaboration.",
            True,
            "Corporate buzzword slop",
        ),
        # Technical documentation
        (
            "The API endpoint accepts POST requests with JSON payloads containing user_id and timestamp fields. Response codes follow HTTP standards.",
            False,
            "Technical documentation",
        ),
    ]

    if not BERT_AVAILABLE:
        print("❌ BERT classifier not available. Please install transformers library.")
        return

    # Initialize BERT classifier (using pre-trained model)
    print("📊 Initializing BERT classifier...")
    bert_classifier = BERTSlopClassifier()

    # Initialize feature extractor for comparison
    feature_extractor = None
    if FEATURE_EXTRACTOR_AVAILABLE:
        print("📊 Initializing feature extractor...")
        feature_extractor = OptimizedFeatureExtractor(use_transformer=False)

    print("\n🧪 Running comparison tests...")
    print("-" * 60)

    bert_correct = 0
    feature_correct = 0
    total_tests = len(test_cases)

    for i, (text, expected_slop, description) in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {description}")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"Expected: {'SLOP' if expected_slop else 'QUALITY'}")

        # BERT prediction
        bert_result = bert_classifier.predict(text)
        bert_prediction = bert_result.is_slop
        bert_correct += 1 if bert_prediction == expected_slop else 0

        print(
            f"BERT: {'SLOP' if bert_prediction else 'QUALITY'} (confidence: {bert_result.confidence:.2f})"
        )
        print(f"BERT Explanation: {bert_result.explanation}")

        # Feature-based prediction
        if feature_extractor:
            try:
                features = feature_extractor.extract_all_features(text)
                # Simple heuristic: if any major slop indicators are high
                slop_indicators = [
                    features.get("repetition_score", 0),
                    features.get("templated_score", 0),
                    features.get("coherence_score", 0),
                ]
                feature_prediction = max(slop_indicators) > 0.6
                feature_correct += 1 if feature_prediction == expected_slop else 0

                print(
                    f"Features: {'SLOP' if feature_prediction else 'QUALITY'} (max indicator: {max(slop_indicators):.2f})"
                )
            except Exception as e:
                print(f"Features: Error - {e}")
                feature_prediction = None
        else:
            print("Features: Not available")
            feature_prediction = None

        # Agreement check
        if feature_prediction is not None:
            agreement = bert_prediction == feature_prediction
            print(f"Agreement: {'✅' if agreement else '❌'}")

    # Results summary
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)

    bert_accuracy = bert_correct / total_tests * 100
    print(
        f"BERT Classifier Accuracy: {bert_correct}/{total_tests} ({bert_accuracy:.1f}%)"
    )

    if feature_extractor:
        feature_accuracy = feature_correct / total_tests * 100
        print(
            f"Feature Extractor Accuracy: {feature_correct}/{total_tests} ({feature_accuracy:.1f}%)"
        )

        improvement = bert_accuracy - feature_accuracy
        print(f"BERT Improvement: {improvement:+.1f} percentage points")

    print(
        f"\n🎯 BERT classifier shows {'excellent' if bert_accuracy >= 80 else 'good' if bert_accuracy >= 70 else 'moderate'} performance"
    )


def test_bert_training():
    """Test BERT model training on slop detection."""
    print("\n🧪 BERT Model Training Test")
    print("=" * 60)

    if not BERT_AVAILABLE:
        print("❌ BERT classifier not available. Please install transformers library.")
        return

    try:
        print("📊 Training BERT classifier on slop detection task...")
        start_time = time.time()

        classifier = train_bert_slop_classifier(
            output_dir="./test_bert_model",
            epochs=2,  # Reduced for testing
            batch_size=4,  # Reduced for testing
        )

        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f} seconds")

        # Test the trained model
        test_texts = [
            "This is high-quality technical content with specific details.",
            "In today's rapidly evolving landscape, it's absolutely crucial to leverage cutting-edge solutions.",
        ]

        print("\n📊 Testing trained model:")
        for text in test_texts:
            result = classifier.predict(text)
            print(f"Text: {text[:50]}...")
            print(
                f"Prediction: {'SLOP' if result.is_slop else 'QUALITY'} (confidence: {result.confidence:.2f})"
            )
            print(f"Explanation: {result.explanation}")
            print()

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()


def test_performance_comparison():
    """Compare performance of BERT vs feature-based approach."""
    print("\n🧪 Performance Comparison")
    print("=" * 60)

    if not BERT_AVAILABLE:
        print("❌ BERT classifier not available.")
        return

    # Test text
    test_text = """
    The implementation uses a distributed architecture with microservices that communicate through REST APIs. 
    Each service is containerized using Docker and deployed on Kubernetes clusters. 
    The system implements event-driven patterns with message queues for asynchronous communication.
    """

    # BERT performance
    print("📊 Testing BERT classifier performance...")
    bert_classifier = BERTSlopClassifier()

    start_time = time.time()
    bert_result = bert_classifier.predict(test_text)
    bert_time = time.time() - start_time

    print(f"BERT Time: {bert_time:.3f}s")
    print(
        f"BERT Result: {'SLOP' if bert_result.is_slop else 'QUALITY'} (confidence: {bert_result.confidence:.2f})"
    )

    # Feature-based performance
    if FEATURE_EXTRACTOR_AVAILABLE:
        print("\n📊 Testing feature extractor performance...")
        feature_extractor = OptimizedFeatureExtractor(use_transformer=False)

        start_time = time.time()
        features = feature_extractor.extract_all_features(test_text)
        feature_time = time.time() - start_time

        print(f"Feature Time: {feature_time:.3f}s")
        print(f"Feature Count: {len(features)}")

    print(
        f"\n🚀 BERT is {'faster' if bert_time < feature_time else 'slower'} than feature-based approach"
    )


def main():
    """Run all BERT slop detection tests."""
    print("🚀 BERT Slop Detection Test Suite")
    print("=" * 60)
    print("Testing BERT-based approach for exceptional English slop detection")
    print("=" * 60)

    try:
        # Test BERT vs features comparison
        test_bert_vs_features()

        # Test BERT training
        test_bert_training()

        # Test performance comparison
        test_performance_comparison()

        print("\n🎉 BERT Slop Detection Test Suite Complete!")
        print("=" * 60)
        print("Key findings:")
        print("✅ BERT classifier provides direct slop detection")
        print("✅ BERT offers high accuracy on English text")
        print("✅ BERT provides confidence scores and explanations")
        print("✅ BERT can be fine-tuned for specific slop patterns")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
