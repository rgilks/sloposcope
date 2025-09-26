#!/usr/bin/env python3
"""
Manual validation test for AI slop detection - testing individual components.
"""

import json
import time

from sloplint.combine import combine_scores, normalize_scores
from sloplint.feature_extractor import FeatureExtractor


def test_feature_extraction():
    """Test feature extraction on sample texts."""
    print("🔍 Testing feature extraction...")

    extractor = FeatureExtractor()

    test_texts = [
        "I woke up this morning feeling groggy. The coffee machine was broken again.",
        "I hope this email finds you well. I wanted to reach out to discuss the upcoming project deadline.",
        "I understand your concern and I'm here to help. Let me provide you with a comprehensive solution.",
    ]

    for i, text in enumerate(test_texts):
        print(f"\nTesting text {i + 1}: {text[:50]}...")

        try:
            features = extractor.extract_all_features(text)
            print(f"✅ Successfully extracted {len(features)} feature categories")

            # Print feature structure
            for feature_name, feature_data in features.items():
                if isinstance(feature_data, dict):
                    print(f"  {feature_name}: {list(feature_data.keys())}")
                else:
                    print(f"  {feature_name}: {type(feature_data)}")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()


def test_individual_features():
    """Test individual feature extraction methods."""
    print("\n🧪 Testing individual feature extraction...")

    extractor = FeatureExtractor()
    text = "This is a test sentence with some content for analysis."

    try:
        # Test basic text processing
        sentences = extractor._split_sentences(text)
        tokens = extractor._tokenize(text)

        print(f"✅ Sentences: {len(sentences)}")
        print(f"✅ Tokens: {len(tokens)}")

        # Test individual features
        from sloplint.features.density import extract_density_features
        from sloplint.features.repetition import extract_repetition_features
        from sloplint.features.verbosity import extract_verbosity_features

        density_features = extract_density_features(text, sentences, tokens)
        print(f"✅ Density features: {density_features}")

        repetition_features = extract_repetition_features(text, sentences, tokens)
        print(f"✅ Repetition features: {repetition_features}")

        verbosity_features = extract_verbosity_features(text, sentences, tokens)
        print(f"✅ Verbosity features: {verbosity_features}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


def test_score_calculation():
    """Test score calculation with manual feature data."""
    print("\n📊 Testing score calculation...")

    # Create sample feature data in the expected format
    sample_features = {
        "density": {"combined_density": 0.5, "perplexity": 25.0, "value": 0.5},
        "repetition": {
            "word_repetition": 0.1,
            "phrase_repetition": 0.05,
            "value": 0.15,
        },
        "verbosity": {"avg_sentence_length": 15.0, "word_count": 100, "value": 0.3},
        "coherence": {"semantic_coherence": 0.7, "value": 0.7},
        "tone": {"formality": 0.6, "value": 0.6},
    }

    try:
        # Test normalization
        normalized = normalize_scores(sample_features, "general")
        print(f"✅ Normalized features: {list(normalized.keys())}")

        # Test score combination
        slop_score, confidence = combine_scores(normalized, "general")
        print(f"✅ Slop Score: {slop_score:.3f}")
        print(f"✅ Confidence: {confidence:.3f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


def test_text_samples():
    """Test with simple text samples."""
    print("\n📝 Testing with text samples...")

    samples = [
        {
            "name": "Human Text",
            "text": "I had a terrible day at work today. My boss was in a bad mood and everything went wrong.",
            "expected": "low",
        },
        {
            "name": "AI Text",
            "text": "I understand your concern and I'm here to help. Let me provide you with a comprehensive solution that addresses all aspects of your inquiry.",
            "expected": "high",
        },
        {
            "name": "Mixed Text",
            "text": "The weather has been unpredictable lately. Yesterday it was sunny and warm, but today it's cold and rainy.",
            "expected": "medium",
        },
    ]

    extractor = FeatureExtractor()

    for sample in samples:
        print(f"\nTesting: {sample['name']}")
        print(f"Text: {sample['text']}")

        try:
            # Extract features
            features = extractor.extract_all_features(sample["text"])
            print(f"✅ Extracted {len(features)} feature categories")

            # Try to normalize and combine scores
            try:
                normalized = normalize_scores(features, "general")
                slop_score, confidence = combine_scores(normalized, "general")
                print(f"✅ Slop Score: {slop_score:.3f}, Confidence: {confidence:.3f}")

                # Determine if result makes sense
                if sample["expected"] == "low" and slop_score < 0.4:
                    print("✅ Result matches expectation (low slop)")
                elif sample["expected"] == "high" and slop_score > 0.6:
                    print("✅ Result matches expectation (high slop)")
                elif sample["expected"] == "medium" and 0.3 <= slop_score <= 0.7:
                    print("✅ Result matches expectation (medium slop)")
                else:
                    print(
                        f"⚠️  Result doesn't match expectation (expected {sample['expected']}, got {slop_score:.3f})"
                    )

            except Exception as e:
                print(f"❌ Score calculation error: {e}")

        except Exception as e:
            print(f"❌ Feature extraction error: {e}")


def run_validation_tests():
    """Run all validation tests."""
    print("🚀 Starting AI Slop Detection Validation Tests")
    print("=" * 60)

    # Test individual components
    test_feature_extraction()
    test_individual_features()
    test_score_calculation()
    test_text_samples()

    print("\n" + "=" * 60)
    print("✅ Validation tests completed!")

    # Save a simple report
    report = {
        "timestamp": time.time(),
        "tests_run": [
            "feature_extraction",
            "individual_features",
            "score_calculation",
            "text_samples",
        ],
        "status": "completed",
    }

    with open("tests/validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("📁 Report saved to: tests/validation_report.json")


if __name__ == "__main__":
    run_validation_tests()
