#!/usr/bin/env python3
"""
Simple validation test for AI slop detection without complex dependencies.
"""

import json
import time

from sloplint.combine import combine_scores, normalize_scores
from sloplint.feature_extractor import FeatureExtractor


def test_basic_slop_detection():
    """Test basic slop detection functionality."""
    print("🧪 Testing basic AI slop detection...")

    # Initialize feature extractor
    extractor = FeatureExtractor()

    # Test cases with expected outcomes
    test_cases = [
        {
            "name": "Clean Human Text",
            "text": "I woke up this morning feeling groggy. The coffee machine was broken again, so I had to make instant coffee. It tasted terrible, but I needed the caffeine to function.",
            "expected_range": "low",
            "domain": "personal",
        },
        {
            "name": "AI-Generated Business Email",
            "text": "I hope this email finds you well. I wanted to reach out to discuss the upcoming project deadline and ensure we're all aligned on the deliverables. Please let me know if you have any questions or concerns.",
            "expected_range": "medium",
            "domain": "business",
        },
        {
            "name": "High AI Content",
            "text": "I understand your concern and I'm here to help. Let me provide you with a comprehensive solution that addresses all aspects of your inquiry. This approach has been proven effective in similar situations.",
            "expected_range": "high",
            "domain": "ai_response",
        },
        {
            "name": "Repetitive AI Text",
            "text": "I understand your concern. I understand your concern. I understand your concern. Let me help you with this matter. Let me help you with this matter. Let me help you with this matter.",
            "expected_range": "very_high",
            "domain": "repetitive",
        },
        {
            "name": "Technical Documentation",
            "text": "To install the package, run 'pip install package-name' in your terminal. Make sure you have Python 3.7 or higher installed on your system.",
            "expected_range": "low",
            "domain": "technical",
        },
    ]

    results = []

    for i, test_case in enumerate(test_cases):
        print(f"Testing {i + 1}/{len(test_cases)}: {test_case['name']}")

        start_time = time.time()

        try:
            # Extract features
            features = extractor.extract_all_features(test_case["text"])

            # Normalize scores
            normalized = normalize_scores(features, test_case["domain"])

            # Combine scores
            slop_score, confidence = combine_scores(normalized, test_case["domain"])

            processing_time = time.time() - start_time

            result = {
                "name": test_case["name"],
                "text_length": len(test_case["text"]),
                "slop_score": slop_score,
                "confidence": confidence,
                "processing_time": processing_time,
                "expected_range": test_case["expected_range"],
                "success": True,
                "features_count": len(features),
            }

            # Determine if result matches expectation
            if test_case["expected_range"] == "low":
                result["correct"] = slop_score < 0.4
            elif test_case["expected_range"] == "medium":
                result["correct"] = 0.2 <= slop_score <= 0.7
            elif test_case["expected_range"] == "high":
                result["correct"] = slop_score > 0.5
            elif test_case["expected_range"] == "very_high":
                result["correct"] = slop_score > 0.7
            else:
                result["correct"] = True  # Unknown range

            print(
                f"  ✅ Slop Score: {slop_score:.3f}, Confidence: {confidence:.3f}, Time: {processing_time:.3f}s"
            )

        except Exception as e:
            processing_time = time.time() - start_time
            result = {
                "name": test_case["name"],
                "text_length": len(test_case["text"]),
                "slop_score": None,
                "confidence": None,
                "processing_time": processing_time,
                "expected_range": test_case["expected_range"],
                "success": False,
                "error": str(e),
                "correct": False,
            }
            print(f"  ❌ Error: {e}")

        results.append(result)

    # Calculate summary statistics
    successful_tests = [r for r in results if r["success"]]
    correct_predictions = [r for r in successful_tests if r["correct"]]

    print("\n📊 Test Results Summary:")
    print(f"  Total Tests: {len(results)}")
    print(f"  Successful: {len(successful_tests)}")
    print(f"  Failed: {len(results) - len(successful_tests)}")
    print(f"  Correct Predictions: {len(correct_predictions)}")
    print(
        f"  Accuracy: {len(correct_predictions) / len(successful_tests) * 100:.1f}%"
        if successful_tests
        else "  Accuracy: N/A"
    )

    if successful_tests:
        avg_processing_time = sum(r["processing_time"] for r in successful_tests) / len(
            successful_tests
        )
        avg_slop_score = sum(r["slop_score"] for r in successful_tests) / len(
            successful_tests
        )
        avg_confidence = sum(r["confidence"] for r in successful_tests) / len(
            successful_tests
        )

        print(f"  Average Processing Time: {avg_processing_time:.3f}s")
        print(f"  Average Slop Score: {avg_slop_score:.3f}")
        print(f"  Average Confidence: {avg_confidence:.3f}")

    # Save results
    output_data = {
        "test_metadata": {
            "timestamp": time.time(),
            "total_tests": len(results),
            "successful_tests": len(successful_tests),
            "accuracy": (
                len(correct_predictions) / len(successful_tests)
                if successful_tests
                else 0
            ),
        },
        "results": results,
    }

    output_path = "tests/simple_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Results saved to: {output_path}")

    return results


def test_dataset_sample():
    """Test a sample from the full dataset."""
    print("\n🎯 Testing sample from full dataset...")

    # Load a sample of the dataset
    with open("tests/test_dataset.json", encoding="utf-8") as f:
        full_dataset = json.load(f)

    # Take first 10 texts for quick testing
    sample_dataset = full_dataset[:10]

    extractor = FeatureExtractor()
    results = []

    for i, item in enumerate(sample_dataset):
        print(f"Testing {i + 1}/{len(sample_dataset)}: {item['doc_id']}")

        start_time = time.time()

        try:
            features = extractor.extract_all_features(item["text"])
            normalized = normalize_scores(features, item["domain"])
            slop_score, confidence = combine_scores(normalized, item["domain"])
            processing_time = time.time() - start_time

            # Determine if prediction matches expectation
            expected_range = item["expected_slop_range"]
            if expected_range < 0.3:
                correct = slop_score < 0.4
            elif expected_range < 0.6:
                correct = 0.2 <= slop_score <= 0.7
            else:
                correct = slop_score > 0.5

            result = {
                "doc_id": item["doc_id"],
                "category": item["category"],
                "domain": item["domain"],
                "text_length": len(item["text"]),
                "slop_score": slop_score,
                "confidence": confidence,
                "processing_time": processing_time,
                "expected_slop_range": expected_range,
                "correct": correct,
                "success": True,
            }

            print(
                f"  ✅ Score: {slop_score:.3f}, Expected: {expected_range:.1f}, Correct: {correct}"
            )

        except Exception as e:
            processing_time = time.time() - start_time
            result = {
                "doc_id": item["doc_id"],
                "category": item["category"],
                "domain": item["domain"],
                "text_length": len(item["text"]),
                "slop_score": None,
                "confidence": None,
                "processing_time": processing_time,
                "expected_slop_range": expected_range,
                "correct": False,
                "success": False,
                "error": str(e),
            }
            print(f"  ❌ Error: {e}")

        results.append(result)

    # Calculate accuracy
    successful_results = [r for r in results if r["success"]]
    correct_predictions = [r for r in successful_results if r["correct"]]

    print("\n📊 Dataset Sample Results:")
    print(f"  Total Tests: {len(results)}")
    print(f"  Successful: {len(successful_results)}")
    print(f"  Correct Predictions: {len(correct_predictions)}")
    print(
        f"  Accuracy: {len(correct_predictions) / len(successful_results) * 100:.1f}%"
        if successful_results
        else "  Accuracy: N/A"
    )

    return results


if __name__ == "__main__":
    # Run basic tests
    basic_results = test_basic_slop_detection()

    # Run dataset sample test
    dataset_results = test_dataset_sample()

    print("\n✅ All tests completed!")
