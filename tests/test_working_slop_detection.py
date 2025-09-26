#!/usr/bin/env python3
"""
Working test for AI slop detection using the actual feature structure.
"""

import json
import time

from sloplint.feature_extractor import FeatureExtractor


def calculate_slop_score_from_features(features):
    """Calculate slop score from features using the actual structure."""

    # Extract key metrics from features
    scores = []

    # Density score
    if "density" in features and "combined_density" in features["density"]:
        scores.append(features["density"]["combined_density"])

    # Repetition score
    if "repetition" in features and "overall_repetition" in features["repetition"]:
        scores.append(features["repetition"]["overall_repetition"])

    # Templated score
    if "templated" in features and "templated_score" in features["templated"]:
        scores.append(features["templated"]["templated_score"])

    # Coherence score
    if "coherence" in features and "coherence_score" in features["coherence"]:
        scores.append(features["coherence"]["coherence_score"])

    # Verbosity score
    if "verbosity" in features and "overall_verbosity" in features["verbosity"]:
        scores.append(features["verbosity"]["overall_verbosity"])

    # Tone score
    if "tone" in features and "tone_score" in features["tone"]:
        scores.append(features["tone"]["tone_score"])

    # Fluency score
    if "fluency" in features and "value" in features["fluency"]:
        scores.append(features["fluency"]["value"])

    # Complexity score
    if "complexity" in features and "value" in features["complexity"]:
        scores.append(features["complexity"]["value"])

    # Relevance score
    if "relevance" in features and "value" in features["relevance"]:
        scores.append(features["relevance"]["value"])

    # Subjectivity score
    if "subjectivity" in features and "value" in features["subjectivity"]:
        scores.append(features["subjectivity"]["value"])

    if not scores:
        return 0.5, 0.0  # Default if no scores available

    # Calculate weighted average
    # Higher scores indicate more AI-like content
    avg_score = sum(scores) / len(scores)

    # Calculate confidence based on score variance
    if len(scores) > 1:
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        confidence = max(0.0, 1.0 - variance)  # Lower variance = higher confidence
    else:
        confidence = 0.5

    return avg_score, confidence


def test_slop_detection():
    """Test slop detection with various text samples."""
    print("🧪 Testing AI Slop Detection")
    print("=" * 50)

    extractor = FeatureExtractor()

    test_cases = [
        {
            "name": "Clean Human Text",
            "text": "I woke up this morning feeling groggy. The coffee machine was broken again, so I had to make instant coffee. It tasted terrible, but I needed the caffeine to function.",
            "expected": "low",
        },
        {
            "name": "Personal Narrative",
            "text": "My grandmother's recipe for apple pie has been passed down for three generations. The secret is in the crust - she uses lard instead of butter, which makes it incredibly flaky.",
            "expected": "low",
        },
        {
            "name": "Technical Writing",
            "text": "The bug was in the database connection pool. When we increased the pool size from 10 to 50, the timeout errors disappeared. The issue was that we had too many concurrent requests.",
            "expected": "low",
        },
        {
            "name": "AI Business Email",
            "text": "I hope this email finds you well. I wanted to reach out to discuss the upcoming project deadline and ensure we're all aligned on the deliverables. Please let me know if you have any questions or concerns.",
            "expected": "medium",
        },
        {
            "name": "Product Description",
            "text": "This innovative product combines cutting-edge technology with user-friendly design to deliver exceptional performance. Its sleek appearance and intuitive interface make it perfect for both beginners and professionals.",
            "expected": "medium",
        },
        {
            "name": "AI Response",
            "text": "I understand your concern and I'm here to help. Let me provide you with a comprehensive solution that addresses all aspects of your inquiry. This approach has been proven effective in similar situations.",
            "expected": "high",
        },
        {
            "name": "Corporate Speak",
            "text": "We are excited to announce our strategic partnership that will revolutionize the industry landscape. This collaboration represents a significant milestone in our journey toward sustainable growth and innovation.",
            "expected": "high",
        },
        {
            "name": "Repetitive AI",
            "text": "I understand your concern. I understand your concern. I understand your concern. Let me help you with this matter. Let me help you with this matter. Let me help you with this matter.",
            "expected": "very_high",
        },
        {
            "name": "Template Content",
            "text": "Welcome to our platform! We're excited to have you join our community of like-minded individuals who are passionate about achieving their goals and making a positive impact in their respective fields.",
            "expected": "high",
        },
        {
            "name": "News Article",
            "text": "Scientists discover new species of deep-sea fish that glows in the dark. The bioluminescent creature was found 2,000 meters below the surface in the Pacific Ocean.",
            "expected": "low",
        },
    ]

    results = []

    for i, test_case in enumerate(test_cases):
        print(f"\n{i + 1}. {test_case['name']}")
        print(f"   Text: {test_case['text'][:60]}...")

        start_time = time.time()

        try:
            # Extract features
            features = extractor.extract_all_features(test_case["text"])

            # Calculate slop score
            slop_score, confidence = calculate_slop_score_from_features(features)

            processing_time = time.time() - start_time

            # Determine if result matches expectation
            correct = False
            if test_case["expected"] == "low" and slop_score < 0.4:
                correct = True
            elif test_case["expected"] == "medium" and 0.2 <= slop_score <= 0.7:
                correct = True
            elif test_case["expected"] == "high" and slop_score > 0.5:
                correct = True
            elif test_case["expected"] == "very_high" and slop_score > 0.7:
                correct = True

            result = {
                "name": test_case["name"],
                "text_length": len(test_case["text"]),
                "slop_score": slop_score,
                "confidence": confidence,
                "processing_time": processing_time,
                "expected": test_case["expected"],
                "correct": correct,
                "success": True,
            }

            status = "✅" if correct else "⚠️"
            print(
                f"   {status} Slop Score: {slop_score:.3f}, Confidence: {confidence:.3f}, Time: {processing_time:.3f}s"
            )
            print(f"   Expected: {test_case['expected']}, Correct: {correct}")

        except Exception as e:
            processing_time = time.time() - start_time
            result = {
                "name": test_case["name"],
                "text_length": len(test_case["text"]),
                "slop_score": None,
                "confidence": None,
                "processing_time": processing_time,
                "expected": test_case["expected"],
                "correct": False,
                "success": False,
                "error": str(e),
            }
            print(f"   ❌ Error: {e}")

        results.append(result)

    # Calculate summary
    successful_results = [r for r in results if r["success"]]
    correct_predictions = [r for r in successful_results if r["correct"]]

    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Total Tests: {len(results)}")
    print(f"   Successful: {len(successful_results)}")
    print(f"   Failed: {len(results) - len(successful_results)}")
    print(f"   Correct Predictions: {len(correct_predictions)}")
    print(
        f"   Accuracy: {len(correct_predictions) / len(successful_results) * 100:.1f}%"
        if successful_results
        else "   Accuracy: N/A"
    )

    if successful_results:
        avg_processing_time = sum(
            r["processing_time"] for r in successful_results
        ) / len(successful_results)
        avg_slop_score = sum(r["slop_score"] for r in successful_results) / len(
            successful_results
        )
        avg_confidence = sum(r["confidence"] for r in successful_results) / len(
            successful_results
        )

        print(f"   Average Processing Time: {avg_processing_time:.3f}s")
        print(f"   Average Slop Score: {avg_slop_score:.3f}")
        print(f"   Average Confidence: {avg_confidence:.3f}")

    # Save results
    output_data = {
        "test_metadata": {
            "timestamp": time.time(),
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            "accuracy": (
                len(correct_predictions) / len(successful_results)
                if successful_results
                else 0
            ),
        },
        "results": results,
    }

    output_path = "tests/working_test_results.json"
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

    # Take first 20 texts for testing
    sample_dataset = full_dataset[:20]

    extractor = FeatureExtractor()
    results = []

    for i, item in enumerate(sample_dataset):
        print(f"Testing {i + 1}/{len(sample_dataset)}: {item['doc_id']}")

        start_time = time.time()

        try:
            features = extractor.extract_all_features(item["text"])
            slop_score, confidence = calculate_slop_score_from_features(features)
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

            status = "✅" if correct else "⚠️"
            print(
                f"  {status} Score: {slop_score:.3f}, Expected: {expected_range:.1f}, Correct: {correct}"
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
    print(f"   Total Tests: {len(results)}")
    print(f"   Successful: {len(successful_results)}")
    print(f"   Correct Predictions: {len(correct_predictions)}")
    print(
        f"   Accuracy: {len(correct_predictions) / len(successful_results) * 100:.1f}%"
        if successful_results
        else "   Accuracy: N/A"
    )

    return results


if __name__ == "__main__":
    # Run basic tests
    basic_results = test_slop_detection()

    # Run dataset sample test
    dataset_results = test_dataset_sample()

    print("\n✅ All tests completed!")
