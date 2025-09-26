#!/usr/bin/env python3
"""
Full dataset test for AI slop detection - runs all 102 texts.
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


def test_full_dataset():
    """Test the full dataset of 102 texts."""
    print("🚀 Testing Full Dataset - 102 AI Slop Detection Samples")
    print("=" * 70)

    # Load the full dataset
    with open("tests/test_dataset.json", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"📊 Loaded {len(dataset)} texts for testing")

    extractor = FeatureExtractor()
    results = []

    # Process each text
    for i, item in enumerate(dataset):
        print(f"Processing {i + 1:3d}/{len(dataset)}: {item['doc_id']}")

        start_time = time.time()

        try:
            # Extract features
            features = extractor.extract_all_features(item["text"])

            # Calculate slop score
            slop_score, confidence = calculate_slop_score_from_features(features)

            processing_time = time.time() - start_time

            # Determine if prediction matches expectation
            expected_range = item["expected_slop_range"]
            if expected_range < 0.3:
                correct = slop_score < 0.4
                expected_category = "Low"
            elif expected_range < 0.6:
                correct = 0.2 <= slop_score <= 0.7
                expected_category = "Medium"
            else:
                correct = (
                    slop_score > 0.70
                )  # Fixed: match updated system classification logic
                expected_category = "High"

            result = {
                "doc_id": item["doc_id"],
                "category": item["category"],
                "domain": item["domain"],
                "text_length": len(item["text"]),
                "slop_score": slop_score,
                "confidence": confidence,
                "processing_time": processing_time,
                "expected_slop_range": expected_range,
                "expected_category": expected_category,
                "correct": correct,
                "success": True,
            }

            status = "✅" if correct else "⚠️"
            print(
                f"  {status} Score: {slop_score:.3f}, Expected: {expected_category}, Correct: {correct}"
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
                "expected_category": expected_category,
                "correct": False,
                "success": False,
                "error": str(e),
            }
            print(f"  ❌ Error: {e}")

        results.append(result)

    # Calculate comprehensive statistics
    successful_results = [r for r in results if r["success"]]
    correct_predictions = [r for r in successful_results if r["correct"]]

    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 70)

    print("\n🎯 Overall Performance:")
    print(f"   Total Texts: {len(results)}")
    print(f"   Successful Analyses: {len(successful_results)}")
    print(f"   Failed Analyses: {len(results) - len(successful_results)}")
    print(f"   Success Rate: {len(successful_results) / len(results) * 100:.1f}%")
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

        print("\n⚡ Performance Metrics:")
        print(f"   Average Processing Time: {avg_processing_time:.3f}s")
        print(
            f"   Total Processing Time: {sum(r['processing_time'] for r in successful_results):.1f}s"
        )
        print(f"   Average Slop Score: {avg_slop_score:.3f}")
        print(f"   Average Confidence: {avg_confidence:.3f}")

    # Analyze by expected category
    print("\n📈 Performance by Expected Category:")
    category_stats = {}
    for result in successful_results:
        category = result["expected_category"]
        if category not in category_stats:
            category_stats[category] = {"total": 0, "correct": 0, "scores": []}

        category_stats[category]["total"] += 1
        if result["correct"]:
            category_stats[category]["correct"] += 1
        category_stats[category]["scores"].append(result["slop_score"])

    for category, stats in category_stats.items():
        accuracy = stats["correct"] / stats["total"] * 100
        avg_score = sum(stats["scores"]) / len(stats["scores"])
        print(
            f"   {category:6s}: {accuracy:5.1f}% accuracy ({stats['correct']:2d}/{stats['total']:2d}), avg score: {avg_score:.3f}"
        )

    # Analyze by text category
    print("\n🏷️  Performance by Text Category:")
    text_category_stats = {}
    for result in successful_results:
        category = result["category"]
        if category not in text_category_stats:
            text_category_stats[category] = {"total": 0, "correct": 0, "scores": []}

        text_category_stats[category]["total"] += 1
        if result["correct"]:
            text_category_stats[category]["correct"] += 1
        text_category_stats[category]["scores"].append(result["slop_score"])

    # Sort by accuracy
    sorted_categories = sorted(
        text_category_stats.items(),
        key=lambda x: x[1]["correct"] / x[1]["total"],
        reverse=True,
    )

    for category, stats in sorted_categories[:10]:  # Show top 10
        accuracy = stats["correct"] / stats["total"] * 100
        avg_score = sum(stats["scores"]) / len(stats["scores"])
        print(
            f"   {category:12s}: {accuracy:5.1f}% accuracy ({stats['correct']:2d}/{stats['total']:2d}), avg score: {avg_score:.3f}"
        )

    # Save detailed results
    output_data = {
        "test_metadata": {
            "timestamp": time.time(),
            "total_texts": len(results),
            "successful_tests": len(successful_results),
            "accuracy": (
                len(correct_predictions) / len(successful_results)
                if successful_results
                else 0
            ),
            "avg_processing_time": avg_processing_time if successful_results else 0,
            "avg_slop_score": avg_slop_score if successful_results else 0,
            "avg_confidence": avg_confidence if successful_results else 0,
        },
        "category_performance": category_stats,
        "text_category_performance": text_category_stats,
        "detailed_results": results,
    }

    output_path = "tests/full_dataset_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Detailed results saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = test_full_dataset()
    print("\n✅ Full dataset test completed!")
