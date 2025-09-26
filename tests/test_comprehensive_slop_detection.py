#!/usr/bin/env python3
"""
Comprehensive test suite for AI slop detection with 100 diverse text samples.
"""

import json
import statistics
import time

from sloplint.combine import combine_scores, normalize_scores
from sloplint.feature_extractor import FeatureExtractor


class SlopDetectionTester:
    """Test AI slop detection across diverse text samples."""

    def __init__(self):
        self.extractor = FeatureExtractor()
        self.results = []
        self.performance_stats = {
            "total_texts": 0,
            "processing_times": [],
            "slop_scores": [],
            "confidence_scores": [],
            "category_accuracy": {},
            "domain_accuracy": {},
        }

    def load_test_dataset(
        self, dataset_path: str = "tests/test_dataset.json"
    ) -> list[dict]:
        """Load the test dataset."""
        with open(dataset_path, encoding="utf-8") as f:
            return json.load(f)

    def analyze_text(self, text: str, doc_id: str, domain: str) -> dict:
        """Analyze a single text for AI slop."""
        start_time = time.time()

        try:
            # Extract features
            features = self.extractor.extract_all_features(text)

            # Normalize scores
            normalized = normalize_scores(features, domain)

            # Combine scores
            slop_score, confidence = combine_scores(normalized, domain)

            processing_time = time.time() - start_time

            return {
                "doc_id": doc_id,
                "domain": domain,
                "text_length": len(text),
                "slop_score": slop_score,
                "confidence": confidence,
                "processing_time": processing_time,
                "features": features,
                "normalized_features": normalized,
                "success": True,
                "error": None,
            }

        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "doc_id": doc_id,
                "domain": domain,
                "text_length": len(text),
                "slop_score": None,
                "confidence": None,
                "processing_time": processing_time,
                "features": None,
                "normalized_features": None,
                "success": False,
                "error": str(e),
            }

    def run_comprehensive_test(
        self, dataset_path: str = "tests/test_dataset.json"
    ) -> dict:
        """Run comprehensive test on all texts in the dataset."""
        print("🚀 Starting comprehensive AI slop detection test...")

        # Load dataset
        dataset = self.load_test_dataset(dataset_path)
        self.performance_stats["total_texts"] = len(dataset)

        print(f"📊 Testing {len(dataset)} texts across multiple categories...")

        # Process each text
        for i, item in enumerate(dataset):
            print(f"Processing {i + 1}/{len(dataset)}: {item['doc_id']}")

            result = self.analyze_text(item["text"], item["doc_id"], item["domain"])

            # Add expected slop range for comparison
            result["expected_slop_range"] = item["expected_slop_range"]
            result["category"] = item["category"]

            self.results.append(result)

            # Update performance stats
            if result["success"]:
                self.performance_stats["processing_times"].append(
                    result["processing_time"]
                )
                self.performance_stats["slop_scores"].append(result["slop_score"])
                self.performance_stats["confidence_scores"].append(result["confidence"])

        # Calculate accuracy metrics
        self._calculate_accuracy_metrics()

        # Generate summary
        summary = self._generate_summary()

        print("✅ Comprehensive test completed!")
        return summary

    def _calculate_accuracy_metrics(self):
        """Calculate accuracy metrics for different categories."""
        for result in self.results:
            if not result["success"]:
                continue

            slop_score = result["slop_score"]
            expected_range = result["expected_slop_range"]
            category = result["category"]
            domain = result["domain"]

            # Determine if prediction is correct
            if expected_range < 0.3:  # Low slop expected
                correct = slop_score < 0.4
            elif expected_range < 0.6:  # Medium slop expected
                correct = 0.2 <= slop_score <= 0.7
            else:  # High slop expected
                correct = (
                    slop_score > 0.70
                )  # Fixed: match updated system classification logic

            # Update category accuracy
            if category not in self.performance_stats["category_accuracy"]:
                self.performance_stats["category_accuracy"][category] = {
                    "correct": 0,
                    "total": 0,
                }

            self.performance_stats["category_accuracy"][category]["total"] += 1
            if correct:
                self.performance_stats["category_accuracy"][category]["correct"] += 1

            # Update domain accuracy
            if domain not in self.performance_stats["domain_accuracy"]:
                self.performance_stats["domain_accuracy"][domain] = {
                    "correct": 0,
                    "total": 0,
                }

            self.performance_stats["domain_accuracy"][domain]["total"] += 1
            if correct:
                self.performance_stats["domain_accuracy"][domain]["correct"] += 1

    def _generate_summary(self) -> dict:
        """Generate comprehensive test summary."""
        successful_results = [r for r in self.results if r["success"]]
        failed_results = [r for r in self.results if not r["success"]]

        summary = {
            "test_summary": {
                "total_texts": len(self.results),
                "successful_analyses": len(successful_results),
                "failed_analyses": len(failed_results),
                "success_rate": (
                    len(successful_results) / len(self.results) if self.results else 0
                ),
            },
            "performance_metrics": {
                "avg_processing_time": (
                    statistics.mean(self.performance_stats["processing_times"])
                    if self.performance_stats["processing_times"]
                    else 0
                ),
                "median_processing_time": (
                    statistics.median(self.performance_stats["processing_times"])
                    if self.performance_stats["processing_times"]
                    else 0
                ),
                "total_processing_time": sum(
                    self.performance_stats["processing_times"]
                ),
                "avg_slop_score": (
                    statistics.mean(self.performance_stats["slop_scores"])
                    if self.performance_stats["slop_scores"]
                    else 0
                ),
                "median_slop_score": (
                    statistics.median(self.performance_stats["slop_scores"])
                    if self.performance_stats["slop_scores"]
                    else 0
                ),
                "avg_confidence": (
                    statistics.mean(self.performance_stats["confidence_scores"])
                    if self.performance_stats["confidence_scores"]
                    else 0
                ),
            },
            "accuracy_by_category": {},
            "accuracy_by_domain": {},
            "slop_score_distribution": self._get_slop_distribution(),
            "failed_analyses": [
                {"doc_id": r["doc_id"], "error": r["error"]} for r in failed_results
            ],
        }

        # Calculate accuracy percentages
        for category, stats in self.performance_stats["category_accuracy"].items():
            summary["accuracy_by_category"][category] = {
                "accuracy": (
                    stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                ),
                "correct": stats["correct"],
                "total": stats["total"],
            }

        for domain, stats in self.performance_stats["domain_accuracy"].items():
            summary["accuracy_by_domain"][domain] = {
                "accuracy": (
                    stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                ),
                "correct": stats["correct"],
                "total": stats["total"],
            }

        return summary

    def _get_slop_distribution(self) -> dict:
        """Get distribution of slop scores."""
        slop_scores = self.performance_stats["slop_scores"]

        if not slop_scores:
            return {}

        distribution = {
            "very_low": len([s for s in slop_scores if s < 0.2]),
            "low": len([s for s in slop_scores if 0.2 <= s < 0.4]),
            "medium": len([s for s in slop_scores if 0.4 <= s < 0.6]),
            "high": len([s for s in slop_scores if 0.6 <= s < 0.8]),
            "very_high": len([s for s in slop_scores if s >= 0.8]),
        }

        return distribution

    def save_results(self, output_path: str = "tests/comprehensive_test_results.json"):
        """Save detailed test results."""
        output_data = {
            "test_metadata": {
                "timestamp": time.time(),
                "total_texts": len(self.results),
                "feature_extractor_version": "1.0.0",  # Could be dynamic
            },
            "summary": self._generate_summary(),
            "detailed_results": self.results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"📁 Detailed results saved to: {output_path}")

    def print_summary(self, summary: dict):
        """Print a formatted summary of the test results."""
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE AI SLOP DETECTION TEST RESULTS")
        print("=" * 80)

        # Test Summary
        test_summary = summary["test_summary"]
        print("\n📊 Test Summary:")
        print(f"   Total Texts: {test_summary['total_texts']}")
        print(f"   Successful Analyses: {test_summary['successful_analyses']}")
        print(f"   Failed Analyses: {test_summary['failed_analyses']}")
        print(f"   Success Rate: {test_summary['success_rate']:.1%}")

        # Performance Metrics
        perf = summary["performance_metrics"]
        print("\n⚡ Performance Metrics:")
        print(f"   Average Processing Time: {perf['avg_processing_time']:.3f}s")
        print(f"   Median Processing Time: {perf['median_processing_time']:.3f}s")
        print(f"   Total Processing Time: {perf['total_processing_time']:.3f}s")
        print(f"   Average Slop Score: {perf['avg_slop_score']:.3f}")
        print(f"   Median Slop Score: {perf['median_slop_score']:.3f}")
        print(f"   Average Confidence: {perf['avg_confidence']:.3f}")

        # Slop Score Distribution
        distribution = summary["slop_score_distribution"]
        print("\n📈 Slop Score Distribution:")
        print(f"   Very Low (0.0-0.2): {distribution.get('very_low', 0)} texts")
        print(f"   Low (0.2-0.4): {distribution.get('low', 0)} texts")
        print(f"   Medium (0.4-0.6): {distribution.get('medium', 0)} texts")
        print(f"   High (0.6-0.8): {distribution.get('high', 0)} texts")
        print(f"   Very High (0.8-1.0): {distribution.get('very_high', 0)} texts")

        # Top Categories by Accuracy
        category_acc = summary["accuracy_by_category"]
        if category_acc:
            print("\n🏆 Top Categories by Accuracy:")
            sorted_categories = sorted(
                category_acc.items(), key=lambda x: x[1]["accuracy"], reverse=True
            )
            for category, stats in sorted_categories[:5]:
                print(
                    f"   {category}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})"
                )

        # Failed Analyses
        failed = summary["failed_analyses"]
        if failed:
            print(f"\n❌ Failed Analyses ({len(failed)}):")
            for failure in failed[:5]:  # Show first 5 failures
                print(f"   {failure['doc_id']}: {failure['error']}")
            if len(failed) > 5:
                print(f"   ... and {len(failed) - 5} more")


def test_comprehensive_slop_detection():
    """Main test function for comprehensive slop detection."""
    tester = SlopDetectionTester()

    # Run comprehensive test
    summary = tester.run_comprehensive_test()

    # Print results
    tester.print_summary(summary)

    # Save detailed results
    tester.save_results()

    # Assertions for test validation
    assert summary["test_summary"]["success_rate"] > 0.9, "Success rate should be > 90%"
    assert summary["performance_metrics"]["avg_processing_time"] < 5.0, (
        "Average processing time should be < 5s"
    )
    assert summary["test_summary"]["total_texts"] >= 70, "Should test at least 70 texts"

    print("\n✅ All test assertions passed!")
    return summary


if __name__ == "__main__":
    test_comprehensive_slop_detection()
