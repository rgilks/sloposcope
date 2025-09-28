#!/usr/bin/env python3
"""
Systematic testing script for sloposcope to identify issues and improvements.
Tests various types of high/low slop content and analyzes results.
"""

import re
import subprocess
import sys


class SlopTestCase:
    def __init__(
        self, name: str, text: str, expected_slop_level: str, domain: str = "general"
    ):
        self.name = name
        self.text = text
        self.expected_slop_level = expected_slop_level
        self.domain = domain


class SlopTester:
    def __init__(self):
        self.test_cases = self._create_test_cases()
        self.results = []

    def _create_test_cases(self) -> list[SlopTestCase]:
        """Create comprehensive test cases covering different slop scenarios."""

        return [
            # High Slop Cases
            SlopTestCase(
                "High Slop - Generic Corporate Speak",
                "In today's rapidly evolving business landscape, stakeholders must leverage synergies and optimize core competencies to achieve sustainable competitive advantages. Our innovative solutions empower organizations to maximize shareholder value while maintaining strategic alignment across all operational silos.",
                "high",
                "general",
            ),
            SlopTestCase(
                "High Slop - AI Generated Nonsense",
                "The quantum mechanics of consciousness interfaces with blockchain technology through neural networks, creating emergent properties that transcend traditional computational paradigms. This paradigm shift enables unprecedented scalability in distributed ledger systems.",
                "high",
                "general",
            ),
            SlopTestCase(
                "High Slop - Excessive Repetition",
                "The project was successful. The team worked hard and the project was successful. Success was achieved through hard work. The successful outcome demonstrates success. Success, success, success - that's what we achieved.",
                "high",
                "general",
            ),
            SlopTestCase(
                "High Slop - Template News Article",
                "Dr. A. Researcher, a renowned expert at a leading university, recently published groundbreaking research on this important topic. Professor B. Scientist, a leading researcher at a major institution, confirmed these findings in a peer-reviewed study. Dr. C. Expert, an authority at a top university, emphasized the significance of these results.",
                "high",
                "news",
            ),
            SlopTestCase(
                "High Slop - Verbose Academic Style",
                "This paper presents a comprehensive analysis of the phenomenon under investigation. The methodology employed in this research endeavor involves multiple stages of data collection and analysis. The findings reveal significant correlations between the variables examined, with particular attention to the moderating factors that influence the observed relationships.",
                "high",
                "general",
            ),
            SlopTestCase(
                "High Slop - Factually Incorrect",
                "Recent studies show that drinking 8 glasses of water per day can cure all forms of cancer. This scientifically proven method has been endorsed by the World Health Organization and leading medical experts worldwide.",
                "high",
                "general",
            ),
            # Low Slop Cases
            SlopTestCase(
                "Low Slop - Clear Human Writing",
                "The meeting went well. We discussed the budget, assigned tasks, and set a deadline for next Friday. Everyone seemed engaged and we made good progress on the project timeline.",
                "low",
                "general",
            ),
            SlopTestCase(
                "Low Slop - Simple Instructions",
                "To make coffee: Fill the pot with water, add 2 tablespoons of ground coffee to the filter, and turn on the machine. Wait 5 minutes for it to brew, then pour and enjoy.",
                "low",
                "general",
            ),
            SlopTestCase(
                "Low Slop - Factual News",
                "The Federal Reserve announced today that interest rates will remain unchanged at 5.25-5.50%. This decision follows recent data showing inflation has cooled to 3.1% annually. Stock markets rose 1.2% in response to the news.",
                "low",
                "news",
            ),
            SlopTestCase(
                "Low Slop - Technical Documentation",
                "The API endpoint accepts POST requests with JSON payload containing 'username' and 'password' fields. Returns a JWT token on success (200) or error message on failure (401). Rate limited to 100 requests per minute.",
                "low",
                "general",
            ),
            SlopTestCase(
                "Low Slop - Creative Writing",
                "The old house stood silent on the hill, its windows like empty eyes staring at the overgrown garden. Wind whispered through the broken fence, carrying the scent of rain and forgotten memories.",
                "low",
                "general",
            ),
            SlopTestCase(
                "Low Slop - Scientific Abstract",
                "We investigated the effects of temperature on enzyme activity using spectrophotometric analysis. Results show optimal activity at 37°C with a 40% decrease at 50°C. Statistical analysis confirms significance (p < 0.01).",
                "low",
                "general",
            ),
        ]

    def run_test(self, test_case: SlopTestCase) -> dict:
        """Run a single test case and return results."""
        try:
            # Create temporary file with test text
            with open("/tmp/test_text.txt", "w") as f:
                f.write(test_case.text)

            # Run sloposcope analysis
            cmd = [
                sys.executable,
                "-m",
                "sloplint.cli",
                "analyze",
                "/tmp/test_text.txt",
                "--domain",
                test_case.domain,
                "--explain",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Parse output to extract key metrics
            output = result.stdout
            slop_score = self._extract_slop_score(output)
            level = self._extract_level(output)
            key_issues = self._extract_key_issues(output)

            return {
                "name": test_case.name,
                "expected": test_case.expected_slop_level,
                "actual_score": slop_score,
                "actual_level": level,
                "key_issues": key_issues,
                "output": output,
                "success": True,
            }

        except Exception as e:
            return {
                "name": test_case.name,
                "expected": test_case.expected_slop_level,
                "error": str(e),
                "success": False,
            }

    def _extract_slop_score(self, output: str) -> float:
        """Extract slop score from CLI output."""
        match = re.search(r"Slop Score: ([0-9.]+)", output)
        return float(match.group(1)) if match else 0.0

    def _extract_level(self, output: str) -> str:
        """Extract slop level from CLI output."""
        match = re.search(r"Slop Score: [0-9.]+ \(([^)]+)\)", output)
        return match.group(1) if match else "unknown"

    def _extract_key_issues(self, output: str) -> list[str]:
        """Extract key issues from CLI output."""
        issues = []
        match = re.search(r"Key Issues:(.*?)(?=Recommendations:|$)", output, re.DOTALL)
        if match:
            issue_text = match.group(1).strip()
            issues = [
                line.strip("• ").strip()
                for line in issue_text.split("\n")
                if line.strip() and "•" in line
            ]
        return issues

    def run_all_tests(self) -> list[dict]:
        """Run all test cases and return results."""
        results = []
        for test_case in self.test_cases:
            print(f"Testing: {test_case.name}")
            result = self.run_test(test_case)
            results.append(result)
            print(
                f"  Result: Score={result.get('actual_score', 'ERROR')}, Level={result.get('actual_level', 'ERROR')}"
            )
        return results

    def analyze_results(self, results: list[dict]) -> dict:
        """Analyze test results to identify patterns and issues."""
        analysis = {
            "total_tests": len(results),
            "successful_tests": sum(1 for r in results if r.get("success", False)),
            "accuracy": {},
            "common_issues": [],
            "improvement_areas": [],
        }

        # Analyze accuracy by expected level
        for expected_level in ["high", "low"]:
            level_results = [
                r
                for r in results
                if r.get("expected") == expected_level and r.get("success", False)
            ]
            if level_results:
                correct = sum(
                    1 for r in level_results if self._matches_expected_level(r)
                )
                accuracy = correct / len(level_results)
                analysis["accuracy"][expected_level] = accuracy

        # Collect all key issues
        all_issues = []
        for result in results:
            if result.get("success", False):
                all_issues.extend(result.get("key_issues", []))

        # Find most common issues
        from collections import Counter

        issue_counts = Counter(all_issues)
        analysis["common_issues"] = issue_counts.most_common(5)

        # Identify improvement areas
        analysis["improvement_areas"] = self._identify_improvement_areas(results)

        return analysis

    def _matches_expected_level(self, result: dict) -> bool:
        """Check if actual level matches expected level."""
        expected = result.get("expected", "")
        actual_level = result.get("actual_level", "")

        if expected == "high":
            return actual_level in ["Sloppy", "High-Slop"]
        elif expected == "low":
            return actual_level in ["Clean", "Watch"]

        return False

    def _identify_improvement_areas(self, results: list[dict]) -> list[str]:
        """Identify areas where the detector needs improvement."""
        improvements = []

        # Calculate accuracy for analysis
        high_results = [
            r
            for r in results
            if r.get("expected") == "high" and r.get("success", False)
        ]
        low_results = [
            r for r in results if r.get("expected") == "low" and r.get("success", False)
        ]

        high_accuracy = (
            sum(1 for r in high_results if self._matches_expected_level(r))
            / len(high_results)
            if high_results
            else 0
        )
        low_accuracy = (
            sum(1 for r in low_results if self._matches_expected_level(r))
            / len(low_results)
            if low_results
            else 0
        )

        if high_accuracy < 0.8:
            improvements.append("Improve detection of high-slop content")
        if low_accuracy < 0.8:
            improvements.append("Improve detection of low-slop (clean) content")

        # Check for common misclassifications
        misclassified = [
            r
            for r in results
            if r.get("success", False) and not self._matches_expected_level(r)
        ]
        if len(misclassified) > 2:
            improvements.append("Reduce false positives/negatives")

        # Check for missing key indicators
        all_issues = []
        for result in results:
            if result.get("success", False):
                all_issues.extend(result.get("key_issues", []))

        from collections import Counter

        issue_counts = Counter(all_issues)

        if issue_counts.get("repetition", 0) == 0:
            improvements.append("Improve repetition detection")
        if issue_counts.get("templated", 0) == 0:
            improvements.append("Improve templated content detection")

        return improvements


def main():
    tester = SlopTester()
    print("🧪 Running comprehensive slop detection tests...\n")

    # Run all tests
    results = tester.run_all_tests()

    # Analyze results
    analysis = tester.analyze_results(results)

    # Print summary
    print("\n📊 TEST RESULTS SUMMARY")
    print(f"Total Tests: {analysis['total_tests']}")
    print(f"Successful Tests: {analysis['successful_tests']}")
    success_rate = (analysis["successful_tests"] / analysis["total_tests"]) * 100
    print(f"Success Rate: {success_rate:.1f}%")

    print("\n🎯 ACCURACY BY EXPECTED SLOP LEVEL:")
    for level, accuracy in analysis["accuracy"].items():
        accuracy_pct = accuracy * 100
        print(f"  {level.upper()}: {accuracy_pct:.1f}%")

    print("\n🚨 MOST COMMON ISSUES DETECTED:")
    for issue, count in analysis["common_issues"]:
        print(f"  {issue}: {count} occurrences")

    print("\n💡 AREAS FOR IMPROVEMENT:")
    for improvement in analysis["improvement_areas"]:
        print(f"  • {improvement}")

    # Detailed results
    print("\n📋 DETAILED RESULTS:")
    for result in results:
        if result.get("success", False):
            status = "✅" if tester._matches_expected_level(result) else "❌"
            score_formatted = f"{result['actual_score']:.3f}"
            print(
                f"{status} {result['name']}: {score_formatted} ({result['actual_level']})"
            )
            if result.get("key_issues"):
                print(f"    Issues: {', '.join(result['key_issues'][:3])}")
        else:
            print(
                f"❌ {result['name']}: ERROR - {result.get('error', 'Unknown error')}"
            )


if __name__ == "__main__":
    main()
