"""
Score normalization and combination for AI slop metrics.

Handles converting raw metric values to normalized scores and combining them
into a composite slop score with domain-specific weighting.
"""

from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


class ScoreNormalizer:
    """Handles normalization of raw metric scores using calibration data."""

    def __init__(self, calibration_dir: Path | None = None):
        """Initialize with calibration data."""
        self.calibration_dir = calibration_dir or Path(__file__).parent / "calibration"
        self.calibration_data = self._load_calibration_data()

    def _load_calibration_data(self) -> dict[str, Any]:
        """Load calibration statistics from data files."""
        # Default calibration values based on research literature
        return {
            "general": {
                "density": {"mean": 0.5, "std": 0.2},
                "relevance": {"mean": 0.6, "std": 0.15},
                "factuality": {"mean": 0.7, "std": 0.18},  # Higher mean for factuality
                "subjectivity": {"mean": 0.4, "std": 0.16},
                "coherence": {"mean": 0.5, "std": 0.18},
                "repetition": {"mean": 0.3, "std": 0.12},
                "templated": {"mean": 0.4, "std": 0.16},
                "verbosity": {"mean": 0.5, "std": 0.14},
                "complexity": {"mean": 0.4, "std": 0.18},  # Lower mean for complexity
                "tone": {"mean": 0.4, "std": 0.15},
                "fluency": {"mean": 0.6, "std": 0.16},  # Higher mean for fluency
            },
            "news": {
                "density": {"mean": 0.6, "std": 0.18},
                "relevance": {"mean": 0.7, "std": 0.12},
                "factuality": {"mean": 0.8, "std": 0.15},  # Higher for news
                "subjectivity": {"mean": 0.3, "std": 0.12},
                "coherence": {"mean": 0.6, "std": 0.16},
                "repetition": {"mean": 0.2, "std": 0.10},
                "templated": {"mean": 0.3, "std": 0.14},
                "verbosity": {"mean": 0.4, "std": 0.13},
                "complexity": {"mean": 0.5, "std": 0.15},
                "tone": {"mean": 0.3, "std": 0.13},
                "fluency": {"mean": 0.7, "std": 0.14},
            },
            "qa": {
                "relevance": {"mean": 0.7, "std": 0.15},  # High for QA
                "factuality": {"mean": 0.6, "std": 0.20},  # Important for QA
                "density": {"mean": 0.5, "std": 0.16},
                "subjectivity": {"mean": 0.4, "std": 0.14},
                "coherence": {"mean": 0.5, "std": 0.17},
                "repetition": {"mean": 0.2, "std": 0.09},
                "templated": {"mean": 0.3, "std": 0.15},
                "verbosity": {"mean": 0.4, "std": 0.14},
                "complexity": {"mean": 0.4, "std": 0.16},
                "tone": {"mean": 0.3, "std": 0.12},
                "fluency": {"mean": 0.6, "std": 0.18},
            },
        }

    def normalize_score(self, metric_name: str, raw_score: float, domain: str) -> float:
        """Normalize a raw metric score using z-score normalization."""
        if domain not in self.calibration_data:
            domain = "general"

        domain_stats = self.calibration_data[domain]

        if metric_name not in domain_stats:
            # Fallback to general domain
            if metric_name in self.calibration_data["general"]:
                domain_stats = self.calibration_data["general"]
            else:
                # No calibration data available, return raw score clipped to [0,1]
                return max(0.0, min(1.0, raw_score))

        mean_val = domain_stats[metric_name]["mean"]
        std_val = domain_stats[metric_name]["std"]

        if std_val == 0:
            return 1.0 if raw_score >= mean_val else 0.0

        # Z-score normalization
        z_score = float((raw_score - mean_val) / std_val)

        # Convert to [0,1] range using sigmoid-like function
        normalized = 1 / (1 + np.exp(-z_score))

        return float(max(0.0, min(1.0, normalized)))


def get_domain_weights(domain: str) -> dict[str, float]:
    """Get domain-specific weights for combining metrics."""
    weights = {
        "general": {
            "density": 0.15,  # Strongest predictor (β=0.05)
            "relevance": 0.20,  # Strongest predictor (β=0.06)
            "factuality": 0.15,  # Highest agreement (AC₁=0.76)
            "coherence": 0.10,
            "repetition": 0.08,
            "templated": 0.08,
            "verbosity": 0.08,
            "complexity": 0.06,
            "tone": 0.05,  # Strong predictor (β=0.05)
            "subjectivity": 0.05,
            "fluency": 0.05,
        },
        "news": {
            "density": 0.18,  # Strong predictor
            "relevance": 0.20,  # Strongest predictor
            "factuality": 0.18,  # High agreement, important for news
            "coherence": 0.12,
            "repetition": 0.08,
            "templated": 0.08,
            "subjectivity": 0.08,  # Important for news objectivity
            "verbosity": 0.05,
            "complexity": 0.03,
            "tone": 0.05,
            "fluency": 0.05,
        },
        "qa": {
            "relevance": 0.25,  # Strongest predictor, crucial for QA
            "factuality": 0.20,  # High agreement, crucial for QA
            "density": 0.15,  # Strong predictor
            "coherence": 0.12,
            "repetition": 0.08,  # Structure metrics
            "templated": 0.08,
            "fluency": 0.08,
            "verbosity": 0.04,
            "tone": 0.05,
            "subjectivity": 0.05,
            "complexity": 0.05,
        },
    }

    return weights.get(domain, weights["general"])


def normalize_scores(
    metrics: dict[str, dict[str, Any]], domain: str
) -> dict[str, dict[str, Any]]:
    """Normalize all metric scores for a domain."""
    normalizer = ScoreNormalizer()

    normalized = {}
    for metric_name, metric_data in metrics.items():
        # Extract the main score value - different extractors use different key names
        raw_score = _extract_main_score(metric_name, metric_data)
        normalized_score = normalizer.normalize_score(metric_name, raw_score, domain)

        normalized[metric_name] = {
            **metric_data,
            "value": normalized_score,
            "normalized": True,
        }

    return normalized


def _extract_main_score(metric_name: str, metric_data: dict[str, Any]) -> float:
    """Extract the main score value from metric data based on metric type."""
    # Map metric names to their main score keys
    score_keys = {
        "density": "combined_density",
        "repetition": "overall_repetition",
        "templated": "templated_score",
        "coherence": "coherence_score",
        "verbosity": "overall_verbosity",
        "tone": "tone_score",
        "relevance": "relevance_score",  # Updated based on paper
        "factuality": "factuality_score",  # New based on paper
        "subjectivity": "subjectivity_score",  # Updated based on paper
        "fluency": "fluency_score",  # Updated based on paper
        "complexity": "complexity_score",  # Updated based on paper
    }

    # Try the specific key first, then fall back to "value"
    key = score_keys.get(metric_name, "value")
    if key in metric_data:
        return float(metric_data[key])
    elif "value" in metric_data:
        return float(metric_data["value"])
    else:
        # If no score found, return 0.5 as neutral
        return 0.5


def combine_scores(
    metrics: dict[str, dict[str, Any]], domain: str
) -> tuple[float, float]:
    """Combine normalized metrics into a composite slop score."""
    weights = get_domain_weights(domain)

    total_weight = 0.0
    weighted_sum = 0.0
    available_metrics = 0

    for metric_name, metric_data in metrics.items():
        if metric_name in weights:
            weight = weights[metric_name]
            score = metric_data["value"]

            weighted_sum += weight * score
            total_weight += weight
            available_metrics += 1

    if total_weight == 0:
        return 0.5, 0.0  # Default fallback

    # Calculate composite score
    composite_score = weighted_sum / total_weight

    # Calculate confidence based on coverage
    expected_metrics = len(weights)
    coverage = available_metrics / expected_metrics
    confidence = min(coverage, 1.0)  # Cap at 1.0

    return composite_score, confidence


def get_slop_level(score: float) -> str:
    """Convert slop score to categorical level."""
    if score <= 0.30:
        return "Clean"
    elif score <= 0.55:
        return "Watch"
    elif score <= 0.75:
        return "Sloppy"
    else:
        return "High-Slop"


def calculate_confidence_intervals(
    metrics: dict[str, dict[str, Any]], domain: str
) -> dict[str, Any]:
    """Calculate confidence intervals for the composite score."""
    # This is a simplified version - in practice would use bootstrap sampling
    weights = get_domain_weights(domain)

    # Calculate standard error based on available metrics
    available_metrics = [name for name in metrics.keys() if name in weights]
    n_metrics = len(available_metrics)

    if n_metrics == 0:
        return {"lower": 0.0, "upper": 1.0, "method": "none"}

    # Simplified confidence interval calculation
    std_error = 1.0 / (n_metrics**0.5)  # Simplified
    z_score = stats.norm.ppf(0.975)  # 95% confidence

    margin = z_score * std_error

    # Get point estimate
    score, _ = combine_scores(metrics, domain)

    return {
        "lower": max(0.0, score - margin),
        "upper": min(1.0, score + margin),
        "method": "normal_approximation",
        "confidence_level": 0.95,
    }
