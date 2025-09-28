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
                "density": {
                    "mean": 0.3,
                    "std": 0.2,
                },  # Lower for better high-slop detection
                "relevance": {"mean": 0.5, "std": 0.15},  # Lower
                "factuality": {"mean": 0.6, "std": 0.18},  # Lower
                "subjectivity": {"mean": 0.3, "std": 0.16},  # Lower
                "coherence": {"mean": 0.4, "std": 0.18},  # Lower
                "repetition": {
                    "mean": 0.10,
                    "std": 0.10,
                },  # Even lower mean to make high repetition scores much higher
                "templated": {
                    "mean": 0.15,
                    "std": 0.15,
                },  # Lower mean to make high templated scores higher
                "verbosity": {"mean": 0.3, "std": 0.14},  # Lower - key slop indicator
                "complexity": {"mean": 0.3, "std": 0.18},  # Lower
                "tone": {"mean": 0.25, "std": 0.18},  # Adjusted for better sensitivity
                "fluency": {"mean": 0.5, "std": 0.16},  # Lower
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

        # Convert to [0,1] range - simpler, more direct approach
        if z_score > 0:
            # More sensitive scaling for positive z-scores
            normalized = min(1.0, 0.4 + (z_score * 0.4))
        else:
            # Sigmoid for negative z-scores
            normalized = 1 / (1 + np.exp(-z_score))

        return float(max(0.0, min(1.0, normalized)))

    def _get_metric_weight(self, metric_name: str) -> float:
        """Get weight for individual metrics within dimensions."""
        # Higher weights for more reliable/important metrics
        metric_weights = {
            # Density metrics - perplexity is most reliable
            "perplexity_score": 2.0,
            "combined_density": 1.5,
            "idea_density_score": 1.0,
            "semantic_density_score": 1.0,
            "conceptual_density_score": 1.0,
            # Repetition metrics - overall and sentence repetition most important
            "overall_repetition": 2.0,
            "sentence_repetition": 1.5,
            "compression_ratio": 1.2,
            "pattern_repetition": 1.0,
            "ngram_repetition": 0.8,
            # Templated metrics - score and hits most important
            "templated_score": 2.0,
            "boilerplate_hits": 1.5,
            "pos_diversity": 1.0,
            # Tone metrics - overall score most important
            "tone_score": 2.0,
            "hedging_ratio": 1.2,
            "sycophancy_ratio": 1.2,
            "formality_ratio": 1.0,
            "passive_ratio": 1.0,
            # Verbosity metrics - overall and key ratios most important
            "overall_verbosity": 2.0,
            "filler_ratio": 1.5,
            "words_per_sentence": 1.2,
            "listiness": 1.0,
            "sentence_variance": 0.8,
            # Coherence metrics - score and continuity most important
            "coherence_score": 2.0,
            "entity_continuity": 1.5,
            "embedding_drift": 1.0,
            # Relevance metrics - score most important
            "relevance_score": 2.0,
            "mean_similarity": 1.2,
            "min_similarity": 1.2,
            "low_relevance_ratio": 1.0,
            "relevance_variance": 0.8,
            # Factuality metrics - score and ratios most important
            "factuality_score": 2.0,
            "unsupported_ratio": 1.5,
            "contradictions_count": 1.2,
            # Subjectivity metrics - score and ratios most important
            "subjectivity_score": 2.0,
            "subjective_ratio": 1.2,
            "bias_ratio": 1.2,
            "neutral_ratio": 1.0,
            # Fluency metrics - score and ratios most important
            "fluency_score": 2.0,
            "grammar_error_ratio": 1.5,
            "unnatural_phrase_ratio": 1.5,
            "fragment_ratio": 1.2,
            # Complexity metrics - score most important
            "complexity_score": 2.0,
            "complex_word_ratio": 1.2,
            "complex_phrase_ratio": 1.2,
            "flesch_kincaid_grade": 1.0,
        }

        return metric_weights.get(metric_name, 1.0)

    def normalize_score_with_inversion(
        self, metric_name: str, raw_score: float, domain: str
    ) -> float:
        """Normalize a raw metric score with proper inversion for certain metrics."""
        # Define which metrics should be inverted (lower is better for slop detection)
        inverted_metrics = {
            # These metrics: lower values = more slop, higher values = less slop
            "perplexity_score",
            "idea_density_score",
            "semantic_density_score",
            "conceptual_density_score",
            "combined_density",
            "sentence_repetition",
            "compression_ratio",
            "overall_repetition",
            "tone_score",
            "hedging_ratio",
            "sycophancy_ratio",
            "formality_ratio",
            "passive_ratio",
            "overall_verbosity",
            "filler_ratio",
            "listiness",
            "sentence_variance",
            "coherence_score",
            "entity_continuity",
            "embedding_drift",
            "relevance_score",
            "mean_similarity",
            "min_similarity",
            "low_relevance_ratio",
            "relevance_variance",
            "factuality_score",
            "unsupported_ratio",
            "contradictions_count",
            "subjectivity_score",
            "subjective_ratio",
            "bias_ratio",
            "neutral_ratio",
            "fluency_score",
            "grammar_error_ratio",
            "unnatural_phrase_ratio",
            "fragment_ratio",
            "complexity_score",
            "complex_word_ratio",
            "complex_phrase_ratio",
            "flesch_kincaid_grade",
            # These are metadata and should be neutral
            "model_name",
            "has_transformer",
            "processing_times",
            "total_processing_time",
            "has_semantic_features",
            "coherence_spans",
            "repetition_spans",
            "templated_spans",
            "tone_spans",
            "verbosity_spans",
            "relevance_spans",
            "factual_claims",
            "hedging_spans",
            "unsupported_spans",
            "contradiction_spans",
            "grammar_errors",
            "unnatural_phrases",
            "fragments",
            "subjective_by_category",
            "bias_by_category",
            "subjective_spans",
            "bias_spans",
            "neutral_spans",
            "complex_words",
            "complex_phrases",
            "value",
        }

        # Normalize the score first
        normalized = self.normalize_score(metric_name, raw_score, domain)

        # Invert if this metric should be inverted
        if metric_name in inverted_metrics:
            return 1.0 - normalized

        return normalized


def get_domain_weights(domain: str) -> dict[str, float]:
    """Get domain-specific weights for combining metrics."""
    weights = {
        "general": {
            "density": 0.10,  # Reduced - can be misleading
            "relevance": 0.12,  # Reduced
            "factuality": 0.10,  # Reduced
            "coherence": 0.08,  # Reduced
            "repetition": 0.25,  # Significantly increased - primary slop indicator
            "templated": 0.20,  # Significantly increased - primary slop indicator
            "verbosity": 0.08,  # Reduced
            "complexity": 0.05,  # Reduced
            "tone": 0.06,  # Reduced
            "subjectivity": 0.04,  # Reduced
            "fluency": 0.03,  # Reduced
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
        # Skip non-scalar features (like spans, complex objects)
        if not isinstance(metric_data, dict) or "value" not in metric_data:
            continue

        # Extract the main score value - different extractors use different key names
        raw_score = _extract_main_score(metric_name, metric_data)
        normalized_score = normalizer.normalize_score_with_inversion(
            metric_name, raw_score, domain
        )

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
        value = metric_data[key]
        # Handle different value types
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, list):
            # For span features and other list-based features, return neutral score
            return 0.5
        elif isinstance(value, str):
            # Try to convert string numbers to float
            try:
                return float(value)
            except ValueError:
                return 0.5
        else:
            return 0.5
    elif "value" in metric_data:
        value = metric_data["value"]
        # Handle different value types
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, list):
            # For span features and other list-based features, return neutral score
            return 0.5
        elif isinstance(value, str):
            # Try to convert string numbers to float
            try:
                return float(value)
            except ValueError:
                return 0.5
        else:
            return 0.5
    else:
        # If no score found, return 0.5 as neutral
        return 0.5


def combine_scores(
    metrics: dict[str, dict[str, Any]],
    domain: str,
    normalizer: ScoreNormalizer | None = None,
) -> tuple[float, float]:
    """Combine normalized metrics into a composite slop score."""
    weights = get_domain_weights(domain)

    # Map individual metrics to core dimensions
    dimension_mapping = {
        # Density dimension
        "combined_density": "density",
        "perplexity_score": "density",
        "idea_density_score": "density",
        "semantic_density_score": "density",
        "conceptual_density_score": "density",
        # Repetition dimension
        "overall_repetition": "repetition",
        "ngram_repetition": "repetition",
        "sentence_repetition": "repetition",
        "compression_ratio": "repetition",
        "pattern_repetition": "repetition",
        # Templated dimension
        "templated_score": "templated",
        "boilerplate_hits": "templated",
        "pos_diversity": "templated",
        # Tone dimension
        "tone_score": "tone",
        "hedging_ratio": "tone",
        "sycophancy_ratio": "tone",
        "formality_ratio": "tone",
        "passive_ratio": "tone",
        # Verbosity dimension
        "overall_verbosity": "verbosity",
        "words_per_sentence": "verbosity",
        "filler_ratio": "verbosity",
        "listiness": "verbosity",
        "sentence_variance": "verbosity",
        # Coherence dimension
        "coherence_score": "coherence",
        "entity_continuity": "coherence",
        "embedding_drift": "coherence",
        # Relevance dimension
        "relevance_score": "relevance",
        "mean_similarity": "relevance",
        "min_similarity": "relevance",
        "low_relevance_ratio": "relevance",
        "relevance_variance": "relevance",
        # Factuality dimension
        "factuality_score": "factuality",
        "unsupported_ratio": "factuality",
        "contradictions_count": "factuality",
        # Subjectivity dimension
        "subjectivity_score": "subjectivity",
        "subjective_ratio": "subjectivity",
        "bias_ratio": "subjectivity",
        "neutral_ratio": "subjectivity",
        # Fluency dimension
        "fluency_score": "fluency",
        "grammar_error_ratio": "fluency",
        "unnatural_phrase_ratio": "fluency",
        "fragment_ratio": "fluency",
        # Complexity dimension
        "complexity_score": "complexity",
        "complex_word_ratio": "complexity",
        "complex_phrase_ratio": "complexity",
        "flesch_kincaid_grade": "complexity",
    }

    # Group metrics by dimension and calculate dimension scores
    dimension_scores = {}
    dimension_weights = {}

    for metric_name, metric_data in metrics.items():
        if metric_name in dimension_mapping:
            dimension = dimension_mapping[metric_name]
            score = metric_data.get("value", 0.5)

            if dimension not in dimension_scores:
                dimension_scores[dimension] = []
                dimension_weights[dimension] = []

            dimension_scores[dimension].append(score)
            # Use metric-specific weights within dimensions for better sensitivity
            if normalizer:
                metric_weight = normalizer._get_metric_weight(metric_name)
            else:
                metric_weight = 1.0  # Default weight if no normalizer
            dimension_weights[dimension].append(metric_weight)

    # Calculate weighted average for each dimension
    total_weight = 0.0
    weighted_sum = 0.0
    available_dimensions = 0

    for dimension, scores in dimension_scores.items():
        if dimension in weights:
            # Calculate weighted average score for this dimension
            if scores and dimension_weights.get(dimension):
                dim_scores = dimension_scores[dimension]
                dim_weights_list = dimension_weights[dimension]

                weighted_sum_dim = sum(
                    s * w for s, w in zip(dim_scores, dim_weights_list, strict=False)
                )
                total_weight_dim = sum(dim_weights_list)

                if total_weight_dim > 0:
                    dimension_score = weighted_sum_dim / total_weight_dim
                else:
                    dimension_score = (
                        sum(dim_scores) / len(dim_scores) if dim_scores else 0.5
                    )

                weight = weights[dimension]
                weighted_sum += weight * dimension_score
                total_weight += weight
                available_dimensions += 1

    if total_weight == 0:
        return 0.5, 0.0  # Default fallback

    # Calculate composite score
    composite_score = weighted_sum / total_weight

    # Calculate confidence based on coverage
    expected_dimensions = len(weights)
    coverage = available_dimensions / expected_dimensions
    confidence = min(coverage, 1.0)  # Cap at 1.0

    return composite_score, confidence


def get_slop_level(score: float) -> str:
    """Convert slop score to categorical level."""
    if score <= 0.50:
        return "Clean"
    elif score <= 0.70:
        return "Watch"
    elif score <= 0.85:
        return "Sloppy"
    else:
        return "High-Slop"


def calculate_confidence_intervals(
    metrics: dict[str, dict[str, Any]],
    domain: str,
    normalizer: ScoreNormalizer | None = None,
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
    score, _ = combine_scores(metrics, domain, normalizer)

    return {
        "lower": max(0.0, score - margin),
        "upper": min(1.0, score + margin),
        "method": "normal_approximation",
        "confidence_level": 0.95,
    }
