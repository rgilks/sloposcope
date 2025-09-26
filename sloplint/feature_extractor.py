"""
Feature extraction orchestrator for AI slop analysis.

Combines all individual feature extractors and provides a unified interface.
"""

import logging
from typing import Any

from .features import (
    extract_coherence_features,
    extract_density_features,
    extract_repetition_features,
    extract_templated_features,
    extract_tone_features,
    extract_verbosity_features,
)
from .features.complexity import extract_features as extract_complexity_features
from .features.factuality import extract_features as extract_factuality_features
from .features.fluency import extract_features as extract_fluency_features
from .features.relevance import extract_features as extract_relevance_features
from .features.subjectivity import extract_features as extract_subjectivity_features
from .nlp.pipeline import NLPPipeline
from .spans import Span, SpanCollection, SpanType

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Orchestrates all feature extraction for AI slop analysis."""

    def __init__(self, language: str = "en"):
        """Initialize feature extractor."""
        self.language = language
        self.nlp_pipeline = NLPPipeline(language=language)

    def extract_all_features(self, text: str) -> dict[str, Any]:
        """Extract all AI slop features from text."""
        logger.info(f"Extracting features from {len(text)} characters of text...")

        # Process text through NLP pipeline
        doc_result = self.nlp_pipeline.process(text)

        if not doc_result:
            logger.error("Failed to process text through NLP pipeline")
            return self._create_empty_features()

        sentences = doc_result["sentences"]
        tokens = doc_result["tokens"]
        pos_tags = doc_result.get("pos_tags", [])

        # Extract features from each module
        features = {}

        try:
            # Information Utility
            density_features = extract_density_features(
                text, sentences, tokens, pos_tags
            )
            features["density"] = density_features

            # Information Quality - now implemented based on paper findings
            relevance_features = extract_relevance_features(text, sentences, tokens)
            features["relevance"] = relevance_features

            factuality_features = extract_factuality_features(text, sentences, tokens)
            features["factuality"] = factuality_features

            subjectivity_features = extract_subjectivity_features(
                text, sentences, tokens
            )
            features["subjectivity"] = subjectivity_features

            # Style Quality
            repetition_features = extract_repetition_features(text, sentences, tokens)
            features["repetition"] = repetition_features

            templated_features = extract_templated_features(
                text, sentences, tokens, pos_tags
            )
            features["templated"] = templated_features

            coherence_features = extract_coherence_features(text, sentences, tokens)
            features["coherence"] = coherence_features

            verbosity_features = extract_verbosity_features(text, sentences, tokens)
            features["verbosity"] = verbosity_features

            # Fluency and Complexity - now implemented based on paper findings
            fluency_features = extract_fluency_features(text, sentences, tokens)
            features["fluency"] = fluency_features

            complexity_features = extract_complexity_features(text, sentences, tokens)
            features["complexity"] = complexity_features

            tone_features = extract_tone_features(text, sentences, tokens)
            features["tone"] = tone_features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return self._create_empty_features()

        return features

    def _create_empty_features(self) -> dict[str, Any]:
        """Create empty feature set for error cases."""
        return {
            "density": {
                "combined_density": 0.0,
                "perplexity": 25.0,
                "idea_density": 0.0,
            },
            "relevance": {"value": 0.0, "mean_sim": 0.0, "low_sim_frac": 0.0},
            "subjectivity": {"value": 0.0, "top_terms": []},
            "repetition": {"overall_repetition": 0.0, "compression_ratio": 0.0},
            "templated": {"templated_score": 0.0, "boilerplate_hits": 0},
            "coherence": {
                "coherence_score": 0.0,
                "entity_continuity": 0.0,
                "embedding_drift": 0.0,
            },
            "verbosity": {"overall_verbosity": 0.0, "words_per_sentence": 0.0},
            "fluency": {"value": 0.0, "grammar_errors_k": 0.0, "ppl_spikes": 0},
            "complexity": {"value": 0.0, "fkgl": 0.0, "fog": 0.0},
            "tone": {"tone_score": 0.0, "hedging_ratio": 0.0, "sycophancy_ratio": 0.0},
        }

    def extract_spans(self, text: str) -> SpanCollection:
        """Extract all problematic spans from text."""
        features = self.extract_all_features(text)

        spans = SpanCollection()

        # Extract spans from each feature module
        try:
            # Repetition spans
            if (
                "repetition" in features
                and "repetition_spans" in features["repetition"]
            ):
                for span_data in features["repetition"]["repetition_spans"]:
                    span = Span(
                        start=span_data["start"],
                        end=span_data["end"],
                        span_type=SpanType.REPETITION,
                        note=span_data.get("note", "Repetitive content"),
                    )
                    spans.add_span(span)

            # Templated spans
            if "templated" in features and "templated_spans" in features["templated"]:
                for span_data in features["templated"]["templated_spans"]:
                    span = Span(
                        start=span_data["start"],
                        end=span_data["end"],
                        span_type=SpanType.TEMPLATED,
                        note=span_data.get("note", "Templated content"),
                    )
                    spans.add_span(span)

            # Coherence spans
            if "coherence" in features and "coherence_spans" in features["coherence"]:
                for span_data in features["coherence"]["coherence_spans"]:
                    span = Span(
                        start=span_data["start"],
                        end=span_data["end"],
                        span_type=SpanType.COHERENCE_BREAK,
                        note=span_data.get("note", "Coherence break"),
                    )
                    spans.add_span(span)

            # Tone spans
            if "tone" in features and "tone_spans" in features["tone"]:
                for span_data in features["tone"]["tone_spans"]:
                    if span_data.get("type") == "hedging":
                        span_type = SpanType.HEDGING
                    else:
                        span_type = SpanType.OFF_TOPIC  # Generic for other tone issues
                    span = Span(
                        start=span_data["start"],
                        end=span_data["end"],
                        span_type=span_type,
                        note=span_data.get("note", "Tone issue"),
                    )
                    spans.add_span(span)

        except Exception as e:
            logger.error(f"Error extracting spans: {e}")

        return spans
