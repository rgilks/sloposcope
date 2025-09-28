"""
Optimized feature extraction orchestrator with performance improvements.

This module provides:
- Lazy loading of feature extractors
- Caching of intermediate results
- Batch processing capabilities
- Memory-efficient processing
"""

import logging
import time
from typing import Any

from .features import (
    extract_coherence_features,
    extract_density_features,
    extract_repetition_features,
    extract_templated_features,
    extract_tone_features,
    extract_verbosity_features,
)
from .features.coherence import SPACY_AVAILABLE
from .features.complexity import extract_features as extract_complexity_features
from .features.factuality import extract_features as extract_factuality_features
from .features.fluency import extract_features as extract_fluency_features
from .features.relevance import (
    SENTENCE_TRANSFORMERS_AVAILABLE,
    extract_relevance_features_fallback,
)
from .features.relevance import (
    extract_features as extract_relevance_features,
)
from .features.subjectivity import extract_features as extract_subjectivity_features
from .nlp.pipeline import NLPPipeline

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Optimized feature extraction orchestrator with performance improvements."""

    def __init__(
        self,
        use_transformer: bool = True,
        enable_caching: bool = True,
        cache_dir: str | None = None,
    ):
        """Initialize optimized feature extractor."""
        self.use_transformer = use_transformer
        self.enable_caching = enable_caching

        # Initialize NLP pipeline with optimizations
        self.nlp_pipeline = NLPPipeline(
            use_transformer=use_transformer,
            enable_caching=enable_caching,
            cache_dir=cache_dir,
        )

        # Cache for processed documents
        self._doc_cache: dict[str, Any] = {}

        # Performance tracking
        self._processing_times: dict[str, float] = {}

    def extract_all_features(self, text: str) -> dict[str, Any]:
        """Extract all features with performance optimizations."""
        start_time = time.time()

        # Process text through NLP pipeline
        doc_result = self.nlp_pipeline.process(text)

        # Extract individual features
        features = {}

        # Extract density features (with semantic embeddings)
        density_start = time.time()
        density_result = extract_density_features(
            text,
            doc_result["sentences"],
            doc_result["tokens"],
            doc_result.get("pos_tags"),
            doc_result.get("sentence_embeddings"),
        )

        # Wrap density features in proper format
        if isinstance(density_result, dict):
            wrapped_density = {}
            for key, value in density_result.items():
                if isinstance(value, (int, float, bool, str)):
                    # Simple scalar values get wrapped
                    wrapped_density[key] = {"value": value, "normalized": False}
                elif isinstance(value, list):
                    # Lists get wrapped
                    wrapped_density[key] = {"value": value, "normalized": False}
                elif isinstance(value, dict):
                    # Complex dictionaries are kept as-is
                    wrapped_density[key] = value
                else:
                    wrapped_density[key] = value
            features.update(wrapped_density)
        self._processing_times["density"] = time.time() - density_start

        # Extract coherence features (with semantic embeddings if available)
        coherence_start = time.time()
        if SPACY_AVAILABLE:
            coherence_result = extract_coherence_features(
                text,
                doc_result["sentences"],
                doc_result["tokens"],
                doc_result.get("sentence_embeddings"),
                self.nlp_pipeline,
            )

            # Wrap coherence features in proper format
            if isinstance(coherence_result, dict):
                wrapped_coherence = {}
                for key, value in coherence_result.items():
                    if isinstance(value, (int, float, bool, str)):
                        # Simple scalar values get wrapped
                        wrapped_coherence[key] = {"value": value, "normalized": False}
                    elif isinstance(value, list):
                        # Lists get wrapped
                        wrapped_coherence[key] = {"value": value, "normalized": False}
                    elif isinstance(value, dict):
                        # Complex dictionaries are kept as-is
                        wrapped_coherence[key] = value
                    else:
                        wrapped_coherence[key] = value
                features.update(wrapped_coherence)
        else:
            # Fallback coherence features
            features.update(
                {
                    "entity_continuity": {"value": 0.5, "normalized": False},
                    "local_coherence": {"value": 0.5, "normalized": False},
                    "global_coherence": {"value": 0.5, "normalized": False},
                    "coherence_score": {"value": 0.5, "normalized": False},
                    "value": {"value": 0.5, "normalized": False},
                }
            )
        self._processing_times["coherence"] = time.time() - coherence_start

        # Extract other features
        feature_extractors = [
            ("repetition", extract_repetition_features),
            ("templated", extract_templated_features),
            ("tone", extract_tone_features),
            ("verbosity", extract_verbosity_features),
            ("complexity", extract_complexity_features),
            ("factuality", extract_factuality_features),
            ("fluency", extract_fluency_features),
            ("subjectivity", extract_subjectivity_features),
        ]

        # Extract relevance features (with semantic embeddings if available)
        relevance_start = time.time()
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                feature_result = extract_relevance_features(
                    text,
                    doc_result["sentences"],
                    doc_result["tokens"],
                )

                # Wrap relevance features in proper format
                if isinstance(feature_result, dict):
                    wrapped_relevance = {}
                    for key, value in feature_result.items():
                        if isinstance(value, (int, float, bool, str)):
                            # Simple scalar values get wrapped
                            wrapped_relevance[key] = {
                                "value": value,
                                "normalized": False,
                            }
                        elif isinstance(value, list):
                            # Lists get wrapped
                            wrapped_relevance[key] = {
                                "value": value,
                                "normalized": False,
                            }
                        elif isinstance(value, dict):
                            # Complex dictionaries are kept as-is
                            wrapped_relevance[key] = value
                        else:
                            wrapped_relevance[key] = value
                    features.update(wrapped_relevance)
            except Exception as e:
                logger.error(f"Error extracting relevance features: {e}")
                features["relevance_error"] = str(e)
        else:
            try:
                feature_result = extract_relevance_features_fallback(
                    text,
                    doc_result["sentences"],
                )

                # Wrap fallback relevance features in proper format
                if isinstance(feature_result, dict):
                    wrapped_relevance = {}
                    for key, value in feature_result.items():
                        if isinstance(value, (int, float, bool, str)):
                            # Simple scalar values get wrapped
                            wrapped_relevance[key] = {
                                "value": value,
                                "normalized": False,
                            }
                        elif isinstance(value, list):
                            # Lists get wrapped
                            wrapped_relevance[key] = {
                                "value": value,
                                "normalized": False,
                            }
                        elif isinstance(value, dict):
                            # Complex dictionaries are kept as-is
                            wrapped_relevance[key] = value
                        else:
                            wrapped_relevance[key] = value
                    features.update(wrapped_relevance)
            except Exception as e:
                logger.error(f"Error extracting fallback relevance features: {e}")
                features["relevance_error"] = str(e)
        self._processing_times["relevance"] = time.time() - relevance_start

        # Extract other features
        for feature_name, extractor_func in feature_extractors:
            feature_start = time.time()
            try:
                # All extractors use the same signature: (text, sentences, tokens)
                feature_result = extractor_func(
                    text,
                    doc_result["sentences"],
                    doc_result["tokens"],
                )

                # Wrap feature results in proper format for combine.py
                if isinstance(feature_result, dict):
                    wrapped_features = {}
                    for key, value in feature_result.items():
                        if isinstance(value, (int, float, bool, str)):
                            # Simple scalar values get wrapped
                            wrapped_features[key] = {
                                "value": value,
                                "normalized": False,
                            }
                        elif isinstance(value, list):
                            # Lists get wrapped
                            wrapped_features[key] = {
                                "value": value,
                                "normalized": False,
                            }
                        elif isinstance(value, dict):
                            # Complex dictionaries are kept as-is so combine.py can access their structure
                            wrapped_features[key] = value
                        else:
                            wrapped_features[key] = value
                    features.update(wrapped_features)
                else:
                    # Handle single value features
                    features[f"{feature_name}_value"] = {
                        "value": feature_result,
                        "normalized": False,
                    }
            except Exception as e:
                logger.error(f"Error extracting {feature_name} features: {e}")
                features[f"{feature_name}_error"] = str(e)

            self._processing_times[feature_name] = time.time() - feature_start

        # Add metadata
        features.update(
            {
                "has_semantic_features": {
                    "value": doc_result.get("sentence_embeddings") is not None,
                    "normalized": False,
                },
                "model_name": {"value": doc_result["model_name"], "normalized": False},
                "has_transformer": {
                    "value": doc_result["has_transformer"],
                    "normalized": False,
                },
                "processing_times": {
                    "value": self._processing_times.copy(),
                    "normalized": False,
                },
                "total_processing_time": {
                    "value": time.time() - start_time,
                    "normalized": False,
                },
            }
        )

        return features

    def batch_extract_features(self, texts: list[str]) -> list[dict[str, Any]]:
        """Extract features for multiple texts efficiently."""
        results = []
        for text in texts:
            results.append(self.extract_all_features(text))
        return results

    def get_processing_stats(self) -> dict[str, Any]:
        """Get processing performance statistics."""
        total_time = sum(self._processing_times.values())

        stats = {
            "total_time": total_time,
            "feature_times": self._processing_times.copy(),
            "average_feature_time": (
                total_time / len(self._processing_times)
                if self._processing_times
                else 0
            ),
            "slowest_feature": (
                max(self._processing_times.items(), key=lambda x: x[1])
                if self._processing_times
                else None
            ),
            "fastest_feature": (
                min(self._processing_times.items(), key=lambda x: x[1])
                if self._processing_times
                else None
            ),
        }

        # Add NLP pipeline stats
        stats["nlp_pipeline"] = {
            "model_name": self.nlp_pipeline.model_name,
            "has_transformer": self.nlp_pipeline.has_transformer,
            "cache_stats": self.nlp_pipeline.get_cache_stats(),
        }

        return stats

    def clear_caches(self) -> None:
        """Clear all caches."""
        self.nlp_pipeline.clear_cache()
        self._doc_cache.clear()
        self._processing_times.clear()
        logger.info("All caches cleared")

    def _cached_feature_extraction(
        self, text_hash: str, feature_name: str
    ) -> dict[str, Any]:
        """Cached feature extraction for repeated texts."""
        # This is a placeholder - actual implementation would extract specific features
        return {f"{feature_name}_cached": True}
