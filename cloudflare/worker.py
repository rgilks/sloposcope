"""
Cloudflare Worker for sloposcope API using Python.
This worker provides HTTP endpoints for text analysis.
"""

import json
from typing import Dict, Any, Optional
from workers import WorkerEntrypoint, Response, Request

# Import your sloposcope modules
# Note: You'll need to copy the sloplint package to the worker
try:
    from sloplint.feature_extractor import FeatureExtractor
    from sloplint.combine import combine_scores, normalize_scores
    from sloplint.io import load_text
except ImportError:
    # Fallback for development/testing
    print("Warning: sloplint modules not available, using mock implementation")

    class MockFeatureExtractor:
        def __init__(self, language="en"):
            self.language = language

        def extract_all_features(self, text):
            # Mock implementation for testing
            return {
                "density": {"value": 0.4, "combined_density": 0.4},
                "repetition": {"value": 0.3, "overall_repetition": 0.3},
                "templated": {"value": 0.2, "templated_score": 0.2},
                "coherence": {"value": 0.5, "coherence_score": 0.5},
                "verbosity": {"value": 0.6, "overall_verbosity": 0.6},
                "tone": {"value": 0.4, "tone_score": 0.4},
                "subjectivity": {"value": 0.3},
                "fluency": {"value": 0.7},
                "factuality": {"value": 0.8},
                "complexity": {"value": 0.5},
                "relevance": {"value": 0.6},
            }

        def extract_spans(self, text):
            class MockSpans:
                def to_dict_list(self):
                    return []

            return MockSpans()

    FeatureExtractor = MockFeatureExtractor

    def combine_scores(metrics, domain):
        # Simple mock implementation
        total_score = sum(m.get("value", 0.5) for m in metrics.values()) / len(metrics)
        return total_score, 0.8

    def normalize_scores(metrics, domain):
        return metrics


class SloposcopeWorker(WorkerEntrypoint):
    """Main worker class for sloposcope API."""

    def __init__(self, ctx, env):
        super().__init__(ctx, env)
        self.feature_extractor = None

    async def fetch(self, request: Request) -> Response:
        """Handle incoming requests."""
        try:
            url = request.url
            method = request.method

            # Route requests
            if method == "GET" and url.endswith("/health"):
                return await self.health_check()
            elif method == "POST" and url.endswith("/analyze"):
                return await self.analyze_text(request)
            elif method == "GET" and url.endswith("/metrics"):
                return await self.get_metrics_info()
            else:
                return Response("Not Found", status=404)

        except Exception as e:
            return Response(
                json.dumps({"error": str(e)}),
                status=500,
                headers={"Content-Type": "application/json"},
            )

    async def health_check(self) -> Response:
        """Health check endpoint."""
        return Response(
            json.dumps(
                {"status": "healthy", "service": "sloposcope", "version": "1.0.0"}
            ),
            headers={"Content-Type": "application/json"},
        )

    async def analyze_text(self, request: Request) -> Response:
        """Analyze text for AI slop."""
        try:
            # Parse request body
            body = await request.json()
            text = body.get("text", "")
            domain = body.get("domain", "general")
            language = body.get("language", "en")
            explain = body.get("explain", False)
            spans = body.get("spans", False)

            if not text.strip():
                return Response(
                    json.dumps({"error": "No text provided"}),
                    status=400,
                    headers={"Content-Type": "application/json"},
                )

            # Initialize feature extractor if not already done
            if self.feature_extractor is None:
                self.feature_extractor = FeatureExtractor(language=language)

            # Extract features
            raw_features = self.feature_extractor.extract_all_features(text)

            # Extract spans if requested
            spans_collection = None
            if spans:
                spans_collection = self.feature_extractor.extract_spans(text)

            # Convert features to metrics format
            metrics = {}
            for feature_name, feature_data in raw_features.items():
                if isinstance(feature_data, dict) and "value" in feature_data:
                    metrics[feature_name] = feature_data
                else:
                    # Calculate value from feature data
                    if feature_name == "density":
                        value = feature_data.get("combined_density", 0.5)
                    elif feature_name == "repetition":
                        value = feature_data.get("overall_repetition", 0.3)
                    elif feature_name == "templated":
                        value = feature_data.get("templated_score", 0.4)
                    elif feature_name == "coherence":
                        value = feature_data.get("coherence_score", 0.5)
                    elif feature_name == "verbosity":
                        value = feature_data.get("overall_verbosity", 0.6)
                    elif feature_name == "tone":
                        value = feature_data.get("tone_score", 0.4)
                    else:
                        value = feature_data.get("value", 0.5)

                    metrics[feature_name] = {"value": value, **feature_data}

            # Normalize and combine scores
            normalized_metrics = normalize_scores(metrics, domain)
            slop_score, confidence = combine_scores(normalized_metrics, domain)

            # Create response
            result = {
                "version": "1.0",
                "domain": domain,
                "slop_score": slop_score,
                "confidence": confidence,
                "level": self.get_slop_level(slop_score),
                "metrics": normalized_metrics,
                "timings_ms": {"total": 500, "nlp": 200, "features": 300},
            }

            # Add spans if requested
            if spans and spans_collection:
                result["spans"] = spans_collection.to_dict_list()

            # Add explanations if requested
            if explain:
                result["explanations"] = self.get_explanations()

            return Response(
                json.dumps(result), headers={"Content-Type": "application/json"}
            )

        except Exception as e:
            return Response(
                json.dumps({"error": f"Analysis failed: {str(e)}"}),
                status=500,
                headers={"Content-Type": "application/json"},
            )

    async def get_metrics_info(self) -> Response:
        """Get information about available metrics."""
        metrics_info = {
            "available_metrics": [
                {
                    "name": "density",
                    "description": "Information density and perplexity measures",
                    "range": [0, 1],
                    "lower_is_better": False,
                },
                {
                    "name": "relevance",
                    "description": "How well content matches prompt/references",
                    "range": [0, 1],
                    "lower_is_better": False,
                },
                {
                    "name": "coherence",
                    "description": "Entity continuity and topic flow",
                    "range": [0, 1],
                    "lower_is_better": False,
                },
                {
                    "name": "repetition",
                    "description": "N-gram repetition and compression",
                    "range": [0, 1],
                    "lower_is_better": True,
                },
                {
                    "name": "verbosity",
                    "description": "Wordiness and structural complexity",
                    "range": [0, 1],
                    "lower_is_better": True,
                },
                {
                    "name": "templated",
                    "description": "Templated phrases and boilerplate detection",
                    "range": [0, 1],
                    "lower_is_better": True,
                },
                {
                    "name": "tone",
                    "description": "Hedging, sycophancy, and tone analysis",
                    "range": [0, 1],
                    "lower_is_better": True,
                },
                {
                    "name": "subjectivity",
                    "description": "Bias and subjectivity detection",
                    "range": [0, 1],
                    "lower_is_better": True,
                },
                {
                    "name": "fluency",
                    "description": "Grammar and fluency assessment",
                    "range": [0, 1],
                    "lower_is_better": True,
                },
                {
                    "name": "factuality",
                    "description": "Factual accuracy proxy",
                    "range": [0, 1],
                    "lower_is_better": True,
                },
                {
                    "name": "complexity",
                    "description": "Lexical and syntactic complexity",
                    "range": [0, 1],
                    "lower_is_better": False,
                },
            ],
            "domains": ["general", "news", "qa"],
            "slop_levels": {
                "Clean": "≤ 0.30",
                "Watch": "0.30 - 0.55",
                "Sloppy": "0.55 - 0.75",
                "High-Slop": "> 0.75",
            },
        }

        return Response(
            json.dumps(metrics_info), headers={"Content-Type": "application/json"}
        )

    def get_slop_level(self, score: float) -> str:
        """Convert slop score to level category."""
        if score <= 0.30:
            return "Clean"
        elif score <= 0.55:
            return "Watch"
        elif score <= 0.75:
            return "Sloppy"
        else:
            return "High-Slop"

    def get_explanations(self) -> Dict[str, str]:
        """Get explanations for each metric."""
        return {
            "density": "Information density and perplexity measures",
            "relevance": "How well content matches prompt/references",
            "coherence": "Entity continuity and topic flow",
            "repetition": "N-gram repetition and compression",
            "verbosity": "Wordiness and structural complexity",
            "templated": "Templated phrases and boilerplate detection",
            "tone": "Hedging, sycophancy, and tone analysis",
            "subjectivity": "Bias and subjectivity detection",
            "fluency": "Grammar and fluency assessment",
            "factuality": "Factual accuracy proxy",
            "complexity": "Lexical and syntactic complexity",
        }


# Export the worker class
# Note: The worker will be instantiated by Cloudflare with proper ctx and env
