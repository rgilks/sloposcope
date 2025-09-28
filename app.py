"""
Sloposcope Web Application - FastAPI implementation for Fly.io deployment
"""

import os
import time
from pathlib import Path
from typing import Any

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sloplint.combine import combine_scores, normalize_scores
from sloplint.feature_extractor import FeatureExtractor
from sloplint.spans import SpanCollection

app = FastAPI(
    title="Sloposcope API", description="AI Slop Detection API", version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global feature extractor instance
feature_extractor: FeatureExtractor | None = None


class AnalysisRequest(BaseModel):
    text: str
    domain: str = "general"
    language: str = "en"
    explain: bool = False
    spans: bool = False


class AnalysisResponse(BaseModel):
    version: str
    domain: str
    slop_score: float
    confidence: float
    level: str
    metrics: dict[str, Any]
    timings_ms: dict[str, int]
    spans: list | None = None
    explanations: dict[str, str] | None = None


def get_feature_extractor() -> FeatureExtractor:
    """Get or create feature extractor instance."""
    global feature_extractor
    if feature_extractor is None:
        feature_extractor = FeatureExtractor()
    return feature_extractor


def get_slop_level(score: float) -> str:
    """Convert slop score to level category."""
    if score <= 0.50:
        return "Clean"
    elif score <= 0.70:
        return "Watch"
    elif score <= 0.85:
        return "Sloppy"
    else:
        return "High-Slop"


def get_index_html() -> str:
    """Load the HTML template from file."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback to inline HTML if template file doesn't exist
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Sloposcope - AI Text Analysis</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-50 min-h-screen">
            <div class="container mx-auto px-4 py-8 max-w-4xl">
                <div class="bg-white rounded-lg shadow-sm border p-6">
                    <h1 class="text-3xl font-bold text-blue-600 mb-4">Sloposcope</h1>
                    <p class="text-gray-600 mb-6">Detect AI-generated text patterns and measure "slop" across multiple dimensions</p>

                    <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                        <h2 class="text-lg font-semibold text-yellow-800 mb-2">Template File Missing</h2>
                        <p class="text-yellow-700">The HTML template file could not be found. Please ensure templates/index.html exists.</p>
                    </div>

                    <div class="mt-6">
                        <h3 class="text-lg font-semibold mb-4">Quick Test</h3>
                        <form class="space-y-4">
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-2">Test the API</label>
                                <textarea
                                    name="text"
                                    rows="3"
                                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                    placeholder="Enter text to test..."
                                >This is a test message for the Sloposcope API.</textarea>
                            </div>
                            <button type="button" onclick="testAPI()" class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors">
                                Test API
                            </button>
                        </form>
                        <div id="testResult" class="mt-4 hidden"></div>
                    </div>
                </div>
            </div>

            <script>
                async function testAPI() {
                    const text = document.querySelector('textarea[name="text"]').value;
                    const resultDiv = document.getElementById('testResult');

                    try {
                        const response = await fetch('/analyze', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ text, domain: 'general' })
                        });

                        if (!response.ok) {
                            throw new Error('API request failed');
                        }

                        const result = await response.json();
                        resultDiv.innerHTML = `
                            <div class="bg-green-50 border border-green-200 rounded-lg p-4">
                                <h4 class="font-medium text-green-800 mb-2">✅ API Test Successful!</h4>
                                <p class="text-green-700 text-sm">Slop Score: ${result.slop_score.toFixed(3)} (${result.level})</p>
                                <p class="text-green-700 text-sm">Confidence: ${result.confidence.toFixed(3)}</p>
                            </div>
                        `;
                    } catch (error) {
                        resultDiv.innerHTML = `
                            <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                                <h4 class="font-medium text-red-800 mb-2">❌ API Test Failed</h4>
                                <p class="text-red-700 text-sm">Error: ${error.message}</p>
                            </div>
                        `;
                    }
                    resultDiv.classList.remove('hidden');
                }
            </script>
        </body>
        </html>
        """


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    return HTMLResponse(content=get_index_html())


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "sloposcope",
        "version": "1.0.0",
        "implementation": "python-fastapi",
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    """Analyze text for AI slop patterns."""
    start_time = time.time()

    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="No text provided")

        # Get feature extractor
        extractor = get_feature_extractor()

        # Extract all features
        analysis_start = time.time()
        raw_features = extractor.extract_all_features(request.text)
        analysis_time = int((time.time() - analysis_start) * 1000)

        # Convert features to metrics format
        metrics = {}
        for feature_name, feature_data in raw_features.items():
            if isinstance(feature_data, dict):
                metrics[feature_name] = feature_data
            else:
                metrics[feature_name] = {
                    "value": float(feature_data)
                    if isinstance(feature_data, (int, float))
                    else 0.5
                }

        # Normalize and combine scores
        normalized_metrics = normalize_scores(metrics, request.domain)
        slop_score, confidence = combine_scores(normalized_metrics, request.domain)
        slop_level = get_slop_level(slop_score)

        # Create spans collection (placeholder for now)
        spans_collection = SpanCollection()

        total_time = int((time.time() - start_time) * 1000)

        # Create response
        response = AnalysisResponse(
            version="1.0",
            domain=request.domain,
            slop_score=slop_score,
            confidence=confidence,
            level=slop_level,
            metrics=normalized_metrics,
            timings_ms={"total": total_time, "analysis": analysis_time},
        )

        # Add explanations if requested
        if request.explain:
            response.explanations = {
                "density": "Information density and perplexity measures",
                "relevance": "How well content matches prompt/references",
                "coherence": "Entity continuity and topic flow",
                "repetition": "N-gram repetition and compression",
                "verbosity": "Wordiness and structural complexity",
                "tone": "Jargon and awkward phrasing detection",
                "templated": "Formulaic and boilerplate patterns",
                "factuality": "Accuracy and truthfulness measures",
                "subjectivity": "Bias and subjective language detection",
                "fluency": "Grammar and natural language patterns",
                "complexity": "Text complexity and readability measures",
            }

        # Add spans if requested
        if request.spans:
            response.spans = spans_collection.to_dict_list()

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}") from e


@app.get("/metrics")
async def get_metrics_info():
    """Get information about available metrics."""
    return {
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
