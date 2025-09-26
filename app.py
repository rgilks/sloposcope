"""
Sloposcope Web Application - FastAPI implementation for Fly.io deployment
"""

import json
import time
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sloplint.feature_extractor import FeatureExtractor
from sloplint.combine import combine_scores, normalize_scores, get_slop_level

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
feature_extractor: Optional[FeatureExtractor] = None


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
    metrics: Dict[str, Any]
    timings_ms: Dict[str, int]
    spans: Optional[list] = None
    explanations: Optional[Dict[str, str]] = None


def get_feature_extractor(language: str = "en") -> FeatureExtractor:
    """Get or create feature extractor instance."""
    global feature_extractor
    if feature_extractor is None or feature_extractor.language != language:
        feature_extractor = FeatureExtractor(language=language)
    return feature_extractor


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sloposcope - AI Text Analysis</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            .slop-clean { @apply bg-green-100 text-green-800; }
            .slop-watch { @apply bg-yellow-100 text-yellow-800; }
            .slop-sloppy { @apply bg-orange-100 text-orange-800; }
            .slop-high-slop { @apply bg-red-100 text-red-800; }
        </style>
    </head>
    <body class="bg-gray-50 min-h-screen">
        <div class="container mx-auto px-4 py-8 max-w-4xl">
            <!-- Header -->
            <header class="bg-white shadow-sm border-b rounded-lg mb-8">
                <div class="px-6 py-4">
                    <h1 class="text-3xl font-bold text-blue-600">Sloposcope</h1>
                    <p class="text-gray-600 mt-2">Detect AI-generated text patterns and measure "slop" across multiple dimensions</p>
                </div>
            </header>

            <!-- Main Content -->
            <div class="bg-white rounded-lg shadow-sm border p-6">
                <h2 class="text-xl font-semibold mb-4">Text Analysis</h2>
                
                <form id="analysisForm" class="space-y-4">
                    <div>
                        <label for="text" class="block text-sm font-medium text-gray-700 mb-2">Enter text to analyze</label>
                        <textarea 
                            id="text" 
                            name="text" 
                            rows="6" 
                            class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            placeholder="Paste your text here to analyze for AI slop patterns..."
                            required
                        ></textarea>
                        <div id="charCount" class="text-sm text-gray-500 mt-1">0 characters</div>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label for="domain" class="block text-sm font-medium text-gray-700 mb-2">Domain</label>
                            <select id="domain" name="domain" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                                <option value="general">General - General purpose text analysis</option>
                                <option value="news">News - News articles and journalism</option>
                                <option value="qa">Q&A - Question and answer content</option>
                            </select>
                        </div>
                        
                        <div>
                            <label for="language" class="block text-sm font-medium text-gray-700 mb-2">Language</label>
                            <select id="language" name="language" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                                <option value="en">English</option>
                                <option value="es">Spanish</option>
                                <option value="fr">French</option>
                                <option value="de">German</option>
                            </select>
                        </div>
                    </div>

                    <div class="bg-gray-50 rounded-lg p-4">
                        <h3 class="text-sm font-medium text-gray-700 mb-3">Analysis Options</h3>
                        <div class="space-y-2">
                            <label class="flex items-center">
                                <input type="checkbox" id="explain" name="explain" class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded">
                                <span class="ml-2 text-sm text-gray-700">Include detailed explanations</span>
                            </label>
                            <label class="flex items-center">
                                <input type="checkbox" id="spans" name="spans" class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded">
                                <span class="ml-2 text-sm text-gray-700">Show character spans for problematic regions</span>
                            </label>
                        </div>
                    </div>

                    <div class="flex space-x-4">
                        <button type="submit" id="analyzeBtn" class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center space-x-2">
                            <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                            </svg>
                            <span>Analyze Text</span>
                        </button>
                        <button type="button" id="clearBtn" class="bg-gray-200 hover:bg-gray-300 text-gray-800 font-medium py-2 px-4 rounded-lg transition-colors">
                            Clear
                        </button>
                    </div>
                </form>

                <!-- Results -->
                <div id="results" class="mt-8 hidden">
                    <h2 class="text-xl font-semibold mb-4">Analysis Results</h2>
                    <div id="resultsContent"></div>
                </div>

                <!-- Error -->
                <div id="error" class="mt-8 hidden">
                    <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                        <div class="flex">
                            <div class="flex-shrink-0">
                                <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                                </svg>
                            </div>
                            <div class="ml-3">
                                <h3 class="text-sm font-medium text-red-800">Error</h3>
                                <div id="errorMessage" class="mt-2 text-sm text-red-700"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Footer -->
            <footer class="bg-gray-900 text-white py-8 mt-16 rounded-lg">
                <div class="px-6">
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
                        <div>
                            <h3 class="text-lg font-semibold mb-4">Sloposcope</h3>
                            <p class="text-gray-400">A comprehensive tool for detecting AI-generated text patterns and measuring slop across multiple dimensions.</p>
                        </div>
                        <div>
                            <h3 class="text-lg font-semibold mb-4">Features</h3>
                            <ul class="text-gray-400 space-y-2">
                                <li>11 different metrics</li>
                                <li>Domain-specific scoring</li>
                                <li>Real-time analysis</li>
                                <li>Detailed explanations</li>
                            </ul>
                        </div>
                        <div>
                            <h3 class="text-lg font-semibold mb-4">Resources</h3>
                            <ul class="text-gray-400 space-y-2">
                                <li><a href="https://github.com/rgilks/sloposcope" class="hover:text-white">GitHub Repository</a></li>
                                <li><a href="/docs" class="hover:text-white">API Documentation</a></li>
                            </ul>
                        </div>
                    </div>
                    <div class="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
                        <p>&copy; 2024 Sloposcope. Open source under Apache 2.0 license.</p>
                    </div>
                </div>
            </footer>
        </div>

        <script>
            // Character counter
            const textArea = document.getElementById('text');
            const charCount = document.getElementById('charCount');
            
            textArea.addEventListener('input', () => {
                charCount.textContent = textArea.value.length + ' characters';
            });

            // Form submission
            const form = document.getElementById('analysisForm');
            const results = document.getElementById('results');
            const error = document.getElementById('error');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const clearBtn = document.getElementById('clearBtn');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData(form);
                const data = {
                    text: formData.get('text'),
                    domain: formData.get('domain'),
                    language: formData.get('language'),
                    explain: formData.get('explain') === 'on',
                    spans: formData.get('spans') === 'on'
                };

                if (!data.text.trim()) {
                    showError('Please enter some text to analyze');
                    return;
                }

                analyzeBtn.disabled = true;
                analyzeBtn.innerHTML = '<svg class="animate-spin h-4 w-4" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg><span class="ml-2">Analyzing...</span>';
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || 'Analysis failed');
                    }

                    const result = await response.json();
                    showResults(result);
                } catch (err) {
                    showError(err.message);
                } finally {
                    analyzeBtn.disabled = false;
                    analyzeBtn.innerHTML = '<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg><span class="ml-2">Analyze Text</span>';
                }
            });

            clearBtn.addEventListener('click', () => {
                form.reset();
                results.classList.add('hidden');
                error.classList.add('hidden');
                charCount.textContent = '0 characters';
            });

            function showResults(result) {
                error.classList.add('hidden');
                
                const slopColor = getSlopColor(result.level);
                const slopPercentage = (result.slop_score * 100).toFixed(1);
                
                const resultsHTML = `
                    <div class="space-y-6">
                        <!-- Overall Score -->
                        <div class="bg-gray-50 rounded-lg p-6">
                            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                                <div class="text-center">
                                    <h3 class="text-sm font-medium text-gray-500 mb-2">Slop Score</h3>
                                    <div class="text-3xl font-bold ${slopColor.replace('slop-', 'text-')}">${slopPercentage}%</div>
                                    <div class="w-full bg-gray-200 rounded-full h-2 mt-2">
                                        <div class="h-2 rounded-full ${slopColor.replace('slop-', 'bg-')}" style="width: ${slopPercentage}%"></div>
                                    </div>
                                    <div class="text-sm text-gray-600 mt-2">Confidence: ${(result.confidence * 100).toFixed(1)}%</div>
                                </div>
                                
                                <div class="text-center">
                                    <h3 class="text-sm font-medium text-gray-500 mb-2">Level</h3>
                                    <div class="text-2xl font-semibold">
                                        <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${slopColor}">${result.level}</span>
                                    </div>
                                </div>
                                
                                <div class="text-center">
                                    <h3 class="text-sm font-medium text-gray-500 mb-2">Processing Time</h3>
                                    <div class="text-2xl font-semibold text-gray-900">${result.timings_ms.total}ms</div>
                                </div>
                            </div>
                        </div>

                        <!-- Metrics -->
                        <div>
                            <h3 class="text-lg font-semibold mb-4">Per-Axis Metrics</h3>
                            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                ${Object.entries(result.metrics).map(([name, data]) => {
                                    const score = data.value;
                                    const percentage = (score * 100).toFixed(1);
                                    const status = getStatus(score);
                                    return `
                                        <div class="bg-white border border-gray-200 rounded-lg p-4">
                                            <div class="flex items-center justify-between mb-2">
                                                <h4 class="text-sm font-medium text-gray-700 capitalize">${name.replace('_', ' ')}</h4>
                                                <span class="text-lg">${status.icon}</span>
                                            </div>
                                            <div class="text-2xl font-bold ${status.color}">${percentage}%</div>
                                            <div class="text-xs text-gray-500 mt-1">${status.text}</div>
                                            <div class="w-full bg-gray-200 rounded-full h-1 mt-2">
                                                <div class="h-1 rounded-full ${status.bgColor}" style="width: ${percentage}%"></div>
                                            </div>
                                        </div>
                                    `;
                                }).join('')}
                            </div>
                        </div>

                        ${result.explanations ? `
                            <div>
                                <h3 class="text-lg font-semibold mb-4">Explanations</h3>
                                <div class="space-y-3">
                                    ${Object.entries(result.explanations).map(([metric, explanation]) => `
                                        <div class="border-l-4 border-blue-200 pl-4">
                                            <h4 class="font-medium text-gray-900 capitalize">${metric}</h4>
                                            <p class="text-gray-600 text-sm">${explanation}</p>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}

                        ${result.spans && result.spans.length > 0 ? `
                            <div>
                                <h3 class="text-lg font-semibold mb-4">Problematic Spans</h3>
                                <div class="space-y-3">
                                    ${result.spans.map(span => `
                                        <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                                            <div class="flex items-center justify-between mb-2">
                                                <span class="font-medium capitalize">${span.type.replace('_', ' ')}</span>
                                                <span class="text-sm text-gray-500">Characters ${span.start}-${span.end}</span>
                                            </div>
                                            <p class="text-sm text-gray-700">${span.description}</p>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}
                    </div>
                `;
                
                document.getElementById('resultsContent').innerHTML = resultsHTML;
                results.classList.remove('hidden');
            }

            function showError(message) {
                results.classList.add('hidden');
                document.getElementById('errorMessage').textContent = message;
                error.classList.remove('hidden');
            }

            function getSlopColor(level) {
                switch(level.toLowerCase()) {
                    case 'clean': return 'slop-clean';
                    case 'watch': return 'slop-watch';
                    case 'sloppy': return 'slop-sloppy';
                    case 'high-slop': return 'slop-high-slop';
                    default: return 'slop-watch';
                }
            }

            function getStatus(score) {
                if (score <= 0.3) {
                    return { icon: '✅', text: 'Good', color: 'text-green-600', bgColor: 'bg-green-500' };
                } else if (score <= 0.55) {
                    return { icon: '⚠️', text: 'Watch', color: 'text-yellow-600', bgColor: 'bg-yellow-500' };
                } else if (score <= 0.75) {
                    return { icon: '🔶', text: 'Sloppy', color: 'text-orange-600', bgColor: 'bg-orange-500' };
                } else {
                    return { icon: '❌', text: 'High-Slop', color: 'text-red-600', bgColor: 'bg-red-500' };
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


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
        extractor = get_feature_extractor(request.language)

        # Extract features
        feature_start = time.time()
        raw_features = extractor.extract_all_features(request.text)
        feature_time = int((time.time() - feature_start) * 1000)

        # Extract spans if requested
        spans_collection = None
        if request.spans:
            spans_collection = extractor.extract_spans(request.text)

        # Convert features to metrics format
        metrics = {}
        for feature_name, feature_data in raw_features.items():
            if isinstance(feature_data, dict) and "value" in feature_data:
                metrics[feature_name] = feature_data
            else:
                # Extract main score based on feature type
                score_keys = {
                    "density": "combined_density",
                    "repetition": "overall_repetition",
                    "templated": "templated_score",
                    "coherence": "coherence_score",
                    "verbosity": "overall_verbosity",
                    "tone": "tone_score",
                    "relevance": "relevance_score",
                    "factuality": "factuality_score",
                    "subjectivity": "subjectivity_score",
                    "fluency": "fluency_score",
                    "complexity": "complexity_score",
                }

                key = score_keys.get(feature_name, "value")
                value = feature_data.get(key, feature_data.get("value", 0.5))

                metrics[feature_name] = {"value": value, **feature_data}

        # Normalize and combine scores
        normalized_metrics = normalize_scores(metrics, request.domain)
        slop_score, confidence = combine_scores(normalized_metrics, request.domain)

        total_time = int((time.time() - start_time) * 1000)

        # Create response
        result = AnalysisResponse(
            version="1.0",
            domain=request.domain,
            slop_score=slop_score,
            confidence=confidence,
            level=get_slop_level(slop_score),
            metrics=normalized_metrics,
            timings_ms={"total": total_time, "nlp": 50, "features": feature_time},
        )

        # Add spans if requested
        if request.spans and spans_collection:
            result.spans = spans_collection.to_dict_list()

        # Add explanations if requested
        if request.explain:
            result.explanations = {
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

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


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
