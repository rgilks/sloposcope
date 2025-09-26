/**
 * Sloposcope Cloudflare Worker - JavaScript Implementation
 * This is a simplified version that provides the API structure
 * while we work on getting the full Python implementation working
 */

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const method = request.method;

    // CORS headers
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    };

    // Handle preflight requests
    if (method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    try {
      // Route requests
      if (method === "GET" && url.pathname.endsWith("/health")) {
        return await healthCheck(corsHeaders);
      } else if (method === "POST" && url.pathname.endsWith("/analyze")) {
        return await analyzeText(request, corsHeaders);
      } else if (method === "GET" && url.pathname.endsWith("/metrics")) {
        return await getMetricsInfo(corsHeaders);
      } else {
        return new Response("Not Found", {
          status: 404,
          headers: corsHeaders,
        });
      }
    } catch (error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
  },
};

async function healthCheck(corsHeaders) {
  return new Response(
    JSON.stringify({
      status: "healthy",
      service: "sloposcope",
      version: "1.0.0",
      implementation: "javascript-mock",
    }),
    {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    }
  );
}

async function analyzeText(request, corsHeaders) {
  try {
    const body = await request.json();
    const text = body.text || "";
    const domain = body.domain || "general";
    const language = body.language || "en";
    const explain = body.explain || false;
    const spans = body.spans || false;

    if (!text.trim()) {
      return new Response(JSON.stringify({ error: "No text provided" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Mock analysis - in a real implementation, this would call your Python code
    const mockMetrics = {
      density: { value: 0.4, combined_density: 0.4 },
      repetition: { value: 0.3, overall_repetition: 0.3 },
      templated: { value: 0.2, templated_score: 0.2 },
      coherence: { value: 0.5, coherence_score: 0.5 },
      verbosity: { value: 0.6, overall_verbosity: 0.6 },
      tone: { value: 0.4, tone_score: 0.4 },
      subjectivity: { value: 0.3 },
      fluency: { value: 0.7 },
      factuality: { value: 0.8 },
      complexity: { value: 0.5 },
      relevance: { value: 0.6 },
    };

    // Calculate overall slop score (simple average for demo)
    const slopScore =
      Object.values(mockMetrics).reduce(
        (sum, metric) => sum + metric.value,
        0
      ) / Object.keys(mockMetrics).length;
    const confidence = 0.8;

    const result = {
      version: "1.0",
      domain: domain,
      slop_score: slopScore,
      confidence: confidence,
      level: getSlopLevel(slopScore),
      metrics: mockMetrics,
      timings_ms: { total: 150, nlp: 50, features: 100 },
    };

    // Add spans if requested
    if (spans) {
      result.spans = [
        {
          start: 10,
          end: 25,
          type: "repetition",
          description: "Repeated phrase detected",
        },
      ];
    }

    // Add explanations if requested
    if (explain) {
      result.explanations = {
        density: "Information density and perplexity measures",
        relevance: "How well content matches prompt/references",
        coherence: "Entity continuity and topic flow",
        repetition: "N-gram repetition and compression",
        verbosity: "Wordiness and structural complexity",
        templated: "Templated phrases and boilerplate detection",
        tone: "Hedging, sycophancy, and tone analysis",
        subjectivity: "Bias and subjectivity detection",
        fluency: "Grammar and fluency assessment",
        factuality: "Factual accuracy proxy",
        complexity: "Lexical and syntactic complexity",
      };
    }

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error) {
    return new Response(
      JSON.stringify({ error: `Analysis failed: ${error.message}` }),
      {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  }
}

async function getMetricsInfo(corsHeaders) {
  const metricsInfo = {
    available_metrics: [
      {
        name: "density",
        description: "Information density and perplexity measures",
        range: [0, 1],
        lower_is_better: false,
      },
      {
        name: "relevance",
        description: "How well content matches prompt/references",
        range: [0, 1],
        lower_is_better: false,
      },
      {
        name: "coherence",
        description: "Entity continuity and topic flow",
        range: [0, 1],
        lower_is_better: false,
      },
      {
        name: "repetition",
        description: "N-gram repetition and compression",
        range: [0, 1],
        lower_is_better: true,
      },
      {
        name: "verbosity",
        description: "Wordiness and structural complexity",
        range: [0, 1],
        lower_is_better: true,
      },
      {
        name: "templated",
        description: "Templated phrases and boilerplate detection",
        range: [0, 1],
        lower_is_better: true,
      },
      {
        name: "tone",
        description: "Hedging, sycophancy, and tone analysis",
        range: [0, 1],
        lower_is_better: true,
      },
      {
        name: "subjectivity",
        description: "Bias and subjectivity detection",
        range: [0, 1],
        lower_is_better: true,
      },
      {
        name: "fluency",
        description: "Grammar and fluency assessment",
        range: [0, 1],
        lower_is_better: true,
      },
      {
        name: "factuality",
        description: "Factual accuracy proxy",
        range: [0, 1],
        lower_is_better: true,
      },
      {
        name: "complexity",
        description: "Lexical and syntactic complexity",
        range: [0, 1],
        lower_is_better: false,
      },
    ],
    domains: ["general", "news", "qa"],
    slop_levels: {
      Clean: "≤ 0.30",
      Watch: "0.30 - 0.55",
      Sloppy: "0.55 - 0.75",
      "High-Slop": "> 0.75",
    },
  };

  return new Response(JSON.stringify(metricsInfo), {
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}

function getSlopLevel(score) {
  if (score <= 0.3) return "Clean";
  else if (score <= 0.55) return "Watch";
  else if (score <= 0.75) return "Sloppy";
  else return "High-Slop";
}
