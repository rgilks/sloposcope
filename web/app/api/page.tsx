export default function API() {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl font-bold text-gray-900 mb-8">
            API Documentation
          </h1>

          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
            <h2 className="text-2xl font-semibold mb-4">Sloposcope API</h2>
            <p className="text-gray-700 mb-6">
              The Sloposcope API provides programmatic access to text analysis
              capabilities. All endpoints are hosted on Cloudflare Workers for
              fast, global performance.
            </p>

            <div className="bg-gray-50 rounded-lg p-4 mb-6">
              <h3 className="text-lg font-semibold mb-2">Base URL</h3>
              <code className="text-sm bg-white px-2 py-1 rounded border">
                https://sloposcope-prod.rob-gilks.workers.dev
              </code>
            </div>

            <h2 className="text-2xl font-semibold mb-4">Endpoints</h2>

            <div className="space-y-6">
              <div className="border border-gray-200 rounded-lg p-6">
                <div className="flex items-center mb-3">
                  <span className="bg-green-100 text-green-800 text-xs font-medium px-2.5 py-0.5 rounded mr-3">
                    GET
                  </span>
                  <code className="text-lg font-mono">/health</code>
                </div>
                <p className="text-gray-700 mb-3">
                  Check the health status of the API service.
                </p>

                <h4 className="font-semibold mb-2">Response</h4>
                <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
                  {`{
  "status": "healthy",
  "service": "sloposcope",
  "version": "1.0.0",
  "implementation": "javascript-mock"
}`}
                </pre>
              </div>

              <div className="border border-gray-200 rounded-lg p-6">
                <div className="flex items-center mb-3">
                  <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded mr-3">
                    POST
                  </span>
                  <code className="text-lg font-mono">/analyze</code>
                </div>
                <p className="text-gray-700 mb-3">
                  Analyze text for AI slop patterns and return detailed metrics.
                </p>

                <h4 className="font-semibold mb-2">Request Body</h4>
                <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
                  {`{
  "text": "Text to analyze",
  "domain": "general", // optional: "general", "news", "qa"
  "language": "en", // optional: "en", "es", "fr", "de"
  "explain": true, // optional: include explanations
  "spans": true // optional: include character spans
}`}
                </pre>

                <h4 className="font-semibold mb-2 mt-4">Response</h4>
                <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
                  {`{
  "version": "1.0",
  "domain": "general",
  "slop_score": 0.482,
  "confidence": 0.8,
  "level": "Watch",
  "metrics": {
    "density": {"value": 0.4},
    "repetition": {"value": 0.3},
    "templated": {"value": 0.2},
    "coherence": {"value": 0.5},
    "verbosity": {"value": 0.6},
    "tone": {"value": 0.4},
    "subjectivity": {"value": 0.3},
    "fluency": {"value": 0.7},
    "factuality": {"value": 0.8},
    "complexity": {"value": 0.5},
    "relevance": {"value": 0.6}
  },
  "timings_ms": {"total": 150, "nlp": 50, "features": 100},
  "explanations": {...}, // if explain=true
  "spans": [...] // if spans=true
}`}
                </pre>
              </div>

              <div className="border border-gray-200 rounded-lg p-6">
                <div className="flex items-center mb-3">
                  <span className="bg-green-100 text-green-800 text-xs font-medium px-2.5 py-0.5 rounded mr-3">
                    GET
                  </span>
                  <code className="text-lg font-mono">/metrics</code>
                </div>
                <p className="text-gray-700 mb-3">
                  Get information about available analysis metrics.
                </p>

                <h4 className="font-semibold mb-2">Response</h4>
                <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
                  {`{
  "available_metrics": [
    {
      "name": "density",
      "description": "Information density and perplexity measures",
      "range": [0, 1],
      "lower_is_better": false
    },
    // ... other metrics
  ],
  "domains": ["general", "news", "qa"],
  "slop_levels": {
    "Clean": "≤ 0.30",
    "Watch": "0.30 - 0.55",
    "Sloppy": "0.55 - 0.75",
    "High-Slop": "> 0.75"
  }
}`}
                </pre>
              </div>
            </div>

            <h2 className="text-2xl font-semibold mb-4 mt-8">Usage Examples</h2>

            <div className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2">cURL</h4>
                <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
                  {`curl -X POST https://sloposcope-prod.rob-gilks.workers.dev/analyze \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "Your text here",
    "domain": "general",
    "explain": true
  }'`}
                </pre>
              </div>

              <div>
                <h4 className="font-semibold mb-2">JavaScript</h4>
                <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
                  {`const response = await fetch('https://sloposcope-prod.rob-gilks.workers.dev/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'Your text here',
    domain: 'general',
    explain: true
  })
});

const result = await response.json();
console.log('Slop Score:', result.slop_score);`}
                </pre>
              </div>

              <div>
                <h4 className="font-semibold mb-2">Python</h4>
                <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
                  {`import requests

response = requests.post(
    'https://sloposcope-prod.rob-gilks.workers.dev/analyze',
    json={
        'text': 'Your text here',
        'domain': 'general',
        'explain': True
    }
)

result = response.json()
print(f"Slop Score: {result['slop_score']}")`}
                </pre>
              </div>
            </div>

            <h2 className="text-2xl font-semibold mb-4 mt-8">Rate Limits</h2>
            <p className="text-gray-700 mb-4">
              Currently, there are no strict rate limits, but please use the API
              responsibly. For high-volume usage, consider implementing your own
              caching and rate limiting.
            </p>

            <h2 className="text-2xl font-semibold mb-4">Error Handling</h2>
            <p className="text-gray-700 mb-4">
              The API returns standard HTTP status codes:
            </p>
            <ul className="list-disc list-inside text-gray-700 space-y-1">
              <li>
                <strong>200:</strong> Success
              </li>
              <li>
                <strong>400:</strong> Bad Request (invalid input)
              </li>
              <li>
                <strong>404:</strong> Not Found
              </li>
              <li>
                <strong>500:</strong> Internal Server Error
              </li>
            </ul>

            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mt-6">
              <h3 className="text-lg font-semibold text-yellow-900 mb-2">
                Note
              </h3>
              <p className="text-yellow-800">
                This API currently uses a mock implementation for demonstration
                purposes. The actual sloposcope analysis engine will be
                integrated in future updates.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
