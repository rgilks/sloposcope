export default function About() {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl font-bold text-gray-900 mb-8">
            About Sloposcope
          </h1>

          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
            <h2 className="text-2xl font-semibold mb-4">What is Sloposcope?</h2>
            <p className="text-gray-700 mb-6">
              Sloposcope is a comprehensive tool for detecting AI-generated text
              patterns and measuring "slop" across multiple dimensions. It
              analyzes text using 11 different metrics to provide detailed
              insights into text quality and authenticity.
            </p>

            <h2 className="text-2xl font-semibold mb-4">Features</h2>
            <ul className="list-disc list-inside text-gray-700 mb-6 space-y-2">
              <li>
                <strong>11 Analysis Metrics:</strong> Density, repetition,
                coherence, templatedness, verbosity, tone, subjectivity,
                fluency, factuality, complexity, and relevance
              </li>
              <li>
                <strong>Domain-Specific Scoring:</strong> Optimized for general,
                news, and Q&A content
              </li>
              <li>
                <strong>Real-Time Analysis:</strong> Fast processing with
                detailed explanations
              </li>
              <li>
                <strong>Character-Level Spans:</strong> Identify specific
                problematic regions in text
              </li>
              <li>
                <strong>Confidence Scoring:</strong> Understand the reliability
                of analysis results
              </li>
            </ul>

            <h2 className="text-2xl font-semibold mb-4">How It Works</h2>
            <p className="text-gray-700 mb-4">
              Sloposcope uses advanced natural language processing techniques to
              analyze text across multiple dimensions:
            </p>
            <ol className="list-decimal list-inside text-gray-700 mb-6 space-y-2">
              <li>
                <strong>Text Processing:</strong> Tokenization, sentence
                segmentation, and linguistic analysis
              </li>
              <li>
                <strong>Feature Extraction:</strong> Computing metrics for each
                analysis dimension
              </li>
              <li>
                <strong>Score Normalization:</strong> Converting raw metrics to
                standardized scores
              </li>
              <li>
                <strong>Domain Weighting:</strong> Applying domain-specific
                weights for accurate scoring
              </li>
              <li>
                <strong>Result Synthesis:</strong> Combining metrics into an
                overall slop score
              </li>
            </ol>

            <h2 className="text-2xl font-semibold mb-4">Use Cases</h2>
            <ul className="list-disc list-inside text-gray-700 mb-6 space-y-2">
              <li>
                <strong>Content Editors:</strong> Identify and improve
                AI-generated content
              </li>
              <li>
                <strong>Quality Assurance:</strong> Ensure high-quality text
                output
              </li>
              <li>
                <strong>Research:</strong> Study patterns in AI-generated text
              </li>
              <li>
                <strong>Education:</strong> Teach about AI text detection
              </li>
              <li>
                <strong>Development:</strong> Integrate into content management
                systems
              </li>
            </ul>

            <h2 className="text-2xl font-semibold mb-4">Technical Details</h2>
            <p className="text-gray-700 mb-4">
              Built with modern web technologies and deployed on Cloudflare's
              global edge network:
            </p>
            <ul className="list-disc list-inside text-gray-700 mb-6 space-y-2">
              <li>
                <strong>Frontend:</strong> Next.js with React and Tailwind CSS
              </li>
              <li>
                <strong>Backend:</strong> Cloudflare Workers with JavaScript
              </li>
              <li>
                <strong>Deployment:</strong> Cloudflare Pages and Workers
              </li>
              <li>
                <strong>Performance:</strong> Global CDN with edge computing
              </li>
            </ul>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-blue-900 mb-2">
                Open Source
              </h3>
              <p className="text-blue-800">
                Sloposcope is open source under the Apache 2.0 license.
                <a
                  href="https://github.com/rgilks/sloposcope"
                  className="underline hover:no-underline"
                >
                  View the source code on GitHub
                </a>
                .
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
