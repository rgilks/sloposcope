export function Footer() {
  return (
    <footer className="bg-gray-900 text-white py-8 mt-16">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-lg font-semibold mb-4">Sloposcope</h3>
            <p className="text-gray-400">
              A comprehensive tool for detecting AI-generated text patterns and
              measuring slop across multiple dimensions.
            </p>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Features</h3>
            <ul className="text-gray-400 space-y-2">
              <li>11 different metrics</li>
              <li>Domain-specific scoring</li>
              <li>Real-time analysis</li>
              <li>Detailed explanations</li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Resources</h3>
            <ul className="text-gray-400 space-y-2">
              <li>
                <a
                  href="https://github.com/rgilks/sloposcope"
                  className="hover:text-white"
                >
                  GitHub Repository
                </a>
              </li>
              <li>
                <a href="/api" className="hover:text-white">
                  API Documentation
                </a>
              </li>
            </ul>
          </div>
        </div>
        <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
          <p>&copy; 2024 Sloposcope. Open source under Apache 2.0 license.</p>
        </div>
      </div>
    </footer>
  );
}
