"use client";

import { AnalysisResult } from "./TextAnalyzer";
import { MetricCard } from "./MetricCard";
import { SlopScoreDisplay } from "./SlopScoreDisplay";
import { SpansDisplay } from "./SpansDisplay";

interface ResultsDisplayProps {
  result: AnalysisResult;
}

export function ResultsDisplay({ result }: ResultsDisplayProps) {
  const getSlopColor = (level: string) => {
    switch (level.toLowerCase()) {
      case "clean":
        return "slop-clean";
      case "watch":
        return "slop-watch";
      case "sloppy":
        return "slop-sloppy";
      case "high-slop":
        return "slop-high-slop";
      default:
        return "slop-watch";
    }
  };

  return (
    <div className="space-y-6">
      {/* Overall Score */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold mb-4">Analysis Results</h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <SlopScoreDisplay
            score={result.slop_score}
            level={result.level}
            confidence={result.confidence}
          />

          <div className="metric-card">
            <h3 className="text-sm font-medium text-gray-500 mb-2">Domain</h3>
            <p className="text-lg font-semibold capitalize">{result.domain}</p>
          </div>

          <div className="metric-card">
            <h3 className="text-sm font-medium text-gray-500 mb-2">
              Processing Time
            </h3>
            <p className="text-lg font-semibold">{result.timings_ms.total}ms</p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-600">Overall Slop Level:</span>
          <span className={`slop-indicator ${getSlopColor(result.level)}`}>
            {result.level}
          </span>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold mb-4">Per-Axis Metrics</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(result.metrics).map(([name, data]) => (
            <MetricCard key={name} name={name} value={data.value} data={data} />
          ))}
        </div>
      </div>

      {/* Spans Display */}
      {result.spans && result.spans.length > 0 && (
        <SpansDisplay spans={result.spans} />
      )}

      {/* Explanations */}
      {result.explanations && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-semibold mb-4">Metric Explanations</h2>
          <div className="space-y-3">
            {Object.entries(result.explanations).map(
              ([metric, explanation]) => (
                <div
                  key={metric}
                  className="border-l-4 border-primary-200 pl-4"
                >
                  <h3 className="font-medium text-gray-900 capitalize">
                    {metric}
                  </h3>
                  <p className="text-gray-600 text-sm">{explanation}</p>
                </div>
              )
            )}
          </div>
        </div>
      )}
    </div>
  );
}
