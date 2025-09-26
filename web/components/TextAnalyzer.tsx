"use client";

import { useState } from "react";
import { AnalyzeButton } from "./AnalyzeButton";
import { ResultsDisplay } from "./ResultsDisplay";
import { DomainSelector } from "./DomainSelector";
import { OptionsPanel } from "./OptionsPanel";

export interface AnalysisResult {
  version: string;
  domain: string;
  slop_score: number;
  confidence: number;
  level: string;
  metrics: Record<string, { value: number; [key: string]: any }>;
  spans?: Array<{
    start: number;
    end: number;
    type: string;
    description: string;
  }>;
  explanations?: Record<string, string>;
  timings_ms: { total: number; nlp: number; features: number };
}

export function TextAnalyzer() {
  const [text, setText] = useState("");
  const [domain, setDomain] = useState("general");
  const [language, setLanguage] = useState("en");
  const [explain, setExplain] = useState(false);
  const [spans, setSpans] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError("Please enter some text to analyze");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(
        "https://sloposcope-prod.rob-gilks.workers.dev/analyze",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text,
            domain,
            language,
            explain,
            spans,
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Analysis failed");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText("");
    setResult(null);
    setError(null);
  };

  return (
    <div className="space-y-6">
      {/* Input Section */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold mb-4">Text Input</h2>

        <div className="space-y-4">
          <div>
            <label
              htmlFor="text-input"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
              Enter text to analyze
            </label>
            <textarea
              id="text-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste your text here to analyze for AI slop patterns..."
              className="w-full h-32 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 resize-none"
            />
            <div className="text-sm text-gray-500 mt-1">
              {text.length} characters
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <DomainSelector value={domain} onChange={setDomain} />

            <div>
              <label
                htmlFor="language"
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Language
              </label>
              <select
                id="language"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                <option value="en">English</option>
                <option value="es">Spanish</option>
                <option value="fr">French</option>
                <option value="de">German</option>
              </select>
            </div>
          </div>

          <OptionsPanel
            explain={explain}
            setExplain={setExplain}
            spans={spans}
            setSpans={setSpans}
          />

          <div className="flex space-x-4">
            <AnalyzeButton
              onClick={handleAnalyze}
              loading={loading}
              disabled={!text.trim()}
            />
            <button
              onClick={handleClear}
              className="btn-secondary"
              disabled={loading}
            >
              Clear
            </button>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg
                className="h-5 w-5 text-red-400"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Error</h3>
              <div className="mt-2 text-sm text-red-700">{error}</div>
            </div>
          </div>
        </div>
      )}

      {/* Results Display */}
      {result && <ResultsDisplay result={result} />}
    </div>
  );
}
