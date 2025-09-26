interface SlopScoreDisplayProps {
  score: number;
  level: string;
  confidence: number;
}

export function SlopScoreDisplay({
  score,
  level,
  confidence,
}: SlopScoreDisplayProps) {
  const getScoreColor = (score: number) => {
    if (score <= 0.3) return "text-green-600";
    if (score <= 0.55) return "text-yellow-600";
    if (score <= 0.75) return "text-orange-600";
    return "text-red-600";
  };

  const getProgressColor = (score: number) => {
    if (score <= 0.3) return "bg-green-500";
    if (score <= 0.55) return "bg-yellow-500";
    if (score <= 0.75) return "bg-orange-500";
    return "bg-red-500";
  };

  return (
    <div className="metric-card">
      <h3 className="text-sm font-medium text-gray-500 mb-2">Slop Score</h3>
      <div className="space-y-3">
        <div className="text-3xl font-bold">
          <span className={getScoreColor(score)}>
            {(score * 100).toFixed(1)}%
          </span>
        </div>

        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full ${getProgressColor(score)}`}
            style={{ width: `${score * 100}%` }}
          />
        </div>

        <div className="text-sm text-gray-600">
          Confidence: {(confidence * 100).toFixed(1)}%
        </div>
      </div>
    </div>
  );
}
