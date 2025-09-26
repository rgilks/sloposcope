interface MetricCardProps {
  name: string;
  value: number;
  data: { value: number; [key: string]: any };
}

export function MetricCard({ name, value, data }: MetricCardProps) {
  const getStatusIcon = (value: number) => {
    if (value <= 0.3) return "✅";
    if (value <= 0.55) return "⚠️";
    if (value <= 0.75) return "🔶";
    return "❌";
  };

  const getStatusText = (value: number) => {
    if (value <= 0.3) return "Good";
    if (value <= 0.55) return "Watch";
    if (value <= 0.75) return "Sloppy";
    return "High-Slop";
  };

  const getStatusColor = (value: number) => {
    if (value <= 0.3) return "text-green-600";
    if (value <= 0.55) return "text-yellow-600";
    if (value <= 0.75) return "text-orange-600";
    return "text-red-600";
  };

  return (
    <div className="metric-card">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-700 capitalize">
          {name.replace("_", " ")}
        </h3>
        <span className="text-lg">{getStatusIcon(value)}</span>
      </div>

      <div className="space-y-2">
        <div className="text-2xl font-bold">
          <span className={getStatusColor(value)}>
            {(value * 100).toFixed(1)}%
          </span>
        </div>

        <div className="text-xs text-gray-500">{getStatusText(value)}</div>

        <div className="w-full bg-gray-200 rounded-full h-1">
          <div
            className={`h-1 rounded-full ${getStatusColor(value).replace("text-", "bg-")}`}
            style={{ width: `${value * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
}
