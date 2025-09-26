interface Span {
  start: number;
  end: number;
  type: string;
  description: string;
}

interface SpansDisplayProps {
  spans: Span[];
}

export function SpansDisplay({ spans }: SpansDisplayProps) {
  const getSpanColor = (type: string) => {
    switch (type.toLowerCase()) {
      case "repetition":
        return "bg-yellow-100 border-yellow-300 text-yellow-800";
      case "templated":
        return "bg-blue-100 border-blue-300 text-blue-800";
      case "hedging":
        return "bg-purple-100 border-purple-300 text-purple-800";
      case "verbosity":
        return "bg-orange-100 border-orange-300 text-orange-800";
      default:
        return "bg-gray-100 border-gray-300 text-gray-800";
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-xl font-semibold mb-4">Problematic Spans</h2>

      {spans.length === 0 ? (
        <p className="text-gray-500 text-center py-4">
          No problematic spans detected
        </p>
      ) : (
        <div className="space-y-3">
          {spans.map((span, index) => (
            <div
              key={index}
              className={`border rounded-lg p-3 ${getSpanColor(span.type)}`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium capitalize">
                  {span.type.replace("_", " ")}
                </span>
                <span className="text-sm opacity-75">
                  Characters {span.start}-{span.end}
                </span>
              </div>
              <p className="text-sm">{span.description}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
