interface OptionsPanelProps {
  explain: boolean;
  setExplain: (value: boolean) => void;
  spans: boolean;
  setSpans: (value: boolean) => void;
}

export function OptionsPanel({
  explain,
  setExplain,
  spans,
  setSpans,
}: OptionsPanelProps) {
  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-700 mb-3">
        Analysis Options
      </h3>
      <div className="space-y-3">
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={explain}
            onChange={(e) => setExplain(e.target.checked)}
            className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
          />
          <span className="ml-2 text-sm text-gray-700">
            Include detailed explanations
          </span>
        </label>

        <label className="flex items-center">
          <input
            type="checkbox"
            checked={spans}
            onChange={(e) => setSpans(e.target.checked)}
            className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
          />
          <span className="ml-2 text-sm text-gray-700">
            Show character spans for problematic regions
          </span>
        </label>
      </div>
    </div>
  );
}
