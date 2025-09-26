interface DomainSelectorProps {
  value: string;
  onChange: (value: string) => void;
}

export function DomainSelector({ value, onChange }: DomainSelectorProps) {
  const domains = [
    {
      value: "general",
      label: "General",
      description: "General purpose text analysis",
    },
    {
      value: "news",
      label: "News",
      description: "News articles and journalism",
    },
    { value: "qa", label: "Q&A", description: "Question and answer content" },
  ];

  return (
    <div>
      <label
        htmlFor="domain"
        className="block text-sm font-medium text-gray-700 mb-2"
      >
        Domain
      </label>
      <select
        id="domain"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
      >
        {domains.map((domain) => (
          <option key={domain.value} value={domain.value}>
            {domain.label} - {domain.description}
          </option>
        ))}
      </select>
    </div>
  );
}
