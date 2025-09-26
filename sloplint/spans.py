"""
Span handling for AI slop analysis.

Manages character-level annotations for problematic regions in text.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SpanType(Enum):
    """Types of problematic spans that can be detected."""
    REPETITION = "repetition"
    TEMPLATED = "templated"
    OFF_TOPIC = "off_topic"
    HEDGING = "hedging"
    COHERENCE_BREAK = "coherence_break"
    VERBOSITY = "verbosity"
    COMPLEXITY = "complexity"


@dataclass
class Span:
    """Represents a character span with associated metadata."""
    start: int
    end: int
    span_type: SpanType
    confidence: float = 1.0
    note: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate span data."""
        if self.start < 0:
            raise ValueError("Start position must be non-negative")
        if self.end <= self.start:
            raise ValueError("End position must be greater than start position")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary representation."""
        return {
            "start": self.start,
            "end": self.end,
            "axis": self.span_type.value,
            "confidence": self.confidence,
            "note": self.note,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Span":
        """Create span from dictionary representation."""
        return cls(
            start=data["start"],
            end=data["end"],
            span_type=SpanType(data["axis"]),
            confidence=data.get("confidence", 1.0),
            note=data.get("note"),
            metadata=data.get("metadata", {}),
        )


class SpanCollection:
    """Manages a collection of spans with merging and filtering capabilities."""

    def __init__(self, spans: list[Span] | None = None):
        """Initialize span collection."""
        self.spans = spans or []
        self._sort_spans()

    def add_span(self, span: Span) -> None:
        """Add a span to the collection."""
        self.spans.append(span)
        self._sort_spans()

    def add_spans(self, spans: list[Span]) -> None:
        """Add multiple spans to the collection."""
        self.spans.extend(spans)
        self._sort_spans()

    def _sort_spans(self) -> None:
        """Sort spans by start position."""
        self.spans.sort(key=lambda s: s.start)

    def merge_overlapping(self, max_gap: int = 0) -> "SpanCollection":
        """Merge overlapping or adjacent spans of the same type."""
        if not self.spans:
            return SpanCollection()

        merged = []
        current = self.spans[0]

        for span in self.spans[1:]:
            if (span.start <= current.end + max_gap and
                span.span_type == current.span_type):

                # Merge spans
                current = Span(
                    start=min(current.start, span.start),
                    end=max(current.end, span.end),
                    span_type=current.span_type,
                    confidence=max(current.confidence, span.confidence),
                    note=current.note or span.note,
                    metadata={**(current.metadata or {}), **(span.metadata or {})}
                )
            else:
                merged.append(current)
                current = span

        merged.append(current)
        return SpanCollection(merged)

    def filter_by_type(self, span_type: SpanType) -> "SpanCollection":
        """Filter spans by type."""
        filtered = [span for span in self.spans if span.span_type == span_type]
        return SpanCollection(filtered)

    def filter_by_confidence(self, min_confidence: float) -> "SpanCollection":
        """Filter spans by minimum confidence."""
        filtered = [span for span in self.spans if span.confidence >= min_confidence]
        return SpanCollection(filtered)

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Convert all spans to dictionary list."""
        return [span.to_dict() for span in self.spans]

    @classmethod
    def from_dict_list(cls, data: list[dict[str, Any]]) -> "SpanCollection":
        """Create span collection from dictionary list."""
        spans = [Span.from_dict(item) for item in data]
        return cls(spans)

    def get_spans_in_range(self, start: int, end: int) -> "SpanCollection":
        """Get spans that overlap with the given range."""
        overlapping = [
            span for span in self.spans
            if span.start < end and span.end > start
        ]
        return SpanCollection(overlapping)

    def remove_duplicates(self) -> "SpanCollection":
        """Remove duplicate spans."""
        seen = set()
        unique_spans = []

        for span in self.spans:
            span_key = (span.start, span.end, span.span_type.value)
            if span_key not in seen:
                seen.add(span_key)
                unique_spans.append(span)

        return SpanCollection(unique_spans)

    def __len__(self) -> int:
        """Return number of spans."""
        return len(self.spans)

    def __iter__(self) -> iter[Span]:  # type: ignore
        """Iterate over spans."""
        return iter(self.spans)

    def __getitem__(self, index: int) -> Span:
        """Get span by index."""
        return self.spans[index]
