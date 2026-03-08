"""
Data models for AgentDBG traces and spans.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SpanKind(str, Enum):
    """Type of span in the trace."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    AGENT_STEP = "agent_step"
    CHAIN = "chain"
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"
    CUSTOM = "custom"


class SpanStatus(str, Enum):
    """Status of a span."""

    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class CostInfo:
    """Token and cost information for a span."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "model": self.model,
        }


@dataclass
class Span:
    """A single span in a trace representing one operation."""

    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_id: str = ""
    parent_id: str | None = None
    name: str = ""
    kind: SpanKind = SpanKind.CUSTOM
    status: SpanStatus = SpanStatus.RUNNING
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float | None = None

    # Input/Output data
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)

    # Metadata
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    # Cost tracking
    cost: CostInfo = field(default_factory=CostInfo)

    # Error info
    error: str | None = None
    error_traceback: str | None = None

    def complete(self, output: dict[str, Any] | None = None, error: str | None = None) -> None:
        """Mark span as completed."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000

        if error:
            self.status = SpanStatus.ERROR
            self.error = error
        else:
            self.status = SpanStatus.COMPLETED

        if output:
            self.output_data = output

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind.value,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "attributes": self.attributes,
            "events": self.events,
            "cost": self.cost.to_dict(),
            "error": self.error,
            "error_traceback": self.error_traceback,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Span:
        """Create span from dictionary."""
        cost_data = data.get("cost", {})
        cost = CostInfo(
            input_tokens=cost_data.get("input_tokens", 0),
            output_tokens=cost_data.get("output_tokens", 0),
            total_tokens=cost_data.get("total_tokens", 0),
            input_cost=cost_data.get("input_cost", 0.0),
            output_cost=cost_data.get("output_cost", 0.0),
            total_cost=cost_data.get("total_cost", 0.0),
            model=cost_data.get("model", ""),
        )

        return cls(
            span_id=data.get("span_id", str(uuid.uuid4())[:8]),
            trace_id=data.get("trace_id", ""),
            parent_id=data.get("parent_id"),
            name=data.get("name", ""),
            kind=SpanKind(data.get("kind", "custom")),
            status=SpanStatus(data.get("status", "running")),
            start_time=data.get("start_time", time.time()),
            end_time=data.get("end_time"),
            duration_ms=data.get("duration_ms"),
            input_data=data.get("input_data", {}),
            output_data=data.get("output_data", {}),
            attributes=data.get("attributes", {}),
            events=data.get("events", []),
            cost=cost,
            error=data.get("error"),
            error_traceback=data.get("error_traceback"),
        )


@dataclass
class Trace:
    """A complete trace containing multiple spans."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: SpanStatus = SpanStatus.RUNNING
    spans: list[Span] = field(default_factory=list)

    # Aggregated cost
    total_cost: float = 0.0
    total_tokens: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_span(self, span: Span) -> None:
        """Add a span to this trace."""
        span.trace_id = self.trace_id
        self.spans.append(span)

        # Update aggregated cost
        self.total_cost += span.cost.total_cost
        self.total_tokens += span.cost.total_tokens

    def get_span(self, span_id: str) -> Span | None:
        """Get a span by ID."""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def complete(self) -> None:
        """Mark trace as completed."""
        self.end_time = time.time()

        # Check if any spans have errors
        has_error = any(s.status == SpanStatus.ERROR for s in self.spans)
        self.status = SpanStatus.ERROR if has_error else SpanStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status.value,
            "spans": [s.to_dict() for s in self.spans],
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        """Create trace from dictionary."""
        trace = cls(
            trace_id=data.get("trace_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            start_time=data.get("start_time", time.time()),
            end_time=data.get("end_time"),
            status=SpanStatus(data.get("status", "running")),
            total_cost=data.get("total_cost", 0.0),
            total_tokens=data.get("total_tokens", 0),
            metadata=data.get("metadata", {}),
        )

        for span_data in data.get("spans", []):
            trace.spans.append(Span.from_dict(span_data))

        return trace
