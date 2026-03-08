"""Tests for data models."""

import time
import pytest
from agentdbg.models import Span, Trace, SpanKind, SpanStatus, CostInfo


class TestCostInfo:
    """Tests for CostInfo model."""

    def test_default_values(self):
        cost = CostInfo()
        assert cost.input_tokens == 0
        assert cost.output_tokens == 0
        assert cost.total_cost == 0.0
        assert cost.model == ""

    def test_to_dict(self):
        cost = CostInfo(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            model="gpt-4o",
        )
        d = cost.to_dict()
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["total_cost"] == 0.003
        assert d["model"] == "gpt-4o"


class TestSpan:
    """Tests for Span model."""

    def test_create_span(self):
        span = Span(
            name="test_span",
            kind=SpanKind.LLM_CALL,
        )
        assert span.name == "test_span"
        assert span.kind == SpanKind.LLM_CALL
        assert span.status == SpanStatus.RUNNING
        assert span.span_id is not None
        assert len(span.span_id) == 8

    def test_span_complete(self):
        span = Span(name="test")
        time.sleep(0.01)  # Small delay
        span.complete(output={"result": "success"})

        assert span.status == SpanStatus.COMPLETED
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms > 0
        assert span.output_data == {"result": "success"}

    def test_span_complete_with_error(self):
        span = Span(name="test")
        span.complete(error="Something went wrong")

        assert span.status == SpanStatus.ERROR
        assert span.error == "Something went wrong"

    def test_span_add_event(self):
        span = Span(name="test")
        span.add_event("checkpoint", {"step": 1})

        assert len(span.events) == 1
        assert span.events[0]["name"] == "checkpoint"
        assert span.events[0]["attributes"]["step"] == 1

    def test_span_to_dict(self):
        span = Span(
            name="test",
            kind=SpanKind.TOOL_CALL,
            input_data={"query": "test"},
        )
        d = span.to_dict()

        assert d["name"] == "test"
        assert d["kind"] == "tool_call"
        assert d["status"] == "running"
        assert d["input_data"] == {"query": "test"}

    def test_span_from_dict(self):
        data = {
            "span_id": "abc12345",
            "trace_id": "trace123",
            "name": "test_span",
            "kind": "llm_call",
            "status": "completed",
            "start_time": 1000.0,
            "end_time": 1001.0,
            "duration_ms": 1000.0,
            "input_data": {"messages": []},
            "output_data": {"content": "Hello"},
            "cost": {"input_tokens": 10, "output_tokens": 5},
        }

        span = Span.from_dict(data)
        assert span.span_id == "abc12345"
        assert span.name == "test_span"
        assert span.kind == SpanKind.LLM_CALL
        assert span.status == SpanStatus.COMPLETED
        assert span.cost.input_tokens == 10


class TestTrace:
    """Tests for Trace model."""

    def test_create_trace(self):
        trace = Trace(name="test_trace")
        assert trace.name == "test_trace"
        assert trace.status == SpanStatus.RUNNING
        assert trace.trace_id is not None
        assert len(trace.spans) == 0

    def test_trace_add_span(self):
        trace = Trace(name="test")
        span = Span(name="span1", kind=SpanKind.LLM_CALL)
        span.cost = CostInfo(total_cost=0.01, total_tokens=100)

        trace.add_span(span)

        assert len(trace.spans) == 1
        assert trace.spans[0].trace_id == trace.trace_id
        assert trace.total_cost == 0.01
        assert trace.total_tokens == 100

    def test_trace_get_span(self):
        trace = Trace(name="test")
        span = Span(name="span1")
        trace.add_span(span)

        found = trace.get_span(span.span_id)
        assert found is not None
        assert found.name == "span1"

        not_found = trace.get_span("nonexistent")
        assert not_found is None

    def test_trace_complete(self):
        trace = Trace(name="test")
        span1 = Span(name="span1")
        span1.complete()
        trace.add_span(span1)

        trace.complete()
        assert trace.status == SpanStatus.COMPLETED
        assert trace.end_time is not None

    def test_trace_complete_with_errors(self):
        trace = Trace(name="test")
        span1 = Span(name="span1")
        span1.complete(error="Failed")
        trace.add_span(span1)

        trace.complete()
        assert trace.status == SpanStatus.ERROR

    def test_trace_to_dict(self):
        trace = Trace(name="test", metadata={"env": "dev"})
        span = Span(name="span1")
        trace.add_span(span)

        d = trace.to_dict()
        assert d["name"] == "test"
        assert d["metadata"] == {"env": "dev"}
        assert len(d["spans"]) == 1

    def test_trace_from_dict(self):
        data = {
            "trace_id": "trace123",
            "name": "test",
            "status": "completed",
            "start_time": 1000.0,
            "end_time": 1001.0,
            "total_cost": 0.05,
            "total_tokens": 500,
            "spans": [
                {"span_id": "s1", "name": "span1", "kind": "llm_call", "status": "completed"}
            ],
        }

        trace = Trace.from_dict(data)
        assert trace.trace_id == "trace123"
        assert trace.total_cost == 0.05
        assert len(trace.spans) == 1
