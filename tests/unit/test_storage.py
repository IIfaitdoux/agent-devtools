"""Tests for storage backends."""

import os
import pytest
import tempfile
import time
from agentdbg.storage import SQLiteStorage
from agentdbg.models import Span, Trace, SpanKind, SpanStatus, CostInfo


@pytest.fixture
def temp_db():
    """Create a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        yield db_path


@pytest.fixture
def storage(temp_db):
    """Create a storage instance."""
    return SQLiteStorage(temp_db)


class TestSQLiteStorage:
    """Tests for SQLiteStorage."""

    def test_create_storage(self, temp_db):
        storage = SQLiteStorage(temp_db)
        assert os.path.exists(temp_db)

    def test_save_and_get_trace(self, storage):
        trace = Trace(name="test_trace", metadata={"env": "test"})
        trace.total_cost = 0.05
        trace.total_tokens = 500
        trace.complete()

        storage.save_trace(trace)

        loaded = storage.get_trace(trace.trace_id)
        assert loaded is not None
        assert loaded.name == "test_trace"
        assert loaded.total_cost == 0.05
        assert loaded.total_tokens == 500
        assert loaded.metadata == {"env": "test"}

    def test_save_and_get_span(self, storage):
        trace = Trace(name="test")
        storage.save_trace(trace)

        span = Span(
            trace_id=trace.trace_id,
            name="llm_call",
            kind=SpanKind.LLM_CALL,
            input_data={"prompt": "Hello"},
            output_data={"response": "Hi"},
        )
        span.cost = CostInfo(
            input_tokens=10,
            output_tokens=5,
            total_cost=0.001,
            model="gpt-4o",
        )
        span.complete()

        storage.save_span(span)

        loaded_trace = storage.get_trace(trace.trace_id)
        assert len(loaded_trace.spans) == 1

        loaded_span = loaded_trace.spans[0]
        assert loaded_span.name == "llm_call"
        assert loaded_span.kind == SpanKind.LLM_CALL
        assert loaded_span.input_data == {"prompt": "Hello"}
        assert loaded_span.cost.input_tokens == 10

    def test_get_traces(self, storage):
        for i in range(5):
            trace = Trace(name=f"trace_{i}")
            trace.complete()
            storage.save_trace(trace)
            time.sleep(0.01)  # Ensure different timestamps

        traces = storage.get_traces(limit=3)
        assert len(traces) == 3

    def test_get_traces_with_offset(self, storage):
        for i in range(5):
            trace = Trace(name=f"trace_{i}")
            trace.complete()
            storage.save_trace(trace)

        traces = storage.get_traces(limit=2, offset=2)
        assert len(traces) == 2

    def test_get_traces_by_status(self, storage):
        trace1 = Trace(name="success")
        trace1.complete()
        storage.save_trace(trace1)

        trace2 = Trace(name="error")
        trace2.status = SpanStatus.ERROR
        storage.save_trace(trace2)

        error_traces = storage.get_traces(status="error")
        assert len(error_traces) == 1
        assert error_traces[0].name == "error"

    def test_delete_trace(self, storage):
        trace = Trace(name="to_delete")
        storage.save_trace(trace)

        span = Span(trace_id=trace.trace_id, name="span1")
        storage.save_span(span)

        result = storage.delete_trace(trace.trace_id)
        assert result is True

        loaded = storage.get_trace(trace.trace_id)
        assert loaded is None

    def test_delete_nonexistent_trace(self, storage):
        result = storage.delete_trace("nonexistent")
        assert result is False

    def test_delete_old_traces(self, storage):
        # Create an old trace (fake old timestamp)
        old_trace = Trace(name="old")
        old_trace.start_time = time.time() - (10 * 86400)  # 10 days ago
        old_trace.complete()
        storage.save_trace(old_trace)

        # Create a recent trace
        new_trace = Trace(name="new")
        new_trace.complete()
        storage.save_trace(new_trace)

        # Delete traces older than 7 days
        deleted = storage.delete_old_traces(days=7)
        assert deleted == 1

        # Old trace should be gone
        assert storage.get_trace(old_trace.trace_id) is None

        # New trace should remain
        assert storage.get_trace(new_trace.trace_id) is not None

    def test_get_stats(self, storage):
        trace1 = Trace(name="trace1")
        trace1.total_cost = 0.05
        trace1.total_tokens = 500
        trace1.complete()
        storage.save_trace(trace1)

        trace2 = Trace(name="trace2")
        trace2.total_cost = 0.03
        trace2.total_tokens = 300
        trace2.status = SpanStatus.ERROR
        storage.save_trace(trace2)

        for t in [trace1, trace2]:
            span = Span(trace_id=t.trace_id, name="span")
            storage.save_span(span)

        stats = storage.get_stats()
        assert stats["trace_count"] == 2
        assert stats["span_count"] == 2
        assert stats["total_cost"] == 0.08
        assert stats["total_tokens"] == 800
        assert stats["error_count"] == 1
        assert stats["error_rate"] == 0.5

    def test_clear(self, storage):
        trace = Trace(name="test")
        storage.save_trace(trace)

        span = Span(trace_id=trace.trace_id, name="span")
        storage.save_span(span)

        storage.clear()

        stats = storage.get_stats()
        assert stats["trace_count"] == 0
        assert stats["span_count"] == 0

    def test_get_recent_traces(self, storage):
        # Create a trace
        trace = Trace(name="recent")
        trace.complete()
        storage.save_trace(trace)

        recent = storage.get_recent_traces(hours=1)
        assert len(recent) == 1
        assert recent[0].name == "recent"
