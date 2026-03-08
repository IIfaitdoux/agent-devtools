"""Tests for core debugger functionality."""

import pytest
import threading
import time
from agentdbg.core import AgentDebugger, DebuggerState, trace, traced, get_debugger
from agentdbg.config import DebugConfig
from agentdbg.models import SpanKind, SpanStatus


class TestDebuggerState:
    """Tests for DebuggerState."""

    def test_initial_state(self):
        state = DebuggerState()
        assert not state.is_paused
        assert not state.is_step_mode

    def test_pause_resume(self):
        state = DebuggerState()

        state.pause()
        assert state.is_paused

        state.resume()
        assert not state.is_paused

    def test_step_mode(self):
        state = DebuggerState()
        state.pause()
        state.step()
        assert state.is_step_mode

    def test_wait_if_paused_not_paused(self):
        state = DebuggerState()
        # Should not block
        state.wait_if_paused()

    def test_wait_if_paused_with_resume(self):
        state = DebuggerState()
        state.pause()

        # Resume from another thread
        def resume_later():
            time.sleep(0.1)
            state.resume()

        t = threading.Thread(target=resume_later)
        t.start()

        start = time.time()
        state.wait_if_paused()
        elapsed = time.time() - start

        t.join()
        assert elapsed >= 0.1
        assert not state.is_paused

    def test_breakpoint(self):
        state = DebuggerState()

        # Add breakpoint that triggers on error
        state.add_breakpoint(lambda span: span.error is not None)

        from agentdbg.models import Span

        # Normal span shouldn't trigger
        normal_span = Span(name="test")
        state.wait_if_paused(normal_span)
        assert not state.is_paused

        # Error span should trigger
        error_span = Span(name="test", error="Failed")
        # Need to resume immediately for test
        def auto_resume():
            time.sleep(0.05)
            state.resume()
        t = threading.Thread(target=auto_resume)
        t.start()

        state.wait_if_paused(error_span)
        t.join()

    def test_callbacks(self):
        state = DebuggerState()
        paused_count = [0]
        resumed_count = [0]

        state.on_pause(lambda: paused_count.__setitem__(0, paused_count[0] + 1))
        state.on_resume(lambda: resumed_count.__setitem__(0, resumed_count[0] + 1))

        state.pause()
        assert paused_count[0] == 1

        state.resume()
        assert resumed_count[0] == 1


class TestAgentDebugger:
    """Tests for AgentDebugger."""

    def test_create_debugger(self):
        debugger = AgentDebugger()
        assert debugger.config is not None
        assert not debugger.is_paused

    def test_create_with_config(self):
        config = DebugConfig(auto_pause_on_error=False)
        debugger = AgentDebugger(config=config)
        assert debugger.config.auto_pause_on_error is False

    def test_start_end_trace(self):
        debugger = AgentDebugger()

        trace = debugger.start_trace(name="test")
        assert trace.name == "test"
        assert debugger.get_current_trace() == trace

        ended = debugger.end_trace()
        assert ended.status == SpanStatus.COMPLETED
        assert debugger.get_current_trace() is None

    def test_start_end_span(self):
        debugger = AgentDebugger()
        debugger.start_trace(name="test")

        span = debugger.start_span(
            name="llm_call",
            kind=SpanKind.LLM_CALL,
            input_data={"prompt": "Hello"},
        )

        assert span.name == "llm_call"
        assert span.kind == SpanKind.LLM_CALL
        assert debugger.get_current_span() == span

        debugger.end_span(output_data={"response": "Hi"})
        assert span.status == SpanStatus.COMPLETED
        assert span.output_data == {"response": "Hi"}

    def test_nested_spans(self):
        debugger = AgentDebugger()
        debugger.start_trace(name="test")

        parent = debugger.start_span(name="parent", kind=SpanKind.CHAIN)
        child = debugger.start_span(name="child", kind=SpanKind.LLM_CALL)

        assert child.parent_id == parent.span_id

        debugger.end_span(child)
        assert debugger.get_current_span() == parent

        debugger.end_span(parent)
        assert debugger.get_current_span() is None

    def test_get_all_traces(self):
        debugger = AgentDebugger()

        debugger.start_trace(name="trace1")
        debugger.end_trace()

        debugger.start_trace(name="trace2")
        debugger.end_trace()

        traces = debugger.get_all_traces()
        assert len(traces) == 2

    def test_clear_traces(self):
        debugger = AgentDebugger()

        debugger.start_trace(name="test")
        debugger.end_trace()

        debugger.clear_traces()
        assert len(debugger.get_all_traces()) == 0

    def test_pause_resume(self):
        debugger = AgentDebugger()

        debugger.pause()
        assert debugger.is_paused

        debugger.resume()
        assert not debugger.is_paused

    def test_span_callbacks(self):
        debugger = AgentDebugger()
        spans_received = []

        debugger.on_span(lambda s: spans_received.append(s))

        debugger.start_trace(name="test")
        debugger.start_span(name="span1")
        debugger.end_span()

        # Should receive span on start and end
        assert len(spans_received) >= 2


class TestTraceContextManager:
    """Tests for the trace context manager."""

    def test_basic_trace(self):
        with trace(name="test", kind=SpanKind.CUSTOM) as span:
            assert span.name == "test"
            assert span.status == SpanStatus.RUNNING

        assert span.status == SpanStatus.COMPLETED

    def test_trace_with_exception(self):
        with pytest.raises(ValueError):
            with trace(name="test") as span:
                raise ValueError("Test error")

        assert span.status == SpanStatus.ERROR
        assert "Test error" in span.error

    def test_trace_input_data(self):
        with trace(name="test", input_data={"query": "hello"}) as span:
            pass

        assert span.input_data == {"query": "hello"}


class TestTracedDecorator:
    """Tests for the @traced decorator."""

    def test_traced_sync_function(self):
        @traced(name="my_function")
        def my_function(x, y):
            return x + y

        result = my_function(1, 2)
        assert result == 3

    def test_traced_captures_args(self):
        debugger = get_debugger()
        debugger.clear_traces()

        @traced(name="add", capture_args=True)
        def add(a, b):
            return a + b

        debugger.start_trace(name="test")
        result = add(5, 3)
        debugger.end_trace()

        assert result == 8
        traces = debugger.get_all_traces()
        assert len(traces) == 1

    def test_traced_captures_result(self):
        debugger = get_debugger()
        debugger.clear_traces()

        @traced(name="multiply", capture_result=True)
        def multiply(a, b):
            return a * b

        debugger.start_trace(name="test")
        result = multiply(4, 5)
        debugger.end_trace()

        assert result == 20

    def test_traced_with_exception(self):
        @traced(name="failing")
        def failing():
            raise RuntimeError("Intentional error")

        with pytest.raises(RuntimeError):
            failing()
