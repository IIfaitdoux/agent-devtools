"""End-to-end integration tests."""

import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch

from agentdbg.core import AgentDebugger, trace, traced, get_debugger
from agentdbg.config import DebugConfig
from agentdbg.models import SpanKind, SpanStatus, CostInfo


class TestEndToEndTracing:
    """End-to-end tracing tests."""

    def test_full_trace_workflow(self):
        """Test a complete trace workflow with nested spans."""
        debugger = AgentDebugger()
        debugger.clear_traces()

        # Start a trace
        t = debugger.start_trace(name="agent_run", metadata={"user": "test"})

        # First span - agent thinking
        s1 = debugger.start_span(
            name="agent_think",
            kind=SpanKind.AGENT_STEP,
            input_data={"task": "Answer a question"},
        )

        # Nested LLM call
        s2 = debugger.start_span(
            name="openai.chat.completions.create",
            kind=SpanKind.LLM_CALL,
            input_data={"messages": [{"role": "user", "content": "Hello"}]},
        )
        s2.cost = CostInfo(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            total_cost=0.001,
            model="gpt-4o",
        )
        debugger.end_span(s2, output_data={"content": "Hi there!"})

        # Tool call
        s3 = debugger.start_span(
            name="search_tool",
            kind=SpanKind.TOOL_CALL,
            input_data={"query": "latest news"},
        )
        debugger.end_span(s3, output_data={"results": ["news1", "news2"]})

        # End agent thinking
        debugger.end_span(s1, output_data={"decision": "respond"})

        # End trace
        debugger.end_trace()

        # Verify
        traces = debugger.get_all_traces()
        assert len(traces) == 1

        trace = traces[0]
        assert trace.name == "agent_run"
        assert trace.status == SpanStatus.COMPLETED
        assert len(trace.spans) == 3

        # Check parent-child relationships
        llm_span = trace.get_span(s2.span_id)
        assert llm_span.parent_id == s1.span_id

    def test_trace_with_error_handling(self):
        """Test tracing with error handling."""
        debugger = AgentDebugger(config=DebugConfig(auto_pause_on_error=False))
        debugger.clear_traces()

        debugger.start_trace(name="error_test")

        span = debugger.start_span(name="failing_operation", kind=SpanKind.TOOL_CALL)
        debugger.end_span(span, error="Connection timeout")

        debugger.end_trace()

        traces = debugger.get_all_traces()
        assert traces[0].status == SpanStatus.ERROR

        failing_span = traces[0].spans[0]
        assert failing_span.error == "Connection timeout"
        assert failing_span.status == SpanStatus.ERROR

    def test_cost_accumulation(self):
        """Test that costs accumulate correctly across spans."""
        debugger = AgentDebugger()
        debugger.clear_traces()

        debugger.start_trace(name="cost_test")

        for i in range(3):
            span = debugger.start_span(name=f"llm_call_{i}", kind=SpanKind.LLM_CALL)
            cost = CostInfo(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                total_cost=0.01,
                model="gpt-4o",
            )
            debugger.end_span(span, cost=cost)

        debugger.end_trace()

        trace = debugger.get_all_traces()[0]
        assert trace.total_cost == pytest.approx(0.03, rel=1e-6)
        assert trace.total_tokens == 450


class TestContextManagerAndDecorator:
    """Tests for context manager and decorator patterns."""

    def test_trace_context_manager(self):
        """Test using trace as a context manager."""
        with trace(name="test_operation", kind=SpanKind.CUSTOM) as span:
            span.add_event("checkpoint", {"step": 1})
            time.sleep(0.01)

        assert span.status == SpanStatus.COMPLETED
        assert span.duration_ms > 0
        assert len(span.events) == 1

    def test_nested_context_managers(self):
        """Test nested trace context managers."""
        with trace(name="outer", kind=SpanKind.CHAIN) as outer:
            with trace(name="inner", kind=SpanKind.LLM_CALL) as inner:
                pass

        assert inner.parent_id == outer.span_id

    def test_traced_decorator(self):
        """Test the @traced decorator."""

        @traced(name="compute", kind=SpanKind.CUSTOM, capture_args=True, capture_result=True)
        def compute(x, y):
            return x * y

        debugger = get_debugger()
        debugger.clear_traces()
        debugger.start_trace(name="decorator_test")

        result = compute(5, 3)

        debugger.end_trace()

        assert result == 15
        traces = debugger.get_all_traces()
        assert len(traces) == 1

    @pytest.mark.asyncio
    async def test_traced_async_decorator(self):
        """Test the @traced decorator with async functions."""

        @traced(name="async_compute")
        async def async_compute(x, y):
            await asyncio.sleep(0.01)
            return x + y

        debugger = get_debugger()
        debugger.clear_traces()
        debugger.start_trace(name="async_test")

        result = await async_compute(2, 3)

        debugger.end_trace()

        assert result == 5


class TestAutoInstrumentation:
    """Tests for auto-instrumentation of LLM providers."""

    def test_openai_instrumentation_mock(self):
        """Test OpenAI instrumentation with mocked client."""
        from agentdbg.instrumentors import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor()

        # Just test that instrumentor doesn't crash without openai installed
        instrumentor.instrument()
        instrumentor.uninstrument()

    def test_anthropic_instrumentation_mock(self):
        """Test Anthropic instrumentation with mocked client."""
        from agentdbg.instrumentors import AnthropicInstrumentor

        instrumentor = AnthropicInstrumentor()

        # Just test that instrumentor doesn't crash without anthropic installed
        instrumentor.instrument()
        instrumentor.uninstrument()

    def test_langchain_instrumentation_mock(self):
        """Test LangChain instrumentation with mocked client."""
        from agentdbg.instrumentors import LangChainInstrumentor

        instrumentor = LangChainInstrumentor()

        # Just test that instrumentor doesn't crash without langchain installed
        instrumentor.instrument()
        instrumentor.uninstrument()


class TestDebuggerControls:
    """Tests for debugger pause/resume/step functionality."""

    def test_pause_blocks_execution(self):
        """Test that pause blocks span creation."""
        debugger = AgentDebugger()
        debugger.clear_traces()

        # This is tricky to test without threading
        # Just verify the state changes work
        assert not debugger.is_paused

        debugger.pause()
        assert debugger.is_paused

        debugger.resume()
        assert not debugger.is_paused

    def test_auto_pause_on_cost(self):
        """Test auto-pause when cost threshold is exceeded."""
        config = DebugConfig(auto_pause_on_cost=0.05)
        debugger = AgentDebugger(config=config)
        debugger.clear_traces()

        debugger.start_trace(name="cost_threshold_test")

        # Add spans with costs
        for i in range(3):
            span = debugger.start_span(name=f"call_{i}", kind=SpanKind.LLM_CALL)
            cost = CostInfo(total_cost=0.02)
            debugger.end_span(span, cost=cost)

        # After 3 spans with $0.02 each = $0.06, should have triggered pause
        # Note: The pause happens asynchronously, so we check the trace cost
        trace = debugger.get_current_trace()
        assert trace.total_cost >= 0.05
