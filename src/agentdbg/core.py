"""
Core AgentDBG debugger implementation.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar

from agentdbg.config import DebugConfig, get_model_cost
from agentdbg.models import CostInfo, Span, SpanKind, SpanStatus, Trace

# Global debugger instance
_debugger: AgentDebugger | None = None
_debugger_lock = threading.Lock()

# Context variable for current span (works with async)
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "_current_span", default=None
)
_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "_current_trace", default=None
)

T = TypeVar("T")


def get_debugger() -> AgentDebugger:
    """Get the global debugger instance."""
    global _debugger
    if _debugger is None:
        with _debugger_lock:
            if _debugger is None:
                _debugger = AgentDebugger()
    return _debugger


class DebuggerState:
    """Manages the debugging state - pause, resume, step."""

    def __init__(self) -> None:
        self._paused = False
        self._step_mode = False
        self._step_event = threading.Event()
        self._step_event.set()  # Initially not blocked

        # Async support
        self._async_step_event: asyncio.Event | None = None

        # Breakpoints
        self._breakpoints: list[Callable[[Span], bool]] = []

        # Callbacks for UI updates
        self._on_pause_callbacks: list[Callable[[], None]] = []
        self._on_resume_callbacks: list[Callable[[], None]] = []

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_step_mode(self) -> bool:
        return self._step_mode

    def pause(self) -> None:
        """Pause execution."""
        self._paused = True
        self._step_event.clear()
        if self._async_step_event:
            self._async_step_event.clear()

        for callback in self._on_pause_callbacks:
            callback()

    def resume(self) -> None:
        """Resume execution."""
        self._paused = False
        self._step_mode = False
        self._step_event.set()
        if self._async_step_event:
            self._async_step_event.set()

        for callback in self._on_resume_callbacks:
            callback()

    def step(self) -> None:
        """Execute one step and pause again."""
        self._step_mode = True
        self._step_event.set()
        if self._async_step_event:
            self._async_step_event.set()

    def wait_if_paused(self, span: Span | None = None) -> None:
        """Block if paused. Call this at each step."""
        # Check breakpoints
        if span:
            for bp in self._breakpoints:
                if bp(span):
                    self.pause()
                    break

        if not self._paused:
            return

        # Wait for resume or step
        self._step_event.wait()

        # If in step mode, pause again after this step
        if self._step_mode:
            self._step_mode = False
            self._step_event.clear()

    async def async_wait_if_paused(self, span: Span | None = None) -> None:
        """Async version of wait_if_paused."""
        if self._async_step_event is None:
            self._async_step_event = asyncio.Event()
            if not self._paused:
                self._async_step_event.set()

        # Check breakpoints
        if span:
            for bp in self._breakpoints:
                if bp(span):
                    self.pause()
                    break

        if not self._paused:
            return

        await self._async_step_event.wait()

        if self._step_mode:
            self._step_mode = False
            self._async_step_event.clear()

    def add_breakpoint(self, condition: Callable[[Span], bool]) -> None:
        """Add a breakpoint condition."""
        self._breakpoints.append(condition)

    def clear_breakpoints(self) -> None:
        """Clear all breakpoints."""
        self._breakpoints.clear()

    def on_pause(self, callback: Callable[[], None]) -> None:
        """Register a callback for when execution is paused."""
        self._on_pause_callbacks.append(callback)

    def on_resume(self, callback: Callable[[], None]) -> None:
        """Register a callback for when execution is resumed."""
        self._on_resume_callbacks.append(callback)


class AgentDebugger:
    """Main debugger class that orchestrates tracing and debugging."""

    def __init__(self, config: DebugConfig | None = None) -> None:
        self.config = config or DebugConfig()
        self.state = DebuggerState()

        # Active traces
        self._traces: dict[str, Trace] = {}
        self._traces_lock = threading.Lock()

        # Callbacks for real-time updates
        self._span_callbacks: list[Callable[[Span], None]] = []
        self._trace_callbacks: list[Callable[[Trace], None]] = []

        # Auto-pause conditions
        if self.config.auto_pause_on_cost:
            self.state.add_breakpoint(
                lambda s: s.cost.total_cost >= self.config.auto_pause_on_cost  # type: ignore
            )

        if self.config.auto_pause_on_tokens:
            self.state.add_breakpoint(
                lambda s: s.cost.total_tokens >= self.config.auto_pause_on_tokens  # type: ignore
            )

        if self.config.auto_pause_on_error:
            self.state.add_breakpoint(lambda s: s.status == SpanStatus.ERROR)

    def start_trace(self, name: str = "", metadata: dict[str, Any] | None = None) -> Trace:
        """Start a new trace."""
        trace = Trace(name=name, metadata=metadata or {})

        with self._traces_lock:
            self._traces[trace.trace_id] = trace

        _current_trace.set(trace)

        # Notify callbacks
        for callback in self._trace_callbacks:
            callback(trace)

        return trace

    def end_trace(self, trace_id: str | None = None) -> Trace | None:
        """End a trace."""
        if trace_id:
            trace = self._traces.get(trace_id)
        else:
            trace = _current_trace.get()

        if trace:
            trace.complete()
            _current_trace.set(None)

            # Notify callbacks
            for callback in self._trace_callbacks:
                callback(trace)

        return trace

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.CUSTOM,
        input_data: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span within the current trace."""
        trace = _current_trace.get()
        if trace is None:
            trace = self.start_trace(name=name)

        parent_span = _current_span.get()

        span = Span(
            trace_id=trace.trace_id,
            parent_id=parent_span.span_id if parent_span else None,
            name=name,
            kind=kind,
            input_data=self._truncate_data(input_data or {}),
            attributes=attributes or {},
        )

        trace.add_span(span)
        _current_span.set(span)

        # Check if we should pause (breakpoints)
        self.state.wait_if_paused(span)

        # Notify callbacks
        for callback in self._span_callbacks:
            callback(span)

        return span

    def end_span(
        self,
        span: Span | None = None,
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
        cost: CostInfo | None = None,
    ) -> None:
        """End a span."""
        if span is None:
            span = _current_span.get()

        if span is None:
            return

        if output_data:
            span.output_data = self._truncate_data(output_data)

        if cost:
            span.cost = cost
            # Update trace totals
            trace = _current_trace.get()
            if trace:
                trace.total_cost += cost.total_cost
                trace.total_tokens += cost.total_tokens

        span.complete(output=span.output_data, error=error)

        # Check if we should pause on error
        if error and self.config.auto_pause_on_error:
            self.state.pause()

        # Check cost threshold
        if self.config.auto_pause_on_cost:
            trace = _current_trace.get()
            if trace and trace.total_cost >= self.config.auto_pause_on_cost:
                self.state.pause()

        # Notify callbacks
        for callback in self._span_callbacks:
            callback(span)

        # Restore parent span
        if span.parent_id:
            trace = _current_trace.get()
            if trace:
                parent = trace.get_span(span.parent_id)
                _current_span.set(parent)
        else:
            _current_span.set(None)

    def _truncate_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Truncate data to configured max size."""
        if not self.config.capture_inputs and not self.config.capture_outputs:
            return {}

        result = {}
        for key, value in data.items():
            if isinstance(value, str) and len(value) > self.config.max_input_size:
                result[key] = value[: self.config.max_input_size] + "... [truncated]"
            else:
                result[key] = value

        return result

    def get_trace(self, trace_id: str) -> Trace | None:
        """Get a trace by ID."""
        return self._traces.get(trace_id)

    def get_current_trace(self) -> Trace | None:
        """Get the current active trace."""
        return _current_trace.get()

    def get_current_span(self) -> Span | None:
        """Get the current active span."""
        return _current_span.get()

    def get_all_traces(self) -> list[Trace]:
        """Get all traces."""
        with self._traces_lock:
            return list(self._traces.values())

    def clear_traces(self) -> None:
        """Clear all traces."""
        with self._traces_lock:
            self._traces.clear()

    # Debugger controls
    def pause(self) -> None:
        """Pause execution."""
        self.state.pause()

    def resume(self) -> None:
        """Resume execution."""
        self.state.resume()

    def step(self) -> None:
        """Step to next span."""
        self.state.step()

    @property
    def is_paused(self) -> bool:
        """Check if debugger is paused."""
        return self.state.is_paused

    # Callback registration
    def on_span(self, callback: Callable[[Span], None]) -> None:
        """Register a callback for span updates."""
        self._span_callbacks.append(callback)

    def on_trace(self, callback: Callable[[Trace], None]) -> None:
        """Register a callback for trace updates."""
        self._trace_callbacks.append(callback)


@contextmanager
def trace(
    name: str = "",
    kind: SpanKind = SpanKind.CUSTOM,
    input_data: dict[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
) -> Generator[Span, None, None]:
    """Context manager for creating a span."""
    debugger = get_debugger()
    span = debugger.start_span(
        name=name,
        kind=kind,
        input_data=input_data,
        attributes=attributes,
    )

    try:
        yield span
    except Exception as e:
        debugger.end_span(
            span,
            error=str(e),
        )
        raise
    else:
        debugger.end_span(span)


def traced(
    name: str | None = None,
    kind: SpanKind = SpanKind.CUSTOM,
    capture_args: bool = True,
    capture_result: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for tracing a function."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            input_data = {}
            if capture_args:
                input_data = {"args": args, "kwargs": kwargs}

            with trace(name=span_name, kind=kind, input_data=input_data) as span:
                result = func(*args, **kwargs)
                if capture_result:
                    span.output_data = {"result": result}
                return result

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            input_data = {}
            if capture_args:
                input_data = {"args": args, "kwargs": kwargs}

            debugger = get_debugger()
            span = debugger.start_span(
                name=span_name,
                kind=kind,
                input_data=input_data,
            )

            try:
                await debugger.state.async_wait_if_paused(span)
                result = await func(*args, **kwargs)
                if capture_result:
                    span.output_data = {"result": result}
                debugger.end_span(span)
                return result
            except Exception as e:
                debugger.end_span(span, error=str(e))
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator
