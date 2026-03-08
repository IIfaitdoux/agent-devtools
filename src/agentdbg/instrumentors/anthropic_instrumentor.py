"""
Anthropic auto-instrumentation.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from agentdbg.config import get_model_cost
from agentdbg.core import get_debugger
from agentdbg.instrumentors.base import BaseInstrumentor
from agentdbg.models import CostInfo, SpanKind

_original_methods: dict[str, Any] = {}


class AnthropicInstrumentor(BaseInstrumentor):
    """Instrumentor for Anthropic API calls."""

    def instrument(self) -> None:
        """Wrap Anthropic client methods."""
        if self._instrumented:
            return

        try:
            import anthropic
        except ImportError:
            return

        self._wrap_messages()
        self._instrumented = True

    def uninstrument(self) -> None:
        """Restore original methods."""
        if not self._instrumented:
            return

        try:
            from anthropic.resources import Messages

            if "Messages.create" in _original_methods:
                Messages.create = _original_methods["Messages.create"]

            _original_methods.clear()
            self._instrumented = False
        except ImportError:
            pass

    def _wrap_messages(self) -> None:
        """Wrap messages.create."""
        try:
            from anthropic.resources import Messages

            original = Messages.create
            _original_methods["Messages.create"] = original

            @functools.wraps(original)
            def wrapped(self_client: Any, *args: Any, **kwargs: Any) -> Any:
                return _trace_anthropic_call(
                    original,
                    self_client,
                    "anthropic.messages.create",
                    *args,
                    **kwargs,
                )

            Messages.create = wrapped  # type: ignore

        except (ImportError, AttributeError):
            pass


def _trace_anthropic_call(
    original_fn: Callable,
    self_client: Any,
    name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Trace an Anthropic API call."""
    debugger = get_debugger()

    input_data = {
        "model": kwargs.get("model", "unknown"),
        "messages": kwargs.get("messages", []),
        "system": kwargs.get("system", ""),
        "max_tokens": kwargs.get("max_tokens"),
        "temperature": kwargs.get("temperature"),
    }
    input_data = {k: v for k, v in input_data.items() if v is not None and v != "" and v != []}

    span = debugger.start_span(
        name=name,
        kind=SpanKind.LLM_CALL,
        input_data=input_data,
        attributes={"provider": "anthropic"},
    )

    try:
        debugger.state.wait_if_paused(span)

        start_time = time.time()
        response = original_fn(self_client, *args, **kwargs)
        duration = time.time() - start_time

        output_data = _extract_anthropic_response(response)
        cost = _calculate_anthropic_cost(response, kwargs.get("model", ""))

        debugger.end_span(span, output_data=output_data, cost=cost)
        return response

    except Exception as e:
        debugger.end_span(span, error=str(e))
        raise


def _extract_anthropic_response(response: Any) -> dict[str, Any]:
    """Extract relevant data from Anthropic response."""
    try:
        output = {
            "id": getattr(response, "id", ""),
            "model": getattr(response, "model", ""),
            "stop_reason": getattr(response, "stop_reason", ""),
            "content": [],
            "tool_use": [],
        }

        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "text"):
                    output["content"].append({"type": "text", "text": block.text})
                elif hasattr(block, "type") and block.type == "tool_use":
                    output["tool_use"].append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

        return output
    except Exception:
        return {"raw": str(response)[:500]}


def _calculate_anthropic_cost(response: Any, model: str) -> CostInfo:
    """Calculate cost from Anthropic response."""
    cost = CostInfo(model=model)

    try:
        if hasattr(response, "usage"):
            usage = response.usage
            cost.input_tokens = getattr(usage, "input_tokens", 0)
            cost.output_tokens = getattr(usage, "output_tokens", 0)
            cost.total_tokens = cost.input_tokens + cost.output_tokens

            input_cost, output_cost = get_model_cost(model, cost.input_tokens, cost.output_tokens)
            cost.input_cost = input_cost
            cost.output_cost = output_cost
            cost.total_cost = input_cost + output_cost
    except Exception:
        pass

    return cost
