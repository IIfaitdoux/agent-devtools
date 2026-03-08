"""
OpenAI auto-instrumentation.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from agentdbg.config import get_model_cost
from agentdbg.core import get_debugger
from agentdbg.instrumentors.base import BaseInstrumentor
from agentdbg.models import CostInfo, SpanKind

# Store original methods for uninstrumentation
_original_methods: dict[str, Any] = {}


class OpenAIInstrumentor(BaseInstrumentor):
    """Instrumentor for OpenAI API calls."""

    def instrument(self) -> None:
        """Wrap OpenAI client methods."""
        if self._instrumented:
            return

        try:
            import openai
        except ImportError:
            return  # OpenAI not installed

        self._wrap_completions()
        self._wrap_chat_completions()
        self._wrap_embeddings()

        self._instrumented = True

    def uninstrument(self) -> None:
        """Restore original OpenAI methods."""
        if not self._instrumented:
            return

        try:
            import openai

            # Restore original methods
            for key, original in _original_methods.items():
                parts = key.split(".")
                obj = openai
                for part in parts[:-1]:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                if obj:
                    setattr(obj, parts[-1], original)

            _original_methods.clear()
            self._instrumented = False
        except ImportError:
            pass

    def _wrap_completions(self) -> None:
        """Wrap completions.create."""
        try:
            from openai.resources import Completions

            original = Completions.create
            _original_methods["resources.Completions.create"] = original

            @functools.wraps(original)
            def wrapped(self_client: Any, *args: Any, **kwargs: Any) -> Any:
                return _trace_openai_call(
                    original,
                    self_client,
                    "openai.completions.create",
                    SpanKind.LLM_CALL,
                    *args,
                    **kwargs,
                )

            Completions.create = wrapped  # type: ignore
        except (ImportError, AttributeError):
            pass

    def _wrap_chat_completions(self) -> None:
        """Wrap chat.completions.create."""
        try:
            from openai.resources.chat import Completions as ChatCompletions

            # Sync version
            original_create = ChatCompletions.create
            _original_methods["resources.chat.Completions.create"] = original_create

            @functools.wraps(original_create)
            def wrapped_create(self_client: Any, *args: Any, **kwargs: Any) -> Any:
                return _trace_openai_call(
                    original_create,
                    self_client,
                    "openai.chat.completions.create",
                    SpanKind.LLM_CALL,
                    *args,
                    **kwargs,
                )

            ChatCompletions.create = wrapped_create  # type: ignore

        except (ImportError, AttributeError):
            pass

    def _wrap_embeddings(self) -> None:
        """Wrap embeddings.create."""
        try:
            from openai.resources import Embeddings

            original = Embeddings.create
            _original_methods["resources.Embeddings.create"] = original

            @functools.wraps(original)
            def wrapped(self_client: Any, *args: Any, **kwargs: Any) -> Any:
                return _trace_openai_call(
                    original,
                    self_client,
                    "openai.embeddings.create",
                    SpanKind.EMBEDDING,
                    *args,
                    **kwargs,
                )

            Embeddings.create = wrapped  # type: ignore
        except (ImportError, AttributeError):
            pass


def _trace_openai_call(
    original_fn: Callable,
    self_client: Any,
    name: str,
    kind: SpanKind,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Trace an OpenAI API call."""
    debugger = get_debugger()

    # Extract input data
    input_data = {
        "model": kwargs.get("model", "unknown"),
        "messages": kwargs.get("messages", []),
        "prompt": kwargs.get("prompt", ""),
        "temperature": kwargs.get("temperature"),
        "max_tokens": kwargs.get("max_tokens"),
    }

    # Clean up None values
    input_data = {k: v for k, v in input_data.items() if v is not None and v != "" and v != []}

    span = debugger.start_span(
        name=name,
        kind=kind,
        input_data=input_data,
        attributes={"provider": "openai"},
    )

    try:
        # Wait if paused
        debugger.state.wait_if_paused(span)

        # Make the actual call
        start_time = time.time()
        response = original_fn(self_client, *args, **kwargs)
        duration = time.time() - start_time

        # Extract output data
        output_data = _extract_openai_response(response)

        # Calculate cost
        cost = _calculate_openai_cost(response, kwargs.get("model", ""))

        debugger.end_span(span, output_data=output_data, cost=cost)
        return response

    except Exception as e:
        debugger.end_span(span, error=str(e))
        raise


def _extract_openai_response(response: Any) -> dict[str, Any]:
    """Extract relevant data from OpenAI response."""
    try:
        # Chat completion response
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                return {
                    "content": choice.message.content,
                    "role": choice.message.role,
                    "finish_reason": choice.finish_reason,
                    "tool_calls": [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in (choice.message.tool_calls or [])
                    ] if hasattr(choice.message, "tool_calls") and choice.message.tool_calls else [],
                }
            elif hasattr(choice, "text"):
                return {
                    "text": choice.text,
                    "finish_reason": choice.finish_reason,
                }

        # Embedding response
        if hasattr(response, "data") and response.data:
            return {
                "embedding_count": len(response.data),
                "model": response.model,
            }

        return {"raw": str(response)[:500]}
    except Exception:
        return {"raw": str(response)[:500]}


def _calculate_openai_cost(response: Any, model: str) -> CostInfo:
    """Calculate cost from OpenAI response."""
    cost = CostInfo(model=model)

    try:
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            cost.input_tokens = getattr(usage, "prompt_tokens", 0)
            cost.output_tokens = getattr(usage, "completion_tokens", 0)
            cost.total_tokens = getattr(usage, "total_tokens", 0)

            input_cost, output_cost = get_model_cost(model, cost.input_tokens, cost.output_tokens)
            cost.input_cost = input_cost
            cost.output_cost = output_cost
            cost.total_cost = input_cost + output_cost
    except Exception:
        pass

    return cost
