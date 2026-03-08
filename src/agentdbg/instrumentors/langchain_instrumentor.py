"""
LangChain auto-instrumentation.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable
from uuid import UUID

from agentdbg.core import get_debugger
from agentdbg.instrumentors.base import BaseInstrumentor
from agentdbg.models import CostInfo, SpanKind

_original_handlers: list[Any] = []


class LangChainInstrumentor(BaseInstrumentor):
    """Instrumentor for LangChain using callbacks."""

    def instrument(self) -> None:
        """Install LangChain callback handler."""
        if self._instrumented:
            return

        try:
            from langchain_core.callbacks import BaseCallbackHandler
            from langchain_core.callbacks.manager import CallbackManager
        except ImportError:
            try:
                from langchain.callbacks.base import BaseCallbackHandler
                from langchain.callbacks.manager import CallbackManager
            except ImportError:
                return

        # Create and register our callback handler
        handler = AgentDBGCallbackHandler()
        _original_handlers.append(handler)

        # Try to set as default callback
        try:
            import langchain
            if hasattr(langchain, "callbacks"):
                if not hasattr(langchain.callbacks, "_default_handlers"):
                    langchain.callbacks._default_handlers = []
                langchain.callbacks._default_handlers.append(handler)
        except Exception:
            pass

        self._instrumented = True

    def uninstrument(self) -> None:
        """Remove callback handler."""
        if not self._instrumented:
            return

        try:
            import langchain
            if hasattr(langchain, "callbacks") and hasattr(langchain.callbacks, "_default_handlers"):
                for handler in _original_handlers:
                    if handler in langchain.callbacks._default_handlers:
                        langchain.callbacks._default_handlers.remove(handler)
        except Exception:
            pass

        _original_handlers.clear()
        self._instrumented = False

    def get_callback_handler(self) -> Any:
        """Get the callback handler for manual use."""
        try:
            from langchain_core.callbacks import BaseCallbackHandler
        except ImportError:
            from langchain.callbacks.base import BaseCallbackHandler

        return AgentDBGCallbackHandler()


class AgentDBGCallbackHandler:
    """LangChain callback handler that creates spans for debugging."""

    def __init__(self) -> None:
        self._span_map: dict[str, Any] = {}  # run_id -> span
        self._debugger = get_debugger()

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts."""
        span = self._debugger.start_span(
            name=f"llm:{serialized.get('name', 'unknown')}",
            kind=SpanKind.LLM_CALL,
            input_data={
                "prompts": prompts,
                "model": serialized.get("kwargs", {}).get("model_name", "unknown"),
            },
            attributes={
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "provider": "langchain",
            },
        )
        self._span_map[str(run_id)] = span

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when LLM ends."""
        span = self._span_map.pop(str(run_id), None)
        if span:
            output_data = {}
            cost = CostInfo()

            try:
                if hasattr(response, "generations"):
                    output_data["generations"] = [
                        [{"text": g.text} for g in gen]
                        for gen in response.generations
                    ]

                if hasattr(response, "llm_output") and response.llm_output:
                    llm_output = response.llm_output
                    if "token_usage" in llm_output:
                        usage = llm_output["token_usage"]
                        cost.input_tokens = usage.get("prompt_tokens", 0)
                        cost.output_tokens = usage.get("completion_tokens", 0)
                        cost.total_tokens = usage.get("total_tokens", 0)
            except Exception:
                pass

            self._debugger.end_span(span, output_data=output_data, cost=cost)

    def on_llm_error(
        self,
        error: Exception,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        span = self._span_map.pop(str(run_id), None)
        if span:
            self._debugger.end_span(span, error=str(error))

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain starts."""
        span = self._debugger.start_span(
            name=f"chain:{serialized.get('name', serialized.get('id', ['unknown'])[-1])}",
            kind=SpanKind.CHAIN,
            input_data=inputs,
            attributes={
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "provider": "langchain",
            },
        )
        self._span_map[str(run_id)] = span

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when chain ends."""
        span = self._span_map.pop(str(run_id), None)
        if span:
            self._debugger.end_span(span, output_data=outputs)

    def on_chain_error(
        self,
        error: Exception,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when chain errors."""
        span = self._span_map.pop(str(run_id), None)
        if span:
            self._debugger.end_span(span, error=str(error))

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts."""
        span = self._debugger.start_span(
            name=f"tool:{serialized.get('name', 'unknown')}",
            kind=SpanKind.TOOL_CALL,
            input_data={"input": input_str},
            attributes={
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "provider": "langchain",
            },
        )
        self._span_map[str(run_id)] = span

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when tool ends."""
        span = self._span_map.pop(str(run_id), None)
        if span:
            self._debugger.end_span(span, output_data={"output": output})

    def on_tool_error(
        self,
        error: Exception,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when tool errors."""
        span = self._span_map.pop(str(run_id), None)
        if span:
            self._debugger.end_span(span, error=str(error))

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when agent takes an action."""
        span = self._debugger.start_span(
            name=f"agent_action:{getattr(action, 'tool', 'unknown')}",
            kind=SpanKind.AGENT_STEP,
            input_data={
                "tool": getattr(action, "tool", "unknown"),
                "tool_input": getattr(action, "tool_input", {}),
                "log": getattr(action, "log", ""),
            },
            attributes={
                "run_id": str(run_id),
                "provider": "langchain",
            },
        )
        self._span_map[f"action_{run_id}"] = span

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when agent finishes."""
        span = self._span_map.pop(f"action_{run_id}", None)
        if span:
            self._debugger.end_span(
                span,
                output_data={
                    "output": getattr(finish, "return_values", {}),
                    "log": getattr(finish, "log", ""),
                },
            )

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever starts."""
        span = self._debugger.start_span(
            name=f"retriever:{serialized.get('name', 'unknown')}",
            kind=SpanKind.RETRIEVAL,
            input_data={"query": query},
            attributes={
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "provider": "langchain",
            },
        )
        self._span_map[str(run_id)] = span

    def on_retriever_end(
        self,
        documents: list[Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when retriever ends."""
        span = self._span_map.pop(str(run_id), None)
        if span:
            self._debugger.end_span(
                span,
                output_data={
                    "documents": [
                        {
                            "content": getattr(d, "page_content", str(d))[:500],
                            "metadata": getattr(d, "metadata", {}),
                        }
                        for d in documents[:10]  # Limit to 10 docs
                    ],
                    "count": len(documents),
                },
            )

    def on_retriever_error(
        self,
        error: Exception,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when retriever errors."""
        span = self._span_map.pop(str(run_id), None)
        if span:
            self._debugger.end_span(span, error=str(error))
