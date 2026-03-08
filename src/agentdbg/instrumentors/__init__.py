"""
Auto-instrumentation for LLM providers.
"""

from agentdbg.instrumentors.base import BaseInstrumentor
from agentdbg.instrumentors.openai_instrumentor import OpenAIInstrumentor
from agentdbg.instrumentors.anthropic_instrumentor import AnthropicInstrumentor
from agentdbg.instrumentors.langchain_instrumentor import LangChainInstrumentor

__all__ = [
    "BaseInstrumentor",
    "OpenAIInstrumentor",
    "AnthropicInstrumentor",
    "LangChainInstrumentor",
]


def auto_instrument() -> None:
    """Auto-instrument all available providers."""
    OpenAIInstrumentor().instrument()
    AnthropicInstrumentor().instrument()
    LangChainInstrumentor().instrument()


def uninstrument() -> None:
    """Remove all instrumentation."""
    OpenAIInstrumentor().uninstrument()
    AnthropicInstrumentor().uninstrument()
    LangChainInstrumentor().uninstrument()
