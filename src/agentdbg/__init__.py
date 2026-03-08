"""
AgentDBG - Chrome DevTools for AI Agents

Real-time debugging, pause, inspect, and step through your AI agent execution.
"""

from agentdbg.core import AgentDebugger, trace, traced, get_debugger
from agentdbg.models import Span, Trace, SpanKind, SpanStatus, CostInfo
from agentdbg.config import DebugConfig

__version__ = "0.1.0"
__all__ = [
    "AgentDebugger",
    "trace",
    "traced",
    "get_debugger",
    "Span",
    "Trace",
    "SpanKind",
    "SpanStatus",
    "CostInfo",
    "DebugConfig",
    "__version__",
]
