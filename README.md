# AgentDBG

**Chrome DevTools for AI Agents** - Real-time debugging, pause, inspect, and step through your AI agent execution.

[![PyPI version](https://badge.fury.io/py/agentdbg.svg)](https://badge.fury.io/py/agentdbg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![AgentDBG Demo](https://via.placeholder.com/800x400?text=AgentDBG+Demo)

## The Problem

Building AI agents is hard. Debugging them is harder.

> "Your AI agent worked perfectly in testing. Then it hit production and called the wrong tool 14 times in a loop, burned $40 of API credits, and returned gibberish to your user. This is not a rare scenario. It's the default scenario."

Existing observability tools show you **what happened** after the fact. AgentDBG lets you **watch it happen** and **stop it** when things go wrong.

## Features

- **Real-time Visualization** - Watch your agent's execution unfold in real-time
- **Pause & Resume** - Stop execution at any point to inspect state
- **Step-through Debugging** - Advance one LLM call at a time
- **Breakpoints** - Pause on cost thresholds, errors, or custom conditions
- **Cost Tracking** - Real-time token and cost tracking per span
- **Auto-instrumentation** - Zero-config support for OpenAI, Anthropic, and LangChain
- **Local-first** - All data stays on your machine, sub-millisecond overhead

## Quick Start

### Installation

```bash
pip install agentdbg
```

### Basic Usage

Run any Python script with AgentDBG instrumentation:

```bash
agentdbg run my_agent.py
```

This will:
1. Auto-instrument OpenAI, Anthropic, and LangChain calls
2. Start the debugging UI at http://localhost:8766
3. Open your browser to the live trace viewer

### Manual Instrumentation

For more control, use the `@traced` decorator or `trace` context manager:

```python
from agentdbg import trace, traced, SpanKind

# Using decorator
@traced(name="process_query", kind=SpanKind.AGENT_STEP)
def process_query(query: str) -> str:
    # Your agent logic here
    return result

# Using context manager
with trace(name="llm_call", kind=SpanKind.LLM_CALL) as span:
    response = call_llm(messages)
    span.output_data = {"response": response}
```

## CLI Commands

```bash
# Run a script with debugging
agentdbg run script.py

# Run with cost limit (pause when exceeded)
agentdbg run script.py --cost-limit 1.0

# Run paused at start
agentdbg run script.py --pause-on-start

# Start server only (for external connections)
agentdbg server

# View recent traces
agentdbg traces

# Show statistics
agentdbg stats

# Clean up old traces
agentdbg cleanup --days 7
```

## Debugger Controls

### In the UI

- **Pause** - Stop execution at the current point
- **Resume** - Continue execution
- **Step** - Execute one span and pause again
- **Clear** - Remove all traces

### Breakpoints

Set breakpoints programmatically:

```python
from agentdbg import get_debugger

debugger = get_debugger()

# Pause when cost exceeds $0.50
debugger.state.add_breakpoint(
    lambda span: span.cost.total_cost > 0.50
)

# Pause on any error
debugger.state.add_breakpoint(
    lambda span: span.error is not None
)

# Pause on specific span name
debugger.state.add_breakpoint(
    lambda span: "dangerous_tool" in span.name
)
```

## Cost Tracking

AgentDBG automatically tracks costs for popular models:

```python
from agentdbg.config import MODEL_COSTS

# Supported models:
# - OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1, o1-mini
# - Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3-5-sonnet
```

Costs are displayed in real-time in the UI and can trigger breakpoints.

## Auto-Instrumentation

### OpenAI

```python
from openai import OpenAI
from agentdbg.instrumentors import auto_instrument

auto_instrument()  # Done automatically by CLI

client = OpenAI()
# All calls are now traced automatically
response = client.chat.completions.create(...)
```

### Anthropic

```python
from anthropic import Anthropic
from agentdbg.instrumentors import auto_instrument

auto_instrument()

client = Anthropic()
# All calls are now traced automatically
response = client.messages.create(...)
```

### LangChain

```python
from langchain_openai import ChatOpenAI
from agentdbg.instrumentors.langchain_instrumentor import AgentDBGCallbackHandler

# Use the callback handler
handler = AgentDBGCallbackHandler()
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[handler],
)
```

## Storage

Traces are stored locally in SQLite:

```python
from agentdbg.storage import SQLiteStorage

storage = SQLiteStorage(".agentdbg/traces.db")

# Get recent traces
traces = storage.get_traces(limit=10)

# Get statistics
stats = storage.get_stats()
print(f"Total cost: ${stats['total_cost']:.2f}")
print(f"Total tokens: {stats['total_tokens']:,}")

# Clean up old data
storage.delete_old_traces(days=7)
```

## Configuration

```python
from agentdbg import DebugConfig, AgentDebugger

config = DebugConfig(
    # Server
    host="127.0.0.1",
    port=8765,
    ui_port=8766,

    # Auto-pause
    auto_pause_on_error=True,
    auto_pause_on_cost=1.0,  # Pause at $1.00
    auto_pause_on_tokens=100000,  # Pause at 100k tokens

    # Data capture
    capture_inputs=True,
    capture_outputs=True,
    max_input_size=10000,
    max_output_size=10000,
)

debugger = AgentDebugger(config=config)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Agent Code                         │
│  (OpenAI, Anthropic, LangChain, Custom)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    AgentDBG SDK                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │ Instrumentor│  │    Core     │  │      Storage        │   │
│  │  (auto-wrap)│  │(trace/span) │  │     (SQLite)        │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    WebSocket Server                          │
│              (Real-time streaming)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Web UI                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │
│  │ Trace List │  │ Span Tree  │  │ Inspector (State/Cost) │ │
│  └────────────┘  └────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Comparison

| Feature | AgentDBG | Langfuse | LangSmith |
|---------|----------|----------|-----------|
| Real-time pause/resume | ✅ | ❌ | ❌ |
| Step-through debugging | ✅ | ❌ | ❌ |
| Breakpoints | ✅ | ❌ | ❌ |
| Local-first | ✅ | ⚠️ Self-host | ❌ |
| Zero-config | ✅ | ⚠️ | ⚠️ |
| Open source | ✅ | ✅ | ❌ |
| Cost tracking | ✅ | ✅ | ✅ |

## Development

```bash
# Clone the repo
git clone https://github.com/agentdbg/agentdbg.git
cd agentdbg

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src tests
mypy src
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with frustration and love by developers who've spent too many hours staring at logs wondering why their agent decided to search Google 47 times in a row.

---

**Stop guessing why your agent failed. See every thought. Pause anywhere. Fix it live.**
