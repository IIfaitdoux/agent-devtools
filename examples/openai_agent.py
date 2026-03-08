"""
OpenAI Agent Example with AgentDBG

This example shows how to trace an OpenAI-powered agent with tool use.

Prerequisites:
    pip install openai

Run with:
    export OPENAI_API_KEY=your_key
    agentdbg run examples/openai_agent.py
"""

import os
import json
from agentdbg import trace, traced, get_debugger, SpanKind
from agentdbg.instrumentors import auto_instrument

# Auto-instrument OpenAI (done automatically by CLI, but shown here for clarity)
auto_instrument()


# Define tools
def get_weather(location: str) -> str:
    """Get weather for a location."""
    # Simulated weather data
    return json.dumps({
        "location": location,
        "temperature": 72,
        "conditions": "Sunny",
        "humidity": 45,
    })


def search_web(query: str) -> str:
    """Search the web."""
    # Simulated search results
    return json.dumps({
        "query": query,
        "results": [
            {"title": "Result 1", "snippet": "This is the first result..."},
            {"title": "Result 2", "snippet": "This is the second result..."},
        ]
    })


TOOLS = {
    "get_weather": get_weather,
    "search_web": search_web,
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


@traced(name="execute_tool", kind=SpanKind.TOOL_CALL)
def execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""
    if tool_name not in TOOLS:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    tool_fn = TOOLS[tool_name]
    return tool_fn(**arguments)


@traced(name="agent_loop", kind=SpanKind.AGENT_STEP)
def run_agent(user_message: str, max_iterations: int = 5) -> str:
    """Run the agent loop."""
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI not installed. Run: pip install openai")
        return "OpenAI not available"

    client = OpenAI()
    messages = [{"role": "user", "content": user_message}]

    for iteration in range(max_iterations):
        # Get LLM response (auto-instrumented by AgentDBG)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

        # Check if we have tool calls
        if assistant_message.tool_calls:
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                # Execute tool (traced)
                result = execute_tool(tool_name, arguments)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            # No tool calls, return the response
            return assistant_message.content

    return "Max iterations reached"


def main():
    """Main function."""
    print("OpenAI Agent with AgentDBG")
    print("=" * 40)

    # Initialize debugger
    debugger = get_debugger()
    debugger.start_trace(name="openai_agent_demo")

    try:
        # Run the agent
        user_query = "What's the weather like in San Francisco?"
        print(f"\nUser: {user_query}")

        response = run_agent(user_query)
        print(f"\nAgent: {response}")

    finally:
        debugger.end_trace()

    # Print cost summary
    traces = debugger.get_all_traces()
    if traces:
        trace = traces[-1]
        print(f"\n--- Trace Summary ---")
        print(f"Total spans: {len(trace.spans)}")
        print(f"Total cost: ${trace.total_cost:.6f}")
        print(f"Total tokens: {trace.total_tokens}")


if __name__ == "__main__":
    main()
