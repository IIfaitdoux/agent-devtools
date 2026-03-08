"""
Basic AgentDBG Usage Example

This example shows how to use AgentDBG to trace and debug AI agent execution.

Run with:
    agentdbg run examples/basic_usage.py

Or import and use directly in your code.
"""

from agentdbg import trace, traced, get_debugger, SpanKind, DebugConfig


# Example 1: Using the @traced decorator
@traced(name="process_query", kind=SpanKind.AGENT_STEP)
def process_user_query(query: str) -> str:
    """Process a user query through multiple steps."""

    # Simulate thinking
    analysis = analyze_query(query)

    # Simulate tool use
    if "search" in query.lower():
        results = search_web(query)
        return f"Found: {results}"

    return f"Analysis: {analysis}"


@traced(name="analyze", kind=SpanKind.LLM_CALL)
def analyze_query(query: str) -> str:
    """Analyze the query (simulates LLM call)."""
    import time
    time.sleep(0.1)  # Simulate API latency
    return f"Query is about: {query[:20]}"


@traced(name="search", kind=SpanKind.TOOL_CALL)
def search_web(query: str) -> list:
    """Search the web (simulates tool call)."""
    import time
    time.sleep(0.05)
    return ["result1", "result2", "result3"]


# Example 2: Using the context manager
def manual_tracing_example():
    """Example using manual trace context manager."""
    debugger = get_debugger()

    # Start a trace for this operation
    debugger.start_trace(name="manual_example", metadata={"user": "demo"})

    with trace(name="step_1", kind=SpanKind.CHAIN) as span:
        span.add_event("starting", {"info": "Beginning step 1"})

        with trace(name="nested_llm_call", kind=SpanKind.LLM_CALL) as llm_span:
            # Simulate LLM work
            import time
            time.sleep(0.05)

        span.add_event("completed", {"info": "Step 1 done"})

    debugger.end_trace()


# Example 3: Cost tracking
def cost_tracking_example():
    """Example showing cost tracking."""
    from agentdbg.models import CostInfo

    debugger = get_debugger()
    debugger.start_trace(name="cost_example")

    # Create a span with cost information
    span = debugger.start_span(
        name="gpt-4o-call",
        kind=SpanKind.LLM_CALL,
        input_data={"messages": [{"role": "user", "content": "Hello!"}]},
    )

    # Simulate the response with cost
    cost = CostInfo(
        input_tokens=10,
        output_tokens=25,
        total_tokens=35,
        input_cost=0.000025,
        output_cost=0.00025,
        total_cost=0.000275,
        model="gpt-4o",
    )

    debugger.end_span(
        span,
        output_data={"content": "Hello! How can I help you today?"},
        cost=cost,
    )

    debugger.end_trace()

    # Print summary
    trace = debugger.get_all_traces()[-1]
    print(f"Total cost: ${trace.total_cost:.6f}")
    print(f"Total tokens: {trace.total_tokens}")


if __name__ == "__main__":
    print("Running AgentDBG Examples...")
    print()

    print("Example 1: Decorated functions")
    result = process_user_query("Please search for the latest news")
    print(f"Result: {result}")
    print()

    print("Example 2: Manual tracing")
    manual_tracing_example()
    print("Manual tracing complete")
    print()

    print("Example 3: Cost tracking")
    cost_tracking_example()
    print()

    print("All examples complete!")
    print("Open the AgentDBG UI to see the traces.")
