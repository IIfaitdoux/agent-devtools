"""
Breakpoints and Pause Demo

This example demonstrates the real-time debugging features:
- Pause execution at any point
- Step through spans one at a time
- Set breakpoints based on conditions

Run with:
    agentdbg run examples/breakpoints_demo.py --pause-on-start
"""

import time
from agentdbg import trace, traced, get_debugger, SpanKind
from agentdbg.models import CostInfo


@traced(name="expensive_call", kind=SpanKind.LLM_CALL)
def make_expensive_call(iteration: int) -> str:
    """Simulate an expensive LLM call."""
    debugger = get_debugger()

    # Simulate processing time
    time.sleep(0.5)

    # Simulate cost
    span = debugger.get_current_span()
    if span:
        span.cost = CostInfo(
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
            total_cost=0.02,  # $0.02 per call
            model="gpt-4o",
        )

    return f"Response {iteration}"


@traced(name="risky_operation", kind=SpanKind.TOOL_CALL)
def risky_operation() -> str:
    """An operation that might fail."""
    import random
    if random.random() < 0.3:  # 30% chance of failure
        raise RuntimeError("Simulated failure!")
    return "Success"


def main():
    """Run the demo."""
    debugger = get_debugger()

    print("Breakpoints and Pause Demo")
    print("=" * 40)
    print()
    print("This demo will:")
    print("1. Make several 'expensive' LLM calls")
    print("2. Occasionally fail with errors")
    print("3. Accumulate costs")
    print()
    print("Open the AgentDBG UI to:")
    print("- Watch execution in real-time")
    print("- Pause at any moment")
    print("- Step through calls one by one")
    print("- Set breakpoints on cost or errors")
    print()

    # Set up a cost breakpoint
    debugger.state.add_breakpoint(
        lambda span: span.cost.total_cost > 0.05  # Pause if single call costs > $0.05
    )

    debugger.start_trace(name="breakpoint_demo", metadata={
        "description": "Demo of breakpoints and pause functionality"
    })

    try:
        for i in range(10):
            print(f"Iteration {i+1}/10...")

            # Make expensive call
            with trace(name=f"iteration_{i+1}", kind=SpanKind.AGENT_STEP) as span:
                result = make_expensive_call(i)
                print(f"  Got: {result}")

                # Sometimes do risky operation
                if i % 3 == 0:
                    try:
                        risky_result = risky_operation()
                        print(f"  Risky op: {risky_result}")
                    except RuntimeError as e:
                        print(f"  Risky op failed: {e}")

            # Small delay between iterations
            time.sleep(0.2)

    finally:
        debugger.end_trace()

    # Print summary
    traces = debugger.get_all_traces()
    if traces:
        trace = traces[-1]
        print()
        print("--- Final Summary ---")
        print(f"Total spans: {len(trace.spans)}")
        print(f"Total cost: ${trace.total_cost:.4f}")
        print(f"Total tokens: {trace.total_tokens}")

        error_spans = [s for s in trace.spans if s.error]
        print(f"Errors: {len(error_spans)}")


if __name__ == "__main__":
    main()
