"""
LangChain Agent Example with AgentDBG

This example shows how to trace a LangChain agent with tools.

Prerequisites:
    pip install langchain langchain-openai

Run with:
    export OPENAI_API_KEY=your_key
    agentdbg run examples/langchain_agent.py
"""

import os
from agentdbg import get_debugger
from agentdbg.instrumentors import auto_instrument
from agentdbg.instrumentors.langchain_instrumentor import AgentDBGCallbackHandler

# Auto-instrument
auto_instrument()


def run_langchain_agent():
    """Run a LangChain agent with AgentDBG tracing."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.tools import tool
    except ImportError:
        print("LangChain not installed. Run: pip install langchain langchain-openai")
        return

    # Define tools
    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_current_time() -> str:
        """Get the current time."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tools = [calculator, get_current_time]

    # Create the agent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to tools."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm, tools, prompt)

    # Create executor with AgentDBG callback
    callback_handler = AgentDBGCallbackHandler()
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        callbacks=[callback_handler],
    )

    # Run the agent
    debugger = get_debugger()
    debugger.start_trace(name="langchain_agent_demo")

    try:
        result = agent_executor.invoke({
            "input": "What is 25 * 4 + 100? Also, what time is it?"
        })
        print(f"\nResult: {result['output']}")
    finally:
        debugger.end_trace()

    # Print summary
    traces = debugger.get_all_traces()
    if traces:
        trace = traces[-1]
        print(f"\n--- Trace Summary ---")
        print(f"Total spans: {len(trace.spans)}")
        print(f"Total cost: ${trace.total_cost:.6f}")


if __name__ == "__main__":
    print("LangChain Agent with AgentDBG")
    print("=" * 40)
    run_langchain_agent()
