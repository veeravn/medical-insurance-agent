from agent_graph import create_react_agent_with_memory
from langgraph.graph.message import add_messages

agent = create_react_agent_with_memory()

if __name__ == "__main__":
    history = []
    while True:
        query = input("\nAsk a question about your insurance policy (or 'exit'): ")
        if query.strip().lower() == "exit":
            break
        result = agent.invoke({"messages": history + ["user: " + query]})
        history = add_messages(history, ["user: " + query, "assistant: " + result])
        print("\n📌 Answer:\n", result)