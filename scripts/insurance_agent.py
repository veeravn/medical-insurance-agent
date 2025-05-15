from agent_factory import create_insurance_agent

agent = create_insurance_agent()

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question about your insurance policy (or 'exit'): ")
        if query.strip().lower() == "exit":
            break
        result = agent.run(query)
        print("\n📌 Answer:\n", result)
