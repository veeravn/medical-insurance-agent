import os
from dotenv import load_dotenv
from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages

load_dotenv()

# Load retriever
def load_retriever():
    embedding_model = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    vectorstore = FAISS.load_local(
        "vectorstore/insurance_faiss",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_type="similarity", k=5)

# Load LLM
def load_llm(streaming=False, callbacks=None):
    return AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        temperature=0,
        streaming=streaming,
        callbacks=callbacks or []
    )

# Build LangGraph ReAct Agent
def create_react_agent_with_memory(streaming=False, callbacks=None):
    llm = load_llm(streaming=streaming, callbacks=callbacks)
    retriever = load_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    def invoke_with_sources(query):
        result = qa_chain.invoke({"query": query})
        sources = "\n\nSources:\n" + "\n".join([
            f"- {doc.metadata.get('source', 'Unknown Source')}" for doc in result.get("source_documents", [])
        ])
        return result["result"] + sources

    tool = Tool(
        name="MedicalInsuranceKnowledgeBase",
        func=invoke_with_sources,
        description="Answer medical insurance policy questions using a knowledge base."
    )

    app = create_react_agent(
    model=llm,
    tools=[tool],
    version="v1"
    )
    return app

# For CLI testing
if __name__ == "__main__":
    agent = create_react_agent_with_memory()
    history = []
    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.strip().lower() == "exit":
            break
        result = agent.invoke({"messages": history + ["user: " + query]})
        history = add_messages(history, ["user: " + query, "assistant: " + result])
        print("\n📌 Answer:\n", result)
