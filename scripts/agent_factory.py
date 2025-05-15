import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

load_dotenv()

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

def get_qa_tool(llm, retriever):
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

    return Tool(
        name="MedicalInsuranceKnowledgeBase",
        func=invoke_with_sources,
        description="Use this to answer detailed questions about medical insurance policies."
    )

def create_insurance_agent(streaming=False, callbacks=None):
    llm = load_llm(streaming=streaming, callbacks=callbacks)
    retriever = load_retriever()
    tool = get_qa_tool(llm, retriever)
    return initialize_agent(
        tools=[tool],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )
