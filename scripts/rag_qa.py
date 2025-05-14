import os
import argparse
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

def main():
    parser = argparse.ArgumentParser(description="Ask a question about your medical insurance documents.")
    parser.add_argument("--query", "-q", required=True, help="The question to ask the Medical Insurance RAG agent")
    args = parser.parse_args()

    load_dotenv()

    VECTORSTORE_PATH = "vectorstore/insurance_faiss"

    embedding_model = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)

    llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    response = qa_chain.invoke({"query": args.query})

    print("\n📌 Answer:\n", response["result"])
    print("\n📎 Sources:")
    for doc in response["source_documents"]:
        print("-", doc.metadata.get("source", "Unknown"))

if __name__ == "__main__":
    main()
