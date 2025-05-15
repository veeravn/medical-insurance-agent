import os
import pickle
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

def embed_and_index_docs():

    with open("./temp/split_docs.pkl", "rb") as f:
        split_docs = pickle.load(f)

    embedding_model = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    embedded_docs = []
    for i, doc in enumerate(split_docs):
        success = False
        retries = 0
        while not success and retries < 5:
            try:
                embedding_model.embed_documents([doc.page_content])  # warm-up call to raise if failing
                embedded_docs.append(doc)
                success = True
            except Exception as e:
                print(f"[Retry {retries+1}] Error embedding chunk {i}: {e}")
                retries += 1
                time.sleep(5 * retries)  # exponential backoff

    vectorstore = FAISS.from_documents(embedded_docs, embedding_model)
    os.makedirs("./vectorstore", exist_ok=True)
    vectorstore.save_local("./vectorstore/insurance_faiss")
    print(f"✅ Vector store created with {len(embedded_docs)} embedded chunks.")