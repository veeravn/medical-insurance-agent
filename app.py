import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

st.set_page_config(page_title="Medical Insurance Assistant", layout="wide")
st.title("🏥 Medical Insurance RAG Assistant")

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

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.tokens = []

    def on_llm_new_token(self, token, **kwargs):
        self.tokens.append(token)
        self.container.markdown("".join(self.tokens), unsafe_allow_html=True)

response_container = st.empty()
handler = StreamHandler(response_container)

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0,
    streaming=True,
    callbacks=[handler]
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

query = st.text_input("Enter your medical insurance question:", "What is covered under emergency hospitalization?")

if st.button("Submit"):
    with st.spinner("Retrieving and generating answer..."):
        result = qa_chain.invoke({"question": query, "chat_history": []})
        st.subheader("\n\nSources")
        for doc in result["source_documents"]:
            st.markdown(f"- {doc.metadata.get('source', 'Unknown Source')}")