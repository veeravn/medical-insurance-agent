import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from scripts.load_and_split import load_and_split_docs
from scripts.embed_and_index import embed_and_index_docs
from scripts.agent_graph import create_react_agent_with_memory
from langgraph.graph.message import add_messages

load_dotenv()

# Check for updated documents
VECTORSTORE_PATH = Path("vectorstore/insurance_faiss/index.faiss")
TIMESTAMP_PATH = Path("temp/timestamp.txt")
LATEST_PDF = max([f.stat().st_mtime for f in Path("data").glob("*.pdf")], default=0)

NEEDS_REBUILD = not VECTORSTORE_PATH.exists()
if TIMESTAMP_PATH.exists():
    with open(TIMESTAMP_PATH) as f:
        try:
            saved_ts = int(f.read().strip())
            if int(LATEST_PDF) > saved_ts:
                NEEDS_REBUILD = True
        except:
            NEEDS_REBUILD = True
else:
    NEEDS_REBUILD = True

if NEEDS_REBUILD:
    print("🔄 Changes detected. Rebuilding index...")
    load_and_split_docs()
    embed_and_index_docs()
else:
    print("✅ No changes in PDFs. Using cached vectorstore.")

# Streamlit app
st.set_page_config(page_title="Medical Insurance Assistant", layout="wide")
st.title("🏥 Medical Insurance Knowledge Base Agent")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.tokens = []

    def on_llm_new_token(self, token, **kwargs):
        self.tokens.append(token)
        self.container.markdown("".join(self.tokens), unsafe_allow_html=True)

response_container = st.empty()
handler = StreamHandler(response_container)

agent = create_react_agent_with_memory(streaming=True, callbacks=[handler])

# Session memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Enter your medical insurance question:", "What is covered under emergency hospitalization?")

if st.button("Submit"):
    with st.spinner("Retrieving and generating answer..."):
        result = agent.invoke({"messages": st.session_state.chat_history + ["user: " + query]})
        st.session_state.chat_history = add_messages(st.session_state.chat_history, ["user: " + query, "assistant: " + result])

        response_text, sources = result.split("\n\nSources:\n") if "\n\nSources:\n" in result else (result, "")

        st.subheader("Answer")
        st.write(response_text.strip())

        if sources:
            st.subheader("Sources")
            for line in sources.strip().split("\n"):
                if line.startswith("- "):
                    doc = line[2:].strip()
                    st.markdown(f"- [{doc}](./data/{doc})")
