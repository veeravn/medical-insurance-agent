import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from scripts.load_and_split import load_and_split_docs
from scripts.embed_and_index import embed_and_index_docs
from scripts.agent_factory import create_insurance_agent

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

agent = create_insurance_agent(streaming=True, callbacks=[handler])

query = st.text_input("Enter your medical insurance question:", "What is covered under emergency hospitalization?")

if st.button("Submit"):
    with st.spinner("Retrieving and generating answer..."):
        raw_response = agent.run(query)
        response_text, sources = raw_response.split("\n\nSources:\n") if "\n\nSources:\n" in raw_response else (raw_response, "")

        st.subheader("Answer")
        st.write(response_text.strip())

        if sources:
            st.subheader("Sources")
            for line in sources.strip().split("\n"):
                if line.startswith("- "):
                    doc = line[2:].strip()
                    st.markdown(f"- [{doc}](./data/{doc})")
