from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

DATA_DIR = "../data"
all_docs = []

for filename in os.listdir(DATA_DIR):
    if filename.lower().endswith(".pdf"):
        path = os.path.join(DATA_DIR, filename)
        loader = PyPDFLoader(path)
        documents = loader.load()
        all_docs.extend(documents)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(all_docs)

os.makedirs("../temp", exist_ok=True)
with open("../temp/split_docs.pkl", "wb") as f:
    import pickle
    pickle.dump(split_docs, f)
print("✅ All PDF documents split and saved.")