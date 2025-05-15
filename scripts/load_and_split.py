import os
import pickle
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_docs():
    pdf_dir = Path("data")
    split_dir = Path("temp")
    split_dir.mkdir(exist_ok=True)
    all_docs = []

    for pdf_file in pdf_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    with open(split_dir / "split_docs.pkl", "wb") as f:
        pickle.dump(split_docs, f)

    with open(split_dir / "timestamp.txt", "w") as f:
        latest = max([pdf_file.stat().st_mtime for pdf_file in pdf_dir.glob("*.pdf")])
        f.write(str(int(latest)))

    print("✅ Documents loaded and split.")
