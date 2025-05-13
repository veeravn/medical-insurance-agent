# Medical Insurance RAG Agent

This project is a Retrieval-Augmented Generation (RAG) pipeline built with **LangChain**, **Azure OpenAI**, and **Streamlit**, designed to answer questions from medical insurance policy documents.

---

## Project Structure

```
medical_insurance_rag/
├── data/                  # Place your PDF documents here
├── temp/                  # Intermediate split documents
├── vectorstore/           # FAISS index files
├── scripts/
│   ├── load_and_split.py        # Loads and splits PDFs
│   ├── embed_and_index.py       # Embeds text into FAISS vector store
│   ├── rag_qa.py                # CLI Q&A
├── app.py                 # Streamlit UI with live streaming
├── .env                   # Your Azure OpenAI credentials (not tracked)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Set Up Environment Variables**

Create a `.env` file with:

```env
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_API_VERSION=2023-12-01
```

### 3. **Prepare Documents**

Place your insurance PDF files into the `data/` folder.

### 4. **Build the Vector Store**

```bash
python scripts/load_and_split.py
python scripts/embed_and_index.py
```

### 5. **Run the Streamlit App**

```bash
streamlit run app.py
```

---

## Features

* Document chunking & semantic search (via FAISS)
* LLM answers grounded in retrieved insurance policy text
* Real-time streaming in the UI
* Azure OpenAI integration (GPT + Embeddings)

---

## Notes

* Uses `AzureOpenAIEmbeddings` and `AzureChatOpenAI` from `langchain-openai`
* Streaming implemented with `ConversationalRetrievalChain` and a custom `StreamHandler`
* Safe retry/backoff logic included for embedding under rate limits

---

## Security Warning

**Do not** check `.env` or any credential files into version control.

---

## Feedback

Feel free to suggest improvements or request features!
