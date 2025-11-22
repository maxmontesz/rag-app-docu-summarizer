# 📄 AI Document Assistant & Summarizer (RAG)

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF or TXT documents, ingest them into a local vector database, and perform Q&A or generate summaries using Google's Gemini AI.

Built with **Streamlit**, **LangChain**, **ChromaDB**, and **Google Gemini 1.5 Flash**.

---

## 🚀 Key Features

* **Document Ingestion:** Upload multiple PDF or Text files directly via the sidebar.
* **Smart Splitting:** Automatically chunks documents into manageable contexts (1000 chars with overlap).
* **Vector Search:** Uses Google Generative AI Embeddings and ChromaDB for semantic search.
* **Q&A Interface:** Ask specific questions about your documents and get cited sources.
* **Summarization:** Generate concise, bullet-point summaries of your uploaded content.
* **Local Persistence:** Embeddings are saved locally, so you don't have to re-process files every time you restart the app (provided the database persists).

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Google Gemini 1.5 Flash (`gemini-1.5-flash`)
* **Embeddings:** Google Generative AI Embeddings (`models/embedding-001`)
* **Vector Store:** ChromaDB
* **Orchestration:** LangChain

---

## 📂 Project Structure

```text
rag-app-docu-summarizer/
├── documents/          # Directory where uploaded files are stored temporarily
├── chroma_db/          # Directory where the vector database is persisted
├── app.py              # Main Streamlit application (Frontend)
├── core.py             # LLM setup, Prompts, and Chain definitions (Backend)
├── ingest.py           # Logic for loading, splitting, and embedding documents
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (API Keys)
└── Dockerfile          # Container configuration
