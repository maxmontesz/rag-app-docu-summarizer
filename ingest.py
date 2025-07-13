# ingest.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# --- Configuration ---

load_dotenv()

# Specify the directory where your documents are stored
DOCS_DIR = "documents"
# Specify the directory where you want to store the vector database
CHROMA_DB_DIR = "chroma_db"

# --- Main Ingestion Function ---
def ingest_documents():
    """
    Ingests all documents from the DOCS_DIR, processes them,
    and stores them in a persistent ChromaDB vector store.
    """
    print("Starting document ingestion...")

    # 1. Load documents from the specified directory
    documents = []
    for filename in os.listdir(DOCS_DIR):
        file_path = os.path.join(DOCS_DIR, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                print(f"Loaded {filename}")
            elif filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
                print(f"Loaded {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue 

    if not documents:
        print("No documents found to ingest. Exiting.")
        return

    # 2. Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunked_documents)} chunks.")

    # 3. Initialize the embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. Create or load the vector store and add the documents
    
    vector_store = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    print("\n✅ Ingestion complete!")
    print(f"Vector store created/updated at: {CHROMA_DB_DIR}")


# --- Script Execution ---
if __name__ == "__main__":
    ingest_documents()