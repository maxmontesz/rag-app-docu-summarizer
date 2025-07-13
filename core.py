import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


# --- Configuration ---
load_dotenv()
CHROMA_DB_DIR = "chroma_db"

# --- Initialize model and vector store ---
def initialize_services():
    """Initializes and returns the LLM, retriever, and memory."""
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, convert_system_message_to_human=True)

    # Initialize the embedding function
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the vector store
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    # Set up the retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) 

    # Set up memory for conversational context
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return llm, retriever, memory

# --- Define Prompt Templates ---

# Prompt for the main Q&A task
QA_TEMPLATE = """
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
---
Context: {context}
---
Question: {question}
Answer:"""

# Prompt for summarization
SUMMARIZE_TEMPLATE = """
Based on the provided document, create a concise summary.
The summary should be in bullet points and highlight the key topics, findings, and conclusions.
---
Document: {context}
---
Summary (in bullet points):"""

# --- Create Chains ---
def create_qa_chain(llm, retriever):
    """Creates the main Question-Answering chain."""
    prompt = PromptTemplate(template=QA_TEMPLATE, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def create_summary_chain(llm, retriever):
    """Creates a chain specifically for summarization."""
    prompt = PromptTemplate(template=SUMMARIZE_TEMPLATE, input_variables=["context"])
    summary_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    return summary_chain