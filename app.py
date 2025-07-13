import streamlit as st
import os
from ingest import ingest_documents
from core import initialize_services, create_qa_chain, create_summary_chain

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- App State Management ---
if 'services_initialized' not in st.session_state:
    st.session_state.services_initialized = False

# --- UI Components ---
st.title("📄 Smart Content Summarizer & Q&A System")
st.markdown("Upload your documents, and I'll help you find answers and create summaries!")

# Sidebar for document upload and processing
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Save uploaded files to the 'documents' directory
        save_path = "documents"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for file in uploaded_files:
            with open(os.path.join(save_path, file.name), "wb") as f:
                f.write(file.getbuffer())
        
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

    if st.button("Process Documents"):
        with st.spinner("Processing... This might take a moment."):
            try:
                ingest_documents()
                st.session_state.services_initialized = True
                st.success("Documents processed and ready!")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

# Main content area for Q&A and Summarization
st.header("2. Ask Questions & Get Summaries")

if not st.session_state.services_initialized:
    st.warning("Please upload and process your documents first.")
else:
    # Initialize services once documents are processed
    llm, retriever, memory = initialize_services()
    qa_chain = create_qa_chain(llm, retriever)
    summary_chain = create_summary_chain(llm, retriever)

    # Q&A Section
    st.subheader("❓ Ask a Question")
    question = st.text_input("Ask anything about your documents:")

    if question:
        with st.spinner("Finding an answer..."):
            try:
                response = qa_chain({"query": question})
                st.markdown("### Answer")
                st.write(response["result"])

                with st.expander("See Sources"):
                    for doc in response["source_documents"]:
                        st.info(f"Source: {os.path.basename(doc.metadata.get('source', 'N/A'))}")
                        st.text(doc.page_content)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Summarization Section
    st.subheader("📝 Get a Summary")
    if st.button("Generate Summary for All Documents"):
        with st.spinner("Creating summary..."):
            try:
                # To summarize, we retrieve all chunks by asking a generic question
                # A better approach for single docs would be to pass content directly
                all_docs_content = "Summarize the key information from all available documents."
                response = summary_chain({"query": all_docs_content})
                st.markdown("### Summary")
                st.write(response["result"])
            except Exception as e:
                st.error(f"An error occurred: {e}")