import streamlit as st
from langserve import RemoteRunnable
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================
# CONFIG
# ============================================

SERVER_URL = "http://localhost:8000/rag"
VECTOR_DB_DIR = "vectorDBstore"

# Connect to LangServe RAG pipeline
rag = RemoteRunnable(SERVER_URL)

st.title("📚 RAG Chatbot with LangServe + LangSmith + Langchain Core + Streamlit")

# ============================================
# 1️⃣ Upload PDF
# ============================================

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded successfully!")

    if st.button("Build Vector Index"):
        # Save uploaded PDF temporarily
        TEMP_PDF = "temp_doc.pdf"
        with open(TEMP_PDF, "wb") as f:
            f.write(uploaded_file.read())

        st.info("🔍 Loading and chunking PDF...")

        # Step 1: Load PDF
        loader = PyPDFLoader(TEMP_PDF)
        docs = loader.load()

        # Step 2: Split chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        # Step 3: Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Step 4: Create Vector DB
        st.info("⚙️ Creating FAISS Vector DB...")
        DB = FAISS.from_documents(chunks, embeddings)

        # Save DB
        DB.save_local(VECTOR_DB_DIR)

        st.success("🎉 Vector DB created and saved successfully!")

# ============================================
# 2️⃣ Ask Questions to RAG via LangServe Server
# ============================================

st.subheader("💬 Ask a question")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if not os.path.exists(VECTOR_DB_DIR):
        st.error("Vector DB not found! Upload a PDF and build the index first.")
    else:
        st.info("Thinking...")

        try:
            response = rag.invoke(query)
            st.success("Response:")
            st.write(response)

        except Exception as e:
            st.error(f"Error: {str(e)}")
