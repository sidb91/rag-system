import streamlit as st
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Import our modular logic
from src.ingestion import process_documents
from src.retrieval import get_advanced_retriever
from src.generation import get_rag_chain
from src.visualization import visualize_embeddings

# Load API Keys
load_dotenv()

st.set_page_config(page_title="RAG your resume", layout="wide")

# --- UI Header ---
st.title("🚀 RAG your resume")
st.markdown("Hybrid Search + Semantic Re-ranking + PCA Visualization")

# --- Sidebar: Data & Visualization ---
with st.sidebar:
    st.header("1. Data Ingestion")
    if st.button("Build/Update Knowledge Base"):
        with st.spinner("Processing documents..."):
            # This triggers ingestion.py logic
            vector_db = process_documents()
            st.success("Database Ready!")
            st.session_state.vector_db = vector_db

    st.divider()
    st.header("2. Analysis")
    if st.button("Visualize Semantic Clusters"):
        if "vector_db" in st.session_state:
            fig = visualize_embeddings(st.session_state.vector_db)
            st.pyplot(fig)
        else:
            st.error("Please build the database first!")

# --- Main Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about your documents..."):
    if "vector_db" not in st.session_state:
        st.error("Knowledge base not initialized. Click 'Build' in the sidebar.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # START RAG PIPELINE
        with st.spinner("Thinking..."):
            # 1. Setup Retriever (Hybrid + Rerank)
            # Note: We pass chunks if we need BM25; here we simplify to the vector retriever
            retriever = get_advanced_retriever(st.session_state.vector_db, None)
            
            # 2. Setup Generation Chain (Gemini)
            rag_chain = get_rag_chain(retriever)
            
            # 3. Get Response
            response = rag_chain.invoke(prompt)
            full_response = response

        # Display Assistant Response
        with st.chat_message("assistant"):
            st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})