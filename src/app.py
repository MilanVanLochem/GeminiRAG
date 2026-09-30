"""
RAG Chatbot Application

A Streamlit-based document Question-Answering chatbot utilizing ChromaDB
for vector storage and Gemini for retrieval-augmented generation.
"""

import os
import chromadb
import streamlit as st

from doc_processor import load_pdf
from vector_store import ingest_documents
from generator import retrieve, generate_rag_response


# --- Configuration Constants ---
EMBEDDING_MODEL = "models/gemini-embedding-2"
DB_PATH = "./rag_db"
COLLECTION_NAME = "knowledge_base"
DEFAULT_DOCUMENT_PATH = "BArtificialIntelligenceTER.pdf"
RETRIEVAL_TOP_K = 10


@st.cache_resource(show_spinner=False)
def initialize_database() -> chromadb.Collection:
    """
    Initializes the ChromaDB client and loads the vector collection.
    Triggers the document ingestion pipeline if the collection is empty.
    """
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    
    if collection.count() == 0:
        with st.spinner("📚 Initializing Knowledge Base. This may take a moment..."):
            pdf_docs = load_pdf(DEFAULT_DOCUMENT_PATH)
            ingest_documents(pdf_docs)
        st.success("Knowledge Base is ready!")
        
    return collection


def apply_custom_theme():
    """Injects custom CSS that respects system Light/Dark mode."""
    st.markdown("""
        <style>
        /* Headers - Serif font for an academic feel */
        h1, h2, h3 {
            font-family: 'Georgia', serif;
        }
        
        /* Chat message containers - using CSS variables to adapt to Light/Dark mode */
        .stChatMessage {
            background-color: var(--secondary-background-color);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
        }
        
        /* Custom horizontal divider */
        hr {
            border-top: 2px solid #7BA05B; /* Sage Green accent */
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit application logic."""
    # 1. Page Configuration
    st.set_page_config(
        page_title="AI BSc TER Navigator", 
        page_icon="🎓", 
        layout="wide"
    )
    
    apply_custom_theme()

    # 2. Sidebar Configuration
    with st.sidebar:
        st.title("🎓 TER Navigator")
        st.markdown("### BSc Artificial Intelligence")
        st.markdown("---")
        st.info(
            "**Welcome!** This assistant is designed to help you navigate the "
            "Teaching and Examination Regulations (TER) Academic year 2026-2027."
        )
        st.markdown("### About the Base")
        st.markdown(
            "- **Document:** `BArtificialIntelligenceTER.pdf`\n"
            "- **Engine:** Gemini RAG + ChromaDB"
        )
        st.warning("Always cross-reference answers with the official TER document for final academic decisions.")

    # 3. Main Header
    st.title("Academic Regulations Assistant")
    st.markdown("*Ask questions about courses, grading, thesis requirements, and program rules.*")
    st.markdown("---")

    # 4. Initialize backend components
    collection = initialize_database()

    # 5. Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI BSc TER Assistant. What would you like to know about the regulations today?"}
        ]

    # 6. Render previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 7. Handle User Input
    if user_query := st.chat_input("E.g., What are the requirements to pass the Bachelor thesis?"):
        
        # Display user message and add to history
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Consulting the regulations..."):
                try:
                    # Retrieve relevant chunks
                    passages = retrieve(user_query, collection, top_k=RETRIEVAL_TOP_K)
                    
                    # Generate the response
                    answer = generate_rag_response(user_query, passages)
                    
                    # Display and save the answer
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"An error occurred while processing your request: {e}")


if __name__ == "__main__":
    main()