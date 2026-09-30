"""
RAG Interactive Chat CLI

This module provides a command-line interface for interacting with a
Retrieval-Augmented Generation (RAG) system using ChromaDB.
"""

import sys
import logging
import chromadb
from chromadb.api.models.Collection import Collection

# Internal module imports
from doc_processor import load_pdf
from vector_store import ingest_documents
from generator import retrieve, generate_rag_response

# --- Configuration ---
EMBEDDING_MODEL = "models/gemini-embedding-2"
DB_PATH = "./rag_db"
COLLECTION_NAME = "knowledge_base"
DEFAULT_PDF_PATH = "BArtificialIntelligenceTER.pdf"
TOP_K_RETRIEVALS = 10

# Configure system logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def setup_database(db_path: str, collection_name: str, pdf_path: str) -> Collection:
    """
    Initialize ChromaDB, retrieve the collection, and ingest documents if empty.
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    if collection.count() == 0:
        logging.info("Database is empty. Starting document ingestion...")
        try:
            pdf_docs = load_pdf(pdf_path)
            ingest_documents(pdf_docs)
            logging.info("Ingestion complete.")
        except Exception as e:
            logging.error(f"Failed to ingest documents from {pdf_path}: {e}")
            sys.exit(1)
    else:
        logging.info(f"Found {collection.count()} chunks in the database. Skipping ingestion.")

    return collection


def chat_loop(collection: Collection) -> None:
    """
    Run the interactive terminal chat loop.
    """
    print("\n" + "=" * 32)
    print("    RAG Chat Initialized")
    print("    Type 'exit' or 'quit' to end")
    print("=" * 32 + "\n")
    
    while True:
        try:
            user_query = input("Question: ").strip()
            
            if user_query.lower() in ['exit', 'quit']:
                print("\nExiting chat. Goodbye!")
                break
                
            if not user_query:
                continue
                
            # Retrieve relevant chunks
            passages = retrieve(user_query, collection, top_k=TOP_K_RETRIEVALS)
            
            # Generate and print the response
            answer = generate_rag_response(user_query, passages)
            
            print("\n--- Answer ---")
            print(answer)
            print("-" * 14 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting chat. Goodbye!")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred during chat: {e}")


def main() -> None:
    """Main execution flow."""
    collection = setup_database(DB_PATH, COLLECTION_NAME, DEFAULT_PDF_PATH)
    chat_loop(collection)


if __name__ == "__main__":
    main()