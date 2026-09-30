import logging
import os
import time
from typing import Any, Dict, List

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from doc_processor import chunk_text

# ==========================================
# Configuration & Setup
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "models/gemini-embedding-2"
DB_PATH = "./rag_db"
COLLECTION_NAME = "knowledge_base"

# Load environment variables from .env file
load_dotenv()

# Initialize clients (genai will automatically pick up GEMINI_API_KEY from env)
client = genai.Client()
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def ingest_documents(
    docs: List[Dict[str, Any]], 
    batch_size: int = 8, 
    delay_between_batches: float = 2.0
) -> None:
    """
    Ingests document chunks with pacing and exponential backoff 
    to respect free-tier rate limits.
    """
    chunk_ids = []
    chunk_texts = []
    chunk_embeddings = []
    metadatas = []

    all_chunks = []
    all_metadatas = []

    # 1. Flatten all chunks from all pages
    for doc in docs:
        chunks = chunk_text(doc.get("text", ""))
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": doc.get("title", "Unknown")})

    if not all_chunks:
        logger.warning("No text extracted from PDF.")
        return

    total_chunks = len(all_chunks)
    logger.info(f"Total chunks to embed: {total_chunks}. Ingesting in batches of {batch_size}...")

    # 2. Batch process with rate-limit retries and pacing
    global_idx = 0
    total_batches = (total_chunks + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        batch_texts = all_chunks[start : start + batch_size]
        batch_meta = all_metadatas[start : start + batch_size]

        # Retry logic with exponential backoff for 429 errors
        max_retries = 5
        wait_seconds = 10.0

        for attempt in range(max_retries):
            try:
                # Process each chunk in the batch individually
                for text, meta in zip(batch_texts, batch_meta):
                    response = client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=text,
                        config=types.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT"
                        )
                    )
                    
                    chunk_ids.append(f"doc_{global_idx}")
                    chunk_texts.append(text)
                    chunk_embeddings.append(response.embeddings[0].values)
                    metadatas.append(meta)
                    global_idx += 1

                logger.info(f"Successfully processed batch {batch_num + 1}/{total_batches}")
                break  # Batch succeeded, break out of retry loop

            except ClientError as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Hit 429 rate limit on batch {batch_num + 1}. "
                            f"Waiting {wait_seconds}s before retry..."
                        )
                        time.sleep(wait_seconds)
                        wait_seconds *= 2 
                    else:
                        logger.error("Max retries reached. Exhausted rate limits.")
                        raise e
                else:
                    raise e
                    
        # Polite delay between successful calls to stay under RPM quotas
        time.sleep(delay_between_batches)

    # 3. Add to ChromaDB
    if chunk_texts:
        collection.add(
            ids=chunk_ids,
            documents=chunk_texts,
            embeddings=chunk_embeddings,
            metadatas=metadatas
        )
        logger.info(f"Done! Indexed {len(chunk_texts)} chunks successfully into ChromaDB.")
    else:
        logger.warning("No chunks were processed successfully.")