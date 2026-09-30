import os
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Constants
EMBEDDING_MODEL = "models/gemini-embedding-2"
GENERATION_MODEL = "models/gemini-3.8-flash"

# Load environment variables
load_dotenv()

# Initialize Gemini client securely using environment variables
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not found. Please set it in your .env file.")

client = genai.Client(api_key=api_key)


def retrieve(query: str, collection: chromadb.Collection, top_k: int = 10) -> list[dict]:
    """
    Generates a query embedding and retrieves the top-k most relevant documents.

    Args:
        query (str): The search query.
        collection (chromadb.Collection): The ChromaDB collection to search against.
        top_k (int, optional): The number of top results to retrieve. Defaults to 10.

    Returns:
        list[dict]: A list of dictionaries containing 'content' and 'metadata' 
                    for each retrieved document.
    """
    query_resp = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    
    query_vector = query_resp.embeddings[0].values

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    retrieved = []
    # Safeguard against empty results
    if not results.get("documents") or not results.get("metadatas"):
        return retrieved

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({"content": doc, "metadata": meta})
        
    return retrieved


def generate_rag_response(query: str, retrieved_passages: list[dict]) -> str:
    """
    Formats retrieved context into a prompt and queries the Gemini model.

    Args:
        query (str): The user's original question.
        retrieved_passages (list[dict]): A list of dictionaries containing 
                                         'content' and 'metadata' (including 'source').

    Returns:
        str: The generated response from the LLM.
    """
    if not retrieved_passages:
        return "I do not have enough context to answer this question."

    # Use .get() to prevent KeyError if metadata is missing fields
    context_blocks = "\n\n".join(
        [f"[Source: {p.get('metadata', {}).get('source', 'Unknown')}]\n{p.get('content', '')}" 
         for p in retrieved_passages]
    )

    system_instruction = (
        "You are an accurate factual assistant. Answer user questions strictly using "
        "the provided Context. If the context does not contain enough information, state that "
        "you do not know rather than speculating. Always cite sources."
    )

    prompt = f"Context:\n{context_blocks}\n\nUser Question: {query}\n"

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2
        )
    )
    
    return response.text