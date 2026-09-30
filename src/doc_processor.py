import logging
from pathlib import Path
from typing import Union

from pypdf import PdfReader
from pypdf.errors import PdfReadError

# Set up a basic logger for the module
logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into sliding-window chunks based on word count.

    Args:
        text: The input string to be chunked.
        chunk_size: The maximum number of words per chunk.
        overlap: The number of overlapping words between consecutive chunks.

    Returns:
        A list of string chunks.

    Raises:
        ValueError: If overlap is greater than or equal to chunk_size.
    """
    if overlap >= chunk_size:
        raise ValueError("Overlap must be strictly less than chunk_size.")

    words = text.split()
    if not words:
        return []

    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
        
    return chunks


def load_pdf(file_path: Union[str, Path]) -> list[dict[str, str]]:
    """
    Reads a PDF file and extracts text along with page-level metadata.

    Args:
        file_path: The file path to the PDF document.

    Returns:
        A list of dictionaries, where each dictionary contains 'title' 
        and 'text' keys for a non-empty page.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        PdfReadError: If the file is corrupted or cannot be parsed by pypdf.
    """
    path = Path(file_path)
    
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    documents = []
    
    try:
        reader = PdfReader(path)
    except Exception as e:
        logger.error(f"Failed to read or parse PDF: {path}")
        raise PdfReadError(f"Could not read {path.name}. Error: {e}")

    file_name = path.name

    for page_number, page in enumerate(reader.pages, start=1):
        # Fallback to empty string if extract_text returns None
        text = page.extract_text() or ""
        clean_text = text.strip()
        
        if clean_text:
            documents.append({
                "title": f"{file_name} - Page {page_number}",
                "text": f"[Document: {file_name} | Page {page_number}]\n{clean_text}"
            })
            
    return documents