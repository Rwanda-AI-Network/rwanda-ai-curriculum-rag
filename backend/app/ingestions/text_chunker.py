
# app/ingestion/text_chunker.py
from app.ingestion.base.processor import BaseDocumentProcessor
from typing import List, Dict, Any

class TextChunker:
    """
    Splits long documents into smaller chunks for better search and processing.
    Ensures chunks have good context boundaries (sentences, paragraphs).
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize text chunker with size parameters.
        Args:
            chunk_size: Target size of each chunk in tokens
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        Args:
            text: Text to split
            metadata: Metadata to attach to each chunk
        Returns:
            List of text chunks with metadata
        """
        pass
    
    def smart_split(self, text: str) -> List[str]:
        """
        Split text at natural boundaries (sentences, paragraphs).
        Args:
            text: Text to split
        Returns:
            List of text segments
        """
        pass
    
    def count_tokens(self, text: str) -> int:
        """
        Count approximate number of tokens in text.
        Args:
            text: Text to count
        Returns:
            Estimated token count
        """
        pass

