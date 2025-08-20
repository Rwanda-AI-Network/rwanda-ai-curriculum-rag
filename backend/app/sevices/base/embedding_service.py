
# app/services/base/embedding_service.py
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any
import numpy as np

class BaseEmbeddingService(ABC):
    """
    Base class for text embedding services.
    Embeddings convert text into numerical vectors for similarity search.
    
    To add a new embedding provider:
    1. Inherit from this class
    2. Implement the abstract methods
    3. Handle your specific model loading/API calls
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        """
        Initialize the embedding service.
        Args:
            model_name: Name of the embedding model
            config: Configuration (API keys, model path, etc.)
        """
        self.model_name = model_name
        self.config = config or {}
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """
        Initialize the specific embedding model.
        Examples:
        - Load SentenceTransformer model
        - Set up OpenAI API client
        - Download and cache model weights
        """
        pass
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """
        Convert a single text into an embedding vector.
        Args:
            text: Text to convert (question, document chunk, etc.)
        Returns:
            List of floating point numbers representing the text
        """
        pass
    
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts into embedding vectors efficiently.
        Args:
            texts: List of texts to convert
        Returns:
            List of embedding vectors, one for each input text
        """
        pass
    
    async def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        Args:
            text1: First text
            text2: Second text
        Returns:
            Similarity score between 0 and 1 (1 = identical meaning)
        """
        pass
    
    def get_embedding_dimension(self) -> int:
        """
        Get the size of the embedding vectors produced by this model.
        Returns:
            Number of dimensions in the embedding vector
        """
        pass
