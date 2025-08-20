
# app/services/embeddings/sentence_transformer.py
from app.services.base.embedding_service import BaseEmbeddingService
from typing import List

class SentenceTransformerService(BaseEmbeddingService):
    """
    Sentence Transformer embedding service.
    Uses sentence-transformers library for creating text embeddings.
    """
    
    def _initialize(self):
        """
        Load the sentence transformer model.
        Download model if not already cached locally.
        """
        pass
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Convert text to embedding using sentence transformer.
        Args:
            text: Text to embed
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts to embeddings efficiently in batch.
        Args:
            texts: List of texts to embed
        Returns:
            List of embedding vectors
        """
        pass
