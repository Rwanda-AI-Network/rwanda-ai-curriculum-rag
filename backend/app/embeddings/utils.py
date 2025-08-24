"""
Embedding Utilities Module

This module provides utility functions for working with embeddings,
including text preprocessing, similarity calculations, and dimension reduction.

Key Features:
- Text preprocessing for embeddings
- Cosine similarity calculations
- Embedding dimension utilities
- Multilingual text handling
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class EmbeddingUtils:
    """
    Utility class for embedding operations.
    
    This class provides helper methods for text preprocessing,
    similarity calculations, and embedding manipulations.
    """
    
    @staticmethod
    def preprocess_text(text: str, language: str = "en") -> str:
        """
        Preprocess text for embedding generation.
        
        Implementation Guide:
        1. Clean text:
           - Remove special characters
           - Normalize whitespace
           - Handle encoding issues
        2. Language specific:
           - Apply language rules
           - Handle diacritics
        3. Return cleaned text
        
        Args:
            text: Input text to preprocess
            language: Language code (en/rw)
            
        Returns:
            Preprocessed text ready for embedding
        """
        # TODO: Implement text preprocessing
        # Basic cleaning for now
        text = re.sub(r'\s+', ' ', text.strip())
        return text.lower()
    
    @staticmethod
    def cosine_similarity(embedding1: List[float], 
                         embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Implementation Guide:
        1. Convert to numpy arrays
        2. Calculate dot product
        3. Calculate magnitudes
        4. Return cosine similarity
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # TODO: Implement proper cosine similarity
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        except Exception:
            return 0.0
    
    @staticmethod
    def batch_cosine_similarity(query_embedding: List[float],
                               corpus_embeddings: List[List[float]]) -> List[float]:
        """
        Calculate cosine similarities for a query against a corpus.
        
        Args:
            query_embedding: Query vector
            corpus_embeddings: List of corpus vectors
            
        Returns:
            List of similarity scores
        """
        # TODO: Implement batch similarity calculation
        similarities = []
        for corpus_emb in corpus_embeddings:
            similarities.append(EmbeddingUtils.cosine_similarity(query_embedding, corpus_emb))
        return similarities
    
    @staticmethod
    def reduce_dimensions(embeddings: List[List[float]], 
                         target_dim: int = 128) -> List[List[float]]:
        """
        Reduce embedding dimensions using PCA.
        
        Implementation Guide:
        1. Convert to numpy array
        2. Apply PCA transformation
        3. Return reduced embeddings
        
        Args:
            embeddings: Input embeddings
            target_dim: Target dimension count
            
        Returns:
            Dimension-reduced embeddings
        """
        # TODO: Implement PCA dimension reduction
        # For now, truncate if too large
        reduced = []
        for emb in embeddings:
            if len(emb) > target_dim:
                reduced.append(emb[:target_dim])
            else:
                reduced.append(emb)
        return reduced
    
    @staticmethod
    def chunk_text(text: str, 
                   chunk_size: int = 512, 
                   overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for embedding.
        
        Implementation Guide:
        1. Split by sentences
        2. Group into chunks
        3. Add overlap
        4. Ensure max size
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum chunk size
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        # TODO: Implement intelligent text chunking
        # Simple word-based chunking for now
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunks.append(' '.join(chunk_words))
            
        return chunks
    
    @staticmethod
    def save_embeddings(embeddings: Dict[str, List[float]], 
                       filepath: Path) -> bool:
        """
        Save embeddings to file.
        
        Args:
            embeddings: Dictionary of embeddings
            filepath: Output file path
            
        Returns:
            True if saved successfully
        """
        # TODO: Implement embedding serialization
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(embeddings, f)
            return True
        except Exception:
            return False
    
    @staticmethod
    def load_embeddings(filepath: Path) -> Dict[str, List[float]]:
        """
        Load embeddings from file.
        
        Args:
            filepath: Input file path
            
        Returns:
            Dictionary of loaded embeddings
        """
        # TODO: Implement embedding deserialization
        try:
            import json
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            return {}


def calculate_embedding_quality(embeddings: List[List[float]]) -> Dict[str, float]:
    """
    Calculate quality metrics for embeddings.
    
    Implementation Guide:
    1. Calculate variance
    2. Check for clustering
    3. Measure separability
    4. Return quality metrics
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        Dictionary with quality metrics
    """
    # TODO: Implement quality assessment
    if not embeddings:
        return {"variance": 0.0, "mean_magnitude": 0.0}
    
    # Basic metrics
    embeddings_array = np.array(embeddings)
    variance = float(np.var(embeddings_array))
    mean_magnitude = float(np.mean([np.linalg.norm(emb) for emb in embeddings]))
    
    return {
        "variance": variance,
        "mean_magnitude": mean_magnitude,
        "dimension": len(embeddings[0]) if embeddings else 0
    }


def detect_language(text: str) -> str:
    """
    Detect language of input text.
    
    Implementation Guide:
    1. Analyze character patterns
    2. Check for language markers
    3. Use statistical methods
    4. Return language code
    
    Args:
        text: Input text to analyze
        
    Returns:
        Language code (en/rw)
    """
    # TODO: Implement proper language detection
    # Simple heuristic for now
    kinyarwanda_markers = ['ubwoba', 'amahoro', 'ubushobozi', 'ubumenyi']
    
    text_lower = text.lower()
    for marker in kinyarwanda_markers:
        if marker in text_lower:
            return "rw"
    
    return "en"
