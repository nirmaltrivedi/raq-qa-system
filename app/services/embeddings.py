"""
Embedding generation service using sentence-transformers.
"""
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from app.core.logging import app_logger as logger
from app.core.config import settings


class EmbeddingService:
    """
    Generate embeddings for text chunks using sentence-transformers.
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize embedding model."""
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        
        try:
            device = "cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)
            
            # Verify embedding dimension matches config
            test_embedding = self._model.encode("test")
            actual_dim = len(test_embedding)
            
            if actual_dim != settings.EMBEDDING_DIMENSION:
                logger.warning(
                    f"Model dimension ({actual_dim}) differs from config ({settings.EMBEDDING_DIMENSION}). "
                    f"Updating config to match model."
                )
                settings.EMBEDDING_DIMENSION = actual_dim
            
            logger.info(f"Model loaded successfully. Embedding dimension: {settings.EMBEDDING_DIMENSION}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings (shape: [len(texts), embedding_dim])
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Encoding {len(texts)} texts...")
        
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    def encode_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            List of floats (embedding vector)
        """
        embedding = self.encode([text])[0]
        return embedding.tolist()
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Add embeddings to chunk dictionaries.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            
        Returns:
            List of chunk dictionaries with added 'embedding' field
        """
        logger.info(f"Embedding {len(chunks)} chunks...")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.encode(texts, show_progress=len(chunks) > 100)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()
        
        logger.info("Embeddings added to all chunks")
        return chunks
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_name": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "device": str(self._model.device) if self._model else "not loaded",
            "max_seq_length": self._model.max_seq_length if self._model else None
        }


# Global instance
embedding_service = EmbeddingService()


def generate_embeddings(chunks: List[Dict]) -> List[Dict]:
    """
    Convenience function to generate embeddings for chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Chunks with embeddings added
    """
    return embedding_service.embed_chunks(chunks)