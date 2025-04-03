import os
import logging
import numpy as np
import openai
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating and comparing text embeddings."""
    
    def __init__(self, use_openai: bool = True):
        """
        Initialize the embedding service.
        
        Args:
            use_openai: Whether to use OpenAI API or local models for embeddings
        """
        self.use_openai = use_openai
        
        if use_openai:
            # Use OpenAI for embeddings
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                logger.warning("OPENAI_API_KEY not found. Falling back to local model.")
                self.use_openai = False
            else:
                openai.api_key = self.api_key
                logger.info("Using OpenAI for embeddings")
        
        if not self.use_openai:
            # Use local model for embeddings (sentence-transformers)
            logger.info("Loading local embedding model...")
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Local embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading local model: {e}")
                raise
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return np.array([])
        
        try:
            if self.use_openai:
                return self._get_openai_embeddings(texts)
            else:
                return self._get_local_embeddings(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _get_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using OpenAI API."""
        logger.info(f"Generating OpenAI embeddings for {len(texts)} texts")
        
        # Split into batches to avoid token limits (OpenAI has a limit per request)
        batch_size = 20
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}")
            
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=batch
            )
            
            # Extract embeddings from response
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def _get_local_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using local SentenceTransformer model."""
        logger.info(f"Generating local embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts)
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
            
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return float(similarity)
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        embeddings = self.get_embeddings([text1, text2])
        if len(embeddings) < 2:
            return 0.0
        return self.compute_similarity(embeddings[0], embeddings[1])
    
    def find_most_similar(self, query_text: str, candidate_texts: List[str]) -> Dict[str, Any]:
        """
        Find the most similar text in a list to the query text.
        
        Args:
            query_text: Text to compare against
            candidate_texts: List of texts to compare with
            
        Returns:
            Dict containing the most similar text, its index and similarity score
        """
        if not candidate_texts:
            return {"text": "", "index": -1, "score": 0.0}
            
        query_embedding = self.get_embeddings([query_text])[0]
        candidate_embeddings = self.get_embeddings(candidate_texts)
        
        # Calculate similarities
        similarities = []
        for i, emb in enumerate(candidate_embeddings):
            sim = self.compute_similarity(query_embedding, emb)
            similarities.append((i, sim))
        
        # Find the most similar
        best_idx, best_score = max(similarities, key=lambda x: x[1])
        
        return {
            "text": candidate_texts[best_idx],
            "index": best_idx,
            "score": best_score
        } 