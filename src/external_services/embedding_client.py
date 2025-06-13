# src/core/external_services/embedding_client.py
import time
from typing import List
import numpy as np

# This library will need to be installed: pip install sentence-transformers
# It will automatically use the GPU on Apple Silicon if available.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("CRITICAL: sentence_transformers not found. EmbeddingClient will not function.")
    SentenceTransformer = None

from ..config import settings
from ..utils.logger_config import setup_logger

logger = setup_logger(__name__, level=settings.LOG_LEVEL.upper() if hasattr(settings, 'LOG_LEVEL') else 'INFO')

class EmbeddingClient:
    """
    A client for generating text embeddings using SentenceTransformer models.
    """
    def __init__(self, model_name: str = None):
        if not SentenceTransformer:
            raise ImportError("sentence_transformers library is required for EmbeddingClient.")

        # Use the provided model name or get it from settings, with a default fallback
        self.model_name = model_name or getattr(settings, 'EMBEDDING_MODEL_NAME', None)
        self.model = None
        
        logger.info(f"Initializing EmbeddingClient with model: {self.model_name}")
        try:
            # The model is downloaded from Hugging Face automatically
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"SentenceTransformer model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model '{self.model_name}': {e}", exc_info=True)
            raise

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generates embeddings for a list of texts.
        """
        if not self.model:
            logger.error("Embedding model not initialized.")
            return []

        logger.info(f"Generating embeddings for {len(texts)} text chunk(s)...")
        start_time = time.time()
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            duration = time.time() - start_time
            logger.info(f"Successfully generated embeddings in {duration:.2f} seconds.")
            return embeddings
        except Exception as e:
            logger.error(f"Error during text embedding: {e}", exc_info=True)
            return []

    def embed_query(self, text: str) -> np.ndarray:
        """
        Generates an embedding for a single text query.
        """
        # Simply uses the batch embedding method for a single item
        result = self.embed_texts([text])
        return result[0] if result is not None and len(result) > 0 else np.array([])