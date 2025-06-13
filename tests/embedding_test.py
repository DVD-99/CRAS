# tests/embedding_test.py
import sys
import os
import asyncio
import numpy as np

# Add the project root to the Python path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.external_services.embedding_client import EmbeddingClient
from src.utils.logger_config import setup_logger

# Setup logger for the test
logger = setup_logger("EmbeddingClientTest")

async def main():
    """Main function to run the embedding client tests."""
    logger.info("--- Starting EmbeddingClient Test ---")
    
    try:
        # 1. Initialize the client
        # This will download the model from Hugging Face on the first run.
        logger.info("Initializing EmbeddingClient...")
        client = EmbeddingClient()
        logger.info("Client initialized successfully.")

        # 2. Test batch embedding with embed_texts
        logger.info("\n--- Testing embed_texts (batch processing) ---")
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Apple's MLX framework is optimized for Apple Silicon.",
            "Sentence embeddings represent text in a high-dimensional vector space."
        ]
        logger.info(f"Input sentences: {sample_texts}")
        
        embeddings = client.embed_texts(sample_texts)
        
        if embeddings is not None and len(embeddings) > 0:
            embeddings_array = np.array(embeddings)
            logger.info(f"Successfully generated {len(embeddings)} embeddings.")
            logger.info(f"Shape of the embedding matrix: {embeddings_array.shape}")
            # all-MiniLM-L6-v2 has a dimension of 384
            assert embeddings_array.shape == (3, 384), "Embedding matrix shape is incorrect!"
            logger.info("Batch embedding test PASSED.")
        else:
            logger.error("Batch embedding test FAILED. No embeddings were returned.")

        # 3. Test single query embedding with embed_query
        logger.info("\n--- Testing embed_query (single query) ---")
        query_text = "What is a cognitive research assistant?"
        logger.info(f"Input query: '{query_text}'")
        
        query_embedding = client.embed_query(query_text)

        if query_embedding is not None and query_embedding.size > 0:
            logger.info("Successfully generated query embedding.")
            logger.info(f"Shape of the query vector: {query_embedding.shape}")
            assert query_embedding.shape == (384,), "Query embedding shape is incorrect!"
            logger.info("Single query test PASSED.")
        else:
            logger.error("Single query test FAILED. No embedding was returned.")
            
    except Exception as e:
        logger.error(f"An error occurred during the test: {e}", exc_info=True)

    logger.info("\n--- EmbeddingClient Test Finished ---")

if __name__ == "__main__":
    asyncio.run(main())
