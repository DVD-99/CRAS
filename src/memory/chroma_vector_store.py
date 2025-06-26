# /src/memory/chroma_vector_store.py
import chromadb
from typing import List, Optional
from uuid import UUID
import numpy as np

from .interfaces import VectorStoreInterface
from ..utils.logger_config import setup_logger

logger = setup_logger(__name__)

class ChromaVectorStore(VectorStoreInterface):
    def __init__(self, db_path: str = "./data/chroma_db", collection_name: str = "cras_memory"):
        collection_metadata = {
        "hnsw:space": "cosine", # Use cosine similarity
        "hnsw:construction_ef": 200, # Build a higher quality index
        "hnsw:M": 32 # Create more links
        }
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata=collection_metadata)
        logger.info(f"ChromaDB client initialized at path: {db_path}")

    def add_documents(self, ids: List[UUID], embeddings: List[np.ndarray], metadatas: List[dict]):
        if not ids:
            return
        
        str_ids = [str(id) for id in ids]
        list_embeddings = [emb.tolist() for emb in embeddings]
        
        self.collection.add(
            ids=str_ids,
            embeddings=list_embeddings,
            metadatas=metadatas  # Add metadata here
        )
        logger.info(f"Added {len(ids)} documents with metadata to ChromaDB.")

    def search(self, query_embedding: np.ndarray, top_k: int, where_filter: Optional[dict] = None) -> List[tuple[UUID, float]]:
        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k
        }
        if where_filter:
            query_params["where"] = where_filter # Add the where filter if provided

        results = self.collection.query(**query_params)
        if not results['ids'] or not results['distances']:
            return []

        # Results are nested for batch queries, we only have one query
        ids = [UUID(id_str) for id_str in results['ids'][0]]
        distances = results['distances'][0]
        
        return list(zip(ids, distances))