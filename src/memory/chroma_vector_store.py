# /src/memory/chroma_vector_store.py
import chromadb
from typing import List
from uuid import UUID
import numpy as np

from .interfaces import VectorStoreInterface
from ..utils.logger_config import setup_logger

logger = setup_logger(__name__)

class ChromaVectorStore(VectorStoreInterface):
    def __init__(self, db_path: str = "./data/chroma_db", collection_name: str = "cras_memory"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(f"ChromaDB client initialized at path: {db_path}")

    def add_documents(self, ids: List[UUID], embeddings: List[np.ndarray]):
        if not ids:
            return
        
        # ChromaDB expects string IDs and standard lists of floats
        str_ids = [str(id) for id in ids]
        list_embeddings = [emb.tolist() for emb in embeddings]
        
        self.collection.add(
            ids=str_ids,
            embeddings=list_embeddings
        )
        logger.info(f"Added {len(ids)} documents to ChromaDB collection.")

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[tuple[UUID, float]]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        if not results['ids'] or not results['distances']:
            return []

        # Results are nested for batch queries, we only have one query
        ids = [UUID(id_str) for id_str in results['ids'][0]]
        distances = results['distances'][0]
        
        return list(zip(ids, distances))