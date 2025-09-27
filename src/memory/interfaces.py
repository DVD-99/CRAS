# /src/memory/interfaces.py
from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
import numpy as np
from .semantic_memory_models import MemoryNote

class VectorStoreInterface(ABC):
    @abstractmethod
    def add_documents(self, ids: List[UUID], embeddings: List[np.ndarray], metadatas: List[dict]):
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int, where_filter: Optional[dict] = None) -> List[tuple[UUID, float]]:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

class DocumentStoreInterface(ABC):
    @abstractmethod
    def upsert_note(self, note: MemoryNote):
        pass

    @abstractmethod
    def get_note_by_id(self, note_id: UUID) -> Optional[MemoryNote]:
        pass
    
    @abstractmethod
    def get_note_by_hash(self, chunk_hash: str) -> Optional[MemoryNote]:
        pass

    @abstractmethod
    def get_notes_by_ids(self, note_ids: List[UUID]) -> List[MemoryNote]:
        pass

    @abstractmethod
    def count(self) -> int:
        pass