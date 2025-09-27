# /src/memory/tinydb_doc_store.py
from tinydb import TinyDB, Query
from typing import List, Optional
from uuid import UUID

from .interfaces import DocumentStoreInterface
from .semantic_memory_models import MemoryNote
from ..utils.logger_config import setup_logger

logger = setup_logger(__name__)

class TinyDBDocumentStore(DocumentStoreInterface):
    def __init__(self, db_path: str = "./data/document_store.json"):
        self.db = TinyDB(db_path)
        logger.info(f"TinyDB client initialized at path: {db_path}")

    def upsert_note(self, note: MemoryNote):
        note_dict = note.model_dump(mode='json')
        self.db.upsert(note_dict, Query().id == str(note.id))

    def get_note_by_id(self, note_id: UUID) -> Optional[MemoryNote]:
        result = self.db.get(Query().id == str(note_id))
        return MemoryNote(**result) if result else None
    
    def get_note_by_hash(self, chunk_hash: str) -> Optional[MemoryNote]:
        result = self.db.get(Query().chunk_hash == chunk_hash)
        return MemoryNote(**result) if result else None

    def get_notes_by_ids(self, note_ids: List[UUID]) -> List[MemoryNote]:
        if not note_ids: return []
        results = self.db.search(Query().id.one_of([str(id) for id in note_ids]))
        return [MemoryNote(**res) for res in results]

    def count(self) -> int:
        """Returns the total number of items in the database."""
        return len(self.db)
