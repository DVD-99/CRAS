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
        # This automatically converts complex types like datetime and UUID
        # into JSON-compatible string formats before saving.
        note_dict = note.model_dump(mode='json')
        
        NoteQuery = Query()
        self.db.upsert(note_dict, NoteQuery.id == str(note.id))

    def get_note_by_id(self, note_id: UUID) -> Optional[MemoryNote]:
        NoteQuery = Query()
        result = self.db.get(NoteQuery.id == str(note_id))
        return MemoryNote(**result) if result else None
    
    def get_note_by_hash(self, chunk_hash: str) -> Optional[MemoryNote]:
        NoteQuery = Query()
        result = self.db.get(NoteQuery.chunk_hash == chunk_hash)
        return MemoryNote(**result) if result else None

    def get_notes_by_ids(self, note_ids: List[UUID]) -> List[MemoryNote]:
        if not note_ids:
            return []
        str_ids = [str(id) for id in note_ids]
        NoteQuery = Query()
        results = self.db.search(NoteQuery.id.one_of(str_ids))
        return [MemoryNote(**res) for res in results]
