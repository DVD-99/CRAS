# /src/memory/semantic_memory_models.py
from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime

class MemoryNote(BaseModel):
    """
    Represents a single, enriched piece of information in the semantic memory.
    """
    id: UUID = Field(default_factory=uuid4)
    chunk_hash: str = Field(index=True)  # To prevent duplicates
    source_document_id: Optional[str] = None
    content: str

    # LLM-generated augmentations for better retrieval
    llm_summary: Optional[str] = None
    llm_keywords: List[str] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed_at: Optional[datetime] = None