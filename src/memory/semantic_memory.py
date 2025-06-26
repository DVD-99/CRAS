# /src/memory/semantic_memory.py
import hashlib
from typing import List, Optional

from .interfaces import VectorStoreInterface, DocumentStoreInterface
from .semantic_memory_models import MemoryNote
from ..external_services.embedding_client import EmbeddingClient
from ..external_services.llm_client import LLMClient
from ..utils.logger_config import setup_logger

logger = setup_logger(__name__)

class SemanticMemoryManager:
    def __init__(
        self,
        vector_store: VectorStoreInterface,
        doc_store: DocumentStoreInterface,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient
    ):
        self.vector_store = vector_store
        self.doc_store = doc_store
        self.embedding_client = embedding_client
        self.llm_client = llm_client

    async def process_and_add_chunk(self, chunk: str, source_id: Optional[str] = None, source_type: str = "document"):
        """Processes a single text chunk and adds it to memory if it's new."""
        # 1. De-duplication: Hash the content to see if it exists
        chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
        existing_note = self.doc_store.get_note_by_hash(chunk_hash)

        if existing_note:
            logger.info(f"Chunk with hash {chunk_hash[:8]}... already exists. Skipping.")
            return

        logger.info(f"New chunk found. Processing and adding to memory...")

        # 2. Contextual Enrichment: Use LLM to summarize and extract keywords
        summary_prompt = "Summarize the following text in a single, concise sentence."
        keywords_prompt = "Extract up to 5 main keywords from the following text. Respond with only a comma-separated list."
        
        llm_summary = await self.llm_client.generate_text(prompt=chunk, system_prompt=summary_prompt, max_tokens=100)
        keywords_raw = await self.llm_client.generate_text(prompt=chunk, system_prompt=keywords_prompt, max_tokens=50)
        llm_keywords = [k.strip() for k in keywords_raw.split(',')]

        # 3. Create and store the new MemoryNote
        new_note = MemoryNote(
            chunk_hash=chunk_hash,
            source_document_id=source_id,
            content=chunk,
            llm_summary=llm_summary.strip(),
            llm_keywords=llm_keywords,
            source_type=source_type
        )
        self.doc_store.upsert_note(new_note)

        # 4. Generate embedding and add to vector store
        embedding = self.embedding_client.embed_query(chunk)
        metadata = {"source_id": str(source_id), "source_type": source_type}
        self.vector_store.add_documents(ids=[new_note.id], embeddings=[embedding], metadatas=[metadata])
        logger.info(f"Successfully added new memory note {new_note.id} for source {source_id}.")

    async def get_relevant_context(self, query: str, top_k: int = 5) -> Optional[str]:
        """
        Searches memory across ALL source types for relevant chunks and uses an LLM
        to distill them into a concise context block for the final prompt.
        """
        logger.info(f"Searching for context relevant to query: '{query}'")
        
        # 1. Search for similar documents in the vector store
        query_embedding = self.embedding_client.embed_query(query)
        doc_results = self.vector_store.search(query_embedding, top_k=top_k, where_filter={"source_type": {"$eq": "document"}})
        convo_results = self.vector_store.search(query_embedding, top_k=top_k, where_filter={"source_type": {"$eq": "conversation_summary"}})

        if not doc_results or not convo_results:
            logger.warning("No relevant chunks found in semantic memory.")
            return None

        retrieved_ids = [result[0] for result in doc_results]
        retrieved_ids += [result[0] for result in convo_results]
        retrieved_notes = self.doc_store.get_notes_by_ids(retrieved_ids)
        
        if not retrieved_notes:
            return None

        # 2. Context Scoping: Use LLM to extract only the most relevant sentences
        context_str = "\n\n---\n\n".join([f"Source: {note.source_document_id or 'Conversation'}\nContent: {note.content}" for note in retrieved_notes])
        
        system_prompt = (
            "You are an expert at extracting information. From the TEXT below, "
            "extract the exact sentences that are most relevant to answering the USER'S QUESTION. "
            "If no sentences are relevant, output nothing."
        )
        extraction_prompt = f"TEXT:\n{context_str}\n\nUSER'S QUESTION:\n{query}"
        
        distilled_context = await self.llm_client.generate_text(
            prompt=extraction_prompt,
            system_prompt=system_prompt,
            max_tokens=500
        )
        
        if not distilled_context.strip():
            logger.warning("LLM distillation returned no relevant sentences. Falling back to top result summary.")
            return retrieved_notes[0].llm_summary

        logger.info("Successfully distilled context using LLM.")
        return distilled_context

    async def save_conversation_summary(self, conversation_log: List[dict]):
        """
        Summarizes a conversation and adds the summary as a new memory note.
        """
        if len(conversation_log) < 2:
            return

        logger.info("Summarizing conversation to create a new memory.")
        conversation_text = "\n".join([f"{entry['role']}: {entry['content']}" for entry in conversation_log])
        
        system_prompt = "Summarize the key facts, entities, and conclusions from the following conversation."
        
        summary = await self.llm_client.generate_text(
            prompt=conversation_text,
            system_prompt=system_prompt,
            max_tokens=250
        )
        
        if summary:
            await self.process_and_add_chunk(chunk=summary, source_id="conversation_summary", source_type="conversation_summary")
