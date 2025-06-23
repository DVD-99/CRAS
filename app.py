import torch
torch.classes.__path__ = []
import streamlit as st
import os
import numpy as np
import asyncio
import nest_asyncio
from uuid import uuid4
from src.utils.logger_config import setup_logger
from src.ingestion.document_parser import TextProcessor
from src.external_services.embedding_client import EmbeddingClient
from src.external_services.llm_client import LLMClient
from src.external_services.asr_client import ASRClient
from src.memory.semantic_memory import SemanticMemoryManager
from src.memory.chroma_vector_store import ChromaVectorStore
from src.memory.tinydb_doc_store import TinyDBDocumentStore

nest_asyncio.apply()
# --- Page Configuration ---
st.set_page_config(
    page_title="CRAS - Cognitive Research Assistant System",
    page_icon="🧠",
    layout="wide"
)

# --- Logger ---
logger = setup_logger("CRAS_App")

# --- Caching and Model Loading ---
# Use Streamlit's cache to load heavy models only once
@st.cache_resource
def get_llm_client():
    logger.info("Loading LLM Client...")
    return LLMClient()

@st.cache_resource
def get_asr_client():
    logger.info("Loading ASR Client...")
    return ASRClient()

@st.cache_resource
def get_embedding_client():
    logger.info("Loading Embedding Client...")

    class EmbeddingClient:
        def embed_texts(self, texts):
            logger.warning("Using placeholder embedding client. All embeddings will be random.")
            return [np.random.rand(384) for _ in texts] # Assuming a 384-dim model
        def embed_query(self, text):
            return np.random.rand(384)
    return EmbeddingClient()

@st.cache_resource
def get_text_processor():
    logger.info("Loading Text Processor Client...")
    return TextProcessor()

@st.cache_resource
def get_memory_manager():
    logger.info("Initializing Semantic Memory Manager...")
    vector_store = ChromaVectorStore()
    doc_store = TinyDBDocumentStore()
    # Get the already loaded clients
    embedding_client = get_embedding_client()
    llm_client = get_llm_client()
    return SemanticMemoryManager(vector_store, doc_store, embedding_client, llm_client)

# --- Load Models ---
llm_client = get_llm_client()
asr_client = get_asr_client()
embedding_client = get_embedding_client()
text_processor = get_text_processor()
memory_manager = get_memory_manager()

def run_async(awaitable):
    """
    Runs an awaitable coroutine in a new thread with its own event loop.
    This is a robust replacement for asyncio.run() in Streamlit.
    """
    result = None
    exception = None

    def run_in_loop():
        nonlocal result, exception
        try:
            result = asyncio.run(awaitable)
        except Exception as e:
            exception = e

    thread = threading.Thread(target=run_in_loop)
    thread.start()
    thread.join()

    if exception:
        raise exception
        
    return result

# --- Session State Initialization ---
# This block ensures all necessary keys exist before they are accessed.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "conversation_log" not in st.session_state:
    st.session_state.conversation_log = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "last_processed_audio" not in st.session_state:
    st.session_state.last_processed_audio = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper Functions ---
def find_relevant_chunks(query_embedding, top_k=3):
    """Finds the most relevant text chunks from the vector store."""
    if not st.session_state.vector_store["embeddings"]:
        return []

    embeddings = np.array(st.session_state.vector_store["embeddings"])
    # Cosine similarity calculation
    similarities = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))
    
    # Get the indices of the top_k most similar chunks
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    
    relevant_chunks = [st.session_state.vector_store["chunks"][i] for i in top_k_indices]
    return relevant_chunks

async def process_files(uploaded_files):
    """Processes uploaded files: parse, chunk, embed, and store."""
    for uploaded_file in uploaded_files:
        # Avoid re-processing the same file
        if uploaded_file.name in st.session_state.processed_files:
            continue

        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Save the file temporarily to get a file path
            temp_dir = "./data/temp_files"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 1. Parse / Transcribe
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension in [".mp3", ".wav", ".m4a"]:
                text = await asr_client.transcribe(file_path, language = "en")
                text = text_processor.clean_text(text)
            elif file_extension == ".pdf":
                text = text_processor.extract_text_from_pdf(file_path)
            else:
                text = text_processor.read_text_file(file_path)

            if not text:
                st.sidebar.error(f"Failed to extract text from {uploaded_file.name}")
                continue

            # 2. Chunk
            chunks = text_processor.chunk_text(text=text)
            logger.info(f"Extracted {len(chunks)} chunks from {uploaded_file.name}")

            # 3. Embed and Store
            if chunks:
                for chunk in chunks:
                    await memory_manager.process_and_add_chunk(chunk, source_id=uploaded_file.name)
            # if chunks:
            #     chunk_embeddings = embedding_client.embed_texts(chunks)
            #     st.session_state.vector_store["chunks"].extend(chunks)
            #     st.session_state.vector_store["embeddings"].extend(chunk_embeddings)
            
            # Mark as processed
            st.session_state.processed_files.add(uploaded_file.name)
            st.sidebar.success(f"Processed {uploaded_file.name} ({len(chunks)} chunks)")
            
            # Clean up temp file
            os.remove(file_path)


# --- UI Layout ---
st.title("🧠 CRAS - Cognitive Research Assistant System")

# Sidebar for file uploads
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or Audio files",
        type=["pdf", "txt", "mp3", "wav", "m4a"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Files"):
            # Run the async function using asyncio
            asyncio.run(process_files(uploaded_files))

    st.header("Processed Files")
    if st.session_state.processed_files:
        for f_name in st.session_state.processed_files:
            st.markdown(f"- `{f_name}`")
    else:
        st.info("No files processed yet for this session.")


# --- Chat Interface ---
# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get new user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
             # This one call replaces embedding the query, searching, and building context
            distilled_context = asyncio.run(memory_manager.get_relevant_context(prompt))

            if not distilled_context:
                response_text = "I'm sorry, I couldn't find any relevant information..."
            else:
                system_prompt = "You are a helpful research assistant. Answer the user's question based *only* on the following distilled context provided."
                full_prompt = f"DISTILLED CONTEXT:\n{distilled_context}\n\nQUESTION:\n{prompt}"
                response_text = asyncio.run(llm_client.generate_text(full_prompt, system_prompt=system_prompt))
            
            st.markdown(response_text)

        # After generating the response, save a summary of the turn
        st.session_state.conversation_log.append({"role": "user", "content": prompt})
        st.session_state.conversation_log.append({"role": "assistant", "content": response_text})
        asyncio.run(memory_manager.save_conversation_summary(st.session_state.conversation_log[-2:])) # Summarize the last Q&A pair

    # Add assistant's response to session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})