import streamlit as st
import os
import numpy as np
import asyncio
from uuid import uuid4
from src.utils.logger_config import setup_logger
from src.ingestion.document_parser import TextProcessor
from src.external_services.embedding_client import EmbeddingClient
from src.external_services.llm_client import LLMClient
from src.external_services.asr_client import ASRClient

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

# --- Load Models ---
llm_client = get_llm_client()
asr_client = get_asr_client()
embedding_client = get_embedding_client()
text_processor = get_text_processor()


# --- Session State Management ---
# Initialize session state variables if they don't exist
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    # This is our simplified, in-memory vector store for this session
    st.session_state.vector_store = {
        "chunks": [],
        "embeddings": []
    }

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

def run_async(awaitable):
    """
    Runs an awaitable coroutine and blocks until it is complete.
    This is a replacement for asyncio.run() in Streamlit.
    """
    try:
        # Try to get the running event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # If no loop is running, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(awaitable)

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

            # 3. Embed and Store
            if chunks:
                chunk_embeddings = embedding_client.embed_texts(chunks)
                st.session_state.vector_store["chunks"].extend(chunks)
                st.session_state.vector_store["embeddings"].extend(chunk_embeddings)
            
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
            run_async(process_files(uploaded_files))

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
            # Embed the user's query
            query_embedding = embedding_client.embed_query(prompt)

            # Find relevant context from the vector store
            context_chunks = find_relevant_chunks(query_embedding)
            
            if not context_chunks:
                response_text = "I couldn't find any relevant information in the uploaded documents to answer your question. Please try processing a file first."
                st.markdown(response_text)
            else:
                # Build the prompt for the LLM
                context_str = "\n\n---\n\n".join(context_chunks)
                system_prompt = "You are a helpful research assistant. Answer the user's question based *only* on the following context provided. If the answer is not in the context, say so."
                full_prompt = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{prompt}"
                
                # Generate the response
                response_text = run_async(llm_client.generate_text(full_prompt, system_prompt=system_prompt))
                st.markdown(response_text)
            
    # Add assistant's response to session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})