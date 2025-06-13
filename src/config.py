import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-f8', extra='ignore')

    # ASR Configuration (Lightning Whisper MLX)
    ASR_MODEL_NAME: str = "large-v3"  # Model size: "tiny", "base", "small", "medium", "large-v2", "large-v3"
    ASR_QUANTIZATION: Optional[str] = None # Quantization: "4bit", "8bit", or None
    ASR_BATCH_SIZE: int = 15 # Adjust based on your VRAM

    # LLM Configuration (MLX LM)
    # For MLX LM, this is typically a Hugging Face model identifier or local path
    LLM_MODEL_PATH: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit" # Choose the model

    # TTS Configuration (MeloTTS)
    TTS_MELOTTS_VOICE: str = "EN" # Example voice, MeloTTS supports various
    TTS_MELOTTS_SPEAKER_ID: Optional[str] = "EN-US" # e.g., "EN-Default" for some MeloTTS versions
    TTS_MELOTTS_DEVICE: str = "mps" # For Apple Silicon, can also be "cpu"

    # Embedding Model
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # Logging Level
    LOG_LEVEL: str = "INFO"

    HUGGING_FACE_TOKEN: Optional[str] = ""

settings = Settings()