# cras_project/cras_core/external_services/tts_client.py
import traceback
import os
from typing import Optional
from ..config import settings
from ..utils.logger_config import setup_logger
from ..config import settings
try:
    from melo.api import TTS as MeloTTS_API
except ImportError:
    print("Warning: MeloTTS not found. TTSClient will not function.")
    MeloTTS_API = None

logger = setup_logger(__name__, level=settings.LOG_LEVEL.upper() if hasattr(settings, 'LOG_LEVEL') else 'INFO')

class TTSClient:
    """
    Client for Text-to-Speech using MeloTTS.
    """
    def __init__(self,
                 language: Optional[str] = None,
                 speaker_id_name: Optional[str] = None,
                 device: Optional[str] = None):
        if not MeloTTS_API:
            raise ImportError("MeloTTS library is required for TTSClient. Please see installation instructions.")

        self.language = language or settings.TTS_MELOTTS_VOICE
        # This is the string name of the speaker, e.g., 'EN-US'
        self.speaker_id_name = speaker_id_name or settings.TTS_MELOTTS_SPEAKER_ID or 'EN-DEFAULT'
        self.device = device or settings.TTS_MELOTTS_DEVICE
        
        logger.info(f"Initializing TTSClient for language: {self.language}, target speaker: '{self.speaker_id_name}', device: {self.device}")
        
        self.melo_tts = None
        self.speaker_ids = {} # To store the mapping from name to integer ID
        try:
            self.melo_tts = MeloTTS_API(language=self.language, device=self.device)
            self.speaker_ids = self.melo_tts.hps.data.spk2id
            logger.info(f"MeloTTS API initialized. Available speakers: {list(self.speaker_ids.keys())}")
        except Exception as e:
            logger.error(f"Error initializing MeloTTS API: {e}")
            logger.error(traceback.format_exc())
            raise

    async def synthesize_speech(self, text: str, output_file_path: str) -> str:
        """
        Synthesizes speech from the given text and saves it to a file.
        """
        if not self.melo_tts:
            logger.error("MeloTTS API not initialized.")
            return "Error: MeloTTS API not initialized."

        
        speaker_int_id = self.speaker_ids[self.speaker_id_name]
        if speaker_int_id is None:
            logger.error(f"Speaker '{self.speaker_id_name}' not found for language '{self.language}'. Available: {list(self.speaker_ids.keys())}")
            return f"Error: Speaker '{self.speaker_id_name}' not found."

        logger.info(f"Synthesizing speech with speaker '{self.speaker_id_name}' (ID: {speaker_int_id}) for text: '{text[:50]}...'")
        try:
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            # Use the looked-up integer ID instead of the string name
            self.melo_tts.tts_to_file(text, speaker_int_id, output_file_path, speed=1.0)
            return output_file_path
        except Exception as e:
            # The error message from MeloTTS can sometimes be the speaker ID itself if it's invalid
            logger.error(f"Error during speech synthesis: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error: Could not synthesize speech. Details logged. Error: {e}"