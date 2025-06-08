# cras_project/cras_core/external_services/asr_client.py
import sys
import time
import traceback # Import traceback module
import asyncio
from typing import Optional
from tqdm import tqdm
from ..config import settings
from ..utils.logger_config import setup_logger

try:
    from lightning_whisper_mlx import LightningWhisperMLX
except ImportError:
    print("CRITICAL: lightning_whisper_mlx not found. ASRClient will not function.")
    LightningWhisperMLX = None

# Setup a logger specific to this module
logger = setup_logger(__name__, level=settings.LOG_LEVEL.upper() if hasattr(settings, 'LOG_LEVEL') else 'INFO')

class ASRClient:
    """
    Client for Automatic Speech Recognition using lightning-whisper-mlx.
    """
    def __init__(self,
                 model_name: Optional[str] = None,
                 quant: Optional[str] = None,
                 batch_size: Optional[int] = None):
        if not LightningWhisperMLX:
            logger.critical("lightning_whisper_mlx library is required but not installed. ASRClient cannot be initialized.")
            raise ImportError("lightning_whisper_mlx library is required for ASRClient.")

        self.model_name = model_name or settings.ASR_MODEL_NAME
        self.quant = quant or settings.ASR_QUANTIZATION
        self.batch_size = batch_size or settings.ASR_BATCH_SIZE
        
        logger.info(f"Initializing ASRClient with model: {self.model_name}, quant: {self.quant}, batch_size: {self.batch_size}")
        
        self.model = None
        try:
            self.model = LightningWhisperMLX(
                model=self.model_name,
                batch_size=self.batch_size,
                quant=self.quant
            )
            logger.info(f"LightningWhisperMLX model '{self.model_name}' (quant: {self.quant or 'None'}) initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing LightningWhisperMLX model '{self.model_name}': {e}")
            logger.error(traceback.format_exc()) # Log the full traceback
            raise # Re-raise the exception to indicate failure

    async def _spinner(self, message: str, start_time: float):
        """A simple, text-based spinner coroutine."""
        spinner_chars = "|/-\\"
        while True:
            for char in spinner_chars:
                # Use sys.stdout.write and \r to stay on the same line
                elapsed = time.time() - start_time
                sys.stdout.write(f'\r{message} {char} ({elapsed:.1f}s)')
                sys.stdout.flush()
                await asyncio.sleep(0.1)

    async def transcribe(self, audio_file_path: str, language: str) -> str:
        """
        Transcribes an audio file with a progress bar indicating activity.
        """
        if not self.model:
            logger.error("ASR model not initialized. Cannot transcribe.")
            return "Error: ASR model not initialized."

        logger.info(f"Preparing to transcribe audio file: {audio_file_path}")
        
        # This is the synchronous, blocking call we need to run
        blocking_transcribe_call = self.model.transcribe
        
        spinner_task = None
        start_time = time.time()

        # Create an indeterminate progress bar by setting `total=None`.
        # This will show an animated bar and elapsed time.
        try:
            loop = asyncio.get_running_loop()
            # Start the spinner as a concurrent task
            spinner_task = asyncio.create_task(self._spinner("Transcribing Audio...", start_time=start_time))
            # Run the blocking function in a separate thread so the UI doesn't freeze
            result = await loop.run_in_executor(
                None,  # Use the default thread pool executor
                blocking_transcribe_call,
                audio_file_path,
                language
            )

        except Exception as e:
            logger.error(f"Error during transcription of '{audio_file_path}': {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error: Could not transcribe audio. Error: {type(e).__name__}"
        finally:
            # Stop and clean up the spinner
            if spinner_task:
                spinner_task.cancel()
                # Clear the spinner line before printing final logs
                sys.stdout.write('\r' + ' ' * 40 + '\r') 
                sys.stdout.flush()

        duration = time.time() - start_time
        transcription = result.get("text", "").strip()
        
        logger.info(f"Transcription complete in {duration:.2f} seconds. Length: {len(transcription)} chars.")
        if not transcription:
            logger.warning(f"Transcription resulted in empty text for {audio_file_path}.")

        return transcription