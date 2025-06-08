import traceback
from lightning_whisper_mlx import LightningWhisperMLX
from src.config import settings
from src.utils.logger_config import setup_logger
from src.external_services.asr_client import ASRClient

logger = setup_logger(__name__, level=settings.LOG_LEVEL.upper() if hasattr(settings, 'LOG_LEVEL') else 'INFO')
# --- Example Usage (for testing this file directly) ---
async def main_test_asr():

    if LightningWhisperMLX:
        logger.info("Starting ASRClient test...")
        try:

            # Test: Test with quantization set to None to isolate the issue
            logger.info("Attempting to initialize ASRClient with quant=None...")
            asr_client_no_quant = ASRClient(quant=None) # Key test
            logger.info("ASRClient (quant=None) initialized.")
            
            # You need a real audio file for this to work.
            # Create a dummy one or use a real one.
            import os
            audio_dir = "./data/audio_samples"
            os.makedirs(audio_dir, exist_ok=True)
            # Use a real audio file path here
            audio_path = os.path.join(audio_dir, "IS1004a.Mix-Lapel.mp3") # <--- TODO:REPLACE THIS

            if not os.path.exists(audio_path):
                logger.warning(f"Test audio file not found: {audio_path}. Please create it or provide a valid path.")
                logger.warning("Skipping transcription test.")
                return

            logger.info(f"Attempting transcription of '{audio_path}' with default language as English and no quantization...")
            text_no_quant = await asr_client_no_quant.transcribe(audio_path, language = "en")
            logger.info(f"Transcription (no quant): {text_no_quant}")

            # If the above works, then try with your original quantization settings (if different)
            if settings.ASR_QUANTIZATION:
                logger.info(f"Attempting to initialize ASRClient with original quant='{settings.ASR_QUANTIZATION}'...")
                asr_client_quant = ASRClient(quant=settings.ASR_QUANTIZATION)
                logger.info("ASRClient (with quant) initialized.")
                logger.info(f"Attempting transcription of '{audio_path}' with quant='{settings.ASR_QUANTIZATION}'...")
                text_quant = await asr_client_quant.transcribe(audio_path, language = "en")
                logger.info(f"Transcription (quant='{settings.ASR_QUANTIZATION}'): '{text_quant}'")


        except Exception as e:
            logger.error(f"An error occurred during ASR test: {e}")
            print(f"ERROR IN ASR MAIN TEST: {e}\n{traceback.format_exc()}")

    else:
        logger.warning("Skipping ASR test as lightning_whisper_mlx is not available.")

if __name__ == "__main__":
    import asyncio
    # Make sure your config.py and utils/logger_config.py are accessible
    # when running with `python -m src.external_services.asr_client`
    asyncio.run(main_test_asr())