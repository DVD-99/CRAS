import os
from melo.api import TTS as MeloTTS
from src.external_services.tts_client import TTSClient

# Example Usage (for testing this file directly):
async def main_test_tts():
    if MeloTTS:
        try:
            # Ensure data/audio_samples directory exists
            output_dir = "./data/tts_output"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "test_speech.mp3")

            # Use settings from config or override here
            tts_client = TTSClient(language="EN") # Or settings.TTS_MELOTTS_VOICE
            text_to_say = "Hello, this is a test of the Melo Text to Speech system using MLX."
            saved_path = await tts_client.synthesize_speech(text_to_say, output_file)

            if "Error:" not in saved_path:
                print(f"Speech saved to: {saved_path}")
                # You can play it using a system player or another library if desired
            else:
                print(saved_path)
        except Exception as e:
            print(f"Could not run TTS test: {e}")
    else:
        print("Skipping TTS test as MeloTTS is not available.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main_test_tts())