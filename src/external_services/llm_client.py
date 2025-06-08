# cras_project/cras_core/external_services/llm_client.py
from typing import Optional, Dict, Any, List
import time
from ..config import settings
from ..utils.logger_config import setup_logger
from huggingface_hub import login
try:
    from mlx_lm import load, generate
except ImportError:
    print("Warning: mlx_lm not found. LLMClient will not function.")
    load = None
    generate = None

logger = setup_logger(__name__, level=settings.LOG_LEVEL.upper() if hasattr(settings, 'LOG_LEVEL') else 'INFO')

class LLMClient:
    """
    Client for interacting with Language Models using MLX LM.
    """
    def __init__(self, model_path: Optional[str] = None):
        if not load:
            raise ImportError("mlx_lm library is required but not installed.")
        
        hf_token = getattr(settings, 'HUGGING_FACE_TOKEN', None)
        if hf_token:
            logger.info("Hugging Face token found. Logging in...")
            login(token=hf_token)
        else:
            logger.warning("Hugging Face token not found in settings. Downloads may fail for gated models.")

        self.model_path = model_path or settings.LLM_MODEL_PATH
        self.model = None
        self.tokenizer = None
        logger.info(f"Initializing LLMClient with model: {self.model_path}")
        try:
            self.model, self.tokenizer = load(self.model_path)
            logger.info(f"LLM model '{self.model_path}' loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading LLM model '{self.model_path}': {e}", exc_info=True)
            raise

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512
    ) -> str:
        """
        Generates text based on the given prompt using a chat template.

        """
        if not self.model or not self.tokenizer:
            logger.error("LLM model or tokenizer not loaded.")
            return "Error: LLM model or tokenizer not loaded."

        # Prepare messages for the chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply the chat template to format the prompt correctly for the model
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        logger.info(f"Generating text for prompt (first 50 chars): '{prompt[:50]}...'")
        start_time = time.time()
        
        try:
            response = generate(
                self.model, 
                self.tokenizer, 
                prompt=formatted_prompt, 
                max_tokens=max_tokens,
                verbose=False
            )
            
            duration = time.time() - start_time
            logger.info(f"LLM text generated in {duration:.2f} seconds.")
            return response
        except Exception as e:
            logger.error(f"Error during LLM text generation: {e}", exc_info=True)
            return f"Error: Could not generate text. Details logged. Error: {type(e).__name__}"