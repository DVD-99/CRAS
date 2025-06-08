
from mlx_lm import load
from src.external_services.llm_client import LLMClient

# Example Usage (for testing this file directly):
async def main_test_llm():
    if load:
        try:
            # Ensure your settings.LLM_MODEL_PATH points to a model compatible with mlx-lm
            # smaller model like "mlx-community/Phi-3.5-mini-instruct-4bit"
            llm_client = LLMClient(model_path="mlx-community/Phi-3.5-mini-instruct-4bit") # Example with a smaller model
            prompt_text = "Explain the concept of an LLM in simple terms."

            # If using a base model
            response = await llm_client.generate_text(prompt_text, max_tokens=512)
            print(f"\nLLM Response:\n{response}")
        except Exception as e:
            print(f"Could not run LLM test: {e}")
    else:
        print("Skipping LLM test as mlx_lm is not available.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main_test_llm())