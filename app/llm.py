import ollama
from app.config import settings


class LLMUnavailableError(RuntimeError):
    pass


class LocalLLM:
    def __init__(self) -> None:
        self.model = settings.ollama_model

    def generate(self, prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": 0.1,
                },
            )
            return response["message"]["content"].strip()
        except Exception as e:
            raise LLMUnavailableError(
                f"Kunde inte nå Ollama/modellen '{self.model}': {type(e).__name__}: {e}"
            ) from e