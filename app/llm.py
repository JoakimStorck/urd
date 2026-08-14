import logging

import ollama
from app.config import settings

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    pass


# Grov tokenuppskattning för svensk text: ~3 tecken per token är en
# konservativ tumregel för mistral-familjens tokenizer på svenska.
# Uppskattningen används bara för trunkeringsvarningen — den behöver
# vara ungefärlig, inte exakt.
_CHARS_PER_TOKEN_ESTIMATE = 3


class LocalLLM:
    def __init__(self) -> None:
        self.model = settings.ollama_model
        self.num_ctx = settings.llm_num_ctx
        self.think = settings.llm_think
        # Äldre versioner av ollama-python känner inte till think.
        # Sätts till False första gången ett anrop avvisas på den
        # grunden, så att fallbacken bara kostar ett försök.
        self._think_supported = True

    def generate(self, prompt: str, format: str | None = None) -> str:
        # Logga promptstorlek och varna om prompten riskerar att
        # trunkeras tyst av Ollama. En trunkerad prompt ger svar som
        # ser ut som modellfel (fabrikation, ignorerade instruktioner)
        # — varningen gör orsaken diagnosbar i loggen.
        est_tokens = len(prompt) // _CHARS_PER_TOKEN_ESTIMATE
        logger.debug(
            "LLM-anrop (%s): %d tecken (~%d tokens, num_ctx=%d)",
            self.model, len(prompt), est_tokens, self.num_ctx,
        )
        if est_tokens > self.num_ctx * 0.9:
            logger.warning(
                "LLM-prompt nära eller över kontextfönstret: ~%d tokens "
                "av num_ctx=%d (%d tecken). Risk för tyst trunkering — "
                "överväg att höja llm_num_ctx eller minska källunderlaget.",
                est_tokens, self.num_ctx, len(prompt),
            )

        try:
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "options": {
                    # temperature 0 för reproducerbarhet. Baslinje-
                    # jämförelsen 2026-08-11 visade att 0.1 räcker för
                    # att klassificering och frågeoperation ska flippa
                    # mellan identiska körningar — då går det inte att
                    # skilja patcheffekt från slump i testbatteriet.
                    # För en evidenscentrerad assistent är deterministisk
                    # generering dessutom rätt även i sak.
                    "temperature": 0.0,
                    "num_ctx": self.num_ctx,
                },
            }
            if format is not None:
                kwargs["format"] = format

            # think sätts ALLTID explicit, av samma skäl som num_ctx:
            # defaultbeteendet varierar mellan modeller och Ollama-
            # versioner, och skillnaden är inte subtil. Se kommentaren
            # vid settings.llm_think.
            if self._think_supported:
                try:
                    response = ollama.chat(think=self.think, **kwargs)
                except TypeError as e:
                    if "think" not in str(e):
                        raise
                    logger.warning(
                        "Installerad ollama-klient stödjer inte think-"
                        "parametern. Resonemangsläget styrs då av "
                        "modellens default och llm_think=%s får ingen "
                        "effekt. Uppgradera ollama-paketet om modellen "
                        "har resonemang påslaget.",
                        self.think,
                    )
                    self._think_supported = False
                    response = ollama.chat(**kwargs)
            else:
                response = ollama.chat(**kwargs)

            return response["message"]["content"].strip()
        except Exception as e:
            raise LLMUnavailableError(
                f"Kunde inte nå Ollama/modellen '{self.model}': {type(e).__name__}: {e}"
            ) from e
