"""
Dämpning av tredjepartsbibliotekens utskrifter.

Transformers, sentence-transformers och huggingface_hub skriver
diagnostik till stdout och stderr vid modelladdning:

    Warning: You are sending unauthenticated requests to the HF Hub...
    XLMRobertaModel LOAD REPORT from: intfloat/multilingual-e5-large
    Key                     | Status     |
    embeddings.position_ids | UNEXPECTED |
    Loading weights: 100%|███...███| 391/391 [00:00<00:00, 14521.00it/s]

Ingenting av det är åtgärdbart för användaren. LOAD REPORT gäller ett
fält som saknas i checkpointen och som modellen inte använder;
HF-varningen gäller nedladdningskvot som inte är relevant när
modellerna redan finns lokalt.

I serverläge är det brus i loggen. I interaktivt läge står det mellan
frågan och svaret, vilket gör gränssnittet oanvändbart.

MÅSTE ANROPAS FÖRE IMPORT av transformers och sentence_transformers,
eftersom flera av utskrifterna sker vid modulinladdning. Därav
miljövariablerna, som biblioteken läser när de initieras.

Dämpningen kan stängas av med URD_VERBOSE_LIBS=1 för felsökning av
modelladdning.
"""

from __future__ import annotations

import logging
import os

_NOISY_LOGGERS = (
    "transformers",
    "transformers.modeling_utils",
    "transformers.configuration_utils",
    "transformers.tokenization_utils_base",
    "sentence_transformers",
    "sentence_transformers.SentenceTransformer",
    "sentence_transformers.cross_encoder.CrossEncoder",
    "huggingface_hub",
    "huggingface_hub.file_download",
    "urllib3",
    "filelock",
    "stanza",
)

_applied = False


def quiet_libraries() -> None:
    """Dämpa biblioteksutskrifter. Idempotent."""
    global _applied
    if _applied or os.environ.get("URD_VERBOSE_LIBS"):
        return
    _applied = True

    # Läses av biblioteken vid init — måste sättas före import.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Tystar "Loading weights: 100%|...".
    os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        # Biblioteket kanske inte är installerat, eller har bytt API.
        # Dämpningen är en bekvämlighet och får aldrig fälla starten.
        pass
