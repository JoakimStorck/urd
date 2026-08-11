"""
Embeddings med instruktionsprefix för E5-modeller.

multilingual-e5-modellerna är TRÄNADE med instruktionsprefix:
frågor ska kodas som "query: ..." och dokumenttext som "passage: ...".
Modellkortet är uttryckligt om att prestandan degraderar utan dem —
och URD körde länge utan, vilket både försvagade den semantiska
sökningen och komprimerade likhetsskalan så att QUD-driftskyddet
blev verkningslöst (uppmätt i baslinjen 2026-08-11: alla likheter
i bandet 0,76–0,90 oavsett ämnesrelation).

Prefixen appliceras automatiskt när modellnamnet ser ut att vara en
E5-modell, annars inte — så att ett modellbyte i config inte tyst
får fel prefixregim. Prefixet läggs på i detta lager, aldrig av
anroparna: embed_query för frågor, embed_texts för dokumenttext.

VIKTIGT: att slå på prefixen ändrar alla vektorer. Indexet måste
byggas om (urd reindex) och qud_drift_threshold omkalibreras
(scripts/calibrate_drift.py) efter denna ändring.
"""

from sentence_transformers import SentenceTransformer
from app.config import settings


def _is_e5_model(model_name: str) -> bool:
    """Avgör om modellen är en E5-modell som kräver instruktionsprefix."""
    return "e5" in model_name.casefold()


class Embedder:
    def __init__(self) -> None:
        self.model = SentenceTransformer(settings.embedding_model)
        self.use_e5_prefixes = _is_e5_model(settings.embedding_model)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Koda dokumenttext (chunkar, evidensobjekt) — passage-sidan."""
        if self.use_e5_prefixes:
            texts = [f"passage: {t}" for t in texts]
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        """Koda en fråga — query-sidan."""
        if self.use_e5_prefixes:
            text = f"query: {text}"
        return self._encode([text])[0]
