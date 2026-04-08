from sentence_transformers import SentenceTransformer
from app.config import settings

class Embedder:
    def __init__(self) -> None:
        self.model = SentenceTransformer(settings.embedding_model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]