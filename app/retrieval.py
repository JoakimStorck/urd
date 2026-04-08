import time
from app.config import settings
from app.embeddings import Embedder
from app.qdrant_store import QdrantStore
from app.llm import LocalLLM
from app.prompting import build_prompt
from app.schemas import ChatResponse


class RagService:
    def __init__(self) -> None:
        self.embedder = Embedder()
        test_vec = self.embedder.embed_query("test")
        self.store = QdrantStore(vector_size=len(test_vec))
        self.llm = LocalLLM()

    def answer(self, question: str) -> ChatResponse:
        t0 = time.perf_counter()
        query_vector = self.embedder.embed_query(question)
        t1 = time.perf_counter()

        hits = self.store.search(query_vector, limit=settings.top_k)
        t2 = time.perf_counter()

        prompt = build_prompt(question, hits)
        t3 = time.perf_counter()

        answer = self.llm.generate(prompt)
        t4 = time.perf_counter()

        return ChatResponse(
            answer=answer,
            sources=hits,
            debug={
                "top_k": settings.top_k,
                "num_hits": len(hits),
                "timing_s": {
                    "embed_query": round(t1 - t0, 3),
                    "search": round(t2 - t1, 3),
                    "build_prompt": round(t3 - t2, 3),
                    "generate": round(t4 - t3, 3),
                    "total": round(t4 - t0, 3),
                },
                "prompt_chars": len(prompt),
            },
        )