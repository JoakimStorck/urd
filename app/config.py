from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseModel):
    docs_path: Path = Path(os.getenv("DOCS_PATH", "./docs"))
    qdrant_path: Path = Path(os.getenv("QDRANT_PATH", "./data/qdrant"))
    collection_name: str = os.getenv("QDRANT_COLLECTION", "iit_docs")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

    ollama_model: str = os.getenv("OLLAMA_MODEL", "mistral")
    preprocess_ollama_model: str = os.getenv(
        "PREPROCESS_OLLAMA_MODEL",
        os.getenv("OLLAMA_MODEL", "mistral"),
    )
    preprocess_semantic_version: str = os.getenv("PREPROCESS_SEMANTIC_VERSION", "v1")

    top_k: int = int(os.getenv("TOP_K", "3"))
    retrieval_candidate_k: int = int(os.getenv("RETRIEVAL_CANDIDATE_K", "12"))

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    preprocess_max_section_chars: int = int(os.getenv("PREPROCESS_MAX_SECTION_CHARS", "6000"))

    rerank_semantic_bonus: float = float(os.getenv("RERANK_SEMANTIC_BONUS", "0.01"))
    rerank_keyword_bonus: float = float(os.getenv("RERANK_KEYWORD_BONUS", "0.06"))
    rerank_role_bonus: float = float(os.getenv("RERANK_ROLE_BONUS", "0.04"))
    rerank_action_bonus: float = float(os.getenv("RERANK_ACTION_BONUS", "0.04"))
    rerank_summary_bonus: float = float(os.getenv("RERANK_SUMMARY_BONUS", "0.03"))
    rerank_document_type_bonus: float = float(os.getenv("RERANK_DOCUMENT_TYPE_BONUS", "0.05"))
    rerank_section_title_bonus: float = float(os.getenv("RERANK_SECTION_TITLE_BONUS", "0.07"))
    rerank_document_title_bonus: float = float(os.getenv("RERANK_DOCUMENT_TITLE_BONUS", "0.05"))
    rerank_boilerplate_penalty: float = float(os.getenv("RERANK_BOILERPLATE_PENALTY", "0.10"))


settings = Settings()