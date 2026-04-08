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
    top_k: int = int(os.getenv("TOP_K", "3"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

settings = Settings()