from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import settings
from app.schemas import DocumentChunk, SourceHit, ChunkMetadata


class QdrantStore:
    def __init__(self, vector_size: int) -> None:
        self.client = QdrantClient(path=str(settings.qdrant_path))
        self.collection_name = settings.collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def upsert_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            payload = {
                "text": chunk.text,
                "source_path": chunk.metadata.source_path,
                "file_name": chunk.metadata.file_name,
                "document_title": chunk.metadata.document_title,
                "category": chunk.metadata.category,
                "section_title": chunk.metadata.section_title,
                "page_number": chunk.metadata.page_number,
                "document_date": chunk.metadata.document_date,
                "chunk_index": chunk.metadata.chunk_index,
            }
            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], limit: int = 6) -> list[SourceHit]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        points = response.points if hasattr(response, "points") else response

        hits: list[SourceHit] = []
        for r in points:
            payload = r.payload or {}
            hits.append(
                SourceHit(
                    chunk_id=str(r.id),
                    score=float(r.score),
                    text=payload.get("text", ""),
                    metadata=ChunkMetadata(
                        source_path=payload.get("source_path", ""),
                        file_name=payload.get("file_name", ""),
                        document_title=payload.get("document_title"),
                        category=payload.get("category"),
                        section_title=payload.get("section_title"),
                        page_number=payload.get("page_number"),
                        document_date=payload.get("document_date"),
                        chunk_index=payload.get("chunk_index", 0),
                    ),
                )
            )
        return hits

    def recreate_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            self.client.delete_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )        