import uuid
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
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

    def _payload_from_chunk(self, chunk: DocumentChunk) -> dict:
        return {
            "text": chunk.text,
            "source_path": chunk.metadata.source_path,
            "file_name": chunk.metadata.file_name,
            "document_title": chunk.metadata.document_title,
            "category": chunk.metadata.category,
            "section_title": chunk.metadata.section_title,
            "section_level": chunk.metadata.section_level,
            "page_number": chunk.metadata.page_number,
            "document_date": chunk.metadata.document_date,
            "document_type": chunk.metadata.document_type,
            "keywords": chunk.metadata.keywords,
            "roles": chunk.metadata.roles,
            "actions": chunk.metadata.actions,
            "time_markers": chunk.metadata.time_markers,
            "applies_to": chunk.metadata.applies_to,
            "section_summary": chunk.metadata.section_summary,
            "source_fingerprint": chunk.metadata.source_fingerprint,
            "semantic_enriched": chunk.metadata.semantic_enriched,
            "semantic_model": chunk.metadata.semantic_model,
            "semantic_version": chunk.metadata.semantic_version,
            "semantic_source_hash": chunk.metadata.semantic_source_hash,
            "chunk_index": chunk.metadata.chunk_index,
        }

    def _point_id_from_chunk_id(self, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def upsert_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            payload = self._payload_from_chunk(chunk)
            point_id = self._point_id_from_chunk_id(chunk.chunk_id)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def _source_hit_from_point(self, r) -> SourceHit:
        payload = r.payload or {}
        return SourceHit(
            chunk_id=str(r.id),  # point-id
            score=float(getattr(r, "score", 0.0)),
            text=payload.get("text", ""),
            metadata=ChunkMetadata(
                source_path=payload.get("source_path", ""),
                file_name=payload.get("file_name", ""),
                document_title=payload.get("document_title"),
                category=payload.get("category"),
                section_title=payload.get("section_title"),
                section_level=payload.get("section_level"),
                page_number=payload.get("page_number"),
                document_date=payload.get("document_date"),
                document_type=payload.get("document_type"),
                keywords=payload.get("keywords") or [],
                roles=payload.get("roles") or [],
                actions=payload.get("actions") or [],
                time_markers=payload.get("time_markers") or [],
                applies_to=payload.get("applies_to") or [],
                section_summary=payload.get("section_summary"),
                source_fingerprint=payload.get("source_fingerprint"),
                semantic_enriched=payload.get("semantic_enriched", False),
                semantic_model=payload.get("semantic_model"),
                semantic_version=payload.get("semantic_version"),
                semantic_source_hash=payload.get("semantic_source_hash"),
                chunk_index=payload.get("chunk_index", 0),
            ),
        )

    def search(self, query_vector: list[float], limit: int = 6) -> list[SourceHit]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        points = response.points if hasattr(response, "points") else response
        return [self._source_hit_from_point(r) for r in points]

    def iter_all_chunks(self, batch_size: int = 256) -> list[SourceHit]:
        offset = None
        all_hits: list[SourceHit] = []

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=batch_size,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )

            if not points:
                break

            all_hits.extend(self._source_hit_from_point(p) for p in points)

            if offset is None:
                break

        return all_hits

    def get_indexed_documents(self, batch_size: int = 256) -> dict[str, str | None]:
        hits = self.iter_all_chunks(batch_size=batch_size)
        by_path: dict[str, str | None] = {}
        for hit in hits:
            if hit.metadata.source_path not in by_path:
                by_path[hit.metadata.source_path] = hit.metadata.source_fingerprint
        return by_path

    def delete_chunks_by_source_path(self, source_path: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source_path",
                        match=MatchValue(value=source_path),
                    )
                ]
            ),
        )

    def update_chunk_metadata_by_point_id(self, point_id: str, metadata_updates: dict) -> None:
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=metadata_updates,
            points=[point_id],
        )

    def bulk_update_chunk_metadata(self, updates: list[tuple[str, dict]]) -> None:
        for point_id, metadata_updates in updates:
            self.update_chunk_metadata_by_point_id(point_id, metadata_updates)

    def recreate_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            self.client.delete_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )