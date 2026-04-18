import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.config import settings
from app.schemas import DocumentChunk, SourceHit, ChunkMetadata, EvidenceObject


class QdrantStore:
    def __init__(self, vector_size: int) -> None:
        self.client = QdrantClient(path=str(settings.qdrant_path))
        self.collection_name = settings.collection_name
        self.evidence_collection_name = f"{settings.collection_name}__evidence"
        self.vector_size = vector_size
        self._ensure_collection(self.collection_name)
        self._ensure_collection(self.evidence_collection_name)

    def _ensure_collection(self, collection_name: str) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def _payload_from_chunk(self, chunk: DocumentChunk) -> dict:
        return {
            "record_type": "chunk",
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

    def _payload_from_evidence(self, evidence: EvidenceObject) -> dict:
        return {
            "record_type": "evidence",
            "evidence_id": evidence.evidence_id,
            "source_path": evidence.source_path,
            "file_name": evidence.file_name,
            "document_title": evidence.document_title,
            "section_title": evidence.section_title,
            "evidence_type": evidence.evidence_type,
            "evidence_text": evidence.evidence_text,
            "supporting_before": evidence.supporting_before,
            "supporting_after": evidence.supporting_after,
            "referring_passages": evidence.referring_passages,
            "source_fingerprint": evidence.source_fingerprint,
            "chunk_ids": evidence.chunk_ids,
        }

    def _point_id_from_chunk_id(self, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def _point_id_from_evidence_id(self, evidence_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, evidence_id))

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

    def upsert_evidence_objects(self, evidence_objects: list[EvidenceObject], vectors: list[list[float]]) -> None:
        points = []
        for evidence, vector in zip(evidence_objects, vectors, strict=True):
            payload = self._payload_from_evidence(evidence)
            point_id = self._point_id_from_evidence_id(evidence.evidence_id)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )
        if points:
            self.client.upsert(collection_name=self.evidence_collection_name, points=points)

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

    def _source_hit_from_evidence_point(self, r) -> SourceHit:
        payload = r.payload or {}
        parts = [payload.get("evidence_text", "")]
        if payload.get("supporting_before"):
            parts.append("Stödtext före:\n" + payload["supporting_before"])
        if payload.get("supporting_after"):
            parts.append("Stödtext efter:\n" + payload["supporting_after"])
        for passage in payload.get("referring_passages") or []:
            parts.append("Referens i text:\n" + passage)

        evidence_type = payload.get("evidence_type") or "evidence"
        section_title = payload.get("section_title")
        if section_title:
            section_title = f"{section_title} [{evidence_type}]"
        else:
            section_title = f"[{evidence_type}]"

        return SourceHit(
            chunk_id=str(r.id),
            score=float(getattr(r, "score", 0.0)),
            text="\n\n".join(p for p in parts if p),
            metadata=ChunkMetadata(
                source_path=payload.get("source_path", ""),
                file_name=payload.get("file_name", ""),
                document_title=payload.get("document_title"),
                category=None,
                section_title=section_title,
                section_level=None,
                page_number=None,
                document_date=None,
                document_type=evidence_type,
                keywords=[],
                roles=[],
                actions=[],
                time_markers=[],
                applies_to=[],
                section_summary=None,
                source_fingerprint=payload.get("source_fingerprint"),
                semantic_enriched=False,
                semantic_model=None,
                semantic_version=None,
                semantic_source_hash=None,
                chunk_index=-1,
            ),
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 6,
        source_paths: list[str] | None = None,
    ) -> list[SourceHit]:
        query_filter = None
        if source_paths:
            conditions = [
                FieldCondition(key="source_path", match=MatchValue(value=path))
                for path in source_paths
            ]
            query_filter = Filter(should=conditions)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
        points = response.points if hasattr(response, "points") else response
        return [self._source_hit_from_point(r) for r in points]

    def search_evidence(self, query_vector: list[float], source_paths: list[str], limit: int = 12) -> list[SourceHit]:
        if not source_paths:
            return []
        conditions = [
            FieldCondition(key="source_path", match=MatchValue(value=path))
            for path in source_paths
        ]
        response = self.client.query_points(
            collection_name=self.evidence_collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=Filter(should=conditions),
        )
        points = response.points if hasattr(response, "points") else response
        return [self._source_hit_from_evidence_point(r) for r in points]

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

    def iter_all_evidence(self, batch_size: int = 256) -> list[SourceHit]:
        offset = None
        all_hits: list[SourceHit] = []

        while True:
            points, offset = self.client.scroll(
                collection_name=self.evidence_collection_name,
                scroll_filter=None,
                limit=batch_size,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )

            if not points:
                break

            all_hits.extend(self._source_hit_from_evidence_point(p) for p in points)

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

    def get_evidence_source_paths(self, batch_size: int = 256) -> set[str]:
        """
        Hämta de source_paths som har minst ett evidensobjekt indexerat.

        Används av ingest-loopen för att upptäcka dokument som har
        chunks men saknar evidensobjekt — t.ex. dokument som
        indexerades innan evidensstrukturen införts, eller där
        evidens-extraktionen misslyckats tidigare.
        """
        hits = self.iter_all_evidence(batch_size=batch_size)
        return {hit.metadata.source_path for hit in hits if hit.metadata.source_path}

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

    def delete_evidence_by_source_path(self, source_path: str) -> None:
        self.client.delete(
            collection_name=self.evidence_collection_name,
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
        """
        Uppdatera metadata för flera punkter. Inte en Qdrant-batch — varje
        punkt uppdateras i ett eget anrop eftersom varje punkt typiskt har
        unika payload-uppdateringar. Namnet "bulk" syftar på att flera
        punkter hanteras i en anropssekvens, inte på Qdrant-batching.
        """
        for point_id, metadata_updates in updates:
            self.update_chunk_metadata_by_point_id(point_id, metadata_updates)

    def recreate_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        for collection_name in (self.collection_name, self.evidence_collection_name):
            if collection_name in collections:
                self.client.delete_collection(collection_name=collection_name)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )