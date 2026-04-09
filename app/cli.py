#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path

import typer
import uvicorn

from app.config import settings
from app.embeddings import Embedder
from app.ingest import iter_document_paths, ingest_path, compute_source_fingerprint
from app.preprocess_llm import SectionMetadataExtractor
from app.qdrant_store import QdrantStore
from app.retrieval import RagService
from app.schemas import SourceHit

app = typer.Typer(
    help="Lokal dokumentchat för IIT-dokument.",
    no_args_is_help=True,
    add_completion=False,
)


def _build_store_and_embedder() -> tuple[QdrantStore, Embedder]:
    embedder = Embedder()
    dim = len(embedder.embed_query("test"))
    store = QdrantStore(vector_size=dim)
    return store, embedder


def _build_store_only() -> QdrantStore:
    # Samlingen finns redan efter ingest/reset-index.
    # Dummy-dimension används bara för init; _ensure_collection skapar inget nytt om samlingen finns.
    return QdrantStore(vector_size=1024)


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", help="Host för webbservern."),
    port: int = typer.Option(8000, help="Port för webbservern."),
    reload: bool = typer.Option(True, help="Kör med autoreload."),
) -> None:
    uvicorn.run("app.api:app", host=host, port=port, reload=reload)


@app.command("reset-index")
def reset_index() -> None:
    store, _ = _build_store_and_embedder()
    store.recreate_collection()
    typer.echo(f"Återskapade samlingen '{settings.collection_name}' i {settings.qdrant_path}")


@app.command("ingest")
def ingest(
    docs_path: Path | None = typer.Option(None, "--docs-path"),
    force: bool = typer.Option(False, "--force", help="Kör om alla dokument oavsett fingerprint."),
    sync_delete: bool = typer.Option(False, "--sync-delete", help="Ta bort indexerade dokument som inte längre finns på disk."),
) -> None:
    root = docs_path or settings.docs_path
    if not root.exists():
        raise typer.BadParameter(f"Dokumentkatalog finns inte: {root}")

    store, embedder = _build_store_and_embedder()
    indexed_docs = store.get_indexed_documents()
    fs_paths = iter_document_paths(root)
    fs_map = {str(p): compute_source_fingerprint(p) for p in fs_paths}

    typer.echo(f"Läser dokument från: {root}")
    typer.echo(f"Antal filer funna: {len(fs_paths)}")
    typer.echo(f"Indexerade dokument i Qdrant: {len(indexed_docs)}")

    total_docs = 0
    total_chunks = 0
    skipped = 0
    updated = 0
    created = 0

    if sync_delete:
        fs_set = set(fs_map.keys())
        indexed_set = set(indexed_docs.keys())
        removed = sorted(indexed_set - fs_set)
        for source_path in removed:
            store.delete_chunks_by_source_path(source_path)
            typer.echo(f"Deleted missing document from index: {source_path}")

    for path in fs_paths:
        source_path = str(path)
        new_fp = fs_map[source_path]
        old_fp = indexed_docs.get(source_path)

        if not force and old_fp == new_fp:
            skipped += 1
            typer.echo(f"Skip unchanged: {source_path}")
            continue

        if old_fp is not None:
            store.delete_chunks_by_source_path(source_path)
            updated += 1
            typer.echo(f"Reingest changed: {source_path}")
        else:
            created += 1
            typer.echo(f"Ingest new: {source_path}")

        chunks = ingest_path(path, root)
        if not chunks:
            typer.echo(f"Hoppar över {path} (inga chunkar)")
            continue

        vectors = embedder.embed_texts([c.text for c in chunks])
        store.upsert_chunks(chunks, vectors)

        total_docs += 1
        total_chunks += len(chunks)
        typer.echo(f"Indexed {path} -> {len(chunks)} chunks")

    typer.echo("")
    typer.echo(
        f"Klart. Processade dokument: {total_docs}, chunkar: {total_chunks}, "
        f"skapade: {created}, uppdaterade: {updated}, hoppade över: {skipped}"
    )


@app.command("reindex")
def reindex(
    docs_path: Path | None = typer.Option(None, "--docs-path"),
) -> None:
    reset_index()
    ingest(docs_path=docs_path, force=False, sync_delete=False)


def _section_key(hit: SourceHit) -> tuple[str, str | None, int | None]:
    m = hit.metadata
    return (m.source_path, m.section_title, m.section_level)


def _section_text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@app.command("enrich")
def enrich(
    batch_size: int = typer.Option(256),
    limit_sections: int | None = typer.Option(None, "--limit-sections"),
    force: bool = typer.Option(False, "--force", help="Annotera alla sektioner igen."),
) -> None:
    store = _build_store_only()
    extractor = SectionMetadataExtractor()

    typer.echo("Läser indexerade chunkar från Qdrant ...")
    hits = store.iter_all_chunks(batch_size=batch_size)
    typer.echo(f"Antal chunkar lästa: {len(hits)}")

    grouped: dict[tuple[str, str | None, int | None], list[SourceHit]] = defaultdict(list)
    for hit in hits:
        grouped[_section_key(hit)].append(hit)

    section_items = list(grouped.items())
    section_items.sort(key=lambda item: (
        item[0][0],
        item[1][0].metadata.chunk_index if item[1] else 0,
    ))

    if limit_sections is not None:
        section_items = section_items[:limit_sections]

    updates: list[tuple[str, dict]] = []
    skipped = 0
    processed = 0

    typer.echo(f"Antal sektioner att pröva: {len(section_items)}")

    for idx, ((source_path, section_title, section_level), group_hits) in enumerate(section_items, start=1):
        group_hits.sort(key=lambda h: h.metadata.chunk_index)
        section_text = "\n\n".join(h.text for h in group_hits if h.text.strip())
        if not section_text.strip():
            skipped += 1
            continue

        source_hash = _section_text_hash(section_text)
        first_meta = group_hits[0].metadata

        already_ok = (
            first_meta.semantic_enriched
            and first_meta.semantic_source_hash == source_hash
            and first_meta.semantic_model == settings.preprocess_ollama_model
            and first_meta.semantic_version == settings.preprocess_semantic_version
        )

        if not force and already_ok:
            skipped += 1
            continue

        semantic = extractor.extract(
            document_title=first_meta.document_title,
            section_title=section_title,
            text=section_text,
        )

        payload_updates = {
            "document_type": semantic.document_type,
            "keywords": semantic.keywords,
            "roles": semantic.roles,
            "actions": semantic.actions,
            "time_markers": semantic.time_markers,
            "applies_to": semantic.applies_to,
            "section_summary": semantic.summary,
            "semantic_enriched": True,
            "semantic_model": settings.preprocess_ollama_model,
            "semantic_version": settings.preprocess_semantic_version,
            "semantic_source_hash": source_hash,
        }

        for hit in group_hits:
            updates.append((hit.chunk_id, payload_updates))

        processed += 1
        typer.echo(
            f"[{idx}/{len(section_items)}] "
            f"{Path(source_path).name} | rubrik={section_title!r} | chunkar={len(group_hits)}"
        )

    typer.echo("Skriver tillbaka metadata till Qdrant ...")
    store.bulk_update_chunk_metadata(updates)

    typer.echo("")
    typer.echo(
        f"Klart. Annoterade sektioner: {processed}, hoppade över: {skipped}, "
        f"uppdaterade chunkar: {len(updates)}"
    )


@app.command("backfill-enrich-status")
def backfill_enrich_status(
    batch_size: int = typer.Option(256),
) -> None:
    store = _build_store_only()
    hits = store.iter_all_chunks(batch_size=batch_size)

    updates: list[tuple[str, dict]] = []

    for hit in hits:
        m = hit.metadata
        has_semantic = bool(
            m.document_type
            or m.keywords
            or m.roles
            or m.actions
            or m.time_markers
            or m.applies_to
            or m.section_summary
        )

        updates.append((
            hit.chunk_id,
            {
                "semantic_enriched": has_semantic,
            },
        ))

    typer.echo("Skriver enrich-status till Qdrant ...")
    store.bulk_update_chunk_metadata(updates)
    typer.echo(f"Klart. Uppdaterade chunkar: {len(updates)}")

@app.command("stats")
def stats(
    docs_path: Path | None = typer.Option(None, "--docs-path"),
    batch_size: int = typer.Option(256),
) -> None:
    """
    Visa översikt över dokument på disk, indexerade chunkar/sektioner
    och enrich-status.
    """
    root = docs_path or settings.docs_path
    if not root.exists():
        raise typer.BadParameter(f"Dokumentkatalog finns inte: {root}")

    store = _build_store_only()

    fs_paths = iter_document_paths(root)
    fs_map = {str(p): compute_source_fingerprint(p) for p in fs_paths}

    hits = store.iter_all_chunks(batch_size=batch_size)
    indexed_docs = store.get_indexed_documents(batch_size=batch_size)

    section_groups: dict[tuple[str, str | None, int | None], list[SourceHit]] = defaultdict(list)
    for hit in hits:
        section_groups[_section_key(hit)].append(hit)

    enriched_chunks = sum(1 for h in hits if h.metadata.semantic_enriched)
    not_enriched_chunks = len(hits) - enriched_chunks

    enriched_sections = 0
    not_enriched_sections = 0

    for _, group_hits in section_groups.items():
        if group_hits and all(h.metadata.semantic_enriched for h in group_hits):
            enriched_sections += 1
        else:
            not_enriched_sections += 1

    indexed_set = set(indexed_docs.keys())
    fs_set = set(fs_map.keys())

    new_docs = sorted(fs_set - indexed_set)
    missing_docs = sorted(indexed_set - fs_set)

    changed_docs = []
    unchanged_docs = []

    for source_path in sorted(fs_set & indexed_set):
        fs_fp = fs_map[source_path]
        idx_fp = indexed_docs.get(source_path)
        if fs_fp == idx_fp:
            unchanged_docs.append(source_path)
        else:
            changed_docs.append(source_path)

    typer.echo("")
    typer.echo("Disk")
    typer.echo("----")
    typer.echo(f"Dokument på disk:        {len(fs_paths)}")

    typer.echo("")
    typer.echo("Index")
    typer.echo("-----")
    typer.echo(f"Indexerade dokument:     {len(indexed_docs)}")
    typer.echo(f"Indexerade sektioner:    {len(section_groups)}")
    typer.echo(f"Indexerade chunkar:      {len(hits)}")

    typer.echo("")
    typer.echo("Enrich")
    typer.echo("------")
    typer.echo(f"Enrichade sektioner:     {enriched_sections}")
    typer.echo(f"Ej enrichade sektioner:  {not_enriched_sections}")
    typer.echo(f"Enrichade chunkar:       {enriched_chunks}")
    typer.echo(f"Ej enrichade chunkar:    {not_enriched_chunks}")

    typer.echo("")
    typer.echo("Synk mot disk")
    typer.echo("------------")
    typer.echo(f"Nya dokument:            {len(new_docs)}")
    typer.echo(f"Ändrade dokument:        {len(changed_docs)}")
    typer.echo(f"Oförändrade dokument:    {len(unchanged_docs)}")
    typer.echo(f"Saknas på disk:          {len(missing_docs)}")

    if new_docs:
        typer.echo("")
        typer.echo("Nya dokument")
        typer.echo("------------")
        for p in new_docs[:20]:
            typer.echo(p)
        if len(new_docs) > 20:
            typer.echo(f"... och {len(new_docs) - 20} till")

    if changed_docs:
        typer.echo("")
        typer.echo("Ändrade dokument")
        typer.echo("----------------")
        for p in changed_docs[:20]:
            typer.echo(p)
        if len(changed_docs) > 20:
            typer.echo(f"... och {len(changed_docs) - 20} till")

    if missing_docs:
        typer.echo("")
        typer.echo("Indexerade men saknas på disk")
        typer.echo("-----------------------------")
        for p in missing_docs[:20]:
            typer.echo(p)
        if len(missing_docs) > 20:
            typer.echo(f"... och {len(missing_docs) - 20} till")

@app.command("ask")
def ask(
    question: str = typer.Argument(...),
    show_sources: bool = typer.Option(True, "--sources/--no-sources"),
    show_debug: bool = typer.Option(False, "--debug"),
) -> None:
    rag = RagService()
    response = rag.answer(question)

    typer.echo("")
    typer.echo("Svar")
    typer.echo("----")
    typer.echo(response.answer)

    if show_sources and response.sources:
        typer.echo("")
        typer.echo("Källor")
        typer.echo("------")
        for i, src in enumerate(response.sources, start=1):
            meta = src.metadata
            typer.echo(f"[{i}] {meta.file_name}")
            if meta.document_title:
                typer.echo(f"    titel: {meta.document_title}")
            if meta.category:
                typer.echo(f"    kategori: {meta.category}")
            if meta.section_title:
                typer.echo(f"    rubrik: {meta.section_title}")
            if meta.section_level is not None:
                typer.echo(f"    nivå: {meta.section_level}")
            if meta.document_type:
                typer.echo(f"    dokumenttyp: {meta.document_type}")
            if meta.document_date:
                typer.echo(f"    datum: {meta.document_date}")
            typer.echo(f"    score: {src.score:.3f}")
            typer.echo(f"    chunk: {meta.chunk_index}")
            typer.echo(f"    väg: {meta.source_path}")

    if show_debug and response.debug:
        typer.echo("")
        typer.echo("Debug")
        typer.echo("-----")
        typer.echo(json.dumps(response.debug, ensure_ascii=False, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()