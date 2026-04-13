#!/usr/bin/env python3
from __future__ import annotations

import os

# Försök tvinga Typer till enkel text-help utan Rich-paneler.
# Måste sättas före import av typer.
os.environ["TYPER_USE_RICH"] = "0"

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import requests
import typer
import uvicorn

from app.config import settings
from app.embeddings import Embedder
from app.ingest import compute_source_fingerprint, ingest_path, iter_document_paths
from app.preprocess_llm import SectionMetadataExtractor
from app.qdrant_store import QdrantStore
from app.retrieval import RagService
from app.schemas import ChatResponse, SourceHit

app = typer.Typer(
    help="Lokal dokumentchat för IIT-dokument.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
)


def _ask_via_server(question: str, base_url: str) -> dict:
    url = base_url.rstrip("/") + "/chat"
    resp = requests.post(url, json={"question": question}, timeout=300)
    if not resp.ok:
        try:
            detail = resp.text
        except Exception:
            detail = "<ingen svarstext>"
        raise RuntimeError(
            f"Serverfel {resp.status_code} från {url}\n--- svarstext ---\n{detail}"
        )
    return resp.json()


def _server_is_available(base_url: str) -> bool:
    try:
        url = base_url.rstrip("/") + "/health"
        resp = requests.get(url, timeout=1.0)
        return resp.ok
    except Exception:
        return False


def _build_store_and_embedder() -> tuple[QdrantStore, Embedder]:
    embedder = Embedder()
    dim = len(embedder.embed_query("test"))
    store = QdrantStore(vector_size=dim)
    return store, embedder


def _build_store_only() -> QdrantStore:
    # Samlingen finns redan efter ingest/reset-index.
    # Dummy-dimension används bara för init; _ensure_collection skapar inget nytt
    # om samlingen redan finns.
    return QdrantStore(vector_size=1024)


def _is_qdrant_lock_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "Storage folder" in msg
        and "already accessed by another instance of Qdrant client" in msg
    )


def _print_response(
    response: ChatResponse,
    show_sources: bool,
    show_debug: bool,
) -> None:
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


@app.command(
    "serve",
    help="Starta lokal API-server för dokumentchatten.",
)
def serve(
    host: str = typer.Option("127.0.0.1", help="Host för webbservern."),
    port: int = typer.Option(8000, help="Port för webbservern."),
    autoreload: bool = typer.Option(
        True,
        "--autoreload/--no-autoreload",
        help="Ladda om servern automatiskt vid kodändringar.",
    ),
) -> None:
    """
    Starta den lokala backend-servern för API och webbgränssnitt.
    """
    uvicorn.run("app.api:app", host=host, port=port, reload=autoreload)


@app.command(
    "reset-index",
    help="Återskapa sökindexet i Qdrant från grunden.",
)
def reset_index() -> None:
    """
    Ta bort nuvarande samling och skapa om indexet.
    """
    try:
        store, _ = _build_store_and_embedder()
        store.recreate_collection()
    except Exception as exc:
        if _is_qdrant_lock_error(exc):
            raise typer.Exit(
                typer.echo(
                    "Qdrant-lagringen är låst av en annan process. "
                    "Stäng eventuell körande server eller annan process som använder indexet."
                )
            )
        raise

    typer.echo(
        f"Återskapade samlingen '{settings.collection_name}' i {settings.qdrant_path}"
    )


@app.command(
    "ingest",
    help="Läs in dokument från disk och indexera nya eller ändrade filer.",
)
def ingest(
    docs_path: Path | None = typer.Option(
        None,
        "--docs-path",
        help="Alternativ dokumentkatalog att läsa från.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Kör om alla dokument oavsett fingerprint.",
    ),
    sync_delete: bool = typer.Option(
        False,
        "--sync-delete",
        help="Ta bort indexerade dokument som inte längre finns på disk.",
    ),
) -> None:
    """
    Läs dokument från disk, extrahera innehåll och indexera chunkar i Qdrant.
    """
    root = docs_path or settings.docs_path
    if not root.exists():
        raise typer.BadParameter(f"Dokumentkatalog finns inte: {root}")

    try:
        store, embedder = _build_store_and_embedder()
    except Exception as exc:
        if _is_qdrant_lock_error(exc):
            typer.echo(
                "Qdrant-lagringen är låst av en annan process. "
                "Stäng eventuell körande server eller annan process som använder indexet."
            )
            raise typer.Exit(code=1)
        raise

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


@app.command(
    "reindex",
    help="Nollställ indexet och bygg upp det igen från dokument på disk.",
)
def reindex(
    docs_path: Path | None = typer.Option(
        None,
        "--docs-path",
        help="Alternativ dokumentkatalog att läsa från.",
    ),
) -> None:
    """
    Kör reset-index följt av ingest.
    """
    reset_index()
    ingest(docs_path=docs_path, force=False, sync_delete=False)


def _section_key(hit: SourceHit) -> tuple[str, str | None, int | None]:
    m = hit.metadata
    return (m.source_path, m.section_title, m.section_level)


def _section_text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@app.command(
    "enrich",
    help="Berika sektioner i indexet med semantisk metadata.",
)
def enrich(
    batch_size: int = typer.Option(
        256,
        help="Antal chunkar att läsa per batch från indexet.",
    ),
    limit_sections: int | None = typer.Option(
        None,
        "--limit-sections",
        help="Begränsa antal sektioner som analyseras.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Annotera alla sektioner igen.",
    ),
) -> None:
    """
    Läs indexerade sektioner och skriv tillbaka semantisk metadata till Qdrant.
    """
    import time

    try:
        store = _build_store_only()
    except Exception as exc:
        if _is_qdrant_lock_error(exc):
            typer.echo(
                "Qdrant-lagringen är låst av en annan process. "
                "Stäng eventuell körande server eller annan process som använder indexet."
            )
            raise typer.Exit(code=1)
        raise

    extractor = SectionMetadataExtractor()

    typer.echo("Läser indexerade chunkar från Qdrant ...")
    hits = store.iter_all_chunks(batch_size=batch_size)
    typer.echo(f"Antal chunkar lästa: {len(hits)}")

    grouped: dict[tuple[str, str | None, int | None], list[SourceHit]] = defaultdict(list)
    for hit in hits:
        grouped[_section_key(hit)].append(hit)

    section_items = list(grouped.items())
    section_items.sort(
        key=lambda item: (
            item[0][0],
            item[1][0].metadata.chunk_index if item[1] else 0,
        )
    )

    if limit_sections is not None:
        section_items = section_items[:limit_sections]

    updates: list[tuple[str, dict]] = []
    skipped = 0
    processed = 0

    total_sections = len(section_items)
    typer.echo(f"Antal sektioner att pröva: {total_sections}")

    t_start = time.perf_counter()

    for idx, ((source_path, section_title, _section_level), group_hits) in enumerate(
        section_items,
        start=1,
    ):
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

        t_section = time.perf_counter()

        semantic = extractor.extract(
            document_title=first_meta.document_title,
            section_title=section_title,
            text=section_text,
        )

        dt_section = time.perf_counter() - t_section
        elapsed = time.perf_counter() - t_start

        # Uppskatta återstående tid baserat på genomsnitt per processad sektion
        avg_per_section = elapsed / processed if processed > 0 else dt_section
        remaining_to_check = total_sections - idx
        eta_seconds = remaining_to_check * avg_per_section

        if eta_seconds >= 3600:
            eta_str = f"{eta_seconds / 3600:.1f}h"
        elif eta_seconds >= 60:
            eta_str = f"{eta_seconds / 60:.0f}min"
        else:
            eta_str = f"{eta_seconds:.0f}s"

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
            f"[{idx}/{total_sections}] "
            f"{Path(source_path).name} | rubrik={section_title!r} | "
            f"chunkar={len(group_hits)} | {dt_section:.1f}s | ETA {eta_str}"
        )

    t_total = time.perf_counter() - t_start

    typer.echo("Skriver tillbaka metadata till Qdrant ...")
    store.bulk_update_chunk_metadata(updates)

    if t_total >= 3600:
        total_str = f"{t_total / 3600:.1f}h"
    elif t_total >= 60:
        total_str = f"{t_total / 60:.1f}min"
    else:
        total_str = f"{t_total:.0f}s"

    typer.echo("")
    typer.echo(
        f"Klart. Annoterade sektioner: {processed}, hoppade över: {skipped}, "
        f"uppdaterade chunkar: {len(updates)}, total tid: {total_str}"
    )


@app.command(
    "backfill-enrich-status",
    help="Återskapa enrich-status utifrån befintlig semantisk metadata.",
)
def backfill_enrich_status(
    batch_size: int = typer.Option(
        256,
        help="Antal chunkar att läsa per batch från indexet.",
    ),
) -> None:
    """
    Sätt semantic_enriched på redan indexerade chunkar utifrån befintliga fält.
    """
    try:
        store = _build_store_only()
    except Exception as exc:
        if _is_qdrant_lock_error(exc):
            typer.echo(
                "Qdrant-lagringen är låst av en annan process. "
                "Stäng eventuell körande server eller annan process som använder indexet."
            )
            raise typer.Exit(code=1)
        raise

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

        updates.append(
            (
                hit.chunk_id,
                {
                    "semantic_enriched": has_semantic,
                },
            )
        )

    typer.echo("Skriver enrich-status till Qdrant ...")
    store.bulk_update_chunk_metadata(updates)
    typer.echo(f"Klart. Uppdaterade chunkar: {len(updates)}")


@app.command(
    "stats",
    help="Visa översikt över dokument på disk, indexerade chunkar, sektioner och enrich-status.",
)
def stats(
    docs_path: Path | None = typer.Option(
        None,
        "--docs-path",
        help="Alternativ dokumentkatalog att jämföra mot.",
    ),
    batch_size: int = typer.Option(
        256,
        help="Antal chunkar att läsa per batch från indexet.",
    ),
) -> None:
    """
    Visa status för dokument på disk och innehåll i indexet.
    """
    root = docs_path or settings.docs_path
    if not root.exists():
        raise typer.BadParameter(f"Dokumentkatalog finns inte: {root}")

    try:
        store = _build_store_only()
    except Exception as exc:
        if _is_qdrant_lock_error(exc):
            typer.echo(
                "Qdrant-lagringen är låst av en annan process. "
                "Stäng eventuell körande server eller annan process som använder indexet."
            )
            raise typer.Exit(code=1)
        raise

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


"""
Uppdatering av ask-kommandot i cli.py.

Ersätt den befintliga ask-funktionen med denna version.
Lägg till modulvariabel efter app = typer.Typer(...):
_cli_active_session_id: str | None = None
"""


_cli_active_session_id: str | None = None


@app.command(
    "ask",
    help="Ställ en fråga till dokumentchatten och visa svar med källor.",
)
def ask(
    question: str = typer.Argument(
        ...,
        help="Frågan som ska ställas till dokumentchatten.",
    ),
    show_sources: bool = typer.Option(
        True,
        "--sources/--no-sources",
        help="Visa eller dölj källor i svaret.",
    ),
    show_debug: bool = typer.Option(
        False,
        "--debug",
        help="Visa debug-information om retrieval och backend.",
    ),
    via_server: bool = typer.Option(
        False,
        "--via-server",
        help="Skicka frågan till en docchat-server.",
    ),
    server_url: str = typer.Option(
        "http://127.0.0.1:8000",
        "--server-url",
        help="URL till docchat-servern. Aktiverar server-läge automatiskt.",
    ),
    new_session: bool = typer.Option(
        False,
        "--new-session",
        help="Starta en ny session (glöm samtalshistorik).",
    ),
) -> None:
    """
    Ställ en fråga till dokumentchatten.

    Följdfrågor fungerar automatiskt inom samma session via servern.
    Använd --new-session för att börja om.

    Exempel:
      docchat ask "Vad gäller för externa forskningsansökningar?"
      docchat ask "Berätta mer om beslutsmötet" --via-server
      docchat ask "Ny fråga" --via-server --server-url http://100.96.76.110:8000
      docchat ask "Annat ämne" --new-session --via-server
    """
    global _cli_active_session_id

    default_url = "http://127.0.0.1:8000"

    if new_session:
        _cli_active_session_id = None

    # --server-url med icke-default värde aktiverar server-läge
    use_server = via_server or server_url != default_url or _server_is_available(default_url)

    if use_server:
        if show_debug:
            typer.echo(f"Backend: server ({server_url})")

        request_payload = {"question": question}
        if _cli_active_session_id:
            request_payload["session_id"] = _cli_active_session_id

        resp = requests.post(
            server_url.rstrip("/") + "/chat",
            json=request_payload,
            timeout=300,
        )
        if not resp.ok:
            raise RuntimeError(f"Serverfel {resp.status_code}: {resp.text}")

        data = resp.json()
        response = ChatResponse.model_validate(data)

        if response.session_id:
            _cli_active_session_id = response.session_id
    else:
        if show_debug:
            typer.echo("Backend: local")
        rag = RagService()
        response = rag.answer(question)

    _print_response(response, show_sources=show_sources, show_debug=show_debug)


def main() -> None:
    app()


if __name__ == "__main__":
    main()