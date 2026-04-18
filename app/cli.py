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
from app.ingest import (
    compute_source_fingerprint,
    ingest_path,
    ingest_evidence_path,
    ingest_path_with_evidence,
    iter_document_paths,
)
from app.preprocess_llm import SectionMetadataExtractor
from app.qdrant_store import QdrantStore
from app.retrieval import RagService
from app.schemas import ChatResponse, SourceHit

app = typer.Typer(
    help="URD Local source of knowledge about document content.",
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
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        help="Antal källträffar att använda (överskriver TOP_K i config).",
    ),
) -> None:
    """
    Starta den lokala backend-servern för API och webbgränssnitt.
    """
    if top_k is not None:
        os.environ["TOP_K"] = str(top_k)
        typer.echo(f"top_k satt till {top_k}")

    uvicorn.run("app.api:app", host=host, port=port, reload=autoreload)


@app.command(
    "connect",
    help="Starta lokal klient som servar webben lokalt och kopplar upp sig mot en urd-server.",
)
def connect(
    server: str | None = typer.Option(
        None,
        "--server",
        help="Upstream-server, t.ex. pop-os:8000 eller http://100.96.76.110:8000",
    ),
    host: str = typer.Option("127.0.0.1", help="Host för den lokala klienten."),
    port: int = typer.Option(8765, help="Port för den lokala klienten."),
    autoreload: bool = typer.Option(
        False,
        "--autoreload/--no-autoreload",
        help="Ladda om klienten automatiskt vid kodändringar.",
    ),
) -> None:
    """
    Starta ett lokalt URD-gränssnitt som proxar /chat och /document till en fjärrserver.

    Server väljs i denna ordning:
    1. --server
    2. config-nyckeln 'server'

    Exempel:
      urd connect --server pop-os:8000
      urd config set server pop-os:8000
      urd connect
    """
    upstream = (server or settings.server or "").strip()
    if not upstream:
        typer.echo(
            "Ingen server är angiven.\n"
            "Ange --server HOST:PORT eller sätt config-värdet 'server', t.ex.:\n"
            "  urd config set server pop-os:8000"
        )
        raise typer.Exit(code=1)

    if "://" not in upstream:
        upstream = "http://" + upstream

    os.environ["URD_UPSTREAM_SERVER"] = upstream

    typer.echo(f"Ansluter till URD-server: {upstream}")
    typer.echo(f"Lokal klient startas på: http://{host}:{port}")

    uvicorn.run(
        "app.connect_api:app",
        host=host,
        port=port,
        reload=autoreload,
        log_level="info",
        access_log=False,
    )


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
    indexed_with_evidence = store.get_evidence_source_paths()
    fs_paths = iter_document_paths(root)
    fs_map = {str(p): compute_source_fingerprint(p) for p in fs_paths}

    typer.echo(f"Läser dokument från: {root}")
    typer.echo(f"Antal filer funna: {len(fs_paths)}")
    typer.echo(f"Indexerade dokument i Qdrant: {len(indexed_docs)}")
    missing_evidence_count = sum(
        1
        for path in fs_paths
        if str(path) in indexed_docs and str(path) not in indexed_with_evidence
    )
    if missing_evidence_count:
        typer.echo(
            f"Dokument utan evidensobjekt: {missing_evidence_count} "
            f"(kommer processas för evidens även om fingerprint är oförändrat)"
        )

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
            store.delete_evidence_by_source_path(source_path)
            typer.echo(f"Deleted missing document from index: {source_path}")

    for path in fs_paths:
        source_path = str(path)
        new_fp = fs_map[source_path]
        old_fp = indexed_docs.get(source_path)

        fingerprint_unchanged = old_fp == new_fp
        has_evidence = source_path in indexed_with_evidence
        needs_evidence_backfill = fingerprint_unchanged and not has_evidence

        if not force and fingerprint_unchanged and has_evidence:
            skipped += 1
            typer.echo(f"Skip unchanged: {source_path}")
            continue

        if needs_evidence_backfill:
            # Chunks är korrekt indexerade men evidens saknas. Lägg
            # bara till evidensobjekt, utan att röra chunks.
            _, evidence_objects = ingest_path_with_evidence(path, root)
            if evidence_objects:
                evidence_vectors = embedder.embed_texts(
                    [e.evidence_text for e in evidence_objects]
                )
                store.upsert_evidence_objects(evidence_objects, evidence_vectors)
                updated += 1
                typer.echo(
                    f"Evidensbackfill: {source_path} -> "
                    f"{len(evidence_objects)} evidensobjekt"
                )
            else:
                skipped += 1
                typer.echo(
                    f"Evidensbackfill: {source_path} (inga evidensobjekt hittades)"
                )
            total_docs += 1
            continue

        if old_fp is not None:
            store.delete_chunks_by_source_path(source_path)
            store.delete_evidence_by_source_path(source_path)
            updated += 1
            typer.echo(f"Reingest changed: {source_path}")
        else:
            created += 1
            typer.echo(f"Ingest new: {source_path}")

        chunks, evidence_objects = ingest_path_with_evidence(path, root)
        if not chunks:
            typer.echo(f"Hoppar över {path} (inga chunkar)")
            continue

        vectors = embedder.embed_texts([c.text for c in chunks])
        store.upsert_chunks(chunks, vectors)

        if evidence_objects:
            evidence_vectors = embedder.embed_texts([e.evidence_text for e in evidence_objects])
            store.upsert_evidence_objects(evidence_objects, evidence_vectors)

        total_docs += 1
        total_chunks += len(chunks)
        typer.echo(
            f"Indexed {path} -> {len(chunks)} chunks"
            + (f", {len(evidence_objects)} evidensobjekt" if evidence_objects else "")
        )

    typer.echo("")
    typer.echo(
        f"Klart. Processade dokument: {total_docs}, chunkar: {total_chunks}, "
        f"skapade: {created}, uppdaterade: {updated}, hoppade över: {skipped}"
    )

    # Om servern körs, uppdatera BM25-indexet
    if total_docs > 0 and _server_is_available("http://127.0.0.1:8000"):
        try:
            resp = requests.post("http://127.0.0.1:8000/refresh", timeout=30)
            if resp.ok:
                data = resp.json()
                typer.echo(f"Serverns sökindex uppdaterat ({data.get('num_chunks', '?')} chunkar).")
            else:
                typer.echo("Varning: kunde inte uppdatera serverns sökindex.")
        except Exception:
            typer.echo("Varning: kunde inte nå servern för indexuppdatering.")


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


@app.command(
    "config",
    help="Visa eller ändra konfiguration i .urd/config.json.",
)
def config_cmd(
    action: str = typer.Argument(
        "show",
        help="Åtgärd: show, get, set, reset",
    ),
    key: str | None = typer.Argument(
        None,
        help="Config-nyckel (för get/set).",
    ),
    value: str | None = typer.Argument(
        None,
        help="Nytt värde (för set).",
    ),
) -> None:
    """
    Visa eller ändra konfiguration.
    """
    from app.config import DEFAULTS, CONFIG_FILE, _load_file_config, save_config_file, _ENV_KEYS

    if action == "show":
        file_config = _load_file_config()
        typer.echo(f"Konfigurationsfil: {CONFIG_FILE}")
        typer.echo("")
        for k, default in DEFAULTS.items():
            file_val = file_config.get(k)
            env_key = _ENV_KEYS.get(k, "")
            env_val = os.getenv(env_key) if env_key else None

            current = getattr(settings, k, default)

            if env_val is not None:
                source = f"env ({env_key})"
            elif file_val is not None and file_val != default:
                source = "config.json"
            else:
                source = "default"

            typer.echo(f"  {k}: {current}  ({source})")

    elif action == "get":
        if not key:
            typer.echo("Ange en nyckel, t.ex.: urd config get top_k")
            raise typer.Exit(code=1)
        if key not in DEFAULTS:
            typer.echo(f"Okänd nyckel: {key}")
            typer.echo(f"Tillgängliga nycklar: {', '.join(sorted(DEFAULTS.keys()))}")
            raise typer.Exit(code=1)
        current = getattr(settings, key, DEFAULTS[key])
        typer.echo(f"{key}: {current}")

    elif action == "set":
        if not key or value is None:
            typer.echo("Användning: urd config set <nyckel> <värde>")
            raise typer.Exit(code=1)
        if key not in DEFAULTS:
            typer.echo(f"Okänd nyckel: {key}")
            typer.echo(f"Tillgängliga nycklar: {', '.join(sorted(DEFAULTS.keys()))}")
            raise typer.Exit(code=1)

        file_config = _load_file_config()
        default = DEFAULTS[key]
        try:
            if isinstance(default, int):
                typed_value = int(value)
            elif isinstance(default, float):
                typed_value = float(value)
            else:
                typed_value = value
        except ValueError:
            typer.echo(f"Ogiltigt värde: {value} (förväntar {type(default).__name__})")
            raise typer.Exit(code=1)

        file_config[key] = typed_value
        save_config_file(file_config)
        typer.echo(f"{key}: {typed_value}")

    elif action == "reset":
        save_config_file(dict(DEFAULTS))
        typer.echo(f"Återställde {CONFIG_FILE} till defaults.")

    else:
        typer.echo(f"Okänd åtgärd: {action}")
        typer.echo("Tillgängliga: show, get, set, reset")
        raise typer.Exit(code=1)


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
        help="Skicka frågan till en urd-server.",
    ),
    server_url: str = typer.Option(
        "http://127.0.0.1:8000",
        "--server-url",
        help="URL till urd-servern. Aktiverar server-läge automatiskt.",
    ),
    new_session: bool = typer.Option(
        False,
        "--new-session",
        help="Starta en ny session (glöm samtalshistorik).",
    ),
) -> None:
    """
    Ställ en fråga till dokumentchatten.
    """
    global _cli_active_session_id

    default_url = "http://127.0.0.1:8000"

    if new_session:
        _cli_active_session_id = None

    use_server = via_server or server_url != default_url or _server_is_available(default_url)

    if use_server:
        if show_debug:
            typer.echo(f"Backend: server ({server_url})")

        request_payload = {"question": question}
        if _cli_active_session_id:
            request_payload["session_id"] = _cli_active_session_id

        try:
            resp = requests.post(
                server_url.rstrip("/") + "/chat",
                json=request_payload,
                timeout=300,
            )
        except requests.ConnectionError:
            typer.echo(
                f"Kunde inte ansluta till servern på {server_url}. "
                f"Starta servern med 'urd serve' först."
            )
            raise typer.Exit(code=1)

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


@app.command(
    "test",
    help="Kör ett sekvensbaserat testbatteri och rapportera utfall per sekvens och tur.",
)
def test(
    test_file: Path = typer.Option(
        "test/questions.json",
        "--file",
        "-f",
        help="Sökväg till JSON-fil med testsekvenser.",
    ),
    server_url: str = typer.Option(
        "http://127.0.0.1:8000",
        "--server-url",
        help="URL till urd-servern.",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Spara resultat till JSON-fil. Default: .urd/results/results_<timestamp>.json",
    ),
    only_sequence: str | None = typer.Option(
        None,
        "--only",
        help="Kör bara sekvensen med detta namn (för snabbare iteration).",
    ),
    show_answers: bool = typer.Option(
        True,
        "--answers/--no-answers",
        help="Visa svar i terminalen.",
    ),
    show_sources: bool = typer.Option(
        False,
        "--sources/--no-sources",
        help="Visa källor i terminalen.",
    ),
    pause_ms: int = typer.Option(
        500,
        "--pause-ms",
        help="Paus i millisekunder mellan turer i samma sekvens (skyddar servern).",
    ),
) -> None:
    """
    Kör testsekvenser mot servern och samla resultat.

    Testfilen ska ha formatet:

      {"version": 3, "sequences": [
        {"name": "...", "description": "...", "turns": [
          {"question": "...", "expect": {...}},
          ...
        ]}
      ]}

    Varje sekvens körs i en egen session (session_id delas mellan
    sekvensens turer). Expect-fälten rapporteras bredvid faktiskt
    utfall. Observationsbara flaggor som valideras uttryckligen:
      - should_find_sources, min_sources
      - should_abstain
      - should_detect_drift
      - expected_intent (matchar classification.intent)
      - expect_new_hits (elaboration har hämtat nytt material)
      - expect_verification_status (verification har producerat
        minst en finding med angiven status: supported/unclear/unsupported)

    Kvalitativa fält (notes, known_issue, sequence_role) rapporteras
    men valideras inte.
    """
    import time as time_module
    from datetime import datetime

    if not test_file.exists():
        typer.echo(f"Testfil saknas: {test_file}")
        typer.echo("")
        typer.echo("Filen ska vara i sekvensformat:")
        typer.echo('  {"version": 3, "sequences": [...]}')
        raise typer.Exit(code=1)

    try:
        with open(test_file, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        typer.echo(f"Kunde inte läsa testfilen: {e}")
        raise typer.Exit(code=1)

    if not isinstance(data, dict) or "sequences" not in data:
        typer.echo(
            "Testfilen måste vara ett objekt med nyckeln 'sequences'. "
            "Se docs för formatet."
        )
        raise typer.Exit(code=1)

    sequences = data.get("sequences", [])
    if not isinstance(sequences, list) or not sequences:
        typer.echo("Inga sekvenser att köra.")
        raise typer.Exit(code=1)

    if only_sequence:
        sequences = [s for s in sequences if s.get("name") == only_sequence]
        if not sequences:
            typer.echo(f"Ingen sekvens med namnet '{only_sequence}'.")
            raise typer.Exit(code=1)

    if not _server_is_available(server_url):
        typer.echo(
            f"Kunde inte ansluta till servern på {server_url}. "
            f"Starta servern med 'urd serve' först."
        )
        raise typer.Exit(code=1)

    typer.echo(f"Testfil: {test_file}")
    typer.echo(f"Sekvenser: {len(sequences)}")
    total_turns = sum(len(s.get("turns", [])) for s in sequences)
    typer.echo(f"Totalt antal turer: {total_turns}")
    typer.echo(f"Server: {server_url}")
    typer.echo("")

    # Resultat per sekvens
    sequence_results: list[dict] = []
    # Globala räknare
    total_flags: list[dict] = []
    all_times: list[float] = []

    for seq_idx, sequence in enumerate(sequences, start=1):
        seq_name = sequence.get("name", f"sequence_{seq_idx}")
        seq_description = sequence.get("description", "")
        turns = sequence.get("turns", [])

        typer.echo(f"=== Sekvens {seq_idx}/{len(sequences)}: {seq_name} ===")
        if seq_description:
            typer.echo(f"    {seq_description}")
        typer.echo("")

        session_id: str | None = None
        turn_results: list[dict] = []
        seq_flags: list[dict] = []

        for turn_idx, turn_spec in enumerate(turns, start=1):
            question = turn_spec.get("question", "").strip()
            expect = turn_spec.get("expect", {}) or {}

            if not question:
                typer.echo(f"  [{turn_idx}] Tom fråga — hoppar över")
                continue

            typer.echo(f"  [{turn_idx}/{len(turns)}] {question}")

            # Paus mellan turer (ej före första)
            if turn_idx > 1 and pause_ms > 0:
                time_module.sleep(pause_ms / 1000.0)

            payload = {"question": question}
            if session_id:
                payload["session_id"] = session_id

            try:
                resp = requests.post(
                    server_url.rstrip("/") + "/chat",
                    json=payload,
                    timeout=300,
                )
            except requests.ConnectionError:
                typer.echo("    Anslutningen bröts — avbryter.")
                turn_results.append({
                    "turn": turn_idx,
                    "question": question,
                    "expect": expect,
                    "error": "connection_error",
                })
                break
            except Exception as e:
                typer.echo(f"    Fel: {e}")
                turn_results.append({
                    "turn": turn_idx,
                    "question": question,
                    "expect": expect,
                    "error": str(e),
                })
                continue

            if not resp.ok:
                typer.echo(f"    Serverfel {resp.status_code}")
                turn_results.append({
                    "turn": turn_idx,
                    "question": question,
                    "expect": expect,
                    "error": f"HTTP {resp.status_code}",
                })
                continue

            try:
                response = ChatResponse.model_validate(resp.json())
            except Exception as e:
                typer.echo(f"    Kunde inte tolka svaret: {e}")
                turn_results.append({
                    "turn": turn_idx,
                    "question": question,
                    "expect": expect,
                    "error": f"parse_error: {e}",
                })
                continue

            # Behåll session_id från första svaret
            if response.session_id and not session_id:
                session_id = response.session_id

            debug = response.debug or {}
            classification = debug.get("classification", {}) or {}
            qud = debug.get("qud", {}) or {}
            qud_drift = debug.get("qud_drift", {}) or {}
            synthesis = debug.get("synthesis", {}) or {}
            timing = debug.get("timing_s", {}) or {}

            total_time = timing.get("total", 0)
            if isinstance(total_time, (int, float)) and total_time > 0:
                all_times.append(float(total_time))

            # Bygg en kompakt resultatrepresentation
            intent_str = classification.get("intent", "?")
            substyle = classification.get("substyle")
            if substyle:
                intent_str = f"{intent_str}/{substyle}"

            num_sources = len(response.sources)
            path = debug.get("path", "?")

            # Sammanfattningsrad i terminalen
            parts = [f"{total_time:.1f}s", f"intent={intent_str}", f"path={path}", f"hits={num_sources}"]
            if qud_drift:
                parts.append(
                    f"drift={qud_drift.get('similarity', '?')}"
                    f"{'*' if qud_drift.get('drift_detected') else ''}"
                )
            if synthesis.get("used_fallback"):
                parts.append(f"FALLBACK={synthesis.get('fallback_reason', '?')}")
            typer.echo(f"    {' | '.join(parts)}")

            # Utvärdera expect-flaggor (valideras)
            # num_new_hits finns bara på rework-vägen (elaboration/verification)
            num_new_hits = debug.get("num_new_hits")
            # status_counts finns bara när verification körts
            verification_status_counts = synthesis.get("status_counts")

            flags = _evaluate_expect(
                expect=expect,
                num_sources=num_sources,
                intent=classification.get("intent", ""),
                qud_drift_detected=qud_drift.get("drift_detected", False),
                abstained=debug.get("abstained", False),
                num_new_hits=num_new_hits,
                verification_status_counts=verification_status_counts,
            )
            for flag in flags:
                icon = "✓" if flag["ok"] else "✗"
                typer.echo(f"    {icon} {flag['label']}")
                if not flag["ok"]:
                    seq_flags.append({
                        "turn": turn_idx,
                        "question": question,
                        **flag,
                    })
                    total_flags.append({
                        "sequence": seq_name,
                        "turn": turn_idx,
                        "question": question,
                        **flag,
                    })

            # Notes från testfilen (informativt)
            if expect.get("notes"):
                typer.echo(f"      anteckning: {expect['notes']}")
            if expect.get("known_issue"):
                typer.echo(f"      känt problem: {expect['known_issue']}")

            if show_answers:
                for line in response.answer.splitlines():
                    typer.echo(f"    | {line}")

            if show_sources:
                for j, src in enumerate(response.sources, start=1):
                    typer.echo(
                        f"    [{j}] {src.metadata.file_name} "
                        f"({src.metadata.section_title}) "
                        f"score={src.score:.3f}"
                    )

            turn_results.append({
                "turn": turn_idx,
                "question": question,
                "expect": expect,
                "answer": response.answer,
                "num_sources": num_sources,
                "sources": [
                    {
                        "file_name": s.metadata.file_name,
                        "section_title": s.metadata.section_title,
                        "score": round(s.score, 3),
                    }
                    for s in response.sources
                ],
                "classification": classification,
                "path": path,
                "qud": qud,
                "qud_drift": qud_drift or None,
                "synthesis": {
                    k: v for k, v in synthesis.items()
                    if k != "evidence_json"
                },
                "timing_s": timing,
                "flags": flags,
            })

            typer.echo("")

        sequence_results.append({
            "name": seq_name,
            "description": seq_description,
            "session_id": session_id,
            "turns": turn_results,
            "failed_flags": seq_flags,
        })

        if seq_flags:
            typer.echo(f"  Sekvensen '{seq_name}' hade {len(seq_flags)} avvikelse(r).")
        typer.echo("")

    # Sammanfattning
    typer.echo("Sammanfattning")
    typer.echo("==============")
    typer.echo(f"Sekvenser körda:    {len(sequence_results)}")
    typer.echo(f"Turer körda:        {sum(len(s['turns']) for s in sequence_results)}")
    typer.echo(f"Avvikande flaggor:  {len(total_flags)}")

    if all_times:
        typer.echo(f"Medeltid per tur:   {sum(all_times) / len(all_times):.1f}s")
        typer.echo(f"Min/max tid:        {min(all_times):.1f}s / {max(all_times):.1f}s")

    if total_flags:
        typer.echo("")
        typer.echo("Avvikelser per sekvens")
        typer.echo("----------------------")
        for seq in sequence_results:
            if seq["failed_flags"]:
                typer.echo(f"  {seq['name']}:")
                for f in seq["failed_flags"]:
                    typer.echo(f"    tur {f['turn']}: {f['label']}")

    # Spara resultat
    if output_file is None:
        results_dir = Path(".urd/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"results_{timestamp}.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_file": str(test_file),
                "server_url": server_url,
                "timestamp": datetime.now().isoformat(),
                "num_sequences": len(sequence_results),
                "num_turns": sum(len(s["turns"]) for s in sequence_results),
                "num_flag_failures": len(total_flags),
                "mean_time_s": round(sum(all_times) / len(all_times), 3) if all_times else None,
                "sequences": sequence_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    typer.echo("")
    typer.echo(f"Resultat sparade: {output_file}")


def _evaluate_expect(
    expect: dict,
    num_sources: int,
    intent: str,
    qud_drift_detected: bool,
    abstained: bool,
    num_new_hits: int | None,
    verification_status_counts: dict | None,
) -> list[dict]:
    """
    Utvärdera observationsbara expect-flaggor.

    Returnerar en lista av {label, ok, detail} för varje flagga som
    faktiskt var angiven i expect. Kvalitativa fält (notes,
    known_issue, sequence_role, same_topic_as_previous) valideras
    INTE — de är rapportering, inte pass/fail.

    Parametrarna num_new_hits och verification_status_counts är bara
    meningsfulla för rework-turer (elaboration/verification). För
    andra turer kan de vara None.
    """
    flags: list[dict] = []

    if "should_find_sources" in expect:
        want = bool(expect["should_find_sources"])
        got = num_sources > 0
        ok = (got == want)
        flags.append({
            "label": f"should_find_sources={want} (faktiskt: {num_sources} källor)",
            "ok": ok,
            "field": "should_find_sources",
            "expected": want,
            "actual": got,
        })

    if "min_sources" in expect:
        minimum = int(expect["min_sources"])
        ok = num_sources >= minimum
        flags.append({
            "label": f"min_sources={minimum} (faktiskt: {num_sources})",
            "ok": ok,
            "field": "min_sources",
            "expected": minimum,
            "actual": num_sources,
        })

    if "should_abstain" in expect:
        want = bool(expect["should_abstain"])
        ok = (abstained == want)
        flags.append({
            "label": f"should_abstain={want} (faktiskt: {abstained})",
            "ok": ok,
            "field": "should_abstain",
            "expected": want,
            "actual": abstained,
        })

    if "should_detect_drift" in expect:
        want = bool(expect["should_detect_drift"])
        ok = (qud_drift_detected == want)
        flags.append({
            "label": f"should_detect_drift={want} (faktiskt: {qud_drift_detected})",
            "ok": ok,
            "field": "should_detect_drift",
            "expected": want,
            "actual": qud_drift_detected,
        })

    if "expected_intent" in expect:
        want = str(expect["expected_intent"])
        ok = (intent == want)
        flags.append({
            "label": f"expected_intent={want} (faktiskt: {intent})",
            "ok": ok,
            "field": "expected_intent",
            "expected": want,
            "actual": intent,
        })

    if "expect_new_hits" in expect:
        want = bool(expect["expect_new_hits"])
        # Elaboration lyckades hämta nytt material om num_new_hits > 0.
        # För icke-rework-turer är num_new_hits None och flaggan är
        # inte meningsfull — markera som fail med förklarande text.
        if num_new_hits is None:
            ok = False
            detail = "(ej rework-tur — num_new_hits saknas)"
        else:
            got = num_new_hits > 0
            ok = (got == want)
            detail = f"(faktiskt: {num_new_hits} nya hits)"
        flags.append({
            "label": f"expect_new_hits={want} {detail}",
            "ok": ok,
            "field": "expect_new_hits",
            "expected": want,
            "actual": num_new_hits,
        })

    if "expect_verification_status" in expect:
        want = str(expect["expect_verification_status"])
        # Kräver att minst en finding har den begärda statusen.
        if verification_status_counts is None:
            ok = False
            detail = "(ingen verification körd)"
            actual_count = None
        else:
            actual_count = verification_status_counts.get(want, 0)
            ok = actual_count > 0
            detail = (
                f"(supported={verification_status_counts.get('supported', 0)}, "
                f"unclear={verification_status_counts.get('unclear', 0)}, "
                f"unsupported={verification_status_counts.get('unsupported', 0)})"
            )
        flags.append({
            "label": f"expect_verification_status={want} {detail}",
            "ok": ok,
            "field": "expect_verification_status",
            "expected": want,
            "actual": actual_count,
        })

    return flags


def main() -> None:
    app()


if __name__ == "__main__":
    main()