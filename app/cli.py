#!/usr/bin/env python3
from __future__ import annotations

import os

# Försök tvinga Typer till enkel text-help utan Rich-paneler.
# Måste sättas före import av typer.
os.environ["TYPER_USE_RICH"] = "0"

import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import requests
import typer
import uvicorn

# Samma logghantering som i api.py: utan handler på rotloggern
# försvinner appmodulernas varningar tyst — bl.a. ingest-lagrets
# "Extraction failed", som är exakt den rad som förklarar varför
# ett dokument saknas i indexet. Konfigurera bara om ingen handler
# redan finns, och dämpa pratiga tredjepartsbibliotek.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
for _noisy in ("httpx", "huggingface_hub", "urllib3", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from app.config import settings
from app.embeddings import Embedder
from app.ingest import (
    compute_source_fingerprint,
    ingest_path,
    ingest_evidence_path,
    ingest_path_with_evidence,
    iter_document_paths,
)
from app.qdrant_store import QdrantStore
from app.retrieval import RagService
from app.schemas import ChatResponse, SourceHit

app = typer.Typer(
    help="URD Local source of knowledge about document content.",
    # no_args_is_help avstängt: `urd` utan argument går till
    # interaktivt läge, som `python`. Hjälpen nås med --help.
    no_args_is_help=False,
    invoke_without_command=True,
    add_completion=False,
    rich_markup_mode=None,
)


@app.callback()
def _root(
    ctx: typer.Context,
    server_url: str = typer.Option(
        "http://127.0.0.1:8000",
        "--server-url",
        help="URL till urd-servern (interaktivt läge).",
    ),
    sources: bool = typer.Option(
        True, "--sources/--no-sources",
        help="Visa källor i interaktivt läge.",
    ),
    debug: bool = typer.Option(
        False, "--debug",
        help="Visa teknisk info i interaktivt läge.",
    ),
) -> None:
    """
    Utan underkommando startas interaktivt läge.

    Skillnaden mot `urd ask` är att sessionen LEVER: QUD, aktiva
    dokument och rework-tillstånd finns kvar mellan frågor, vilket är
    hela poängen med URD:s samtalsmodell. Varje `urd ask` startar en ny
    process och tappar allt det.
    """
    if ctx.invoked_subcommand is not None:
        return
    from app.repl import run
    run(server_url=server_url, show_sources=sources, show_debug=debug)


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
    # PID-fil så att 'urd stop' kan avsluta servern kontrollerat.
    #
    # PID-fil hellre än en shutdown-endpoint: en HTTP-väg som dödar
    # processen bör inte ligga öppen ens på localhost utan
    # auktorisering, och signalvägen fungerar även om servern hängt sig.
    #
    # Med autoreload kör uvicorn TVÅ processer — en reloader och en
    # arbetare. os.getpid() här är reloaderns, vilket är den rätta:
    # dödar man arbetaren startar reloadern bara en ny.
    pid_file = Path(".urd") / "server.pid"
    try:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(f"{os.getpid()}\n{host}:{port}\n", encoding="utf-8")
    except OSError as e:
        logger.warning("Kunde inte skriva %s (%s).", pid_file, e)

    try:
        uvicorn.run("app.api:app", host=host, port=port, reload=autoreload)
    finally:
        try:
            if pid_file.exists() and pid_file.read_text(
                encoding="utf-8"
            ).split("\n")[0].strip() == str(os.getpid()):
                pid_file.unlink()
        except OSError:
            pass


@app.command(
    "stop",
    help="Avsluta en lokalt körande urd-server.",
)
def stop(
    force: bool = typer.Option(
        False, "--force",
        help="Skicka SIGKILL i stället för SIGTERM.",
    ),
) -> None:
    """
    Avsluta servern via PID-filen som 'urd serve' skriver.

    Kontrollerar att processen fortfarande lever innan signalen skickas,
    så att en kvarlämnad PID-fil efter en krasch inte får kommandot att
    döda en obesläktad process som råkat få samma PID.
    """
    import signal

    pid_file = Path(".urd") / "server.pid"
    if not pid_file.exists():
        typer.echo(
            "Ingen server.pid i .urd/. Antingen kör ingen server, eller så "
            "startades den utan 'urd serve'.\n"
            "Hitta processen med:  lsof data/qdrant/.lock"
        )
        raise typer.Exit(code=1)

    lines = pid_file.read_text(encoding="utf-8").splitlines()
    try:
        pid = int(lines[0].strip())
    except (IndexError, ValueError):
        typer.echo(f"Kunde inte läsa PID ur {pid_file}.")
        raise typer.Exit(code=1)

    address = lines[1].strip() if len(lines) > 1 else "okänd adress"

    try:
        os.kill(pid, 0)          # lever processen?
    except ProcessLookupError:
        typer.echo(
            f"Ingen process med PID {pid} — servern verkar redan avslutad."
        )
        pid_file.unlink(missing_ok=True)
        raise typer.Exit(code=0)
    except PermissionError:
        typer.echo(f"Saknar behörighet att signalera PID {pid}.")
        raise typer.Exit(code=1)

    sig = signal.SIGKILL if force else signal.SIGTERM
    os.kill(pid, sig)
    typer.echo(f"Skickade {sig.name} till PID {pid} ({address}).")
    pid_file.unlink(missing_ok=True)


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
    # Dokument som inte kunde indexeras: (source_path, orsak).
    # Rapporteras samlat i slutet — en enskild rad mitt i hundratals
    # skip-rader är i praktiken osynlig, och ett dokument som saknas
    # i indexet är osynligt för all sökning.
    extraction_failures: list[tuple[str, str]] = []

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
            _, evidence_objects, _err = ingest_path_with_evidence(path, root)
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

        chunks, evidence_objects, extract_error = ingest_path_with_evidence(path, root)
        if not chunks:
            reason = extract_error or "inga chunkar producerades"
            typer.echo(f"MISSLYCKADES: {path} — {reason}")
            extraction_failures.append((source_path, reason))
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

    if extraction_failures:
        typer.echo("")
        typer.echo(f"VARNING: {len(extraction_failures)} dokument kunde INTE indexeras:")
        for failed_path, reason in extraction_failures:
            typer.echo(f"  - {failed_path}")
            typer.echo(f"      orsak: {reason}")
        typer.echo(
            "Dessa dokument är osynliga för all sökning tills de indexerats. "
            "Ej indexerade dokument försöks om automatiskt vid nästa 'urd ingest'."
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


@app.command(
    "stats",
    help="Visa översikt över dokument på disk, indexerade chunkar och sektioner.",
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
    typer.echo("Attest")
    typer.echo("------")
    try:
        from app import attest
        conn = attest.connect()
        cov = attest.coverage(conn, list(indexed_docs.keys()))
        totals = attest.stats(conn)
        typer.echo(f"Observationer:           {totals['observations']}")
        typer.echo(f"Dokument med belägg:     {cov['documents_with_observations']}")
        typer.echo(f"Dokument utan belägg:    {cov['documents_without_observations']}")
        if cov["stale_documents"]:
            typer.echo(
                f"Stale i attest.db:       {len(cov['stale_documents'])}"
                "  (kör attest-build)"
            )
        for path in cov["without_observations"][:10]:
            typer.echo(f"    {Path(path).name}")
        if cov["documents_without_observations"] > 10:
            typer.echo(
                f"    ... och {cov['documents_without_observations'] - 10} till"
            )
    except Exception as e:
        # Attest är ett tillägg, inte en förutsättning för stats.
        typer.echo(f"Attest otillgängligt: {e}")

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
        help="Åtgärd: show, get, set, reset, validate",
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
    from app.config import (
        DEFAULTS, CONFIG_FILE, _load_file_config, save_config_file, _ENV_KEYS,
        parse_bool as _parse_bool, _BOOL_TRUE, _BOOL_FALSE,
    )

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
            # bool MÅSTE prövas före int: bool är en subklass till int i
            # Python, så isinstance(True, int) är sant. Utan den här
            # ordningen hamnar booleska nycklar i int()-grenen, och
            # 'urd config set predication_enabled true' fallerar med
            # "Ogiltigt värde: true (förväntar bool)" — ett felmeddelande
            # som beskriver rätt typ men ändå vägrar den.
            if isinstance(default, bool):
                typed_value = _parse_bool(value)
            elif isinstance(default, int):
                typed_value = int(value)
            elif isinstance(default, float):
                typed_value = float(value)
            else:
                typed_value = value
        except ValueError:
            if isinstance(default, bool):
                typer.echo(
                    f"Ogiltigt värde: {value} (förväntar bool — "
                    f"{', '.join(sorted(_BOOL_TRUE))} eller "
                    f"{', '.join(sorted(_BOOL_FALSE))})"
                )
            else:
                typer.echo(
                    f"Ogiltigt värde: {value} (förväntar {type(default).__name__})"
                )
            raise typer.Exit(code=1)

        file_config[key] = typed_value
        save_config_file(file_config)
        typer.echo(f"{key}: {typed_value}")

    elif action == "reset":
        save_config_file(dict(DEFAULTS))
        typer.echo(f"Återställde {CONFIG_FILE} till defaults.")

    elif action == "validate":
        from app.config_validation import validate_config_files, format_report_lines

        report = validate_config_files(
            synonyms_path=settings.synonyms_path,
            concepts_path=settings.concepts_path,
            question_operations_path=settings.question_operations_path,
        )
        for line in format_report_lines(report):
            typer.echo(line)
        typer.echo("")
        if report.ok:
            typer.echo("Konfigurationen är giltig.")
        else:
            typer.echo(
                "Konfigurationen har FEL. Funktioner med felmarkerade filer "
                "är helt eller delvis avstängda tills felen rättas."
            )
            raise typer.Exit(code=1)

    else:
        typer.echo(f"Okänd åtgärd: {action}")
        typer.echo("Tillgängliga: show, get, set, reset, validate")
        raise typer.Exit(code=1)


_cli_active_session_id: str | None = None


@app.command(
    "attest-build",
    help=(
        "Bygg Attest-indexet: grammatiska observationer ur hela beståndet. "
        "Kräver att servern är avstängd."
    ),
)
def attest_build(
    limit: int = typer.Option(
        0, "--limit",
        help="Bygg bara de N första dokumenten (för att pröva innan full körning).",
    ),
    only_changed: bool = typer.Option(
        False, "--only-changed",
        help="Hoppa över dokument vars fingerprint är oförändrat (inkrementellt).",
    ),
    pattern: str = typer.Option(
        "", "--pattern",
        help=(
            "Bygg bara dokument vars sökväg innehåller denna substräng. "
            "--limit ensamt tar de alfabetiskt första, vilket i beståndet "
            "ger enbart regeldokument — konstruktionerna skiljer sig "
            "mellan textsorter, så pröva mot t.ex. 'IL-protokoll'."
        ),
    ),
) -> None:
    from app import attest
    from app.retrieval import RagService

    typer.echo("Laddar RagService (kräver att servern är avstängd)...")
    rag = RagService()
    chunks = rag.bm25_index.hits
    if pattern:
        chunks = [c for c in chunks if pattern.lower() in c.metadata.source_path.lower()]
        typer.echo(f"Filtrerat på {pattern!r}: {len(chunks)} chunkar")
    conn = attest.connect()

    def progress(i, total, path, n):
        typer.echo(f"  [{i}/{total}] {Path(path).name}: {n} observationer")

    try:
        result = attest.build(
            chunks, conn, only_changed=only_changed,
            limit=limit or None, progress=progress,
        )
    except RuntimeError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    typer.echo("")
    typer.echo(f"Dokument byggda:    {result['documents']}")
    typer.echo(f"Överhoppade:        {result['skipped']}")
    typer.echo(f"Observationer:      {result['observations']}")
    typer.echo(f"Tid:                {result['seconds']}s")
    typer.echo("")
    typer.echo(f"Totalt i indexet:   {attest.stats(conn)}")


@app.command(
    "attest-sample",
    help=(
        "Slumpa observationer för HANDKLASSNING. Precision per "
        "konstruktion måste mätas innan en konstruktion får användas."
    ),
)
def attest_sample(
    n: int = typer.Option(30, "--n", help="Antal observationer att slumpa."),
    kind: str = typer.Option("identitet", "--kind", help="Dragtyp."),
    construction: str = typer.Option("", "--construction", help="Filtrera på konstruktion."),
) -> None:
    from app import attest
    import sqlite3

    conn = attest.connect()
    conn.row_factory = sqlite3.Row
    sql = "SELECT * FROM observations WHERE kind = ?"
    args = [kind]
    if construction:
        sql += " AND construction = ?"
        args.append(construction)
    sql += " ORDER BY RANDOM() LIMIT ?"
    args.append(n)

    rows = list(conn.execute(sql, args))
    if not rows:
        typer.echo("Inga observationer matchade.")
        return

    for i, r in enumerate(rows, start=1):
        # Tvetydiga observationer får inte bokföras som fel vid
        # handklassning: konstruktionen tillåter mer än en läsning, och
        # att systemet redovisar det är rätt beteende.
        flag = "  [TVETYDIG]" if r["ambiguous"] else ""
        typer.echo(
            f"{i:3}. [{r['construction']}] {r['subject']}  ->  {r['object']}{flag}"
        )
        typer.echo(f"     {(r['sentence'] or '')[:120]}")
        typer.echo(f"     {r['file_name']}")
        typer.echo("")
    typer.echo(
        f"{len(rows)} observationer. Klassa varje som korrekt eller inte och\n"
        "räkna precision per konstruktion. Under 90 % bör konstruktionen\n"
        "strykas ur uttaget — bedömning på stickprov, inte på intryck."
    )


@app.command(
    "attest-coverage",
    help="Mät uttagets täckning för en term: textförekomster mot observationer.",
)
def attest_coverage(
    term: str = typer.Argument(..., help="Term att mäta, t.ex. studierektor"),
) -> None:
    from app import attest, inspect as ins

    try:
        store = _build_store_only()
    except Exception as exc:
        if _is_qdrant_lock_error(exc):
            raise StorageLockedError() from exc
        raise

    conn = attest.connect()
    try:
        cov = ins.term_coverage(term, store, conn=conn)
    finally:
        conn.close()

    typer.echo(ins.format_term_coverage(cov))
    # Grep-konventionen: 1 = inget att rapportera, inte ett fel.
    if cov.text_occurrences == 0:
        raise typer.Exit(code=1)


@app.command(
    "attest-lookup",
    help="Slå upp vad beståndet belägger om en term.",
)
def attest_lookup(
    term: str = typer.Argument(..., help="Roll eller namn att slå upp."),
    by: str = typer.Option(
        "object", "--by",
        help="'object' för 'vem är X', 'subject' för 'vad är X'.",
    ),
) -> None:
    from app import attest

    conn = attest.connect()
    fn = attest.lookup_object if by == "object" else attest.lookup_subject
    cands = fn(conn, term)

    if not cands:
        typer.echo(f"Inga observationer för {term!r}.")
        return

    typer.echo(f"{term!r} — {len(cands)} kandidat(er), rangordnade efter relevans.")
    typer.echo("")
    for c in cands:
        flag = "  [ENDAST TVETYDIGA BELÄGG]" if c.ambiguous_only else ""
        span = f"{c.first_date or '?'} – {c.last_date or '?'}"
        typer.echo(f"  {c.subject}  ->  {c.object}{flag}")
        typer.echo(
            f"      relevans {c.relevance:.2f}"
            f"  (styrka {c.strength:.2f} × aktualitet {c.recency:.2f})"
        )
        typer.echo(
            f"      {c.documents} dokument"
            f" varav {c.unambiguous_documents} entydiga,"
            f" {c.observations} observationer, {span}"
        )
        typer.echo(f"      konstruktioner: {', '.join(c.constructions)}")
        if c.scopes:
            typer.echo(f"      avser: {', '.join(c.scopes)}")
        if c.statuses:
            typer.echo(
                f"      status: {', '.join(c.statuses)}"
                f"  ({c.confirmed_documents} bekräftade dokument)"
            )
        for sent in c.sentences[:2]:
            typer.echo(f"      \"{sent[:100]}\"")
        typer.echo("")
    typer.echo("Beläggning är inte sanning: siffrorna mäter hur ofta något")
    typer.echo("skrivits i beståndet, inte om det stämmer. Flera kandidater kan")
    typer.echo("vara korrekta samtidigt — en roll kan ha flera innehavare.")


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
        None,
        "--file",
        "-f",
        help=(
            "Sökväg till JSON-fil med testsekvenser. Utan flagga används "
            ".urd/questions.json om den finns, annars "
            "test/questions.example.json."
        ),
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
    jsonl: bool = typer.Option(
        True,
        "--jsonl/--no-jsonl",
        help="Skriv fullständigt diagnostikspår som JSONL (en rad per tur, "
             "inkl. hela debug-blocket med rerank_top). Jämför två körningar "
             "med scripts/compare_test_runs.py.",
    ),
) -> None:
    """
    Kör testsekvenser mot servern och samla resultat.

    Testfilen ska ha formatet:

      {"version": 4, "sequences": [
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
      - expected_docs (substrängar som ska matcha filnamnen bland de
        källor som bar svaret)
      - expected_docs_in_retrieval (substrängar som ska matcha filnamn
        i retrieval-rankningen, dokumentnivå-hit@k; k styrs med
        retrieval_top_k, default 5)
      - answer_must_contain (substrängar som ska finnas i svaret;
        whitespace-okänslig matchning så att "10 000" matchar "10 000:-")
      - answer_must_not_contain (substrängar som INTE får finnas i
        svaret; samma normalisering. För felbindningar: ett namn eller
        en roll som svaret bevisligen inte ska tillskriva frågans
        subjekt)

    Kvalitativa fält (notes, known_issue, sequence_role) rapporteras
    men valideras inte.

    Skillnaden mellan expected_docs och expected_docs_in_retrieval är
    diagnostisk: den första mäter om rätt dokument BAR svaret, den
    andra om retrieval alls FANN dokumentet. Faller den första men
    inte den andra sitter felet i urvalet, inte i sökningen.
    """
    import time as time_module
    from datetime import datetime

    # Instansens batteri går före repots exempel.
    #
    # Detta MÅSTE ske före test_file.exists(): utan --file är test_file
    # None, och kontrollen kraschade med AttributeError. Blocket låg
    # tidigare längre ned, efter en rad som aldrig nåddes.
    #
    # Testfall som mäter rollbindning och aktualitet kräver verkliga
    # namn ur beståndet för att mäta något, och de hör därför hemma i
    # instansen — samma princip som docs/ och .urd/.
    if test_file is None:
        instance_file = Path(".urd") / "questions.json"
        example_file = Path("test") / "questions.example.json"
        if instance_file.exists():
            test_file = instance_file
        elif example_file.exists():
            test_file = example_file
            typer.echo(
                "Använder exempelbatteriet. Instansens eget batteri läggs i "
                ".urd/questions.json och versionshanteras inte."
            )
        else:
            typer.echo(
                "Hittade varken .urd/questions.json eller "
                "test/questions.example.json. Ange fil med --file."
            )
            raise typer.Exit(code=1)

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

    # Bestäm utdatavägar FÖRE körningen så att JSONL-spåret kan
    # skrivas löpande, en rad per tur. Kraschar körningen halvvägs
    # finns spåret fram till dess — det är hela poängen med en
    # append-logg för diagnostik.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file is None:
        results_dir = Path(".urd/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        output_file = results_dir / f"results_{timestamp}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    jsonl_file: Path | None = None
    jsonl_handle = None
    if jsonl:
        jsonl_file = output_file.with_suffix(".jsonl")
        jsonl_handle = open(jsonl_file, "w", encoding="utf-8")
        jsonl_handle.write(json.dumps({
            "type": "run_meta",
            "timestamp": datetime.now().isoformat(),
            "test_file": str(test_file),
            "server_url": server_url,
            "git_commit": _current_git_commit(),
            # Modell och resonemangsläge i spåret: två körningar med
            # samma commit kan annars inte skiljas åt i efterhand.
            "ollama_model": settings.ollama_model,
            "llm_think": settings.llm_think,
            "llm_num_ctx": settings.llm_num_ctx,
        }, ensure_ascii=False) + "\n")

    def _write_jsonl(record: dict) -> None:
        if jsonl_handle is not None:
            jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_handle.flush()

    # Resultat per sekvens
    sequence_results: list[dict] = []
    # Globala räknare
    total_flags: list[dict] = []
    all_flags: list[dict] = []
    all_times: list[float] = []
    # Turer som aldrig producerade ett svar (nätverksfel, HTTP-fel,
    # otolkbart svar). En sådan tur utvärderas inte mot sina expect-
    # fält och får därför aldrig en avvikande flagga — utan egen
    # räkning blir den osynlig i sammanfattningen, och körningen ser
    # normal ut trots att turer saknas. Bekräftat 2026-08-14: en
    # omstart av ollama mitt i en körning dödade två turer utan att
    # något syntes i summeringen.
    failed_turns: list[dict] = []

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
                _write_jsonl({
                    "type": "turn", "sequence": seq_name, "turn": turn_idx,
                    "question": question, "expect": expect,
                    "error": "connection_error",
                })
                break
            except Exception as e:
                typer.echo(f"    Fel: {e}")
                failed_turns.append({
                    "sequence": seq_name,
                    "turn": turn_idx,
                    "question": question,
                    "error": str(e),
                })
                turn_results.append({
                    "turn": turn_idx,
                    "question": question,
                    "expect": expect,
                    "error": str(e),
                })
                _write_jsonl({
                    "type": "turn", "sequence": seq_name, "turn": turn_idx,
                    "question": question, "expect": expect, "error": str(e),
                })
                continue

            if not resp.ok:
                typer.echo(f"    Serverfel {resp.status_code}")
                failed_turns.append({
                    "sequence": seq_name,
                    "turn": turn_idx,
                    "question": question,
                    "error": f"HTTP {resp.status_code}",
                })
                turn_results.append({
                    "turn": turn_idx,
                    "question": question,
                    "expect": expect,
                    "error": f"HTTP {resp.status_code}",
                })
                _write_jsonl({
                    "type": "turn", "sequence": seq_name, "turn": turn_idx,
                    "question": question, "expect": expect,
                    "error": f"HTTP {resp.status_code}",
                })
                continue

            try:
                response = ChatResponse.model_validate(resp.json())
            except Exception as e:
                typer.echo(f"    Kunde inte tolka svaret: {e}")
                failed_turns.append({
                    "sequence": seq_name,
                    "turn": turn_idx,
                    "question": question,
                    "error": f"parse_error: {e}",
                })
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
                drift_str = f"drift={qud_drift.get('similarity', '?')}"
                if qud_drift.get("doc_similarity") is not None:
                    drift_str += f"/doc={qud_drift.get('doc_similarity')}"
                if qud_drift.get("drift_detected"):
                    drift_str += "*"
                parts.append(drift_str)
            if debug.get("context_fallback"):
                cf = debug["context_fallback"]
                parts.append(
                    "kontextfallback="
                    + ("räddad" if cf.get("rescued") else "hjälpte ej")
                )
            if synthesis.get("used_fallback"):
                parts.append(f"FALLBACK={synthesis.get('fallback_reason', '?')}")
            typer.echo(f"    {' | '.join(parts)}")

            # Utvärdera expect-flaggor (valideras)
            # num_new_hits finns bara på rework-vägen (elaboration/verification)
            num_new_hits = debug.get("num_new_hits")
            # status_counts finns bara när verification körts
            verification_status_counts = synthesis.get("status_counts")

            # Filnamn för dokumentmetrik. source_file_names är de källor
            # som bar svaret; retrieval_file_names är dokumentnivå-
            # rankningen ur rerank_top (ordnad efter score, exkl. chunkar
            # som cross-encodern filtrerat bort, dedupliceras per fil).
            source_file_names = [
                s.metadata.file_name for s in response.sources
            ]
            retrieval_file_names: list[str] = []
            _seen_files: set[str] = set()
            for entry in debug.get("rerank_top", []) or []:
                fn = entry.get("file_name")
                if not fn or entry.get("filtered") or fn in _seen_files:
                    continue
                _seen_files.add(fn)
                retrieval_file_names.append(fn)

            flags = _evaluate_expect(
                expect=expect,
                num_sources=num_sources,
                intent=classification.get("intent", ""),
                qud_drift_detected=qud_drift.get("drift_detected", False),
                abstained=debug.get("abstained", False),
                num_new_hits=num_new_hits,
                verification_status_counts=verification_status_counts,
                answer=response.answer,
                source_file_names=source_file_names,
                retrieval_file_names=retrieval_file_names,
                source_guard=debug.get("source_guard"),
            )
            for flag in flags:
                icon = "✓" if flag["ok"] else "✗"
                typer.echo(f"    {icon} {flag['label']}")
                all_flags.append({"sequence": seq_name, "turn": turn_idx, **flag})
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
                "num_new_hits": num_new_hits,
                "abstained": debug.get("abstained", False),
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

            # JSONL-spåret får HELA debug-blocket (inkl. rerank_top,
            # evidence_top, synonym_additions m.m.) — det är detta som
            # gör två körningar diffbara på retrieval-nivå, inte bara
            # på flaggnivå.
            _write_jsonl({
                "type": "turn",
                "sequence": seq_name,
                "turn": turn_idx,
                "question": question,
                "expect": expect,
                "answer": response.answer,
                "sources": [
                    {
                        "file_name": s.metadata.file_name,
                        "section_title": s.metadata.section_title,
                        "score": round(s.score, 4),
                    }
                    for s in response.sources
                ],
                "flags": flags,
                "debug": debug,
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
    if failed_turns:
        typer.echo(f"Uteblivna svar:     {len(failed_turns)}  ← KÖRNINGEN ÄR OFULLSTÄNDIG")

    if all_times:
        typer.echo(f"Medeltid per tur:   {sum(all_times) / len(all_times):.1f}s")
        typer.echo(f"Min/max tid:        {min(all_times):.1f}s / {max(all_times):.1f}s")

    if all_flags:
        by_field: dict[str, list[bool]] = defaultdict(list)
        for f in all_flags:
            by_field[f["field"]].append(bool(f["ok"]))
        typer.echo("")
        typer.echo("Flaggor per fält (ok/utvärderade)")
        typer.echo("---------------------------------")
        for field_name in sorted(by_field):
            oks = by_field[field_name]
            typer.echo(f"  {field_name}: {sum(oks)}/{len(oks)}")

    if total_flags:
        typer.echo("")
        typer.echo("Avvikelser per sekvens")
        typer.echo("----------------------")
        for seq in sequence_results:
            if seq["failed_flags"]:
                typer.echo(f"  {seq['name']}:")
                for f in seq["failed_flags"]:
                    typer.echo(f"    tur {f['turn']}: {f['label']}")

    if failed_turns:
        typer.echo("")
        typer.echo("Uteblivna svar (turen utvärderades aldrig)")
        typer.echo("-----------------------------------------")
        for ft in failed_turns:
            typer.echo(f"  {ft['sequence']} tur {ft['turn']}: {ft['error']}")
        typer.echo("")
        typer.echo(
            "  Dessa turer saknar svar och är därför inte utvärderade mot sina\n"
            "  expect-fält. Körningen kan INTE användas som baslinje eller\n"
            "  jämförelsepunkt. Vanlig orsak: ollama startades om under\n"
            "  körningen (uppdatering, modellbyte, systemctl restart)."
        )

    # Spara resultat (output_file bestämdes före körningen)
    # run_summary skrivs sist i JSONL: compare_test_runs läser den för
    # att kunna vägra jämföra mot en ofullständig körning.
    _write_jsonl({
        "type": "run_summary",
        "num_turns": sum(len(s["turns"]) for s in sequence_results),
        "num_flag_failures": len(total_flags),
        "num_failed_turns": len(failed_turns),
        "incomplete": bool(failed_turns),
        "failed_turns": failed_turns,
    })
    if jsonl_handle is not None:
        jsonl_handle.close()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_file": str(test_file),
                "server_url": server_url,
                "timestamp": datetime.now().isoformat(),
                "num_sequences": len(sequence_results),
                "num_turns": sum(len(s["turns"]) for s in sequence_results),
                "num_flag_failures": len(total_flags),
                "num_failed_turns": len(failed_turns),
                "incomplete": bool(failed_turns),
                "failed_turns": failed_turns,
                "mean_time_s": round(sum(all_times) / len(all_times), 3) if all_times else None,
                "sequences": sequence_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    typer.echo("")
    typer.echo(f"Resultat sparade: {output_file}")
    if jsonl_file is not None:
        typer.echo(f"Diagnostikspår:   {jsonl_file}")
        typer.echo(
            "Jämför två körningar: "
            "python -m scripts.compare_test_runs <gammal>.jsonl <ny>.jsonl"
        )


def _current_git_commit() -> str | None:
    """Bäst-effort: aktuell git-commit för spårbarhet i run_meta."""
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _normalize_for_contains(text: str) -> str:
    """
    Normalisera text för substrängsmatchning: ta bort all whitespace
    och casefolda. Gör att "10 000" matchar "10 000:-", "10  000" och
    "10000" oavsett hur källan råkar formatera beloppet.
    """
    return re.sub(r"\s+", "", text or "").casefold()


def _evaluate_expect(
    expect: dict,
    num_sources: int,
    intent: str,
    qud_drift_detected: bool,
    abstained: bool,
    num_new_hits: int | None,
    verification_status_counts: dict | None,
    answer: str = "",
    source_file_names: list[str] | None = None,
    retrieval_file_names: list[str] | None = None,
    source_guard: dict | None = None,
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

    source_file_names är filnamnen för de källor som bar svaret.
    retrieval_file_names är dokumentnivå-rankningen ur rerank_top
    (bara retrieval-turer har en; rework-turer skickar tom lista).
    """
    flags: list[dict] = []
    source_file_names = source_file_names or []
    retrieval_file_names = retrieval_file_names or []

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

    if "expected_docs" in expect:
        wanted = [str(x) for x in expect["expected_docs"]]
        missing = [
            w for w in wanted
            if not any(w.casefold() in fn.casefold() for fn in source_file_names)
        ]
        ok = not missing
        if ok:
            detail = "(alla bland svarets källor)"
        else:
            detail = f"(saknas: {missing}; källor: {source_file_names or 'inga'})"
        flags.append({
            "label": f"expected_docs={wanted} {detail}",
            "ok": ok,
            "field": "expected_docs",
            "expected": wanted,
            "actual": source_file_names,
        })

    if "expected_docs_in_retrieval" in expect:
        wanted = [str(x) for x in expect["expected_docs_in_retrieval"]]
        k = int(expect.get("retrieval_top_k", 5))
        top_docs = retrieval_file_names[:k]
        missing = [
            w for w in wanted
            if not any(w.casefold() in fn.casefold() for fn in top_docs)
        ]
        ok = not missing
        if not retrieval_file_names:
            detail = "(ingen retrieval-rankning på denna tur)"
        elif ok:
            detail = f"(hit@{k} på dokumentnivå)"
        else:
            detail = f"(saknas i topp-{k}: {missing}; topp: {top_docs})"
        flags.append({
            "label": f"expected_docs_in_retrieval={wanted} {detail}",
            "ok": ok,
            "field": "expected_docs_in_retrieval",
            "expected": wanted,
            "actual": top_docs,
        })

    if "answer_numbers_must_be_sourced" in expect:
        want = bool(expect["answer_numbers_must_be_sourced"])
        if source_guard is None:
            ok = False
            detail = "(ingen källvakt kördes på denna tur — fältet gäller bara huvudsyntesvägen)"
            actual: object = None
        else:
            unsourced = source_guard.get("unsourced_numbers") or []
            got = not unsourced
            ok = (got == want)
            checked = source_guard.get("numbers_checked") or []
            if got:
                detail = f"(alla {len(checked)} kontrollerade tal belagda)"
            else:
                detail = f"(obelagda tal: {unsourced})"
            actual = unsourced
        flags.append({
            "label": f"answer_numbers_must_be_sourced={want} {detail}",
            "ok": ok,
            "field": "answer_numbers_must_be_sourced",
            "expected": want,
            "actual": actual,
        })

    if "answer_must_not_contain" in expect:
        needles = [str(x) for x in expect["answer_must_not_contain"]]
        haystack = _normalize_for_contains(answer)
        present = [
            n for n in needles
            if _normalize_for_contains(n) in haystack
        ]
        ok = not present
        detail = "(inget förbjudet i svaret)" if ok else f"(förekommer i svaret: {present})"
        flags.append({
            "label": f"answer_must_not_contain={needles} {detail}",
            "ok": ok,
            "field": "answer_must_not_contain",
            "expected": needles,
            "actual": present,
        })

    if "answer_must_contain" in expect:
        needles = [str(x) for x in expect["answer_must_contain"]]
        haystack = _normalize_for_contains(answer)
        missing = [
            n for n in needles
            if _normalize_for_contains(n) not in haystack
        ]
        ok = not missing
        detail = "(allt finns i svaret)" if ok else f"(saknas i svaret: {missing})"
        flags.append({
            "label": f"answer_must_contain={needles} {detail}",
            "ok": ok,
            "field": "answer_must_contain",
            "expected": needles,
            "actual": missing,
        })

    return flags


def main() -> None:
    """
    Entry point.

    StorageLockedError fångas här i stället för i varje kommando: den
    kan uppstå i allt som öppnar den inbäddade lagringen, och
    meddelandet är detsamma oavsett vilket kommando som råkade ut för
    det. Ett förutsägbart fel med självklar åtgärd ska mötas med ett
    meddelande, inte med ett traceback på hundra rader där orsaken
    står sist.
    """
    from app.qdrant_store import StorageLockedError

    try:
        app()
    except StorageLockedError as e:
        typer.echo("")
        typer.echo(str(e))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    main()
