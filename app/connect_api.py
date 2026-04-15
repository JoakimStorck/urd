from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import unquote

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("urd.connect")

app = FastAPI(title="Local IIT URD Client")

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def _get_upstream_base_url() -> str:
    raw = (os.getenv("URD_UPSTREAM_SERVER") or "").strip()
    if not raw:
        raise RuntimeError(
            "Ingen upstream-server angiven. Sätt URD_UPSTREAM_SERVER eller starta via 'urd connect --server ...'."
        )
    if "://" not in raw:
        raw = "http://" + raw
    return raw.rstrip("/")


def _proxy_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
    }


def _short_document_label(path: str) -> str:
    decoded = unquote(path)
    p = Path(decoded)
    if len(p.parts) >= 2:
        return str(Path(*p.parts[-2:]))
    return decoded


@app.on_event("startup")
def on_startup() -> None:
    upstream = _get_upstream_base_url()
    logger.info("URD connect startad")
    logger.info("Upstream-server: %s", upstream)


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health() -> dict:
    upstream = _get_upstream_base_url()
    try:
        resp = requests.get(f"{upstream}/health", timeout=5)
        upstream_ok = resp.ok
        upstream_payload = resp.json() if resp.ok else {"status": "error"}
    except Exception as e:
        upstream_ok = False
        upstream_payload = {"status": "error", "detail": f"{type(e).__name__}: {e}"}

    return {
        "status": "ok",
        "mode": "client",
        "upstream": upstream,
        "upstream_ok": upstream_ok,
        "upstream_health": upstream_payload,
    }


@app.post("/chat")
def chat(req: Request, req_body: dict) -> Response:
    upstream = _get_upstream_base_url()
    client_host = req.client.host if req.client else "unknown"

    try:
        resp = requests.post(
            f"{upstream}/chat",
            json=req_body,
            headers=_proxy_headers(),
            timeout=300,
        )
    except requests.RequestException as e:
        logger.warning(
            "POST /chat misslyckades | client=%s | upstream=%s | fel=%s: %s",
            client_host,
            upstream,
            type(e).__name__,
            e,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Kunde inte nå URD-servern på {upstream}: {type(e).__name__}: {e}",
        ) from e

    logger.info(
        "POST /chat -> %s | client=%s",
        resp.status_code,
        client_host,
    )

    content_type = resp.headers.get("content-type", "application/json")
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=content_type.split(";")[0],
        headers={"content-type": content_type},
    )


@app.get("/document")
def get_document(
    request: Request,
    path: str = Query(..., description="Relativ sökväg under docs/"),
):
    upstream = _get_upstream_base_url()
    client_host = request.client.host if request.client else "unknown"
    doc_label = _short_document_label(path)

    try:
        resp = requests.get(
            f"{upstream}/document",
            params={"path": path},
            timeout=300,
            stream=True,
        )
    except requests.RequestException as e:
        logger.warning(
            "GET /document misslyckades | client=%s | dokument=%s | upstream=%s | fel=%s: %s",
            client_host,
            doc_label,
            upstream,
            type(e).__name__,
            e,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Kunde inte nå URD-servern på {upstream}: {type(e).__name__}: {e}",
        ) from e

    logger.info(
        "GET /document -> %s | client=%s | dokument=%s",
        resp.status_code,
        client_host,
        doc_label,
    )

    content = resp.content
    headers = {}
    for key in ("content-type", "content-disposition", "content-length", "etag", "last-modified"):
        value = resp.headers.get(key)
        if value:
            headers[key] = value

    return Response(
        content=content,
        status_code=resp.status_code,
        headers=headers,
        media_type=resp.headers.get("content-type", "application/octet-stream").split(";")[0],
    )
