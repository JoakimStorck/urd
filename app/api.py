import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.config_validation import validate_config_files, format_report_lines
from app.schemas import ChatRequest, ChatResponse
from app.retrieval import RagService

logger = logging.getLogger(__name__)

# Uvicorn konfigurerar bara sina egna loggers. Appens loggers (den här
# modulens, loadernas i synonyms/concepts/question_operations, llm:s
# trunkeringsvarning) hamnar på rotloggern som saknar handler — och
# försvinner då tyst. Det underminerar hela poängen med synlig
# konfigstatus. Konfigurera därför rotloggern här, men bara om ingen
# handler redan finns (så att en inbäddande process kan styra själv).
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:     %(message)s",
    )

# Tredjepartsbibliotek som loggar varje HTTP-anrop på INFO-nivå
# dränker appens egna rader. Dämpa dem till WARNING — deras fel
# ska fortfarande synas, men inte deras vardagsprat.
for _noisy in ("httpx", "huggingface_hub", "urllib3", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

app = FastAPI(title="Local IIT URD")
rag = RagService()
# Sessionerna ägs numera av RagService (rag.sessions). Modulnivåvariabeln
# är borttagen — två sessionsregister mot samma tjänst vore en tyst
# felkälla där webbgränssnittet och andra klienter fick olika minne.

# Validera instansens konfigurationsfiler vid uppstart. Trasiga filer
# stoppar inte servern (tyst fallback i loaders gäller fortfarande),
# men status ska alltid vara synlig: en rad per fil i startloggen och
# samma bild i /health. Se app/config_validation.py för bakgrund.
_config_report = validate_config_files(
    synonyms_path=settings.synonyms_path,
    concepts_path=settings.concepts_path,
    question_operations_path=settings.question_operations_path,
)
for _line in format_report_lines(_config_report):
    if _line.startswith("[FEL]") or _line.strip().startswith("fel:"):
        logger.warning("konfig: %s", _line)
    else:
        logger.info("konfig: %s", _line)
if not _config_report.ok:
    logger.warning(
        "konfig: en eller flera konfigurationsfiler har FEL — kör "
        "'urd config validate' för detaljer. Berörda funktioner är avstängda."
    )

# Aktiv modell och LLM-inställningar i startloggen.
#
# Modellen sätts i .urd/config.json men kan överstyras av OLLAMA_MODEL,
# och think/num_ctx avgör både svarskvalitet och svarstid. Utan den här
# raden går det inte att se vad som faktiskt kör: 2026-08-14 kostade det
# en hel testkörning att upptäcka att gemma4:12b hade resonemang påslaget
# (107,6 s median per tur mot 9,0 s med det avstängt), och en körning
# till att reda ut om en modellväxling ens hade slagit igenom.
logger.info(
    "modell: %s | think=%s | num_ctx=%d | enrich-modell=%s",
    settings.ollama_model,
    settings.llm_think,
    settings.llm_num_ctx,
    settings.preprocess_ollama_model,
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Resolve docs root en gång vid uppstart
_docs_root = settings.docs_path.resolve()



@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "config_files": _config_report.as_dict(),
        # Klienten (och urd connect) ska kunna se vad servern faktiskt
        # kör utan att läsa serverloggen.
        "llm": {
            "model": settings.ollama_model,
            "think": settings.llm_think,
            "num_ctx": settings.llm_num_ctx,
            "enrich_model": settings.preprocess_ollama_model,
        },
    }


@app.post("/refresh")
def refresh() -> dict[str, int]:
    """Återbygg BM25-index efter ingest. Anropas av CLI."""
    num_chunks = rag.refresh_index()
    return {"status": "ok", "num_chunks": num_chunks}


@app.get("/document")
def get_document(path: str = Query(..., description="Relativ sökväg under docs/")):
    """
    Servera ett originaldokument. Validerar att sökvägen pekar
    in i docs-katalogen för att förhindra path traversal.
    """
    resolved = (_docs_root / path).resolve()

    if not resolved.is_relative_to(_docs_root):
        raise HTTPException(status_code=404, detail="Dokumentet hittades inte.")

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Dokumentet hittades inte.")

    return FileResponse(resolved, filename=resolved.name)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    HTTP-lager över RagService.converse.

    Samtalslogiken — QUD, drift, rework, sessionstillstånd — flyttades
    till kärnan 2026-08-16. Den hör hemma där enligt white paper och
    ska vara tillgänglig för varje klient, inte bara för den som talar
    HTTP. Den här funktionen mappar request till anrop och undantag
    till statuskod; inget mer.
    """
    try:
        return rag.converse(req.question, session_id=req.session_id)
    except Exception:
        import traceback
        tb = traceback.format_exc()
        logger.error("chat misslyckades:\n%s", tb)
        raise HTTPException(status_code=500, detail=tb)
