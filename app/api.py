import logging
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.config_validation import validate_config_files, format_report_lines
from app.schemas import ChatRequest, ChatResponse
from app.retrieval import RagService
from app import auth

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
    "modell: %s | think=%s | num_ctx=%d",
    settings.ollama_model,
    settings.llm_think,
    settings.llm_num_ctx,
)

# AUTENTISERING.
#
# Användarna läses vid uppstart och OM när filen ändrats. Ett stat-
# anrop per skyddad förfrågan avgör saken; innehållet läses bara när
# ändringstid eller storlek skiljer sig.
#
# Skälet är inte bekvämlighet utan återkallelse. En användare som
# tagits bort med 'urd auth remove' ska förlora åtkomsten då, inte vid
# nästa omstart — allt annat gör kommandots besked osant. En trasig
# eller borttagen fil ger tom uppsättning och därmed avslag för alla,
# aldrig en fortsättning på den gamla; se auth.reload_if_changed.
_users = auth.load_users(settings.users_path)
_users_lock = threading.Lock()
for _fel in _users.errors:
    logger.error("users: %s", _fel)

if settings.auth_enabled:
    logger.info(
        "auth: PÅ | %d användare ur %s", _users.loaded, settings.users_path
    )
    if not _users.loaded:
        logger.error(
            "auth: PÅSLAGEN MEN INGA ANVÄNDARE — varje anrop kommer att "
            "avvisas. Lägg upp någon med 'urd auth add <namn>'."
        )
else:
    logger.warning(
        "auth: AV — varje klient som når porten kan läsa hela "
        "dokumentbeståndet. Ofarligt vid bindning till loopback."
    )


def _current_users() -> auth.UserStore:
    """
    Användaruppsättningen, omläst om filen ändrats sedan sist.

    Låset hindrar att flera samtidiga förfrågningar läser om samma
    ändring; tilldelningen i sig är atomär i CPython, så en läsare
    utan låset ser antingen den gamla eller den nya uppsättningen,
    aldrig något halvfärdigt.
    """
    global _users
    with _users_lock:
        store, ändrad = auth.reload_if_changed(_users)
        if ändrad:
            _users = store
            for _fel in store.errors:
                logger.error("users: %s", _fel)
            logger.info(
                "auth: läste om %s — %d användare", store.path, store.loaded
            )
            if not store.loaded:
                logger.error(
                    "auth: användarfilen ger INGA användare — varje anrop "
                    "avvisas tills den är rättad."
                )
    return _users


def require_principal(
    authorization: str | None = Header(default=None),
) -> auth.Principal:
    """
    Identifiera den som frågar.

    Skyddar allt utom /health, som medvetet är öppen: klienten måste
    kunna se om servern lever och vilket protokoll den talar innan den
    har något att autentisera sig med. Svaret där innehåller ingen
    uppgift ur dokumentbeståndet.

    Med autentisering avstängd är principalen LOCAL, som är
    oavgränsad. Det är rätt för en enanvändarmaskin bunden till
    loopback, och servern vägrar starta i det läget med en bindning
    som når nätverket.
    """
    if not settings.auth_enabled:
        return auth.LOCAL

    principal = _current_users().verify(auth.bearer_token(authorization))
    if principal is None:
        # Samma svar oavsett om token saknas eller är okänd: att skilja
        # dem åt berättar för en angripare vilken av gissningarna som
        # var nära.
        raise HTTPException(status_code=401, detail="Ogiltig eller saknad token.")
    return principal


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
        # Klienten behöver veta om den måste autentisera sig innan den
        # försöker. Uppgiften avslöjar inget om beståndet.
        "auth_required": settings.auth_enabled,
        "config_files": _config_report.as_dict(),
        # Klienten (och urd connect) ska kunna se vad servern faktiskt
        # kör utan att läsa serverloggen.
        "llm": {
            "model": settings.ollama_model,
            "think": settings.llm_think,
            "num_ctx": settings.llm_num_ctx,
        },
    }


@app.post("/refresh")
def refresh(
    principal: auth.Principal = Depends(require_principal),
) -> dict:
    """
    Återbygg BM25-index efter ingest. Anropas av CLI.

    RETURTYPEN VAR FEL och FastAPI validerar mot den: annoteringen sade
    dict[str, int] medan svaret bär "status": "ok". Endpointen svarade
    därför 500 i drift, och CLI:t skrev "Varning: kunde inte uppdatera
    serverns sökindex" — ett meddelande som ser ut som ett
    nätverksproblem. Följden var att serverns BM25-index behöll gamla
    chunkar efter varje ingest tills servern startades om.

    Upptäckt 2026-08-18 när endpointen för första gången anropades i
    test. Den hade aldrig prövats med en körande server: felet syns
    inte i importkontroll, inte i py_compile, och CLI:t sväljer det.
    """
    logger.info("refresh | principal=%s", principal.name)
    num_chunks = rag.refresh_index()
    return {"status": "ok", "num_chunks": num_chunks}


@app.get("/document")
def get_document(
    path: str = Query(..., description="Relativ sökväg under docs/"),
    principal: auth.Principal = Depends(require_principal),
):
    """
    Servera ett originaldokument. Validerar att sökvägen pekar
    in i docs-katalogen för att förhindra path traversal.
    """
    resolved = (_docs_root / path).resolve()

    if not resolved.is_relative_to(_docs_root):
        raise HTTPException(status_code=404, detail="Dokumentet hittades inte.")

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Dokumentet hittades inte.")

    logger.info("document | principal=%s | %s", principal.name, resolved.name)
    return FileResponse(resolved, filename=resolved.name)


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    principal: auth.Principal = Depends(require_principal),
) -> ChatResponse:
    """
    HTTP-lager över RagService.converse.

    Samtalslogiken — QUD, drift, rework, sessionstillstånd — flyttades
    till kärnan 2026-08-16. Den hör hemma där enligt white paper och
    ska vara tillgänglig för varje klient, inte bara för den som talar
    HTTP. Den här funktionen mappar request till anrop och undantag
    till statuskod; inget mer.
    """
    # ANSLUTNINGSLOGG. Vem och när, inte vad. Frågeinnehållet kan
    # avslöja vad en administratör utreder, och att logga det är en
    # integritetsavvägning som ska beslutas tillsammans med
    # verksamheten — inte uppstå som en bieffekt av felsökning.
    logger.info(
        "chat | principal=%s | session=%s | längd=%d",
        principal.name,
        (req.session_id or "ny")[:8],
        len(req.question),
    )
    try:
        return rag.converse(req.question, session_id=req.session_id)
    except Exception:
        import traceback
        tb = traceback.format_exc()
        logger.error("chat misslyckades:\n%s", tb)
        raise HTTPException(status_code=500, detail=tb)
