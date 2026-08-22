import logging
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
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

# Sessioner från inloggning med lösenord, och strypningen som skyddar
# den vägen. Båda lever i minnet och försvinner vid omstart. För
# sessioner är det avsiktligt — en glömd inloggning ska inte överleva
# att maskinen startas om. För strypningen är det en känd svaghet, men
# en omstart kräver åtkomst till maskinen, och då är spärren inte det
# som skyddar.
_sessions = auth.SessionStore(
    ttl_seconds=settings.session_ttl_seconds,
    idle_seconds=settings.session_idle_seconds,
)
_throttle = auth.Throttle()

if settings.auth_enabled:
    logger.info(
        "auth: PÅ | %d användare ur %s",
        _users.loaded,
        settings.users_path.resolve(),
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
                "auth: läste om %s — %d användare",
                store.path.resolve() if store.path else "(ingen fil)",
                store.loaded,
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

    token = auth.bearer_token(authorization)
    # Kontotoken först, sessionstoken därefter. Båda bärs i samma
    # Authorization-huvud, vilket är avsiktligt: protokollet ändras
    # inte av att det finns två sätt att skaffa ett bevis. Maskiner
    # (urd test, skript) har långlivade kontotokens; människor får en
    # kortlivad session genom /login.
    principal = _current_users().verify(token) or _sessions.verify(token)
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


def _require_confidential(request) -> None:
    """
    Vägra lösenord över en förbindelse som kan avlyssnas.

    Samma mönster som 'urd serve' redan använder när den vägrar binda
    utanför loopback utan autentisering: villkoret upprätthålls av
    koden, inte av att någon minns det på driftsättningsdagen.

    Skälet att vara strängare här än för tokens är att lösenord
    återanvänds. En läckt token drabbar URD; ett läckt lösenord följer
    ofta användaren till andra system.

    Tre vägar är godtagbara: förbindelsen är https, den kommer från
    loopback (klienten och servern är samma maskin, ingen tråd att
    avlyssna), eller så säger driften uttryckligen att TLS avslutas
    uppströms. X-Forwarded-Proto godtas INTE — ett huvud från en okänd
    mellanhand är inget bevis för att förbindelsen var krypterad.
    """
    if settings.tls_terminated_upstream:
        return
    if request.url.scheme == "https":
        return
    värd = request.client.host if request.client else ""
    if auth.is_loopback(värd):
        return
    raise HTTPException(
        status_code=421,
        detail=(
            "Lösenordsinloggning kräver TLS. Anslut över https, eller sätt "
            "tls_terminated_upstream när TLS avslutas i en betrodd proxy."
        ),
    )


@app.post("/login")
def login(payload: dict, request: Request) -> dict:
    """
    Byt namn och lösenord mot en kortlivad sessionstoken.

    Samma svar oavsett om namnet är okänt eller lösenordet fel, och
    verify_password kostar lika mycket i båda fallen — annars avslöjar
    svaret vilka konton som finns, och en lista över anställda med
    konton i systemet är i sig en uppgift värd att skydda.
    """
    if not settings.auth_enabled:
        raise HTTPException(
            status_code=400,
            detail="Autentisering är avstängd på den här servern.",
        )
    _require_confidential(request)

    name = str(payload.get("name") or "").strip()
    password = str(payload.get("password") or "")
    if not name or not password:
        raise HTTPException(status_code=400, detail="Namn och lösenord krävs.")

    # Nyckeln bär både namn och avsändare, så att en angripare varken
    # kan låsa ute en enskild användare genom att gissa i hens namn
    # eller kringgå spärren genom att byta namn.
    värd = request.client.host if request.client else "okänd"
    nyckel = f"{name.lower()}|{värd}"
    kvar = _throttle.locked_for(nyckel)
    if kvar > 0:
        logger.warning("auth: inloggning spärrad för %s (%.0f s kvar)", värd, kvar)
        raise HTTPException(
            status_code=429,
            detail=f"För många misslyckade försök. Försök igen om {int(kvar) + 1} s.",
        )

    principal = _current_users().verify_password(
        name, password, throttle=_throttle, key=nyckel
    )
    if principal is None:
        logger.warning("auth: misslyckad inloggning från %s", värd)
        raise HTTPException(status_code=401, detail="Fel namn eller lösenord.")

    token, ttl = _sessions.create(principal)
    _sessions.prune()
    logger.info(
        "auth: %s loggade in från %s (%d aktiva sessioner)",
        principal.name, värd, _sessions.active,
    )
    return {
        "token": token,
        "expires_in": int(ttl),
        "principal": {"name": principal.name, "groups": list(principal.groups)},
    }


@app.post("/logout")
def logout(
    authorization: str | None = Header(default=None),
    principal: auth.Principal = Depends(require_principal),
) -> dict:
    """
    Avsluta den session anropet bärs av.

    En kontotoken kan inte avslutas här — den är ett konto och inte en
    session, och tas bort med 'urd auth remove'. Svaret säger vilket
    som hände så att klienten inte tror sig ha loggat ut när den inte
    har det.
    """
    borttagen = _sessions.revoke(auth.bearer_token(authorization))
    if borttagen:
        logger.info("auth: %s loggade ut", principal.name)
    return {"logged_out": borttagen, "principal": principal.name}


@app.get("/whoami")
def whoami(
    principal: auth.Principal = Depends(require_principal),
) -> dict:
    """
    Vem servern anser att anroparen är.

    Billig, utan sidoeffekter, och tre saker på en gång: klienten kan
    pröva en uppgift innan den sparas, användaren kan se sitt namn i
    gränssnittet, och när behörighetsfiltret införs blir den svaret på
    "vad får jag se" — grupperna avgör det.
    """
    return {
        "name": principal.name,
        "groups": list(principal.groups),
        "unrestricted": principal.unrestricted,
    }


@app.post("/enroll")
def enroll(payload: dict, request: Request) -> dict:
    """
    Växla in en inbjudan mot ett lösenord.

    Token identifierar posten — den ÄR identiteten, och något namn
    behöver därför inte anges. Efter inväxlingen tas token bort ur
    posten: en inbjudan gäller en gång.

    Samma strypning och samma TLS-krav som /login. Strypningen är här
    ännu viktigare, eftersom en gissad inbjudan ger rätten att SÄTTA
    ett lösenord och inte bara att pröva ett.
    """
    if not settings.auth_enabled:
        raise HTTPException(
            status_code=400,
            detail="Autentisering är avstängd på den här servern.",
        )
    _require_confidential(request)

    token = str(payload.get("token") or "").strip()
    password = str(payload.get("password") or "")
    if not token or not password:
        raise HTTPException(status_code=400, detail="Token och lösenord krävs.")

    värd = request.client.host if request.client else "okänd"
    nyckel = f"enroll|{värd}"
    kvar = _throttle.locked_for(nyckel)
    if kvar > 0:
        raise HTTPException(
            status_code=429,
            detail=f"För många misslyckade försök. Försök igen om {int(kvar) + 1} s.",
        )

    record = _current_users().enrollment_for(token)
    if record is None:
        _throttle.record_failure(nyckel)
        logger.warning("auth: ogiltig inbjudan från %s", värd)
        raise HTTPException(status_code=401, detail="Ogiltig inbjudan.")

    name = record.principal.name
    problem = auth.password_problems(password, name=name)
    if problem:
        # INTE ett misslyckat försök: token var giltig, lösenordet dög
        # inte. Att räkna det som gissning skulle låta en användare med
        # svagt lösenordsval spärra sig själv.
        raise HTTPException(status_code=400, detail=" · ".join(problem))

    if not auth.set_password(
        settings.users_path, name, password, consume_enrollment=True
    ):
        raise HTTPException(status_code=409, detail="Användaren finns inte längre.")

    _throttle.record_success(nyckel)
    logger.info("auth: %s växlade in sin inbjudan från %s", name, värd)

    principal = record.principal
    token_ny, ttl = _sessions.create(principal)
    return {
        "token": token_ny,
        "expires_in": int(ttl),
        "principal": {"name": principal.name, "groups": list(principal.groups)},
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
