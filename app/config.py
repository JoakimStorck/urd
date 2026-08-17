from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import logging
import os

logger = logging.getLogger(__name__)

load_dotenv()

URD_DIR = Path(".urd")
CONFIG_FILE = URD_DIR / "config.json"

# Hårdkodade defaults — dessa skrivs till .urd/config.json om filen saknas
DEFAULTS = {
    "docs_path": "./docs",
    "qdrant_path": "./data/qdrant",
    "question_operations_path": ".urd/question_operations.yaml",
    "collection_name": "iit_docs",
    "embedding_model": "intfloat/multilingual-e5-large",
    "reranker_model": "jeffwan/mmarco-mMiniLMv2-L12-H384-v1",
    "ollama_model": "gemma4:12b",
    "llm_num_ctx": 8192,
    "llm_think": False,
    "predication_enabled": False,
    "attest_selection": False,
    "attest_boost": 0.15,
    "chunk_size": 1200,
    "chunk_overlap": 150,
    "server": "",
    "qud_background_turns": 1,
    "social_history_turns": 4,
    "classification_history_turns": 2,
    "qud_drift_threshold": 0.80,
    "qud_drift_doc_threshold": 0.80,
    "max_hits": 10,
    "min_desired_hits": 3,
    # Sannolikhetsskala (sigmoid på cross-encoderns logits) — se
    # kommentarer i Settings. Gamla logit-skalade nycklar
    # (expansion_score_threshold, expanded_filter_floor,
    # min_relevance_floor, relevance_ratio, evidence_*_boost) är
    # avsiktligt BORTTAGNA: kvarvarande värden i äldre config.json
    # ignoreras, så att gamla logit-tal inte tolkas som
    # sannolikheter.
    "select_min_prob": 0.5,
    "expansion_min_prob": 0.55,
    "expanded_min_prob": 0.27,
    "evidence_section_prob_boost": 0.15,
    "evidence_document_prob_boost": 0.05,
}

# Mapping: config-nyckel → miljövariabel
_ENV_KEYS = {
    "docs_path": "DOCS_PATH",
    "qdrant_path": "QDRANT_PATH",
    "question_operations_path": "QUESTION_OPERATIONS_PATH",
    "collection_name": "QDRANT_COLLECTION",
    "embedding_model": "EMBEDDING_MODEL",
    "reranker_model": "RERANKER_MODEL",
    "ollama_model": "OLLAMA_MODEL",
    "llm_num_ctx": "LLM_NUM_CTX",
    "llm_think": "LLM_THINK",
    "predication_enabled": "PREDICATION_ENABLED",
    "attest_selection": "ATTEST_SELECTION",
    "attest_boost": "ATTEST_BOOST",
    "chunk_size": "CHUNK_SIZE",
    "chunk_overlap": "CHUNK_OVERLAP",
    "server": "URD_SERVER",
    "qud_background_turns": "QUD_BACKGROUND_TURNS",
    "social_history_turns": "SOCIAL_HISTORY_TURNS",
    "classification_history_turns": "CLASSIFICATION_HISTORY_TURNS",
    "qud_drift_threshold": "QUD_DRIFT_THRESHOLD",
    "qud_drift_doc_threshold": "QUD_DRIFT_DOC_THRESHOLD",
    "max_hits": "MAX_HITS",
    "min_desired_hits": "MIN_DESIRED_HITS",
    "select_min_prob": "SELECT_MIN_PROB",
    "expansion_min_prob": "EXPANSION_MIN_PROB",
    "expanded_min_prob": "EXPANDED_MIN_PROB",
    "evidence_section_prob_boost": "EVIDENCE_SECTION_PROB_BOOST",
    "evidence_document_prob_boost": "EVIDENCE_DOCUMENT_PROB_BOOST",
}


def _load_file_config() -> dict:
    """Läs .urd/config.json om den finns."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _ensure_config_file() -> None:
    """Skapa .urd/config.json med defaults om den inte finns."""
    if not CONFIG_FILE.exists():
        URD_DIR.mkdir(parents=True, exist_ok=True)
        save_config_file(dict(DEFAULTS))


def save_config_file(data: dict) -> None:
    """Skriv config till .urd/config.json."""
    URD_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _resolve_value(key: str, file_config: dict) -> str | int | float:
    """
    Resolva ett config-värde med prioritet:
    1. Miljövariabel
    2. .urd/config.json
    3. Hårdkodad default
    """
    env_key = _ENV_KEYS.get(key)
    env_val = os.getenv(env_key) if env_key else None

    if env_val is not None:
        return env_val

    if key in file_config:
        return file_config[key]

    return DEFAULTS[key]


_BOOL_TRUE = {"1", "true", "yes", "ja", "on", "sant"}
_BOOL_FALSE = {"0", "false", "no", "nej", "off", "falskt"}


def parse_bool(value: str) -> bool:
    """
    Tolka en boolesk konfigsträng. Kastar ValueError vid okänt värde.

    Delas av miljövariabeltolkningen och 'urd config set' så att de två
    vägarna aldrig kan hamna i otakt om vad som räknas som sant.
    """
    v = value.strip().lower()
    if v in _BOOL_TRUE:
        return True
    if v in _BOOL_FALSE:
        return False
    raise ValueError(f"okänt booleskt värde: {value!r}")


def _build_settings() -> "Settings":
    """Bygg Settings med rätt prioritetsordning."""
    _ensure_config_file()
    file_config = _load_file_config()

    def s(key: str) -> str:
        return str(_resolve_value(key, file_config))

    def i(key: str) -> int:
        return int(_resolve_value(key, file_config))

    def f(key: str) -> float:
        return float(_resolve_value(key, file_config))

    def b(key: str) -> bool:
        raw = _resolve_value(key, file_config)
        if isinstance(raw, bool):
            return raw
        try:
            return parse_bool(str(raw))
        except ValueError:
            # Tyst fallback till False vore samma sorts osynliga fel som
            # den trasiga synonyms.yaml: en felstavad miljövariabel skulle
            # stänga av en funktion utan att någon märkte det.
            logger.warning(
                "config: %s har ogiltigt booleskt värde %r — använder %r.",
                key, raw, DEFAULTS[key],
            )
            return bool(DEFAULTS[key])

    server = s("server").strip() or None

    return Settings(
        docs_path=Path(s("docs_path")),
        qdrant_path=Path(s("qdrant_path")),
        question_operations_path=Path(s("question_operations_path")),
        collection_name=s("collection_name"),
        embedding_model=s("embedding_model"),
        reranker_model=s("reranker_model"),
        ollama_model=s("ollama_model"),
        llm_num_ctx=i("llm_num_ctx"),
        llm_think=b("llm_think"),
        predication_enabled=b("predication_enabled"),
        attest_selection=b("attest_selection"),
        attest_boost=f("attest_boost"),
        chunk_size=i("chunk_size"),
        chunk_overlap=i("chunk_overlap"),
        server=server,
        qud_background_turns=i("qud_background_turns"),
        social_history_turns=i("social_history_turns"),
        classification_history_turns=i("classification_history_turns"),
        qud_drift_threshold=f("qud_drift_threshold"),
        qud_drift_doc_threshold=f("qud_drift_doc_threshold"),
        max_hits=i("max_hits"),
        min_desired_hits=i("min_desired_hits"),
        select_min_prob=f("select_min_prob"),
        expansion_min_prob=f("expansion_min_prob"),
        expanded_min_prob=f("expanded_min_prob"),
        evidence_section_prob_boost=f("evidence_section_prob_boost"),
        evidence_document_prob_boost=f("evidence_document_prob_boost"),
    )


class Settings(BaseModel):
    docs_path: Path = Path("./docs")
    qdrant_path: Path = Path("./data/qdrant")
    synonyms_path: Path = Path(".urd/synonyms.yaml")
    concepts_path: Path = Path(".urd/concepts.yaml")
    question_operations_path: Path = Path(".urd/question_operations.yaml")

    collection_name: str = "iit_docs"

    embedding_model: str = "intfloat/multilingual-e5-large"
    reranker_model: str = "jeffwan/mmarco-mMiniLMv2-L12-H384-v1"

    # Syntesmodell. gemma4:12b sedan 2026-08-17, efter att
    # modellutvärderingen visat ett tydligt lyft mot mistral-nemo.
    # Nemos språkbrus var dokumenterat och låg utanför vad prompt och
    # vakter kan åtgärda: inverterad delegationsordning, trasslig
    # negation, uppfunna ord i genererad text. Vakterna prövar tal och
    # referenser, inte formuleringar — den felklassen kunde bara mötas
    # med en bättre modell.
    #
    # Defaulten styr bara NYA instanser: en befintlig .urd/config.json
    # har företräde, och miljövariabeln OLLAMA_MODEL över båda.
    # llm_think måste vara av (se nedan).
    ollama_model: str = "gemma4:12b"

    # Kontextfönster för Ollama-anropen. Ollama har ett lågt default
    # (2048–4096 tokens beroende på version/modellfil) och TRUNKERAR
    # TYST prompt som inte får plats — vilket ger exakt de symptom
    # som annars ser ut som modellfel: fabrikation, ignorerade
    # instruktioner, svar som bara bygger på delar av källorna.
    # Därför sätts num_ctx alltid explicit. 8192 rymmer huvudsyntesens
    # och rework-vägarnas prompter med god marginal.
    llm_num_ctx: int = 8192

    # Resonemangsläge ("thinking") i modeller som stödjer det.
    # AV som default. Uppmätt 2026-08-14 med gemma4:12b: påslaget
    # resonemang gav 107,6 s median per tur mot Nemos 3,2 s — och
    # KORTARE svar, eftersom tankespåret kostar generering men
    # returneras i ett eget fält som URD inte läser. Effekten på
    # kvaliteten var förödande: answer_must_contain föll 0/6, alla
    # belopp försvann ur arvodessvaret. För källbunden syntes ur
    # given text tillför resonemang ingenting; uppgiften är att
    # återge, inte att härleda.
    llm_think: bool = False

    # Predikationslagret (app/grammar.py + app/predication.py).
    # AV som default: i nuvarande version är det ett skuggläge som
    # bara skriver till debugspåret. Kräver att stanza och den
    # svenska modellen är installerade; saknas de loggas det en
    # gång och lagret förblir inaktivt.
    predication_enabled: bool = False

    # Attestsignal i urvalet för entitetsfrågor ("vem är X").
    # AV som default: signalen ÄNDRAR svaren och måste mätas mot
    # baslinjen. I skuggläge ändrar den ingenting och blir omätbar,
    # så här är en configflagga rätt mekanism i stället.
    # Kräver att .urd/attest.db är byggd med 'urd attest-build'.
    attest_selection: bool = False

    # Påslagets storlek i sannolikhetsskala, viktat med kandidatens
    # relevans. 0.15 betyder att en bindning med full relevans lyfter
    # en träff 0.15 — nog för att passera golvet 0.50 från 0.36, men
    # aldrig nog att kasta om en klart bättre chunk. Provisoriskt
    # värde; kalibreras mot batteriet.
    attest_boost: float = 0.15

    chunk_size: int = 1200
    chunk_overlap: int = 150

    server: str | None = None

    # Samtalskontext — hur mycket historik som skickas med i olika steg.
    # Varje värde räknas i "turer" där en tur = ett fråga-svar-par.
    # qud_background_turns används för related_to_qud och
    # verification_or_challenge, där föregående turer ges som bakgrund
    # i evidensextraktionen.
    qud_background_turns: int = 1
    social_history_turns: int = 4
    classification_history_turns: int = 2

    # -------------------------------------------------------------
    # Relevansscore i SANNOLIKHETSSKALA.
    #
    # Cross-encoderns råa logits normaliseras genom sigmoid till
    # (0, 1) direkt i rerankern. Alla trösklar och boostar nedan är
    # därmed tolkningsbara sannolikheter: 0.5 = "mer sannolikt
    # relevant än inte", 0.27 = sigmoid(-1) ≈ "osäker men inte
    # avfärdad". Tidigare blandades obundna logits, additiva boostar
    # (+3.0 på en ±5-skala) och kvotregler som är meningslösa på
    # logits — med följden att svaga träffar kunde tränga undan
    # klart bättre, och att syntesen ströps till 1 källa på
    # godtyckliga grunder.
    # -------------------------------------------------------------

    # Lägsta relevanssannolikhet för att en chunk ska väljas till
    # svarsunderlaget (pass 1 i urvalet).
    select_min_prob: float = 0.5

    # expansion_min_prob: minsta sannolikhet som krävs för att ett
    #   dokument ska expanderas (motsvarar gamla logit-tröskeln 0.2).
    # expanded_min_prob: lägsta sannolikhet som tillåts för chunkar
    #   från expanderade dokument — lägre än select_min_prob eftersom
    #   dokumentet som helhet redan visat sig relevant (motsvarar
    #   gamla logit-golvet -1.0).
    expansion_min_prob: float = 0.55
    expanded_min_prob: float = 0.27

    # QUD-drift-skydd. Om embedding-likhet mellan aktuell fråga och
    # current_qud_text understiger detta värde, överrids en
    # related_to_qud-klassificering till new_main_question.
    #
    # OBS: tröskeln måste kalibreras mot den faktiska embeddingregimen.
    # Med E5-prefix (query:) ligger likheterna i ett annat band än
    # utan. Kör scripts/calibrate_drift.py efter modell- eller
    # prefixändring och sätt värdet med
    # 'urd config set qud_drift_threshold <värde>'. Defaultvärdet här
    # är en provisorisk nivå för multilingual-e5-large MED prefix.
    qud_drift_threshold: float = 0.80

    # Dokumentbaserad drift: när aktiva chunkar finns avgörs driften
    # av högsta likheten mellan yttringen (query) och chunktexterna
    # (passage) — se qud_drift.py. PROVISORISK nivå: båda likheterna
    # loggas i debug/JSONL vid varje drift-mätning; läs av
    # fördelningen efter en testkörning och justera med
    # 'urd config set qud_drift_doc_threshold <värde>'.
    qud_drift_doc_threshold: float = 0.80

    # Tak respektive golv för antalet valda hits. Pass 2 i urvalet
    # fyller på till min_desired_hits med chunkar i sannolikhets-
    # spannet [0.35, select_min_prob) — osäkra men inte avfärdade —
    # så att elaboration och liknande vägar får material att arbeta
    # mot.
    max_hits: int = 10
    min_desired_hits: int = 3

    # Evidensboost i sannolikhetspoäng (adderas och kapas vid 1.0):
    #   - evidence_section_prob_boost: evidensobjektet delar sektion
    #     med en högt rankad textchunk. Stark indikation.
    #   - evidence_document_prob_boost: samma dokument men annan
    #     sektion. Svagare indikation.
    # Section-boost och document-boost är ömsesidigt uteslutande
    # (sektion vinner). Ett evidensobjekt kan därmed lyftas över
    # urvalströskeln av sitt textstöd, men aldrig hoppa över en
    # tydligt bättre textchunk med mer än boostens storlek — till
    # skillnad från gamla additiva +3.0 på logitskalan, som i
    # praktiken alltid vann.
    evidence_section_prob_boost: float = 0.15
    evidence_document_prob_boost: float = 0.05


settings = _build_settings()
