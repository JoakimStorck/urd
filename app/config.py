from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import os

load_dotenv()

URD_DIR = Path(".urd")
CONFIG_FILE = URD_DIR / "config.json"

# Hårdkodade defaults — dessa skrivs till .urd/config.json om filen saknas
DEFAULTS = {
    "docs_path": "./docs",
    "qdrant_path": "./data/qdrant",
    "collection_name": "iit_docs",
    "embedding_model": "intfloat/multilingual-e5-large",
    "reranker_model": "jeffwan/mmarco-mMiniLMv2-L12-H384-v1",
    "ollama_model": "mistral-nemo",
    "preprocess_ollama_model": "mistral",
    "preprocess_semantic_version": "v1",
    "top_k": 3,
    "chunk_size": 1200,
    "chunk_overlap": 150,
    "preprocess_max_section_chars": 6000,
    "server": "",
    "qud_background_turns": 1,
    "social_history_turns": 4,
    "classification_history_turns": 2,
    "expansion_score_threshold": 0.2,
    "expanded_filter_floor": -1.0,
    "qud_drift_threshold": 0.55,
    "min_relevance_floor": 0.0,
    "relevance_ratio": 0.3,
    "max_hits": 10,
    "min_desired_hits": 3,
    "evidence_section_boost": 3.0,
    "evidence_document_boost": 0.5,
}

# Mapping: config-nyckel → miljövariabel
_ENV_KEYS = {
    "docs_path": "DOCS_PATH",
    "qdrant_path": "QDRANT_PATH",
    "collection_name": "QDRANT_COLLECTION",
    "embedding_model": "EMBEDDING_MODEL",
    "reranker_model": "RERANKER_MODEL",
    "ollama_model": "OLLAMA_MODEL",
    "preprocess_ollama_model": "PREPROCESS_OLLAMA_MODEL",
    "preprocess_semantic_version": "PREPROCESS_SEMANTIC_VERSION",
    "top_k": "TOP_K",
    "chunk_size": "CHUNK_SIZE",
    "chunk_overlap": "CHUNK_OVERLAP",
    "preprocess_max_section_chars": "PREPROCESS_MAX_SECTION_CHARS",
    "server": "URD_SERVER",
    "qud_background_turns": "QUD_BACKGROUND_TURNS",
    "social_history_turns": "SOCIAL_HISTORY_TURNS",
    "classification_history_turns": "CLASSIFICATION_HISTORY_TURNS",
    "expansion_score_threshold": "EXPANSION_SCORE_THRESHOLD",
    "expanded_filter_floor": "EXPANDED_FILTER_FLOOR",
    "qud_drift_threshold": "QUD_DRIFT_THRESHOLD",
    "min_relevance_floor": "MIN_RELEVANCE_FLOOR",
    "relevance_ratio": "RELEVANCE_RATIO",
    "max_hits": "MAX_HITS",
    "min_desired_hits": "MIN_DESIRED_HITS",
    "evidence_section_boost": "EVIDENCE_SECTION_BOOST",
    "evidence_document_boost": "EVIDENCE_DOCUMENT_BOOST",
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

    server = s("server").strip() or None

    return Settings(
        docs_path=Path(s("docs_path")),
        qdrant_path=Path(s("qdrant_path")),
        collection_name=s("collection_name"),
        embedding_model=s("embedding_model"),
        reranker_model=s("reranker_model"),
        ollama_model=s("ollama_model"),
        preprocess_ollama_model=s("preprocess_ollama_model"),
        preprocess_semantic_version=s("preprocess_semantic_version"),
        top_k=i("top_k"),
        chunk_size=i("chunk_size"),
        chunk_overlap=i("chunk_overlap"),
        preprocess_max_section_chars=i("preprocess_max_section_chars"),
        server=server,
        qud_background_turns=i("qud_background_turns"),
        social_history_turns=i("social_history_turns"),
        classification_history_turns=i("classification_history_turns"),
        expansion_score_threshold=f("expansion_score_threshold"),
        expanded_filter_floor=f("expanded_filter_floor"),
        qud_drift_threshold=f("qud_drift_threshold"),
        min_relevance_floor=f("min_relevance_floor"),
        relevance_ratio=f("relevance_ratio"),
        max_hits=i("max_hits"),
        min_desired_hits=i("min_desired_hits"),
        evidence_section_boost=f("evidence_section_boost"),
        evidence_document_boost=f("evidence_document_boost"),
    )


class Settings(BaseModel):
    docs_path: Path = Path("./docs")
    qdrant_path: Path = Path("./data/qdrant")
    synonyms_path: Path = Path(".urd/synonyms.yaml")
    concepts_path: Path = Path(".urd/concepts.yaml")
    collection_name: str = "iit_docs"

    embedding_model: str = "intfloat/multilingual-e5-large"
    reranker_model: str = "jeffwan/mmarco-mMiniLMv2-L12-H384-v1"

    ollama_model: str = "mistral-nemo"
    preprocess_ollama_model: str = "mistral"

    preprocess_semantic_version: str = "v1"

    top_k: int = 3

    chunk_size: int = 1200
    chunk_overlap: int = 150

    preprocess_max_section_chars: int = 6000
    server: str | None = None

    # Samtalskontext — hur mycket historik som skickas med i olika steg.
    # Varje värde räknas i "turer" där en tur = ett fråga-svar-par.
    # qud_background_turns används för related_to_qud och
    # verification_or_challenge, där föregående turer ges som bakgrund
    # i evidensextraktionen.
    qud_background_turns: int = 1
    social_history_turns: int = 4
    classification_history_turns: int = 2

    # Retrieval-trösklar.
    # expansion_score_threshold: minsta cross-encoder-score som krävs
    #   för att ett dokument ska expanderas. Sänk för att vara mer
    #   generös med borderline-relevanta dokument.
    # expanded_filter_floor: lägsta score som tillåts för chunkar som
    #   kommer från expanderade dokument. Negativt värde betyder att
    #   även chunkar cross-encodern är osäker på släpps igenom,
    #   eftersom dokumentet som helhet redan visat sig relevant.
    expansion_score_threshold: float = 0.2
    expanded_filter_floor: float = -1.0

    # QUD-drift-skydd. Om embedding-likhet mellan aktuell fråga och
    # current_qud_text understiger detta värde, överrids en
    # related_to_qud-klassificering till new_main_question.
    # Värdet är kalibrerat för multilingual-e5-large.
    qud_drift_threshold: float = 0.55

    # Relevansbaserat hit-urval (ersätter hårdkodat top_k).
    # Strategin: alla hits med score ≥ max(min_relevance_floor,
    # top_score × relevance_ratio) tas med, upp till max_hits.
    # Om färre än min_desired_hits passerar men fler finns med
    # positiv score, tas upp till min_desired_hits totalt.
    #
    # min_relevance_floor = 0.0 innebär att vi litar på den relativa
    # cutoff:en. Ett absolut golv används inte — det är den relativa
    # fördelningen som avgör, precis som en människa skulle göra en
    # bedömning baserat på hur tydligt bästa träffen sticker ut.
    min_relevance_floor: float = 0.0
    relevance_ratio: float = 0.3
    max_hits: int = 10
    min_desired_hits: int = 3

    # Evidensboost: evidensobjekt (tabeller, listor, figurer) får
    # ett pålägg på sin cross-encoder-score när det finns stöd för
    # att objektet är relevant för just denna fråga, utöver sin
    # egen språkliga matchning. Pålägg adderas i två steg:
    #   - evidence_section_boost: evidensobjektet delar sektion med
    #     en högt rankad textchunk. Stark indikation att den
    #     förklarande texten hör ihop med objektet.
    #   - evidence_document_boost: evidensobjektet är från samma
    #     dokument som en högt rankad textchunk. Svagare indikation
    #     men ger en lätt förmån åt strukturella objekt i relevanta
    #     dokument.
    # Section-boost ges bara, document-boost bara om section inte
    # redan matchade.
    evidence_section_boost: float = 3.0
    evidence_document_boost: float = 0.5


settings = _build_settings()