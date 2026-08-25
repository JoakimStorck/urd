"""
Deliberation: åtagandet som löper genom svarskedjan.

Detta är lagrets första, TYSTA form (white paper 3.0, införandevägen):
objektet byggs, loggas per tur i debug/JSONL — och påverkar ingenting.
Makt kommer först när divergensmätningen visat var lagrets avgöranden
skiljer sig från generationens på rätt ställen.

RAMEN ÄR GRAMMATISK, INTE TYPOLOGISK. En inskränkning är varje led i
frågan som avgränsar vilka belägg som är tillåtliga: "för doktorander",
"vid IIT", "enligt delegationsordningen", "år 2024". Extraktionen känner
igen FORMEN — prepositionsfraser och adverbial ur dependensparsen — inte
en lista över slag. Uppmätt 2026-08-23 över 67 frågor: 47 bar uttryckta
inskränkningar, markören "för" vanligast (17), årtal förekom i noll.
En typlista byggd på utvecklingsarbetets exempel hade alltså börjat i
fel ände.

EJ UPPRÄTTHÅLLEN ÄR ETT TILLSTÅND, INTE ETT FEL. De flesta inskränkningar
kan i dag inte mötas mot någon dimension i indexet ("för doktorander" har
inget fält). De släpps INTE tyst: de ligger kvar i åtagandet, märkta, och
loggas. Mängden ej upprätthållna inskränkningar är arbetslistan för
kommande indexdimensioner — genererad ur drift, inte ur exempel.

RAMEN ÄR EN EGENSKAP HOS FRÅGAN UNDER EN QUD. Tolv av 67 uppmätta frågor
var underspecificerade följdturer vars ram måste ÄRVAS — en femtedel av
samtalet, inte ett kantfall. Arvet styrs av samtalsrollen: en ny
huvudfråga börjar tomt, en relaterad tur ärver QUD-turens inskränkningar.
Version ett gör UNION UTAN ERSÄTTNING: markören räcker inte för att
avgöra när en ny inskränkning ersätter en ärvd ("för IIT" och "för
doktorander" delar markör men inte dimension), och i ett tyst lager
kostar det inget att vänta med ersättningsreglerna tills mätningen visar
var de behövs.

Representationen är kontraktet och modelloberoende; igenkänningen är
utbytbar. Byts parsern, eller lär sig en kommande klassificerare lösa
upp "då" mot rätt tur, fylls samma struktur bättre — beslutstabell,
loggar och divergensmätning står orörda.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app import grammar

# Årtal och intervall: "2024", "2023/2024", "2023-2025".
_YEAR = re.compile(r"\b(?:19|20)\d{2}(?:\s*[/–-]\s*(?:19|20)?\d{2})?\b")

# Tidsdeiktika: uttryck vars referent ligger i samtalet eller i
# yttrandeögonblicket, inte i frågetexten. De kan aldrig upprätthållas
# ur frågan ensam — de är ARVSBEROENDE per konstruktion.
_DEICTIC = re.compile(
    r"\b(då|nuvarande|dåvarande|senaste?|tidigare|framöver|"
    r"i ?dag|idag|för närvarande|just nu|numera)\b",
    re.IGNORECASE,
)

# Adverbial som är diskurspartiklar snarare än inskränkningar.
_NOISE_ADVERBS = {"inte", "också", "bara", "ju", "väl", "nog", "egentligen"}

STATUS_EJ_UPPRATTHALLEN = "ej_upprätthållen"
STATUS_ARVSBEROENDE = "arvsberoende"

URSPRUNG_YTTRAD = "yttrad"
URSPRUNG_ARVD = "ärvd"


@dataclass(frozen=True)
class Inskrankning:
    """
    En avgränsning av vilka belägg som är tillåtliga.

    markor är formen som kände igen den — prepositionens lemma,
    "adverbial", "artal" eller "deiktisk" — inte en semantisk typ.
    Avbildningen markör -> beläggsdimension hör hemma i konfiguration
    den dag någon dimension kan upprätthållas, inte här.
    """
    markor: str
    fras: str
    ursprung: str = URSPRUNG_YTTRAD
    fran_tur: int | None = None
    status: str = STATUS_EJ_UPPRATTHALLEN

    def as_debug(self) -> dict:
        return {
            "markor": self.markor,
            "fras": self.fras,
            "ursprung": self.ursprung,
            "fran_tur": self.fran_tur,
            "status": self.status,
        }


@dataclass
class Atagande:
    """
    Vad systemet åtagit sig för denna tur: operation och ram.

    Utfallsmängd och beslutstabell tillkommer i nästa steg — objektet
    byggs smalt och växer med mätningarna, inte före dem.
    """
    intent: str
    operation: str | None
    inskrankningar: list[Inskrankning] = field(default_factory=list)

    @property
    def ej_uppratthallna(self) -> list[Inskrankning]:
        return [i for i in self.inskrankningar
                if i.status != "upprätthållen"]

    def as_debug(self) -> dict:
        return {
            "intent": self.intent,
            "operation": self.operation,
            "inskrankningar": [i.as_debug() for i in self.inskrankningar],
            "antal_ej_uppratthallna": len(self.ej_uppratthallna),
        }


def extract_constraints(question: str) -> list[Inskrankning]:
    """
    Frågans egna, yttrade inskränkningar.

    Grammatisk nivå ur dependensparsen när den finns; årtal och
    deiktika på textnivå alltid. Utan parser extraheras mindre — och
    det syns i loggen som färre inskränkningar, aldrig som ett fel:
    lagret är tyst och får inte kunna sänka en fråga.
    """
    ut: list[Inskrankning] = []
    sedda: set[tuple[str, str]] = set()

    def lagg(markor: str, fras: str, status: str) -> None:
        nyckel = (markor, fras.lower())
        if nyckel in sedda:
            return
        sedda.add(nyckel)
        ut.append(Inskrankning(markor=markor, fras=fras, status=status))

    for m in _YEAR.finditer(question):
        lagg("artal", m.group(0), STATUS_EJ_UPPRATTHALLEN)
    for m in _DEICTIC.finditer(question):
        lagg("deiktisk", m.group(0), STATUS_ARVSBEROENDE)

    pipeline = grammar._get_pipeline()
    if pipeline is None:
        return ut
    try:
        doc = pipeline(question)
    except Exception:
        return ut
    for mening in doc.sentences:
        barn: dict[int, list] = {}
        for w in mening.words:
            barn.setdefault(w.head, []).append(w)
        for w in mening.words:
            if w.deprel in ("obl", "obl:tmod", "nmod"):
                case = [b for b in barn.get(w.id, []) if b.deprel == "case"]
                if not case:
                    continue
                led = [case[0].text, w.text] + [
                    b.text for b in barn.get(w.id, [])
                    if b.deprel in ("flat", "flat:name", "compound", "amod")
                ]
                lagg(case[0].lemma.lower(), " ".join(led),
                     STATUS_EJ_UPPRATTHALLEN)
            elif w.deprel == "advmod" and w.upos == "ADV":
                if w.lemma.lower() not in _NOISE_ADVERBS:
                    lagg("adverbial", w.text, STATUS_EJ_UPPRATTHALLEN)
    return ut


def compose(question: str, intent: str, operation: str | None,
            state) -> Atagande:
    """
    Turens åtagande: yttrade inskränkningar plus arv efter samtalsroll.

    En ny huvudfråga börjar tomt — och dess åtagande lagras på
    sessionen som QUD:ns, så att kommande turer har något att ärva.
    Relaterade turer, elaboration och verification ärver QUD-turens
    inskränkningar; sociala turer ärver ingenting och lagrar ingenting.

    Anropas EFTER driftkontrollen, så att en omtolkad tur får sin
    slutliga rolls arvsregel — en tur som driftskyddet gjort till ny
    huvudfråga ska inte ärva den QUD den just lämnade.
    """
    yttrade = extract_constraints(question)

    if intent == "new_main_question":
        atagande = Atagande(intent=intent, operation=operation,
                            inskrankningar=yttrade)
        state.qud_commitment = atagande
        return atagande

    if intent == "social_or_meta":
        return Atagande(intent=intent, operation=operation,
                        inskrankningar=yttrade)

    arvda: list[Inskrankning] = []
    tidigare: Atagande | None = getattr(state, "qud_commitment", None)
    if tidigare is not None:
        egna = {(i.markor, i.fras.lower()) for i in yttrade}
        for i in tidigare.inskrankningar:
            if (i.markor, i.fras.lower()) in egna:
                continue
            arvda.append(Inskrankning(
                markor=i.markor, fras=i.fras,
                ursprung=URSPRUNG_ARVD,
                fran_tur=state.current_qud_turn_index,
                status=i.status,
            ))
    return Atagande(intent=intent, operation=operation,
                    inskrankningar=yttrade + arvda)


# ---------------------------------------------------------------------
# Domen: utfallsklassning och den första maktklassen
# ---------------------------------------------------------------------

_TABLE_CACHE: dict | None = None


def load_table() -> dict:
    """
    Beslutstabellen ur repots rot. Feltålig: en saknad eller trasig
    tabell ger ett TYST lager (tom makt-lista), aldrig ett undantag —
    deliberationen får aldrig kunna sänka en fråga, och utan tabell
    faller systemet tillbaka till exakt det beteende det hade före
    0050.
    """
    global _TABLE_CACHE
    if _TABLE_CACHE is not None:
        return _TABLE_CACHE
    try:
        import yaml
        rot = Path(__file__).resolve().parent.parent
        data = yaml.safe_load(
            (rot / "deliberation_table.yaml").read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    _TABLE_CACHE = data
    return data


def enforced_outcomes() -> set[str]:
    makt = load_table().get("makt") or []
    return {str(m) for m in makt} if isinstance(makt, list) else set()


# Operationer vars löfte är en namngiven innehavare.
NAMING_OPERATIONS = {"entity_lookup", "entity_aggregation"}

# Intenter vars svar prövar ett TIDIGARE påstående och inte ska dömas
# som nya innehavarfrågor.
EXCLUDED_INTENTS = {"verification_or_challenge", "social_or_meta"}

# Personform kontra funktionsform, grammatiskt skild: "vem ÄR X" —
# kopula med rollpredikat — lovar en PERSON, medan "vem BESLUTAR om X"
# — agentverb — lovar en FUNKTION, och "prefekten beslutar" är då ett
# fullständigt svar. Uppmätt 2026-08-25: tre av sex divergenta var
# funktionsfrågor korrekt besvarade med funktion.
_COPULAS = {"är", "var", "blir", "heter", "vart"}
_HOLDER_PHRASES = ("uppdraget som", "rollen som", "utsedd till")


def person_form_kind(question: str) -> str | None:
    """
    Frågans innehavarform — eller None när formen inte är personbunden.

    "direkt": rollen är given, personen efterfrågas — "vem ÄR X",
    "vem har uppdraget som X". Utfallet beskriver-men-namnger-inte är
    definierat här: källorna kan beskriva rollen utan att peka ut
    någon.

    "omvand": personen är given, rollen efterfrågas — "vilken roll har
    <namn>". Här är beskriver-men-namnger-inte ODEFINIERAT: frågan
    namnger ju själv innehavaren. Uppmätt 2026-08-25, lagrets första
    felaktiga maktutövning: beskedet "namnger ingen innehavare"
    författades rakt ovanpå ett svar fullt av personens namn. Den
    omvända formens brist är en annan — bindningen bärs av pronomen
    och är OBEKRÄFTAD — och dess åtgärd är prövningen, inte
    författningen.
    """
    låg = (question or "").lower()
    # Uppdragsfraserna prövas FÖRE vem-slingan: "vem har uppdraget som
    # X" inleds med ett icke-kopulärt verb, och slingan skulle annars
    # avfärda formen innan frasen hunnit ses. Provet fällde precis den
    # ordningsföljden.
    if any(f in låg for f in _HOLDER_PHRASES):
        return "direkt"
    ord_ = låg.replace("?", " ").split()
    for i, w in enumerate(ord_):
        if w in ("vem", "vilka") and i + 1 < len(ord_):
            return "direkt" if ord_[i + 1] in _COPULAS else None
    if låg.startswith("vilken roll") or "vilken roll har" in låg:
        return "omvand"
    return None


def is_person_form(question: str) -> bool:
    """Lovar frågan en person eller en personbunden roll?"""
    return person_form_kind(question) is not None


def role_phrase(question: str) -> str:
    """
    Det efterfrågade ledet ur en personformad fråga: "Vem är X?" -> X.

    Deterministisk och medvetet enkel — misslyckas den blir det
    generiska "rollen", aldrig ett undantag. Systemförfattade meningar
    får inte kunna sänka en fråga.
    """
    ord_ = (question or "").replace("?", " ").split()
    låg = [w.lower() for w in ord_]
    for i, w in enumerate(låg):
        if w in ("vem", "vilka") and i + 1 < len(ord_) and låg[i + 1] in _COPULAS:
            rest = " ".join(ord_[i + 2:]).strip()
            return rest or "rollen"
    return "rollen"


def _binder_namn(bindning: dict, namnpredikat) -> bool:
    if not bindning.get("verifierbar"):
        return False
    return (namnpredikat(bindning.get("subjekt", ""))
            or namnpredikat(bindning.get("predikat", "")))


def judge_naming_outcome(operation: str, intent: str, question: str,
                         abstained: bool, claims: dict | None,
                         namnpredikat) -> str | None:
    """
    Utfallsklass för en innehavarfråga — eller None när domen inte
    är tillämplig (annan operation, funktionsform, undantagen intent).

    Delas av driftvägen och measure_divergence: en dom, två
    konsumenter. namnpredikat injiceras (grammar.looks_like_person_name)
    så att funktionen är prövbar utan modulimportens sidoeffekter.
    """
    if operation not in NAMING_OPERATIONS:
        return None
    if intent in EXCLUDED_INTENTS:
        return None
    if not is_person_form(question):
        return None
    if abstained:
        return "avstar"
    bindningar = (claims or {}).get("bindningar", [])
    if person_form_kind(question) == "omvand":
        # Den omvända formen: svaret ska binda den GIVNA personen till
        # en roll. En pronomenbindning är obekräftad — prövningens
        # fall. Ett svar utan overifierbara bindningar lämnas odömt
        # tills prövningen kan jämföra bindningen mot frågans namn.
        if any(not b.get("verifierbar") for b in bindningar):
            return "obekraftad_bindning"
        return None
    if any(_binder_namn(b, namnpredikat) for b in bindningar):
        return "namnger"
    return "beskriver_men_namnger_inte"


def author_unnamed_holder(question: str) -> str:
    """
    Det systemförfattade utfallet "beskriver men namnger inte".

    En SATS med sin grund, inte en osäkerhetsredovisning: den påstår
    vad källorna gör och inte gör. Avgränsningen är ärlig — "källorna"
    är de som bar svaret, inte beståndet som helhet, för det är vad
    denna tur faktiskt vet.
    """
    return (f"Källorna beskriver {role_phrase(question)} "
            "men namnger ingen innehavare.")
