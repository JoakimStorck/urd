"""
Grammatisk dragutvinning ur svensk text.

Modulen är medvetet fri från URD-begrepp: den vet ingenting om chunkar,
källor, roller, lärarkategorier eller belopp. Den tar text och returnerar
prövbara semantiska drag. Kopplingen till URD:s begrepp sker i
`predication.py`.

BAKGRUND

Tre bekräftade fel 2026-08-14 har samma logiska form: svaret påstår ett
samband som källan inte bär.

    Källa:  "Prefekten uppdrog åt HR-specialist Thomas Bodegrim att utreda."
    Svar:   "Thomas Bodegrim har rollen som prefekt."

Felet överlever modellbyte (Nemo -> gemma4:12b) och avstängt resonemang,
och det kan inte fångas med strängmatchning: två testassertioner har redan
glidit förbi på omformuleringar. Det kan heller inte fångas med
likhetsmått — "X är Y" och "Y uppdrog åt X" ligger nära varandra i varje
embeddingrum, eftersom de handlar om samma personer och samma roller.
Skillnaden är syntaktisk, inte distributionell.

DRAGEN

1. IDENTITET / ÅTSKILLNAD — betecknar två uttryck samma sak?
2. AGENS — vem gör vad (predikat och dess argument)
3. MODALITET — ska / bör / får / kan; kravnivå i normativ text
4. KVANTITET — minst / högst / längst; gränsriktning kring tal

REGLERNA ÄR EN AVBILDNING FRÅN UD-RELATIONER, INTE EN LISTA ÖVER FALL

Varje regel ska kunna formuleras som en sats om syntax, inte om innehåll.
"Apposition binder titel till namn" duger. "Rollordet proprefekt binder
till närmaste egennamn" duger inte — det är en lookup-tabell förklädd
till grammatik.

VAD LAGRET INTE KAN

- Parafras ("personalspecialist" mot "HR-specialist").
- Koreferens över meningsgräns ("Hon är proprefekt").
- Räckvidd för negation och villkor syns inte alltid i trädet.

Dessa gränser mäts i skuggläge innan något byggs ovanpå.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datamodell
# ---------------------------------------------------------------------------

@dataclass
class Feature:
    """
    Ett utvunnet drag ur en mening.

    kind      identitet | atskillnad | agens | modalitet | kvantitet
    a, b      de två uttryck draget relaterar (b kan vara None)
    relation  vilken konstruktion som bar draget (UD-relation eller mönster)
    sentence  meningen draget kom ur, för spårbarhet
    strength  'asserterad' eller 'presupponerad' — se nedan
    """
    kind: str
    a: str
    b: str | None = None
    relation: str = ""
    sentence: str = ""
    strength: str = "asserterad"
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# Skillnaden mellan asserterad och presupponerad är inte kosmetisk.
#
#   "Ewa Wäckelgård är proprefekt"            -> asserterad
#   "Proprefekt Ewa Wäckelgård föredrog ..."  -> presupponerad
#
# Appositionen PÅSTÅR inte rollen, den förutsätter den — vilket syns på att
# den överlever negation: "Proprefekt Ewa Wäckelgård föredrog inte ärendet"
# tar fortfarande rollen för given. Presupposition är giltig evidens men
# svagare, och den säger dessutom att personen är EN innehavare av rollen,
# inte DEN. Räckvidden (organisation, tidpunkt) ligger utanför meningen och
# måste hämtas ur dokumentets metadata.
ASSERTED = "asserterad"
PRESUPPOSED = "presupponerad"


# ---------------------------------------------------------------------------
# Modalitet och kvantitet — slutna ordmängder
# ---------------------------------------------------------------------------

# Kravnivåer i svensk normativ text. Mängden är sluten och ordnad efter
# styrka; det är skillnaden mellan "ska inhämtas" och "bör normalt ha" som
# är hela poängen med ett regelverk.
MODAL_LEVELS = {
    "ska": "krav",
    "skall": "krav",
    "måste": "krav",
    "bör": "rekommendation",
    "borde": "rekommendation",
    "får": "tillåtelse",
    "kan": "möjlighet",
}

# Gränsmarkörer. Siffervakten kontrollerar att talet finns i källan — inte
# att riktningen stämmer. "Högst tre" där källan säger "minst tre" passerar
# idag utan anmärkning.
QUANTIFIER_DIRECTION = {
    "minst": "undre",
    "lägst": "undre",
    "högst": "övre",
    "längst": "övre",
    "max": "övre",
    "maximalt": "övre",
    "minimum": "undre",
    "över": "övre",
    "under": "undre",
}

_NEGATIONS = {"inte", "ej", "icke", "aldrig"}


# ---------------------------------------------------------------------------
# Parserinladdning — lat, valfri, aldrig fatal
# ---------------------------------------------------------------------------

_PIPELINE = None
_PIPELINE_FAILED = False


def _get_pipeline():
    """
    Ladda Stanzas svenska pipeline vid första behov.

    Laddningen är lat av två skäl: modulen ska kunna importeras utan att
    stanza är installerat, och servern ska starta lika snabbt som förut när
    lagret är avstängt. Misslyckas laddningen loggas det EN gång och
    modulen faller tillbaka på tomma resultat — aldrig ett undantag upp i
    svarskedjan. Ett analyslager i skuggläge får inte kunna sänka en fråga.
    """
    global _PIPELINE, _PIPELINE_FAILED
    if _PIPELINE is not None or _PIPELINE_FAILED:
        return _PIPELINE
    try:
        import stanza  # type: ignore
        _PIPELINE = stanza.Pipeline(
            lang="sv",
            processors="tokenize,pos,lemma,depparse",
            download_method=None,   # ingen nedladdning i drift
            logging_level="WARN",
            use_gpu=False,          # CPU: modellen är liten, GPU:n bär gemma
        )
        logger.info("grammatik: stanza-pipeline (sv) laddad")
    except Exception as e:
        _PIPELINE_FAILED = True
        logger.warning(
            "grammatik: kunde inte ladda stanza (%s). Predikationslagret "
            "är inaktivt. Installera med 'pip install stanza' och hämta "
            "modellen med stanza.download('sv').", e,
        )
    return _PIPELINE


def is_available() -> bool:
    return _get_pipeline() is not None


# ---------------------------------------------------------------------------
# Textfilter — vad som är värt att parsa
# ---------------------------------------------------------------------------

_TABLE_MARKERS = ("[Tabell]", "|")
_LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

# Källhänvisningar är systemets egen notation, inte innehåll. Utan
# sanering parsas "[Källa 1]" som text och ger drag av typen
# "har -> 1" — uppmätt 2026-08-15. Mönstret tillåter förvanskade
# varianter ("[Käur 4]", "[Klla 1]") som gemma4 producerar.
# Ett kort ord följt av en siffra inom hakparentes. Bredare än
# "Källa N" med flit: gemma4 producerar förvanskade varianter, och
# att matcha formen i stället för ordet gör mönstret robust mot
# stavningar vi inte sett.
_CITATION = re.compile(r"\[\s*[^\]\d]{0,12}\d+(?:\s*[,;]\s*[^\]\d]{0,12}\d+)*\s*\]")

# Metapåståenden om underlaget är systemets tal om sig självt, inte
# påståenden om verkligheten: "Det framgår inte i källorna att ...".
# Uppmätt gav de drag som "Det -> källorna", vilka aldrig kan ha stöd
# eftersom de handlar om källmängden och inte om dess innehåll.
_META_MARKERS = (
    "källorna", "källan", "källtexten", "de tillgängliga källorna",
    "de indexerade dokumenten", "tidigare svar",
)


def strip_citations(text: str) -> str:
    return _CITATION.sub(" ", text)


def _is_meta_sentence(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _META_MARKERS)


def sentence_like_lines(text: str) -> list[str]:
    """
    Behåll bara rader som liknar löpande meningar.

    Tabellrader och punktlistor undantas medvetet: en dependensparser på
    OCR-bruten tabellstruktur producerar skräp med hög konfidens, vilket är
    värre än inget resultat alls. Evidensobjektmodellen är rätt mekanism
    för strukturerat material och fungerar redan.
    """
    out: list[str] = []
    for line in text.splitlines():
        stripped = strip_citations(line).strip()
        if len(stripped) < 25:
            continue
        if _is_meta_sentence(stripped):
            continue
        if any(m in stripped for m in _TABLE_MARKERS):
            continue
        if _LIST_LINE.match(stripped):
            continue
        if not re.search(r"[a-zåäöA-ZÅÄÖ]{3}", stripped):
            continue
        out.append(stripped)
    return out


# ---------------------------------------------------------------------------
# Dragutvinning
# ---------------------------------------------------------------------------

def _word_text(words, idx: int) -> str:
    """Text för ett ord givet 1-baserat UD-id."""
    return words[idx - 1].text if 0 < idx <= len(words) else ""


def _phrase(words, head_idx: int) -> str:
    """
    Ytlig fras kring ett huvudord: huvudordet plus dess omedelbara
    nominala modifierare. Räcker för att få med "HR-specialist Thomas
    Bodegrim" som en enhet utan att bygga en fullständig frasstruktur.
    """
    parts = {head_idx}
    for w in words:
        if w.head == head_idx and w.deprel in ("flat", "flat:name", "compound", "amod", "nmod:poss"):
            parts.add(w.id)
    return " ".join(_word_text(words, i) for i in sorted(parts)).strip()


def extract_features(text: str, max_sentences: int = 40) -> list[Feature]:
    """
    Utvinn drag ur text. Returnerar tom lista om parsern saknas.

    max_sentences begränsar arbetet per anrop; en chunk på 1200 tecken
    rymmer sällan fler.
    """
    nlp = _get_pipeline()
    if nlp is None:
        return []

    lines = sentence_like_lines(text)
    if not lines:
        return []

    try:
        doc = nlp("\n".join(lines))
    except Exception as e:
        logger.warning("grammatik: parsning misslyckades (%s)", e)
        return []

    features: list[Feature] = []
    for sent in doc.sentences[:max_sentences]:
        words = sent.words
        stext = sent.text
        features.extend(_identity(words, stext))
        features.extend(_distinction(words, stext))
        features.extend(_agency(words, stext))
        features.extend(_modality(words, stext))
        # Kvantitet gav 2 drag av 291 i mätningen 2026-08-15 och bär
        # inte sin underhållskostnad. Funktionen behålls men anropas
        # inte; gränsriktning är fortfarande en verklig felklass och
        # kan återaktiveras om ett testfall visar att den behövs.
        # features.extend(_quantity(words, stext))
    features.extend(_parenthetical_identity(text))
    return features


# Nominala ordklasser. Identitet kräver att BÅDA leden är nominala:
# utan kravet plockas relativsatser upp som appositioner, vilket gav
# draget "som -> nuvarande proprefekten" i mätningen 2026-08-15.
_NOMINAL_UPOS = {"NOUN", "PROPN"}

# Predikativmarkörer i sluten klass. "som" och "till" är funktionsord,
# inte en lista över verb — det är skillnaden mot att räkna upp
# "har rollen som", "innehar uppdraget som", "utses till".
_PREDICATIVE_MARKS = {"som", "till"}


def _is_nominal(words, idx: int) -> bool:
    return 0 < idx <= len(words) and words[idx - 1].upos in _NOMINAL_UPOS


def _identity(words, stext: str) -> list[Feature]:
    """
    Identitet uttryckt som villkor på dependensrelationer.

    Tre konstruktioner, alla formulerade som satser om syntax:

    1. APPOSITION (appos), båda leden nominala.
       "Proprefekt Ewa Wäckelgård föredrog ärendet."
       Presupponerad: rollen påstås inte, den förutsätts — vilket syns
       på att den överlever negation.

    2. KOPULA (cop), predikativet nominalt.
       "Ewa Wäckelgård är proprefekt." Asserterad.

    3. PREDIKATIV MED MARKÖR: ett verb som tar ett nominalt led inlett
       av "som" eller "till", bundet till verbets subjekt.
       "Thomas Bodegrim har rollen som prefekt", "N utses till X",
       "N tjänstgör som X", "N verkar som X".

       Verbets betydelse används INTE. Det som gör konstruktionen till
       en identitet är att predikativet är nominalt och inlett av en
       markör ur en sluten klass — inte vilket verb som råkar stå där.
       Därmed täcks även konstruktioner vi inte har sett.

       Mätningen 2026-08-15 visade att avsaknaden av denna regel gjorde
       lagret blint för sitt eget testfall: "Thomas Bodegrim har rollen
       som prefekt" gav ingen identitet alls.
    """
    out: list[Feature] = []
    for w in words:
        if w.deprel == "appos":
            if not (_is_nominal(words, w.head) and _is_nominal(words, w.id)):
                continue
            out.append(Feature(
                kind="identitet", a=_phrase(words, w.head), b=_phrase(words, w.id),
                relation="appos", sentence=stext, strength=PRESUPPOSED,
            ))

        if w.deprel == "cop":
            pred = w.head
            if not _is_nominal(words, pred):
                continue
            subj = next(
                (x.id for x in words
                 if x.head == pred and x.deprel in ("nsubj", "nsubj:pass")
                 and _is_nominal(words, x.id)),
                None,
            )
            if subj:
                out.append(Feature(
                    kind="identitet", a=_phrase(words, subj), b=_phrase(words, pred),
                    relation="cop", sentence=stext, strength=ASSERTED,
                ))

        # Predikativ med markör
        if w.deprel in ("xcomp", "obl", "obl:arg", "nmod") and _is_nominal(words, w.id):
            mark = next(
                (x for x in words
                 if x.head == w.id and x.deprel in ("mark", "case")
                 and x.text.lower() in _PREDICATIVE_MARKS),
                None,
            )
            if not mark:
                continue
            head = w.head
            subj = next(
                (x.id for x in words
                 if x.head == head and x.deprel in ("nsubj", "nsubj:pass")
                 and _is_nominal(words, x.id)),
                None,
            )
            if subj:
                out.append(Feature(
                    kind="identitet", a=_phrase(words, subj), b=_phrase(words, w.id),
                    relation=f"predikativ:{mark.text.lower()}", sentence=stext,
                    strength=ASSERTED,
                ))
    return out


# Sluten klass av svenska funktionsord. Att stryka dem framför en
# parentes är en sats om syntax (prepositioner och konjunktioner kan
# inte vara appositionens huvudord), inte en lista över domänfall.
_FUNCTION_WORDS = {
    "av", "för", "till", "i", "på", "med", "hos", "om", "som", "och",
    "efter", "vid", "från", "genom", "att", "den", "det", "en", "ett",
    "är", "fattas", "beslutas",
}


_REFERENCE_MARKERS = {
    "se", "jfr", "dvs", "exempelvis", "bilaga", "not", "enligt",
    "eller", "resp", "respektive",
}


def _strip_leading_function_words(phrase: str) -> str:
    words = phrase.split()
    while words and words[0].lower() in _FUNCTION_WORDS:
        words.pop(0)
    return " ".join(words).strip()


def _parenthetical_identity(text: str) -> list[Feature]:
    """
    Parentetisk apposition: "vicekansler (Vice Chancellor)".

    Beståndet definierar sina egna termekvivalenser, tvåspråkigt, och ingen
    läser dem idag: synonymlistan har 12 handskrivna grupper. Detta är
    white paperns tredje synonymväg — utnyttja dokumentens egen variation.
    """
    out: list[Feature] = []
    for m in re.finditer(
        r"((?:[\wÅÄÖåäö\-]+[ ]){0,2}[\wÅÄÖåäö\-]+)[ ]*\(([A-ZÅÄÖa-zåäö][\w\- ]{1,40}?)\)",
        text,
    ):
        a = _strip_leading_function_words(m.group(1))
        b = m.group(2).strip()
        if not a or a.lower() == b.lower():
            continue
        # "(se ovan)", "(jfr bilaga 2)" är hänvisningar, inte
        # appositioner. En apposition har ett nominalt huvudord.
        if b.split()[0].lower() in _REFERENCE_MARKERS | _FUNCTION_WORDS:
            continue
        out.append(Feature(
            kind="identitet", a=a, b=b, relation="parentes",
            sentence=m.group(0), strength=PRESUPPOSED,
        ))
    return out


def _distinction(words, stext: str) -> list[Feature]:
    """
    Åtskillnad: skilda argument till samma predikat har normalt skilda
    referenter.

        "Prefekten uppdrog åt HR-specialist Thomas Bodegrim att utreda."

    Subjekt och objekt kan inte vara samma person här — vilket är den enda
    negativa slutsats som går att dra mekaniskt ur en sats, och exakt den
    som fäller "Thomas Bodegrim har rollen som prefekt". Reflexiva
    konstruktioner är undantaget och uppträder inte i den här formen.
    """
    out: list[Feature] = []
    verbs = {w.id for w in words if w.upos in ("VERB", "AUX")}
    for v in verbs:
        args = [w for w in words if w.head == v and w.deprel in ("nsubj", "obj", "iobj", "obl", "obl:arg")]
        for i, x in enumerate(args):
            for y in args[i + 1:]:
                out.append(Feature(
                    kind="atskillnad", a=_phrase(words, x.id), b=_phrase(words, y.id),
                    relation=f"{x.deprel}/{y.deprel}", sentence=stext,
                    extra={"predikat": _word_text(words, v)},
                ))
    return out


def _agency(words, stext: str) -> list[Feature]:
    """
    Vem gör vad. Sannolikt det vanligaste påståendet i URD:s svar —
    "prefekt beslutar om medfinansiering", "rektor utser proprefekt".

    Felklassen är densamma som rollbindning men på verbet: bekräftat fall
    i baslinjen 2026-08-12 var "Vice rektor har delegerat till rektor",
    inverterad delegationsordning. nsubj/obj är parsningens mest
    tillförlitliga relationer, så detta är mekaniskt det enklaste draget.
    """
    out: list[Feature] = []
    for w in words:
        if w.upos != "VERB":
            continue
        subj = next((x for x in words if x.head == w.id and x.deprel in ("nsubj", "nsubj:pass")), None)
        if not subj:
            continue
        obj = next((x for x in words if x.head == w.id and x.deprel in ("obj", "obl", "obl:arg", "xcomp")), None)
        negated = any(
            x.head == w.id and x.deprel == "advmod" and x.text.lower() in _NEGATIONS
            for x in words
        )
        passive = subj.deprel == "nsubj:pass"
        out.append(Feature(
            kind="agens", a=_phrase(words, subj.id),
            b=_phrase(words, obj.id) if obj else None,
            relation=w.lemma or w.text, sentence=stext,
            extra={"negerad": negated, "passiv": passive},
        ))
    return out


def _modality(words, stext: str) -> list[Feature]:
    """Kravnivå: ska / bör / får / kan, knuten till sitt huvudverb."""
    out: list[Feature] = []
    for w in words:
        level = MODAL_LEVELS.get(w.text.lower())
        if not level or w.upos not in ("AUX", "VERB"):
            continue
        head = _word_text(words, w.head) if w.head else ""
        subj = next((x for x in words if x.head == w.head and x.deprel in ("nsubj", "nsubj:pass")), None)
        out.append(Feature(
            kind="modalitet", a=_phrase(words, subj.id) if subj else head,
            b=head, relation=w.text.lower(), sentence=stext,
            extra={"nivå": level},
        ))
    return out


def _quantity(words, stext: str) -> list[Feature]:
    """
    Gränsriktning kring tal: "minst tre ledamöter", "längst två år".

    Siffervakten kontrollerar att talet finns i källan, inte åt vilket håll
    gränsen går. "Högst tre" där källan säger "minst tre" passerar idag.
    """
    out: list[Feature] = []
    for w in words:
        direction = QUANTIFIER_DIRECTION.get(w.text.lower())
        if not direction:
            continue
        num = next(
            (x for x in words if abs(x.id - w.id) <= 3 and (x.upos == "NUM" or x.text.isdigit())),
            None,
        )
        if not num:
            continue
        out.append(Feature(
            kind="kvantitet", a=w.text.lower(), b=num.text,
            relation=_word_text(words, num.head) if num.head else "",
            sentence=stext, extra={"riktning": direction},
        ))
    return out


def summarize(features: list[Feature]) -> dict:
    """Frekvenser per dragtyp — diagnostik, inte ändamål."""
    counts: dict[str, int] = {}
    for f in features:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    return {
        "total": len(features),
        "per_kind": counts,
        "asserted": sum(1 for f in features if f.strength == ASSERTED),
        "presupposed": sum(1 for f in features if f.strength == PRESUPPOSED),
    }
