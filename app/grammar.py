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
    ambiguous draget kommer ur en konstruktion som tillåter mer än en
              läsning; se _title_identity
    """
    kind: str
    a: str
    b: str | None = None
    relation: str = ""
    sentence: str = ""
    strength: str = "asserterad"
    ambiguous: bool = False
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


# Kontextprefixet som ingest lägger på varje chunk:
#
#     Dokument: 20250603_Information och teknik_protokoll
#     Avsnitt: 13 Beslut: nomineringskommitté för uppdrag som proprefekt IIT
#     ---
#     <källtext>
#
# Detta är URD:s EGEN metadata, inte källtext, och den får inte
# generera drag. Uppmätt 2026-08-15: rubrikraden flöt ihop med den
# efterföljande meningen och gav den fabricerade identiteten
# "IIT Prefekt -> proprefekt", samtidigt som samordningen i
# "Prefekt och HR-expert Thomas Bodegrim" bröts sönder — draget för
# prefekt försvann helt och HR-expert tappade sin tvetydighetsflagga.
#
# Med prefixet strippat ger samma text båda dragen med ambiguous=True.
# Radvis parsning prövades och gav identiskt resultat, alltså behövs
# den inte.
#
# Dokumentkontexten går inte förlorad: den finns i chunkens metadata
# (document_title, section_path) och hämtas därifrån när ett drag ska
# knytas till sin källa.
_CONTEXT_PREFIX = re.compile(
    r"\A(?:(?:Dokument|Avsnitt|Document|Section):[^\n]*\n)+---\n",
)


def strip_context_prefix(text: str) -> str:
    """Ta bort ingest-prefixet före parsning."""
    return _CONTEXT_PREFIX.sub("", text, count=1)


def strip_citations(text: str) -> str:
    return _CITATION.sub(" ", text)


def _is_meta_sentence(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _META_MARKERS)


# En rad som bär subjekt och finit verb är en sats även om den är kort
# eller inleds av en listmarkör. Uppmätt 2026-08-15: det gamla filtret
# (>=25 tecken, ingen listmarkör) sållade bort "Prefekt och HR-expert
# Thomas Bodegrim presenterade ärendet." i vissa chunkar och samtliga
# föredragande- och närvaroförteckningar — alltså exakt den textform
# där rollbindningar lever. Källidentiteterna låg därför kvar på 169
# trots att titeluttaget bevisligen fungerar på verkliga träd.
_MIN_SENTENCE_LEN = 12


def sentence_like_lines(text: str) -> list[str]:
    """
    Behåll rader som kan bära en sats.

    Tabellrader undantas fortfarande: en dependensparser på OCR-bruten
    tabellstruktur producerar skräp med hög konfidens, vilket är värre
    än inget resultat alls. Evidensobjektmodellen är rätt mekanism för
    strukturerat material och fungerar redan.

    Listmarkören strippas i stället för att diskvalificera raden.
    Numrerade processteg i beslutsdokument ("3. Prefekt föredrar
    slutgiltigt förslag i IL.") är fullständiga satser med agens, och
    de är bland de mest informationsrika raderna i beståndet.
    """
    out: list[str] = []
    for line in text.splitlines():
        stripped = strip_citations(line).strip()
        if any(m in stripped for m in _TABLE_MARKERS):
            continue
        # Listmarkören är formatering, inte innehåll.
        stripped = _LIST_LINE.sub("", stripped).strip()
        if len(stripped) < _MIN_SENTENCE_LEN:
            continue
        if _is_meta_sentence(stripped):
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
    # Parentetiska appositioner ("vicekansler (Vice Chancellor)")
    # utvinns med regex på RÅTEXTEN, före radfiltret och oberoende av
    # parsern. De sitter ofta i tabellceller och rubriker som aldrig
    # når dependensparsningen, och de är beståndets egen tvåspråkiga
    # termordlista — den som de tolv handskrivna synonymgrupperna
    # saknar. Uppmätt 2026-08-15: samtliga sådana par extraherades ur
    # SVAREN men var obelagda, eftersom källsidan filtrerades bort.
    text = strip_context_prefix(text)
    parenthetical = _parenthetical_identity(text)

    nlp = _get_pipeline()
    if nlp is None:
        return parenthetical

    lines = sentence_like_lines(text)
    if not lines:
        return parenthetical

    try:
        doc = nlp("\n".join(lines))
    except Exception as e:
        logger.warning("grammatik: parsning misslyckades (%s)", e)
        return []

    features: list[Feature] = []
    for sent in doc.sentences[:max_sentences]:
        words = sent.words
        stext = sent.text
        if not _has_finite_verb(words):
            continue
        features.extend(_identity(words, stext))
        features.extend(_distinction(words, stext))
        features.extend(_agency(words, stext))
        features.extend(_modality(words, stext))
        # Kvantitet gav 2 drag av 291 i mätningen 2026-08-15 och bär
        # inte sin underhållskostnad. Funktionen behålls men anropas
        # inte; gränsriktning är fortfarande en verklig felklass och
        # kan återaktiveras om ett testfall visar att den behövs.
        # features.extend(_quantity(words, stext))
    features.extend(parenthetical)
    return features


# Nominala ordklasser. Identitet kräver att BÅDA leden är nominala:
# utan kravet plockas relativsatser upp som appositioner, vilket gav
# draget "som -> nuvarande proprefekten" i mätningen 2026-08-15.
# Disjunktionsmarkörer. "Examinatorn (alternativt betygsnämnd)" är
# INTE en identitet — ordet säger uttryckligen att de är olika
# alternativ. Uppmätt 2026-08-15 extraherades två sådana som
# identiteter, alltså med rakt motsatt innebörd mot källans.
_DISJUNCTION_MARKERS = {
    "alternativt", "respektive", "eller", "resp", "ev", "eventuellt",
}

# Lagreferenser. "HF 4 kap. 4 §" gav draget "HF -> kap." — numrerade
# hänvisningar är inte påståenden om entiteter.
_LEGAL_REF = re.compile(
    r"\b(HF|SFS|kap\.?|§|dnr|C\s*\d{4}/\d+)\b", re.I
)

# Abstrakta platshållare bär inte referens i sig: "rollen", "kraven",
# "ersättningen" är relationsplatser, inte entiteter. Uppmätt gav de
# drag som "följande krav -> professor" och "Ersättningen -> opponent".
# Listan är sluten och grammatisk till sin natur — orden är
# relationssubstantiv — men den är en ordlista och ska hållas kort.
# Bär vänsterledet ett av dessa ord som huvudord söks referenten via
# verbets subjekt i stället (se predikativregeln).
_PLACEHOLDER_HEADS = {
    "roll", "rollen", "uppdrag", "uppdraget", "krav", "kraven",
    "ersättning", "ersättningen", "befattning", "befattningen",
    "funktion", "funktionen", "post", "posten", "titel", "titeln",
}


# Rums- och objektskoder är inte titlar. Uppmätt 2026-08-15 gav det
# öppnade radfiltret drag som "B422 -> Tomas Person" och
# "B422 -> B421" ur labbansvarigdokumentet: en kod bunden till ett
# namn ser strukturellt ut som en titelkonstruktion.
#
# Villkoret är formmässigt, inte en lista: ett titelled är ett ord med
# gemener som inte är dominerat av siffror. "Prefekt", "HR-expert" och
# "proprefekt" passerar; "B422", "5.1.1.3" och "§13" gör det inte.
def _is_code_like(phrase: str) -> bool:
    """
    Är frasen en kod snarare än ett namn eller en titel?

    Skiljelinjen går vid SIFFROR, inte vid versaler. Rena
    versalförkortningar (IL, RL, HR) är legitima termer och utgör
    dessutom kärnan i beståndets egen ordlista — "Institutionens
    ledningsråd (IL)". Koder blandar versaler med siffror (B422) eller
    domineras av siffror (5.1.1.3, §13).
    """
    if not re.search(r"[\wÅÄÖåäö]", phrase):
        return True
    digits = sum(c.isdigit() for c in phrase)
    letters = sum(c.isalpha() for c in phrase)
    if digits == 0:
        return False
    if digits >= letters:
        return True
    # Siffror i en fras utan gemener: rumskod, paragrafnummer.
    return not re.search(r"[a-zåäö]", phrase)


def _has_disjunction(words, idx: int) -> bool:
    """Ingår ledet i en uppräkning av ALTERNATIV snarare än likheter?"""
    for x in words:
        if x.head == idx and x.deprel in ("cc", "advmod", "mark"):
            if x.text.lower().rstrip(".") in _DISJUNCTION_MARKERS:
                return True
    return False


def _is_placeholder(phrase: str) -> bool:
    tokens = [t.lower() for t in re.findall(r"[\wÅÄÖåäö\-]+", phrase)]
    return bool(tokens) and any(t in _PLACEHOLDER_HEADS for t in tokens)


_NOMINAL_UPOS = {"NOUN", "PROPN"}

# Predikativmarkörer i sluten klass. "som" och "till" är funktionsord,
# inte en lista över verb — det är skillnaden mot att räkna upp
# "har rollen som", "innehar uppdraget som", "utses till".
# ENDAST "som". "till" togs bort 2026-08-15 efter mätning: den gav 175
# av 427 identitetsdrag i ett urval om 20 dokument, och nästan
# uteslutande skräp — "Nomineringar -> prefekt" ur "Nomineringar lämnas
# enskilt till prefekt", "Försäkringskassan -> arbete", "erfarenhet ->
# disputation".
#
# Antagandet att "till" fungerar som "som" i "utses till X" var fel.
# "som" är i predikativ ställning en identitetsmarkör; "till" är
# framför allt en riktnings- och mottagarpreposition, och att skilja
# tillsättningsverb från övriga kräver kunskap om vad verbet betyder —
# vilket lagret medvetet inte har. Konstruktionen "utses till X" fångas
# ändå som agens.
_PREDICATIVE_MARKS = {"som"}


def _is_nominal(words, idx: int) -> bool:
    return 0 < idx <= len(words) and words[idx - 1].upos in _NOMINAL_UPOS


def _governing_verb(words, idx: int, max_steps: int = 6) -> int | None:
    """Följ head-kedjan uppåt till närmaste verb. 1-baserat id, eller None."""
    seen = 0
    cur = idx
    while seen < max_steps and 0 < cur <= len(words):
        head = words[cur - 1].head
        if head == 0:
            return cur if words[cur - 1].upos in ("VERB", "AUX") else None
        if words[head - 1].upos in ("VERB", "AUX"):
            return head
        cur = head
        seen += 1
    return None


# Bestämningar framför en titel bär ingen egen betydelse: "Vår nye
# HR-specialist" och "HR-specialist" är samma roll. Utan normalisering
# räknas de som skilda titlar och beläggen splittras — uppmätt
# 2026-08-15 gav beståndet tre HR-specialist-belägg varav ett i den
# längre formen.
# ENDAST semantiskt tomma bestämningar. "tillförordnad", "tf" och
# "biträdande" ÄNDRAR rollen och får aldrig strippas: en tf proprefekt
# är inte proprefekten (Moudud Alam mot Ewa Wäckelgård), och
# biträdande lektor är en annan lärarkategori än lektor — hela
# kapitelskillnaden i anställningsordningen.
_TITLE_MODIFIERS = {
    "vår", "vårt", "våra", "nye", "nya", "ny", "nytt",
    "den", "det", "de", "en", "ett",
}


def normalize_title(title: str) -> str:
    """Strippa bestämningar framför ett titelled."""
    parts = title.split()
    while parts and parts[0].lower().strip(".") in _TITLE_MODIFIERS:
        parts.pop(0)
    return " ".join(parts).strip() or title.strip()


def _title_identity(words, stext: str) -> list[Feature]:
    """
    Titel bunden till egennamn.

    UPPMÄTT PARSERBETEENDE 2026-08-15 (Stanza, sv):

        Proprefekt Ewa Wäckelgård föredrog ärendet.
          1 Proprefekt  NOUN  head=2 nmod       <- titel FÖRE namn: nmod
          2 Ewa         PROPN head=4 nsubj
          3 Wäckelgård  PROPN head=2 flat:name

        Thomas Bodegrim, HR-specialist, presenterade ärendet.
          1 Thomas      PROPN head=6 nsubj
          4 HR-specialist NOUN head=1 appos     <- titel EFTER namn: appos

    Samma semantiska relation, två UD-etiketter beroende på ordföljd —
    och i båda fallen hänger titeln UNDER namnet. Den tidigare regeln
    letade bara appos och var därför blind för den vanligaste svenska
    formen: obestämd titel före namn. Det förklarar varför Ewa
    Wäckelgård aldrig extraherades ur beståndet trots att hon
    förekommer som föredragande i ett stort antal protokoll.

    Konstruktionen är stabil oavsett satsroll: i passiv form ("Ärendet
    föredrogs av proprefekt Ewa Wäckelgård") hänger titeln likadant
    under namnet, som i sin tur är obl:agent.

    CASE-KRAVET. nmod under egennamn är inte alltid en titel:
    "rapporten om Bodegrim" ger samma relation. Titelkonstruktionen
    saknar preposition mellan titel och namn, medan om/av/från alltid
    ger en case-markör. Kravet skiljer konstruktionerna åt utan att
    veta vad orden betyder.

    TVETYDIGHET. "Prefekt och HR-expert Thomas Bodegrim presenterade
    ärendet" har två giltiga läsningar:

        [Prefekt och HR-expert] Thomas Bodegrim   -> en person, två titlar
        [Prefekt] och [HR-expert Thomas Bodegrim] -> två personer

    Stanza väljer den första; i det verkliga protokollet avsågs den
    andra. Tvetydigheten sitter i källan, inte i parsern, och den kan
    inte avgöras ur meningen ensam. Drag för båda titlarna produceras
    därför med ambiguous=True — att välja en läsning vore att gissa,
    och systemet ska avstå hellre än gissa.
    """
    out: list[Feature] = []
    for w in words:
        if w.deprel not in ("nmod", "appos"):
            continue
        if not _is_nominal(words, w.id):
            continue
        # Huvudordet ska vara ett PERSONNAMN, inte bara ett PROPN.
        #
        # Mätning 2026-08-15: kravet på PROPN ensamt gav "Diarienummer C
        # -> Rektor" och "Universitetsadjunkt -> gästlärare" i
        # regeltext, eftersom Stanza taggar rolltitlar och beteckningar
        # som egennamn där. Titelkonstruktionen är dessutom en
        # PROTOKOLLföreteelse: i regeltext står roller i generisk form
        # utan namn, så regeln har där inget legitimt utfall alls.
        #
        # Villkoret är formmässigt: ett personnamn i svensk
        # förvaltningstext skrivs ut med minst två namnled, vilket i UD
        # ger ett PROPN med minst ett flat:name-barn. "Ewa Wäckelgård"
        # och "Thomas Bodegrim" uppfyller det; "Rektor" och
        # "Diarienummer C" gör det inte.
        if not (0 < w.head <= len(words)) or words[w.head - 1].upos != "PROPN":
            continue
        if not any(
            x.head == w.head and x.deprel in ("flat", "flat:name")
            and words[x.id - 1].upos == "PROPN"
            for x in words
        ):
            continue
        # Preposition mellan leden => inte en titelkonstruktion.
        if any(x.head == w.id and x.deprel == "case" for x in words):
            continue
        if _has_disjunction(words, w.id):
            continue

        name = _phrase(words, w.head)
        if _LEGAL_REF.search(name) or _LEGAL_REF.search(_phrase(words, w.id)):
            continue
        if _is_placeholder(name):
            continue
        if _is_code_like(name) or _is_code_like(_phrase(words, w.id)):
            continue
        # En titel är inte ett namn. Närvarolistor ("Thomas Bodegrim,
        # Xingxing Zhang, Moudud Alam") tolkas annars som appositioner
        # och gav drag som "Thomas Bodegrim -> Moudud Alam" — uppmätt
        # 2026-08-15. Titelledet ska vara ett vanligt substantiv.
        if words[w.id - 1].upos == "PROPN":
            continue
        titles = [w] + [
            x for x in words if x.head == w.id and x.deprel == "conj"
        ]
        ambiguous = len(titles) > 1

        # Samordnade PERSONNAMN delar titeln. "Studierektorerna Mia
        # Xiaoyun Zhao och Xingxing Zhang presenterade läget" band
        # tidigare bara den första — uppmätt 2026-08-15. Det
        # underskattar beläggningen systematiskt, och just för roller
        # med flera innehavare är det den siffran som avgör om rollen
        # bedöms unik.
        #
        # Detta är INTE samma tvetydighet som samordnade titlar framför
        # ett namn: här är det entydigt att båda personerna bär titeln.
        names = [w.head] + [
            x.id for x in words
            if x.head == w.head and x.deprel == "conj"
            and words[x.id - 1].upos == "PROPN"
        ]
        for nid in names:
            n = _phrase(words, nid)
            if _is_code_like(n) or _is_placeholder(n):
                continue
            for t in titles:
                out.append(Feature(
                    kind="identitet", a=n,
                    b=normalize_title(_phrase(words, t.id)),
                    relation="titel:" + w.deprel, sentence=stext,
                    strength=PRESUPPOSED, ambiguous=ambiguous,
                ))
    return out


def _has_finite_verb(words) -> bool:
    """
    Innehåller satsen ett predikat?

    En teckensekvens utan finit verb är inte ett påstående, och ingen
    parsning kan utvinna en relation ur något som inte påstår något.

    Detta träffar en konkret och vanlig felkälla: signaturblock och
    närvarolistor i protokoll. "Vid protokollet / Annette Lenne /
    Joakim Storck" är en LAYOUTstruktur — sannolikt två kolumner eller
    en tabell i PDF:en — som Docling platt ut till radbrytningar. I
    handklassningen 2026-08-15 stod sju av elva titelfel för den
    formen, med drag som "Annette Lenne Joakim Storck -> protokollet".

    Att Annette Lenne var sekreterare är sant, men det följer av
    genrekunskap om svensk mötesformalia, inte av texten. Att låta
    parsern gissa det vore att producera belägg som ser lästa ut men
    är härledda. Ska närvaro och sekreterarskap fångas är rätt väg en
    egen behandling av protokollens standardsektioner vid ingest, där
    formatet används i stället för att en dependensparser gissar.

    Villkoret är grammatiskt och textsortsoberoende: samtliga korrekta
    drag i stickprovet stod i fullständiga satser.
    """
    for w in words:
        if w.upos in ("VERB", "AUX"):
            feats = w.feats or ""
            if "VerbForm=Fin" in feats or "Mood=" in feats or "Tense=" in feats:
                return True
    return False


def _identity(words, stext: str) -> list[Feature]:
    """
    Identitet i en sats.

    Kvar står EN konstruktion: titel bunden till personnamn, via nmod
    (titel före namn) eller appos (titel efter namn med komma). Se
    _title_identity.

    KOPULA BORTTAGEN 2026-08-15. Handklassning av 40 observationer gav
    1 rätt av 14. Kopula i löptext uttrycker vilken predikation som
    helst, inte identitet mellan entiteter: "samarbete är en viktig
    del", "två år är en kort tid", "Utgångspunkten för arbetet är
    material som togs fram". Att den gav 3/3 i det första stickprovet
    var slumpen i ett litet urval — vilket är hela skälet till att
    precision mäts per konstruktion i stället för bedöms på intryck.

    Riktiga identiteter i kopulaform ("Ewa är adjungerad ledamot i XU")
    finns, men de går inte att skilja från övriga predikationer utan
    att veta vad orden betyder.

    PREDIKATIV MED MARKÖR BORTTAGEN samma dag: 2 rätt av 5. "Föreslås
    som ersättare" betyder att personen får den roll som blev ledig —
    "ersättare" är ingen roll. Att skilja det från "Frank Fiedler som
    ämnesansvarig" kräver en ordlista över rollord.

    Funktionerna _governing_verb och _has_disjunction behålls: de
    kostar inget och är rätt mekanismer om konstruktionerna återinförs
    med bättre avgränsning.
    """
    return _title_identity(words, stext)


# Parentesen bär tre SKILDA relationer, som handklassningen
# 2026-08-15 blandade ihop:
#
#   Linus Kallin (Studentrepresentant)   -> roll
#   Mats Rönnelid (HDa)                  -> organisationstillhörighet
#   Utvärderingsutskotten (UUU)          -> förkortning
#
# Alla tre var korrekta observationer, men de betyder olika saker.
# Sju av de tolv rätta i stickprovet var (HDa) — att räkna dem som
# identiteter gör aggregeringen missvisande: "Joakim Storck -> HDa"
# och "Joakim Storck -> prefekt" är inte samma slags påstående, och
# en rollfråga får inte besvaras med en arbetsplats.
#
# Formen skiljer dem åt utan ordlista: en versalförkortning är en
# förkortning; ett känt organisationssuffix eller en känd
# organisationsform är tillhörighet; övrigt är roll.
_ORG_MARKERS = (
    "universitet", "högskola", "högskolan", "institut", "avdelning",
    "myndighet", "kommun", "region", "ab", "hb", "kb",
)


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
    parts = phrase.split()
    while parts and parts[0].lower() in _FUNCTION_WORDS:
        parts.pop(0)
    return " ".join(parts).strip()


# Verbändelser som markerar finit form i svenska. Listan är
# morfologisk, inte lexikal: den säger något om ordform, inte om vilka
# ord som finns. Riktig morfologi (Kanns Stava/Inflector) skulle
# ersätta den.
_FINITE_ENDINGS = ("as", "ar", "er", "de", "te", "ts", "ades", "ade")


def _looks_predicative(phrase: str) -> bool:
    """Innehåller frasen något som ser ut som ett finit verb?"""
    for tok in re.findall(r"[\wÅÄÖåäö]+", phrase.lower()):
        if len(tok) < 4:
            continue
        if tok.endswith(_FINITE_ENDINGS):
            return True
    return False


def _parenthesis_kind(b: str) -> str:
    """Vilken relation bär parentesen?"""
    stripped = b.strip()
    letters = [c for c in stripped if c.isalpha()]
    # Förkortning: kort, ett enda led, övervägande versaler. Kravet är
    # ÖVERVÄGANDE och inte enbart — "HDa" (Högskolan Dalarna) och
    # "SÄVA" är båda förkortningar trots blandat skiftläge.
    if (
        letters and " " not in stripped and len(stripped) <= 6
        and sum(c.isupper() for c in letters) >= len(letters) / 2
    ):
        return "forkortning"
    if any(m in stripped.lower() for m in _ORG_MARKERS):
        return "tillhorighet"
    return "identitet"


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
        if b.split()[0].lower().rstrip(".") in _DISJUNCTION_MARKERS:
            continue
        if _LEGAL_REF.search(a) or _LEGAL_REF.search(b):
            continue
        if _is_placeholder(a):
            continue
        if _is_code_like(a) or _is_code_like(b):
            continue
        # En roll är en nominalfras. Parentetiska FÖRTYDLIGANDEN
        # innehåller predikat: "(nominering diskuteras i kollegiet)",
        # "(sett till antal HST)", "(prefekt godkänner)". De gav 6 av 9
        # fel i handklassningen 2026-08-15. Samma princip som
        # verbkravet på satser, applicerad på parentesens innehåll —
        # men omvänt: här diskvalificerar predikatet, eftersom en
        # apposition inte påstår något, den benämner.
        if _looks_predicative(b):
            continue
        out.append(Feature(
            kind=_parenthesis_kind(b), a=a, b=b,
            relation="parentes:" + _parenthesis_kind(b).split(":")[-1],
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
