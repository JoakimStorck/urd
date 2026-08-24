"""
Vad påstår svaret? Bindningar utvunna ur systemets egen text.

Deliberationens prövningssteg (white paper 3.0) jämför svarets
påståenden med turens åtagande. Detta är utvinningen — den läser
svaret, inte beståndet, och den är TYST: den avgör ingenting.

VARFÖR GRAMMATIK OCH INTE STRÄNGAR. Testbatteriets negativa assertioner
prövar förekomst av strängar, och en uppmätt tur passerade sin
förbudslista samtidigt som felet inträffade: förbudet gällde namnet,
och svaret skrev "han". Ett fält som mäter strängar kan inte bära ett
påstående om bindning. Samma brist finns i källvakten, som vaktar tal
men inte formuleringar.

KOPULA ÄR RÄTT HÄR MEN FEL I BESTÅNDET. Konstruktionen togs medvetet
bort ur beståndsutvinningen: den kan inte skilja identitet från allmän
predikation utan att veta vad orden betyder ("samarbete är en viktig
del"). För svarstexten är uppgiften en annan — inte om påståendet är
sant, utan vad som påstås och om påståendet är KONTROLLERBART. Det
kräver ingen ordkunskap, bara subjektets art.

UTVINNINGEN SER ALLT, BEVAKNINGEN VÄLJER. Modulen tar varje nominal
predikation den hittar och märker subjektets art. Vilka bindningar som
sedan ska prövas strängt är ett senare beslut, styrt av operationen och
åtagandet. Att bygga in en lista över bevakade roller redan i
utvinningen vore att göra mekanismen instansformad — den skulle bara se
det vi råkat leta efter.

OVERIFIERBAR ÄR EN EGEN KATEGORI, inte ett fel. "Han är prefekt" är ett
bindningspåstående som ingen korpuskontroll kan pröva, eftersom
referenten ligger utanför satsen. Att märka det är hela poängen: en
overifierbar bindning på en bevakad roll är skäl nog att flagga, utan
att koreferens behöver lösas.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app import grammar

# Källhänvisningar är systemets egen notation och ska inte parsas.
_SOURCE_REF = re.compile(r"\[\s*Källa[^\]]*\]", re.IGNORECASE)

# Subjektets art avgör om påståendet går att kontrollera mot beståndet.
SUBJEKT_NAMN = "namn"           # PROPN — kontrollerbart
SUBJEKT_PRONOMEN = "pronomen"   # PRON — referenten ligger utanför satsen
SUBJEKT_APPELLATIV = "appellativ"  # vanligt substantiv

_MODIFIER_DEPRELS = (
    "flat", "flat:name", "compound", "amod", "nmod", "case", "det",
)


@dataclass(frozen=True)
class Bindning:
    """
    Ett påstående i svaret som binder ett uttryck till ett annat.

    subjekt/predikat  de två leden, som de står i svaret
    subjekt_art       namn | pronomen | appellativ
    konstruktion      vilken form som bar påståendet
    verifierbar       falskt när referenten inte finns i satsen
    mening            för spårbarhet
    """
    subjekt: str
    predikat: str
    subjekt_art: str
    konstruktion: str
    verifierbar: bool
    mening: str

    def as_debug(self) -> dict:
        return {
            "subjekt": self.subjekt,
            "predikat": self.predikat,
            "subjekt_art": self.subjekt_art,
            "konstruktion": self.konstruktion,
            "verifierbar": self.verifierbar,
        }


def strip_source_refs(text: str) -> str:
    return _SOURCE_REF.sub("", text)


def _phrase(word, barn: dict) -> str:
    """
    Ordet med sitt modifierarled, i ordföljd.

    REKURSIVT över tillåtna relationer. En ytlig hämtning av direkta
    barn räcker inte: i "Prefekt vid IIT" hänger prepositionen på IIT
    och inte på huvudordet, så frasen blev "Prefekt IIT" — läsbart men
    fel, och fel på ett sätt som inte syns förrän man jämför med
    källtexten.

    Endast nominella modifierare följs. Bisatser och samordningar
    lämnas utanför, så att frasen förblir en fras och inte växer till
    hela satsen.
    """
    samlade: dict[int, object] = {word.id: word}

    def följ(w) -> None:
        for b in barn.get(w.id, []):
            if b.deprel in _MODIFIER_DEPRELS and b.id not in samlade:
                samlade[b.id] = b
                följ(b)

    följ(word)
    led = sorted(samlade.values(), key=lambda w: w.id)
    return " ".join(w.text for w in led).strip()


def _subject_kind(word) -> str:
    if word.upos == "PRON":
        return SUBJEKT_PRONOMEN
    if word.upos == "PROPN":
        return SUBJEKT_NAMN
    return SUBJEKT_APPELLATIV


def extract_bindings(answer: str) -> list[Bindning]:
    """
    Bindningspåståenden i ett svar.

    Två källor: nominala predikationer med kopula ur parsen, och de
    konstruktioner beståndsutvinningen redan känner igen (titel,
    apposition, tillsättning) tillämpade på svarstexten.

    Utan parser returneras det grammatikmodulen ändå kan ge. Lagret är
    tyst och får aldrig kunna sänka en fråga — samma regel som gäller
    predikationslagret i övrigt.
    """
    text = strip_source_refs(answer or "").strip()
    if not text:
        return []

    ut: list[Bindning] = []
    sedda: set[tuple[str, str]] = set()

    def lagg(b: Bindning) -> None:
        nyckel = (b.subjekt.lower(), b.predikat.lower())
        if nyckel in sedda:
            return
        sedda.add(nyckel)
        ut.append(b)

    pipeline = grammar._get_pipeline()
    if pipeline is not None:
        try:
            doc = pipeline(text)
        except Exception:
            doc = None
        if doc is not None:
            for mening in doc.sentences:
                barn: dict[int, list] = {}
                for w in mening.words:
                    barn.setdefault(w.head, []).append(w)
                for head in mening.words:
                    if head.upos not in ("NOUN", "PROPN", "ADJ"):
                        continue
                    kopula = [b for b in barn.get(head.id, [])
                              if b.deprel == "cop"]
                    subj = [b for b in barn.get(head.id, [])
                            if b.deprel in ("nsubj", "nsubj:pass")]
                    if not kopula or not subj:
                        continue
                    s = subj[0]
                    art = _subject_kind(s)
                    lagg(Bindning(
                        subjekt=_phrase(s, barn),
                        predikat=_phrase(head, barn),
                        subjekt_art=art,
                        konstruktion="kopula",
                        # ETT PRONOMENSUBJEKT GÖR PÅSTÅENDET
                        # OKONTROLLERBART: referenten ligger utanför
                        # satsen, och ingen korpuskontroll kan nå den.
                        verifierbar=art != SUBJEKT_PRONOMEN,
                        mening=mening.text,
                    ))

    # Konstruktionerna beståndsutvinningen redan känner igen. De
    # förutsätter namnlika led och ger därför bara verifierbara
    # bindningar — men de fångar former kopulan missar, som
    # "Prefekt N.N. beslutade" och "N.N. (sekreterare)".
    for f in grammar.extract_features(text):
        if f.kind != "identitet" or not f.b:
            continue
        lagg(Bindning(
            subjekt=f.a, predikat=f.b,
            subjekt_art=SUBJEKT_NAMN,
            konstruktion=f.relation,
            verifierbar=True,
            mening=f.sentence,
        ))
    return ut


def summarize(bindningar: list[Bindning]) -> dict:
    """Aggregat för debug/JSONL — utan att bära texten vidare."""
    return {
        "antal": len(bindningar),
        "antal_overifierbara": sum(1 for b in bindningar if not b.verifierbar),
        "konstruktioner": sorted({b.konstruktion for b in bindningar}),
        "bindningar": [b.as_debug() for b in bindningar],
    }
