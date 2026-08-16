"""
Predikationslager: kopplar grammatiska drag till URD:s begrepp.

`grammar.py` är fri från URD-begrepp. Den här modulen binder ihop drag med
chunkar, frågan och det genererade svaret, och producerar den diagnostik
som steg 0 i utvecklingsplanen ska mäta.

SKUGGLÄGE

I nuvarande version påverkar lagret INGENTING. Det läser, jämför och
skriver till debugspåret. Inga svar ändras, inga flaggor sätts, inga
träffar väljs bort. Aktivering sker i senare steg och först när
mätningen visar att parserkvaliteten håller (acceptanskriterium: ≥90 %
precision på titelappositioner och nsubj/obj i ett stickprov om hundra).

Detta är samma trappa som källvakten gick igenom — men till skillnad från
källvakten byggs lagret på rätt plats från början: i kedjan, med tillgång
till frågan och kandidatpoolen, inte som ett efterhandsskript mot sparade
resultatfiler. Ett efterhandsskript kan bara se det som redan hänt och kan
aldrig påverka urvalet, vilket är där den största vinsten sitter.

VARFÖR DETTA INTE ÄR CLAIMSLAGRET IGEN

Drag utvinns vid frågetillfället ur det material frågan drog fram. Ingen
ny sanning lagras, ingen omindexering krävs, och en ny frågetyp kräver
ingen ny extraktion. Cachen är en cache över parsningar, inte en tabell
över påståenden.
"""

from __future__ import annotations

import logging
import re
import time

from app.config import settings
from app.grammar import (
    Feature,
    extract_features,
    is_available,
    summarize,
)
from app.morphology import is_inflection_of

logger = logging.getLogger(__name__)


# Cache: chunk-fingerprint -> drag. En chunk är statisk mellan
# omindexeringar, så den parsas en gång och läses därefter. Kostnaden är
# hög första gången ett dokument nås och noll därefter.
_CACHE: dict[str, list[Feature]] = {}
_CACHE_MAX = 5000


def _cache_key(hit) -> str:
    md = hit.metadata
    return f"{getattr(md, 'source_fingerprint', '')}:{getattr(md, 'chunk_index', 0)}"


def features_for_hits(hits) -> tuple[list[Feature], dict]:
    """Drag ur källchunkar, med cache och tidsmätning."""
    t0 = time.perf_counter()
    out: list[Feature] = []
    hits_cached = 0
    for hit in hits:
        key = _cache_key(hit)
        cached = _CACHE.get(key)
        if cached is None:
            cached = extract_features(hit.text)
            if len(_CACHE) < _CACHE_MAX:
                _CACHE[key] = cached
        else:
            hits_cached += 1
        out.extend(cached)
    return out, {
        "num_hits": len(hits),
        "num_cached": hits_cached,
        "seconds": round(time.perf_counter() - t0, 3),
    }


# ---------------------------------------------------------------------------
# Jämförelse mellan svarets och källornas drag
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "den", "det", "de", "en", "ett", "som", "vilken", "vilket", "vilka",
    "denna", "detta", "dessa", "han", "hon", "hen", "man", "sitt", "sin",
}


def _content_words(phrase: str) -> set[str]:
    """Innehållsord i en fras: allt utom pronomen och artiklar."""
    return {
        w for w in re.findall(r"[\wÅÄÖåäö\-]+", phrase.lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }


def _terms_match(a: str, b: str) -> bool:
    """
    Termjämförelse på innehållsord, med böjningstolerans.

    SKÄRPT 2026-08-15. Tidigare version accepterade delsträngsmatchning
    i båda riktningarna ("rollen" matchade "rollen som prefekt",
    "beslut" matchade "beslutsmöte"). Med hundratals åtskillnadsdrag i
    kandidatpoolen blev slumpmässiga överlappningar oundvikliga, och
    samtliga tre motsägelser i mätningen var falsklarm — en av dem
    citerade dessutom en källmening utan samband med påståendet.

    Nu krävs att minst ett INNEHÅLLSORD sammanfaller, jämfört ordagrant
    eller som böjningsvariant. Böjningstoleransen behövs för att
    beståndet innehåller "Sara Lundquist" i labbansvarigbeslutet och
    "Sara Lundqvist" som studierektor — en felstavning i källan. URD
    ska återge vad källan säger, inte tyst rätta den, men jämförelsen
    får inte räkna två stavningar som två personer.
    """
    aw, bw = _content_words(a), _content_words(b)
    if not aw or not bw:
        return False
    if aw & bw:
        return True
    return any(
        is_inflection_of(x, y) or is_inflection_of(y, x)
        for x in aw for y in bw
    )


def _pair_match(f: Feature, g: Feature) -> bool:
    """Samma uttryckspar, oavsett ordning."""
    if f.b is None or g.b is None:
        return _terms_match(f.a, g.a)
    return (
        (_terms_match(f.a, g.a) and _terms_match(f.b, g.b))
        or (_terms_match(f.a, g.b) and _terms_match(f.b, g.a))
    )


def _is_supporting(f: Feature, g: Feature) -> bool:
    """
    Bär källdraget g samma påstående som svarsdraget f?

    Jämförelsen är dragspecifik. Ett generellt parmatchningsvillkor
    räcker inte: "ska" och "bör" relaterar samma subjekt till samma verb
    och skulle då räknas som stödjande trots att de säger olika saker om
    kravnivån. Det är precis den förvanskning draget finns för att fånga.
    """
    if f.kind != g.kind:
        return False
    if f.kind == "modalitet":
        return _pair_match(f, g) and f.extra.get("nivå") == g.extra.get("nivå")
    if f.kind == "kvantitet":
        # Kvantitetsdrag paras på talet och dess huvudord, inte på
        # gränsordet — det är gränsordet som ska jämföras.
        return (
            _terms_match(f.b or "", g.b or "")
            and _terms_match(f.relation, g.relation)
            and f.extra.get("riktning") == g.extra.get("riktning")
        )
    return _pair_match(f, g)


def _both_terms_match(f: Feature, g: Feature) -> bool:
    """
    Strikt parmatchning: BÅDA leden måste sammanfalla.

    _pair_match räcker inte för motsägelser. Ett falsklarm i mätningen
    2026-08-15 parade ihop svarets påstående om ett beslutsmötes längd
    med en källmening om prefektens beredning, enbart för att ett led
    överlappade. En motsägelse är en stark flagga och måste vila på att
    källan talar om samma två saker.
    """
    if f.b is None or g.b is None:
        return False
    return (
        (_terms_match(f.a, g.a) and _terms_match(f.b, g.b))
        or (_terms_match(f.a, g.b) and _terms_match(f.b, g.a))
    )


def _contradiction(f: Feature, g: Feature) -> dict | None:
    """
    Bär källdraget g något som är oförenligt med svarsdraget f?

    SKÄRPT 2026-08-15: samtliga tre motsägelser i första mätningen var
    falsklarm. Kraven är nu att båda leden matchar och att källdraget
    har en meningskontext att visa upp — en flagga som citerar fel
    mening är värre än ingen flagga, eftersom felet blir svårt att
    genomskåda.
    """
    if not g.sentence:
        return None
    # Identitet mot åtskillnad: "TB är prefekt" mot "prefekten uppdrog
    # åt HR-specialist TB". Detta är det bekräftade felet 2026-08-14.
    if f.kind == "identitet" and g.kind == "atskillnad" and _both_terms_match(f, g):
        return {"status": "motsagd", "via": g.relation, "kalla": g.sentence}

    # Omvänd agens: "Vice rektor har delegerat till rektor".
    if f.kind == "agens" and g.kind == "agens" and f.relation == g.relation:
        # Kräv att BÅDA dragen har objekt. Ett falsklarm i mätningen
        # utlöstes av ett källdrag utan objekt, ur en mening som
        # klippts mitt i satsen.
        if (
            f.a and f.b and g.a and g.b
            and _terms_match(f.a, g.b) and _terms_match(f.b, g.a)
            and not _terms_match(f.a, g.a)
        ):
            return {"status": "motsagd", "via": "omvänd_agens", "kalla": g.sentence}

    # Kravnivå: svaret säger "ska" där källan säger "bör".
    if f.kind == "modalitet" and g.kind == "modalitet" and _both_terms_match(f, g):
        if f.extra.get("nivå") != g.extra.get("nivå"):
            return {
                "status": "motsagd", "via": "kravnivå",
                "svar_nivå": f.extra.get("nivå"),
                "kalla_nivå": g.extra.get("nivå"),
                "kalla": g.sentence,
            }

    # Gränsriktning: "högst tre" där källan säger "minst tre".
    if f.kind == "kvantitet" and g.kind == "kvantitet":
        if (
            _terms_match(f.b or "", g.b or "")
            and _terms_match(f.relation, g.relation)
            and f.extra.get("riktning") != g.extra.get("riktning")
        ):
            return {
                "status": "motsagd", "via": "gränsriktning",
                "svar_riktning": f.extra.get("riktning"),
                "kalla_riktning": g.extra.get("riktning"),
                "kalla": g.sentence,
            }
    return None


def classify_answer_feature(f: Feature, source_features: list[Feature]) -> dict:
    """
    Klassificera ett drag i svaret mot källornas drag.

    belagd    — källan bär samma drag
    tvetydig  — källan bär draget, men i en konstruktion som tillåter
                mer än en läsning
    motsagd   — källan bär ett oförenligt drag
    obelagd   — ingen källa bär draget

    Motsägelse prövas FÖRE stöd. En källa som håller uttrycken isär väger
    tyngre än en annan källa som råkar para ihop dem, eftersom
    motsägelsen är den starkare signalen och den felklass lagret finns
    för.

    VAKTENS ASYMMETRI (gäller när lagret aktiveras): den får avstå men
    aldrig bekräfta. Ett parsningsfel kostar då ett falsklarm, inte ett
    falskt godkännande — samma felviktning som i övriga systemet. Det
    skyddar också mot cirkularitet när samma parser både väljer in källan
    och granskar svaret.
    """
    for g in source_features:
        contra = _contradiction(f, g)
        if contra:
            return contra

    supporting = [g for g in source_features if _is_supporting(f, g)]
    if supporting:
        # Tvetydighet är information och får inte tystas av att en
        # läsning skulle ge stöd. "Prefekt och HR-expert Anna
        # Andersson presenterade ärendet" tillåter två läsningar — en
        # person med två titlar, eller två personer varav en onämnd.
        # Stanza väljer den första; i det verkliga protokollet avsågs
        # den andra. Tvetydigheten sitter i källan, och ett system som
        # tvingar fram en läsning ur en tvetydig mening är sämre än ett
        # som redovisar att meningen tillåter två.
        #
        # Bär NÅGON stödjande källa tvetydigheten räcker det: svaret
        # vilar då på en mening som inte entydigt binder påståendet.
        ambiguous = [g for g in supporting if g.ambiguous]
        if ambiguous:
            return {
                "status": "tvetydig",
                "via": ambiguous[0].relation,
                "strength": ambiguous[0].strength,
                "kalla": ambiguous[0].sentence,
            }
        g = supporting[0]
        return {"status": "belagd", "via": g.relation, "strength": g.strength}

    return {"status": "obelagd"}


def analyze(question: str, answer: str, hits) -> dict:
    """
    Kör lagret för en tur. Returnerar debugpost, ändrar ingenting.

    Undantag fångas: ett analyslager i skuggläge får aldrig kunna sänka
    en fråga.
    """
    if not settings.predication_enabled:
        return {"enabled": False}
    if not is_available():
        return {"enabled": True, "parser": "saknas"}

    try:
        t0 = time.perf_counter()
        source_features, src_timing = features_for_hits(hits)
        answer_features = extract_features(answer)

        findings = []
        for f in answer_features:
            result = classify_answer_feature(f, source_features)
            findings.append({
                "kind": f.kind, "a": f.a, "b": f.b,
                "relation": f.relation, **result,
            })

        by_status: dict[str, int] = {}
        for x in findings:
            by_status[x["status"]] = by_status.get(x["status"], 0) + 1

        return {
            "enabled": True,
            "parser": "stanza",
            "answer": summarize(answer_features),
            "sources": summarize(source_features),
            "status_counts": by_status,
            "findings": findings[:40],
            "timing_s": {
                "sources": src_timing["seconds"],
                "total": round(time.perf_counter() - t0, 3),
            },
            "cache": {
                "hits_cached": src_timing["num_cached"],
                "hits_total": src_timing["num_hits"],
                "size": len(_CACHE),
            },
        }
    except Exception as e:
        logger.warning("predikation: analys misslyckades (%s)", e)
        return {"enabled": True, "error": str(e)}
