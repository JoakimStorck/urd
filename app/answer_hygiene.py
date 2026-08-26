"""
Deterministisk efterbearbetning av svarstext: likalydande meningar.

VARFÖR MEKANISM OCH INTE PROMPT. Två promptomgångar (0064, 0066) har
försökt lära modellen att slå ihop belägg som säger samma sak. Båda
misslyckades på samma felklass: ett innehavarsvar upprepade "Enligt
[Källa N] (datum) är X proprefekt" fem gånger, en gång per källa. Det
är inte fem uppgifter utan en uppgift med fem belägg. Enligt systemets
egen arbetsregel är två misslyckade promptvarv signalen att beskriva
felets form i stället för att formulera om instruktionen en tredje
gång.

MEKANISMEN FANNS REDAN TILL HÄLFTEN. rework.py klipper stycken vars
6-gram-shingles till övervägande del återfinns i FÖREGÅENDE svar, och
dess kända lucka har varit upprepning INOM samma svar. Detta är den
saknade halvan — med en skillnad: här klipps inte, här SLÅS IHOP.
Källhänvisningarna är svarets verifierbarhet och får aldrig försvinna
för att texten städas.

VILLKORET ÄR STRIKT LIKHET EFTER AVSKALNING. Två meningar slås ihop
endast när de är identiska sedan källhänvisningar, datum och
interpunktion tagits bort. Det är trubbigt med flit: ett lager som
skriver om svarstext ska hellre missa en upprepning än slå ihop två
meningar som säger olika saker. Nära-likhet med tröskel vore lätt att
införa och svår att lita på.

SPANNET ÄR EN UPPLYSNING, inte bara en hopslagning. När flera belägg
bär samma bindning säger tidigaste och senaste datum något om
bindningens varaktighet, vilket är mer upplysande än en rad lösryckta
datum.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Systemets egen citatnotation och ISO-datum. Båda är formaliserade och
# kan därför skalas av utan att meningens innehåll rörs.
_CITATION = re.compile(r"\[\s*Källa[^\]]*\]", re.IGNORECASE)
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_SOURCE_NUM = re.compile(r"Källa\s*(\d+)", re.IGNORECASE)
# Meningsgräns: punkt, utrop eller fråga följt av blanksteg och versal.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÅÄÖ])")


def _skeleton(sentence: str) -> str:
    """Meningen utan källhänvisningar, datum och interpunktion."""
    s = _CITATION.sub(" ", sentence)
    s = _ISO_DATE.sub(" ", s)
    s = re.sub(r"[^\wåäöÅÄÖ]+", " ", s, flags=re.UNICODE)
    return " ".join(s.casefold().split())


def _source_numbers(sentence: str) -> list[int]:
    return [int(m.group(1)) for m in _SOURCE_NUM.finditer(sentence)]


def merge_repeated_sentences(answer: str) -> tuple[str, int]:
    """
    Slå ihop meningar som säger samma sak.

    Returnerar (bearbetat svar, antal hopslagna meningar). Den första
    förekomsten behålls på sin plats — svarets ordning är syntesens
    bedömning av vad som är viktigast och ska inte kastas om här — och
    får de övrigas källhänvisningar och datumspann. Följande
    förekomster tas bort.

    Ett svar utan upprepning återlämnas oförändrat, och räknaren blir
    noll. Lagret är rent additivt i den meningen: det kan aldrig göra
    ett korrekt svar sämre än att en mening blir längre.
    """
    text = (answer or "").strip()
    if not text:
        return answer, 0

    # Endast inom stycken; styckegränser bär struktur (rubriker,
    # punktlistor) som inte ska slås ihop över.
    stycken = re.split(r"(\n\s*\n)", text)
    ut: list[str] = []
    sammanslagna = 0

    for block in stycken:
        if not block.strip() or block.startswith("\n"):
            ut.append(block)
            continue
        meningar = _SENTENCE_SPLIT.split(block)
        if len(meningar) < 2:
            ut.append(block)
            continue

        forst: dict[str, int] = {}
        behall: list[str | None] = list(meningar)
        extra_kallor: dict[int, list[int]] = {}
        extra_datum: dict[int, list[str]] = {}

        for i, mening in enumerate(meningar):
            nyckel = _skeleton(mening)
            if not nyckel:
                continue
            if nyckel not in forst:
                forst[nyckel] = i
                continue
            j = forst[nyckel]
            extra_kallor.setdefault(j, []).extend(_source_numbers(mening))
            extra_datum.setdefault(j, []).extend(
                m.group(1) for m in _ISO_DATE.finditer(mening)
            )
            behall[i] = None
            sammanslagna += 1
            logger.info(
                "Sammanslagning: likalydande mening togs bort (%r)",
                mening[:100],
            )

        for j, nya_kallor in extra_kallor.items():
            behall[j] = _merge_into(
                meningar[j], nya_kallor, extra_datum.get(j, []),
            )
        ut.append(" ".join(m for m in behall if m is not None))

    return "".join(ut), sammanslagna


def _merge_into(sentence: str, extra_sources: list[int],
                extra_dates: list[str]) -> str:
    """Ge den behållna meningen de borttagnas källor och datumspann."""
    alla_kallor = sorted(set(_source_numbers(sentence)) | set(extra_sources))
    alla_datum = sorted(set(
        [m.group(1) for m in _ISO_DATE.finditer(sentence)] + extra_dates
    ))

    resultat = sentence
    if alla_kallor:
        ersattning = "[" + ", ".join(f"Källa {n}" for n in alla_kallor) + "]"
        resultat = _CITATION.sub(ersattning, resultat, count=1)
    if len(alla_datum) > 1:
        # Platshållare först: annars träffar städningen av övriga datum
        # spannets egen slutpunkt och lämnar "2023-01-31 till".
        # Provet fällde precis den ordningen.
        spann = f"{alla_datum[0]} till {alla_datum[-1]}"
        resultat = _ISO_DATE.sub("\x00SPANN\x00", resultat, count=1)
        resultat = _ISO_DATE.sub("", resultat)
        resultat = resultat.replace("\x00SPANN\x00", spann)
    return re.sub(r"\s{2,}", " ", resultat).strip()


# ---------------------------------------------------------------------------
# Bindningssammanställning
# ---------------------------------------------------------------------------

def format_bindings(bindings: list[dict], hits, frame_year: int | None = None) -> str:
    """
    Beståndets bindningar i sammanställd form, för syntesprompten.

    VARFÖR SAMMANSTÄLLNINGEN GES FÄRDIG. Uppslaget har redan grupperat
    observationerna per person och roll, räknat dokument och beräknat
    spann. Att skicka enbart råtext och hoppas att modellen gör om den
    grupperingen är att be den härleda något systemet redan vet — och
    uppmätt 2026-08-26 gör den det dåligt: tre belägg om samma person
    blev tre meningar på olika detaljnivå i stället för en.

    AGGREGATET BÄR INTE SVARET. Sammanställningen är ett läshjälpmedel
    härlett ur passagerna som ändå skickas med; originaltexten bär
    fortfarande påståendet, och prompten säger det uttryckligen. Detta
    är samma hållning som reservationskanalen: Attest pekar ut var
    bindningen står, syntesen formulerar ur källan.

    Källhänvisningarna beräknas mot den FAKTISKA numreringen i
    hits_for_synthesis, så att raden pekar på samma [Källa N] som
    källblocket. En bindning vars belägg inte finns bland källorna tas
    inte med — den kan inte verifieras av läsaren och hör därför inte
    hemma i svarets underlag.
    """
    if not bindings:
        return ""
    index_by_name: dict[str, list[int]] = {}
    for i, h in enumerate(hits, start=1):
        index_by_name.setdefault(h.metadata.file_name, []).append(i)

    rader: list[str] = []
    for b in bindings:
        if frame_year and (b.get("first_date") or "")[:4].isdigit():
            if int(b["first_date"][:4]) > frame_year:
                continue
        nr: list[int] = []
        for namn in b.get("sources", []):
            nr.extend(index_by_name.get(namn, []))
        if not nr:
            continue
        rad = f"- {b['subjekt']} — {b['roll']}"
        if b.get("avser"):
            rad += f" (avser {b['avser']})"
        forst, sist = b.get("first_date"), b.get("last_date")
        if forst and sist and forst != sist:
            rad += f", belagt {forst} till {sist}"
        elif forst or sist:
            rad += f", belagt {forst or sist}"
        rad += " [" + ", ".join(f"Källa {n}" for n in sorted(set(nr))) + "]"
        rader.append(rad)

    if not rader:
        return ""
    return (
        "BINDNINGAR SOM BESTÅNDET UTTRYCKER (sammanställda ur källorna "
        "nedan — formulera svaret ur källtexten, inte ur denna lista):\n"
        + "\n".join(rader)
    )
