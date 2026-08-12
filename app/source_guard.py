"""
Mekanisk källvakt: deterministisk efterkontroll av huvudsyntesens svar.

Huvudvägens källtrohet vilar annars enbart på prompten, och
verification körs bara när användaren själv prövar ett svar.
Däremellan finns ett tomrum: ingen kontroll av att svaret faktiskt
är belagt i de källor som skickades till syntesen. Fabricerade
siffror är den farligaste felklassen i styrdokumentsvar — och den
är maskinkontrollerbar med ren strängbearbetning.

Två vakter, båda utan LLM-anrop (millisekundnivå):

1. **Siffervakt.** Alla tal i svaret (belopp, årtal, tidsfrister,
   procentsatser) verifieras mot källtexterna som syntesen fick.
   Matchningen är whitespace-normaliserad så att "10 000" belägger
   "10 000:-", "10  000" och "10000". Listmarkörer ("1. ", "2) ")
   och siffror i källhänvisningar ("[Källa 2]") räknas inte som
   faktatal. Ensiffriga tal kontrolleras inte (för brusiga:
   uppräkningar, kapitelnummer, "4 kap. 4 §") — en dokumenterad
   avgränsning i v1.

2. **Källreferensvakt.** Varje [Källa N] i svaret ska referera en
   källa som faktiskt skickades (1 ≤ N ≤ antal källor). Dessutom
   räknas längre stycken utan någon källhänvisning (rapporteras,
   fälls inte).

Vakten står AVSIKTLIGT utanför genereringsvägen: den kan inte
komprimera bort detaljer (tvåstegssyntesens gamla problem) eftersom
den granskar efteråt. Utfallet är trappat: rapport i debug/JSONL
alltid; en synlig varningsrad i svaret när obelagda tal hittas.
Ett strikt läge (omgenerering) kan läggas till senare om mätningen
motiverar det.

Matchningen är medvetet GENERÖS (hellre missa en fabrikation än
larma falskt): substrängmatchning på normaliserade sifferföljder
betyder t.ex. att "100" beläggs av "5100". Vaktens värde är att den
fångar tal som inte förekommer ALLS i underlaget — den vanligaste
och farligaste fabrikationsformen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Whitespace inkl. hårt mellanslag — svenska belopp skrivs "10 000".
_WS_RE = re.compile(r"[\s  ]+")

# Sifferkandidater: siffergrupper eventuellt separerade av
# mellanslag/hårt mellanslag/punkt/komma ("10 000", "1.500", "2,5").
_NUMBER_RE = re.compile(r"\d+(?:[   .,]\d+)*")

# Listmarkörer i början av rad: "1. ", "12) ", "3.2 " — formatering,
# inte faktapåståenden.
_LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+(?:\.\d+)*[.)]?\s+")

# Källhänvisningar: [Källa 2], Källa 12 — siffran är en referens,
# inte ett faktatal.
_CITATION_RE = re.compile(r"\[?\s*[Kk]älla\s*(\d+)\s*\]?")


def _normalize(text: str) -> str:
    """Ta bort all whitespace — gör siffermatchning formatokänslig."""
    return _WS_RE.sub("", text)


@dataclass
class SourceGuardReport:
    numbers_checked: list[str] = field(default_factory=list)
    unsourced_numbers: list[str] = field(default_factory=list)
    citations_used: list[int] = field(default_factory=list)
    citations_out_of_range: list[int] = field(default_factory=list)
    paragraphs_without_citation: int = 0
    ok: bool = True

    def as_dict(self) -> dict:
        return {
            "numbers_checked": self.numbers_checked,
            "unsourced_numbers": self.unsourced_numbers,
            "citations_used": self.citations_used,
            "citations_out_of_range": self.citations_out_of_range,
            "paragraphs_without_citation": self.paragraphs_without_citation,
            "ok": self.ok,
        }


def _extract_fact_numbers(answer: str) -> list[str]:
    """
    Extrahera faktatal ur svaret: alla sifferkandidater utom
    listmarkörer, källhänvisningssiffror och ensiffriga tal.
    Returnerar deduplicerade, whitespace-normaliserade tal i
    förekomstordning.
    """
    cleaned = _CITATION_RE.sub(" ", answer)
    cleaned = _LIST_MARKER_RE.sub(" ", cleaned)

    numbers: list[str] = []
    seen: set[str] = set()
    for match in _NUMBER_RE.finditer(cleaned):
        normalized = _normalize(match.group(0))
        # Ensiffriga tal kontrolleras inte (dokumenterad avgränsning).
        digits_only = re.sub(r"\D", "", normalized)
        if len(digits_only) < 2:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        numbers.append(normalized)
    return numbers


def check_answer(answer: str, source_texts: list[str]) -> SourceGuardReport:
    """
    Kontrollera ett syntessvar mot texterna i de källor syntesen fick.

    Returnerar en rapport; ok=False när obelagda tal eller
    källhänvisningar utanför räckvidd hittats.
    """
    report = SourceGuardReport()

    if not answer or not answer.strip():
        return report

    haystack = _normalize(" ".join(source_texts))

    # 1. Siffervakt
    report.numbers_checked = _extract_fact_numbers(answer)
    for number in report.numbers_checked:
        if number not in haystack:
            report.unsourced_numbers.append(number)

    # 2. Källreferensvakt
    num_sources = len(source_texts)
    for match in _CITATION_RE.finditer(answer):
        ref = int(match.group(1))
        if ref not in report.citations_used:
            report.citations_used.append(ref)
        if not (1 <= ref <= num_sources):
            if ref not in report.citations_out_of_range:
                report.citations_out_of_range.append(ref)

    # Längre stycken utan källhänvisning — rapporteras men fäller inte
    # (att mekaniskt avgöra vad som är ett "faktapåstående" är inte
    # möjligt; räknaren gör mönstret mätbart över tid i JSONL).
    for paragraph in answer.split("\n\n"):
        p = paragraph.strip()
        if len(p) < 120:
            continue
        if p.startswith("Relaterade begrepp"):
            continue
        if not _CITATION_RE.search(p):
            report.paragraphs_without_citation += 1

    report.ok = not report.unsourced_numbers and not report.citations_out_of_range
    return report


def format_warning(report: SourceGuardReport) -> str | None:
    """
    Formatera användarsynlig varning, eller None om ingen behövs.

    Bara obelagda tal ger synlig varning i v1 — referenser utanför
    räckvidd loggas men är oftast formateringsfel snarare än
    fabrikation.
    """
    if not report.unsourced_numbers:
        return None

    shown = ", ".join(report.unsourced_numbers[:5])
    if len(report.unsourced_numbers) > 5:
        shown += ", …"
    return (
        f"⚠ Källvakt: uppgiften {shown} i svaret kunde inte beläggas i de "
        "visade källorna — kontrollera mot originaldokumenten."
    )
