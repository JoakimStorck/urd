"""
Regelbaserad förklassificering av frågeoperation.

LLM-klassificeraren missar frågeoperationer även i entydiga fall —
uppmätt 2026-08-11: "Vilka labbansvariga finns på IIT?" klassades som
direct_lookup trots att mönstret "vilka … finns" står ordagrant i
klassificerarens instruktion, med följden att aggregeringsblockets
"sammanställ till komplett lista"-instruktion aldrig aktiverades och
svaret blev ofullständigt.

För operationer med entydiga språkliga markörer avgör därför
deterministiska regler FÖRE LLM:en. Reglerna är avsiktligt smala:
bara mönster där operationen är otvetydig. Allt annat lämnas till
LLM-klassificeraren som förut. Intent (samtalsrelationen) berörs
inte — reglerna gäller enbart frågeoperationen, som är en separat
axel.

Regelträffar loggas i debug (classification.operation_source) så att
regelfel är diagnosbara i JSONL-spåren.
"""

from __future__ import annotations

import re

# Ordningen spelar roll: comparison prövas före aggregation så att
# "vilka skillnader finns mellan X och Y" blir comparison, inte
# aggregation.
_OPERATION_RULES: list[tuple[str, re.Pattern]] = [
    (
        # "Vilka professorer finns vid IIT?" — frågan efter ALLA
        # innehavare av en roll. Skiljer sig från entity_lookup i att
        # svaret är en mängd, och från aggregation i att mängden består
        # av personer bundna till en roll.
        #
        # Ingen enskild källa innehåller listan; den finns bara som en
        # sammanräkning över beståndet. Prövas FÖRE entity_lookup, som
        # annars fångar "vilka är studierektorer".
        "entity_aggregation",
        re.compile(
            r"^\s*vilka\s+\w+(?:er|ar|or|ter|na)\b[^?]*\b(?:finns|är|har)\b"
            r"|^\s*vilka\s+(?:är|arbetar|tjänstgör)\s+(?:som\s+)?\w{4,}",
            re.IGNORECASE,
        ),
    ),
    (
        # "Vem är proprefekt?" — frågan efter en NAMNGIVEN INNEHAVARE.
        # Skiljer sig från direct_lookup i vad som efterfrågas: inte
        # rollens innehåll utan vem som bär den. Cross-encodern kan
        # inte göra den skillnaden — den mäter aboutness, och på
        # "vem är proprefekt" handlar varje kandidatpassage om
        # proprefekten. Se attestsignalen i retrieval.
        #
        # "vem/vilka" följt av vara/ha/ansvara fångar formen utan att
        # veta något om vilka roller som finns.
        "entity_lookup",
        re.compile(
            r"^\s*vem\b"
            r"|^\s*vilka\s+(?:är|har|innehar|ansvarar|sitter)\b"
            r"|\bvem (?:är|har|innehar|ansvarar|utses|utsågs)\b"
            # Frågan åt andra hållet: från person till roll. "Vilken
            # roll har X", "vilket uppdrag har X". Uppmätt 2026-08-17
            # klassades den formen inte som entity_lookup, med följden
            # att signalen aldrig aktiverades och svaret byggdes på en
            # enda tvetydig källa.
            # "har/innehar/hade" krävs: "vilken roll SPELAR forskningen"
            # är ingen entitetsfråga.
            r"|\bvilk(?:en|et)\s+(?:roll|uppdrag|befattning|titel|funktion)"
            r"\s+(?:har|innehar|hade|innehade)\b"
            r"|\bvad (?:är|har) .{0,40}\bför (?:roll|uppdrag|titel)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "comparison",
        re.compile(
            r"\bskillnad(?:en|er|erna)?\b"
            r"|\bjämför\b"
            r"|\bjämfört med\b"
            r"|\bhur skiljer\b",
            re.IGNORECASE,
        ),
    ),
    (
        "aggregation",
        re.compile(
            # "Vilka X finns ..." — uppräkningsfrågans grundform
            r"^\s*vilka\b[^?]*\bfinns\b"
            # "vilka typer/kategorier/sorter av ..."
            r"|\bvilka (?:typer|kategorier|sorter)\b"
            # "lista alla/upp ..."
            r"|\blista (?:alla|upp)\b",
            re.IGNORECASE,
        ),
    ),
]


def rule_based_operation(question: str) -> str | None:
    """
    Returnera frågeoperationen om ett entydigt mönster matchar,
    annars None (LLM-klassificeraren avgör).
    """
    if not question or not question.strip():
        return None

    for operation, pattern in _OPERATION_RULES:
        if pattern.search(question):
            return operation

    return None
