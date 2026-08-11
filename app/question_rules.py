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
