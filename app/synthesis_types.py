"""
Gemensamma datamodeller för syntesvägarna.

Dessa typer delas mellan synthesis.py (huvudvägen) och rework.py
(elaboration och verification). De bor i en egen modul för att
undvika cirkulära beroenden mellan de två.

SynthesisResult är den gemensamma returtypen för alla syntesvägar.
verification-fältet sätts endast av verify() i rework.py — alla
andra syntesvägar lämnar det som None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


VerificationStatus = Literal["supported", "unclear", "unsupported"]


@dataclass
class VerificationFinding:
    """Ett enskilt påstående och dess stöd i källorna."""
    claim: str
    status: VerificationStatus
    source: str | None  # "Källa N" eller None för unsupported-påståenden


@dataclass
class VerificationReport:
    """Strukturerad granskning av tidigare svar mot källorna."""
    findings: list[VerificationFinding] = field(default_factory=list)
    raw_json: str | None = None


@dataclass
class SynthesisResult:
    """
    Gemensam returtyp för synthesize() (huvudvägen) och elaborate()/
    verify() (rework-vägarna).

    verification är satt endast av verify() — för övriga syntesvägar
    är det None.
    """
    answer: str
    verification: VerificationReport | None = None
    used_fallback: bool = False
    fallback_reason: str | None = None
    timing_s: dict = field(default_factory=dict)