"""
Korpuskontroll av rollbindningar i svaret.

Prövar rollbindningar i ett formulerat svar mot vad HELA beståndet
belägger, och kompletterar när svaret bygger på ett svagare belägg än
det bästa som finns.

    Svar:    "Anna Andersson har rollen som prefekt."
    Attest:  Anna Andersson -> prefekt        relevans 0.03, tvetydig, 1 dok
             Anna Andersson -> HR-specialist  relevans 0.86, entydig,  3 dok

    Tillägg: "Beståndet binder Anna Andersson till HR-specialist i tre
              dokument (senast 2025-05-20). Uppgiften om prefekt kommer
              från en källa som tillåter två läsningar."

SKILLNADEN MOT source_guard

source_guard prövar svaret mot de källor som skickades till syntesen.
Den frågan är "är svaret troget sin kontext". Den här modulen frågar
"stämmer svaret med vad beståndet i övrigt säger" — ett svar kan vara
helt troget en enda tvetydig källa och ändå strida mot tjugo entydiga
belägg någon annanstans.

VARFÖR EFTER SYNTESEN OCH INTE I URVALET

Attestsignalen i retrieval påverkar vilka dokument som når syntesen.
Den är död så snart kontexten är låst av andra skäl — uppmätt
2026-08-17 ankrades en följdfråga till ett enda dokument, och signalen
fick aldrig chansen trots att beståndet hade ett starkt belagt svar.
En signal som bara verkar i urvalet är verkningslös just när urvalet
är för snävt, alltså när den behövs mest.

KOMPLETTERAR, SKRIVER INTE OM

Svaret får behålla sitt påstående. Att tyst byta ut en uppgift vore
att låta aggregatet bära svaret, vilket white paper avvisar:
originaltexten bär, aggregatet pekar ut. Läsaren ser båda uppgifterna
med sitt underlag och avgör själv.

BELÄGGNING ÄR INTE SANNING

En felaktig uppgift upprepad i tio protokoll får hög relevans. Därför
formuleras tillägget som vad beståndet SÄGER, aldrig som en rättelse.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Ett tillägg är bara motiverat när skillnaden är tydlig. Marginella
# skillnader i relevans betyder att båda bindningarna är rimliga, och
# då har systemet inget att invända.
MIN_RELEVANCE_GAP = 0.3

# Under detta är bindningen i svaret så svagt belagd att den bör
# kommenteras även om ingen starkare finns.
WEAK_RELEVANCE = 0.15


@dataclass
class BindingCheck:
    person: str
    claimed_role: str
    claimed_relevance: float
    claimed_ambiguous: bool
    claimed_documents: int
    best_role: str | None = None
    best_relevance: float = 0.0
    best_documents: int = 0
    best_last_date: str | None = None
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "person": self.person,
            "claimed_role": self.claimed_role,
            "claimed_relevance": round(self.claimed_relevance, 3),
            "claimed_ambiguous": self.claimed_ambiguous,
            "claimed_documents": self.claimed_documents,
            "best_role": self.best_role,
            "best_relevance": round(self.best_relevance, 3),
            "best_documents": self.best_documents,
            "best_last_date": self.best_last_date,
            "reason": self.reason,
        }


@dataclass
class CorpusReport:
    checked: list[str] = field(default_factory=list)
    findings: list[BindingCheck] = field(default_factory=list)
    ok: bool = True

    def as_dict(self) -> dict:
        return {
            "checked_persons": self.checked,
            "findings": [f.as_dict() for f in self.findings],
            "ok": self.ok,
        }


# Rollbindning i svarstext: "X är/har rollen som/utsågs till Y".
# Mönstret är avsiktligt snävt — det ska hitta EXPLICITA bindningar,
# inte gissa. Missar är ofarliga; falska träffar producerar
# förvirrande tillägg.
# "som X" har företräde framför "är X". I "N.N. är tillsvidareanställd
# SOM professor" är rollen professor, inte tillsvidareanställd —
# uppmätt 2026-08-17 tog uttaget participet och producerade ett
# obegripligt tillägg. Anställningsform, tidsbegränsning och
# rekryteringsstatus står regelmässigt mellan verbet och rollen i
# förvaltningstext.
_BINDING_PATTERNS = [
    re.compile(
        r"\b(?P<person>[A-ZÅÄÖ][a-zåäöé\-]+(?:\s+[A-ZÅÄÖ][a-zåäöé\-]+)+)\s+"
        r"(?:är|var)\s+[^.,;:]{0,40}?\bsom\s+(?:en\s+|den\s+)?"
        r"(?P<role>[a-zåäöA-ZÅÄÖ\-]{4,40})",
    ),
    re.compile(
        r"\b(?P<person>[A-ZÅÄÖ][a-zåäöé\-]+(?:\s+[A-ZÅÄÖ][a-zåäöé\-]+)+)\s+"
        r"har\s+(?:rollen|uppdraget|befattningen)\s+som\s+(?:en\s+|den\s+)?"
        r"(?P<role>[a-zåäöA-ZÅÄÖ\-]{4,40})",
    ),
    # Enkel form sist: bara när ingen "som"-konstruktion matchat.
    re.compile(
        r"\b(?P<person>[A-ZÅÄÖ][a-zåäöé\-]+(?:\s+[A-ZÅÄÖ][a-zåäöé\-]+)+)\s+"
        r"(?:är|var)\s+(?:en\s+|den\s+)?"
        r"(?P<role>[a-zåäöA-ZÅÄÖ\-]{4,40})",
    ),
]

# Ord som ser ut som roller men inte är det i den här konstruktionen.
_NOT_ROLES = {
    "utsedd", "utsett", "föreslagen", "närvarande", "frånvarande",
    "med", "och", "eller", "samt", "inte", "även", "också", "här",
    "kvar", "borta", "ansvarig",
    # Anställningsform och status, inte roll.
    "tillsvidareanställd", "tidsbegränsad", "tidsbegränsat",
    "anställd", "vikarierande", "tillförordnad", "adjungerad",
    "rekryterad", "befordrad", "antagen", "aktuell", "aktuellt",
}


def extract_bindings(answer: str) -> list[tuple[str, str]]:
    """Hitta explicita rollbindningar i svarstexten."""
    found: list[tuple[str, str]] = []
    seen_persons: set[str] = set()
    for i, pattern in enumerate(_BINDING_PATTERNS):
        for m in pattern.finditer(answer):
            person = m.group("person").strip()
            role = m.group("role").strip().lower().rstrip(".,;:")
            if role in _NOT_ROLES or len(role) < 4:
                continue
            # Sista mönstret är den enkla "är X"-formen. Har en
            # "som"-konstruktion redan bundit personen är den mer
            # precis, och den enkla formen skulle bara fånga
            # participet före rollordet.
            if i == len(_BINDING_PATTERNS) - 1 and person in seen_persons:
                continue
            if (person, role) not in found:
                found.append((person, role))
                seen_persons.add(person)
    return found


def check_answer(answer: str, conn=None) -> CorpusReport:
    """
    Pröva svarets rollbindningar mot Attest.

    conn är en öppen sqlite-anslutning; None öppnar en egen. Fel här
    får aldrig fälla ett svar — kontrollen är ett tillägg, inte en
    förutsättning.
    """
    report = CorpusReport()
    bindings = extract_bindings(answer)
    if not bindings:
        return report

    try:
        from app import attest
        if conn is None:
            conn = attest.connect()
    except Exception as e:
        logger.debug("korpuskontroll: Attest otillgängligt (%s)", e)
        return report

    for person, role in bindings:
        try:
            cands = attest.lookup_subject(conn, person)
        except Exception:
            continue
        if not cands:
            continue
        report.checked.append(person)

        # Kandidaterna är rangordnade på relevans.
        best = cands[0]
        claimed = next(
            (c for c in cands if _same_role(c.object, role)), None
        )

        if claimed is None:
            # FRÅNVARO AV BELÄGG ÄR INTE ETT FYND.
            #
            # Att Attest saknar en bindning betyder inte att uppgiften
            # är fel — uttaget missar ungefär vart femte fall, och
            # ingen enskild person har bara en roll. Uppmätt
            # 2026-08-17 gav den tidigare regeln tillägget "uppgiften
            # om tillsvidareanställd har inget motsvarande stöd" om ett
            # korrekt svar, eftersom personen också var studierektor.
            #
            # Kontrollen jämför därför bara belägg systemet FAKTISKT
            # HAR mot varandra. Samma princip som abstain: att inte
            # hitta något är inte ett fynd.
            continue

        gap = best.relevance - claimed.relevance
        weak = claimed.ambiguous_only or claimed.relevance < WEAK_RELEVANCE

        if weak and gap >= MIN_RELEVANCE_GAP and not _same_role(
            best.object, claimed.object
        ):
            report.findings.append(BindingCheck(
                person=person, claimed_role=claimed.object,
                claimed_relevance=claimed.relevance,
                claimed_ambiguous=claimed.ambiguous_only,
                claimed_documents=claimed.documents,
                best_role=best.object, best_relevance=best.relevance,
                best_documents=best.documents,
                best_last_date=best.last_date,
                reason="svagare_an_basta",
            ))
            report.ok = False

    return report


def _same_role(a: str, b: str) -> bool:
    """Jämför roller toleransfritt mot skiftläge och bindestreck."""
    def norm(x: str) -> str:
        return x.lower().replace("-", "").replace(".", "").strip()
    na, nb = norm(a), norm(b)
    return na == nb or na.endswith(nb) or nb.endswith(na)


def format_addition(report: CorpusReport) -> str:
    """
    Formulera tillägget.

    Säger vad beståndet BELÄGGER, aldrig att svaret är fel. Skillnaden
    är inte kosmetisk: en felaktig uppgift upprepad i tio protokoll får
    hög relevans, och systemet vet inte vilken uppgift som är sann —
    bara vilken som är mest belagd.
    """
    if report.ok or not report.findings:
        return ""

    lines: list[str] = []
    for f in report.findings:
        dok = "dokument" if f.best_documents != 1 else "dokument"
        datum = f" (senast {f.best_last_date})" if f.best_last_date else ""
        grund = (
            "en källa som tillåter två läsningar"
            if f.claimed_ambiguous
            else f"{f.claimed_documents} dokument"
        )
        lines.append(
            f"Beståndet binder {f.person} till {f.best_role} i "
            f"{f.best_documents} {dok}{datum}, medan uppgiften om "
            f"{f.claimed_role} vilar på {grund}."
        )

    return "Observera: " + " ".join(lines)

# ---------------------------------------------------------------------------
# Uppräkning av rollinnehavare
# ---------------------------------------------------------------------------

def format_role_holders(candidates, role_term: str, max_rows: int = 12) -> str:
    """
    Sammanställ alla personer som beståndet binder till en roll.

    AGGREGATET BÄR HÄR SVARET, till skillnad från övriga vägar. Det är
    en medveten precisering av white paperns regel, inte ett avsteg:

    Ingen enskild källa innehåller listan — den finns bara som en
    sammanräkning över beståndet. Regelns syfte är att systemet inte
    ska FORMULERA påståenden ur ett aggregat i stället för ur text. En
    uppräkning gör inte det: varje rad är en bindning med sina egna
    källor och datum, och läsaren kan gå till dokumenten. Aggregatet
    organiserar, det påstår inget nytt.

    Villkoret är att underlaget redovisas PER RAD och att listan är
    märkt som en sammanställning, inte som ett citat ur en källa.

    Svaga belägg stryks INTE. Att tyst utesluta en person med ett enda
    gammalt belägg vore att låta systemet avgöra vem som räknas; att
    visa beläggningen låter läsaren göra det.
    """
    if not candidates:
        return ""

    rows: list[str] = []
    for c in candidates[:max_rows]:
        dok = f"{c.documents} dokument"
        datum = f", senast {c.last_date}" if c.last_date else ""
        if c.ambiguous_only:
            flagga = "  (tvetydig källa)"
        elif c.statuses == ["föreslagen"]:
            # Endast förslag: bindningen kan ha bifallits utan att
            # namnet upprepades, eller ha fallit. Beståndet vet inte.
            flagga = "  (endast föreslagen)"
        elif c.statuses:
            flagga = f"  ({', '.join(c.statuses)})"
        else:
            flagga = ""
        rows.append(
            f"- {c.subject} — {c.object} ({dok}{datum}){flagga}"
        )

    fler = ""
    if len(candidates) > max_rows:
        fler = f"\n({len(candidates) - max_rows} till med svagare beläggning.)"

    return (
        f"Sammanställning ur beståndet — personer som bundits till "
        f"{role_term}:\n" + "\n".join(rows) + fler + "\n\n"
        "Listan är sammanräknad ur dokumentens formuleringar och visar "
        "vad beståndet belägger, inte en fastställd förteckning. "
        "Antalet dokument anger hur väl bindningen är belagd."
    )
