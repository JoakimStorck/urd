"""
Validering av instansens konfigurationsfiler i .urd/.

Bakgrund: loaders för synonyms.yaml, concepts.yaml och
question_operations.yaml har medvetet tyst fallback — URD ska kunna
köra utan dem. Men tyst fallback utan synlig status betyder att ett
skrivfel i en YAML-fil stänger av en hel funktion utan att någon
märker det. Exakt det hände: en rad med två listposter gjorde hela
synonymlistan oladdbar, och en labels-post som råkade vara en sträng
i stället för en lista gjorde ett begrepp dött.

Denna modul gör felen synliga på tre ställen:

1. `urd config validate` — kör validering och skriv en rapport.
2. Serverns startlogg — en rad per fil med status.
3. `/health` — samma status som JSON, så att en klient (eller
   `urd connect`) kan se att instansens konfiguration är hel.

Modulen är avsiktligt fri från beroenden på resten av appen
(ingen import av app.config eller tunga modeller) så att den kan
köras fristående och testas utan att dra igång embeddings/Qdrant.
Valideringen speglar de faktiska loadernas förväntningar — den ska
ge fel/varningar för precis de strukturer som loadern skulle
tappa bort eller misstolka.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Kända frågeoperationer — hålls i synk med intent.QuestionOperation
# och intent._VALID_OPERATIONS. Definieras lokalt (inte importerat) för
# att hålla modulen fri från appberoenden.
#
# LISTAN SLÄPAR EFTER OM DEN INTE UNDERHÅLLS, och det är just den
# felformen som redan kostat en gång: en operation som saknades i
# intent._VALID_OPERATIONS avvisades tyst med fallback till
# direct_lookup, med följden att en signal aldrig aktiverades trots att
# regeln satte rätt operation. Här är utfallet mildare — valideringen
# varnar bara — men en falsk varning om en operation som faktiskt
# fungerar lär användaren att ignorera varningarna, vilket är samma
# skada i långsam form.
KNOWN_OPERATIONS = {
    "direct_lookup",
    "entity_lookup",
    "entity_aggregation",
    "relation_membership",
    "comparison",
    "requirements",
    "process",
    "aggregation",
}


@dataclass
class FileReport:
    """Valideringsresultat för en konfigurationsfil."""

    name: str                 # t.ex. "synonyms"
    path: str
    status: str               # "ok" | "missing" | "error"
    summary: str              # kort människoläsbar rad
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "status": self.status,
            "summary": self.summary,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class ValidationReport:
    files: list[FileReport]

    @property
    def ok(self) -> bool:
        return all(f.status != "error" for f in self.files)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "files": [f.as_dict() for f in self.files],
        }


# ---------------------------------------------------------------------------
# Hjälpfunktioner
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> tuple[object | None, str | None]:
    """
    Läs och parsa en YAML-fil.

    Returnerar (data, felmeddelande). Vid parsefel är data None och
    felmeddelandet innehåller radinformation från parsern — det är
    just radnumret som gör felet hittbart.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        return None, "PyYAML är inte installerat — filen kan inte läsas."

    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f), None
    except yaml.YAMLError as e:  # type: ignore[attr-defined]
        return None, f"YAML-parsefel: {e}"
    except OSError as e:
        return None, f"Kunde inte läsa filen: {e}"


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


# ---------------------------------------------------------------------------
# Validering per fil
# ---------------------------------------------------------------------------

def validate_synonyms_file(path: Path) -> FileReport:
    name = "synonyms"
    if not path.exists():
        return FileReport(
            name=name, path=str(path), status="missing",
            summary="fil saknas — kör utan synonymexpansion (ofarligt)",
        )

    data, load_error = _load_yaml(path)
    if load_error:
        return FileReport(
            name=name, path=str(path), status="error",
            summary="kunde inte parsas — ALL synonymexpansion är avstängd",
            errors=[load_error],
        )

    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict):
        errors.append("Roten ska vara ett objekt med nyckeln 'synonyms'.")
        return FileReport(
            name=name, path=str(path), status="error",
            summary="fel struktur — ALL synonymexpansion är avstängd",
            errors=errors,
        )

    raw_groups = data.get("synonyms")
    if not isinstance(raw_groups, list):
        errors.append("'synonyms' ska vara en lista av grupper.")
        return FileReport(
            name=name, path=str(path), status="error",
            summary="fel struktur — ALL synonymexpansion är avstängd",
            errors=errors,
        )

    valid_groups = 0
    seen_groups: set[tuple[str, ...]] = set()
    for i, group in enumerate(raw_groups):
        if not isinstance(group, list):
            errors.append(
                f"Grupp {i + 1} är inte en lista (fick {type(group).__name__}). "
                "Kontrollera att varje grupp står på en egen rad: '- [term1, term2]'."
            )
            continue
        terms = [str(t).strip() for t in group if str(t).strip()]
        if len(terms) < 2:
            warnings.append(
                f"Grupp {i + 1} ({terms or group}) har färre än två termer "
                "— den ger ingen expansion och hoppas över av loadern."
            )
            continue
        key = tuple(sorted(t.casefold() for t in terms))
        if key in seen_groups:
            warnings.append(f"Grupp {i + 1} ({terms}) är en dubblett av en tidigare grupp.")
        seen_groups.add(key)
        valid_groups += 1

    status = "error" if errors else "ok"
    summary = f"{valid_groups} synonymgrupper"
    if errors:
        summary += " — filen har fel som gör att grupper tappas"
    return FileReport(
        name=name, path=str(path), status=status,
        summary=summary, errors=errors, warnings=warnings,
    )


def validate_concepts_file(path: Path) -> FileReport:
    name = "concepts"
    if not path.exists():
        return FileReport(
            name=name, path=str(path), status="missing",
            summary="fil saknas — kör utan begreppsmodell (ofarligt)",
        )

    data, load_error = _load_yaml(path)
    if load_error:
        return FileReport(
            name=name, path=str(path), status="error",
            summary="kunde inte parsas — begreppsmodellen är avstängd",
            errors=[load_error],
        )

    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict) or not isinstance(data.get("concepts"), list):
        errors.append("Roten ska vara ett objekt med nyckeln 'concepts' (lista).")
        return FileReport(
            name=name, path=str(path), status="error",
            summary="fel struktur — begreppsmodellen är avstängd",
            errors=errors,
        )

    concept_ids: set[str] = set()
    all_broader: list[tuple[str, str]] = []  # (concept_id, broader_id)
    valid_concepts = 0

    for i, item in enumerate(data["concepts"]):
        label = f"begrepp {i + 1}"
        if not isinstance(item, dict):
            errors.append(f"{label}: ska vara ett objekt med 'id' och 'labels'.")
            continue

        concept_id = str(item.get("id", "")).strip()
        if not concept_id:
            errors.append(f"{label}: saknar 'id' — hoppas över av loadern.")
            continue
        label = f"begrepp '{concept_id}'"

        if concept_id in concept_ids:
            warnings.append(f"{label}: id förekommer flera gånger.")
        concept_ids.add(concept_id)

        labels = item.get("labels")
        if isinstance(labels, str):
            # Det farligaste felet: loadern itererar strängen tecken
            # för tecken och begreppet blir tyst obrukbart.
            errors.append(
                f"{label}: 'labels' är en sträng, inte en lista. "
                "Skriv varje label som en egen listpost: '- term'. "
                "Som sträng blir begreppet obrukbart (itereras tecken för tecken)."
            )
            continue
        if not _is_string_list(labels) or not labels:
            errors.append(f"{label}: 'labels' ska vara en icke-tom lista av strängar.")
            continue

        broader = item.get("broader", [])
        if broader and not _is_string_list(broader):
            errors.append(f"{label}: 'broader' ska vara en lista av begrepps-id:n.")
            continue
        for b in broader or []:
            all_broader.append((concept_id, str(b).strip()))

        valid_concepts += 1

    for concept_id, broader_id in all_broader:
        if broader_id not in concept_ids:
            warnings.append(
                f"begrepp '{concept_id}': broader-referensen '{broader_id}' "
                "matchar inget definierat begrepp."
            )

    status = "error" if errors else "ok"
    summary = f"{valid_concepts} begrepp"
    if errors:
        summary += " — filen har fel som gör att begrepp tappas"
    return FileReport(
        name=name, path=str(path), status=status,
        summary=summary, errors=errors, warnings=warnings,
    )


def validate_question_operations_file(path: Path) -> FileReport:
    name = "question_operations"
    if not path.exists():
        return FileReport(
            name=name, path=str(path), status="missing",
            summary="fil saknas — kör utan operationsstyrd expansion (ofarligt)",
        )

    data, load_error = _load_yaml(path)
    if load_error:
        return FileReport(
            name=name, path=str(path), status="error",
            summary="kunde inte parsas — operationsstyrd expansion är avstängd",
            errors=[load_error],
        )

    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict) or not isinstance(data.get("operations"), dict):
        errors.append("Roten ska vara ett objekt med nyckeln 'operations' (objekt).")
        return FileReport(
            name=name, path=str(path), status="error",
            summary="fel struktur — operationsstyrd expansion är avstängd",
            errors=errors,
        )

    valid_ops = 0
    for op_name, op_data in data["operations"].items():
        op_name = str(op_name).strip()
        if op_name not in KNOWN_OPERATIONS:
            warnings.append(
                f"operation '{op_name}' är inte en känd frågeoperation "
                f"({', '.join(sorted(KNOWN_OPERATIONS))}) — den kommer aldrig att användas."
            )
        if not isinstance(op_data, dict):
            errors.append(f"operation '{op_name}': ska vara ett objekt — hoppas över av loadern.")
            continue
        for field_name in ("expansion_terms", "preferred_section_terms"):
            value = op_data.get(field_name, [])
            if value and not _is_string_list(value):
                errors.append(
                    f"operation '{op_name}': '{field_name}' ska vara en lista av strängar."
                )
        valid_ops += 1

    status = "error" if errors else "ok"
    summary = f"{valid_ops} operationer"
    return FileReport(
        name=name, path=str(path), status=status,
        summary=summary, errors=errors, warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Samlad validering
# ---------------------------------------------------------------------------

def validate_config_files(
    synonyms_path: Path,
    concepts_path: Path,
    question_operations_path: Path,
) -> ValidationReport:
    """Validera instansens tre YAML-konfigfiler."""
    return ValidationReport(files=[
        validate_synonyms_file(synonyms_path),
        validate_concepts_file(concepts_path),
        validate_question_operations_file(question_operations_path),
    ])


def format_report_lines(report: ValidationReport) -> list[str]:
    """
    Formatera rapporten som människoläsbara rader — används både av
    CLI-kommandot och serverns startlogg.
    """
    lines: list[str] = []
    status_labels = {"ok": "OK", "missing": "SAKNAS", "error": "FEL"}
    for f in report.files:
        lines.append(f"[{status_labels.get(f.status, f.status)}] {f.name} ({f.path}): {f.summary}")
        for e in f.errors:
            lines.append(f"    fel: {e}")
        for w in f.warnings:
            lines.append(f"    varning: {w}")
    return lines
