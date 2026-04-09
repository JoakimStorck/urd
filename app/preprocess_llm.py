import json
import re
import ollama

from app.config import settings
from app.schemas import SectionSemanticMetadata
from app.llm import LLMUnavailableError


SYSTEM_PROMPT = """Du extraherar strukturerad metadata ur interna styrdokument.

Returnera endast giltig JSON.
Använd endast information som uttryckligen stöds i texten.
Hitta inte på roller, tidsfrister eller begrepp.
Om något saknas, använd null eller tom lista.

Schema:
{
  "document_type": string | null,
  "keywords": [string],
  "roles": [string],
  "actions": [string],
  "time_markers": [string],
  "applies_to": [string],
  "summary": string | null
}

Regler:
- Använd svenska
- Högst 8 keywords
- Högst 8 roles
- Högst 8 actions
- Högst 6 time_markers
- Högst 6 applies_to
- summary max 40 ord
- Ta bara med sådant som tydligt stöds i texten
"""


def _normalize_list(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        s = re.sub(r"\s+", " ", item.strip())
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _normalize_semantic(meta: SectionSemanticMetadata) -> SectionSemanticMetadata:
    return SectionSemanticMetadata(
        document_type=meta.document_type.strip() if meta.document_type else None,
        keywords=_normalize_list(meta.keywords),
        roles=_normalize_list(meta.roles),
        actions=_normalize_list(meta.actions),
        time_markers=_normalize_list(meta.time_markers),
        applies_to=_normalize_list(meta.applies_to),
        summary=meta.summary.strip() if meta.summary else None,
    )


class SectionMetadataExtractor:
    def __init__(self) -> None:
        self.model = settings.preprocess_ollama_model

    def extract(
        self,
        document_title: str | None,
        section_title: str | None,
        text: str,
    ) -> SectionSemanticMetadata:
        clipped = text[: settings.preprocess_max_section_chars]

        user_prompt = f"""Dokumenttitel: {document_title or ""}
Sektionsrubrik: {section_title or ""}

Text:
{clipped}
"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": 0.0,
                },
                format="json",
            )
        except Exception as e:
            raise LLMUnavailableError(
                f"Kunde inte nå Ollama/modellen '{self.model}' för preprocess: {type(e).__name__}: {e}"
            ) from e

        raw = response["message"]["content"].strip()

        try:
            data = json.loads(raw)
            meta = SectionSemanticMetadata.model_validate(data)
            return _normalize_semantic(meta)
        except Exception:
            return SectionSemanticMetadata()