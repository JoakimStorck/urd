from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return [
        tok
        for tok in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(tok) >= 2
    ]


_VALID_ENDINGS = {
    "",
    "s",
    "er",
    "ers",
    "erna",
    "ernas",
    "en",
    "ens",
    "et",
    "ets",
    "n",
    "ns",
    "na",
    "nas",
    "ar",
    "ars",
    "arna",
    "arnas",
    "or",
    "orna",
    "e",
}


def _is_inflection_of(word: str, term: str) -> bool:
    word_lower = word.lower()
    term_lower = term.lower()
    if not word_lower.startswith(term_lower):
        return False
    tail = word_lower[len(term_lower):]
    return tail in _VALID_ENDINGS


@dataclass
class Concept:
    concept_id: str
    labels: list[str]
    broader: list[str]


@dataclass
class ConceptIndex:
    concepts: dict[str, Concept]
    label_index: list[tuple[str, list[str], str]]
    # [(original_label, tokenized_label, concept_id)]

    @classmethod
    def from_data(cls, raw_concepts: list[dict]) -> "ConceptIndex":
        concepts: dict[str, Concept] = {}
        label_index: list[tuple[str, list[str], str]] = []

        for item in raw_concepts:
            concept_id = str(item.get("id", "")).strip()
            if not concept_id:
                continue

            labels = [
                str(label).strip()
                for label in item.get("labels", [])
                if str(label).strip()
            ]
            broader = [
                str(b).strip()
                for b in item.get("broader", [])
                if str(b).strip()
            ]

            concept = Concept(
                concept_id=concept_id,
                labels=labels,
                broader=broader,
            )
            concepts[concept_id] = concept

            for label in labels:
                label_tokens = _tokenize(label)
                if label_tokens:
                    label_index.append((label, label_tokens, concept_id))

        return cls(concepts=concepts, label_index=label_index)

    def _question_matches_label(
        self,
        q_tokens: list[str],
        label_tokens: list[str],
    ) -> bool:
        if not label_tokens:
            return False
        for label_token in label_tokens:
            if not any(_is_inflection_of(qt, label_token) for qt in q_tokens):
                return False
        return True

    def find_matching_concept_ids(self, question: str) -> list[str]:
        q_tokens = _tokenize(question)
        if not q_tokens:
            return []

        matched: list[str] = []
        seen: set[str] = set()

        for _label, label_tokens, concept_id in self.label_index:
            if concept_id in seen:
                continue
            if self._question_matches_label(q_tokens, label_tokens):
                matched.append(concept_id)
                seen.add(concept_id)

        return matched

    def broader_labels(self, question: str) -> list[str]:
        """
        Returnera labels för överordnade begrepp till de begrepp som
        matchar frågan.

        Exempel:
          fråga = "Vilka uppgifter har en lektor?"
          -> ["lärare", "teacher", "teachers"] om lecturer -> teacher
        """
        matched_ids = self.find_matching_concept_ids(question)
        if not matched_ids:
            return []

        added: list[str] = []
        seen: set[str] = set()

        for concept_id in matched_ids:
            concept = self.concepts.get(concept_id)
            if concept is None:
                continue

            for broader_id in concept.broader:
                broader = self.concepts.get(broader_id)
                if broader is None:
                    continue
                for label in broader.labels:
                    key = label.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    added.append(label)

        return added


def load_concepts(path: Path) -> ConceptIndex:
    if not path.exists():
        logger.info("Ingen begreppsfil hittad på %s — kör utan begreppsexpansion.", path)
        return ConceptIndex.from_data([])

    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning(
            "PyYAML är inte installerat — concepts.yaml kan inte laddas. "
            "Kör utan begreppsexpansion."
        )
        return ConceptIndex.from_data([])

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.warning(
            "Kunde inte läsa begreppsfil %s: %s. Kör utan begreppsexpansion.",
            path, e,
        )
        return ConceptIndex.from_data([])

    if not isinstance(data, dict):
        logger.warning(
            "Begreppsfil %s har fel struktur (rot ska vara ett objekt). "
            "Kör utan begreppsexpansion.",
            path,
        )
        return ConceptIndex.from_data([])

    raw_concepts = data.get("concepts", [])
    if not isinstance(raw_concepts, list):
        logger.warning(
            "Begreppsfil %s: 'concepts' ska vara en lista. Kör utan begreppsexpansion.",
            path,
        )
        return ConceptIndex.from_data([])

    index = ConceptIndex.from_data(raw_concepts)
    logger.info("Laddade %d begrepp från %s.", len(index.concepts), path)
    return index