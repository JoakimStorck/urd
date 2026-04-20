from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OperationPolicy:
    expansion_terms: list[str]
    preferred_section_terms: list[str]


@dataclass
class OperationConfig:
    operations: dict[str, OperationPolicy]

    @classmethod
    def empty(cls) -> "OperationConfig":
        return cls(operations={})

    def get(self, name: str) -> OperationPolicy:
        return self.operations.get(
            name,
            OperationPolicy(expansion_terms=[], preferred_section_terms=[]),
        )


def load_question_operations(path: Path) -> OperationConfig:
    if not path.exists():
        logger.info(
            "Ingen question_operations-fil hittad på %s — kör utan operationsstyrd expansion.",
            path,
        )
        return OperationConfig.empty()

    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning(
            "PyYAML är inte installerat — question_operations.yaml kan inte laddas."
        )
        return OperationConfig.empty()

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.warning("Kunde inte läsa %s: %s", path, e)
        return OperationConfig.empty()

    if not isinstance(data, dict):
        logger.warning("Fel struktur i %s — rot ska vara objekt.", path)
        return OperationConfig.empty()

    raw_ops = data.get("operations", {})
    if not isinstance(raw_ops, dict):
        logger.warning("Fel struktur i %s — 'operations' ska vara objekt.", path)
        return OperationConfig.empty()

    parsed: dict[str, OperationPolicy] = {}
    for op_name, op_data in raw_ops.items():
        if not isinstance(op_data, dict):
            continue

        expansion_terms = [
            str(x).strip()
            for x in op_data.get("expansion_terms", [])
            if str(x).strip()
        ]
        preferred_section_terms = [
            str(x).strip()
            for x in op_data.get("preferred_section_terms", [])
            if str(x).strip()
        ]

        parsed[str(op_name).strip()] = OperationPolicy(
            expansion_terms=expansion_terms,
            preferred_section_terms=preferred_section_terms,
        )

    logger.info("Laddade %d frågeoperationer från %s.", len(parsed), path)
    return OperationConfig(operations=parsed)