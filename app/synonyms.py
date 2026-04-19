"""
Synonymexpansion för retrieval.

Modulen laddar en instans-specifik synonymlista från .urd/synonyms.yaml
och tillhandahåller en expand_bm25_query-funktion som breddar söktexten
för BM25 med kända termvarianter.

Designprinciper:

- Expansionen påverkar ENDAST BM25. Embedding och cross-encoder-rerank
  arbetar på den ursprungliga frågan. Detta är samma princip som QUD-
  ankringen: hjälp kandidaturvalet, lämna relevansbedömningen ren.

- Matchningen är flexibel genom prefix-matchning. Svensk stemming är
  notoriskt krånglig — "lektor" och "lektorer" stammas inte nödvändigtvis
  till samma form av standardstemmers. I stället jämför vi frågeord mot
  synonymlistans termer med prefix-logik: ett frågeord matchar en
  synonymterm om det börjar med termen och skillnaden är en rimlig
  böjningsändelse (upp till ~5 tecken). Det är transparent, förutsägbart
  och kräver ingen lingvistisk kunskap.

- Symmetrisk expansion inom en grupp: om frågan innehåller EN term ur
  en synonymgrupp, läggs ALLA andra termer i gruppen till söktexten.

- Tyst fallback: om synonyms.yaml saknas eller är felaktig returneras
  originalfrågan oförändrad och en varning loggas. URD ska fungera
  utan synonymlista.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenisering — samma principer som i retrieval.py:s BM25-tokenisering
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Extrahera ord ur text, gemener, minst 2 tecken."""
    return [
        tok
        for tok in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(tok) >= 2
    ]


# ---------------------------------------------------------------------------
# Flexibel matchning genom prefix
# ---------------------------------------------------------------------------

# Ett frågeord räknas som en böjning av en term om frågeordet börjar
# med termen och "resten" är en känd svensk böjningsändelse. Listan
# behöver inte vara uttömmande för svensk morfologi — den behöver
# bara täcka de vanliga böjningsformerna för substantiv och bestämda
# former som förekommer i sökfrågor.
_VALID_ENDINGS = {
    "",        # oböjd form
    "s",       # genitiv: lektors
    "er",      # pluralt: lektorer
    "ers",     # pluralt genitiv
    "erna",    # bestämt pluralt: lektorerna
    "ernas",   # bestämt pluralt genitiv
    "en",      # bestämd singular: lektorn kan också stavas lektoren i äldre texter
    "ens",     # bestämd singular genitiv
    "et",      # neutrum bestämd: ärendet
    "ets",
    "n",       # bestämd singular: lektorn
    "ns",      # lektorns
    "na",      # bestämt pluralt svag böjning
    "nas",
    "ar",      # pluralt: stolar
    "ars",
    "arna",
    "arnas",
    "or",      # pluralt: flickor — förekommer
    "orna",
    "e",       # adjektivböjning: stora, biträdande
}


def _is_inflection_of(word: str, term: str) -> bool:
    """
    Returnerar True om `word` är en böjd form av `term`.

    Logik: word börjar med term (case-insensitive), och resten av word
    efter termen är en känd böjningsändelse. Detta är asymmetriskt —
    vi testar om frågeordet är en böjning av synonymlistans term, inte
    tvärtom.

    Exempel:
    - word="lektorer", term="lektor" -> True (ändelse "er")
    - word="lektor", term="lektor" -> True (ändelse "")
    - word="lektorn", term="lektor" -> True (ändelse "n")
    - word="universitetslektor", term="lektor" -> False (börjar inte med term)
    - word="lekt", term="lektor" -> False (börjar inte med hela term)
    """
    word_lower = word.lower()
    term_lower = term.lower()
    if not word_lower.startswith(term_lower):
        return False
    tail = word_lower[len(term_lower):]
    return tail in _VALID_ENDINGS


# ---------------------------------------------------------------------------
# SynonymIndex — den laddade och indexerade synonymlistan
# ---------------------------------------------------------------------------

@dataclass
class SynonymIndex:
    """
    Indexerad representation av synonymgrupper.

    Varje grupp är en lista av utbytbara termer (strängar, kan innehålla
    flera ord). En term kan bestå av flera ord ("biträdande
    universitetslektor"). När en sådan term ska matcha i en fråga
    behöver alla dess ord (eller böjningar av dem) finnas i frågan.
    Ordningen kontrolleras inte — "lektor biträdande" matchar alltså
    mot "biträdande lektor". I praktiken är det nästan alltid rätt
    beteende för retrieval.
    """

    groups: list[list[str]]
    # För varje grupp (index), lista av term-ord-listor.
    # Varje term är en lista av ord (ett för enordstermer, flera för
    # sammansatta uttryck). Orden lagras i originalform och jämförs
    # med _is_inflection_of mot frågeord.
    _group_term_words: list[list[list[str]]]

    @classmethod
    def from_groups(cls, groups: list[list[str]]) -> "SynonymIndex":
        group_term_words: list[list[list[str]]] = []
        for group in groups:
            term_words_list = []
            for term in group:
                words = _tokenize(term)
                term_words_list.append(words)
            group_term_words.append(term_words_list)
        return cls(groups=groups, _group_term_words=group_term_words)

    def _question_matches_term(
        self, q_tokens: list[str], term_words: list[str]
    ) -> bool:
        """
        En term matchar om varje ord i termen har en böjningsform
        representerad bland frågeorden.
        """
        if not term_words:
            return False
        for term_word in term_words:
            if not any(_is_inflection_of(qt, term_word) for qt in q_tokens):
                return False
        return True

    def find_matching_groups(self, question: str) -> list[int]:
        """
        Returnera index på de synonymgrupper som matchar mot frågan.

        En grupp matchar om någon av dess termer förekommer i frågan
        (med flexibel böjningsmatchning). Flera grupper kan matcha
        samtidigt — alla returneras.
        """
        q_tokens = _tokenize(question)
        if not q_tokens:
            return []

        matched: list[int] = []
        for group_idx, term_words_list in enumerate(self._group_term_words):
            for term_words in term_words_list:
                if self._question_matches_term(q_tokens, term_words):
                    matched.append(group_idx)
                    break  # en term räcker för att gruppen ska matcha
        return matched

    def expand_terms(self, question: str) -> list[str]:
        """
        Returnera de ytterligare termer som ska läggas till BM25-söktexten.

        För varje matchande grupp läggs alla *andra* termer från den
        gruppen till. Originalfrågan innehåller redan den matchande
        termen (eller en böjning av den) — den behöver inte dubbleras.

        Dedupliceras: om flera grupper introducerar samma term läggs
        den bara till en gång.

        Notera att frågan kan trigga flera grupper samtidigt när en
        smal term innehåller ord som också matchar en bred term
        (t.ex. "biträdande lektor" triggar både {biträdande
        lektor-gruppen} och {lektor-gruppen}). Det är avsiktligt:
        expansion är additiv, och cross-encodern (som inte ser
        expansionen) avgör själv relevansbedömningen.
        """
        matched_groups = self.find_matching_groups(question)
        if not matched_groups:
            return []

        q_tokens = _tokenize(question)

        added: list[str] = []
        seen: set[str] = set()
        for group_idx in matched_groups:
            for term in self.groups[group_idx]:
                term_words = _tokenize(term)
                # Hoppa över termer som redan finns i frågan (via böjningsmatch)
                if self._question_matches_term(q_tokens, term_words):
                    continue
                term_lower = term.lower()
                if term_lower in seen:
                    continue
                seen.add(term_lower)
                added.append(term)
        return added


# ---------------------------------------------------------------------------
# Laddning från fil
# ---------------------------------------------------------------------------

def load_synonyms(path: Path) -> SynonymIndex:
    """
    Ladda synonymgrupper från en YAML-fil.

    Förväntat format:

        synonyms:
          - [lektor, universitetslektor]
          - [biträdande lektor, biträdande universitetslektor, bitr. lektor]

    Om filen saknas, är tom, eller inte kan parsas returneras ett tomt
    SynonymIndex. En varning loggas men ingenting kastas — URD ska
    fungera utan synonymlista.
    """
    if not path.exists():
        logger.info("Ingen synonymlista hittad på %s — kör utan expansion.", path)
        return SynonymIndex.from_groups([])

    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning(
            "PyYAML är inte installerat — synonymlistan kan inte laddas. "
            "Kör utan expansion."
        )
        return SynonymIndex.from_groups([])

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.warning(
            "Kunde inte läsa synonymlista %s: %s. Kör utan expansion.",
            path, e,
        )
        return SynonymIndex.from_groups([])

    if not isinstance(data, dict):
        logger.warning(
            "Synonymlista %s har fel struktur (rot ska vara ett objekt). "
            "Kör utan expansion.",
            path,
        )
        return SynonymIndex.from_groups([])

    raw_groups = data.get("synonyms", [])
    if not isinstance(raw_groups, list):
        logger.warning(
            "Synonymlista %s: 'synonyms' ska vara en lista. Kör utan expansion.",
            path,
        )
        return SynonymIndex.from_groups([])

    groups: list[list[str]] = []
    for i, group in enumerate(raw_groups):
        if not isinstance(group, list):
            logger.warning(
                "Synonymlista %s: grupp %d är inte en lista — hoppar över.",
                path, i,
            )
            continue
        cleaned = [str(term).strip() for term in group if str(term).strip()]
        if len(cleaned) >= 2:
            groups.append(cleaned)
        else:
            logger.warning(
                "Synonymlista %s: grupp %d har färre än 2 termer — hoppar över.",
                path, i,
            )

    if groups:
        logger.info("Laddade %d synonymgrupper från %s.", len(groups), path)
    return SynonymIndex.from_groups(groups)


# ---------------------------------------------------------------------------
# Publikt API — används av retrieval.py
# ---------------------------------------------------------------------------

def expand_bm25_query(question: str, synonyms: SynonymIndex) -> str:
    """
    Bredda söktexten för BM25 med kända termvarianter.

    Returnerar den ursprungliga frågan om inga synonymer matchar.
    Annars returneras frågan följd av de tillagda termerna, separerade
    med mellanslag. BM25:s tokenisering behandlar den utökade strängen
    som en utökad ordmängd.

    Denna funktion påverkar INTE embedding eller cross-encoder. Se
    retrieval.py för hur det säkerställs.
    """
    added = synonyms.expand_terms(question)
    if not added:
        return question
    return question + " " + " ".join(added)


# ---------------------------------------------------------------------------
# Manuell test — kör `python -m app.synonyms` för att verifiera
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Enkla exempel som verifierar att expansionen gör rätt saker."""
    index = SynonymIndex.from_groups([
        ["lektor", "universitetslektor"],
        ["biträdande lektor", "biträdande universitetslektor", "bitr. lektor"],
        ["adjunkt", "universitetsadjunkt"],
    ])

    test_cases = [
        ("Vad gäller för lektorer?", ["universitetslektor"]),
        ("Vad är en universitetslektor?", ["lektor"]),
        # "biträdande lektor" triggar både den smala gruppen (bitr. lektor)
        # och den breda (lektor/universitetslektor) eftersom ordet "lektor"
        # matchar i båda. Det är avsiktligt: expansion är additiv, och
        # cross-encodern får ändå avgöra vad som är relevant. Se
        # SynonymIndex.expand_terms för resonemanget.
        ("Hur blir man biträdande lektor?",
         ["universitetslektor", "biträdande universitetslektor", "bitr. lektor"]),
        ("Vad är skillnaden mellan en professor och en adjunkt?",
         ["universitetsadjunkt"]),
        ("Vilka är rektorns uppgifter?", []),  # ingen match
        ("", []),  # tom fråga
    ]

    print("Självtest för synonymexpansion")
    print("=" * 50)
    all_ok = True
    for question, expected in test_cases:
        got = index.expand_terms(question)
        ok = sorted(got) == sorted(expected)
        all_ok = all_ok and ok
        marker = "✓" if ok else "✗"
        print(f"{marker} {question!r}")
        print(f"   förväntat: {expected}")
        print(f"   fick:      {got}")

    print()
    if all_ok:
        print("Alla tester OK.")
    else:
        print("Några tester fallerade — granska ovan.")

    print()
    print("Exempel på expand_bm25_query:")
    for question, _ in test_cases[:3]:
        expanded = expand_bm25_query(question, index)
        print(f"  {question!r}")
        print(f"  -> {expanded!r}")


if __name__ == "__main__":
    _self_test()