"""
Svensk böjningsmatchning.

Modulen samlar den prefixbaserade böjningsheuristik som tidigare låg
duplicerad i `synonyms.py` och `concepts.py` — identiska kopior av
`_VALID_ENDINGS` och `_is_inflection_of`, med risk att divergera vid
nästa ändring.

Detta är en HEURISTIK, inte morfologi. Den känner igen att "lektorer"
är en böjning av "lektor" genom att resten efter prefixet råkar vara
en känd ändelse. Den klarar inte omljud (bok/böcker), inte
oregelbundna former, och den ger falska träffar för ord som råkar
börja likadant. Riktig svensk morfologi — i stil med Kanns Stava och
Inflector — skulle ersätta den här filen, och det är en av de
öppna utvecklingslinjerna.

Konsumenterna är medvetet oförändrade i beteende: samma ändelsemängd,
samma asymmetriska jämförelse, samma resultat.
"""

# Ändelserna täcker de vanliga böjningsformerna för substantiv och
# bestämda former som förekommer i sökfrågor. Listan gör inte anspråk
# på fullständighet.
VALID_ENDINGS = {
    "",        # oböjd form
    "s",       # genitiv: lektors
    "er",      # pluralt: lektorer
    "ers",     # pluralt genitiv
    "erna",    # bestämt pluralt: lektorerna
    "ernas",   # bestämt pluralt genitiv
    "en",      # bestämd singular: lektoren i äldre texter
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
    "or",      # pluralt: flickor
    "orna",
    "e",       # adjektivböjning: stora, biträdande
}


def is_inflection_of(word: str, term: str) -> bool:
    """
    Returnerar True om `word` är en böjd form av `term`.

    Logik: word börjar med term (case-insensitive), och resten av word
    efter termen är en känd böjningsändelse. Jämförelsen är
    ASYMMETRISK — vi testar om frågeordet är en böjning av listans
    term, inte tvärtom.

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
    return tail in VALID_ENDINGS
