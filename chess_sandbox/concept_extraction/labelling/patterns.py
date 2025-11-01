"""Regex patterns for detecting chess concepts in annotations."""

import re

# Tactical concepts
PIN_PATTERNS = [
    r"\bpin(?:s|ned|ning)?\b",
    r"pinned?\s+(?:to|against)",
]

FORK_PATTERNS = [
    r"\bfork(?:s|ing|ed)?\b",
    r"double\s+attack",
    r"knight\s+(?:forks?|forking)",
]

SKEWER_PATTERNS = [
    r"\bskewer(?:s|ed|ing)?\b",
]

DISCOVERED_ATTACK_PATTERNS = [
    r"\bdiscovered\s+attack",
    r"\bdiscovered\s+check",
]

SACRIFICE_PATTERNS = [
    r"\bsacrifice(?:s|d)?\b",
    r"\bsac(?:s|rifice)?\b",
]

# Strategic concepts
PASSED_PAWN_PATTERNS = [
    r"\bpassed\s+pawn",
]

OUTPOST_PATTERNS = [
    r"\boutpost(?:s)?\b",
]

WEAK_SQUARE_PATTERNS = [
    r"\bweak\s+square",
    r"\bweakness(?:es)?\b",
]

INITIATIVE_PATTERNS = [
    r"\binitiative\b",
]

ZUGZWANG_PATTERNS = [
    r"\bzugzwang\b",
]

# King safety concepts
MATING_THREAT_PATTERNS = [
    r"\bmating\s+(?:threat|attack)",
    r"\bmate\b",
    r"\bcheckmate\b",
]

EXPOSED_KING_PATTERNS = [
    r"\bexposed\s+king",
    r"\bking\s+safety",
]

# Concept registry with compiled patterns
CONCEPT_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "pin": [re.compile(p, re.IGNORECASE) for p in PIN_PATTERNS],
    "fork": [re.compile(p, re.IGNORECASE) for p in FORK_PATTERNS],
    "skewer": [re.compile(p, re.IGNORECASE) for p in SKEWER_PATTERNS],
    "discovered_attack": [re.compile(p, re.IGNORECASE) for p in DISCOVERED_ATTACK_PATTERNS],
    "sacrifice": [re.compile(p, re.IGNORECASE) for p in SACRIFICE_PATTERNS],
    "passed_pawn": [re.compile(p, re.IGNORECASE) for p in PASSED_PAWN_PATTERNS],
    "outpost": [re.compile(p, re.IGNORECASE) for p in OUTPOST_PATTERNS],
    "weak_square": [re.compile(p, re.IGNORECASE) for p in WEAK_SQUARE_PATTERNS],
    "initiative": [re.compile(p, re.IGNORECASE) for p in INITIATIVE_PATTERNS],
    "zugzwang": [re.compile(p, re.IGNORECASE) for p in ZUGZWANG_PATTERNS],
    "mating_threat": [re.compile(p, re.IGNORECASE) for p in MATING_THREAT_PATTERNS],
    "exposed_king": [re.compile(p, re.IGNORECASE) for p in EXPOSED_KING_PATTERNS],
}


def get_all_concepts() -> list[str]:
    """Get list of all available concept names.

    >>> concepts = get_all_concepts()
    >>> 'pin' in concepts
    True
    >>> 'fork' in concepts
    True
    >>> len(concepts) >= 10
    True
    """
    return list(CONCEPT_PATTERNS.keys())
