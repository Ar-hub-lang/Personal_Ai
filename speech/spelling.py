import json
import re
from functools import lru_cache
from pathlib import Path

DICTIONARY_PATH = Path(__file__).with_name("spelling_corrections.json")
IT_SPELLINGS = json.dumps({
    "d-field global protect": "Deerfield GlobalProtect",
    "d-field globalprotect": "Deerfield GlobalProtect",
    "global protect": "GlobalProtect",
    "globalprotect": "GlobalProtect",
    "edu roam": "eduroam",
    "eduroam": "eduroam",
    "paper cut print": "PaperCut print",
    "papercut": "PaperCut",
})


@lru_cache(maxsize=2)
def load_corrections(contents: str) -> tuple[dict[str, str], re.Pattern[str]]:
    entries = json.loads(contents)
    if not isinstance(entries, dict) or any(
        not alias.strip() or not isinstance(spelling, str) or not spelling.strip()
        for alias, spelling in entries.items()
    ):
        raise ValueError("Expected a JSON object mapping spoken variants to correct spellings.")
    spellings = {" ".join(alias.casefold().split()): spelling for alias, spelling in entries.items()}
    aliases = sorted(
        (alias for alias in spellings if not re.fullmatch(r"\w+", alias)),
        key=len,
        reverse=True,
    )
    phrases = [r"\s+".join(map(re.escape, alias.split())) for alias in aliases]
    # Check phrases before looking up individual words in the dictionary.
    alternatives = "|".join([*phrases, r"\w+"])
    pattern = re.compile(r"(?<!\w)(?:" + alternatives + r")(?!\w)", re.IGNORECASE)
    return spellings, pattern


def correct_spelling(transcript: str, *, it_support: bool = False) -> str:
    contents = IT_SPELLINGS if it_support else DICTIONARY_PATH.read_text(encoding="utf-8")
    spellings, pattern = load_corrections(contents)

    def replacement(match: re.Match[str]) -> str:
        original = match[0]
        key = " ".join(original.casefold().split())
        spelling = spellings.get(key, original)
        if original[0].isupper() and spelling[0].islower():
            return spelling[0].upper() + spelling[1:]
        return spelling

    return pattern.sub(replacement, transcript)
