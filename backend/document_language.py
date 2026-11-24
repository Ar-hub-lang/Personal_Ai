"""Normalize PDF questions and supply close spellings for evidence matching."""

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path


WORD = re.compile(r"[^\W_]+", re.UNICODE)
QUESTION_TOKEN = re.compile(
    r"`[^`]*`|https?://\S+|\b[\w.+-]+@[\w.-]+\b|"
    r"\b(?:[\w-]+\.)+[\w-]+\b|(?:[A-Za-z]:[\\/]|(?<!\w)/)\S+|"
    r"\b[\w]+(?:[-_]\w+)+\b|[^\W_]+",
    re.IGNORECASE,
)
SPELLING_PATH = Path(__file__).with_name("Correct_Spelling.json")
SPELLING = json.loads(SPELLING_PATH.read_text(encoding="utf-8"))
RELATED_WORDS = (
    {"college", "university", "universities", "polytechnic"},
    {"shuttle", "bus"},
    {"stop", "station", "halt"},
    {"fee", "cost", "price", "charge"},
    {"percent", "percentage", "share", "proportion"},
    {"duration", "runtime", "length"},
    {"prohibited", "forbidden", "banned"},
    {"permitted", "allowed", "authorized"},
    {"recommend", "suggest", "advise"},
    {"yearly", "annual"},
    {"fridge", "refrigerator"},
)


def normalize_question(question: str) -> str:
    question = unicodedata.normalize("NFKC", question).replace("’", "'")

    def replace_word(match: re.Match) -> str:
        word = match[0]
        return SPELLING.get(word.casefold(), word)

    question = QUESTION_TOKEN.sub(replace_word, question)
    return re.sub(r"\bwhats\b", "what's", question, flags=re.I)


@lru_cache(maxsize=16384)
def close_spelling(word: str, candidate: str) -> bool:
    """Match one adjacent swap or a missing/extra letter, with the first letter fixed."""
    if (len(word) < 6 or len(candidate) < 5 or word[0] != candidate[0] or abs(len(word) - len(candidate)) > 1
            or not word.isalpha() or not candidate.isalpha()):
        return False
    first = next((index for index, (a, b) in enumerate(zip(word, candidate)) if a != b), min(len(word), len(candidate)))
    if len(word) == len(candidate):
        return (first + 1 < len(word) and word[first:first + 2] == candidate[first:first + 2][::-1]
                and word[first + 2:] == candidate[first + 2:])
    shorter, longer = sorted((word, candidate), key=len)
    return shorter[first:] == longer[first + 1:]
