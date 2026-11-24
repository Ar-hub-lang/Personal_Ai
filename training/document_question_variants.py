"""Create repeatable question variants without changing evidence or labels."""

import random
import re


REPHRASES = (
    (r"\bwhat is\b", "can you tell me"),
    (r"\bwhich\b", "what"),
    (r"\bhow many\b", "what is the number of"),
    (r"\blisted\b", "shown"),
    (r"\bidentify\b", "name"),
    (r"\bhow far back\b", "how long ago"),
    (r"\bwhat year\b", "when"),
)


def question_variants(question: str, rng: random.Random) -> list[str]:
    variants = []
    words = list(re.finditer(r"\b[a-z]{6,}\b", question))
    if words:
        word = rng.choice(words)
        text = word[0]
        positions = [i for i in range(1, len(text) - 1) if text[i] != text[i + 1]]
        if positions:
            i = rng.choice(positions)
            typo = text[:i] + text[i + 1] + text[i] + text[i + 2:]
            variants.append(question[:word.start()] + typo + question[word.end():])
    for pattern, replacement in REPHRASES:
        altered, count = re.subn(pattern, replacement, question, count=1, flags=re.I)
        if count:
            variants.append(altered)
            break
    return variants


def variant_rows(rows: list[tuple[str, list[str], int]], *, seed: int) -> list[tuple[str, list[str], int]]:
    rng = random.Random(seed)
    return [(alternate, passages, label) for question, passages, label in rows
            for alternate in question_variants(question, rng)]
