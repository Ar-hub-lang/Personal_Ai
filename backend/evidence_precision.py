"""Keep cited replies extractive while dropping unrelated passage text."""

import re
from collections import Counter

from backend.documents import STOP_WORDS
from backend.rag_search import Passage, quote_pattern

# Keep a numbered step or a bullet intact when it contains multiple related clauses.
LIST_SPLIT = re.compile(r"(?<=\s)(?=(?:[•●▪]\s*|\d{1,2}[.)]\s+(?=[A-Z])))")
SENTENCE_SPLIT = re.compile(r"(?<!\b[A-Za-z]\.)(?<=[.!?])\s+(?=[A-Z“])")
LIST_MARKER = re.compile(r"^(?:[•●▪]\s*|\d{1,2}[.)]\s+)")
PROCEDURE = re.compile(
    r"\b(?:steps|procedure|walk me through|step.by.step)\b"
    r"|^how (?:do|can|should|to)\b.*\b(?:install|set up|configure|connect|troubleshoot|reset)\b",
    re.I,
)
START_ACTION = re.compile(
    r"^how (?:do|does|can|should|to)\b.*?\b(?:start|begin)\s+(?:to\s+)?([a-z]{4,})\b",
    re.I,
)
START_RESPONSE_STEP = re.compile(r"^(\d{1,2})[.)]\s+(.+)")
EXPLICIT_OPTIONS = re.compile(
    r"\bwhere\b.*\b(?:click|select|tap|choose)\b"
    r"|\brequir\w*\b.*\b(?:digits?|numbers?|characters?)\b"
    r"|\b(?:digits?|numbers?|characters?)\b.*\brequir\w*\b",
    re.I,
)
CONDITION = re.compile(
    r"^(?:only if|if|unless|before|after|otherwise|important\b|prerequisites?\b|"
    r"you (?:must|need to)\b|do not\b|make sure\b)", re.I,
)
WORD = re.compile(r"[^\W_]+")
LOW_INFORMATION = STOP_WORDS | set(
    "absolutely according answer associated here listed look must note one option options please required"
    " select should shown step steps stated use using will would".split()
)


BRIEF_REQUEST = re.compile(
    r"(?:^|[.?!]\s+)(?:short answer(?: only)?|just (?:the )?[\w -]{1,40}"
    r"|(?:the )?(?:month and year|name|number|value|date) only)[.!?]*$",
    re.I,
)
SINGLE_FACT = re.compile(
    r"^(?:who\b|how many\b|which (?:model|dataset|(?:programming )?language)\b"
    r"|what did .+\b(?:display|show|return)\b"
    r"|what(?:'s| is| was| percentage| version| year| date)\b"
    r".*\b(?:accuracy|percentage|version|year|date|score|name|dataset|language|hostname)\b)",
    re.I,
)
LIST_REQUEST = re.compile(r"^(?:list\b|which\b.*?\b(?:libraries|packages)\b)", re.I)
CONDITIONAL_VALUE = re.compile(
    r"\b(?:only if|if|unless|except|depending on|provided that|subject to|not (?:yet |currently )?"
    r"(?:approved|allowed|available|supported|recommended))\b",
    re.I,
)
NUMBER_QUALIFIER = re.compile(
    r"(?:[~≈<>≤≥]|\b(?:approximately|roughly|around|about|at least|at most|up to|"
    r"less than|more than))\s*$", re.I,
)
METRIC = re.compile(r"\b(?:accuracy|precision|recall|f1[ -]?score)\b", re.I)
NUMBER = re.compile(r"(?<![\w.,])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*%?(?![\w]|[.,]\d)")
QUOTED_LABEL = re.compile(r"(?<!\w)(['\"])([A-Za-z][A-Za-z0-9 -]{0,28})\1(?!\w)")
UNCERTAIN_VALUE = {"yes", "no", "none", "unknown", "not specified", "not available", "n/a"}


def requested_metric(question: str) -> str | None:
    if not re.match(r"^what(?:'s| is| was| accuracy| precision| recall| f1[ -]?score)\b", question, re.I):
        return None
    if re.fullmatch(r"what(?:'s| is) (?:accuracy|precision|recall|f1[ -]?score)[?.!]*", question, re.I):
        return None
    metrics = metric_labels(question)
    return next(iter(metrics)) if len(metrics) == 1 else None


def numeric_values(text: str) -> set[str]:
    return {re.sub(r"[,\s]", "", match[0]) for match in NUMBER.finditer(text)}


def metric_labels(text: str) -> set[str]:
    return {re.sub(r"[- ]", "", match[0].casefold()) for match in METRIC.finditer(text)}


def isolated_metric(sentence: str, metric: str) -> list[str]:
    labels = metric_labels(sentence)
    if labels == {metric}:
        return [sentence]
    if metric not in labels:
        return []
    # A score can be followed by mentions of other metrics with no numeric result.
    label = r"f1[ -]?score" if metric == "f1score" else re.escape(metric)
    value = r"\d+(?:\.\d+)?(?:\s*%)?"
    adjacent = re.compile(
        rf"\b{label}\b\s*(?:(?:was|of|at)\s+|[:=]\s*)?{value}"
        rf"|{value}\s+(?:(?:classification|test|validation|overall|model)\s+)*\b{label}\b",
        re.I,
    )
    return [match[0] for match in adjacent.finditer(sentence)]


def metric_supported(metric: str, question: str, answer: str, excerpts: list[Passage]) -> bool:
    values = numeric_values(answer)
    if len(values) != 1:
        return False
    subject = terms(METRIC.sub("", question)) - {"exact", "short", "achieve", "achieved", "get", "number"}
    for _, _, text in excerpts:
        for sentence in SENTENCE_SPLIT.split(text):
            for result in isolated_metric(sentence, metric):
                # Tie a score to the requested subject in a compact comparison.
                clauses = re.split(r"(?<=[)%])(?:,\s+|\s+and\s+)", result, flags=re.I)
                ranked = [(len(subject & terms(clause)), numeric_values(clause)) for clause in clauses]
                best_overlap = max(overlap for overlap, _ in ranked)
                if any(overlap == best_overlap and values <= numbers for overlap, numbers in ranked):
                    return True
    return False


def verified_brief(question: str, answer: str, excerpts: list[Passage]) -> list[Passage]:
    """Return a short literal value when its source and scope can be preserved."""
    if PROCEDURE.search(question) or EXPLICIT_OPTIONS.search(question):
        return []
    listed = bool(LIST_REQUEST.search(question) and "," in answer)
    metric = requested_metric(question)
    if not (listed or metric or BRIEF_REQUEST.search(question) or SINGLE_FACT.search(question)):
        return []
    candidate = " ".join(answer.split()).rstrip(".")
    candidate = QUOTED_LABEL.sub(r"\2", candidate.replace("“", "").replace("”", ""))
    if metric:
        scores = list(NUMBER.finditer(candidate))
        if len(scores) != 1:
            return []
        candidate = scores[0][0].strip()
    if not candidate or candidate.casefold() in UNCERTAIN_VALUE or len(candidate.split()) > 12:
        return []
    if len({filename for filename, _, _ in excerpts}) != 1:
        return []
    if listed:
        items = [value.strip() for value in re.split(r",\s*(?:and\s+)?|\s+and\s+", candidate)]
        if not all(items):
            return []
        patterns = [quote_pattern(item) for item in items]
    else:
        if re.fullmatch(r"\d{4,}", candidate):
            groups = f"{int(candidate):,}".split(",")
            patterns = [re.compile(r"(?<!\w)" + r",?".join(groups) + r"(?!\w)")]
        else:
            patterns = [quote_pattern(candidate)]

    matches = {}
    for filename, page, text in excerpts:
        units = split_units(" ".join(text.split()))
        for index, unit in enumerate(units):
            # Printed quotation marks around names and labels do not change their values.
            source = QUOTED_LABEL.sub(r"\2", unit.replace("“", "").replace("”", ""))
            hits = [pattern.search(source) for pattern in patterns]
            if not all(hits):
                continue
            neighbors = units[max(index - 1, 0):index + 2]
            if (CONDITIONAL_VALUE.search(source) or any(
                CONDITION.search(LIST_MARKER.sub("", part)) for part in neighbors
            )):
                return []
            values = []
            for match in sorted(hits, key=lambda hit: hit.start()):
                before, after = source[:match.start()], source[match.end():]
                if re.search(r"\d[.,]$", before) or re.match(r"[.,]\d", after):
                    return []
                value = match[0]
                if re.match(r"\d", value):
                    if qualifier := NUMBER_QUALIFIER.search(before):
                        value = qualifier[0] + value
                    if percent := re.match(r"\s*%", after):
                        value += percent[0]
                values.append(value)
            matches.setdefault(", ".join(values), (filename, page))
    if len(matches) != 1:
        return []
    value, (filename, page) = matches.popitem()
    return [(filename, page, value)]


def terms(text: str) -> set[str]:
    return {
        token for token in WORD.findall(text.casefold())
        if token not in LOW_INFORMATION and (len(token) > 2 or token.isdigit())
    }


def split_units(text: str) -> list[str]:
    pieces = []
    for row in LIST_SPLIT.split(text):
        row = row.strip()
        if not row:
            continue
        if LIST_MARKER.match(row):
            pieces.append(row)
        else:
            pieces.extend(SENTENCE_SPLIT.split(row))
    return pieces


def _deduplicate(excerpts: list[Passage]) -> list[Passage]:
    # Overlapping retrieval chunks may repeat the same passage on the same page.
    seen = set()
    unique = []
    for filename, page, text in excerpts:
        key = filename, page, " ".join(text.casefold().split())
        if key not in seen:
            seen.add(key)
            unique.append((filename, page, text))
    return unique


def before_completion(question: str, excerpts: list[Passage]) -> list[Passage]:
    """Keep the beginning of a numbered workflow for explicit start questions."""
    action = START_ACTION.search(question)
    if not action or len({filename for filename, _, _ in excerpts}) != 1:
        return excerpts
    steps = [(filename, page, START_RESPONSE_STEP.match(unit), unit)
             for filename, page, text in excerpts for unit in split_units(text)]
    if not steps or any(match is None for _, _, match, _ in steps):
        return excerpts
    if [int(match[1]) for _, _, match, _ in steps] != list(range(1, len(steps) + 1)):
        return excerpts
    completion = re.compile(rf"\bafter\s+{re.escape(action[1])}\b", re.I)
    for index, (_, _, match, _) in enumerate(steps[2:], 2):
        sentences = SENTENCE_SPLIT.split(match[2])
        if completion.match(sentences[-1]) and all(
            re.match(r"^This (?:may|can) take\b", sentence, re.I) for sentence in sentences[:-1]
        ):
            return [(filename, page, unit) for filename, page, _, unit in steps[:index]]
    return excerpts


def focus_excerpts(question: str, answer: str, excerpts: list[Passage]) -> list[Passage]:
    """Select only source units that collectively cover the model's supported answer."""
    unique = _deduplicate(excerpts)
    beginning = before_completion(question, unique)
    if beginning != unique:
        return beginning
    excerpts = unique
    if not excerpts or PROCEDURE.search(question) or EXPLICIT_OPTIONS.search(question):
        return excerpts

    answer_terms = terms(answer)
    question_terms = terms(question)
    units = [
        (source_index, filename, page, unit, terms(unit))
        for source_index, (filename, page, passage) in enumerate(excerpts)
        for unit in split_units(passage)
    ]
    if len(units) < 2:
        return excerpts
    source_terms = set().union(*(unit[-1] for unit in units))
    supported = answer_terms & source_terms
    distinctive = supported - question_terms - {"yes", "none"}
    if not distinctive:
        return excerpts

    # A vague answer gives too little information to choose between substantive source units.
    # A short exact identifier is still sufficient when it occurs only in the cited answer line.
    if len(supported) == 1:
        value = next(iter(distinctive))
        matching = [unit for unit in units if value in unit[-1]]
        if len(matching) != 1 or len(value) < 4:
            return excerpts

    # Keep the original reply when trimming could mask an ungrounded value.
    answer_values = {token for token in answer_terms if token.isdigit()}
    if not answer_values <= source_terms:
        return excerpts

    needed = set(supported)
    selected = set()
    occurrences = Counter(term for unit in units for term in unit[-1] if term in needed)
    while needed:
        def rank(unit):
            covered = needed & unit[-1]
            return (
                len(covered),
                sum(1 / occurrences[token] for token in covered & distinctive),
                len(unit[-1] & question_terms),
                -len(unit[3]),
            )

        index = max(range(len(units)), key=lambda item: rank(units[item]))
        selected.add(index)
        needed.difference_update(units[index][-1])

    # Keep adjacent conditions that govern a selected instruction.
    for index in tuple(selected):
        for neighbor_index in (index - 1, index + 1):
            if not 0 <= neighbor_index < len(units):
                continue
            neighbor = units[neighbor_index]
            if neighbor[0] == units[index][0] and CONDITION.search(LIST_MARKER.sub("", neighbor[3])):
                selected.add(neighbor_index)

    focused = [units[index][1:4] for index in sorted(selected)]
    return focused if len(" ".join(text for _, _, text in focused)) < sum(len(text) for _, _, text in excerpts) else excerpts
