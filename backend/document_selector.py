"""Select cited excerpts with the local TinyGPT reader."""

import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby

from backend.document_language import normalize_question
from backend.documents import (
    SUBMITTER_QUESTION, collection_enabled, collection_scope, document_submitter, indexed_passages, page_texts,
)
from backend.rag_search import DATE_EVENTS, Passage
from backend.rag_trace import trace_event, tracing
from training.document_reader import (
    CHECKPOINT, NUMBER, STOP, YEAR, DocumentReader, features, load_reader, sentence_spans, stem, words,
)


MONEY = re.compile(r"[$₹€£]\s*\d|\b\d[\d.,]*\s*(?:dollars?|rupees?|euros?|pounds?|usd|inr|eur|gbp)\b", re.I)
PRICE_QUESTION = re.compile(
    r"^(?:how much\b.*\b(?:cost|charge|pay)|"
    r"what (?:is|was|are|were) (?!included\b).*\b(?:fee|fees|price|cost|charge|rate)\b|"
    r"what (?:does|do|did)\b.*\bcost\b|which (?:\w+ )?(?:fee|price|rate)\b)",
    re.I,
)
PRICE_LABEL = re.compile(r"\b(?:fee|fees|price|cost|charge|rate|amount)\b", re.I)
PRICE_QUALIFIERS = tuple(re.compile(pattern, re.I) for pattern in (
    r"\b(?:daily|per day)\b", r"\b(?:weekly|per week)\b", r"\b(?:monthly|per month)\b",
    r"\b(?:yearly|annual|per year)\b", r"\b(?:purchase|retail|sale)\b",
))
COST_TERMS = frozenset(
    stem(word) for word in "fee fees price cost charge rate monthly weekly daily yearly annual much pay per".split()
)
CLOCK_TIME = re.compile(r"\b(?:\d{1,2}(?::\d{2})?\s*[ap]\.?\s*m\.?|(?:[01]?\d|2[0-3]):[0-5]\d)\b", re.I)
DATE_VALUE = re.compile(
    rf"{YEAR.pattern}|\b\d+(?:\.\d+)?\s+(?:thousand|million|billion)\s+years?\b"
    r"|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}\b",
    re.I,
)
DATE_QUESTION = re.compile(r"^(?:when\b|(?:in )?(?:what|which) (?:year|date)\b|what\b.*\bdate\b)", re.I)
DATE_RANGE_QUESTION = re.compile(r"^(?:what|which)\b.*\b(?:date|year) range\b", re.I)
DATE_RANGE = re.compile(r"\d(?:[\d.,°˚]*\s*)?(?:[-–—]|to|and)\s*\d", re.I)
LOOKBACK_QUESTION = re.compile(r"^how far back\b", re.I)
ENTRY_QUESTION = re.compile(r"^(?:which|what) (?:\w+ )?(?:gate|door|entrance)\b.*\b(?:enter|use|go|access)\b", re.I)
ACCESS = re.compile(r"\b(?:enter\w*|entrance|entry|access|admit\w*|use[sd]?)\b", re.I)
BRING_QUESTION = re.compile(r"^what\b.*\b(?:bring|pack)\b", re.I)
BRING = re.compile(r"\b(?:bring|pack|required items|supplies include|must have)\b", re.I)
DESTINATION = re.compile(r"\b(?:to|in|on|at|inside|under|near|beside|into)\s+(?!be\b)\w+", re.I)
PERSON_NAME = re.compile(r"\b(?!The\b|This\b|On\b|In\b)[A-Z][a-z]+(?:[ '-][A-Z][a-z]+){1,3}\b")
PERSON_ROLE = re.compile(r"\b(?:coordinator|steward|clerk|manager|supervisor|keyholder|host|officer|lead|staff)\b", re.I)
RESTRICTION = re.compile(r"\b(?:allow\w*|permit\w*|prohibit\w*|forbid\w*|bann?\w*|may|must|can(?:not)?)\b", re.I)
RULE_PLACE = re.compile(r"\b(?:on|in|at)\s+([\w -]+?)\??$", re.I)
WEEKDAY = re.compile(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
LABEL_QUESTION = re.compile(r"^(?:which|what) (?:type of )?label\b", re.I)
MEASUREMENT_QUESTION = re.compile(r"\b(?:temperature range|temperature.*degrees|what share|percentage|what percent)\b", re.I)
SUPPLY_LIST_QUESTION = re.compile(r"^what\b.*\b(?:provide|supply|give)\b", re.I)
SUPPLY_LIST_EVIDENCE = re.compile(r"\b(?:provide\w*|supply|supplies|giv\w*|source)\b", re.I)

ANSWER_PATTERNS = tuple((re.compile(question, re.I), re.compile(evidence, re.I)) for question, evidence in (
    (r"\b(?:temperature range|temperature.*degrees)\b", r"\d\s*[°˚º]\s*[CF]\b|\d\s+degrees?\b"),
    (r"\b(?:duration|how long.*\b(?:last|take|tour|visit|wait|stay))\b", r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight)\s*(?:min(?:ute)?s?|hr?s?|hours?|days?|weeks?|months?|years?)\b"),
    (r"\b(?:what share|percentage|what percent)\b", r"\d\s*(?:%|percent\b)"),
    (r"\b(?:what year|which year)\b", YEAR.pattern),
    (r"\b(?:what date|which date)\b", DATE_VALUE.pattern),
    (r"\b(?:what does .* (?:mean|tell)|what is the definition of)\b", r"\b(?:means|measure|indicates|defined|refers|definition)\b"),
    (r"\b(?:which .* hours|what .* hours|opening hours|closing hours)\b", r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b"),
    (r"^what time\b", CLOCK_TIME.pattern),
    (r"\b(?:mileage|how far(?!\s+back\b)|what distance)\b", r"\d\s*(?:mi\b|miles?\b|km\b|kilomet(?:er|re)s?\b)"),
    (r"\b(?:ridership|passenger count|number of riders)\b", r"\d[\d,.]*\s+(?:daily\s+|annual\s+)?(?:riders|passengers|trips)\b|(?:ridership|passenger count)\s*[:=]?\s*\d"),
    (r"\b(?:(?:which|what) .*\b(?:number|stop)|(?:which|what) stop|how many|how much|how long|which .*zone)\b", NUMBER.pattern),
))
ROW_START = re.compile(r"^\d{1,3}(?:$|\s*[-–|]\s*\S|\s+[A-Z])")
ROW_LOOKUP = re.compile(r"\b(?:which|what)\b.*\b(?:stop|station|route number|climate zone|record number)\b", re.I)

TOPIC_STOP = STOP | set(
    "and any before after each from get given go guide does document fact file i listed mentions must "
    "new our pdf says section sheet should shows some the their then through this very want we year your".split()
)
RELATIONS = (
    "entrance enter entry access admit",
    "visitor guest attendee people participant",
    "coordinate coordinates coordinator coordination organize organizer organise arrange oversees manages handles",
    "appointment appointments bookings booking reservation reservations",
    "show shown see present",
    "move moving transfer",
    "return returned send deposit leave",
    "old used spent",
    "ship shipping shipment delivery courier",
    "require requires needed use used",
    "receive receives deliver take",
    "close closed closing shuts shut",
    "allow allows allowed permit permits permitted",
    "prohibit prohibits prohibited ban banned forbidden",
    "fee fees price cost charge rate",
    "house household residential home",
    "typical typically average",
    "share percent percentage proportion part",
    "usage use consumption",
    "recommend recommended set suggested",
    "refrigerators refrigeration fridge",
    "water swimming boating paddling kayaking canoeing",
    "supply provide give source",
)
RELATION_STEMS = {}
for relation in RELATIONS:
    group = {stem(word) for word in relation.split()}
    for member in group:
        RELATION_STEMS.setdefault(member, set()).update(group)


@dataclass(frozen=True)
class Evidence:
    passage: Passage
    columns: tuple[str, ...] = ()
    heading: str = ""

    @property
    def ranking_text(self) -> str:
        return " ".join(part for part in (self.heading, *self.columns, self.passage[2]) if part)


@lru_cache(maxsize=1)
def cached_reader(checkpoint_mtime: int) -> DocumentReader:
    reader, _ = load_reader()
    return reader


def evidence_snippets(page: str) -> set[str]:
    page = re.sub(r"(?<=\w)-\n(?=[a-z])", "", page)
    lines = page.splitlines()
    prose = " ".join(lines)
    spans = sentence_spans(prose)
    snippets = {prose[left:right] for left, right in spans}
    # Keep the subject beside facts in adjacent sentences, within the excerpt limit.
    snippets.update(prose[left:right] for (left, _), (_, right) in zip(spans, spans[1:]))
    # Preserve rows whose first field is a number or whose columns have separators.
    for index, line in enumerate(lines):
        if "|" in line or re.search(r"\d\s*(?:a\.?m\.?|p\.?m\.?)\b", line, re.I):
            snippets.add(line)
        if ROW_START.match(line):
            snippets.add(line)
            snippets.update(re.split(r"\s+(?=\d{1,3}\s*[-–|]\s*[A-Z])", line))
            following = []
            for next_line in lines[index + 1:index + 3]:
                if ROW_START.match(next_line):
                    break
                following.append(next_line)
            if following:
                snippets.add(" ".join((line, *following)))
    return {" ".join(text.split()) for text in snippets if 14 <= len(text.strip()) <= 360}


def evidence_passages(filename: str, pages: list[tuple[int, str]]) -> list[Evidence]:
    passages = []
    for page, text in pages:
        for tabular, lines in groupby(text.splitlines(), key=lambda line: "|" in line):
            block = list(lines)
            if not tabular:
                passages.extend(Evidence((filename, page, span)) for span in evidence_snippets("\n".join(block)))
                headings = [i for i, line in enumerate(block)
                            if 2 <= len(line.split()) <= 7 and len(line) <= 60
                            and line.isupper() and all(char.isalpha() or char in " &/-" for char in line)]
                for start, end in zip(headings, [*headings[1:], len(block)]):
                    excerpt = "\n".join(block[start + 1:min(end, start + 12)])
                    passages.extend(Evidence((filename, page, span), heading=block[start])
                                    for span in evidence_snippets(excerpt))
                continue
            rows = [tuple(cell.strip() for cell in line.split("|")) for line in block]
            columns = rows[0] if len(rows) > 1 and not NUMBER.search(block[0]) else ()
            for line, row in zip(block[bool(columns):], rows[bool(columns):]):
                header = columns if len(columns) == len(row) else ()
                passages.append(Evidence((filename, page, line), header))
    return passages


@lru_cache(maxsize=8192)
def topic_terms(text: str) -> frozenset[str]:
    return frozenset(stem(word) for word in words(text) if word not in TOPIC_STOP)


def matched_terms(query: frozenset[str], source: frozenset[str]) -> set[str]:
    return {term for term in query if term in source or RELATION_STEMS.get(term, set()) & source}


def contains_price(evidence: Evidence) -> bool:
    if MONEY.search(evidence.passage[2]):
        return True
    values = evidence.passage[2].split("|") if evidence.columns else ()
    return any(
        PRICE_LABEL.search(label) and re.fullmatch(r"\s*\d[\d.,]*\s*", value)
        for label, value in zip(evidence.columns, values)
    )


def rank_evidence(
    question: str, passages: list[Evidence], reader: DocumentReader, *, previous_part: str = "",
) -> Passage | None:
    intent = normalize_question(question)
    expected = next((value for prompt, value in ANSWER_PATTERNS if prompt.search(intent)), None)
    polite = re.match(r"can you (?:tell|show|give) me\b", intent, re.I)
    rule_question = not polite and re.match(r"(?:can|does|do|is|are|were|was)\b", intent, re.I)
    date_question = DATE_QUESTION.search(intent)
    events = [event for event in DATE_EVENTS if event.search(intent)] if date_question else []
    date_range = DATE_RANGE_QUESTION.search(intent)
    lookback = LOOKBACK_QUESTION.search(intent)
    price_question = PRICE_QUESTION.search(intent)
    qualifiers = [pattern for pattern in PRICE_QUALIFIERS if pattern.search(intent)] if price_question else []
    entry_question = ENTRY_QUESTION.search(intent)
    bring_question = BRING_QUESTION.search(intent)
    person_question = re.match(r"who\b", intent, re.I)
    place_question = re.match(r"where\b", intent, re.I)
    clock_question = re.match(r"what time\b", intent, re.I)
    label_question = LABEL_QUESTION.search(intent)
    measurement = MEASUREMENT_QUESTION.search(intent)
    supply_list = SUPPLY_LIST_QUESTION.search(intent) and bool(re.search(r"\b(?:two|three|both)\b", intent, re.I))
    restriction_question = rule_question or re.search(r"\b(?:prohibit\w*|forbid\w*|ban)\b", intent, re.I)
    weekdays = {day.casefold() for day in WEEKDAY.findall(intent)}
    topic = topic_terms(intent)
    date_focus = topic - topic_terms("date range far back place give approximately estimate period usual")
    supply_focus = topic - topic_terms("two three both things give supply provide healthy resident")
    price_topic = topic - COST_TERMS
    if price_question and not price_topic:
        price_topic = topic_terms(previous_part) - topic_terms("bring pack kind much need")
        if not price_topic:
            return None
    rule_place = RULE_PLACE.search(intent) if restriction_question else None
    rule_topic = topic_terms(rule_place[1]) if rule_place else frozenset()
    rule_action = topic - rule_topic - topic_terms("prohibit forbid ban allow activities two ordinary")
    remainder = re.sub(r"^(?:which|what|where|when|how|can you tell me)\b\s*", "", intent, flags=re.I)
    named = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", remainder)
    entity = named[-1].casefold() if named and ROW_LOOKUP.search(intent) else None
    candidates = []
    for evidence in passages:
        text = evidence.passage[2]
        ranking_text = evidence.ranking_text
        if expected is not None and not expected.search(text):
            continue
        if events and not any(event.search(sentence) and (DATE_VALUE.search(sentence) or CLOCK_TIME.search(sentence))
                              for sentence in re.split(r"(?<=[.!?])\s+", text) for event in events):
            continue
        if (lookback or date_range) and not DATE_VALUE.search(text):
            continue
        if date_range and not DATE_RANGE.search(text):
            continue
        if rule_question and not re.search(r"\b(?:allowed|prohibited|permitted|forbidden|may|can|must|are|were|is)\b", text, re.I):
            continue
        if weekdays and not weekdays.issubset(day.casefold() for day in WEEKDAY.findall(text)):
            continue
        if entry_question and not ACCESS.search(text):
            continue
        if bring_question and not BRING.search(text):
            continue
        if label_question and not re.search(r"\blabels?\b", text, re.I):
            continue
        source = topic_terms(text)
        location = source | topic_terms(evidence.heading)
        matches = matched_terms(topic, source)
        coverage = len(matches) / max(1, len(topic))
        date_match = len(matched_terms(date_focus, source)) / max(1, len(date_focus))
        if (lookback or date_range) and date_match < 0.65:
            continue
        if price_question:
            if (not contains_price(evidence)
                    or not all(pattern.search(ranking_text) for pattern in qualifiers)
                    or not matched_terms(price_topic, source - topic_terms("fee price cost charge"))
                    or len(matched_terms(price_topic, topic_terms(ranking_text))) / len(price_topic) < 0.65):
                continue
        exact, related, density, phrase, year_match, number_match, _, _ = features(question, ranking_text)
        named_row = bool(entity and entity in text.casefold() and ROW_START.match(text))
        grounded = bool(
            price_question
            or (entry_question and coverage >= 0.6)
            or (bring_question and coverage >= 0.55)
            or (clock_question and coverage >= 0.6)
            or (label_question and coverage >= 0.6)
            or (measurement and coverage >= 0.65)
            or (lookback or date_range)
            or (supply_list and len(supply_focus) >= 2 and matched_terms(supply_focus, source) == supply_focus
                and SUPPLY_LIST_EVIDENCE.search(text) and re.search(r"\band\b", text, re.I))
            or (person_question and len(matches) >= 2 and coverage >= 0.55
                and (PERSON_NAME.search(text) or PERSON_ROLE.search(text)))
            or (place_question and len(matches) >= 2 and coverage >= 0.6 and DESTINATION.search(text))
            or (restriction_question and rule_topic and rule_action
                and matched_terms(rule_topic, location) == rule_topic
                and matched_terms(rule_action, source) and RESTRICTION.search(text))
        )
        if not named_row and not grounded and related < (0.25 if expected is not None else 0.35):
            continue
        if rule_topic and not matched_terms(rule_topic, location):
            continue
        score = 2 * exact + 2 * related + density + phrase + 0.2 * (year_match + number_match)
        row_bonus = 1.3 if named_row else 0.0
        bonus = row_bonus
        if grounded:
            focus = price_topic if price_question else topic
            bonus += 1.5 + 1.5 * len(matched_terms(focus, source)) / max(1, len(focus))
        candidates.append((score + bonus, bonus, evidence))
    if not candidates:
        trace_event("retrieval", query=question, mode="tinygpt", rows=[])
        return None
    ranked = sorted(candidates, key=lambda row: (-row[0], row[2].passage[1], len(row[2].passage[2]), row[2].passage[2]))[:24]
    finalists = [row[2] for row in ranked]
    scores = [score + row[1] for score, row in zip(reader.rank(question, [item.ranking_text for item in finalists]), ranked)]
    highest = max(range(len(scores)), key=scores.__getitem__)
    if tracing():
        trace_event("retrieval", query=question, mode="tinygpt", rows=[
            {"score": scores[i], "passage": finalists[i].passage}
            for i in sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        ])
    return finalists[highest].passage if scores[highest] > 0 else None


def select_document_evidence(questions: list[str], *, context_query: str | None = None) -> list[Passage | None]:
    rows = indexed_passages()
    collection = collection_enabled()
    scope = collection_scope(context_query or " ".join(questions)) if collection else set()
    if scope:
        rows = [row for row in rows if row[0] in scope]
    if not rows:
        return []
    sources = [(filename, list(page_texts(group))) for filename, group in groupby(rows, key=lambda row: row[0])]
    reader = None
    selected = []
    previous_part = ""
    for question in questions:
        intent = " ".join(normalize_question(question).casefold().split()).rstrip(" ?.!")
        if not collection and SUBMITTER_QUESTION.fullmatch(intent):
            filename, pages = sources[0]
            credit = document_submitter(pages)
            match = (filename, *credit) if credit else None
            trace_event("shortcut", name="submitter", evidence=[match] if match else [])
        else:
            if reader is None:
                reader = cached_reader(CHECKPOINT.stat().st_mtime_ns)
                passages = [passage for filename, pages in sources for passage in evidence_passages(filename, pages)]
            match = rank_evidence(context_query or question, passages, reader, previous_part=previous_part)
        selected.append(match)
        previous_part = question
    return selected
