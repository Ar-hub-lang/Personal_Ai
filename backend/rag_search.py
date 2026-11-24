import asyncio
import logging
import re
import textwrap
from itertools import groupby, zip_longest

import httpx
from fastapi.concurrency import run_in_threadpool

from backend.documents import collection_enabled, collection_scope, indexed_passages, page_texts, question_terms, search_passages
from backend.local_settings import OLLAMA_URL
from backend.rag_trace import trace_event, tracing

EMBED_MODEL = "nomic-embed-text:v1.5"
EMBED_URL = f"{OLLAMA_URL}/api/embed"
SEARCH_DEPTH = 6
SINGLE_QUESTION_CONTEXT = 5
MULTI_QUESTION_CONTEXT = 8
Passage = tuple[str, int, str]
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
CITATION_LABEL = r"(?:\[(?:page\s+)?\d+\]|\((?:page\s+)?\d+\))"
CITATION_PREFIX = re.compile(rf"^(?:{CITATION_LABEL}\s*)+", re.I)
CITATION_SUFFIX = re.compile(rf"(?:\s*{CITATION_LABEL})+$", re.I)
DATE_EVENTS = tuple(re.compile(pattern, re.I) for pattern in (
    r"\b(?:open(?:ed|ing)?|launch(?:ed|ing)?|inaugurat\w*|establish\w*|founded|founding)\b",
    r"\b(?:print(?:ed|ing)?|publish\w*|publication)\b",
    r"\b(?:install\w*|commission\w*|set up|enter(?:ed|s)? service|put into service)\b",
    r"\b(?:inspect\w*|audit\w*|examin\w*)\b",
    r"\b(?:complet\w*|finish\w*|graduat\w*)\b",
    r"\b(?:revis\w*|updat\w*|amend\w*)\b",
    r"\b(?:renovat\w*|refurbish\w*|remodel\w*)\b",
    r"\b(?:purchas\w*|bought|buy)\b",
))
logger = logging.getLogger(__name__)
_cache_lock = asyncio.Lock()
_cached_rows: tuple[Passage, ...] = ()
_cached_vectors: list[list[float]] = []


async def embed(client: httpx.AsyncClient, inputs: list[str]) -> list[list[float]]:
    response = await client.post(EMBED_URL, json={"model": EMBED_MODEL, "input": inputs})
    response.raise_for_status()
    vectors = response.json()["embeddings"]
    if len(vectors) != len(inputs) or any(not vector for vector in vectors):
        raise ValueError("Incomplete embedding response.")
    return vectors


async def semantic_passages(question: str, rows: tuple[Passage, ...]) -> list[Passage]:
    global _cached_rows, _cached_vectors
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=2.0), trust_env=False) as client:
            async with _cache_lock:
                if rows != _cached_rows:
                    vectors = []
                    for start in range(0, len(rows), 16):
                        batch = [f"search_document: {row[2]}" for row in rows[start:start + 16]]
                        vectors.extend(await embed(client, batch))
                    _cached_rows, _cached_vectors = rows, vectors
                vectors = _cached_vectors
            query, = await embed(client, [f"search_query: {question}"])
        ranked = sorted(
            ((sum(left * right for left, right in zip(query, vector)), row) for row, vector in zip(rows, vectors)),
            key=lambda item: item[0], reverse=True,
        )[:SEARCH_DEPTH]
        if tracing():
            trace_event("semantic", rows=[{"score": score, "passage": row} for score, row in ranked])
        return [row for _, row in ranked]
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        trace_event("semantic", unavailable=True)
        logger.info("PDF semantic search unavailable; using the local full-text index.")
        return []


def format_evidence_entry(key: int, passage: Passage, *, collection: bool) -> str:
    filename, page, text = passage
    unit = "page" if filename.lower().endswith(".pdf") else "section"
    label = f"{filename}, " if collection else ""
    return f"[{key}] ({label}{unit} {page}) {text}"


def format_passages(passages: list[Passage]) -> str:
    collection = collection_enabled()
    return "\n\n".join(format_evidence_entry(index, row, collection=collection)
                       for index, row in enumerate(passages, 1))


def numbered_evidence(passages: list[Passage]) -> dict[int, Passage]:
    """Label source spans while keeping their existing filename and PDF page."""
    spans = []
    for filename, page, passage in passages:
        blocks = re.split(r"\n\s*\n|\n(?=(?:[•●*-]\s+|\d+(?:\.\d+)*[.)]\s+))", passage)
        for block in blocks:
            text = " ".join(block.split())
            if not text:
                continue
            if len(text) <= 520:
                spans.append((filename, page, text))
                continue
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text):
                spans.extend(
                    (filename, page, piece)
                    for piece in textwrap.wrap(sentence, width=520, break_long_words=False, break_on_hyphens=False)
                )
    return dict(enumerate(dict.fromkeys(spans), 1))


def quote_pattern(quote: str) -> re.Pattern:
    """Allow spacing changes around punctuation while matching all source tokens."""
    tokens = re.findall(r"\w+|[^\w\s]", quote)
    parts = []
    for index, token in enumerate(tokens):
        if index:
            previous_is_word = bool(re.fullmatch(r"\w+", tokens[index - 1]))
            token_is_word = bool(re.fullmatch(r"\w+", token))
            parts.append(r"\s+" if previous_is_word and token_is_word else r"\s*")
        parts.append(re.escape(token))
    return re.compile(r"(?<!\w)" + "".join(parts) + r"(?!\w)", re.I)


def verified_quotes(quotes: list[str], passages: list[Passage]) -> list[Passage]:
    sources = []
    for filename, page, text in passages:
        lines = [" ".join(line.split()) for line in text.splitlines()]
        sources.append((filename, page, " ".join(lines), "\n".join(lines)))
    verified = []
    for quote in quotes:
        clean = " ".join(quote.split())
        if not clean:
            continue
        unlabelled = CITATION_SUFFIX.sub("", CITATION_PREFIX.sub("", clean))
        heading, separator, body = unlabelled.partition(": ")
        spacing = quote_pattern(unlabelled) if unlabelled else None
        for filename, page, source, linewise in sources:
            # Match the text, since the model can confuse a passage number with a page number.
            match = re.search(re.escape(clean), source, re.I)
            if not match and unlabelled and unlabelled != clean:
                match = re.search(re.escape(unlabelled), source, re.I)
            if not match and separator and body:
                match = re.search(rf"^{re.escape(heading)}\n{re.escape(body)}", linewise, re.I | re.M)
            if not match and spacing is not None:
                match = spacing.search(source)
            if match:
                verified.append((filename, page, " ".join(match[0].split())))
                break
    return verified


def supports_years(answer: str, question: str, excerpts: list[Passage]) -> bool:
    years = set(YEAR.findall(answer))
    if not years:
        return True
    texts = [text for _, _, text in excerpts]
    if re.search(r"\b(?:year|date|when)\b", question, re.I):
        events = [event for event in DATE_EVENTS if event.search(question)]
        if events:
            texts = [sentence for text in texts
                     for sentence in re.split(r"(?<=[.!?])\s+", text)
                     if any(event.search(sentence) for event in events)]
    return years.issubset(YEAR.findall("\n".join(texts)))


def literal_evidence(answer: str, question: str, passages: list[Passage], quoted_text: str) -> list[Passage]:
    # A matching year or date in a nearby row can belong to a different event.
    if re.search(r"\b(?:date|year|when)\b", question, re.I) or YEAR.search(answer):
        return []
    answer = " ".join(answer.split()).strip('".')
    values = re.findall(r"[$₹€£]\s*\d+(?:,\d+)*(?:\.\d+)?", answer)
    if not values and len(answer) <= 64 and 1 <= len(answer.split()) <= 4:
        if answer.casefold() not in {"yes", "no", "none", "unknown", "not specified"}:
            values = [answer]

    topic = set(question_terms(question))
    recovered = []
    for value in dict.fromkeys(values):
        literal = re.compile(rf"(?<![\w.,]){re.escape(value)}(?!\w|[.,]\d)", re.I)
        if literal.search(quoted_text):
            continue
        relevant_words = topic.difference(question_terms(value))
        matches = {
            (filename, page, line)
            for filename, page, passage in passages
            for line in passage.splitlines()
            if len(line) <= 280 and literal.search(line) and relevant_words.intersection(question_terms(line))
        }
        if len(matches) == 1:
            recovered.extend(matches)
    return recovered


def comparison_queries(question: str) -> list[str]:
    comparison = re.fullmatch(r"compare (?:the )?(.+?) (?:of|for) (.+?)[?.!]?", question.strip(), re.I)
    if not comparison:
        return []
    attribute, items = comparison.groups()
    subjects = [part.strip() for part in re.split(r",\s*(?:and\s+)?|\s+and\s+", items, flags=re.I)]
    return [f"{attribute} of {subject}" for subject in subjects] if 2 <= len(subjects) <= 4 else []


async def hybrid_ranking(
    question: str, rows: tuple[Passage, ...], known: set[Passage], *, scope: set[str] | None = None,
) -> list[dict]:
    keyword_rows = (
        await run_in_threadpool(search_passages, question, filenames=scope)
        if scope else await run_in_threadpool(search_passages, question)
    )
    keyword = [row for row in keyword_rows if row in known][:SEARCH_DEPTH]
    semantic = await semantic_passages(question, rows)
    positions = [{row: index for index, row in enumerate(ranking, 1)} for ranking in (keyword, semantic)]
    scores = {}
    for ranking in positions:
        for row, index in ranking.items():
            scores[row] = scores.get(row, 0) + 1 / (index + 2)
    return [
        {"score": scores[row], "keyword_rank": positions[0].get(row),
         "semantic_rank": positions[1].get(row), "passage": row, "query": question}
        for row in sorted(scores, key=scores.__getitem__, reverse=True)
    ]


async def retrieve(question: str, *, max_full_text_bytes: int, subquestions: tuple[str, ...] = ()) -> list[Passage]:
    rows = tuple(await run_in_threadpool(indexed_passages))
    scope = await run_in_threadpool(collection_scope, question) if collection_enabled() else None
    if scope:
        rows = tuple(row for row in rows if row[0] in scope)
        trace_event("document_scope", filenames=sorted(scope))
    if not rows:
        trace_event("retrieval", query=question, mode="empty", rows=[])
        return []
    if max_full_text_bytes > 0:
        pages = [(filename, page, text) for filename, group in groupby(rows, key=lambda row: row[0])
                 for page, text in page_texts(group)]
        if len(format_passages(pages).encode("utf-8")) <= max_full_text_bytes:
            if tracing():
                trace_event("retrieval", query=question, mode="full_text",
                            rows=[{"score": None, "passage": row} for row in pages])
            return pages
    known = set(rows)
    rankings = [await hybrid_ranking(query, rows, known, scope=scope) for query in (subquestions or (question,))]
    selected = {}
    limit = MULTI_QUESTION_CONTEXT if subquestions else SINGLE_QUESTION_CONTEXT
    for round_of_results in zip_longest(*rankings):
        for result in round_of_results:
            if result is None or len(selected) == limit:
                continue
            row = result["passage"]
            if row not in selected and len(format_passages([*selected, row]).encode("utf-8")) <= max_full_text_bytes:
                selected[row] = result
        if len(selected) == limit:
            break
    if tracing():
        trace_event("retrieval", query=question, mode="hybrid", rows=list(selected.values()))
    return list(selected)
