import json
import os
import re
import sqlite3
from bisect import bisect_right
from contextlib import closing, contextmanager
from io import BytesIO
from itertools import groupby
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from backend.document_language import normalize_question
from backend.rag_trace import trace_event, tracing

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = (
    PROJECT_ROOT / "collections/it-support/index/documents.sqlite3"
    if os.environ.get("PERSONAL_AI_COLLECTION") == "1"
    else PROJECT_ROOT / "memory/documents.sqlite3"
)
DATABASE_PATH = Path(os.environ.get("PERSONAL_AI_DOCUMENT_DB") or DEFAULT_INDEX)
MAX_PDF_BYTES = 10 * 1024 * 1024
CHUNK_OVERLAP_WORDS = 20
SECTION_CHUNK_MIN_CHARS = 500
SECTION_CHUNK_MAX_CHARS = 1000
STOP_WORDS = set(
    "about actually after also and are been but can could did does for from had has have "
    "how into its may more most much only our please says see should tell than that the "
    "their them then there these they this those too under use very was were what when "
    "where which who why will with would you your document pdf project used uses in is it we".split()
)
TECHNOLOGY_TERMS = (
    "framework", "platform", "stack", "language", "pwa", "native", "android", "ios",
    "website", "database", "backend", "frontend", "runtime",
)
SECTION_HEADING = re.compile(
    r"\bPhase\s+\d+:\s+(?P<phase>[A-Z][A-Za-z ]{0,60}?(?:\s*\([^)]{1,40}\))?)"
    r"(?=\s+\d{1,3}\.\s+[A-Z])"
    r"|\b(?P<title>(?:[A-Z][a-z]+\s+){1,3}(?:Features|Plan))\b"
)
NUMBERED_ITEM = re.compile(r"(?<![\w.])(\d{1,3})\.\s+(?=[A-Z])")
CLASS_12 = r"(?:class\s+(?:12(?:th)?|xii)|12th(?:\s+(?:class|grade|standard))?|intermediate|higher secondary|senior secondary)"
LOCAL_SETUP_QUESTION = re.compile(
    r"\b(?:local (?:setup|development)|(?:run|start|launch|set ?up)\b[^?.!]*\blocally)\b", re.I
)
LOCAL_SETUP_HEADING = re.compile(r"(?mi)^(?:local (?:setup|development(?: setup)?)|development setup):?$")
EDUCATION_HEADING = re.compile(r"(?mi)^(?:education|academic qualifications):?$")
SKILLS_HEADING = re.compile(r"(?:technical |core |key )?skills:?", re.I)
SKILL_FIELD = re.compile(r"^([^:]{1,40}):")
SKILL_FIELDS = {
    "database": "database", "databases": "database", "database skills": "database", "database tools": "database",
    "frontend": "frontend", "front end": "frontend",
    "backend": "backend", "back end": "backend",
    "programming": "programming", "programming languages": "programming",
    "machine learning": "machine learning",
    "tools & deployment": "tools and deployment", "tools and deployment": "tools and deployment",
}
RESUME_HEADING = re.compile(
    r"(?mi)^(?:(?:professional |career )?(?:summary|profile|objective)|"
    r"(?:work |professional |employment )?experience|(?:technical |core |key )?skills|"
    r"(?:selected |academic |personal )?projects|certifications?|awards|interests|references):?$"
)
RESUME_NAME_QUESTION = re.compile(
    r"(?:(?:what (?:is|was)|what's|(?:can you )?tell me|show me) )?"
    r"(?:(?:the |my |(?:the )?(?:applicant|candidate|person)'s )?"
    r"(?:full )?name(?: (?:on|in) (?:this|the) (?:resume|résumé|cv))?|"
    r"whose (?:resume|résumé|cv) is this|who is this (?:resume|résumé|cv) for|"
    r"who does this (?:resume|résumé|cv) belong to)"
)
INSTITUTION = re.compile(r"\b(?:university|college|institute|polytechnic)\b", re.I)
COLLEGE_QUESTION = re.compile(
    r"(?:(?:what (?:is|are|was)|what's|(?:can you )?tell me|show me) )?(?:the |my )?"
    r"(?:(?:name|names) of (?:the |my )?(?:college|collage|university|uni)|(?:college|university|uni) name)"
    r"|(?:what|which) (?:college|university) (?:did i attend|do i attend|is listed|did i go to|did i graduate from)"
    r"|where did (?:i|the (?:applicant|candidate)) (?:study|go to college|graduate from|(?:do|complete) (?:my |the )?(?:bca|degree))"
    r"|what (?:was|is) (?:my|the) (?:college|university) called"
)
PROJECT_STACK_LINE = re.compile(
    r"^(?:tech(?:nology|nologies)?(?: stack)?\s*:.*"
    r"|[A-Za-z][\w.+#/-]*(?:\s*,\s*[A-Za-z][\w.+#/-]*)+(?:\s*\|.*)?)$",
    re.I,
)
PROJECT_HEADING = re.compile(r"(?mi)^(?:(?:selected|personal|academic) )?projects:?$")
UPPERCASE_HEADING = re.compile(r"(?m)^[A-Z][A-Z0-9 &/()-]{2,60}:?$")
BULLET = re.compile(r"^(?:[•●▪◦*-]|\d{1,3}[.)])(?:\s+|$)")
SECTION_QUESTION = re.compile(
    r"(?:(?:what (?:is|are)(?: the)?|list|show(?: me)?|tell me about|summarize|"
    r"how many(?: total(?: number of)?)?|count(?: the)?|(?:what is the )?(?:total )?number of) )?"
    r"(?:the )?(?P<title>.+?)(?: (?:are there|in it|in the (?:pdf|document)|does it have))?[?.!]*"
)
LIBRARY_LIST_QUESTION = re.compile(
    r"(?:(?:what|which) (?:software |python )?librar(?:y|ies)(?: and dependencies)?"
    r"(?: (?:is|are|was|were) (?:used|listed|required))?(?: (?:in|by|for) (?:this|the|our) (?:project|report|document))?"
    r"|(?:list|show me) (?:the )?librar(?:y|ies)(?: and dependencies)?)",
    re.I,
)
LIBRARY_HEADING = re.compile(r"^(?:\d+\.\s*)?(?:software )?(?:libraries(?: and dependencies)?|software dependencies)\s*:?$", re.I)
SUBMISSION_SUBJECT = (
    r"(?:it|(?:(?:the|this|my|our) )?(?:(?:mini |final )?project report|report|project|document|pdf"
    r"|thesis|dissertation|assignment|paper|proposal))"
)
SUBMITTER_QUESTION = re.compile(
    rf"(?:submitted by|who (?:is|was) (?:{SUBMISSION_SUBJECT} )?submitted by"
    rf"|who submitted(?: {SUBMISSION_SUBJECT})?|by whom (?:is|was) {SUBMISSION_SUBJECT} submitted"
    r"|(?:who is|what is the name of) (?:the )?submitter)"
)
SUBMISSION_LABEL = re.compile(r"\bsubmitted\s+by\b\s*[:–—-]?\s*", re.I)
SUBMISSION_END = re.compile(
    r"[,;]?\s+(?:to|for|on)\s+|\s+in\s+(?=\d)"
    r"|(?P<field>\s+under (?:the )?(?:guidance|supervision)\b"
    r"|[\s,;]+(?:supervisor|advisor|guide|roll (?:no\.?|number)|student id)\s*:)"
    r"|[ \t]*\(\d[\w -]*\)",
    re.I,
)
SUBMISSION_PROSE = re.compile(r"\b(?:is|was|were|has been)\s+$", re.I)
NAME_WORD = re.compile(r"[^\W\d_]+(?:['’.-][^\W\d_]+)*\.?", re.UNICODE)
NAME_PARTICLES = {"al", "and", "bin", "da", "de", "del", "di", "dos", "la", "le", "van", "von"}
NON_PERSON_CREDIT = re.compile(
    r"^(?:the|a|an|all|under|department|university|supervisor|advisor|guide|student|"
    r"tbd|pending|unknown|n/a|none|today|tomorrow|noon|midnight|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|^(?:january|february|march|april|may|june|july|august|september|october|november|december)$",
    re.I,
)


@contextmanager
def open_index():
    with closing(sqlite3.connect(DATABASE_PATH)) as connection, connection:
        connection.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(
                filename UNINDEXED, page UNINDEXED, text, tokenize='porter unicode61'
            )"""
        )
        if collection_enabled():
            connection.execute(
                """CREATE TABLE IF NOT EXISTS document_records (
                    filename TEXT PRIMARY KEY, updated_at TEXT NOT NULL, aliases TEXT NOT NULL,
                    sections TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS document_web_sources (
                    filename TEXT PRIMARY KEY, url TEXT NOT NULL
                )"""
            )
        yield connection


def collection_enabled() -> bool:
    return os.environ.get("PERSONAL_AI_COLLECTION") == "1"


def index_pdf(filename: str, data: bytes, *, aliases: tuple[str, ...] = ()) -> dict[str, str | int]:
    if not filename.lower().endswith(".pdf") or not data.startswith(b"%PDF-"):
        raise ValueError("Choose a PDF file.")
    pages = []
    ocr_pages = []
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise ValueError("Choose a PDF that does not require a password.")
        if len(reader.pages) > 100:
            raise ValueError("Choose a PDF with at most 100 pages.")
        for page, source in enumerate(reader.pages, start=1):
            raw = source.extract_text() or ""
            if collection_enabled() and (
                not raw.strip() or sum(ord(char) < 32 and not char.isspace() for char in raw) > len(raw) * 0.03
            ):
                ocr_pages.append(page)
                continue
            lines = raw.replace("\x7f", "•").splitlines()
            text = "\n".join(" ".join(line.split()) for line in lines if line.strip())
            if text:
                pages.append((page, text))
    except (PyPdfError, RecursionError) as exc:
        raise ValueError("The PDF could not be read.") from exc

    if ocr_pages:
        from backend.document_intake import recognize_pdf_pages

        pages.extend(recognize_pdf_pages(data, ocr_pages))
        pages.sort()
    if not pages:
        raise ValueError("No readable text was found in the PDF.")
    return index_text_pages(filename, pages, section_aware=os.environ.get("RAG_SECTION_CHUNKS") == "1", aliases=aliases)


def page_chunks(text: str, *, section_aware: bool):
    words = list(re.finditer(r"\S+", text))
    if not section_aware:
        for anchor in range(0, len(words), 140):
            start = max(0, anchor - CHUNK_OVERLAP_WORDS)
            end = min(anchor + 140, len(words))
            yield text[words[start].start():words[end - 1].end()]
        return
    ends = [word.end() for word in words]
    boundaries = set()
    for index, (word, following) in enumerate(zip(words, words[1:]), 1):
        separator = text[word.end():following.start()]
        if separator.count("\n") >= 2 or (separator == "\n" and word[0].endswith((".", "?", "!"))):
            boundaries.add(index)
        elif separator == "\n" and re.match(r"(?:[•*-] |\d+(?:\.\d+)*\.\s+)", text[following.start():]):
            boundaries.add(index)
    start = 0
    while start < len(words):
        base = words[start].start()
        upper = bisect_right(ends, base + SECTION_CHUNK_MAX_CHARS, lo=start)
        upper = min(len(words), max(upper, start + CHUNK_OVERLAP_WORDS + 1))
        lower = max(start + CHUNK_OVERLAP_WORDS + 1, bisect_right(ends, base + SECTION_CHUNK_MIN_CHARS, lo=start))
        candidates = boundaries.intersection(range(lower, upper + 1))
        end = max(candidates, default=upper) if upper < len(words) else upper
        yield text[base:ends[end - 1]]
        if end == len(words):
            break
        start = end - CHUNK_OVERLAP_WORDS


def index_text_pages(
    filename: str, pages: list[tuple[int, str]], *, section_aware: bool,
    aliases: tuple[str, ...] = (), sections: dict[int, str] | None = None,
    source_url: str | None = None,
) -> dict[str, str | int]:
    if sum(len(text) for _, text in pages) > 400_000:
        raise ValueError("This document contains too much text for the local index.")
    passages = [(filename, page, chunk) for page, text in pages for chunk in page_chunks(text, section_aware=section_aware)]
    if not passages:
        raise ValueError("No readable text was found in the document.")
    with open_index() as connection:
        if collection_enabled():
            connection.execute("DELETE FROM passages WHERE filename = ?", (filename,))
            previous = connection.execute(
                "SELECT aliases FROM document_records WHERE filename = ?", (filename,)
            ).fetchone()
            connection.execute(
                """INSERT OR REPLACE INTO document_records VALUES (
                    ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'), ?, ?
                )""",
                (filename, json.dumps(aliases) if aliases or not previous else previous[0], json.dumps(sections or {})),
            )
            if source_url is not None:
                connection.execute("INSERT OR REPLACE INTO document_web_sources VALUES (?, ?)", (filename, source_url))
            else:
                connection.execute("DELETE FROM document_web_sources WHERE filename = ?", (filename,))
        else:
            connection.execute("DELETE FROM passages")
        connection.executemany("INSERT INTO passages VALUES (?, ?, ?)", passages)
    return {"filename": filename, "pages": len({page for _, page, _ in passages})}


def collection_documents() -> list[dict]:
    with open_index() as connection:
        rows = connection.execute(
            """SELECT p.filename, COUNT(DISTINCT p.page), d.updated_at, w.url
               FROM passages p LEFT JOIN document_records d ON p.filename = d.filename
               LEFT JOIN document_web_sources w ON p.filename = w.filename
               GROUP BY p.filename, d.updated_at, w.url ORDER BY p.filename COLLATE NOCASE"""
        ).fetchall()
    return [
        {"filename": filename, "pages": pages, "updated_at": updated_at} | ({"url": url} if url else {})
        for filename, pages, updated_at, url in rows
    ]


def remove_collection_document(filename: str) -> bool:
    with open_index() as connection:
        removed = connection.execute("DELETE FROM passages WHERE filename = ?", (filename,)).rowcount
        connection.execute("DELETE FROM document_records WHERE filename = ?", (filename,))
        connection.execute("DELETE FROM document_web_sources WHERE filename = ?", (filename,))
    return removed > 0


def collection_scope(question: str) -> set[str]:
    cleaned = " ".join(re.findall(r"[^\W_]+", question.casefold()))
    with open_index() as connection:
        records = connection.execute("SELECT filename, aliases FROM document_records").fetchall()
    return {
        filename for filename, aliases in records if any(
            f" {alias} " in f" {cleaned} "
            for value in json.loads(aliases)
            if (alias := " ".join(re.findall(r"[^\W_]+", value.casefold())))
        )
    }


def source_metadata(filename: str, page: int) -> dict:
    source = {"filename": filename, "page": page}
    if collection_enabled() and not filename.lower().endswith(".pdf"):
        with open_index() as connection:
            record = connection.execute(
                """SELECT d.sections, w.url FROM document_records d
                   LEFT JOIN document_web_sources w ON d.filename = w.filename WHERE d.filename = ?""",
                (filename,),
            ).fetchone()
        source["section"] = json.loads(record[0]).get(str(page), f"Section {page}") if record else f"Section {page}"
        if record and record[1]:
            source["url"] = record[1]
    return source


def active_document() -> dict[str, str | int] | None:
    with open_index() as connection:
        row = connection.execute(
            "SELECT filename, COUNT(DISTINCT page) FROM passages GROUP BY filename"
        ).fetchone()
    return {"filename": row[0], "pages": row[1]} if row else None


def answer_document_title(question: str) -> dict | None:
    normalized = " ".join(question.casefold().split()).rstrip(" ?.!")
    if not re.fullmatch(
        r"(?:(?:what is|what's|tell me) )?(?:the )?"
        r"(?:(?:name|title) of (?:the |our |this )?(?:project|pdf|document)|"
        r"(?:project|pdf|document) (?:name|title))",
        normalized,
    ):
        return None
    with open_index() as connection:
        row = connection.execute("SELECT filename, page, text FROM passages ORDER BY page, rowid LIMIT 1").fetchone()
    if not row:
        return None
    filename, page, text = row
    title_words = re.findall(r"\w+", Path(filename).stem)
    if not title_words:
        return None
    title_pattern = r"[^\w]+".join(map(re.escape, title_words))
    title = re.match(rf"{title_pattern}\b", text, re.IGNORECASE)
    if not title:
        return None
    heading = title[0]
    name, separator, subtitle = heading.partition(":")
    reply = f"The PDF heading is “{heading}.”"
    if "project" in normalized and separator and subtitle.strip().casefold() in {"feature list", "project plan"}:
        reply = f"The PDF calls the project “{name}” in the heading “{heading}.”"
    return {"reply": reply, "sources": [{"filename": filename, "page": page}]}


def answer_section(question: str) -> dict | None:
    normalized = " ".join(question.casefold().split())
    query = SECTION_QUESTION.fullmatch(normalized)
    if not query or not re.search(r"\b(?:features|plan)\b", query["title"]):
        return None
    wants_count = bool(re.search(r"\b(?:how many|count|number of)\b", normalized))

    with open_index() as connection:
        rows = connection.execute("SELECT filename, page, text FROM passages ORDER BY page, rowid").fetchall()
    if not rows:
        return None
    filename = rows[0][0]
    footer_title = r"[\W_]+".join(map(re.escape, re.findall(r"\w+", Path(filename).stem)))
    footer = re.compile(
        rf"(?:{footer_title}\s+)?\bPage\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE
    )
    pages = []
    offsets = []
    parts = []
    length = 0
    for page, page_text in page_texts(rows):
        text = footer.sub("", " ".join(page_text.split())).strip()
        if text:
            pages.append(page)
            offsets.append(length)
            parts.append(text)
            length += len(text) + 1
    document = " ".join(parts)
    headings = list(SECTION_HEADING.finditer(document))
    for index, heading in enumerate(headings):
        title = heading["phase"] or heading["title"]
        if title.casefold() != query["title"].strip():
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(document)
        section = document[heading.end() : end]
        items = list(NUMBERED_ITEM.finditer(section))
        if not items and wants_count:
            return None
        if items:
            numbers = [int(item[1]) for item in items]
            if section[:items[0].start()].strip() or numbers != list(range(numbers[0], numbers[-1] + 1)):
                return None
            excerpts = []
            for item_index, item in enumerate(items):
                item_end = items[item_index + 1].start() if item_index + 1 < len(items) else len(section)
                body = section[item.end() : item_end].strip()
                excerpt = re.split(r"(?<=[.!?])\s+", body, maxsplit=1)[0]
                excerpts.append(f"{item[1]}. {excerpt}")
            reply = f"There are {len(items)} entries under {title}." if wants_count else "\n".join(excerpts)
        else:
            reply = section.strip()
            if len(reply) > 1800:
                return None
        if not reply:
            return None
        first = bisect_right(offsets, heading.start()) - 1
        last = bisect_right(offsets, end - 1) - 1
        return {"reply": reply, "sources": [{"filename": filename, "page": page} for page in pages[first : last + 1]]}
    return None


def answer_library_list(question: str) -> dict | None:
    if not LIBRARY_LIST_QUESTION.fullmatch(" ".join(question.split()).rstrip(" ?.!,").strip()):
        return None
    rows = indexed_passages()
    for page, text in page_texts(rows):
        lines = text.splitlines()
        for start, heading in enumerate(lines):
            if not LIBRARY_HEADING.fullmatch(heading):
                continue
            body = []
            for line in lines[start + 1:]:
                if re.fullmatch(r"(?:\d+(?:\.\d+)*\.\s+)?\w[^:]{1,79}:", line) or UPPERCASE_HEADING.fullmatch(line):
                    break
                body.append(line)
            if body and BULLET.match(body[0]) and len("\n".join(body)) <= 2000:
                return cited_reply("\n".join((heading, *body)), rows[0][0], page)
    return None


def page_texts(rows):
    for page, group in groupby(rows, key=lambda row: row[1]):
        parts = []
        previous_tail = []
        for _, _, chunk in group:
            words = list(re.finditer(r"\S+", chunk))
            if (len(previous_tail) == CHUNK_OVERLAP_WORDS
                    and previous_tail == [word.group() for word in words[:CHUNK_OVERLAP_WORDS]]):
                parts.append(chunk[words[CHUNK_OVERLAP_WORDS - 1].end():])
            else:
                parts.append("\n" + chunk)
            previous_tail = [word.group() for word in words[-CHUNK_OVERLAP_WORDS:]]
        yield page, "".join(parts).lstrip("\n")


def document_submitter(pages: list[tuple[int, str]]) -> tuple[int, str] | None:
    """Read a submitter credit, preferring a labeled line over prose."""
    prose_credit = None
    for page, text in pages:
        lines = text.splitlines()
        for index, line in enumerate(lines):
            label = SUBMISSION_LABEL.search(line)
            if label is None:
                continue
            credit = line[label.end():]
            prefix = line[:label.start()]
            if not credit and not prefix.strip() and index + 1 < len(lines):
                credit = lines[index + 1]
            end = SUBMISSION_END.search(credit)
            names = (credit[:end.start()] if end else credit).strip(" ,;:.")
            words = names.replace(",", " ").replace("&", " ").split()
            if (not 1 <= len(words) <= 16 or NON_PERSON_CREDIT.match(names)
                    or not all(NAME_WORD.fullmatch(word) and (not word[0].islower() or word in NAME_PARTICLES)
                               for word in words)):
                continue
            if not prefix.strip() or prefix.rstrip().endswith((":", "–", "—", "|")) or (end and end["field"]):
                return page, names
            if SUBMISSION_PROSE.search(prefix) and prose_credit is None:
                prose_credit = page, names
    return prose_credit


def answer_resume_skills(pages: list[tuple[int, str]], filename: str, field: str | None) -> dict | None:
    in_skills = False
    include = field is None
    excerpts = []
    for page, content in pages:
        for line in content.splitlines():
            if SKILLS_HEADING.fullmatch(line):
                in_skills = True
                include = field is None
                continue
            if RESUME_HEADING.fullmatch(line) or EDUCATION_HEADING.fullmatch(line):
                in_skills = False
            if not in_skills:
                continue
            if field is not None and (label := SKILL_FIELD.match(line)):
                include = SKILL_FIELDS.get(label[1].strip().casefold()) == field
            if include and line.strip():
                excerpts.append((page, line))
    if not excerpts:
        return None
    return {
        "reply": "\n".join(line for _, line in excerpts),
        "sources": [{"filename": filename, "page": page} for page in sorted({page for page, _ in excerpts})],
    }


def answer_resume_question(question: str, *, filename: str | None = None) -> dict | None:
    normalized = " ".join(normalize_question(question).casefold().split()).rstrip(" ?.!")
    asks_name = RESUME_NAME_QUESTION.fullmatch(normalized)
    asks_college = COLLEGE_QUESTION.fullmatch(normalized)
    skills_label = re.sub(
        r"^(?:(?:what|which)(?: (?:is|are))? |list |show me |(?:can you )?tell me )?(?:the |my |his |her |their )?",
        "", normalized,
    )
    skills_label = re.sub(
        r" (?:does (?:he|she|the (?:candidate|applicant)) (?:know|use|have)|(?:are|is) listed)$",
        "", skills_label,
    )
    asks_skills = SKILLS_HEADING.fullmatch(skills_label)
    skill_field = SKILL_FIELDS.get(skills_label)
    if not (asks_name or asks_college or asks_skills or skill_field):
        return None

    rows = indexed_passages(filename=filename)
    if not rows:
        return None
    filename = rows[0][0]
    pages = list(page_texts(rows))
    if asks_skills or skill_field:
        return answer_resume_skills(pages, filename, skill_field)
    if asks_name:
        # A short line at the top of an arbitrary PDF can be its title. Require
        # recognizable résumé sections and contact information near the header.
        page, text = pages[0]
        header = text.splitlines()[:6]
        education = any(EDUCATION_HEADING.search(content) for _, content in pages)
        if not education or not RESUME_HEADING.search(text) or not re.search(
            r"\b(?:phone|email|mobile|linkedin)\b|@", " ".join(header), re.I,
        ):
            return None
        if header[0].casefold() in {"resume", "résumé", "curriculum vitae", "cv"}:
            header = header[1:]
        name = re.sub(r"^name\s*:\s*", "", header[0], flags=re.I) if header else ""
        letters = name.translate(str.maketrans("", "", " .-'’"))
        if 1 <= len(name.split()) <= 5 and letters.isalpha() and not INSTITUTION.search(name):
            return cited_reply(name, filename, page)
        return None

    in_education = False
    institutions = {}
    for page, text in pages:
        for line in text.splitlines():
            if EDUCATION_HEADING.fullmatch(line):
                in_education = True
            elif RESUME_HEADING.fullmatch(line):
                in_education = False
            elif in_education and not re.search(rf"\b(?:{CLASS_12}|high school|class\s+(?:10|x))\b", line, re.I):
                for field in re.split(r"\s+[-–—|]\s+|\s+at\s+|,\s+", line):
                    if INSTITUTION.search(field):
                        name = re.split(r"\s+(?:\(?(?:19|20)\d{2}\b|[|•])", field, maxsplit=1)[0].strip()
                        if len(name.split()) >= 2 and INSTITUTION.search(name):
                            institutions.setdefault(name, page)
    if not institutions:
        return None
    return {
        "reply": "\n".join(institutions),
        "sources": [{"filename": filename, "page": page} for page in sorted(set(institutions.values()))],
    }


def project_titles(lines: list[str]) -> list[str]:
    def looks_like_title(line: str) -> bool:
        return bool(line and line[0].isupper() and len(line) <= 160 and not line.endswith((".", "!", "?", ":")))

    if not lines or not looks_like_title(lines[0]):
        return []
    first_bullet = next((i for i, line in enumerate(lines) if BULLET.match(line)), None)
    if first_bullet is None or BULLET.fullmatch(lines[-1]):
        return []

    # The first title can have a technology line and summary before its bullets.
    # Later titles introduce another bullet group. A marker on its own line is
    # followed by description text, even when that text has no final full stop.
    titles = [lines[0]]
    for index in range(first_bullet + 1, len(lines) - 1):
        if (looks_like_title(lines[index]) and BULLET.match(lines[index + 1])
                and not BULLET.fullmatch(lines[index - 1])):
            previous = lines[index - 1]
            title = previous if PROJECT_STACK_LINE.fullmatch(lines[index]) and looks_like_title(previous) else lines[index]
            titles.append(title)
    return titles


def answer_structured_question(question: str, *, filename: str | None = None) -> dict | None:
    normalized = " ".join(question.casefold().split()).rstrip(" ?.!")
    local_setup = LOCAL_SETUP_QUESTION.search(normalized)
    education_year = re.search(r"\b(?:year|when)\b", normalized) and re.search(
        rf"\b(?:{CLASS_12}|(?:my|completed|finished|passed)\s+12)\b", normalized
    )
    project_count = re.match(r"^(?:how many|count(?: the)?|(?:total )?number of) projects\b", normalized)
    project_names = re.fullmatch(
        r"(?:(?:what (?:is|are)|what's|list|show(?: me)?|tell me) )?"
        r"(?:the |my |our )?(?:projects?|projects? names?|names? of (?:my |the |our )?projects?)",
        normalized,
    )
    if project_count and re.search(r"\band\b.*\bnames?\b", normalized):
        project_names = True
    if not (local_setup or education_year or project_count or project_names):
        return None

    rows = indexed_passages(filename=filename)
    for page, text in page_texts(rows):
        setup = LOCAL_SETUP_HEADING.search(text) if local_setup else None
        if setup and text[setup.end():].strip() and not re.search(
            r"(?mi)^(?:table of )?contents:?$", text[:setup.start()]
        ):
            return cited_reply(text[setup.start():], rows[0][0], page)
        education = EDUCATION_HEADING.search(text) if education_year else None
        if education:
            lines = text[education.end():].splitlines()
            for line, following in zip(lines, lines[1:] + [""]):
                if not re.match(rf"{CLASS_12}\b", line, re.I):
                    continue
                if re.search(r"\b(?:19|20)\d{2}\b", line):
                    return cited_reply(line, rows[0][0], page)
                # A separate date line can follow an entry. Leave that layout to retrieval.
                if re.match(r"(?:high school|class\s+(?:10|x))\b", following, re.I):
                    return cited_reply(
                        "The PDF lists your Class 12 (Intermediate) entry without a completion year.",
                        rows[0][0], page,
                    )
        if project_count or project_names:
            heading = PROJECT_HEADING.search(text)
            if not heading:
                continue
            end = next((
                match for match in UPPERCASE_HEADING.finditer(text, heading.end() + 1)
                if not BULLET.match(text[match.end() + 1:])
            ), None)
            if not end and page != rows[-1][1]:
                continue
            section_end = end.start() if end else len(text)
            lines = text[heading.end():section_end].strip().splitlines()
            titles = project_titles(lines)
            if titles:
                reply = f"The PDF lists {len(titles)} {'project' if len(titles) == 1 else 'projects'}."
                if project_names:
                    reply += "\n" + "\n".join(f"- {title}" for title in titles)
                return cited_reply(reply, rows[0][0], page)
    return None


def cited_reply(reply: str, filename: str, page: int) -> dict:
    return {"reply": reply, "sources": [{"filename": filename, "page": page}]}


def question_terms(question: str) -> list[str]:
    return list(dict.fromkeys(
        word for word in re.findall(r"[^\W_]+", question.casefold())
        if word not in STOP_WORDS and (len(word) > 2 or word.isdigit())
    ))


def asks_about_technology(question: str) -> bool:
    return bool(re.search(r"\b(?:technolog(?:y|ies)|tech|stack)\b", question, re.IGNORECASE))


def search_passages(question: str, *, filenames: set[str] | None = None) -> list[tuple[str, int, str]]:
    tokens = question_terms(question)
    if asks_about_technology(question):
        tokens = list(dict.fromkeys([*tokens, *TECHNOLOGY_TERMS]))
    query = " OR ".join(f'"{word}"' for word in tokens[:24])
    if not query:
        return []
    clause = f" AND filename IN ({','.join('?' for _ in filenames)})" if filenames else ""
    with open_index() as connection:
        ranked = connection.execute(
            f"SELECT filename, page, text, rank FROM passages WHERE passages MATCH ?{clause} ORDER BY rank, rowid LIMIT 6",
            (query, *sorted(filenames or ())),
        ).fetchall()
    if tracing():
        trace_event("keyword", query=question, fts_query=query, rows=[
            {"score": score, "passage": (filename, page, text)} for filename, page, text, score in ranked
        ])
    return [(filename, page, text) for filename, page, text, _ in ranked]


def has_indexed_document(filename: str) -> bool:
    with open_index() as connection:
        return connection.execute("SELECT 1 FROM passages WHERE filename = ? LIMIT 1", (filename,)).fetchone() is not None


def indexed_passages(*, filename: str | None = None) -> list[tuple[str, int, str]]:
    with open_index() as connection:
        if filename is not None:
            return connection.execute(
                "SELECT filename, page, text FROM passages WHERE filename = ? ORDER BY rowid", (filename,),
            ).fetchall()
        return connection.execute("SELECT filename, page, text FROM passages ORDER BY rowid").fetchall()


def search_excerpts(question: str) -> list[tuple[str, int, str]]:
    terms = set(question_terms(question))
    if asks_about_technology(question):
        terms.update(TECHNOLOGY_TERMS)
    candidates = {}
    for filename, page, passage in search_passages(question):
        for text in re.split(r"(?<=[.!?])\s+(?=[A-Z]|\d+\.)", passage):
            text = text.strip()
            if len(text) < 12:
                continue
            matches = terms & set(question_terms(text))
            if matches:
                candidates.setdefault(text, (len(matches), filename, page, text))
    ranked = sorted(candidates.values(), key=lambda row: -row[0])
    return [
        (filename, page, text) for _, filename, page, text in ranked
        if text.endswith((".", "!", "?")) or not any(
            other.startswith(text + " ") for other in candidates
        )
    ][:14]


def rollout_excerpts(question: str, excerpts: list[tuple[str, int, str]]) -> list[tuple[str, int, str]]:
    if not re.match(
        r"^(?:do|did|will|are|is) we (?:use|using|have)\b",
        " ".join(question.casefold().split()),
    ):
        return []
    terms = question_terms(question)
    if len(terms) != 1:
        return []
    feature = re.escape(terms[0])
    now = next(
        (row for row in excerpts if re.search(
            rf"^No\s+{feature}\b[^.!?]*\bfor now\b", row[2], re.IGNORECASE
        )),
        None,
    )
    later = next(
        (row for row in excerpts if row != now and re.search(
            rf"\b{feature}\b[^.!?]*\b(?:added|enabled|introduced)\s+"
            r"(?:later|when [^.!?]*\bready\b)",
            row[2], re.IGNORECASE,
        )),
        None,
    )
    return [now, later] if now and later else []
