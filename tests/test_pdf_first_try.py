import asyncio
import json

import httpx

from backend import documents, main, rag_search
from backend.rag_trace import capture_trace
from test_documents import sample_pdf


FIELD_GUIDE_PAGES = (
    "North Pier field guide\nThe North Pier duty coordinator is Luis Ortiz. He checks the pier after sunset.",
    "Shift assignments\nZone | Lead | Pager\nOuter gate | Priya Rao | 214\nGenerator bay | Luis Ortiz | 390",
    "Storm preparation\n- Close the outer gate.\n- Move portable sensors to the shed.\n- Notify the on-call lead.",
    "Revision history\nThe shift schedule was updated in 2024.",
    "Storage areas\nThe east shed contains portable sensors and spare marker flags.",
    "Backup power\nThe generator bay contains one backup generator.",
)

FIELD_GUIDE_CASES = (
    (
        "Who checks the pier after sunset?",
        "Luis Ortiz",
        ("The North Pier duty coordinator is Luis Ortiz. He checks the pier after sunset.",),
        (1,),
    ),
    ("What is the pager number for the generator bay?", "390", ("Generator bay | Luis Ortiz | 390",), (2,)),
    (
        "What should we do to prepare for a storm?",
        "Prepare the pier.",
        ("Close the outer gate.", "Move portable sensors to the shed.", "Notify the on-call lead."),
        (3,),
    ),
    (
        "How many backup generators does the pier have?",
        "One",
        ("The generator bay contains one backup generator.",),
        (6,),
    ),
    ("In what year was the backup generator installed?", None, (), ()),
)


def test_tinygpt_reader_cites_split_table_rows_and_declines_missing_facts(monkeypatch, tmp_path):
    from backend import document_selector

    class TableReader:
        def rank(self, question, candidates):
            return [3.0 if "17 Mirror Lake Trailhead" in text else -1.0 for text in candidates]

    checkpoint = tmp_path / "reader.pt"
    checkpoint.touch()
    monkeypatch.setattr(document_selector, "CHECKPOINT", checkpoint)
    monkeypatch.setattr(document_selector, "cached_reader", lambda _: TableReader())
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "documents.sqlite3")

    async def unexpected_model(*args, **kwargs):
        raise AssertionError("The TinyGPT route must use only its local evidence scorer.")

    monkeypatch.setattr(main, "ask_ollama", unexpected_model)

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            upload = await client.post("/documents", params={"filename": "map.pdf"}, content=sample_pdf(
                "The trailhead parking lot opens at sunrise.",
                "Stops\n17\nMirror Lake\nTrailhead\n18\nLower Pines\nCampground",
            ))
            upload.raise_for_status()
            for question in ("Which stop number serves Mirror Lake?", "Whcih shutle stop numbr servse Mirror Lake?"):
                response = await client.post("/documents/ask", json={"prompt": question, "reader": "tinygpt"})
                assert response.json() == {"reply": "17 Mirror Lake Trailhead", "sources": [{"filename": "map.pdf", "page": 2}]}
            missing = await client.post("/documents/ask", json={
                "prompt": "What year was the first passport issued?", "reader": "tinygpt",
            })
            assert missing.json()["sources"] == []

    asyncio.run(check())


def test_citation_labels_and_heading_punctuation_do_not_change_source_pages():
    passages = [
        ("guide.pdf", 5, "Loading bay access\nThe orange clearance badge allows entry."),
        ("guide.pdf", 6, "Show it to the shift steward before moving any cargo."),
    ]
    assert rag_search.verified_quotes([
        "(page 2) Loading bay access: The orange clearance badge allows entry.",
        "Show it to the shift steward before moving any cargo. [1]",
        "(page 5) The orange badge is valid for 30 days.",
    ], passages) == [
        ("guide.pdf", 5, "Loading bay access The orange clearance badge allows entry."),
        ("guide.pdf", 6, "Show it to the shift steward before moving any cargo."),
    ]


def test_quote_spacing_cannot_change_source_words_or_numbers():
    passages = [("report.pdf", 1, "December,2024\nDr.Sanjay Roka\nThe weight is 64.5 g.")]
    assert rag_search.verified_quotes(
        ["December, 2024", "Dr. Sanjay Roka", "The weight is 65.4 g.", "December, 2025"], passages
    ) == [("report.pdf", 1, "December,2024"), ("report.pdf", 1, "Dr.Sanjay Roka")]


def test_evidence_ids_cite_only_the_selected_source_span(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_EVIDENCE_IDS", "1")
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "documents.sqlite3")
    filename = "service-guide.pdf"
    documents.index_pdf(filename, sample_pdf("The meeting room is Birch 12.", "The repair desk is in Cedar 204."))
    responses = iter([
        '{"answer":"Cedar 204","evidence_ids":[2,2]}',
        '{"answer":"Cedar 204","evidence_ids":[2,999]}',
        '{"answer":null,"evidence_ids":[2]}',
    ])

    async def ollama(prompt, instructions, *, response_format):
        assert "evidence_ids" in response_format["properties"]
        assert "[2] (page 2) The repair desk is in Cedar 204." in prompt
        return next(responses)

    monkeypatch.setattr(main, "ask_ollama", ollama)

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            for supported in (True, False, False):
                response = await client.post("/documents/ask", json={"prompt": "Where is the repair desk?"})
                response.raise_for_status()
                if supported:
                    assert response.json() == {
                        "reply": "The repair desk is in Cedar 204.", "sources": [{"filename": filename, "page": 2}],
                    }
                else:
                    assert response.json()["sources"] == []
                    assert "couldn’t find" in response.json()["reply"]

    asyncio.run(check())


def test_section_chunks_keep_the_original_page_text():
    text = "\n".join(
        f"{number}. Component specification\n" + ("A technical sentence supplies context. " * 12).strip()
        for number in range(1, 6)
    )
    chunks = list(documents.page_chunks(text, section_aware=True))
    assert len(chunks) > 1
    assert all(len(chunk) <= 1000 for chunk in chunks)
    assert list(documents.page_texts([("guide.pdf", 1, chunk) for chunk in chunks])) == [(1, text)]


def test_retrieval_preserves_evidence_for_each_comparison_and_records_scores(monkeypatch):
    alpha = ("report.pdf", 3, "Alpha runs at 10 units and is documented in the test report.")
    beta = ("report.pdf", 7, "Beta runs at 18 units and is documented in the test report.")
    others = tuple(("report.pdf", page, f"Unrelated evidence for page {page}. " * 18) for page in (1, 2, 5))
    monkeypatch.setattr(rag_search, "indexed_passages", lambda: (alpha, beta, *others))
    monkeypatch.setattr(rag_search, "search_passages", lambda query: [alpha] if "Alpha" in query else [beta])

    async def semantic(question, rows):
        return []

    monkeypatch.setattr(rag_search, "semantic_passages", semantic)
    question = "Compare the throughput of Alpha and Beta."
    with capture_trace() as events:
        result = asyncio.run(rag_search.retrieve(
            question, max_full_text_bytes=240, subquestions=tuple(rag_search.comparison_queries(question)),
        ))
    assert result == [alpha, beta]
    [retrieval] = events
    assert [row["query"] for row in retrieval["rows"]] == ["throughput of Alpha", "throughput of Beta"]
    assert all(row["keyword_rank"] == 1 and row["score"] > 0 for row in retrieval["rows"])


def test_tinygpt_checks_table_values_and_preserves_supported_parts(monkeypatch, tmp_path):
    from backend import document_selector

    class Reader:
        def rank(self, question, candidates):
            return [-0.25] * len(candidates)

    checkpoint = tmp_path / "reader.pt"
    checkpoint.touch()
    monkeypatch.setattr(document_selector, "CHECKPOINT", checkpoint)
    monkeypatch.setattr(document_selector, "cached_reader", lambda _: Reader())
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "documents.sqlite3")
    documents.index_pdf("lab.pdf", sample_pdf(
        "Leather workshop\nBring a mallet and a work apron to the leather workshop.",
        "Class prices\nClass | Monthly Fee | Schedule\nLeather workshop | $40 | Tuesday\n"
        "Glass workshop | $25 | Thursday",
        "Equipment log\nThe room sensor passed inspection in 2024.",
        "Wetland history\nThe delta wet phase occurred about 3.1-2.8 million years ago.",
        "Publication history\nThe workshop bulletin was printed on May 9, 2024.",
    ))

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            for question, expected_pages, fragments in (
                ("What should I bring to the leather workshop and what is the fee?",
                 [1, 2], ("mallet", "$40")),
                ("What should I bring to the leather workshop and what year was the room sensor installed?",
                 [1], ("mallet", "I couldn’t find evidence", "room sensor installed")),
                ("What is the annual fee for the leather workshop?", [], ("couldn’t find",)),
                ("What is the monthly fee for the digital workshop?", [], ("couldn’t find",)),
                ("What date range is given for the delta wet phase?", [4], ("3.1-2.8 million years",)),
                ("How far back does the wetland record place the delta wet phase?", [4], ("3.1-2.8 million years",)),
            ):
                response = await client.post("/documents/ask", json={"prompt": question, "reader": "tinygpt"})
                response.raise_for_status()
                result = response.json()
                assert result["sources"] == [{"filename": "lab.pdf", "page": page} for page in expected_pages]
                assert all(fragment in result["reply"] for fragment in fragments)

    asyncio.run(check())


def test_print_year_does_not_support_opening_year():
    quote = [("guide.pdf", 1, "The grounds opened to the public. This booklet was printed in 2024.")]
    assert not rag_search.supports_years("2024", "What year did the grounds open?", quote)
    assert rag_search.supports_years("2024", "Which year was the booklet published?", quote)


def test_short_pdf_uses_every_page_on_first_question(monkeypatch, tmp_path):
    filename = "north-pier-field-guide.pdf"
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "documents.sqlite3")
    remaining_cases = iter(FIELD_GUIDE_CASES)

    def no_ranking(*args, **kwargs):
        raise AssertionError("A short PDF should reach the model without ranking pages.")

    async def ollama(prompt, instructions, *, response_format):
        question, answer, quotes, _ = next(remaining_cases)
        assert question in prompt
        positions = [prompt.index(page) for page in FIELD_GUIDE_PAGES]
        assert positions == sorted(positions)
        if answer == "Luis Ortiz":
            quotes = [f"North Pier field guide: {quotes[0]}"]
        if answer == "390":
            quotes = ["(2) Shift assignments\nPager: 390"]
        if answer is None:
            answer = "2024"
            quotes = ["Revision history: The shift schedule was updated in 2024."]
        return json.dumps({"answer": answer, "quotes": quotes})

    monkeypatch.setattr(rag_search, "search_passages", no_ranking)
    monkeypatch.setattr(rag_search, "semantic_passages", no_ranking)
    monkeypatch.setattr(main, "ask_ollama", ollama)

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            upload = await client.post(
                "/documents", params={"filename": filename}, content=sample_pdf(*FIELD_GUIDE_PAGES)
            )
            assert upload.status_code == 200
            assert upload.json()["document"]["pages"] == len(FIELD_GUIDE_PAGES)
            for question, _, quotes, pages in FIELD_GUIDE_CASES:
                response = await client.post("/documents/ask", json={"prompt": question})
                assert response.status_code == 200
                result = response.json()
                assert result["sources"] == [{"filename": filename, "page": page} for page in pages]
                if quotes:
                    assert all(quote in result["reply"] for quote in quotes)
                else:
                    assert "couldn’t find" in result["reply"]

    asyncio.run(check())


def test_long_pdf_limits_context_to_retrieved_evidence(monkeypatch, tmp_path):
    filename = "estuary-maintenance-log.pdf"
    reference = "Apply sealant to the east dock on Thursdays."
    pages = [
        f"Unrelated maintenance ledger {number}\n" + "Routine inspection details. " * 65
        for number in range(1, 25)
    ]
    pages[11] = "Dock work calendar\n" + reference
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "documents.sqlite3")
    ranked = []

    async def semantic_passages(question, rows):
        ranked.append(question)
        assert any(page == 12 and reference in text for _, page, text in rows)
        return []

    async def ollama(prompt, instructions, *, response_format):
        assert reference in prompt
        assert "Unrelated maintenance ledger 1\n" not in prompt
        assert "Unrelated maintenance ledger 24\n" not in prompt
        return json.dumps({"answer": "Thursdays", "quotes": [reference]})

    monkeypatch.setattr(rag_search, "semantic_passages", semantic_passages)
    monkeypatch.setattr(main, "ask_ollama", ollama)

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            upload = await client.post("/documents", params={"filename": filename}, content=sample_pdf(*pages))
            assert upload.status_code == 200
            question = "When should sealant be applied to the east dock?"
            response = await client.post("/documents/ask", json={"prompt": question})
            assert response.status_code == 200
            assert response.json() == {"reply": reference, "sources": [{"filename": filename, "page": 12}]}
            assert ranked == [question]

    asyncio.run(check())
