import asyncio
from io import BytesIO
from zipfile import ZipFile

import httpx

from backend import documents, main
from backend.document_intake import index_document
from test_documents import sample_pdf


def test_collection_index_preserves_files_and_cites_the_named_guide(monkeypatch, tmp_path):
    monkeypatch.setenv("PERSONAL_AI_COLLECTION", "1")
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "it.sqlite3")
    index_document("staff.pdf", sample_pdf("The dashboard is at Cedar 204."), aliases=("Example University",))
    index_document("guest.pdf", sample_pdf("The dashboard is at Ash 217."), aliases=("Other College",))
    index_document("guest.pdf", sample_pdf("The dashboard is at Ash 250."))
    assert len(documents.collection_documents()) == 2
    assert documents.collection_scope("Where is the Other College dashboard?") == {"guest.pdf"}

    async def ollama(prompt, instructions, *, response_format):
        assert "staff.pdf" in prompt and "guest.pdf" not in prompt
        return '{"answer":"Cedar 204","evidence_ids":[1]}'

    monkeypatch.setattr(main, "ask_ollama", ollama)

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            response = await client.post("/documents/ask", json={"prompt": "Where is the Example University dashboard?"})
            assert response.json()["sources"] == [{"filename": "staff.pdf", "page": 1}]
            deleted = await client.delete("/documents", params={"filename": "guest.pdf"})
            assert [item["filename"] for item in deleted.json()["documents"]] == ["staff.pdf"]

    asyncio.run(check())


def test_docx_index_cites_real_section_titles(monkeypatch, tmp_path):
    monkeypatch.setenv("PERSONAL_AI_COLLECTION", "1")
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "it.sqlite3")
    word = BytesIO()
    with ZipFile(word, "w") as archive:
        archive.writestr("word/document.xml", """
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Printer Setup</w:t></w:r></w:p>
            <w:p><w:r><w:t>Open the printer queue and select North Wing.</w:t></w:r></w:p>
          </w:body>
        </w:document>
        """)
    index_document("guide.docx", word.getvalue())
    assert documents.indexed_passages() == [("guide.docx", 1, "Printer Setup\nOpen the printer queue and select North Wing.")]
    assert documents.source_metadata("guide.docx", 1)["section"] == "Printer Setup"


def test_single_pdf_scope_overrides_alias_and_separates_chat_history(monkeypatch, tmp_path):
    monkeypatch.setenv("PERSONAL_AI_COLLECTION", "1")
    monkeypatch.setenv("IT_SUPPORT_PRODUCT", "1")
    monkeypatch.setenv("PERSONAL_AI_MEMORY_DB", str(tmp_path / "chat.sqlite3"))
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "it.sqlite3")
    index_document("A & B.PDF", sample_pdf("The helpdesk ticket prefix is MAPLE."), aliases=("Campus A",))
    index_document("campus-b.pdf", sample_pdf("The helpdesk ticket prefix is CEDAR."), aliases=("Campus B",))
    index_document("notes.md", b"The helpdesk ticket prefix is BIRCH.")
    calls = []

    async def ollama(prompt, instructions, *, response_format):
        calls.append(prompt)
        assert "CEDAR" not in prompt and "BIRCH" not in prompt and "campus-b.pdf" not in prompt
        if "summarize" in prompt.casefold():
            return '{"answer":null,"evidence_ids":[]}'
        return '{"answer":"MAPLE","evidence_ids":[1]}'

    monkeypatch.setattr(main, "ask_ollama", ollama)
    monkeypatch.setattr(main, "synthesize", lambda _: b"test audio")

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            scoped = {"prompt": "What is the Campus B helpdesk ticket prefix?", "pdf_filename": "A & B.PDF"}
            response = await client.post("/documents/ask", json=scoped)
            assert response.json()["sources"] == [{"filename": "A & B.PDF", "page": 1}]
            assert "MAPLE" in response.json()["reply"]
            stream = await client.post("/documents/ask/stream", json=scoped)
            assert '"filename": "A & B.PDF"' in stream.text
            voice = await client.post("/voice/reply", json=scoped)
            assert voice.json()["sources"] == [{"filename": "A & B.PDF", "page": 1}]
            scoped_history = await client.get("/conversations/current", params={"pdf_filename": "A & B.PDF"})
            assert len(scoped_history.json()["turns"]) == 2
            assert (await client.get("/conversations/current")).json()["turns"] == []
            other_history = await client.get("/conversations/current", params={"pdf_filename": "campus-b.pdf"})
            assert other_history.json()["turns"] == []
            missing = await client.post("/chat", json=scoped | {"prompt": "Summarize this PDF."})
            assert missing.json() == {"reply": "I couldn’t find an answer in the selected PDF.", "sources": []}
            invalid = await client.post("/documents/ask", json=scoped | {"pdf_filename": "notes.md"})
            assert invalid.status_code == 422
            await client.delete("/documents", params={"filename": "A & B.PDF"})
            removed = await client.post("/documents/ask", json=scoped)
            assert removed.status_code == 404
            assert len(calls) == 4

    asyncio.run(check())


def test_single_pdf_scope_is_enforced_in_hybrid_retrieval(monkeypatch, tmp_path):
    from backend import rag_search
    from benchmarks.pdf_qa import make_pdf

    monkeypatch.setenv("PERSONAL_AI_COLLECTION", "1")
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "it.sqlite3")
    index_document("staff.pdf", make_pdf([
        f"Ticket ref STAFF-{index}. " + "Open the helpdesk for a printer ticket. " * 6
        for index in range(1, 8)
    ]))
    index_document("guest.pdf", sample_pdf("Printer ticket ref GUEST-1."), aliases=("Guest",))

    async def semantic(question, rows):
        assert rows and all(row[0] == "staff.pdf" for row in rows)
        return list(rows[:1])

    monkeypatch.setattr(rag_search, "semantic_passages", semantic)
    rows = asyncio.run(rag_search.retrieve(
        "Which printer ticket is used for Guest?", max_full_text_bytes=600, pdf_filename="staff.pdf",
    ))
    assert rows and all(row[0] == "staff.pdf" for row in rows)
    assert asyncio.run(rag_search.retrieve(
        "Which printer ticket is used for Guest?", max_full_text_bytes=600, pdf_filename="missing.pdf",
    )) == []
