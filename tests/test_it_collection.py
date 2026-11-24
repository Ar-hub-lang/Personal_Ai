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
