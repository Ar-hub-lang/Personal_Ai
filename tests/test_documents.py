import asyncio
from io import BytesIO

import httpx
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from backend import documents, main, rag_search


def sample_pdf(*pages: str) -> bytes:
    writer = PdfWriter()
    for text in pages or ("The study plan has three sessions.", "The workshop room is Cedar 204."):
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        contents = DecodedStreamObject()
        contents.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
        page.replace_contents(contents)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_upload_and_answer_with_verified_citation(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "documents.sqlite3")
    calls = []

    async def no_semantic(question, rows):
        return []

    monkeypatch.setattr(rag_search, "semantic_passages", no_semantic)

    async def ollama(prompt, instructions, *, response_format):
        calls.append(prompt)
        if "Question: What is the workshop room?" in prompt:
            return '{"answer":"Cedar 204","quotes":["The workshop room is Cedar 204."]}'
        if "What year" in prompt:
            return '{"answer":"2023","quotes":["The release was tested in Chrome."]}'
        if "Question: What tests are included?" in prompt:
            return '{"answer":"Chrome","quotes":["The release was tested in Chrome."]}'
        if "Question: Who approved the budget?" in prompt:
            return '{"answer":null,"quotes":[]}'
        return '{"answer":"Six sessions","quotes":["The plan has six sessions."]}'

    monkeypatch.setattr(main, "ask_ollama", ollama)

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            upload = await client.post("/documents", params={"filename": "plan.pdf"}, content=sample_pdf(
                "The study plan has three sessions.",
                "The workshop room is Cedar 204.",
                "The release was tested in Chrome.",
            ))
            assert upload.json() == {"document": {"filename": "plan.pdf", "pages": 3}}
            invalid = await client.post("/documents", params={"filename": "bad.pdf"}, content=b"bad")
            assert invalid.status_code == 422
            assert (await client.get("/documents")).json() == upload.json()

            missing = await client.post("/documents/ask", json={"prompt": "Who approved the budget?"})
            assert missing.json()["sources"] == []
            assert len(calls) == 1

            answer = await client.post("/documents/ask", json={"prompt": "What is the workshop room?"})
            assert answer.json() == {
                "reply": "The workshop room is Cedar 204.",
                "sources": [{"filename": "plan.pdf", "page": 2}],
            }

            tests = await client.post("/documents/ask", json={"prompt": "What tests are included?"})
            assert tests.json() == {
                "reply": "The release was tested in Chrome.",
                "sources": [{"filename": "plan.pdf", "page": 3}],
            }

            wrong_passage = await client.post("/documents/ask", json={"prompt": "What does the study plan say?"})
            assert wrong_passage.json()["sources"] == []

            unsupported_year = await client.post("/documents/ask", json={"prompt": "What year was the release tested?"})
            assert unsupported_year.json()["sources"] == []

    asyncio.run(check())


def test_structured_pdf_keeps_project_titles_and_education_dates_separate(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "documents.sqlite3")
    documents.index_pdf("example.pdf", sample_pdf(
        "PROJECTS\nWeather App\n- Implemented alerts.\n- Wrote tests.\nVISITOR MAP\n- Added a map.\n"
        "EDUCATION\nBachelor of Computing 2023-2026\nIntermediate - Example School | 75%\nHigh School - Example School",
        "Contents\nLocal setup\nSee the quickstart.",
        "RUNBOOK\nLocal setup\nInstall the dependencies.\nnpm run dev",
    ))
    assert documents.answer_structured_question("How many projects did I build?")["reply"] == "The PDF lists 2 projects."
    assert documents.answer_structured_question("My project name")["reply"] == (
        "The PDF lists 2 projects.\n- Weather App\n- VISITOR MAP"
    )
    assert documents.answer_structured_question("In which year i have done my 12?") == {
        "reply": "The PDF lists your Class 12 (Intermediate) entry without a completion year.",
        "sources": [{"filename": "example.pdf", "page": 1}],
    }
    assert documents.answer_structured_question("How do I run it locally?") == {
        "reply": "Local setup\nInstall the dependencies.\nnpm run dev",
        "sources": [{"filename": "example.pdf", "page": 3}],
    }


def test_resume_name_and_college_questions_cite_indexed_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "documents.sqlite3")
    documents.index_pdf("candidate.pdf", sample_pdf(
        "PRIYA\nSoftware Developer\nEmail: priya@example.com\nPROFESSIONAL SUMMARY\nBuilds local tools.\n"
        "EDUCATION\nBachelor of Computing - Cedar University\n2023 - 2026\n"
        "Intermediate - Hill College | 80%\nPROJECTS\nWorked with Lake University.",
    ))

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            for question, value in (
                ("What is name?", "PRIYA"),
                ("Who is this CV for?", "PRIYA"),
                ("Waht is the naem on this CV?", "PRIYA"),
                ("What is the name of College?", "Cedar University"),
                ("Whch univresity did I attend?", "Cedar University"),
                ("Where did I study?", "Cedar University"),
                ("What’s my clg naem?", "Cedar University"),
            ):
                response = await client.post("/documents/ask", json={"prompt": question, "reader": "tinygpt"})
                assert response.json() == {"reply": value, "sources": [{"filename": "candidate.pdf", "page": 1}]}

    asyncio.run(check())
    documents.index_pdf("guide.pdf", sample_pdf("CITY GUIDE\nContact: help@example.com\nPLAN\nInformation for visitors."))
    assert documents.answer_resume_question("What is name?") is None


def test_resume_projects_with_wrapped_descriptions_and_standalone_bullets(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "documents.sqlite3")
    documents.index_pdf("candidate.pdf", sample_pdf(
        "PROJECTS\nNeighborhood Shop\nReact, Node.js, PostgreSQL\nOnline store built for local makers.\n"
        "-\nCreated a catalog and payment flow\n-\nDeployed the frontend with a\ncustom domain.\n"
        "Transit Tracker\n-\nShows bus departures from local timetables.\nEDUCATION\nCedar University",
    ))

    for question in ("how many projects he made", "Total number of projects"):
        assert documents.answer_structured_question(question) == {
            "reply": "The PDF lists 2 projects.",
            "sources": [{"filename": "candidate.pdf", "page": 1}],
        }
    assert documents.answer_structured_question("projects name") == {
        "reply": "The PDF lists 2 projects.\n- Neighborhood Shop\n- Transit Tracker",
        "sources": [{"filename": "candidate.pdf", "page": 1}],
    }


def test_resume_skills_queries_use_the_labelled_section(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "documents.sqlite3")
    documents.index_pdf("candidate.pdf", sample_pdf(
        "PROFESSIONAL SUMMARY\nBuilt an application from database design to deployment.\n"
        "TECHNICAL SKILLS\nFrontend: HTML, CSS\nDatabase: MongoDB, SQL,",
        "PostgreSQL\nProgramming: Python\nEXPERIENCE\nBuilt applications with Redis.",
    ))
    sources = [{"filename": "candidate.pdf", "page": page} for page in (1, 2)]
    database = "Database: MongoDB, SQL,\nPostgreSQL"
    skills = f"Frontend: HTML, CSS\n{database}\nProgramming: Python"

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            for question, reply in (
                ("Database", database),
                ("databse", database),
                ("datadse", database),
                ("Which databse does he use?", database),
                ("Which databases does he use?", database),
                ("Technical Skills", skills),
                ("tecnical skils", skills),
            ):
                response = await client.post("/documents/ask", json={"prompt": question, "reader": "tinygpt"})
                assert response.status_code == 200
                assert response.json() == {"reply": reply, "sources": sources}

    asyncio.run(check())


def test_tinygpt_returns_pdf_submitter_names_with_stable_citations(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "documents.sqlite3")
    documents.index_pdf("report.pdf", sample_pdf(
        "ENVIRONMENTAL FIELD REPORT\nSubmitted by\nNora Patel and Ivan Torres\n"
        "Under the guidance of\nDr. Cedar\nDEPARTMENT OF ENVIRONMENTAL SCIENCE",
        "Certificate\nsubmitted by Nora Patel, to Cedar University.\nName of Guide: Dr. Cedar",
    ))

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            previous = None
            for question in (
                "who is submitted by", "who is submitted by", "Who submitted this report?",
                "Who was it submitted by?", "By whom was the report submited?",
            ):
                response = await client.post("/documents/ask", json={
                    "prompt": question, "reader": "tinygpt", "previous_question": previous,
                })
                assert response.status_code == 200
                assert response.json() == {
                    "reply": "Nora Patel and Ivan Torres",
                    "sources": [{"filename": "report.pdf", "page": 1}],
                }
                previous = question

    asyncio.run(check())


def test_submitter_lookup_accepts_inline_credits_and_skips_deadlines():
    assert documents.document_submitter([
        (1, "REPORT ON RIVER HEALTH Submitted by Asha Rao Under the guidance of Dr. Cedar"),
    ]) == (1, "Asha Rao")
    assert documents.document_submitter([
        (1, "Under the guidance of Dr. Cedar\nAll forms must be submitted by Monday."),
        (2, "The report was submitted by J. K. Rivera on June 3."),
    ]) == (2, "J. K. Rivera")
    assert documents.document_submitter([
        (1, "Applications must be submitted by June 3.\nSubmitted by: Pending"),
        (2, "Submitted by Monday\nSupervisor: Dr. Cedar"),
    ]) is None


def test_numbered_section_across_pages_excludes_next_heading(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "documents.sqlite3")
    documents.index_pdf(
        "example.pdf",
        sample_pdf(
            "Rollout Plan Launch the website first. Phase 2: Safety Features 9. Add an alert.",
            "10. Verify reports. Support Features 11. Offer chat help. Phase 3: Later 12. Add email.",
        ),
    )
    safety = documents.answer_section("What are the safety features?")
    assert safety == {
        "reply": "9. Add an alert.\n10. Verify reports.",
        "sources": [{"filename": "example.pdf", "page": 1}, {"filename": "example.pdf", "page": 2}],
    }
    assert documents.answer_section("How many total number of safety features in it?") == {
        **safety, "reply": "There are 2 entries under Safety Features."
    }
    assert documents.answer_section("What are the support features?")["reply"] == "11. Offer chat help."
    assert documents.answer_section("What is the rollout plan?")["reply"] == "Launch the website first."


def test_library_list_stops_at_the_next_numbered_section(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "documents.sqlite3")
    documents.index_pdf("guide.pdf", sample_pdf(
        "3. Libraries and Dependencies:\n- Matplotlib: Used for plotting figures.\n"
        "- Pillow: Used for loading images.\n4. Environment Management:\n- Anaconda: Used for packaging.",
    ))
    assert documents.answer_library_list("Which library is used in this project?") == {
        "reply": "3. Libraries and Dependencies:\n- Matplotlib: Used for plotting figures.\n- Pillow: Used for loading images.",
        "sources": [{"filename": "guide.pdf", "page": 1}],
    }
    assert documents.answer_section("Who approved the rollout plan?") is None


def test_title_uses_printed_heading_and_technology_search_expands_terms(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "documents.sqlite3")
    documents.index_pdf(
        "Workshop Feature List.pdf",
        sample_pdf(
            "Workshop: Feature List The project starts with a website.",
            "PWA first, then wrap the working PWA with Capacitor for Android and iOS.",
        ),
    )
    assert documents.answer_document_title("Name of project") == {
        "reply": "The PDF calls the project “Workshop” in the heading “Workshop: Feature List.”",
        "sources": [{"filename": "Workshop Feature List.pdf", "page": 1}],
    }
    excerpts = documents.search_excerpts("Which technology we used in it")
    assert any(page == 2 and "Capacitor" in text for _, page, text in excerpts)


def test_rollout_excerpts_show_current_and_later_status(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "documents.sqlite3")
    documents.index_pdf("plan.pdf", sample_pdf(
        "No MFA verification for now.", "MFA verification will be added later."
    ))
    question = "Do we use MFA in our project?"
    excerpts = documents.search_excerpts(question)
    assert documents.rollout_excerpts(question, excerpts) == [
        ("plan.pdf", 1, "No MFA verification for now."),
        ("plan.pdf", 2, "MFA verification will be added later."),
    ]


def test_retrieve_uses_semantic_matches_and_followup_context(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "documents.sqlite3")
    documents.index_pdf("plan.pdf", sample_pdf(
        "The session is in Cedar 204. Bring a pencil.",
        "Unrelated background about weather reports and seasonal conditions. " * 140,
    ))
    queries = []

    async def semantic(question, rows):
        queries.append(question)
        return list(rows)

    async def ollama(prompt, instructions, *, response_format):
        assert "Where is the workshop?" in prompt
        return '{"answer":"Cedar 204","quotes":["The session is in Cedar 204."]}'

    monkeypatch.setattr(rag_search, "semantic_passages", semantic)
    monkeypatch.setattr(main, "ask_ollama", ollama)

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            answer = await client.post(
                "/documents/ask",
                json={"prompt": "Where is it?", "previous_question": "Where is the workshop?"},
            )
            assert answer.json() == {
                "reply": "The session is in Cedar 204.",
                "sources": [{"filename": "plan.pdf", "page": 1}],
            }
    asyncio.run(check())
    assert queries == ["Where is it?\nTopic from the previous question: Where is the workshop?"]


def test_optional_standalone_rewrite_uses_the_previous_question(monkeypatch):
    async def ollama(prompt, instructions, *, response_format):
        assert "Where is the workshop?" in prompt
        assert "Where is it?" in prompt
        assert response_format["title"] == "StandaloneQuestion"
        return '{"question":"Where is the workshop?"}'

    monkeypatch.setattr(main, "ask_ollama", ollama)
    assert asyncio.run(main.rewrite_document_question("Where is it?", "Where is the workshop?")) == "Where is the workshop?"
    assert asyncio.run(main.rewrite_document_question("Where is the workshop?", "Where is it?")) is None
