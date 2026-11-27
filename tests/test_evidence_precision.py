import asyncio

import httpx

from backend import documents, main, rag_search
from backend.evidence_precision import focus_excerpts, metric_supported, verified_brief
from benchmarks.pdf_qa import make_pdf


def test_focus_excerpts_keeps_qualifiers_and_drops_unrelated_passages():
    evidence = [
        ("kiosk.pdf", 1,
         "Only if your supervisor approves, use the VIOLET-736 access code. "
         "The storage closet is on the third floor."),
        ("kiosk.pdf", 2, "The loading bay uses an unrelated electronic lock."),
    ]
    assert focus_excerpts("What is the kiosk access code?", "VIOLET-736", evidence) == [
        ("kiosk.pdf", 1, "Only if your supervisor approves, use the VIOLET-736 access code.")
    ]


def test_focus_excerpts_preserves_workflow_and_vague_answer_fallback():
    guide = [("kiosk.pdf", 1, "Before connecting the reader, turn the kiosk off. "
              "1. Connect the reader. 2. Turn the kiosk back on.")]
    assert focus_excerpts("How do I connect the reader?", "Connect the reader.", guide) == guide
    settings = [("kiosk.pdf", 2, "The faculty endpoint is staff.example.edu. The guest endpoint is guest.example.edu.")]
    assert focus_excerpts("Which endpoint is used by faculty?", "Faculty", settings) == settings


def test_only_supporting_page_is_cited_for_a_scoped_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("PERSONAL_AI_COLLECTION", "1")
    monkeypatch.setenv("IT_SUPPORT_PRODUCT", "1")
    monkeypatch.setenv("RAG_EVIDENCE_IDS", "1")
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "it.sqlite3")
    documents.index_pdf("scanner.pdf", make_pdf([
        "The scanner asset tag is IRIS-734. Return packaging to the supply cupboard.",
        "Look up supply stock in the inventory app.",
    ]))

    async def ollama(prompt, instructions, *, response_format):
        assert "[1] (scanner.pdf, page 1)" in prompt
        return '{"answer":"The scanner asset tag is IRIS-734.","evidence_ids":[1,2]}'

    monkeypatch.setattr(main, "ask_ollama", ollama)

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            response = await client.post("/documents/ask", json={
                "prompt": "What is the scanner asset tag?", "pdf_filename": "scanner.pdf",
            })
            assert response.raise_for_status().json() == {
                "reply": "The scanner asset tag is IRIS-734.",
                "sources": [{"filename": "scanner.pdf", "page": 1}],
            }

    asyncio.run(check())


def test_start_question_leaves_out_actions_after_completion():
    guide = [
        ("token.pdf", 1, "1. Open your account. 2. Enter your username. 3. Choose Bind token."),
        ("token.pdf", 2, "4. This may take a minute. After enrolling, open the dashboard."),
    ]
    answer = "Open your account, enter your username, bind the token, and open the dashboard."
    assert focus_excerpts("How do I start enrolling a token?", answer, guide) == [
        ("token.pdf", 1, "1. Open your account."),
        ("token.pdf", 1, "2. Enter your username."),
        ("token.pdf", 1, "3. Choose Bind token."),
    ]
    assert focus_excerpts("What are the steps for enrolling a token?", answer, guide) == guide


def test_password_requirements_keep_the_source_list_when_the_model_paraphrases():
    guide = [("account.pdf", 3, "• Have at least 1 number • Have at least 1 special character")]
    assert focus_excerpts(
        "What does the guide require for digits and special characters?",
        "It requires one digit and one special character.",
        guide,
    ) == guide

def test_brief_answers_keep_approximation_and_scope():
    results = [("model.pdf", 3,
                "Cedar got the highest accuracy (~98%), followed by Alder (96%) and Birch (93%).")]
    assert verified_brief("What accuracy did Cedar achieve?", "98%", results) == [
        ("model.pdf", 3, "~98%"),
    ]
    assert metric_supported("accuracy", "What accuracy did Alder achieve?", "96%", results)
    assert not metric_supported("accuracy", "What accuracy did Alder achieve?", "93%", results)
    assert not metric_supported("f1score", "What was Alder's F1-score?", "96%", results)
    mixed_metrics = [("model.pdf", 3, "Reported 96% classification accuracy and evaluated precision and recall.")]
    assert metric_supported("accuracy", "What accuracy was reported?", "96%", mixed_metrics)
    assert not metric_supported("precision", "What precision was reported?", "96%", mixed_metrics)
    assert verified_brief("What is the code? Short answer.", "AB-321", [
        ("setup.pdf", 2, "The code is AB-321 only if your manager approves."),
    ]) == []


def test_brief_verified_list_requires_every_item_and_keeps_source_order():
    snippets = [("tools.pdf", 2, "The tools are Pandas and NumPy for data, and Matplotlib/Seaborn for charts.")]
    assert verified_brief("List the libraries.", "NumPy, Pandas, Seaborn, Matplotlib", snippets) == [
        ("tools.pdf", 2, "Pandas, NumPy, Matplotlib, Seaborn"),
    ]
    assert verified_brief("List the libraries.", "Pandas, NumPy, MissingPackage", snippets) == []


def test_retrieval_restores_adjacent_selected_page_text_within_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("PERSONAL_AI_COLLECTION", "1")
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "index.sqlite3")
    filename = "options.pdf"
    page = "The supported libraries are Pandas. " + "label " * 150 + " Matplotlib is also supported."
    documents.index_text_pages(filename, [(1, page), (2, "unrelated " * 300)], section_aware=False)
    tail = documents.indexed_passages(filename=filename)[1]

    async def ranked(question, rows, known, *, scope):
        assert scope == {filename}
        return [{"score": 1, "passage": tail}]

    monkeypatch.setattr(rag_search, "hybrid_ranking", ranked)
    result = asyncio.run(rag_search.retrieve(
        "Which libraries are supported?", max_full_text_bytes=1300, pdf_filename=filename,
    ))
    assert len(result) == 1
    assert result[0][:2] == (filename, 1)
    assert all(name in result[0][2] for name in ("Pandas", "Matplotlib"))


def test_metric_retries_wrong_score_once_and_abstains_for_missing_metric(monkeypatch, tmp_path):
    monkeypatch.setenv("PERSONAL_AI_COLLECTION", "1")
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "index.sqlite3")
    guide = "scores.pdf"
    documents.index_pdf(guide, make_pdf([
        "Cedar got the highest accuracy (~98%), followed by Alder (96%) and Birch (93%).",
    ]))
    replies = iter(("93%", "96%", "96%", "96%"))

    async def ask(prompt, instructions, *, response_format):
        return '{"evidence_ids":[1],"answer":"' + next(replies) + '"}'

    monkeypatch.setattr(main, "ask_ollama", ask)

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            for prompt, expected in (
                ("What accuracy did Alder achieve?", {
                    "reply": "96%", "sources": [{"filename": guide, "page": 1}],
                }),
                ("What was Alder's exact F1-score?", {
                    "reply": "I couldn’t find an answer in the selected PDF.", "sources": [],
                }),
            ):
                response = await client.post("/documents/ask", json={"prompt": prompt, "pdf_filename": guide})
                assert response.raise_for_status().json() == expected

    asyncio.run(check())

def test_resume_shortcut_reads_only_selected_pdf(monkeypatch, tmp_path):
    monkeypatch.setenv("PERSONAL_AI_COLLECTION", "1")
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "index.sqlite3")
    documents.index_pdf("other.pdf", make_pdf(["TECHNICAL SKILLS\nDatabases: SQLite\nEXPERIENCE\nWork sample"]))
    documents.index_pdf("selected.pdf", make_pdf([
        "TECHNICAL SKILLS\nDatabases: PostgreSQL, MongoDB, SQL\nEXPERIENCE\nWork sample",
    ]))

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            response = await client.post("/documents/ask", json={
                "prompt": "Which databases are listed?", "pdf_filename": "selected.pdf",
            })
            assert response.raise_for_status().json() == {
                "reply": "Databases: PostgreSQL, MongoDB, SQL",
                "sources": [{"filename": "selected.pdf", "page": 1}],
            }

    asyncio.run(check())


def test_project_title_can_be_followed_by_a_technology_list():
    assert documents.project_titles([
        "Warehouse stock dashboard", "React, PostgreSQL", "•", "Built the dashboard.",
        "Notebook classifier", "Python, CNN", "•", "Evaluated its predictions.",
    ]) == ["Warehouse stock dashboard", "Notebook classifier"]
