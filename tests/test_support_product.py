import asyncio
import base64
import json
import subprocess

import httpx

from backend import documents, main
from speech.spelling import correct_spelling


def enable_product(monkeypatch, tmp_path):
    monkeypatch.setenv("PERSONAL_AI_COLLECTION", "1")
    monkeypatch.setenv("IT_SUPPORT_PRODUCT", "1")
    monkeypatch.setattr(documents, "DATABASE_PATH", tmp_path / "support.sqlite3")


def test_only_approved_knowledge_enters_support_index(monkeypatch, tmp_path):
    enable_product(monkeypatch, tmp_path)
    approval = {"X-Knowledge-Approval": "yes"}
    note = {"title": "VPN guide", "text": "# Setup\nOpen GlobalProtect.",
            "url": "https://support.example.org/vpn"}
    filename = "Website - VPN guide.md"

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            denied = await client.post("/documents/notes", json=note)
            assert denied.status_code == 403
            denied = await client.post("/documents", params={"filename": "other.txt"}, content=b"Unapproved")
            assert denied.status_code == 403
            assert documents.indexed_passages() == []
            invalid = await client.post("/documents/notes", json=note | {"url": "http://localhost/private"}, headers=approval)
            assert invalid.status_code == 422
            saved = await client.post("/documents/notes", json=note, headers=approval)
            assert saved.json()["support_product"] is True
            assert saved.json()["documents"][0]["url"] == note["url"]
            assert documents.source_metadata(filename, 1) == {
                "filename": filename, "page": 1, "section": "Setup", "url": note["url"]
            }
            updated = await client.post("/documents/notes", json=note | {"text": "# Revised\nContact support."}, headers=approval)
            assert updated.status_code == 200
            assert documents.indexed_passages() == [(filename, 1, "Revised\nContact support.")]
            deleted = await client.delete("/documents", params={"filename": filename})
            assert deleted.json()["documents"] == []
            assert documents.indexed_passages() == []

    asyncio.run(check())


def test_support_chat_stream_and_voice_use_same_cited_answer(monkeypatch, tmp_path):
    enable_product(monkeypatch, tmp_path)
    result = {"reply": "Open GlobalProtect.", "sources": [{"filename": "vpn.pdf", "page": 1}]}

    async def answer(request):
        assert main.requested_model.get() == "qwen3:4b"
        assert request.previous_question == "GlobalProtect VPN"
        return result

    def transcribe(audio, prompt):
        assert "GlobalProtect" in prompt
        return "How do I connect?"

    monkeypatch.setattr(main, "answer_document", answer)
    monkeypatch.setattr(main, "transcribe", transcribe)
    monkeypatch.setattr(main, "synthesize", lambda reply: b"wave audio")

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            draft = await client.post("/voice/transcribe", content=b"recording")
            body = {"prompt": draft.json()["transcript"], "model": "qwen3:4b", "previous_question": "GlobalProtect VPN"}
            chat = await client.post("/chat", json=body)
            assert chat.json() == result
            assert main.CONVERSATION_COOKIE in chat.cookies
            response = await client.post("/documents/ask/stream", json=body)
            status, *chunks, final = map(json.loads, response.text.splitlines())
            assert "".join(event["delta"] for event in chunks) == result["reply"]
            assert "status" in status and final == {"result": result}
            spoken = await client.post("/voice/reply", json=body)
            assert spoken.json() == result | {"audio": base64.b64encode(b"wave audio").decode()}
            assert "set-cookie" not in spoken.headers
            assert main.requested_model.get() is None
            assert (await client.post("/documents/ask", json=body | {"reader": "tinygpt"})).status_code == 422

            def broken_tts(reply):
                raise subprocess.CalledProcessError(1, "say")

            monkeypatch.setattr(main, "synthesize", broken_tts)
            text_only = await client.post("/voice/reply", json=body)
            assert text_only.json()["sources"] == result["sources"]
            assert text_only.json()["audio"] is None
            history = (await client.get("/conversations/current")).json()["turns"]
            assert len(history) == 4 and all(turn["prompt"] == body["prompt"] for turn in history)
            reset = await client.post("/conversations/new")
            assert reset.cookies[main.CONVERSATION_COOKIE] != chat.cookies[main.CONVERSATION_COOKIE]
            assert (await client.get("/conversations/current")).json()["turns"] == []

    asyncio.run(check())


def test_unavailable_default_uses_backup_but_explicit_comparison_does_not(monkeypatch):
    original_client = httpx.AsyncClient
    calls = []

    def ollama(request):
        payload = json.loads(request.content)
        model = payload["model"]
        calls.append((model, list(payload["format"]["properties"])))
        if model == "qwen3:4b":
            return httpx.Response(404)
        return httpx.Response(200, json={"message": {"content": "Fallback reply"}})

    def stub_client(**kwargs):
        return original_client(transport=httpx.MockTransport(ollama), **kwargs)

    monkeypatch.setattr(main, "CHAT_MODEL", "qwen3:4b")
    monkeypatch.setattr(httpx, "AsyncClient", stub_client)

    async def check():
        assert await main.ask_ollama("question", "instructions", response_format=main.EVIDENCE_IDS_FIRST_SCHEMA) == "Fallback reply"
        token = main.requested_model.set("qwen3:4b")
        try:
            try:
                await main.ask_ollama("question", "instructions", response_format=main.EVIDENCE_IDS_FIRST_SCHEMA)
            except main.HTTPException as failure:
                assert failure.status_code == 503
            else:
                raise AssertionError("An explicit comparison model must not silently fall back.")
        finally:
            main.requested_model.reset(token)

    asyncio.run(check())
    assert calls == [
        ("qwen3:4b", ["evidence_ids", "answer"]),
        ("qwen3:1.7b", ["answer", "evidence_ids"]),
        ("qwen3:4b", ["evidence_ids", "answer"]),
    ]


def test_it_transcript_terms_do_not_use_personal_spelling_guesses():
    assert correct_spelling(
        "How do I connect to the D-Field Global Protect VPN?", it_support=True
    ) == "How do I connect to the Deerfield GlobalProtect VPN?"
    assert correct_spelling("Open the paper cut print queue.", it_support=True) == "Open the PaperCut print queue."
    assert correct_spelling("I have a paper cut.", it_support=True) == "I have a paper cut."
    assert correct_spelling("D-Field has customers.", it_support=True) == "D-Field has customers."


def test_stream_shows_verified_parts_then_final_citations(monkeypatch, tmp_path):
    enable_product(monkeypatch, tmp_path)
    monkeypatch.setenv("RAG_EVIDENCE_IDS", "1")
    monkeypatch.setenv("RAG_PARTWISE_EVIDENCE", "1")
    passages = [("vpn.pdf", 1, "Use GlobalProtect."), ("vpn.pdf", 2, "Ask the helpdesk.")]

    async def retrieve(query, **kwargs):
        return passages

    async def select(evidence, question, suffix):
        return [evidence[0 if question.startswith("Which") else 1]]

    monkeypatch.setattr(main, "retrieve", retrieve)
    monkeypatch.setattr(main, "select_evidence_ids", select)

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            response = await client.post("/documents/ask/stream", json={"prompt": "Which VPN do I use and who can help?"})
            events = [json.loads(line) for line in response.text.splitlines()]
            parts = [entry["part"] for entry in events if "part" in entry]
            assert [part["sources"][0]["page"] for part in parts] == [1, 2]
            assert events[-1]["result"] == {
                "reply": "Use GlobalProtect.\nAsk the helpdesk.",
                "sources": [{"filename": "vpn.pdf", "page": 1}, {"filename": "vpn.pdf", "page": 2}],
            }

    asyncio.run(check())


def test_4b_orders_evidence_before_answer_for_it_guides(monkeypatch, tmp_path):
    enable_product(monkeypatch, tmp_path)
    monkeypatch.setenv("RAG_EVIDENCE_IDS", "1")
    passages = [("vpn.pdf", 1, "Use GlobalProtect.")]
    orders = []

    async def retrieve(query, **kwargs):
        return passages

    async def ask(prompt, instructions, *, response_format):
        orders.append(list(response_format["properties"]))
        return '{"evidence_ids":[1],"answer":"Use GlobalProtect"}'

    monkeypatch.setattr(main, "retrieve", retrieve)
    monkeypatch.setattr(main, "ask_ollama", ask)

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            for model in ("qwen3:4b", "qwen3:1.7b"):
                response = await client.post("/documents/ask", json={"prompt": "Which VPN do I use?", "model": model})
                assert response.json()["sources"] == [{"filename": "vpn.pdf", "page": 1}]
            monkeypatch.setattr(main, "CHAT_MODEL", "qwen3:4b")
            response = await client.post("/documents/ask", json={"prompt": "Which VPN do I use?"})
            assert response.json()["sources"] == [{"filename": "vpn.pdf", "page": 1}]

    asyncio.run(check())
    assert orders == [["evidence_ids", "answer"], ["answer", "evidence_ids"], ["evidence_ids", "answer"]]
