import asyncio

import httpx

from backend import main
from memory import store
from speech.spelling import correct_spelling


def test_correct_spelling_matches_complete_words_and_longest_phrases():
    assert correct_spelling("My name is Arpitu.") == "My name is Arpit."
    assert correct_spelling("My name is Arpetu.") == "My name is Arpit."
    assert correct_spelling("Use FAST A P I with s q l lite.") == "Use FastAPI with SQLite."
    assert correct_spelling("Arpitus ate an apple.") == "Arpitus ate an apple."
    assert correct_spelling("Recieve a reply.") == "Receive a reply."
    assert correct_spelling("The univeristy has a useful artcile.") == (
        "The university has a useful article."
    )
    assert correct_spelling("The university has a useful article.") == (
        "The university has a useful article."
    )


def test_voice_uses_corrected_transcript_for_reply(monkeypatch):
    monkeypatch.setattr(main, "saved_name", lambda: None)
    monkeypatch.setattr(main, "voice_names", lambda category: [])
    monkeypatch.setattr(main, "transcribe", lambda audio, name: "My name is Arpitu.")
    monkeypatch.setattr(main, "synthesize", lambda reply: b"audio")

    async def reply(prompt):
        return f"Received: {prompt}"

    monkeypatch.setattr(main, "generate_reply", reply)

    async def send():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            return await client.post("/voice", content=b"recording")

    response = asyncio.run(send())
    assert response.status_code == 200
    assert response.json()["transcript"] == "My name is Arpit."
    assert response.json()["reply"] == "Received: My name is Arpit."


def test_indian_names_and_places_use_the_spelling_dictionary():
    assert correct_spelling("State uttarpradesh, district meerut.") == (
        "State Uttar Pradesh, district Meerut."
    )
    assert correct_spelling("District dehra doon, post office rishikesh.") == (
        "District Dehradun, post office Rishikesh."
    )
    assert correct_spelling("Village sisai bolan, district hansi, state haryana.") == (
        "Village Sisai Bolan, district Hansi, state Haryana."
    )
    assert correct_spelling("My name is priyanka. I met someone named rahul.") == (
        "My name is Priyanka. I met someone named Rahul."
    )


def test_corrects_misheard_saharanpur_introduction():
    expected = "I am from Saharanpur, Uttar Pradesh. My village name is Haripur and post is Behat."
    assert correct_spelling(
        "I am from Saharanpur, Uttargadesh. My village name is Haripoor and post is Beheet."
    ) == expected
    assert correct_spelling(
        "I am from Saharanpur, Udhar Pradesh. My village name is Harryper and post is Behat."
    ) == expected


def test_everyday_phrases_correct_names_without_rewriting_ordinary_words():
    assert correct_spelling(
        "My name is Arpithu. My post is Bade and district is Chharanpur."
    ) == "My name is Arpit. My post is Behat and district is Saharanpur."
    assert correct_spelling("What is my intro? Where i am located?") == (
        "What is my introduction? Where am I located?"
    )
    assert correct_spelling("What is my inter duction? Start the back end server.") == (
        "What is my introduction? Start the backend server."
    )
    assert correct_spelling("He bade farewell at the calender machine in Charanpur.") == (
        "He bade farewell at the calender machine in Charanpur."
    )


def test_corrects_full_name_variants_and_preserves_distinct_surnames():
    assert correct_spelling("Call anu bhav toe mer.") == "Call Anubhav Tomer."
    assert correct_spelling("His name is enough half-tomer.") == "His name is Anubhav Tomer."
    assert correct_spelling("I spoke to anubhav tomar.") == "I spoke to Anubhav Tomar."
    assert correct_spelling("Ask aditi sharma and simran kaur.") == (
        "Ask Aditi Sharma and Simran Kaur."
    )


def test_voice_review_waits_for_approval_and_hints_the_recognizer(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DATABASE_PATH", tmp_path / "assistant.sqlite3")
    monkeypatch.setattr(main, "synthesize", lambda reply: b"audio")

    def transcribe(audio, hint):
        assert hint == "Person name: Anubhav Tomer."
        return "Remember that my friend is Anubov Thomas."

    monkeypatch.setattr(main, "transcribe", transcribe)

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            approved = await client.post("/chat", json={"prompt": "Remember voice name: Anubhav Tomer"})
            assert approved.status_code == 200
            assert "Anubhav Tomer" in approved.json()["reply"]

            draft = await client.post("/voice/transcribe", content=b"recording")
            assert draft.json() == {"transcript": "Remember that my friend is Anubov Thomas."}
            assert store.recent_memories() == []

            sent = await client.post(
                "/voice/reply", json={"prompt": "Remember that my friend is Anubhav Tomer."}
            )
            assert sent.status_code == 200
            assert sent.json()["reply"] == "I saved this in memory: my friend is Anubhav Tomer."

    asyncio.run(check())
    assert store.recent_memories() == ["my friend is Anubhav Tomer."]


def test_voice_name_groups_filter_hints_and_keep_legacy_names(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DATABASE_PATH", tmp_path / "assistant.sqlite3")
    store.remember_voice_name("Priya Sharma")
    hints = []

    def transcribe(audio, hint):
        hints.append(hint)
        return "Please review my notes."

    monkeypatch.setattr(main, "transcribe", transcribe)

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            for group, name in (("friends", "Anubhav Tomer"), ("family", "आरव शर्मा")):
                saved = await client.post(
                    "/chat", json={"prompt": f"Remember voice name for {group}: {name}"}
                )
                assert saved.json()["reply"] == f"I saved {name} in the {group} group for Voice mode."

            for group in ("friends", "family", "contacts", "auto"):
                response = await client.post(
                    "/voice/transcribe", params={"category": group}, content=b"recording"
                )
                assert response.json() == {"transcript": "Please review my notes."}
            invalid = await client.post(
                "/voice/transcribe", params={"category": "invalid"}, content=b"recording"
            )
            assert invalid.status_code == 422
            listing = await client.post("/chat", json={"prompt": "List voice names for friends"})
            assert listing.json()["reply"] == "Recent voice names: Anubhav Tomer"

    asyncio.run(check())
    assert hints == [
        "Person name: Anubhav Tomer.",
        "Person name: आरव शर्मा.",
        "",
        "Person name: आरव शर्मा. Anubhav Tomer. Priya Sharma.",
    ]
