import asyncio
import json

import httpx

from backend import main
from memory import store


def test_forget_memory_removes_only_the_exact_fact(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DATABASE_PATH", tmp_path / "assistant.sqlite3")
    fact = "I prefer 25-minute study sessions."
    other = "I prefer 25-minute study sessions on Sundays."
    store.remember(fact)
    store.remember(other)

    assert asyncio.run(main.generate_reply("Forget memory:")).startswith("In Chat")
    assert asyncio.run(main.generate_reply(f"Forget memory: {fact.lower()}")).startswith(
        "I couldn’t find that exact fact."
    )
    assert store.recent_memories() == [other, fact]
    assert asyncio.run(main.generate_reply(f"Forget memory: {fact}")) == (
        "I removed that fact from memory."
    )
    assert store.recent_memories() == [other]
    store.remember(fact)
    assert asyncio.run(main.generate_reply(f"Forget memory : {fact}")) == (
        "I removed that fact from memory."
    )
    assert store.recent_memories() == [other]


def test_memory_saves_only_explicit_requests_and_passes_facts_to_ollama(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DATABASE_PATH", tmp_path / "assistant.sqlite3")
    requests = []

    def ollama(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "A short reply."}})

    client_type = httpx.AsyncClient
    transport = httpx.MockTransport(ollama)
    monkeypatch.setattr(
        main.httpx, "AsyncClient", lambda **kwargs: client_type(transport=transport, **kwargs)
    )

    fact = "I prefer 25-minute study sessions."
    assert asyncio.run(main.generate_reply(fact)) == "A short reply."
    assert store.recent_memories() == []

    saved = asyncio.run(main.generate_reply(f"Remember that {fact}"))
    assert saved == f"I saved this in memory: {fact}"
    assert store.recent_memories() == [fact]
    recalled = asyncio.run(main.generate_reply("What do you remember about me?"))
    assert recalled.endswith(fact)
    assert len(requests) == 1

    asyncio.run(main.generate_reply("How long do I like to study?"))
    assert fact in requests[-1]["messages"][0]["content"]


def test_saved_name_is_available_after_other_memories(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DATABASE_PATH", tmp_path / "assistant.sqlite3")
    assert store.saved_name() is None

    store.remember("My name is Arpitu.")
    store.remember("my name is Arpit.")
    for index in range(9):
        store.remember(f"Study preference {index}.")

    assert "my name is Arpit." not in store.recent_memories()
    assert store.saved_name() == "Arpit"


def test_address_update_replaces_old_addresses_in_chat_and_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DATABASE_PATH", tmp_path / "assistant.sqlite3")
    monkeypatch.setattr(main, "OLLAMA_CHAT_URL", "unused://ollama")
    store.remember("My address is: 2 Park Lane.")
    store.remember("I live at 5 Green Lane.")
    store.remember("I prefer 25-minute study sessions.")

    monkeypatch.setattr(main, "transcribe", lambda audio, name: "Update my address to 14 Rose Lane.")
    monkeypatch.setattr(main, "synthesize", lambda reply: b"audio")

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            empty = await client.post("/chat", json={"prompt": "Update my address to."})
            assert empty.json()["reply"].startswith("Say or type")
            for prompt in (
                "I have updated my address to myself Arpit. I am from Saharanpur, Uttar Pradesh.",
                "Update my address to myself Arpit.",
            ):
                result = await client.post("/chat", json={"prompt": prompt})
                assert result.json()["reply"] == main.ADDRESS_UPDATE_HELP
            assert store.current_address() == "I live at 5 Green Lane."
            voice = await client.post("/voice", content=b"recording")
            assert voice.json()["reply"] == (
                "I updated your saved address. My current address is 14 Rose Lane."
            )
            chat = await client.post("/chat", json={"prompt": "What is my current address?"})
            assert chat.json()["reply"] == "My current address is 14 Rose Lane."

    asyncio.run(check())
    assert store.recent_memories() == [
        "My current address is 14 Rose Lane.",
        "I prefer 25-minute study sessions.",
    ]


def test_update_memory_removes_only_the_matching_old_fact(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DATABASE_PATH", tmp_path / "assistant.sqlite3")
    old = "I prefer 25-minute study sessions."
    new = "I prefer 40-minute study sessions."
    unrelated = "I like to study in the morning."
    store.remember(old)
    store.remember(new)
    store.remember(unrelated)

    missing = asyncio.run(main.generate_reply("Update memory: Unknown fact. -> New fact."))
    assert missing.startswith("I couldn’t find that exact fact.")
    assert store.recent_memories() == [unrelated, new, old]

    confirmation = asyncio.run(main.generate_reply(f"Update memory: {old} -> {new}"))
    assert confirmation == f"I replaced the old memory with: {new}"
    assert store.recent_memories() == [new, unrelated]


def test_update_name_removes_older_name_statements_only(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DATABASE_PATH", tmp_path / "assistant.sqlite3")
    store.remember("My name is Arpithu.")
    store.remember("My name is Rp2.")
    store.remember("I prefer to study in the morning.")

    for prompt in ("Update my name to Rp2.", "Update my name, my name is not Rp2."):
        assert asyncio.run(main.generate_reply(prompt)) == main.NAME_UPDATE_HELP
    assert len(store.recent_memories()) == 3

    assert asyncio.run(main.generate_reply("Update my name to Arpit.")) == (
        "I updated your saved name to Arpit."
    )
    assert asyncio.run(main.generate_reply("What is my name?")) == "Your saved name is Arpit."
    assert store.recent_memories() == ["My name is Arpit.", "I prefer to study in the morning."]


def test_introduction_and_location_use_current_address_instead_of_old_intro(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DATABASE_PATH", tmp_path / "assistant.sqlite3")
    monkeypatch.setattr(main, "OLLAMA_CHAT_URL", "unused://ollama")
    store.replace_name("Arpit")
    address = "Village Haripur, Post Behat, District Saharanpur, Uttar Pradesh"
    store.replace_address(f"My current address is {address}.")
    store.remember("My introduction is: I live in Chharanpur and my post is Bade.")

    assert asyncio.run(main.generate_reply("What is my introduction?")) == (
        f"My name is Arpit. My address is {address}."
    )
    assert asyncio.run(main.generate_reply("where i am located ?")) == (
        f"Your saved address is {address}."
    )
