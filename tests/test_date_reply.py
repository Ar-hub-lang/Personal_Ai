import asyncio
from datetime import date

import pytest

from backend import main


@pytest.mark.parametrize("question", ["what is the date today", "What’s today’s date?"])
def test_current_date_uses_local_clock_without_ollama(monkeypatch, question):
    class LocalDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 11, 23)

    monkeypatch.setattr(main, "date", LocalDate)
    monkeypatch.setattr(main, "OLLAMA_CHAT_URL", "unused://ollama")

    assert asyncio.run(main.generate_reply(question)) == "Today is Monday, November 23, 2026."
