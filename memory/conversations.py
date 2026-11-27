"""Persist the browser's IT Support conversation separately from approved knowledge."""

import json
import os
import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from backend import documents

SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
SUMMARY = re.compile(r"\b(?:summari[sz]e|summary|recap)\b.*\b(?:conversation|chat|session|discussion)\b", re.I)
RECALL = re.compile(
    r"\b(?:i (?:told|gave|shared|mentioned|said)|(?:first|earlier|previous) (?:chat )?message"
    r"|did i (?:say|tell|share|mention|give))\b", re.I,
)
STOP = set("what which where when how was were did does that this the about told tell gave give "
           "mentioned mention shared share first earlier previous message chat conversation said say "
           "my me you for from ask asked session user is it in a an".split())


@contextmanager
def open_history():
    path = Path(os.environ.get("PERSONAL_AI_MEMORY_DB") or documents.DATABASE_PATH.with_name("chat.sqlite3"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.row_factory = sqlite3.Row
        connection.execute(
            """CREATE TABLE IF NOT EXISTS support_chat_turns (
                id INTEGER PRIMARY KEY, conversation TEXT NOT NULL, prompt TEXT NOT NULL,
                reply TEXT NOT NULL, sources TEXT NOT NULL
            )"""
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS support_chat_conversation ON support_chat_turns(conversation, id)"
        )
        yield connection


def read_turns(conversation: str, limit: int = 100) -> list[dict]:
    with open_history() as connection:
        rows = connection.execute(
            """SELECT prompt, reply, sources FROM support_chat_turns
               WHERE conversation = ? ORDER BY id DESC LIMIT ?""", (conversation, limit),
        ).fetchall()
    return [{"prompt": row["prompt"], "reply": row["reply"], "sources": json.loads(row["sources"])}
            for row in reversed(rows)]


def save_turn(conversation: str, prompt: str, result: dict) -> None:
    with open_history() as connection:
        connection.execute(
            "INSERT INTO support_chat_turns (conversation, prompt, reply, sources) VALUES (?, ?, ?, ?)",
            (conversation, prompt, result["reply"], json.dumps(result["sources"], ensure_ascii=False)),
        )


def summarize(turns: list[dict]) -> str:
    highlights = {}
    for turn in turns:
        sentences = [part.strip() for part in SENTENCES.split(turn["prompt"]) if part.strip()]
        for sentence in sentences[:2]:
            if not sentence.endswith("?") and not SUMMARY.search(sentence):
                highlights.setdefault(sentence.casefold(), sentence)
    if not highlights:
        return "There are no user statements to summarize yet."
    return "Conversation summary\n" + "\n".join(f"• {text}" for text in highlights.values())


def recall(prompt: str, turns: list[dict]) -> str | None:
    if not RECALL.search(prompt):
        return None
    if not turns:
        return "I couldn’t find that in this conversation."
    if re.search(r"\bfirst (?:chat )?message\b", prompt, re.I):
        return f"Your first message was “{turns[0]['prompt']}”"
    keywords = set(re.findall(r"[^\W\d_]+", prompt.casefold())) - STOP
    best = ""
    best_score = 0
    for turn in reversed(turns):
        for sentence in SENTENCES.split(turn["prompt"]):
            terms = set(re.findall(r"[^\W\d_]+", sentence.casefold()))
            score = len(keywords & terms)
            if not sentence.endswith("?") and not RECALL.search(sentence) and score > best_score:
                best, best_score = sentence, score
    return f"You said “{best.strip()}”" if best else "I couldn’t find that in this conversation."
