import os
import re
import sqlite3
import unicodedata
from contextlib import closing, contextmanager
from pathlib import Path

DATABASE_PATH = Path(
    os.environ.get("PERSONAL_AI_MEMORY_DB")
    or Path(__file__).with_name("assistant.sqlite3")
)
MAX_FACT_LENGTH = 240
ADDRESS_FILTER = """
    lower(fact) GLOB 'my address is[ :]*'
    OR lower(fact) GLOB 'my current address is[ :]*'
    OR lower(fact) GLOB 'my home address is[ :]*'
    OR lower(fact) GLOB 'i live at[ :]*'
"""
NAME_FILTER = "lower(fact) GLOB 'my name is[ :]*'"


@contextmanager
def open_database():
    with closing(sqlite3.connect(DATABASE_PATH)) as connection, connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                fact TEXT NOT NULL UNIQUE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS voice_names (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS voice_name_groups (
                name TEXT PRIMARY KEY COLLATE NOCASE,
                category TEXT NOT NULL CHECK (category IN ('family', 'friends', 'contacts'))
            )
            """
        )
        yield connection


def save_fact(connection: sqlite3.Connection, fact: str) -> None:
    connection.execute(
        """
        INSERT INTO memories (fact) VALUES (?)
        ON CONFLICT(fact) DO UPDATE SET id = excluded.id
        """,
        (fact,),
    )


def remember(fact: str) -> None:
    with open_database() as connection:
        save_fact(connection, fact)


def remember_voice_name(name: str, category: str = "auto") -> None:
    with open_database() as connection:
        connection.execute(
            """
            INSERT INTO voice_names (name) VALUES (?)
            ON CONFLICT(name) DO UPDATE SET id = excluded.id, name = excluded.name
            """,
            (name,),
        )
        if category != "auto":
            connection.execute(
                """
                INSERT INTO voice_name_groups (name, category) VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET category = excluded.category
                """,
                (name, category),
            )


def voice_names(category: str = "auto") -> list[str]:
    with open_database() as connection:
        rows = connection.execute(
            """
            SELECT names.name
            FROM voice_names AS names
            LEFT JOIN voice_name_groups AS groups ON names.name = groups.name
            WHERE ? = 'auto' OR groups.category = ?
            ORDER BY names.id DESC
            LIMIT 8
            """,
            (category, category),
        )
        return [name for (name,) in rows]


def replace_memory(previous: str, replacement: str) -> bool:
    with open_database() as connection:
        deleted = connection.execute("DELETE FROM memories WHERE fact = ?", (previous,))
        if not deleted.rowcount:
            return False
        save_fact(connection, replacement)
        return True


def forget_memory(fact: str) -> bool:
    with open_database() as connection:
        deleted = connection.execute("DELETE FROM memories WHERE fact = ?", (fact,))
        return deleted.rowcount > 0


def replace_address(fact: str) -> None:
    with open_database() as connection:
        connection.execute(f"DELETE FROM memories WHERE {ADDRESS_FILTER}")
        connection.execute("INSERT INTO memories (fact) VALUES (?)", (fact,))


def replace_name(name: str) -> None:
    with open_database() as connection:
        connection.execute(f"DELETE FROM memories WHERE {NAME_FILTER}")
        save_fact(connection, f"My name is {name}.")


def current_address() -> str | None:
    with open_database() as connection:
        row = connection.execute(
            f"SELECT fact FROM memories WHERE {ADDRESS_FILTER} ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row[0] if row else None


def recent_memories() -> list[str]:
    with open_database() as connection:
        rows = connection.execute("SELECT fact FROM memories ORDER BY id DESC LIMIT 8")
        return [fact for (fact,) in rows]


def valid_name(name: str) -> bool:
    return (
        len(name) <= 80
        and not name.casefold().startswith(("not ", "myself ", "my name is "))
        and all(
            word[:1].isalpha()
            and all(char.isalpha() or unicodedata.category(char).startswith("M") for char in word)
            for word in re.split(r"[ '\u2019-]", name)
        )
    )


def saved_name() -> str | None:
    with open_database() as connection:
        row = connection.execute(
            f"SELECT fact FROM memories WHERE {NAME_FILTER} ORDER BY id DESC LIMIT 1"
        ).fetchone()

    if row:
        name = row[0][len("my name is"):].strip(" :.!?")
        if valid_name(name):
            return name
    return None
