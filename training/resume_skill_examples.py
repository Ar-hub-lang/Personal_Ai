"""Create labeled skills questions using fictional résumé sections."""

import random


PROFILES = {
    "train": (
        ("MySQL", "Java", "Angular", "Spring Boot"),
        ("SQLite", "Python", "Vue", "Flask"),
        ("MariaDB", "JavaScript", "React", "Express"),
        ("Redis", "Go", "Svelte", "Gin"),
        ("Cassandra", "Scala", "Lit", "Play"),
        ("DynamoDB", "TypeScript", "Preact", "NestJS"),
        ("CockroachDB", "Ruby", "Stimulus", "Rails"),
        ("Oracle Database", "C#", "Blazor", "ASP.NET Core"),
        ("IBM Db2", "Kotlin", "Ember", "Ktor"),
        ("HBase", "Clojure", "Reagent", "Pedestal"),
        ("Neo4j", "PHP", "Alpine.js", "Laravel"),
        ("Microsoft SQL Server", "F#", "Elm", "Giraffe"),
    ),
    "validation": (
        ("Firebird", "Dart", "Flutter Web", "Dart Frog"),
        ("ArangoDB", "Elixir", "Phoenix LiveView", "Phoenix"),
        ("ClickHouse", "Swift", "OpenSilver", "Vapor"),
        ("OpenSearch", "Erlang", "Htmx", "Cowboy"),
    ),
    "test": (
        ("DuckDB", "Rust", "Leptos", "Axum"),
        ("Couchbase", "Lua", "Mithril", "Lapis"),
        ("TimescaleDB", "Haskell", "Reflex", "Servant"),
        ("TiDB", "C++", "Qwik", "Drogon"),
    ),
}

DATABASE_LABELS = ("Database", "Databases", "Database tools", "Database skills")
DATABASE_QUESTIONS = (
    "databse", "datadse", "Which database does this candidate use?",
    "What datbase tools are listed?", "Show the databses in the skills section.",
    "Which databsae technlogies are listed?", "What are the database skills?",
    "Which storage systems are in the technical skills?",
)
SKILLS_QUESTIONS = (
    "Technical Skills", "tecnical skils", "Show me the technical skills.",
    "List the candidate's core skills.",
)
PROJECT_SNIPPETS = (
    "Designed and tested the database for a library reservation app.",
    "Migrated the reporting database and verified recovery procedures.",
    "Built a data dashboard from database design to deployment.",
    "Automated a check for database connection errors in the support portal.",
)


def resume_skill_examples(split: str, abstain: int) -> list[tuple[str, list[str], int]]:
    rng = random.Random({"train": 61, "validation": 91, "test": 131}[split])
    rows = []

    def add(question: str, passages: list[str], answer: str | None) -> None:
        rng.shuffle(passages)
        rows.append((question, passages, abstain if answer is None else passages.index(answer)))

    for index, (database, language, frontend, backend) in enumerate(PROFILES[split]):
        label = DATABASE_LABELS[index % len(DATABASE_LABELS)]
        database_row = f"{label}: {database}"
        language_row = f"Programming languages: {language}"
        frontend_row = f"Frontend: {frontend}"
        backend_row = f"Backend: {backend}"
        project = PROJECT_SNIPPETS[index % len(PROJECT_SNIPPETS)]
        skills = f"Technical Skills\n{language_row}\n{database_row}\n{frontend_row}\n{backend_row}"

        for question in DATABASE_QUESTIONS[index % 4::4]:
            add(question, [project, language_row, database_row, backend_row], database_row)
        add("Which programing langauges are listed?", [project, database_row, language_row, frontend_row], language_row)
        add("Which frontend frameworks are listed?", [project, backend_row, database_row, frontend_row], frontend_row)
        add(SKILLS_QUESTIONS[index % len(SKILLS_QUESTIONS)],
            [project, language_row, database_row, skills], skills)
        add("Which database product is named?", [project, language_row, frontend_row, backend_row], None)
        add("Which machine learning libraries are listed?", [project, skills, database_row, backend_row], None)

    return rows
