"""Evaluate IT answers after retrieving across the whole support collection."""

import argparse
import asyncio
import json
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from statistics import median
from time import perf_counter
from tempfile import TemporaryDirectory

import httpx

from backend import documents, main
from backend.document_intake import index_document
from backend.rag_trace import capture_trace
from benchmarks.rag_eval import grade


TYPOS = {
    "VPN": "VPNN", "GlobalProtect": "glboalprotect", "MFA": "MFAA",
    "password": "pasword", "printer": "pritner", "printing": "printng",
    "queue": "qeue", "eduroam": "eduram", "installation": "instalation",
    "software": "softwre", "disconnect": "disconect", "network": "netwrok",
}
TYPO_TERM = re.compile(r"\b(?:" + "|".join(TYPOS) + r")\b")


def grade_with_filename(question, body, traces, filename):
    scoped = []
    for event in traces:
        if event["stage"] == "retrieval":
            scoped.append({**event, "rows": [row for row in event["rows"] if row["passage"][0] == filename]})
        else:
            scoped.append(event)
    result = grade(question, body, scoped, 5, False)
    if body and question["answer"] is not None and any(source["filename"] != filename for source in body["sources"]):
        result.update(correct=False, failure="incorrect_source")
    return result


async def evaluate(root: Path, split: str, noisy: bool, index: Path | None, model: str | None = None) -> dict:
    model = model or main.CHAT_MODEL
    manifest = json.loads((root / "evaluation" / f"{split}.json").read_text(encoding="utf-8"))
    sources = json.loads((root / "sources.json").read_text(encoding="utf-8"))["documents"]
    cases = []
    with TemporaryDirectory(prefix="personal-ai-collection-eval-") as directory:
        documents.DATABASE_PATH = Path(directory) / "index.sqlite3"
        if index:
            with closing(sqlite3.connect(index.resolve().as_uri() + "?mode=ro", uri=True)) as original:
                with closing(sqlite3.connect(documents.DATABASE_PATH)) as copied:
                    original.backup(copied)
            if {item["filename"] for item in sources} - {item["filename"] for item in documents.collection_documents()}:
                raise ValueError("The supplied collection index is missing guides from sources.json.")
        else:
            for item in sources:
                filename = item["filename"]
                index_document(filename, (root / "documents" / filename).read_bytes(), aliases=tuple(item.get("aliases", ())))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://collection-test") as client:
            for document in manifest["documents"]:
                for question in document["questions"]:
                    prompt = question["question"]
                    if noisy:
                        prompt = TYPO_TERM.sub(lambda match: TYPOS[match[0]], prompt, count=1)
                    started = perf_counter()
                    with capture_trace() as trace:
                        response = await client.post("/documents/ask", json={"prompt": prompt, "reader": "qwen", "model": model})
                    duration = perf_counter() - started
                    body = response.json() if response.is_success else None
                    result = grade_with_filename(question, body, trace, document["filename"])
                    cases.append({"id": question["id"], "prompt": prompt, "typo_injected": prompt != question["question"],
                                  "expected_file": document["filename"], "response": body,
                                  "seconds": round(duration, 3), **result})
                    print(question["id"], "PASS" if result["correct"] else result["failure"], flush=True)
    def counts(key):
        values = [case[key] for case in cases if case[key] is not None]
        return [sum(values), len(values)]

    return {"model": model, "split": split, "summary": {"retrieval_hits": counts("hit"), "correct_answers": counts("correct"),
                        "unsupported_answers_on_missing_questions": counts("hallucinated"),
                        "typo_injected": sum(case["typo_injected"] for case in cases),
                        "runtime_errors": sum(case["failure"] == "runtime_error" for case in cases),
                        "median_seconds": round(median(case["seconds"] for case in cases), 3) if cases else None},
            "cases": cases}


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("collections/it-support"))
    parser.add_argument("--split", choices=("tuning", "holdout"), default="tuning")
    parser.add_argument("--noisy", action="store_true", help="Introduce one known typo per matching question.")
    parser.add_argument("--index", type=Path, help="Use an isolated copy of an existing collection index.")
    parser.add_argument("--model", choices=("qwen3:1.7b", "qwen3:4b"),
                        help="Select one model explicitly and disable the default fallback.")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    os.environ["PERSONAL_AI_COLLECTION"] = "1"
    result = asyncio.run(evaluate(args.root, args.split, args.noisy, args.index, args.model))
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main_cli()
