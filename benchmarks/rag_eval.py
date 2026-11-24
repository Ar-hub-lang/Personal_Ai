"""Measure local PDF retrieval, grounded answers, and abstention."""

import argparse
import asyncio
import json
import os
import re
import sqlite3
import unicodedata
from collections import Counter
from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx

from backend import documents, main, rag_search
from backend.rag_trace import capture_trace


class StopAtPrompt(Exception):
    """Finish diagnostics without asking the language model to generate."""


def normalized(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"(?<=\d)\s*(?:x|×|\*)\s*(?=\d)", "x", text)
    text = re.sub(r"(?<=\d)\s+(?=(?:gb|mb|%)\b|%)", "", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=[a-z])", "", text)
    return " ".join(text.replace("–", "-").replace("—", "-").split())


def contains(text, terms):
    return all(len(re.findall(rf"(?<!\w){re.escape(term)}(?!\w)", text)) >= count
               for term, count in Counter(normalized(value) for value in terms).items())


def evidence_hit(question, passages):
    # Count distinct evidence fragments on a source page even across adjacent chunks.
    by_page = {}
    for _, page, text in passages:
        by_page.setdefault(page, set()).add(normalized(text))
    return all(
        any(contains(" ".join(sorted(by_page.get(page, ()))), source["contains"]) for page in source["pages"])
        for source in question["sources"]
    )


def grade(question, body, trace, top_k, retrieval_only):
    events = [item for item in trace if item["stage"] == "retrieval"]
    passages = [
        row["passage"] for event in events
        for row in (event["rows"][:top_k] if event["mode"] == "tinygpt" else event["rows"])
    ]
    # Multi-part requests can answer one part through a structured shortcut.
    passages.extend(
        passage for event in trace if event["stage"] == "shortcut" for passage in event.get("evidence", ())
    )
    missing = question["answer"] is None
    hit = evidence_hit(question, passages) if events and not missing else None
    if retrieval_only:
        return {"hit": hit, "correct": None, "hallucinated": None}
    if body is None:
        return {"hit": hit, "correct": False, "hallucinated": None, "failure": "runtime_error"}
    pages = {source["page"] for source in body["sources"]}
    if missing:
        correct = not pages and "couldn’t find" in body["reply"]
        return {"hit": None, "correct": correct, "hallucinated": not correct,
                "failure": None if correct else "unsupported_answer"}
    expected_pages = set().union(*(source["pages"] for source in question["sources"]))
    cited = pages.issubset(expected_pages) and all(
        pages.intersection(source["pages"]) for source in question["sources"]
    )
    correct = cited and contains(normalized(body["reply"]), question["contains"])
    failure = None
    if not correct:
        model = next((event for event in trace if event["stage"] == "generation"), None)
        produced = model and model.get("answer") and contains(normalized(model["answer"]), question["contains"])
        if hit is False:
            failure = "retrieval"
        elif hit:
            failure = "verification_or_answer_assembly" if produced else "generation_or_selection"
        else:
            failure = "structured_lookup"
    return {"hit": hit, "correct": correct, "hallucinated": None, "failure": failure}


async def run(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["documents"]
    original_index = documents.DATABASE_PATH
    active = documents.active_document() if original_index.exists() else None
    provided = {path.name: path for path in args.pdf}
    for item in manifest:
        filename = item["filename"]
        if filename not in provided and (not active or active["filename"] != filename):
            raise SystemExit(f"Provide --pdf '/path/to/{filename}' or upload that PDF to the app first.")
    results = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://local-eval") as client:
        for item in manifest:
            filename = item["filename"]
            with TemporaryDirectory(prefix="personal-ai-rag-eval-") as directory:
                db = Path(directory) / "documents.sqlite3"
                with patch.object(documents, "DATABASE_PATH", db):
                    if path := provided.get(filename):
                        upload = await client.post("/documents", params={"filename": filename}, content=path.read_bytes())
                        upload.raise_for_status()
                    else:
                        with closing(sqlite3.connect(original_index.resolve().as_uri() + "?mode=ro", uri=True)) as src:
                            with closing(sqlite3.connect(db)) as dst:
                                src.backup(dst)
                    if args.section_chunks:
                        pages = list(documents.page_texts(documents.indexed_passages()))
                        documents.index_text_pages(filename, pages, section_aware=True)
                    indexed_pages = [
                        (filename, page, text) for page, text in documents.page_texts(documents.indexed_passages())
                    ]
                    for question in item["questions"]:
                        gold = evidence_hit(question, indexed_pages)
                        if question["answer"] is not None and not gold:
                            raise SystemExit(f"Reference evidence does not match the indexed PDF for {question['id']}.")
                        with capture_trace() as trace:
                            try:
                                response = await client.post("/documents/ask", json={
                                    "prompt": question["question"], "reader": args.reader,
                                    "previous_question": question.get("previous_question"),
                                })
                                body = response.json() if response.is_success else None
                                error = None if response.is_success else f"HTTP {response.status_code}: {response.text}"
                            except StopAtPrompt:
                                body = error = None
                        result = {
                            "id": question["id"], "filename": filename, "question": question["question"],
                            "kind": question["kind"], "expected": question["answer"], "response": body, "error": error,
                            **grade(question, body, trace, args.top_k, args.retrieval_only),
                        }
                        if args.traces:
                            result["trace"] = trace
                        results.append(result)
                        status = "HIT" if result["hit"] else "MISS" if result["hit"] is False else "-"
                        answer = "PASS" if result["correct"] else "FAIL" if result["correct"] is False else "-"
                        print(f"{result['id']:12} {status:4} {answer:4} {result.get('failure') or ''}", flush=True)
    return results


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/personal_pdf_questions.json"))
    parser.add_argument("--pdf", type=Path, action="append", default=[], help="A local PDF matching a manifest filename.")
    parser.add_argument("--reader", choices=("qwen", "tinygpt"), default="qwen")
    parser.add_argument("--top-k", type=int, default=5, help="Candidate cutoff per TinyGPT question part.")
    parser.add_argument("--keyword-only", action="store_true", help="Measure FTS without calling the embedding model.")
    parser.add_argument("--section-chunks", action="store_true", help="Try 500–1,000 character chunks in the temporary index.")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip Qwen generation and report retrieval metrics.")
    parser.add_argument("--report", type=Path, help="Save the per-question results as local JSON.")
    parser.add_argument("--traces", action="store_true", help="Include private context and prompts in the local report.")
    args = parser.parse_args()
    if args.top_k < 1 or (args.traces and not args.report):
        parser.error("--top-k must be positive, and --traces requires --report.")
    if args.retrieval_only and args.reader != "qwen":
        parser.error("--retrieval-only applies to Qwen.")

    async def keyword_only(question, rows):
        return []

    async def stop_at_prompt(*args, **kwargs):
        if kwargs.get("response_format", {}).get("title") == "StandaloneQuestion":
            return await original_ollama(*args, **kwargs)
        raise StopAtPrompt

    original_ollama = main.ask_ollama
    with ExitStack() as stack:
        if args.keyword_only:
            stack.enter_context(patch.object(rag_search, "semantic_passages", keyword_only))
            if args.retrieval_only:
                stack.enter_context(patch.dict(os.environ, {"RAG_QUERY_REWRITE": "0"}))
        if args.retrieval_only:
            stack.enter_context(patch.object(main, "ask_ollama", stop_at_prompt))
        results = asyncio.run(run(args))

    hits = [case["hit"] for case in results if case["hit"] is not None]
    correctness = [case["correct"] for case in results if case["correct"] is not None]
    missing = [case["hallucinated"] for case in results if case["hallucinated"] is not None]
    summary = {
        "retrieval_hits": [sum(hits), len(hits)],
        "correct_answers": [sum(correctness), len(correctness)] if correctness else None,
        "unsupported_answers_on_missing_questions": [sum(missing), len(missing)] if missing else None,
        "runtime_errors": sum(bool(case["error"]) for case in results),
    }
    print(json.dumps(summary))
    if args.report:
        args.report.write_text(json.dumps({"summary": summary, "cases": results}, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")


if __name__ == "__main__":
    main_cli()
