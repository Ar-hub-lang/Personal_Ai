"""Build and check TinyGPT's document evidence-selection examples."""

import argparse
import hashlib
import json
import logging
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader

from training.tokenizer import ByteTokenizer

DATA = Path(__file__).with_name("document_qa")
SPLITS = ("train", "validation", "test")
CHOICES = "ABCD"
NO_ANSWER = "NO_ANSWER"


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}|\d+", text.casefold()))


def balanced_positions(seed: str) -> Iterator[int]:
    rng = random.Random(seed)
    while True:
        yield from rng.sample(range(len(CHOICES)), len(CHOICES))


def make_example(
    document: dict, suffix: str, question: str, kind: str,
    evidence_id: str | None = None, gold_position: int | None = None,
) -> dict:
    case_id = f"{document['id']}-{suffix}"
    rng = random.Random(case_id)
    rows = document["evidence"]
    gold = next((row for row in rows if row["id"] == evidence_id), None)
    other_rows = [row for row in rows if row is not gold]
    rng.shuffle(other_rows)
    query_terms = terms(question)
    other_rows.sort(key=lambda row: len(query_terms & terms(row["text"])), reverse=True)
    candidates = other_rows[:3] if gold_position is not None else other_rows[:4]
    rng.shuffle(candidates)
    if gold_position is not None:
        if gold is None:
            raise ValueError(f"No matching evidence for {case_id}.")
        candidates.insert(gold_position, gold)
    choices = [
        {"id": choice, "evidence_id": row["id"], "page": row["page"], "text": row["text"]}
        for choice, row in zip(CHOICES, candidates, strict=True)
    ]
    target = NO_ANSWER if gold_position is None else CHOICES[gold_position]
    example = {
        "id": case_id, "document_id": document["id"], "kind": kind,
        "question": question, "candidates": choices, "target": target,
    }
    if "context" in document:
        example["context"] = document["context"]
    return example


def make_splits(documents: list[dict]) -> dict[str, list[dict]]:
    splits = {split: [] for split in SPLITS}
    positions = {split: balanced_positions(split) for split in SPLITS}
    for document in documents:
        split = document["split"]
        examples = splits[split]
        for number, row in enumerate(document["evidence"], 1):
            for variant, question in enumerate(row["questions"], 1):
                examples.append(make_example(
                    document, f"q{number:02}-{variant}", question, "answerable",
                    row["id"], next(positions[split]),
                ))
            # Reuse the wording with the supporting passage withheld to teach abstention.
            if split == "train" or number % 2 == 1:
                examples.append(make_example(
                    document, f"q{number:02}-no-evidence", row["questions"][0], "evidence_removed", row["id"],
                ))
        for number, question in enumerate(document["missing_detail_questions"], 1):
            examples.append(make_example(document, f"missing-{number:02}", question, "missing_detail"))
    return splits


def format_prompt(example: dict) -> str:
    lines = [f"Question: {example['question']}"]
    if context := example.get("context"):
        lines.append(context)
    lines.extend(f"{candidate['id']}: {candidate['text']}" for candidate in example["candidates"])
    lines.append("Evidence:")
    return "\n".join(lines)


def encode_example(example: dict, tokenizer: ByteTokenizer) -> list[int]:
    return tokenizer.encode(f"{format_prompt(example)} {example['target']}", add_boundaries=True)


def read_split(split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"Unknown document QA split: {split}.")
    path = DATA / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_sources(documents: list[dict], pdf_dir: Path | None) -> None:
    benchmark = DATA.parents[1] / "benchmarks" / "real_pdf_questions.json"
    held_out = json.loads(benchmark.read_text(encoding="utf-8"))["documents"]
    seen_ids = set()
    seen_urls = {document["source_url"] for document in held_out}
    seen_hashes = {document["sha256"] for document in held_out}
    for document in documents:
        name = document["id"]
        url = document["source_url"]
        checksum = document["sha256"]
        if name in seen_ids or url in seen_urls or checksum in seen_hashes:
            raise ValueError(f"Duplicate source or benchmark overlap: {name}.")
        seen_ids.add(name)
        seen_urls.add(url)
        seen_hashes.add(checksum)

        evidence = document["evidence"]
        if len(evidence) < 5 or len({row["id"] for row in evidence}) != len(evidence):
            raise ValueError(f"{name} needs at least five unique evidence rows.")
        if len({normalize(row["text"]) for row in evidence}) != len(evidence):
            raise ValueError(f"Duplicate evidence text in {name}.")
        for row in evidence:
            if not row["questions"] or not 1 <= row["page"] <= document["pages"]:
                raise ValueError(f"Invalid page or missing question for {row['id']}.")

        if pdf_dir is not None:
            path = pdf_dir / f"{name}.pdf"
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Checksum mismatch for {path}.")
            reader = PdfReader(path)
            if len(reader.pages) != document["pages"]:
                raise ValueError(f"Unexpected page count for {path}.")
            page_text = {page: normalize(reader.pages[page - 1].extract_text() or "")
                         for page in {row["page"] for row in evidence}}
            for row in evidence:
                if normalize(row["text"]) not in page_text[row["page"]]:
                    raise ValueError(f"{row['id']} does not match PDF page {row['page']}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate the split JSONL files from sources.json.")
    parser.add_argument("--pdf-dir", type=Path, help="Verify against local PDFs named by source ID.")
    options = parser.parse_args()
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    documents = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))["documents"]
    try:
        verify_sources(documents, options.pdf_dir)
        tokenizer = ByteTokenizer()
        longest = 0
        for split, examples in make_splits(documents).items():
            if not examples or len({example["id"] for example in examples}) != len(examples):
                raise ValueError(f"{split} is empty or contains duplicate example IDs.")
            path = DATA / f"{split}.jsonl"
            expected = "".join(json.dumps(example, ensure_ascii=False) + "\n" for example in examples)
            if options.write:
                path.write_text(expected, encoding="utf-8")
            elif path.read_text(encoding="utf-8") != expected:
                raise ValueError(f"{path.name} differs from the sources. Rerun with --write to regenerate it.")
            counts = Counter(example["kind"] for example in examples)
            longest = max(longest, max(len(encode_example(example, tokenizer)) for example in examples))
            print(f"{split}: {len(examples)} examples, {len({row['document_id'] for row in examples})} PDFs "
                  f"({counts['answerable']} answerable, {counts['evidence_removed']} evidence removed, "
                  f"{counts['missing_detail']} missing details)")
        print(f"Longest prompt plus target and boundary tokens: {longest} byte tokens.")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
