"""Check first-try PDF answers against the installed local Ollama models."""

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import httpx
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


@dataclass(frozen=True)
class Question:
    text: str
    pages: tuple[int, ...]
    evidence: tuple[str, ...] = ()


ACCESS_PAGES = (
    "Hawthorn garden access\nThe public entrance is the west gate on Cedar Street.\n"
    "The gate closes to all visitors at 7 p.m.",
    "Seasonal plots\nPlot | Crop | Monthly fee\nPlot A | Beans | $12\nPlot C | Kale | $28",
    "Check-in\nGarden keyholder Mira Sen signs visitors in before they enter.\n"
    "Return borrowed gloves to the gray cabinet when you leave.",
    "Publication record\nThe garden booklet was first printed in 2024.",
)
ACCESS_QUESTIONS = (
    Question("Which gate should a visitor enter through?", (1,), ("west gate",)),
    Question("What is the monthly fee for the kale plot?", (2,), ("Kale", "$28")),
    Question("Who signs people in and where should borrowed gloves go?", (3,), ("Mira Sen", "gray cabinet")),
    Question("What year did Hawthorn garden open to the public?", ()),
)

# The document length forces retrieval, even though each individual page is short.
OPERATIONS_PAGES = [
    f"Millbank service manual / Ledger {number}\n" + "\n".join(
        f"Record {record}. Audit the storage cabinets and report broken hinges.\n"
        "The supervisor files the checklist with the equipment inventory."
        for record in range(1, 7)
    )
    for number in range(1, 21)
]
OPERATIONS_PAGES[4] = (
    "Loading bay access\nThe orange clearance badge allows entry to the loading bay."
)
OPERATIONS_PAGES[5] = (
    "Loading bay access continued\nShow it to the shift steward before moving any cargo."
)
OPERATIONS_PAGES[10] = (
    "Power checks\nThe backup generator passed its inspection in 2023."
)
OPERATIONS_PAGES[13] = (
    "Service requests\nEquipment | Dispatch code | Holding shelf\n"
    "Forklift | R17 | North shelf\nFloor scrubber | S28 | South shelf"
)
OPERATIONS_QUESTIONS = (
    Question("Which dispatch code is assigned to the floor scrubber?", (14,), ("Floor scrubber", "S28")),
    Question("What year was the backup generator inspected?", (11,), ("inspection", "2023")),
    Question("What year was the backup generator installed?", ()),
    Question(
        "Which badge opens the loading bay and who must see it before cargo moves?",
        (5, 6), ("orange clearance badge", "shift steward"),
    ),
)

WORKSHOP_PAGES = (
    "Birch repair cooperative\nThe Saturday coordinator is Nisha Kapur.",
    "Before a sewing repair\nBring the machine, a matching bobbin, and one fabric scrap.",
    "Pricing\nRepair | Fee | Turnaround\nSewing machine | $16 | Three days\nDesk fan | $8 | One day",
    "Restrictions\nThe cooperative only repairs portable equipment.\n"
    "An on-site appliance repair requires a separate contractor.",
)
WORKSHOP_QUESTIONS = (
    Question("Who coordinates Saturday appointments?", (1,), ("Nisha Kapur",)),
    Question("What should I bring for a sewing repair?", (2,), ("matching bobbin", "fabric scrap")),
    Question("What should I bring for sewing repair and what is the fee?", (2, 3), ("matching bobbin", "$16")),
    Question("What is the cooperative's refund policy?", ()),
)


def make_pdf(pages: list[str] | tuple[str, ...]) -> bytes:
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        lines = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in text.splitlines()]
        stream = DecodedStreamObject()
        content = " 0 -15 Td ".join(f"({line}) Tj" for line in lines)
        stream.set_data(f"BT /F1 11 Tf 54 738 Td {content} ET".encode("ascii"))
        page.replace_contents(stream)
    result = BytesIO()
    writer.write(result)
    return result.getvalue()


def grade(question: Question, reply: str, sources: list[dict], filename: str) -> str:
    if not question.pages:
        return "pass" if not sources and "couldn’t find" in reply else "unsupported answer"
    if not sources:
        return "missed answer"
    normalized = " ".join(reply.casefold().split())
    pages = {source["page"] for source in sources}
    cited = pages == set(question.pages) and all(source["filename"] == filename for source in sources)
    complete = all(" ".join(fact.casefold().split()) in normalized for fact in question.evidence)
    if not cited:
        return "incorrect citations" if complete else "incorrect citations and evidence"
    return "pass" if complete else "incomplete or incorrect evidence"


async def run(app, reader: str = "qwen") -> int:
    fixtures = (
        ("hawthorn-garden.pdf", ACCESS_PAGES, ACCESS_QUESTIONS),
        ("millbank-service-manual.pdf", OPERATIONS_PAGES, OPERATIONS_QUESTIONS),
        ("birch-repair-cooperative.pdf", WORKSHOP_PAGES, WORKSHOP_QUESTIONS),
    )
    passed = total = 0
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://benchmark") as client:
        for filename, pages, questions in fixtures:
            upload = await client.post("/documents", params={"filename": filename}, content=make_pdf(pages))
            upload.raise_for_status()
            print(f"\n{filename} ({upload.json()['document']['pages']} pages)", flush=True)
            for question in questions:
                start = perf_counter()
                response = await client.post("/documents/ask", json={"prompt": question.text, "reader": reader})
                result = response.json()
                sources = result.get("sources", [])
                outcome = (
                    grade(question, result["reply"], sources, filename)
                    if response.is_success else f"HTTP {response.status_code}"
                )
                passed += outcome == "pass"
                total += 1
                print(f"{outcome.upper():33} {perf_counter() - start:5.1f}s  {question.text}", flush=True)
                if outcome != "pass":
                    print(f"    Expected pages {list(question.pages)} and evidence {question.evidence}", flush=True)
                    print(f"    Actual pages {[source['page'] for source in sources]}  {result}", flush=True)
    print(f"\n{passed}/{total} first-try checks passed.", flush=True)
    return int(passed != total)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", choices=("qwen", "tinygpt"), default="qwen")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="personal-ai-benchmark-") as directory:
        os.environ["PERSONAL_AI_DOCUMENT_DB"] = str(Path(directory) / "documents.sqlite3")
        from backend.main import OLLAMA_CHAT_URL, app

        if args.reader == "qwen":
            try:
                endpoint = OLLAMA_CHAT_URL.rsplit("/", 1)[0] + "/tags"
                httpx.get(endpoint, timeout=3, trust_env=False).raise_for_status()
            except httpx.HTTPError:
                print("Start local Ollama before running this benchmark.", file=sys.stderr)
                return 2
        return asyncio.run(run(app, args.reader))


if __name__ == "__main__":
    raise SystemExit(main())

