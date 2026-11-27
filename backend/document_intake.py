"""Read local IT guides and index them with their source locations."""

import argparse
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from zipfile import BadZipFile, ZipFile

from backend import documents


DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}
WORD_XML = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
WORD_TAG = "{" + WORD_XML["w"] + "}"


def recognize_pdf_pages(data: bytes, pages: list[int]) -> list[tuple[int, str]]:
    script = Path(__file__).with_name("pdf_ocr.swift")
    cache = Path(gettempdir()) / "personal-ai-ocr-module-cache"
    cache.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="personal-ai-ocr-") as directory:
        source = Path(directory) / "source.pdf"
        source.write_bytes(data)
        try:
            result = subprocess.run(
                ["swift", str(script), str(source), *map(str, pages)],
                env=os.environ | {"CLANG_MODULE_CACHE_PATH": str(cache)},
                capture_output=True, text=True, timeout=180, check=True,
            )
            text = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise ValueError("Local PDF OCR failed. Check that Swift and the macOS Vision framework are available.") from exc
    if len(text) != len(pages):
        raise ValueError("Local PDF OCR did not return every requested page.")
    return [(page, content) for page, content in zip(pages, text) if content.strip()]


def docx_blocks(data: bytes) -> list[tuple[str, bool]]:
    try:
        with ZipFile(BytesIO(data)) as archive:
            member = archive.getinfo("word/document.xml")
            if member.file_size > 4_000_000:
                raise ValueError("This Word document contains too much text for the local index.")
            root = ET.fromstring(archive.read(member))
    except (BadZipFile, KeyError, ET.ParseError) as exc:
        raise ValueError("The Word document could not be read.") from exc
    body = root.find("w:body", WORD_XML)
    if body is None:
        return []

    def paragraph(node: ET.Element) -> str:
        return "".join(part.text or "" for part in node.findall(".//w:t", WORD_XML)).strip()

    blocks = []
    for child in body:
        if child.tag == WORD_TAG + "p":
            style = child.find("w:pPr/w:pStyle", WORD_XML)
            heading = style is not None and bool(re.fullmatch(r"Title|Heading\d+", style.get(WORD_TAG + "val", ""), re.I))
            blocks.append((paragraph(child), heading))
        elif child.tag == WORD_TAG + "tbl":
            for row in child.findall("w:tr", WORD_XML):
                cells = [" ".join(paragraph(node) for node in cell.findall("w:p", WORD_XML))
                         for cell in row.findall("w:tc", WORD_XML)]
                blocks.append((" | ".join(cells), False))
    return blocks


def text_documents(data: bytes, suffix: str) -> list[tuple[str, bool]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Use UTF-8 text for TXT or Markdown files.") from exc
    if suffix == ".txt":
        return [(line, False) for line in text.splitlines()]
    return [(line.lstrip("# ").strip(), True) if re.match(r"^#{1,6}\s+", line) else (line, False)
            for line in text.splitlines()]


def index_document(
    filename: str, data: bytes, *, aliases: tuple[str, ...] = (), source_url: str | None = None,
) -> dict:
    suffix = Path(filename).suffix.casefold()
    if len(data) > documents.MAX_PDF_BYTES:
        raise ValueError("Choose a document of 10 MB or less.")
    if suffix == ".pdf":
        return documents.index_pdf(filename, data, aliases=aliases)
    if not documents.collection_enabled() or suffix not in DOCUMENT_EXTENSIONS:
        raise ValueError("Choose a PDF, Word document, TXT, or Markdown file for the collection.")
    blocks = docx_blocks(data) if suffix == ".docx" else text_documents(data, suffix)
    grouped: list[tuple[str, list[str]]] = [("Introduction", [])]
    for text, heading in blocks:
        if text and heading:
            grouped.append((text, [text]))
        elif text.strip():
            grouped[-1][1].append(text)
    sections = {number: title for number, (title, _) in enumerate(
        (group for group in grouped if group[1]), 1,
    )}
    pages = [(number, "\n".join(lines)) for number, lines in enumerate(
        (lines for _, lines in grouped if lines), 1,
    )]
    return documents.index_text_pages(filename, pages, section_aware=os.environ.get("RAG_SECTION_CHUNKS") == "1",
                                     aliases=aliases, sections=sections, source_url=source_url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--sources", type=Path, help="Optional publisher and alias metadata for the guides.")
    args = parser.parse_args()
    if not documents.collection_enabled():
        parser.error("Set PERSONAL_AI_COLLECTION=1 and point PERSONAL_AI_DOCUMENT_DB at the collection index.")
    sources = json.loads(args.sources.read_text(encoding="utf-8"))["documents"] if args.sources else []
    aliases = {row["filename"]: tuple(row.get("aliases", [row["publisher"]])) for row in sources}
    paths = sorted(path for path in args.folder.iterdir() if path.is_file() and path.suffix.casefold() in DOCUMENT_EXTENSIONS)
    for path in paths:
        result = index_document(path.name, path.read_bytes(), aliases=aliases.get(path.name, ()))
        print(f"{result['filename']} · {result['pages']} indexed locations", flush=True)
    print(f"{len(paths)} files indexed.", flush=True)


if __name__ == "__main__":
    main()
