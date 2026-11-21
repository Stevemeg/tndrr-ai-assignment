"""Extract the PDF text page by page and split it into word-based chunks.

Run:  python chunking.py
Writes index/chunks_150.json and index/chunks_500.json
"""

import json
import re

from pypdf import PdfReader

from config import CHUNK_SIZES, INDEX_DIR, PDF_PATH


def clean_text(text):
    """Fix two common PDF-extraction artefacts."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words split at a line break: "regula-\ntory" -> "regulatory"
    return " ".join(text.split())                 # collapse newlines and repeated spaces into single spaces


def extract_pages(pdf_path=PDF_PATH):
    """Return [{"page": n, "text": ...}] where n is the 1-based PDF page number."""
    reader = PdfReader(pdf_path)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"^NIST AI 100-1 AI RMF 1\.0\n", "", text)  # running header on every page
        text = re.sub(r"\nPage \d+\s*$", "", text)                 # printed page-number footer
        pages.append({"page": number, "text": clean_text(text)})
    return pages


def chunk_pages(pages, chunk_size, overlap):
    """Slide a window of `chunk_size` words over the whole document.
    Each step moves `chunk_size - overlap` words, so neighbouring chunks share
    `overlap` words and a sentence cut at one border is complete in the other chunk.
    Every chunk remembers the pages its first and last word came from."""
    words, word_pages = [], []
    for page in pages:
        for word in page["text"].split():
            words.append(word)
            word_pages.append(page["page"])

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunks.append({
            "id": len(chunks),
            "page_start": word_pages[start],
            "page_end": word_pages[end - 1],
            "text": " ".join(words[start:end]),
        })
        if end == len(words):  # the last chunk reached the end of the document
            break
    return chunks


def main():
    pages = extract_pages()
    total_words = sum(len(p["text"].split()) for p in pages)
    print(f"Extracted {len(pages)} pages, {total_words} words from {PDF_PATH.name}")

    INDEX_DIR.mkdir(exist_ok=True)
    for size, overlap in CHUNK_SIZES.items():
        chunks = chunk_pages(pages, size, overlap)
        path = INDEX_DIR / f"chunks_{size}.json"
        path.write_text(json.dumps(chunks, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{size}-word chunks, {overlap}-word overlap: {len(chunks)} chunks -> {path.name}")


if __name__ == "__main__":
    main()
