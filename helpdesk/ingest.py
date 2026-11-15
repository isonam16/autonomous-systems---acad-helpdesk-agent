"""Build the persistent Chroma index using Gemini embeddings."""
import hashlib
import json
import re

import chromadb
from google import genai
from google.genai import types
from bs4 import BeautifulSoup
from loguru import logger

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from . import config as C

client = genai.Client(api_key=C.GEMINI_API_KEY)
WEB_NOISE = "###147852369$$$_RedirectToLoginPage_%%%963258741!!!"
WEB_FOOTER = re.compile(r"Important links\s+CCCD @ IITJ.*", re.S)


def embed(texts: list[str], task_type: str = "RETRIEVAL_QUERY") -> list[list[float]]:
    vectors = []
    for start in range(0, len(texts), 32):
        response = client.models.embed_content(
            model=C.EMBED_MODEL,
            contents=texts[start:start + 32],
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768,
            ),
        )
        vectors.extend(item.values for item in response.embeddings)
    return vectors


def split_text(text: str) -> list[str]:
    paragraphs = [
        re.sub(r"\s+", " ", part).strip()
        for part in re.split(r"\n\s*\n", text)
    ]
    paragraphs = [part for part in paragraphs if len(part) > 20]
    output, current = [], ""
    for paragraph in paragraphs:
        while len(paragraph) > C.CHUNK_CHARS:
            output.append(paragraph[:C.CHUNK_CHARS])
            paragraph = paragraph[C.CHUNK_CHARS - C.CHUNK_OVERLAP:]
        if current and len(current) + len(paragraph) > C.CHUNK_CHARS:
            output.append(current)
            current = current[-C.CHUNK_OVERLAP:] + " " + paragraph
        else:
            current = (current + " " + paragraph).strip()
    if current:
        output.append(current)
    return output


def parse_file(path) -> list[dict]:
    items = []
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        for page_number, page in enumerate(fitz.open(path), start=1):
            items.extend(
                {"text": piece, "source": path.name, "loc": f"page {page_number}"}
                for piece in split_text(page.get_text())
            )
    elif suffix in {".html", ".htm"}:
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        items.extend(
            {"text": piece, "source": path.name, "loc": "html"}
            for piece in split_text(soup.get_text("\n"))
        )
    elif suffix in {".txt", ".md"}:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"SOURCE_URL:\s*(\S+)", raw)
        location = match.group(1) if match else "text"
        raw = WEB_FOOTER.sub("", raw.replace(WEB_NOISE, " "))
        items.extend(
            {"text": piece, "source": path.name, "loc": location}
            for piece in split_text(raw)
        )
    return items


def add_context(chunk: dict, doc_title: str) -> str:
    prompt = (
        f"Document: {doc_title} ({chunk['loc']})\n"
        f"Chunk:\n{chunk['text']}\n\n"
        "In one short sentence, say what part of the document this chunk is about."
    )
    response = client.models.generate_content(model=C.GEMINI_MODEL, contents=prompt)
    return (response.text or "").strip()


def main():
    C.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    C.CHUNKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    chunks, seen = [], set()
    files = [
        path for path in sorted(C.DOCS_DIR.iterdir())
        if path.is_file() and not path.name.startswith(".")
    ]
    for path in files:
        for item in parse_file(path):
            key = hashlib.md5(
                re.sub(r"\W+", "", item["text"].lower()).encode()
            ).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            item["id"] = f"c{len(chunks)}"
            item["embed_text"] = item["text"]
            if C.USE_CONTEXTUAL_RETRIEVAL:
                item["embed_text"] = add_context(item, path.name) + "\n" + item["text"]
            chunks.append(item)

    if not chunks:
        raise SystemExit("No chunks produced. Add documents to data/docs/.")
    database = chromadb.PersistentClient(path=str(C.CHROMA_DIR))
    try:
        database.delete_collection("helpdesk")
    except Exception:
        pass
    collection = database.create_collection("helpdesk", metadata={"hnsw:space": "cosine"})
    for start in range(0, len(chunks), 32):
        batch = chunks[start:start + 32]
        collection.add(
            ids=[item["id"] for item in batch],
            embeddings=embed([item["embed_text"] for item in batch], "RETRIEVAL_DOCUMENT"),
            documents=[item["text"] for item in batch],
            metadatas=[{"source": item["source"], "loc": item["loc"]} for item in batch],
        )
        logger.info(f"Indexed {min(start + len(batch), len(chunks))}/{len(chunks)} chunks")
    C.CHUNKS_JSON.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    logger.info("Gemini embedding ingestion complete")


if __name__ == "__main__":
    main()
