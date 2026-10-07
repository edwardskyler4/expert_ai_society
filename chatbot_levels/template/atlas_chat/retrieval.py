"""Chunking, SQLite BM25 search, and optional embedding rank fusion."""
import io
import json
import math
import re
import time

from providers import ProviderError
from store import identifier

STOPWORDS = set("a an and are as at be by can do for from how i in is it me my of on or that the this to was what when where which who why with you your".split())


def chunks(text, size=1800, overlap=250):
    if not 0 <= overlap < size:
        raise ValueError("Chunk overlap must be smaller than the chunk size.")
    text = text.replace("\r\n", "\n").replace("\x00", "").strip()
    result, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(text.rfind("\n", start + size // 2, end), text.rfind(" ", start + size // 2, end))
            if boundary > start:
                end = boundary
        part = text[start:end].strip()
        if part:
            result.append(part)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return result


def cosine(left, right):
    if len(left) != len(right) or not left:
        return -1.0
    norm = math.sqrt(sum(x * x for x in left) * sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right)) / norm if norm else -1.0


def fuse(rankings, limit=5):
    scores = {}
    for ranking in rankings:
        for position, value in enumerate(ranking, 1):
            scores[value] = scores.get(value, 0) + 1 / (60 + position)
    return sorted(scores, key=lambda value: (-scores[value], value))[:limit]


def extract(name, data):
    if name.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            from pypdf.errors import PyPdfError
        except ImportError as exc:
            raise ValueError("PDF support requires: python -m pip install pypdf") from exc
        try:
            reader = PdfReader(io.BytesIO(data))
            if len(reader.pages) > 200:
                raise ValueError("PDFs are limited to 200 pages.")
            pages = [page.extract_text() or "" for page in reader.pages]
        except (PyPdfError, OSError) as exc:
            raise ValueError("Cannot read this PDF. It may be damaged or password protected.") from exc
        if not any(page.strip() for page in pages):
            raise ValueError("This PDF has no extractable text. Scanned PDFs require OCR.")
        return "\n\n".join(f"[Page {index + 1}]\n{text}" for index, text in enumerate(pages))
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Upload a UTF-8 text file or a text-based PDF.") from exc


class Retriever:
    def __init__(self, store, providers, settings):
        self.store, self.providers, self.settings = store, providers, settings

    def ingest(self, name, data):
        text = extract(name, data)
        if not text.strip():
            raise ValueError("The document contains no text.")
        if len(text) > self.settings.max_document_chars:
            raise ValueError("Document exceeds the 400,000 character limit.")
        passages = chunks(text)
        with self.store.connect() as db:
            count = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        if count + len(passages) > 5000:
            raise ValueError("This workspace is limited to 5,000 chunks. Delete documents first.")
        vectors = self.providers.embeddings(passages)
        if vectors is not None and any(not vector or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector) for vector in vectors):
            raise ProviderError("Invalid embedding vector received.")
        value = identifier()
        signature = self.settings.embedding_signature if vectors is not None else "none"
        with self.store.connect() as db:
            if db.execute("SELECT count(*) FROM chunks").fetchone()[0] + len(passages) > 5000:
                raise ValueError("This workspace is limited to 5,000 chunks. Delete documents first.")
            db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?)", (value, name, len(text), len(passages), signature, time.time()))
            for index, passage in enumerate(passages):
                chunk_id = identifier()
                vector = json.dumps(vectors[index]) if vectors is not None else None
                db.execute("INSERT INTO chunks VALUES (?,?,?,?,?)", (chunk_id, value, index, passage, vector))
                db.execute("INSERT INTO chunks_fts VALUES (?,?,?)", (chunk_id, value, passage))
        return next(item for item in self.store.list_documents() if item["id"] == value)

    def search(self, query, limit=5):
        terms = [term for term in dict.fromkeys(re.findall(r"\w+", query.lower())) if term not in STOPWORDS and len(term) > 1][:24]
        lexical, semantic, warnings = [], [], []
        with self.store.connect() as db:
            rows = {row["id"]: dict(row) for row in db.execute("""
                SELECT c.*, d.name, d.embedding_signature FROM chunks c JOIN documents d ON d.id=c.document_id
            """)}
            if terms:
                expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
                lexical = [row[0] for row in db.execute("SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT 30", (expression,))]
        candidates = [row for row in rows.values() if row["vector"] and row["embedding_signature"] == self.settings.embedding_signature]
        if candidates and self.settings.embedding_provider != "none":
            try:
                query_vector = self.providers.embeddings([query])[0]
                scored = [(cosine(query_vector, json.loads(row["vector"])), row["id"]) for row in candidates]
                semantic = [value for score, value in sorted(scored, reverse=True)[:30] if score > 0.1]
            except (ProviderError, ValueError, KeyError) as exc:
                warnings.append("Semantic search was unavailable; used keyword search.")
        sources = []
        for index, value in enumerate(fuse([lexical, semantic], limit), 1):
            row = rows[value]
            sources.append({
                "kind": "document", "label": f"D{index}", "document_id": row["document_id"],
                "title": row["name"], "chunk": row["ordinal"] + 1, "text": row["text"],
            })
        return sources, warnings
