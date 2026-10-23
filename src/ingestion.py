"""Persistent PDF ingestion for the Financial RAG project.

New PDFs are extracted page-by-page, chunked, embedded with the existing model,
added to FAISS, and included in a rebuilt BM25 corpus. The original index is
preserved; only the new document's embeddings are computed.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any

import faiss
import fitz
import numpy as np
from rank_bm25 import BM25Okapi

from retrieval_engine import tokenize, load_jsonl

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_CHARS = 1200
OVERLAP_CHARS = 200


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _chunk_text(text: str) -> list[str]:
    """Character chunks with overlap; never combine content across PDF pages."""
    normalized = re.sub(r"[ \t]+", " ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= MAX_CHARS:
        return [normalized]
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + MAX_CHARS, len(normalized))
        # Prefer a paragraph/line boundary near the target to preserve table rows.
        if end < len(normalized):
            boundary = max(normalized.rfind("\n", start + MAX_CHARS // 2, end),
                           normalized.rfind(" ", start + MAX_CHARS // 2, end))
            if boundary > start:
                end = boundary
        part = normalized[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(normalized):
            break
        start = max(start + 1, end - OVERLAP_CHARS)
    return chunks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_pdf(pdf_path: Path, project_root: Path, model: Any) -> dict[str, Any]:
    """Ingest one PDF into canonical data, FAISS and BM25. Caller must serialize writes."""
    project_root = Path(project_root)
    processed = project_root / "data" / "processed"
    emb_dir = processed / "embeddings"
    raw_dir = project_root / "data" / "raw"
    processed.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    filename = pdf_path.name
    document_id = pdf_path.stem
    file_hash = _sha256(pdf_path)
    registry_path = processed / "ingested_documents.json"
    registry = json.loads(registry_path.read_text("utf-8")) if registry_path.exists() else {"documents": []}
    for doc in registry.get("documents", []):
        if doc.get("sha256") == file_hash:
            return {"status": "duplicate", "document_id": doc["document_id"], "filename": doc["filename"],
                    "chunk_count": doc.get("chunk_count", 0), "page_count": doc.get("page_count", 0)}

    # Do not silently overwrite a different report that has the same filename.
    existing_ids = {str(x.get("document_id")) for x in registry.get("documents", [])}
    if document_id in existing_ids:
        document_id = f"{document_id}_{file_hash[:8]}"
        filename = f"{document_id}.pdf"
        renamed = raw_dir / filename
        if pdf_path.resolve() != renamed.resolve():
            pdf_path = renamed
            # The API stages uploads under a temporary name, so move to unique name.
            # If already stored, this is a no-op.
            if not pdf_path.exists():
                raise FileNotFoundError(f"Uploaded PDF was not stored at expected path: {pdf_path}")

    try:
        pdf = fitz.open(pdf_path)
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc
    if pdf.is_encrypted and not pdf.authenticate(""):
        pdf.close()
        raise ValueError("Password-protected PDFs are not supported.")
    page_records: list[dict[str, Any]] = []
    new_chunks: list[dict[str, Any]] = []
    page_count = len(pdf)
    if page_count == 0:
        pdf.close()
        raise ValueError("The PDF has no pages.")
    for page_idx, page in enumerate(pdf):
        text = page.get_text("text", sort=True).strip()
        if not text:
            continue
        page_records.append({
            "document_id": document_id, "filename": filename,
            "page_number": page_idx + 1, "total_pages": page_count, "text": text,
        })
        for chunk_idx, chunk_text in enumerate(_chunk_text(text), 1):
            new_chunks.append({
                "chunk_id": f"{document_id}_p{page_idx + 1:04d}_c{chunk_idx:03d}",
                "document_id": document_id, "filename": filename,
                "page_number": page_idx + 1, "total_pages": page_count,
                "chunk_index": chunk_idx, "start_char": 0,
                "end_char": len(chunk_text), "chunk_size_config": MAX_CHARS,
                "chunk_overlap_config": OVERLAP_CHARS,
                "character_count": len(chunk_text), "word_count": len(chunk_text.split()),
                "text": chunk_text,
            })
    pdf.close()
    if not new_chunks:
        raise ValueError("No selectable text was found in this PDF. OCR is not yet enabled; use a text-based PDF.")

    chunks_path = processed / "embedding_ready_chunks.jsonl"
    old_chunks = _read_jsonl(chunks_path)
    existing_document = next((c for c in old_chunks if str(c.get("document_id")) == document_id
                              or str(c.get("filename", "")).lower() == filename.lower()), None)
    if existing_document:
        raise ValueError(
            f"A report named '{filename}' is already indexed. Rename the new file if it is a different edition."
        )
    old_ids = {str(c.get("chunk_id")) for c in old_chunks}
    collisions = [c["chunk_id"] for c in new_chunks if c["chunk_id"] in old_ids]
    if collisions:
        raise ValueError("Chunk ID collision detected; upload was not indexed.")

    # Create embeddings before mutating any persistent index files.
    vectors = model.encode(
        [c["text"] for c in new_chunks],
        batch_size=32, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 384 or not np.isfinite(vectors).all():
        raise ValueError(f"Embedding output has invalid shape or values: {vectors.shape}")

    index_path = emb_dir / "financial_rag.index"
    dense_meta_path = emb_dir / "embedding_metadata.jsonl"
    bm25_path = emb_dir / "bm25_index.pkl"
    bm25_meta_path = emb_dir / "bm25_metadata.jsonl"
    index = faiss.read_index(str(index_path))
    dense_metadata = _read_jsonl(dense_meta_path)
    if index.ntotal != len(dense_metadata):
        raise ValueError("Existing FAISS index and metadata are misaligned; upload cancelled.")
    if index.d != vectors.shape[1]:
        raise ValueError(f"Embedding dimension mismatch: existing={index.d}, new={vectors.shape[1]}")
    if len({str(x.get("chunk_id")) for x in dense_metadata + new_chunks}) != len(dense_metadata) + len(new_chunks):
        raise ValueError("Duplicate dense metadata chunk IDs; upload cancelled.")

    start_vector = index.ntotal
    index.add(vectors)
    appended_dense = dense_metadata + [
        {"vector_index": start_vector + i, "chunk_id": c["chunk_id"],
         "document_id": document_id, "filename": filename,
         "page_number": c["page_number"], "text": c["text"]}
        for i, c in enumerate(new_chunks)
    ]

    all_chunks = old_chunks + new_chunks
    tokenized_corpus = [tokenize(str(c.get("text", ""))) for c in all_chunks]
    if any(not tokens for tokens in tokenized_corpus):
        raise ValueError("At least one chunk has no searchable tokens; upload cancelled.")
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_payload = {"bm25": bm25, "tokenized_corpus": tokenized_corpus}
    bm25_metadata = [
        {"chunk_id": c["chunk_id"], "document_id": c["document_id"],
         "filename": c["filename"], "page_number": c["page_number"], "text": c["text"]}
        for c in all_chunks
    ]

    # Validate the auxiliary vector archive before changing any persistent files.
    embeddings_npy = emb_dir / "embeddings.npy"
    combined_vectors = None
    if embeddings_npy.exists():
        old_vectors = np.load(embeddings_npy).astype(np.float32)
        if old_vectors.ndim != 2 or old_vectors.shape[1] != vectors.shape[1] or old_vectors.shape[0] != start_vector:
            raise ValueError("Existing embeddings.npy is not aligned with FAISS; upload cancelled.")
        combined_vectors = np.vstack([old_vectors, vectors])

    # Write data and index files using atomic replacement for each file.
    _write_jsonl_atomic(chunks_path, all_chunks)
    _write_jsonl_atomic(dense_meta_path, appended_dense)
    if combined_vectors is not None:
        fd, tmp_vectors = tempfile.mkstemp(prefix="embeddings-", suffix=".npy", dir=emb_dir)
        try:
            # Pass the open file handle to NumPy so it does not append a second ".npy".
            with os.fdopen(fd, "wb") as vector_file:
                np.save(vector_file, combined_vectors)
            os.replace(tmp_vectors, embeddings_npy)
        finally:
            if os.path.exists(tmp_vectors):
                os.unlink(tmp_vectors)
    _write_jsonl_atomic(bm25_meta_path, bm25_metadata)
    fd, tmp_index = tempfile.mkstemp(prefix="faiss-", suffix=".index", dir=emb_dir)
    os.close(fd)
    try:
        faiss.write_index(index, tmp_index)
        os.replace(tmp_index, index_path)
    finally:
        if os.path.exists(tmp_index):
            os.unlink(tmp_index)
    fd, tmp_bm25 = tempfile.mkstemp(prefix="bm25-", suffix=".pkl", dir=emb_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(bm25_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_bm25, bm25_path)
    except Exception:
        try: os.unlink(tmp_bm25)
        except OSError: pass
        raise

    # Keep the raw extracted page archive append-only for auditability.
    _write_jsonl_atomic(processed / "extracted_pages.jsonl",
                        _read_jsonl(processed / "extracted_pages.jsonl") + page_records)
    registry.setdefault("documents", []).append({
        "document_id": document_id, "filename": filename, "sha256": file_hash,
        "page_count": page_count, "chunk_count": len(new_chunks), "status": "indexed",
    })
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "indexed", "document_id": document_id, "filename": filename,
            "page_count": page_count, "chunk_count": len(new_chunks), "vector_count": len(new_chunks)}
