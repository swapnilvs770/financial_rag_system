"""Reusable hybrid retrieval engine for the Financial RAG application.

This module reuses the project's existing FAISS, BM25, metadata, and canonical
chunk files. It does not rebuild or mutate either index.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_RETRIEVAL_K = 50
DEFAULT_FINAL_K = 10
DEFAULT_RRF_K = 60
TOKEN_PATTERN = re.compile(r"[a-z]+(?:[./'-][a-z0-9]+)*|[0-9]+(?:\.[0-9]+)?")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path}, line {line_no}: {exc}") from exc
    return rows


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def record_chunk_id(record: dict[str, Any]) -> str:
    value = record.get("chunk_id") or record.get("id")
    if not value:
        raise ValueError(f"Metadata row has no chunk_id: {list(record)}")
    return str(value)


class HybridRetrievalEngine:
    """Loads existing indexes once and exposes document-aware hybrid search."""

    def __init__(self, project_root: Path = PROJECT_ROOT, model: Any = None):
        self.project_root = Path(project_root)
        processed = self.project_root / "data" / "processed"
        emb = processed / "embeddings"
        self.faiss_index = faiss.read_index(str(emb / "financial_rag.index"))
        self.dense_metadata = load_jsonl(emb / "embedding_metadata.jsonl")
        with (emb / "bm25_index.pkl").open("rb") as handle:
            saved_bm25 = pickle.load(handle)
        self.bm25 = next((v for v in saved_bm25.values() if hasattr(v, "get_scores")), None) if isinstance(saved_bm25, dict) else saved_bm25
        if self.bm25 is None or not hasattr(self.bm25, "get_scores"):
            raise TypeError("BM25 pickle does not contain an object with get_scores().")
        self.bm25_metadata = load_jsonl(emb / "bm25_metadata.jsonl")
        self.chunks = {record_chunk_id(x): x for x in load_jsonl(processed / "embedding_ready_chunks.jsonl")}
        self.model = model if model is not None else SentenceTransformer(MODEL_NAME)
        self._validate_alignment()

    def _validate_alignment(self) -> None:
        if self.faiss_index.ntotal != len(self.dense_metadata):
            raise ValueError(f"FAISS/metadata mismatch: {self.faiss_index.ntotal} vectors, {len(self.dense_metadata)} metadata rows")
        if len(self.bm25_metadata) != len(self.bm25.get_scores([])):
            # BM25Okapi's get_scores returns one score per corpus row.
            raise ValueError("BM25/metadata count mismatch.")
        dense_ids = [record_chunk_id(x) for x in self.dense_metadata]
        bm25_ids = [record_chunk_id(x) for x in self.bm25_metadata]
        if len(set(dense_ids)) != len(dense_ids):
            raise ValueError("Duplicate chunk IDs in dense metadata.")
        if len(set(bm25_ids)) != len(bm25_ids):
            raise ValueError("Duplicate chunk IDs in BM25 metadata.")

    @staticmethod
    def _matches_document(record: dict[str, Any], document_id: str | None) -> bool:
        if not document_id:
            return True
        return str(record.get("document_id", "")) == document_id or Path(str(record.get("filename", ""))).stem == document_id or str(record.get("filename", "")) == document_id

    def list_documents(self) -> list[dict[str, Any]]:
        docs: dict[str, dict[str, Any]] = {}
        for record in self.chunks.values():
            doc_id = str(record.get("document_id") or Path(str(record.get("filename", "unknown"))).stem)
            item = docs.setdefault(doc_id, {
                "document_id": doc_id,
                "filename": str(record.get("filename", f"{doc_id}.pdf")),
                "total_pages": record.get("total_pages"),
                "chunk_count": 0,
            })
            item["chunk_count"] += 1
        return sorted(docs.values(), key=lambda x: x["filename"].lower())

    def search(self, question: str, document_id: str | None = None, retrieval_k: int = DEFAULT_RETRIEVAL_K, final_k: int = DEFAULT_FINAL_K) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        if retrieval_k < 1 or final_k < 1:
            raise ValueError("retrieval_k and final_k must be positive.")

        # Filter the metadata population before truncating rankings. Since the
        # corpus is only ~8.7k chunks, exact all-vector search is acceptable and
        # avoids the previous failure mode where global top-50 omitted a target
        # document before the document filter could be applied.
        query_vector = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        dense_k = int(self.faiss_index.ntotal)
        dense_scores, dense_indices = self.faiss_index.search(query_vector, dense_k)
        dense_candidates = []
        for row_idx, score in zip(dense_indices[0], dense_scores[0]):
            if row_idx < 0:
                continue
            meta = self.dense_metadata[int(row_idx)]
            if self._matches_document(meta, document_id):
                dense_candidates.append((int(row_idx), float(score), meta))
        dense_candidates.sort(key=lambda x: (-x[1], x[0]))
        dense_candidates = dense_candidates[:retrieval_k]
        dense_results = [
            {"chunk_id": record_chunk_id(meta), "rank": rank, "score": score, "metadata": meta}
            for rank, (_, score, meta) in enumerate(dense_candidates, 1)
        ]

        bm25_scores = self.bm25.get_scores(tokenize(question))
        lexical_candidates = []
        for idx, score in enumerate(bm25_scores):
            meta = self.bm25_metadata[idx]
            if self._matches_document(meta, document_id):
                lexical_candidates.append((idx, float(score), meta))
        lexical_candidates.sort(key=lambda x: (-x[1], x[0]))
        lexical_results = []
        for rank, (_, score, meta) in enumerate(lexical_candidates, 1):
            if score <= 0:
                continue
            lexical_results.append({"chunk_id": record_chunk_id(meta), "rank": rank, "score": score, "metadata": meta})
            if len(lexical_results) >= retrieval_k:
                break

        fused: dict[str, dict[str, Any]] = {}
        for method, results in (("dense", dense_results), ("bm25", lexical_results)):
            for result in results:
                cid = result["chunk_id"]
                entry = fused.setdefault(cid, {
                    "chunk_id": cid, "rrf_score": 0.0,
                    "dense_rank": None, "dense_score": None,
                    "bm25_rank": None, "bm25_score": None,
                    "metadata": result["metadata"],
                })
                entry["rrf_score"] += 1.0 / (DEFAULT_RRF_K + result["rank"])
                entry[f"{method}_rank"] = result["rank"]
                entry[f"{method}_score"] = result["score"]

        ranked = sorted(fused.values(), key=lambda x: (-x["rrf_score"], x["chunk_id"]))[:final_k]
        results = []
        for rank, item in enumerate(ranked, 1):
            canonical = self.chunks.get(item["chunk_id"], item["metadata"])
            results.append({
                "rank": rank,
                "chunk_id": item["chunk_id"],
                "document_id": canonical.get("document_id"),
                "filename": canonical.get("filename") or canonical.get("source_file"),
                "page_number": canonical.get("page_number") or canonical.get("physical_page"),
                "text": canonical.get("text") or canonical.get("chunk_text") or "",
                "rrf_score": item["rrf_score"],
                "dense_rank": item["dense_rank"],
                "bm25_rank": item["bm25_rank"],
            })
        return {"question": question, "scope": "document" if document_id else "collection", "document_id": document_id, "result_count": len(results), "results": results}
