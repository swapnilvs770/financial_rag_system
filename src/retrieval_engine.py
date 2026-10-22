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

    @staticmethod
    def expand_query(question: str) -> str:
        """Add finance-domain synonyms without injecting answers or company-specific facts."""
        q = question.lower()
        expansions: list[str] = []

        synonym_groups = (
            (("employee cost", "employee costs", "staff cost", "staff costs", "personnel cost", "personnel costs"),
             "employee benefit expenses employee expenses staff expenses personnel expenses compensation"),
            (("profit and loss", "income statement", "statement of profit and loss"),
             "income statement earnings operating expenses"),
            (("revenue", "sales", "turnover"),
             "revenue from operations total revenue net sales"),
            (("net profit", "profit after tax", "pat"),
             "profit after tax net income earnings"),
            (("operating profit", "ebit"),
             "operating income earnings before interest and taxes"),
            (("cash flow", "cashflows"),
             "cash generated operating investing financing activities"),
            (("borrowings", "debt", "loans"),
             "total debt loans outstanding liabilities"),
            (("capital expenditure", "capex"),
             "purchase of property plant equipment investing activities"),
            (("accounts receivable", "trade receivables"),
             "trade debtors receivables amounts due from customers"),
            (("accounts payable", "trade payables"),
             "trade creditors payables amounts due to suppliers"),
        )

        for triggers, terms in synonym_groups:
            if any(trigger in q for trigger in triggers):
                expansions.append(terms)

        # Preserve all original terms; append only distinct expansion terms.
        original_tokens = set(tokenize(question))
        added_tokens: list[str] = []
        seen = set(original_tokens)
        for phrase in expansions:
            for token in tokenize(phrase):
                if token not in seen:
                    seen.add(token)
                    added_tokens.append(token)
        return question if not added_tokens else f"{question} {' '.join(added_tokens)}"

    @staticmethod
    def _financial_relevance_bonus(question: str, chunk_text: str) -> tuple[float, dict[str, bool]]:
        """Apply small, explainable boosts for financial-statement evidence.

        This uses only question/chunk wording and never injects expected answers
        or company-specific numbers.
        """
        q = question.lower()
        text = chunk_text.lower()
        bonus = 0.0
        features = {
            "metric_match": False,
            "statement_heading": False,
            "notes_table": False,
            "narrative_only": False,
            "consolidated": False,
            "standalone": False,
        }

        metric_groups = (
            (("employee cost", "employee costs", "employee benefit expense", "staff cost", "personnel cost"),
             ("employee benefit expenses", "employee expenses", "salaries, incentives")),
            (("revenue", "sales", "turnover"),
             ("revenue from operations", "total revenue", "net sales")),
            (("net profit", "profit after tax", "pat"),
             ("profit for the year", "profit after tax", "net income")),
            (("operating profit", "ebit"),
             ("operating profit", "profit from operations", "ebit")),
            (("cash flow", "cashflows"),
             ("cash flows from operating activities", "net cash generated")),
            (("borrowings", "debt", "loans"),
             ("borrowings", "total debt", "loans payable")),
            (("capital expenditure", "capex"),
             ("property, plant and equipment", "capital expenditure", "purchase of property")),
        )
        for question_terms, evidence_terms in metric_groups:
            if any(term in q for term in question_terms) and any(term in text for term in evidence_terms):
                bonus += 0.018
                features["metric_match"] = True
                break

        statement_headings = (
            "statement of profit and loss",
            "consolidated balance sheet",
            "balance sheet",
            "statement of cash flows",
            "cash flow statement",
            "statement of changes in equity",
        )
        if any(heading in text for heading in statement_headings):
            bonus += 0.035
            features["statement_heading"] = True

        # Stronger boost when the chunk itself is the primary consolidated
        # statement and contains the requested metric, rather than a note
        # that repeats the same metric on another reporting basis.
        primary_statement = (
            "consolidated statement of profit and loss" in text
            or "consolidated balance sheet" in text
            or "consolidated statement of cash flows" in text
            or "consolidated cash flow statement" in text
        )
        if primary_statement and features["metric_match"]:
            bonus += 0.035
            features["primary_consolidated_statement"] = True
        else:
            features["primary_consolidated_statement"] = False

        if "consolidated" in text:
            features["consolidated"] = True
            if "consolidated" in q:
                bonus += 0.025
        if "standalone" in text or "separate financial statements" in text:
            features["standalone"] = True
            if "standalone" in q or "stand-alone" in q:
                bonus += 0.025
            elif "consolidated" in q:
                bonus -= 0.035

        if "notes to the financial statements" in text or "employee benefit obligations" in text:
            features["notes_table"] = True
            bonus += 0.004

        narrative_cues = ("employee well-being", "employee wellbeing", "learning and development",
                          "workforce engagement", "people practices", "employee experience")
        if any(cue in text for cue in narrative_cues):
            features["narrative_only"] = True
            bonus -= 0.025

        return bonus, features

    def search(self, question: str, document_id: str | None = None, retrieval_k: int = DEFAULT_RETRIEVAL_K, final_k: int = DEFAULT_FINAL_K) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        if retrieval_k < 1 or final_k < 1:
            raise ValueError("retrieval_k and final_k must be positive.")

        expanded_query = self.expand_query(question)
        dense_k = int(self.faiss_index.ntotal)

        def dense_search_for(query_text: str) -> list[dict[str, Any]]:
            query_vector = self.model.encode(
                [query_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            dense_scores, dense_indices = self.faiss_index.search(query_vector, dense_k)
            candidates = []
            for row_idx, score in zip(dense_indices[0], dense_scores[0]):
                if row_idx < 0:
                    continue
                meta = self.dense_metadata[int(row_idx)]
                if self._matches_document(meta, document_id):
                    candidates.append((int(row_idx), float(score), meta))
            candidates.sort(key=lambda x: (-x[1], x[0]))
            return [
                {"chunk_id": record_chunk_id(meta), "rank": rank, "score": score, "metadata": meta}
                for rank, (_, score, meta) in enumerate(candidates[:retrieval_k], 1)
            ]

        def bm25_search_for(query_text: str) -> list[dict[str, Any]]:
            scores = self.bm25.get_scores(tokenize(query_text))
            candidates = []
            for idx, score in enumerate(scores):
                meta = self.bm25_metadata[idx]
                if self._matches_document(meta, document_id):
                    candidates.append((idx, float(score), meta))
            candidates.sort(key=lambda x: (-x[1], x[0]))
            results = []
            for _, score, meta in candidates:
                if score <= 0:
                    continue
                results.append({
                    "chunk_id": record_chunk_id(meta),
                    "rank": len(results) + 1,
                    "score": score,
                    "metadata": meta,
                })
                if len(results) >= retrieval_k:
                    break
            return results

        # Run both retrievers on the original and expanded query. Document
        # filtering happens inside each retriever before truncating its ranking.
        dense_results = dense_search_for(question)
        bm25_results = bm25_search_for(question)
        if expanded_query != question:
            expanded_dense_results = dense_search_for(expanded_query)
            expanded_bm25_results = bm25_search_for(expanded_query)
        else:
            expanded_dense_results = []
            expanded_bm25_results = []

        fused: dict[str, dict[str, Any]] = {}
        ranking_sets = (
            ("dense", dense_results),
            ("bm25", bm25_results),
            ("expanded_dense", expanded_dense_results),
            ("expanded_bm25", expanded_bm25_results),
        )
        for method, results_for_method in ranking_sets:
            for result in results_for_method:
                cid = result["chunk_id"]
                entry = fused.setdefault(cid, {
                    "chunk_id": cid,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                    "expanded_dense_rank": None,
                    "expanded_bm25_rank": None,
                    "metadata": result["metadata"],
                })
                entry["rrf_score"] += 1.0 / (DEFAULT_RRF_K + result["rank"])
                entry[f"{method}_rank"] = result["rank"]
                if method in ("dense", "bm25"):
                    entry[f"{method}_score"] = result["score"]

        # Rerank only the retrieved candidate pool using transparent financial
        # evidence features. RRF remains the base score; boosts are intentionally
        # small and explainable.
        for item in fused.values():
            canonical = self.chunks.get(item["chunk_id"], item["metadata"])
            chunk_text = str(canonical.get("text") or canonical.get("chunk_text") or "")
            bonus, features = self._financial_relevance_bonus(question, chunk_text)
            item["rerank_bonus"] = bonus
            item["rerank_features"] = features
            item["final_score"] = item["rrf_score"] + bonus

        ranked = sorted(
            fused.values(),
            key=lambda x: (-x["final_score"], -x["rrf_score"], x["chunk_id"]),
        )[:final_k]

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
                "rerank_bonus": item["rerank_bonus"],
                "final_score": item["final_score"],
                "rerank_features": item["rerank_features"],
                "dense_rank": item["dense_rank"],
                "bm25_rank": item["bm25_rank"],
                "expanded_dense_rank": item["expanded_dense_rank"],
                "expanded_bm25_rank": item["expanded_bm25_rank"],
            })
        return {
            "question": question,
            "expanded_query": expanded_query,
            "scope": "document" if document_id else "collection",
            "document_id": document_id,
            "result_count": len(results),
            "results": results,
        }
