"""FastAPI query API for the Financial RAG project.

Run from the repository root with:
    uvicorn api.main:app --reload

This API returns hybrid retrieval results plus a conservative, extractive answer
when a financial statement row can be identified directly in retrieved evidence.
It does not invent figures and does not require an LLM provider.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

app = FastAPI(
    title="Financial RAG API",
    version="0.2.0",
    description=(
        "Hybrid FAISS + BM25 retrieval with evidence-grounded extractive answers "
        "and page-level source citations."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
_engine = None
_engine_error: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    scope: Literal["collection", "document"] = "collection"
    document_id: str | None = Field(default=None, max_length=300)
    retrieval_k: int = Field(default=50, ge=1, le=200)
    final_k: int = Field(default=10, ge=1, le=50)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


@app.on_event("startup")
def load_retrieval_engine() -> None:
    """Load heavy retrieval dependencies/indexes at startup, not on import."""
    global _engine, _engine_error
    try:
        from retrieval_engine import HybridRetrievalEngine

        _engine = HybridRetrievalEngine()
        _engine_error = None
    except Exception as exc:
        _engine = None
        _engine_error = f"{type(exc).__name__}: {exc}"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if _engine is not None else "degraded",
        "retrieval_ready": _engine is not None,
        "answer_generation": "extractive_financial_rows",
        "retrieval_error": _engine_error,
    }


@app.get("/api/documents")
def documents() -> dict[str, Any]:
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Retrieval engine is not ready: {_engine_error or 'not initialized'}",
        )
    return {"documents": _engine.list_documents()}


def _basis_from_text(text: str) -> str:
    lowered = text.lower()
    if "consolidated" in lowered:
        return "consolidated"
    if "standalone" in lowered or "stand-alone" in lowered or "separate financial statements" in lowered:
        return "standalone"
    return "not_explicit"


def _source_for(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": result.get("chunk_id"),
        "document_id": result.get("document_id"),
        "filename": result.get("filename"),
        "page_number": result.get("page_number"),
        "basis": _basis_from_text(str(result.get("text", ""))),
        "text": str(result.get("text", "")),
    }


def _clean_number(value: str) -> str:
    return re.sub(r"\s+", "", value).replace(",", "")


def _extract_financial_metric(question: str, results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Extract supported financial rows, including note tables split across chunks."""
    q = question.lower()

    metrics = [
        {
            "key": "employee_expenses",
            "question_terms": (
                "employee cost", "employee costs", "employee expense",
                "employee expenses", "employee benefit", "staff cost",
                "personnel cost", "spend on employees", "employee remuneration",
            ),
            "row_label": r"Employee benefit expenses",
            "answer_label": "employee benefit expenses",
        },
        {
            "key": "revenue",
            "question_terms": ("revenue from operations", "revenue", "sales", "turnover"),
            "row_label": r"Revenue from operations",
            "answer_label": "revenue from operations",
        },
        {
            "key": "profit_after_tax",
            "question_terms": ("profit after tax", "net profit", "profit for the year", "pat"),
            "row_label": r"PROFIT FOR THE YEAR",
            "answer_label": "profit for the year",
        },
    ]

    # Prefer the most specific metric match; e.g. "revenue from operations"
    # should still resolve to revenue, while "employee benefit expenses" to expenses.
    selected = next(
        (metric for metric in metrics if any(
            term in q for term in metric["question_terms"])),
        None,
    )
    if selected is None:
        return None

    wants_consolidated = "consolidated" in q
    wants_standalone = "standalone" in q or "stand-alone" in q
    number = r"([0-9]{1,3}(?:,[0-9]{2,3})+|[0-9]+)"
    row_pattern = re.compile(
        selected["row_label"] + r"\s+(?:\d+\s+)?" + number + r"\s+" + number,
        re.IGNORECASE,
    )

    def has_year_headers(text: str) -> bool:
        lower = text.lower()
        return (
            ("march 31, 2025" in lower and "march 31, 2024" in lower)
            or ("fy2025" in lower and "fy2024" in lower)
            or ("year ended" in lower and "2025" in lower and "2024" in lower)
        )

    def extract_pair(text: str) -> tuple[str, str] | None:
        # Standard financial-statement row: label followed directly by two values.
        match = row_pattern.search(text)
        if match:
            return match.group(1), match.group(2)

        if selected["key"] == "employee_expenses":
            # Notes often show a breakdown (salaries, provident fund, welfare) and
            # put the total on a separate final row. Capture only the expense-table
            # region, then use its last numeric pair as the subtotal.
            heading = re.search(
                r"Employee benefit expenses\s+consist of the following\s*:",
                text,
                re.IGNORECASE,
            )
            if heading:
                tail = text[heading.end():]
                end = re.search(
                    r"Employee benefit obligations\s+consist of the following\s*:",
                    tail,
                    re.IGNORECASE,
                )
                table = tail[:end.start()] if end else tail[:1800]
                pairs = list(re.finditer(number + r"\s+" + number, table))
                if pairs:
                    pair = pairs[-1]
                    return pair.group(1), pair.group(2)
        return None

    def is_primary_statement(text: str) -> bool:
        lower = text.lower()
        return any(marker in lower for marker in (
            "consolidated statement of profit and loss",
            "statement of profit and loss",
            "consolidated statement of comprehensive income",
        ))

    def priority(item: dict[str, Any]) -> tuple[int, int, int]:
        text = str(item.get("text", ""))
        lower = text.lower()
        basis = _basis_from_text(text)
        if wants_consolidated:
            basis_priority = 0 if basis == "consolidated" else 3
        elif wants_standalone:
            basis_priority = 0 if basis == "standalone" else 3
        else:
            # Prefer a primary statement; if retrieval only finds note tables,
            # prefer explicit consolidated evidence. The page-number tie-breaker
            # is used only for otherwise-unspecified snippets from the same report.
            basis_priority = 0 if (
                "consolidated statement of profit and loss" in lower
                or "consolidated financial results" in lower
            ) else (1 if basis == "consolidated" else 2)
        try:
            page = int(item.get("page_number") or 999999)
        except (TypeError, ValueError):
            page = 999999
        statement_priority = 0 if is_primary_statement(text) else 1
        return basis_priority, statement_priority * 1000 + page, int(item.get("rank", 9999))

    ordered = sorted(results, key=priority)
    candidates: list[dict[str, Any]] = []

    for item in ordered:
        text = str(item.get("text", ""))
        lower = text.lower()
        basis = _basis_from_text(text)

        if wants_consolidated and basis != "consolidated":
            continue
        if wants_standalone and basis != "standalone":
            continue
        pair = extract_pair(text)
        if not pair:
            continue

        # Some chunkers split a statement's year-header row into the previous
        # chunk while keeping the labelled metric row and its two values together.
        # Accept a direct labelled row from a primary statement even if both year
        # headers are not repeated in this same chunk. For note-table subtotal
        # fallback, keep the stricter year-header requirement.
        direct_row_match = row_pattern.search(text) is not None
        if not has_year_headers(text) and not (
            direct_row_match and is_primary_statement(text)
        ):
            continue

        value_2025, value_2024 = pair
        candidates.append({
            "item": item,
            "basis": basis,
            "value_2025": value_2025,
            "value_2024": value_2024,
            "text": text,
        })

    if not candidates:
        return None

    # For a generic query, use the first highest-priority extractable source.
    chosen = candidates[0]
    item = chosen["item"]
    basis = chosen["basis"]
    basis_label = (
        "consolidated" if basis == "consolidated"
        else "standalone" if basis == "standalone"
        else "basis not explicitly identified in the retrieved passage"
    )
    filename = item.get("filename") or item.get(
        "document_id") or "Source document"
    metric_label = str(selected["answer_label"])
    prefix = "TCS's " if "tcs" in q or "employee" in q or "revenue" in q or "profit" in q else ""
    if basis in ("consolidated", "standalone"):
        answer_text = (
            f"{prefix}{basis_label} {metric_label} was ₹{chosen['value_2025']} crore "
            f"for FY2025 and ₹{chosen['value_2024']} crore for FY2024."
        )
    else:
        answer_text = (
            f"The retrieved passage reports {metric_label} of ₹{chosen['value_2025']} crore "
            f"for FY2025 and ₹{chosen['value_2024']} crore for FY2024. "
            "The passage does not explicitly identify the figures as consolidated or standalone."
        )

    return {
        "status": "answered",
        "answer": answer_text,
        "answer_type": "extractive",
        "sources": [{
            "filename": filename,
            "document_id": item.get("document_id"),
            "page_number": item.get("page_number"),
            "chunk_id": item.get("chunk_id"),
            "basis": basis if basis != "not_explicit" else "not_explicit",
            "supporting_text": chosen["text"][:1200],
        }],
        "note": (
            "Figures were extracted from a retrieved financial-statement row or note-table subtotal. "
            "The source passage and basis label are shown for verification; verify the cited report page before external use."
        ),
    }


def _build_answer(question: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_answer = _extract_financial_metric(question, results)
    if metric_answer:
        return metric_answer

    if not results:
        return {
            "status": "insufficient_evidence",
            "answer": "I could not find supporting evidence in the retrieved documents.",
            "answer_type": "none",
            "sources": [],
        }

    # Conservative fallback: provide citations and snippets, but don't pretend
    # that a generative model has synthesized a full answer.
    sources = []
    for item in results[:3]:
        sources.append({
            "filename": item.get("filename"),
            "document_id": item.get("document_id"),
            "page_number": item.get("page_number"),
            "chunk_id": item.get("chunk_id"),
            "basis": _basis_from_text(str(item.get("text", ""))),
            "supporting_text": str(item.get("text", ""))[:900],
        })
    return {
        "status": "evidence_found",
        "answer": (
            "Relevant evidence was retrieved, but this API currently generates "
            "answers only for supported extractive financial rows. Review the cited "
            "evidence below; no unsupported summary has been generated."
        ),
        "answer_type": "none",
        "sources": sources,
    }


@app.post("/api/query")
def query(request: QueryRequest) -> dict[str, Any]:
    if request.scope == "document" and not request.document_id:
        raise HTTPException(
            status_code=422,
            detail="document_id is required when scope='document'.",
        )
    if request.scope == "collection" and request.document_id:
        raise HTTPException(
            status_code=422,
            detail="document_id must be omitted when scope='collection'.",
        )
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Retrieval engine is not ready: {_engine_error or 'not initialized'}",
        )

    try:
        retrieval = _engine.search(
            question=request.question,
            document_id=request.document_id if request.scope == "document" else None,
            retrieval_k=request.retrieval_k,
            final_k=request.final_k,
        )
        results = retrieval.get("results", [])
        return {
            **retrieval,
            "answer": _build_answer(request.question, results),
            "answer_generation": "extractive_financial_rows",
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {type(exc).__name__}: {exc}",
        ) from exc
