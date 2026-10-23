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
import threading
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
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
_ingestion_lock = threading.Lock()
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


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
        "answer_generation": "financial_rows_plus_evidence_extractive_fallback",
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


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload, extract, chunk, embed, and index a new text-based PDF."""
    global _engine, _engine_error
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Retrieval engine is not ready: {_engine_error or 'not initialized'}",
        )
    filename = Path(file.filename or "").name
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file with a .pdf extension.")
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / filename
    if destination.exists():
        raise HTTPException(status_code=409, detail=f"A file named '{filename}' already exists. Rename the upload to avoid overwriting it.")
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the 50 MB upload limit.")
    # Basic PDF signature validation before writing anything to the raw library.
    if not payload.lstrip().startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="The uploaded file does not have a valid PDF signature.")
    destination.write_bytes(payload)
    try:
        with _ingestion_lock:
            from ingestion import ingest_pdf
            result = ingest_pdf(destination, ROOT, _engine.model)
            from retrieval_engine import HybridRetrievalEngine
            # Reload the new persistent indexes while reusing the already-loaded model.
            _engine = HybridRetrievalEngine(project_root=ROOT, model=_engine.model)
            _engine_error = None
        return {"message": "PDF processed and indexed successfully.", **result,
                "documents": _engine.list_documents()}
    except ValueError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Keep the uploaded PDF for diagnosis only if indexing may have partially committed.
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {type(exc).__name__}: {exc}") from exc


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
            "question_terms": (
                "profit after tax", "net profit", "profit for the year", "pat",
                "net income", "profit attributable",
            ),
            "row_label": (
                r"(?:Profit after tax attributable to shareholders of the Company|"
                r"Profit attributable to (?:owners|shareholders)(?: of the parent| of the Company)?|"
                r"Profit for the year|Net income|PROFIT FOR THE YEAR)"
            ),
            "answer_label": "profit after tax / net profit",
        },
    ]

    # Prefer the most specific metric match; e.g. "revenue from operations"
    # should still resolve to revenue, while "employee benefit expenses" to expenses.
    selected = next(
        (metric for metric in metrics if any(term in q for term in metric["question_terms"])),
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
            or (re.search(r"fy\s*2025", lower) and re.search(r"fy\s*2024", lower))
            or ("year ended" in lower and "2025" in lower and "2024" in lower)
        )

    def extract_pair(text: str) -> tuple[str, str] | None:
        # Some financial-results overview tables place multiple FY columns side by
        # side: FY2025 Amount, % of Revenue, % Growth, FY2024 Amount, % of Revenue.
        # In that layout, taking the first two numbers is wrong (e.g. 255,324 and
        # 100.0). Select the first and fourth numeric cells for revenue/profit rows.
        if selected["key"] in ("revenue", "profit_after_tax"):
            label_match = re.search(selected["row_label"], text, re.IGNORECASE)
            if label_match:
                tail = text[label_match.end():label_match.end() + 500]
                numeric_tokens = re.findall(
                    r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?",
                    tail,
                )
                table_has_percent_columns = (
                    "% of" in text.lower() or "% growth" in text.lower()
                    or "financial results" in text.lower()
                )
                if table_has_percent_columns and len(numeric_tokens) >= 4:
                    first, fourth = numeric_tokens[0], numeric_tokens[3]
                    return first, fourth
                if len(numeric_tokens) >= 2:
                    # Ordinary two-period statements have only the two amount
                    # cells immediately following the row label.
                    return numeric_tokens[0], numeric_tokens[1]

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
        has_financial_table_context = any(
            marker in text.lower()
            for marker in (
                "consolidated financial results",
                "financial performance overview",
                "revenue from operations",
            )
        )
        if not has_year_headers(text) and not (
            direct_row_match
            and (is_primary_statement(text) or has_financial_table_context)
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
    filename = item.get("filename") or item.get("document_id") or "Source document"
    metric_label = str(selected["answer_label"])
    stem = Path(str(filename)).stem
    company_match = re.match(r"(?:Annual_Report_|annual_report_)?(.+?)(?:_20\d{2})?$", stem, re.IGNORECASE)
    company = company_match.group(1).replace("_", " ").strip() if company_match else ""
    prefix = f"{company}'s " if company and company.lower() not in {"annual report", "report"} else ""
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

    # Evidence-grounded fallback for non-tabular questions. Select sentences
    # using question-term overlap; do not claim to perform abstractive synthesis.
    # This supports qualitative questions without requiring an external LLM key.
    stop_words = {
        "what", "which", "when", "where", "who", "why", "how", "much", "many",
        "does", "did", "was", "were", "is", "are", "the", "a", "an", "of",
        "in", "on", "for", "to", "and", "or", "from", "with", "by", "about",
        "company", "report", "please", "tell", "me", "their", "its", "this",
        "that", "year", "years", "fy", "during", "compare", "explain", "summarize",
    }
    terms = {
        token for token in re.findall(r"[a-z][a-z0-9-]{2,}", question.lower())
        if token not in stop_words
    }
    ranked_sentences: list[tuple[int, int, dict[str, Any], str]] = []
    for result_index, item in enumerate(results):
        text = str(item.get("text", "")).strip()
        # PDFs frequently flatten tables; keep line breaks as useful boundaries.
        pieces = re.split(r"(?<=[.!?])\\s+|\\n+", text)
        seen = set()
        for sentence in pieces:
            sentence = re.sub(r"\\s+", " ", sentence).strip(" -|\\t")
            if len(sentence) < 25:
                continue
            normalized = sentence.lower()
            overlap = sum(1 for term in terms if term in normalized)
            # Retain a little evidence for short or paraphrased questions, but
            # never return a generic success message as though it were an answer.
            if overlap:
                key = normalized[:240]
                if key not in seen:
                    ranked_sentences.append((overlap, -result_index, item, sentence))
                    seen.add(key)
    ranked_sentences.sort(key=lambda row: (row[0], row[1]), reverse=True)

    sources = []
    used_sentences = []
    used_chunks = set()
    for overlap, _, item, sentence in ranked_sentences[:4]:
        used_sentences.append(sentence[:650])
        cid = item.get("chunk_id") or (item.get("filename"), item.get("page_number"))
        if cid not in used_chunks:
            used_chunks.add(cid)
            sources.append({
                "filename": item.get("filename"),
                "document_id": item.get("document_id"),
                "page_number": item.get("page_number"),
                "chunk_id": item.get("chunk_id"),
                "basis": _basis_from_text(str(item.get("text", ""))),
                "supporting_text": str(item.get("text", ""))[:900],
            })

    if used_sentences:
        answer_text = "The retrieved evidence states: " + " ".join(used_sentences)
        status = "answered"
        answer_type = "extractive_evidence"
        note = (
            "This is an extractive answer assembled from sentences in retrieved passages, "
            "not an LLM-generated interpretation. Check the cited pages and surrounding context."
        )
    else:
        answer_text = (
            "I retrieved passages, but could not identify a sentence that directly answers "
            "this question. Try naming the metric, reporting period, or topic more precisely, "
            "or inspect the retrieved evidence before drawing a conclusion."
        )
        status = "insufficient_evidence"
        answer_type = "none"
        note = "No answer was inferred because question-relevant sentence evidence was not identified."
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
        "status": status,
        "answer": answer_text,
        "answer_type": answer_type,
        "sources": sources,
        "note": note,
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
            "answer_generation": "financial_rows_plus_evidence_extractive_fallback",
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {type(exc).__name__}: {exc}",
        ) from exc
