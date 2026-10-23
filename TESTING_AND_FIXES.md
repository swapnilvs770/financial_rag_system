# Financial RAG — Fixes and Verification Notes

## What changed
- Replaced the same generic fallback response with question-term-ranked extractive evidence sentences for non-tabular questions.
- The fallback now returns `answered` only when question-relevant sentence evidence is found; otherwise it returns `insufficient_evidence` and explains what is missing.
- Returned source metadata includes filename, document ID, page, chunk ID, reporting basis, and supporting text.
- Kept the existing specialized financial-row extractor intact to avoid silently changing its established TCS financial-row behavior.
- Added regression tests for qualitative risk questions, distinct questions, insufficient evidence, and no-result behavior.

## Important scope/limitations
This remains a **retrieval + extractive evidence** system, not a full generative LLM RAG answerer. It selects sentences from retrieved text based on question-term overlap. It does not reliably calculate arbitrary ratios, reconcile values across multiple tables, or synthesize complex narrative explanations. It must not be described as a full LLM-generated answer layer. The current specialized financial-row extractor has existing company/currency/unit assumptions that need a separate, source-aware metric parser before treating all companies' values as universally correct.

## Backend setup (Windows PowerShell)
Use Python 3.13 for this project unless you have validated all dependencies on another Python version.

```powershell
# from the extracted project root
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
uvicorn api.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

## Frontend setup
In another PowerShell terminal:

```powershell
cd frontend
npm install
npm run dev
```

Use the URL Vite prints, normally http://127.0.0.1:5173.

## Manual acceptance tests
1. `GET /health` — expect HTTP 200; `retrieval_ready` should be true once indexes and model load.
2. `GET /api/documents` — should return the indexed reports.
3. Ask a numeric row question supported by the report — expect a metric answer and cited page.
4. Ask a narrative question such as “What risks does the company identify?” — expect extracted risk sentences, not the previous generic fallback.
5. Ask an unrelated question not supported by the selected report — expect `insufficient_evidence`, not an invented answer.
6. Repeat the same question in `scope=document` and `scope=collection`; document scope must not cite another report.
7. Upload a valid text-based PDF — verify page count, chunk count, document listing, and successful query afterward.
8. Upload a non-PDF, empty file, oversized file, and duplicate filename — expect clear 4xx responses.
9. Check source filename/page/chunk IDs in the UI against the actual PDF.
10. Re-run `python -m pytest -q` after any change.

## Test status in the build environment
The backend API unit tests passed: **10 passed**. Two FastAPI deprecation warnings remain for `@app.on_event("startup")`. This environment did not validate a full model/index load, a real-PDF ingestion round trip, or a production frontend build. Perform those integration checks on the target Windows machine before presenting the project as fully verified.

## Data safety
Back up `data/processed/` before testing upload/index mutations. Do not commit `.env`, API keys, user data, or credentials. This app currently has no authentication and is intended for local development only.
