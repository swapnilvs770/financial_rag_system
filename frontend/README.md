# FinSight — Financial RAG frontend

React + Vite + Tailwind CSS frontend for the existing Financial RAG FastAPI backend.

## Requirements
- Node.js 20.19+ or 22.12+ recommended
- Existing FastAPI backend running at `http://127.0.0.1:8000`

## Run on Windows PowerShell

1. Extract this folder somewhere convenient, or copy its files into a `frontend` folder inside your existing `Financial-RAG` project.
2. Open PowerShell in the folder containing this `package.json`.
3. Run:

```powershell
npm install
npm run dev
```

4. Open `http://127.0.0.1:5173`.

In a second PowerShell window, start your backend from the Financial-RAG project root:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload
```

Backend docs remain at `http://127.0.0.1:8000/docs`.

## API assumptions
The frontend calls:
- `GET /health`
- `GET /api/documents`
- `POST /api/query`

Query payload:

```json
{
  "question": "How much did TCS spend on employee costs in FY2025?",
  "scope": "collection",
  "document_id": null,
  "retrieval_k": 50,
  "final_k": 10
}
```

The UI is defensive about response shape and includes an expandable raw-response inspector. It expects the answer to be in `response.answer` and sources in `answer.sources` or common equivalent fields. If your API uses a different schema, inspect the raw response and adjust `src/App.jsx`.

## Optional API URL override

Create `.env` in this frontend folder:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Restart Vite after changing `.env`.

## Troubleshooting
- **Backend offline:** make sure `uvicorn api.main:app --reload` is running.
- **CORS error:** configure FastAPI CORS middleware to allow `http://127.0.0.1:5173`. If your current API already has CORS configured, no change is needed.
- **No reports shown:** inspect `GET /api/documents`; the UI supports an array response or arrays under `documents`, `items`, `data`, or `results`.
- **Query validation error:** compare the request body with the schema shown at `http://127.0.0.1:8000/docs`. Keep the backend schema as the source of truth.
