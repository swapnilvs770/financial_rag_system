# Financial RAG — local deployment

Hybrid retrieval over annual reports using FastAPI, FAISS, BM25, and BGE embeddings.

## Requirements
- Python 3.11–3.13 recommended (the project dependencies may not yet support every newer Python release).
- Node.js 20+ and npm for the React/Vite frontend.
- Existing project dataset/index files under `data/processed/embeddings/`.

## Backend setup (PowerShell)
From the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn api.main:app --reload
```

The API is available at `http://127.0.0.1:8000`; Swagger is at `http://127.0.0.1:8000/docs`.

## Frontend setup
In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal (normally `http://127.0.0.1:5173`).

Set `VITE_API_BASE_URL` to the deployed API URL when building the frontend for another environment.

## Upload and indexing
Use **Add an annual report** in the UI, or call `POST /api/documents/upload` with multipart form field `file`.
The endpoint accepts text-based PDFs up to 50 MB. It extracts text page by page, creates overlapping chunks, embeds the new chunks with the same `BAAI/bge-small-en-v1.5` model, appends their vectors to FAISS, and rebuilds BM25 over the canonical chunk corpus. Index metadata and canonical chunks are persisted under `data/processed/`.

The upload endpoint intentionally rejects filename collisions rather than overwriting an existing report. Scanned/image-only PDFs are not OCR-processed yet and return a clear error.

## API endpoints
- `GET /health`
- `GET /api/documents`
- `POST /api/documents/upload`
- `POST /api/query`

For a collection query, set `scope` to `collection` and omit `document_id`. For a single-report query, set `scope` to `document` and provide a valid `document_id`.

## Data and security notes
- Do not expose the local development server directly to the public internet.
- Add authentication, rate limiting, HTTPS, private storage, and an appropriate upload policy before public deployment.
- Never commit API keys or secrets.
- Keep backups of `data/processed/` before testing ingestion against important indexes. Index files are updated locally when an upload succeeds.
