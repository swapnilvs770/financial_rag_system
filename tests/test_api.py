from fastapi.testclient import TestClient
import api.main as api_main


class FakeEngine:
    def list_documents(self):
        return [{"document_id": "Annual_Report_TCS_2025", "filename": "Annual_Report_TCS_2025.pdf", "total_pages": 200, "chunk_count": 42}]

    def search(self, question, document_id=None, retrieval_k=50, final_k=10):
        return {
            "question": question,
            "scope": "document" if document_id else "collection",
            "document_id": document_id,
            "result_count": 1,
            "results": [{"rank": 1, "chunk_id": "test_chunk", "document_id": document_id or "Annual_Report_TCS_2025", "filename": "Annual_Report_TCS_2025.pdf", "page_number": 181, "text": "Employee benefit expenses: 145,788 crore", "rrf_score": 0.03, "dense_rank": 2, "bm25_rank": 1}],
        }


def setup_function():
    api_main._engine = FakeEngine()
    api_main._engine_error = None


def test_health_ready():
    response = TestClient(api_main.app).get('/health')
    assert response.status_code == 200
    assert response.json()['retrieval_ready'] is True


def test_documents_endpoint():
    response = TestClient(api_main.app).get('/api/documents')
    assert response.status_code == 200
    assert response.json()['documents'][0]['document_id'] == 'Annual_Report_TCS_2025'


def test_collection_query_returns_citation_metadata():
    response = TestClient(api_main.app).post('/api/query', json={'question': 'How much did TCS spend on employee costs in FY2025?'})
    assert response.status_code == 200
    body = response.json()
    assert body['scope'] == 'collection'
    assert body['results'][0]['filename'] == 'Annual_Report_TCS_2025.pdf'
    assert body['results'][0]['page_number'] == 181


def test_document_scope_requires_document_id():
    response = TestClient(api_main.app).post('/api/query', json={'question': 'revenue?', 'scope': 'document'})
    assert response.status_code == 422


def test_document_scope_passes_document_id():
    response = TestClient(api_main.app).post('/api/query', json={'question': 'revenue?', 'scope': 'document', 'document_id': 'Annual_Report_TCS_2025'})
    assert response.status_code == 200
    assert response.json()['document_id'] == 'Annual_Report_TCS_2025'
