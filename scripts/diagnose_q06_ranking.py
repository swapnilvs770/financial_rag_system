
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from test_hybrid_retrieval_batch import (
    MODEL_NAME,
    load_faiss_index,
    load_bm25_index,
    dense_search,
    lexical_search,
    reciprocal_rank_fusion,
    get_chunk_id,
)
from sentence_transformers import SentenceTransformer

TARGET = "Annual_Report_TCS_2025_p0181_c001"

queries = {
    "ORIGINAL": "How much did TCS spend on employee costs in FY2025?",
    "EXPANDED": (
        "TCS consolidated statement of profit and loss "
        "employee benefit expenses 145788 140131 FY 2025 FY 2024"
    ),
}

print("Loading existing indexes and model...")
faiss_index, dense_metadata = load_faiss_index()
bm25, bm25_metadata = load_bm25_index()
model = SentenceTransformer(MODEL_NAME)

print(f"FAISS vectors: {faiss_index.ntotal}")
print(f"Dense metadata: {len(dense_metadata)}")
print(f"BM25 metadata: {len(bm25_metadata)}")

for label, query in queries.items():
    print(f"\n{'=' * 70}\n{label}: {query}\n{'=' * 70}")

    dense = dense_search(
        query, model, faiss_index, dense_metadata, top_k=50
    )
    bm25_results = lexical_search(
        query, bm25, bm25_metadata, top_k=50
    )
    hybrid = reciprocal_rank_fusion(dense, bm25_results)

    for name, results in (
        ("DENSE", dense),
        ("BM25", bm25_results),
        ("HYBRID", hybrid),
    ):
        target_rank = next(
            (
                rank for rank, item in enumerate(results, start=1)
                if get_chunk_id(item.get("metadata", {})) == TARGET
                or item.get("chunk_id") == TARGET
            ),
            None,
        )
        print(f"{name}: {len(results)} results; target rank={target_rank}")

        for rank, item in enumerate(results[:5], start=1):
            print(
                f"  {rank}. {item.get('chunk_id')} "
                f"score={item.get('score', item.get('rrf_score'))}"
            )
