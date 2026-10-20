
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FAISS_INDEX_PATH = EMBEDDINGS_DIR / "financial_rag.index"
DENSE_METADATA_PATH = EMBEDDINGS_DIR / "embedding_metadata.jsonl"

BM25_INDEX_PATH = EMBEDDINGS_DIR / "bm25_index.pkl"
BM25_METADATA_PATH = EMBEDDINGS_DIR / "bm25_metadata.jsonl"

CHUNKS_PATH = PROCESSED_DIR / "embedding_ready_chunks.jsonl"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
RETRIEVAL_K = 20
FINAL_K = 5
RRF_K = 60


# Keep the same tokenizer used to build your BM25 index.
TOKEN_PATTERN = re.compile(
    r"[a-z]+(?:[./'-][a-z0-9]+)*|[0-9]+(?:\.[0-9]+)?"
)


# --------------------------------------------------
# File utilities
# --------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

    return records


def get_chunk_id(record: dict) -> str:
    """Extract a chunk identifier from a metadata record."""
    chunk_id = record.get("chunk_id") or record.get("id")

    if not chunk_id:
        raise ValueError(
            "A metadata record has no 'chunk_id' or 'id' field. "
            f"Available fields: {list(record.keys())}"
        )

    return str(chunk_id)


def tokenize(text: str) -> list[str]:
    """Tokenize text using the BM25 index's tokenizer."""
    return TOKEN_PATTERN.findall(text.lower())


def get_chunk_text(record: dict) -> str:
    """Support the common text field names used in chunk records."""
    for field in ("text", "chunk_text", "content"):
        value = record.get(field)

        if isinstance(value, str):
            return value

    return ""


# --------------------------------------------------
# Load retrieval systems
# --------------------------------------------------

def load_faiss_index():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_INDEX_PATH}\n"
            "Run src/build_faiss_index.py first."
        )

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    metadata = load_jsonl(DENSE_METADATA_PATH)

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index and dense metadata counts do not match: "
            f"{index.ntotal} vectors vs {len(metadata)} metadata records."
        )

    return index, metadata


def load_bm25_index():
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {BM25_INDEX_PATH}\n"
            "Run src/build_bm25_index.py first."
        )

    import pickle

    with BM25_INDEX_PATH.open("rb") as file:
        saved_index = pickle.load(file)

    metadata = load_jsonl(BM25_METADATA_PATH)

    # Support either a directly pickled BM25 object or a dictionary
    # containing the BM25 object and related index data.
    if isinstance(saved_index, dict):
        print(
            "BM25 pickle dictionary keys:",
            list(saved_index.keys()),
        )

        bm25 = next(
            (
                value
                for value in saved_index.values()
                if hasattr(value, "get_scores")
            ),
            None,
        )

        if bm25 is None:
            raise TypeError(
                "The BM25 pickle is a dictionary, but no value exposes "
                "'get_scores'. Inspect the printed dictionary keys and "
                "the structure saved by src/build_bm25_index.py."
            )
    else:
        bm25 = saved_index

    if not hasattr(bm25, "get_scores"):
        raise TypeError(
            f"Loaded BM25 object has unsupported type: {type(bm25).__name__}"
        )

    # Validate corpus alignment when the object exposes a corpus.
    corpus = getattr(bm25, "corpus", None)

    if corpus is not None and len(corpus) != len(metadata):
        raise ValueError(
            "BM25 corpus and metadata counts do not match: "
            f"{len(corpus)} documents vs {len(metadata)} metadata records."
        )

    return bm25, metadata


# --------------------------------------------------
# Individual retrieval methods
# --------------------------------------------------

def dense_search(
    query: str,
    model: SentenceTransformer,
    index,
    metadata: list[dict],
    top_k: int = RETRIEVAL_K,
) -> list[dict]:
    """Retrieve candidates using FAISS cosine similarity."""
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    search_k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vector, search_k)

    results = []

    for rank, (score, row_index) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        if row_index < 0:
            continue

        record = metadata[int(row_index)]

        results.append({
            "chunk_id": get_chunk_id(record),
            "rank": rank,
            "score": float(score),
            "metadata": record,
        })

    return results


def lexical_search(
    query: str,
    bm25,
    metadata: list[dict],
    top_k: int = RETRIEVAL_K,
) -> list[dict]:
    """Retrieve candidates using BM25 lexical matching."""
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    search_k = min(top_k, len(scores))

    # Stable sorting makes tied scores deterministic.
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: (-float(scores[i]), i),
    )[:search_k]

    results = []

    for rank, row_index in enumerate(ranked_indices, start=1):
        score = float(scores[row_index])

        # Skip documents with no lexical match.
        if score <= 0:
            continue

        record = metadata[row_index]

        results.append({
            "chunk_id": get_chunk_id(record),
            "rank": rank,
            "score": score,
            "metadata": record,
        })

    return results


# --------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------

def reciprocal_rank_fusion(
    dense_results: list[dict],
    lexical_results: list[dict],
    rrf_k: int = RRF_K,
) -> list[dict]:
    """
    Fuse dense and lexical rankings.

    RRF score = sum(1 / (rrf_k + rank))

    The original similarity scores are not compared directly because
    FAISS and BM25 use different score scales.
    """
    fused = {}

    for method, results in (
        ("dense", dense_results),
        ("bm25", lexical_results),
    ):
        for result in results:
            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "metadata": result["metadata"],
                }

            fused[chunk_id]["rrf_score"] += (
                1.0 / (rrf_k + result["rank"])
            )

            fused[chunk_id][f"{method}_rank"] = result["rank"]
            fused[chunk_id][f"{method}_score"] = result["score"]

    return sorted(
        fused.values(),
        key=lambda item: (
            -item["rrf_score"],
            item["chunk_id"],
        ),
    )


# --------------------------------------------------
# Display results
# --------------------------------------------------

def get_source(record: dict) -> str:
    return str(
        record.get("source_file")
        or record.get("filename")
        or record.get("source")
        or record.get("document_name")
        or "Unknown source"
    )


def get_page(record: dict) -> str:
    for field in ("page_number", "page", "physical_page"):
        if record.get(field) is not None:
            return str(record[field])

    return "Unknown"


def display_results(
    query: str,
    dense_results: list[dict],
    lexical_results: list[dict],
    fused_results: list[dict],
) -> None:
    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    print("\nTop dense retrieval results (FAISS):")
    for item in dense_results[:5]:
        print(
            f"  Rank {item['rank']:>2} | "
            f"Similarity: {item['score']:.4f} | "
            f"{item['chunk_id']}"
        )

    print("\nTop lexical retrieval results (BM25):")
    for item in lexical_results[:5]:
        print(
            f"  Rank {item['rank']:>2} | "
            f"BM25: {item['score']:.4f} | "
            f"{item['chunk_id']}"
        )

    print("\nTop hybrid retrieval results (RRF):")

    for position, item in enumerate(fused_results[:FINAL_K], start=1):
        record = item["metadata"]
        dense_rank = (
            str(item["dense_rank"])
            if item["dense_rank"] is not None else "-"
        )
        bm25_rank = (
            str(item["bm25_rank"])
            if item["bm25_rank"] is not None else "-"
        )

        print("\n" + "-" * 80)
        print(
            f"Hybrid Rank: {position} | "
            f"RRF Score: {item['rrf_score']:.6f}"
        )
        print(f"Dense Rank: {dense_rank} | BM25 Rank: {bm25_rank}")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Source: {get_source(record)}")
        print(f"Physical page: {get_page(record)}")

        text = get_chunk_text(record)
        if text:
            preview = text[:1000]
            if len(text) > 1000:
                preview += "..."
            print(f"Evidence:\n{preview}")
        else:
            print("Evidence text is unavailable in the metadata record.")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> int:
    required_files = [
        FAISS_INDEX_PATH,
        DENSE_METADATA_PATH,
        BM25_INDEX_PATH,
        BM25_METADATA_PATH,
        CHUNKS_PATH,
    ]

    missing = [str(path) for path in required_files if not path.exists()]

    if missing:
        print("ERROR: The following required files are missing:")
        for path in missing:
            print(f"  - {path}")
        print("\nBuild the existing indexes before running this test.")
        return 1

    try:
        print("Loading FAISS index and metadata...")
        faiss_index, dense_metadata = load_faiss_index()

        print("Loading BM25 index and metadata...")
        bm25, bm25_metadata = load_bm25_index()

        # Validate unique IDs and build a canonical evidence lookup.
        chunks = load_jsonl(CHUNKS_PATH)
        chunks_by_id = {}

        for chunk in chunks:
            chunk_id = get_chunk_id(chunk)

            if chunk_id in chunks_by_id:
                raise ValueError(
                    f"Duplicate chunk ID in embedding-ready chunks: {chunk_id}"
                )

            chunks_by_id[chunk_id] = chunk

        for label, records in (
            ("FAISS", dense_metadata),
            ("BM25", bm25_metadata),
        ):
            ids = [get_chunk_id(record) for record in records]

            if len(ids) != len(set(ids)):
                raise ValueError(f"{label} metadata contains duplicate IDs.")

            missing_ids = set(ids) - set(chunks_by_id)

            if missing_ids:
                raise ValueError(
                    f"{label} metadata contains {len(missing_ids)} chunk IDs "
                    "that are absent from embedding_ready_chunks.jsonl. "
                    f"Example: {next(iter(missing_ids))}"
                )

        print(f"Embedding-ready chunks: {len(chunks_by_id)}")
        print(f"FAISS vectors: {faiss_index.ntotal}")
        print(f"BM25 documents: {len(bm25_metadata)}")

        print(f"\nLoading embedding model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)

        test_queries = [
            (
                "Natural-language query",
                "What were TCS's employee cost-to-revenue ratio, "
                "EBIT-to-revenue ratio, and profit-after-tax-to-revenue "
                "ratio for FY 2025?",
            ),
            (
                "Exact financial table labels",
                "Employee Cost / Total Revenue EBIT / Total Revenue "
                "Profit After Tax / Total Revenue FY 2025",
            ),
        ]

        for label, query in test_queries:
            print(f"\n\nTEST: {label}")

            dense_results = dense_search(
                query, model, faiss_index, dense_metadata
            )
            lexical_results = lexical_search(
                query, bm25, bm25_metadata
            )

            fused_results = reciprocal_rank_fusion(
                dense_results, lexical_results
            )

            # Use the canonical chunk records for evidence text and source.
            for item in fused_results:
                item["metadata"] = chunks_by_id[item["chunk_id"]]

            display_results(
                query,
                dense_results,
                lexical_results,
                fused_results,
            )

        print("\n" + "=" * 100)
        print("HYBRID RETRIEVAL TEST COMPLETED")
        print("=" * 100)
        print(
            "Inspect whether the TCS FY 2025 ratio-table chunk "
            "appears near the top for both query styles."
        )
        print(
            "Expected target chunk ID: "
            "Annual_Report_TCS_2025_p0096_c001_s001"
        )

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
