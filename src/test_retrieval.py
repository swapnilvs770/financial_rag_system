
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_DIR = ROOT / "data" / "processed" / "embeddings"

INDEX_FILE = EMBEDDING_DIR / "financial_rag.index"
METADATA_FILE = EMBEDDING_DIR / "embedding_metadata.jsonl"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 5

QUESTION = (
    "What percentage of TCS revenue went to employee costs in FY2025?"
)


def main():
    index = faiss.read_index(str(INDEX_FILE))

    with METADATA_FILE.open("r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]

    if index.ntotal != len(metadata):
        raise ValueError("Index and metadata counts do not match.")

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    query_vector = model.encode(
        [QUESTION],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores, indices = index.search(query_vector, TOP_K)

    print("\nQUESTION:", QUESTION)
    print(f"\nTOP {TOP_K} RETRIEVED CHUNKS")
    print("=" * 80)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        if idx < 0:
            continue

        item = metadata[int(idx)]

        print(f"\nRank: {rank}")
        print(f"Cosine similarity: {float(score):.4f}")
        print(f"Document: {item['filename']}")
        print(f"Page: {item['page_number']}")
        print(f"Chunk ID: {item['chunk_id']}")
        print("-" * 80)
        print(item["text"][:1200])

    print("\nRetrieval test completed.")


if __name__ == "__main__":
    main()
