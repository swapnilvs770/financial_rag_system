
import json
import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_DIR = ROOT / "data" / "processed" / "embeddings"

INDEX_FILE = EMBEDDING_DIR / "bm25_index.pkl"
METADATA_FILE = EMBEDDING_DIR / "bm25_metadata.jsonl"

QUESTION = (
    "Employee Cost / Total Revenue "
    "EBIT / Total Revenue "
    "Profit After Tax / Total Revenue "
    "FY 2025"
)

TOP_K = 5


def tokenize(text):
    return re.findall(
        r"[a-z]+(?:[./'-][a-z0-9]+)*|[0-9]+(?:\.[0-9]+)?",
        text.lower(),
    )


def main():
    # Load only the index created by your own trusted build script.
    with INDEX_FILE.open("rb") as f:
        saved = pickle.load(f)

    bm25 = saved["bm25"]

    with METADATA_FILE.open("r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]

    if len(metadata) != len(saved["tokenized_corpus"]):
        raise ValueError("BM25 corpus and metadata counts do not match.")

    query_tokens = tokenize(QUESTION)
    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    print("QUESTION:", QUESTION)
    print(f"\nTOP {TOP_K} BM25 RESULTS")
    print("=" * 80)

    for rank, idx in enumerate(top_indices, start=1):
        item = metadata[int(idx)]

        print(f"\nRank: {rank}")
        print(f"BM25 score: {float(scores[idx]):.4f}")
        print(f"Document: {item['filename']}")
        print(f"Page: {item['page_number']}")
        print(f"Chunk ID: {item['chunk_id']}")
        print("-" * 80)
        print(item["text"][:1200])

    print("\nBM25 retrieval test completed.")


if __name__ == "__main__":
    main()

