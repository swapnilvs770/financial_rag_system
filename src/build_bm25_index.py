
import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "embedding_ready_chunks.jsonl"
OUTPUT_DIR = ROOT / "data" / "processed" / "embeddings"

INDEX_FILE = OUTPUT_DIR / "bm25_index.pkl"
METADATA_FILE = OUTPUT_DIR / "bm25_metadata.jsonl"


def tokenize(text):
    """Lowercase tokenization that preserves financial labels and numbers."""
    return re.findall(r"[a-z]+(?:[./'-][a-z0-9]+)*|[0-9]+(?:\.[0-9]+)?", text.lower())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = []
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    if not chunks:
        raise ValueError("No chunks found.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]

    if any(not tokens for tokens in tokenized_corpus):
        raise ValueError("At least one chunk has no searchable tokens.")

    bm25 = BM25Okapi(tokenized_corpus)

    with INDEX_FILE.open("wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "tokenized_corpus": tokenized_corpus,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    with METADATA_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("BM25 index built successfully.")
    print("Indexed chunks:", len(chunks))
    print("Unique chunk IDs:", len(set(chunk_ids)))
    print("Saved index:", INDEX_FILE)
    print("Saved metadata:", METADATA_FILE)


if __name__ == "__main__":
    main()
