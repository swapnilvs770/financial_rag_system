
import json
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_DIR = ROOT / "data" / "processed" / "embeddings"

VECTORS_FILE = EMBEDDING_DIR / "embeddings.npy"
METADATA_FILE = EMBEDDING_DIR / "embedding_metadata.jsonl"
INDEX_FILE = EMBEDDING_DIR / "financial_rag.index"

def main():
    vectors = np.load(VECTORS_FILE).astype(np.float32)

    with METADATA_FILE.open("r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]

    if vectors.ndim != 2 or vectors.shape[1] != 384:
        raise ValueError(f"Unexpected vector shape: {vectors.shape}")

    if len(metadata) != vectors.shape[0]:
        raise ValueError("Vector count does not match metadata count.")

    if len({item["chunk_id"] for item in metadata}) != len(metadata):
        raise ValueError("Duplicate chunk IDs found in metadata.")

    if not np.isfinite(vectors).all():
        raise ValueError("Vectors contain non-finite values.")

    # Inner product on L2-normalized vectors is cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    if index.ntotal != len(metadata):
        raise ValueError("FAISS index count does not match metadata.")

    faiss.write_index(index, str(INDEX_FILE))

    # Reload the saved index to verify that it is readable.
    loaded_index = faiss.read_index(str(INDEX_FILE))

    print("FAISS index built successfully.")
    print("FAISS version:", faiss.__version__)
    print("Index type: IndexFlatIP")
    print("Similarity metric: Cosine similarity")
    print("Vector dimension:", loaded_index.d)
    print("Indexed vectors:", loaded_index.ntotal)
    print("Metadata records:", len(metadata))
    print("Saved index:", INDEX_FILE)

if __name__ == "__main__":
    main()

