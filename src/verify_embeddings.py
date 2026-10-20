
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_DIR = ROOT / "data" / "processed" / "embeddings"
CHUNKS_FILE = ROOT / "data" / "processed" / "embedding_ready_chunks.jsonl"

vectors = np.load(EMBEDDING_DIR / "embeddings.npy", mmap_mode="r")

with (EMBEDDING_DIR / "embedding_metadata.jsonl").open(
    encoding="utf-8"
) as f:
    metadata = [json.loads(line) for line in f]

with CHUNKS_FILE.open(encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

assert vectors.shape == (8728, 384), f"Unexpected shape: {vectors.shape}"
assert len(metadata) == len(vectors), "Metadata/vector count mismatch"
assert len(chunks) == len(vectors), "Chunk/vector count mismatch"

metadata_ids = [item["chunk_id"] for item in metadata]
chunk_ids = [item["chunk_id"] for item in chunks]

assert metadata_ids == chunk_ids, "Metadata order does not match chunks"
assert len(set(metadata_ids)) == len(metadata_ids), "Duplicate metadata IDs"

# Check that vectors are normalized without loading the full matrix into RAM.
norms = np.linalg.norm(np.asarray(vectors), axis=1)
assert np.isfinite(norms).all(), "Non-finite vector norms found"

max_norm_error = float(np.max(np.abs(norms - 1.0)))
assert max_norm_error < 1e-4, f"Vectors are not normalized: {max_norm_error}"

print("Embedding verification PASSED")
print("Vector shape:", vectors.shape)
print("Metadata records:", len(metadata))
print("Chunk records:", len(chunks))
print("Chunk IDs aligned:", metadata_ids == chunk_ids)
print("Maximum normalization error:", max_norm_error)
