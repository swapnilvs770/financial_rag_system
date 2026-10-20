
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "processed" / "embedding_ready_chunks.jsonl"
OUTPUT_DIR = ROOT / "data" / "processed" / "embeddings"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = []
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    if not chunks:
        raise ValueError("No chunks found in the input dataset.")

    texts = [chunk["text"] for chunk in chunks]
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.shape != (len(chunks), 384):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    vectors_path = OUTPUT_DIR / "embeddings.npy"
    metadata_path = OUTPUT_DIR / "embedding_metadata.jsonl"
    info_path = OUTPUT_DIR / "embedding_info.json"

    np.save(vectors_path, embeddings)

    # Each metadata record corresponds to the vector at the same row.
    with metadata_path.open("w", encoding="utf-8") as f:
        for row, chunk in enumerate(chunks):
            record = {
                "vector_index": row,
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    info = {
        "model_name": MODEL_NAME,
        "embedding_dimension": int(embeddings.shape[1]),
        "number_of_embeddings": int(embeddings.shape[0]),
        "normalized": True,
        "dtype": str(embeddings.dtype),
        "source_file": str(INPUT_FILE.relative_to(ROOT)),
    }

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("\nEmbedding generation completed.")
    print("Embedding shape:", embeddings.shape)
    print("Embedding dimension:", embeddings.shape[1])
    print("All values finite:", bool(np.isfinite(embeddings).all()))
    print("Vectors:", vectors_path)
    print("Metadata:", metadata_path)
    print("Model info:", info_path)


if __name__ == "__main__":
    main()
