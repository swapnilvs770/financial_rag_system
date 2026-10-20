
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device="cpu")

sentences = [
    "Amazon reported revenue growth in its annual report.",
    "Apple reported its annual revenue and operating expenses.",
]

embeddings = model.encode(
    sentences,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

print("Model loaded successfully.")
print("Embedding shape:", embeddings.shape)
print("Embedding dimension:", model.get_sentence_embedding_dimension())
print("Similarity:", round(float(embeddings[0] @ embeddings[1]), 4))
print("All embedding values finite:", bool(__import__("numpy").isfinite(embeddings).all()))
