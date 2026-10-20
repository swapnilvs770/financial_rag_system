
import json
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "embedding_ready_chunks.jsonl"

tokenizer = AutoTokenizer.from_pretrained(
    "BAAI/bge-small-en-v1.5",
    use_fast=True
)

total = 0
oversized = []
empty = 0
seen_ids = set()
duplicate_ids = []

with INPUT_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)
        total += 1

        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen_ids:
            duplicate_ids.append(chunk_id)
        seen_ids.add(chunk_id)

        text = chunk.get("text", "")
        if not text.strip():
            empty += 1
            continue

        token_count = len(tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False
        ))

        if token_count > 512:
            oversized.append({
                "chunk_id": chunk_id,
                "tokens_including_special_tokens": token_count
            })

print("Total embedding-ready chunks:", total)
print("Empty chunks:", empty)
print("Duplicate chunk IDs:", len(duplicate_ids))
print("Chunks over 512 tokens:", len(oversized))

for chunk in oversized:
    print(chunk)

if empty or duplicate_ids or oversized:
    print("VALIDATION FAILED")
else:
    print("VALIDATION PASSED")
