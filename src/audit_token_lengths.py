
import json
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

EXCLUSIONS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "excluded_documents.json"
)

with EXCLUSIONS_FILE.open("r", encoding="utf-8") as file:
    exclusion_data = json.load(file)

EXCLUDED_FILENAMES = {
    item["filename"]
    for item in exclusion_data["excluded_documents"]
    if item.get("status") == "temporarily_excluded"
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

total_chunks = 0
token_lengths = []
empty_chunks = 0
over_limit = 0

with CHUNKS_FILE.open("r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        chunk = json.loads(line)
        if chunk.get("filename") in EXCLUDED_FILENAMES:
            continue
        text = chunk.get("text", "")

        total_chunks += 1

        if not text.strip():
            empty_chunks += 1
            continue

        token_ids = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        length = len(token_ids)
        token_lengths.append(length)

        if length > 512:
            over_limit += 1

token_lengths.sort()

if not token_lengths:
    raise RuntimeError("No non-empty chunks were found.")


def percentile(values, p):
    index = round((len(values) - 1) * p)
    return values[index]

print("Excluded documents:", sorted(EXCLUDED_FILENAMES))
print("Total chunks:", total_chunks)
print("Non-empty chunks:", len(token_lengths))
print("Empty chunks:", empty_chunks)
print("Minimum tokens:", min(token_lengths))
print("Median tokens:", percentile(token_lengths, 0.50))
print("95th percentile tokens:", percentile(token_lengths, 0.95))
print("99th percentile tokens:", percentile(token_lengths, 0.99))
print("Maximum tokens:", max(token_lengths))
print("Chunks exceeding 512 tokens:", over_limit)
print(
    "Percentage exceeding limit:",
    round(over_limit / len(token_lengths) * 100, 2),
)

