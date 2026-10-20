
import json
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


EXCLUSIONS_FILE = (
    ROOT / "data" / "processed" / "excluded_documents.json"
)

with EXCLUSIONS_FILE.open("r", encoding="utf-8") as file:
    exclusion_data = json.load(file)

EXCLUDED_FILENAMES = {
    item["filename"]
    for item in exclusion_data["excluded_documents"]
    if item.get("status") == "temporarily_excluded"
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

count = 0

with CHUNKS_FILE.open("r", encoding="utf-8") as file:
    for line in file:

        if not line.strip():
            continue

        chunk = json.loads(line)
        text = chunk.get("text", "")
        if chunk.get("filename") in EXCLUDED_FILENAMES:
            continue
        token_ids = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        if len(token_ids) > 512:
            count += 1
            print(
                f"Chunk ID: {chunk.get('chunk_id')}\n"
                f"Document: {chunk.get('filename')}\n"
                f"Page: {chunk.get('page_number')}\n"
                f"Tokens: {len(token_ids)}\n"
                f"Characters: {len(text)}\n"
                f"{'-' * 60}"
            )

print(f"Total oversized chunks: {count}")
