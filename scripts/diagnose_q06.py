
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
target = "Annual_Report_TCS_2025_p0181_c001"

for name in (
    "data/processed/embeddings/embedding_metadata.jsonl",
    "data/processed/embeddings/bm25_metadata.jsonl",
):
    path = root / name
    if not path.exists():
        print(f"NOT FOUND: {path}")
        continue

    found = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunk_id = record.get("chunk_id") or record.get("id")
            if chunk_id == target:
                print(f"\nFOUND in {path.name}:")
                print(json.dumps(record, indent=2)[:2000])
                found = True
                break

    if not found:
        print(f"Target not found in {path.name}")
