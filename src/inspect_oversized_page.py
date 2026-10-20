
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"

TARGET_FILENAME = "Annual_Report_Nvidia_2025.pdf"
TARGET_PAGE = 128

with CHUNKS_FILE.open("r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        chunk = json.loads(line)

        if (
            chunk.get("filename") == TARGET_FILENAME
            and chunk.get("page_number") == TARGET_PAGE
        ):
            print(f"\nChunk: {chunk['chunk_id']}")
            print(f"Characters: {len(chunk.get('text', ''))}")
            print("-" * 70)
            print(chunk.get("text", ""))
            print("=" * 70)
