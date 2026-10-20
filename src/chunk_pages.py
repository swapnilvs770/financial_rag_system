
from pathlib import Path
import csv
import json

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_DIR / "data" / "processed" / "extracted_pages.jsonl"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

CHUNKS_PATH = OUTPUT_DIR / "chunks.jsonl"
REPORT_PATH = OUTPUT_DIR / "chunking_report.csv"

# Baseline configuration
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be positive.")

if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE."
    )


def create_chunks(page_record):
    """Split one page into overlapping character-based chunks."""

    text = page_record.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return []

    text = text.strip()
    chunks = []
    start = 0
    chunk_index = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        chunk_text = text[start:end]

        if chunk_text.strip():
            chunk_index += 1

            chunks.append({
                "chunk_id": (
                    f"{page_record['document_id']}"
                    f"_p{page_record['page_number']:04d}"
                    f"_c{chunk_index:03d}"
                ),
                "document_id": page_record["document_id"],
                "filename": page_record["filename"],
                "page_number": page_record["page_number"],
                "total_pages": page_record["total_pages"],
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": end,
                "chunk_size_config": CHUNK_SIZE,
                "chunk_overlap_config": CHUNK_OVERLAP,
                "character_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "text": chunk_text,
            })

        if end == len(text):
            break

        start += step

    return chunks


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    empty_pages = 0
    total_chunks = 0
    document_stats = {}

    # Write incrementally so the complete chunk collection
    # does not need to remain in memory.
    with (
        INPUT_PATH.open("r", encoding="utf-8") as source,
        CHUNKS_PATH.open("w", encoding="utf-8") as destination,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                page_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on input line {line_number}"
                ) from exc

            required = [
                "document_id",
                "filename",
                "page_number",
                "total_pages",
                "text",
            ]
            missing = [
                key for key in required
                if key not in page_record
            ]
            if missing:
                raise ValueError(
                    f"Line {line_number} missing fields: {missing}"
                )

            total_pages += 1
            filename = page_record["filename"]

            stats = document_stats.setdefault(filename, {
                "filename": filename,
                "pages_processed": 0,
                "empty_pages_skipped": 0,
                "chunks_created": 0,
                "total_chunk_characters": 0,
            })
            stats["pages_processed"] += 1

            page_chunks = create_chunks(page_record)

            if not page_chunks:
                empty_pages += 1
                stats["empty_pages_skipped"] += 1
                continue

            for chunk in page_chunks:
                destination.write(
                    json.dumps(chunk, ensure_ascii=False) + "\n"
                )

                total_chunks += 1
                stats["chunks_created"] += 1
                stats["total_chunk_characters"] += (
                    chunk["character_count"]
                )

    report_fields = [
        "filename",
        "pages_processed",
        "empty_pages_skipped",
        "chunks_created",
        "total_chunk_characters",
    ]

    with REPORT_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as report_file:
        writer = csv.DictWriter(
            report_file, fieldnames=report_fields
        )
        writer.writeheader()

        for filename in sorted(document_stats):
            writer.writerow(document_stats[filename])

    print("\nChunking completed.")
    print(f"Page records processed: {total_pages}")
    print(f"Empty pages skipped: {empty_pages}")
    print(f"Chunks created: {total_chunks}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Overlap: {CHUNK_OVERLAP} characters")
    print(f"Chunk output: {CHUNKS_PATH}")
    print(f"Summary report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
