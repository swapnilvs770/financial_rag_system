
from pathlib import Path
import csv
import json
import fitz  # PyMuPDF

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATH = OUTPUT_DIR / "extracted_pages.jsonl"
REPORT_PATH = OUTPUT_DIR / "extraction_report.csv"


def extract_pdf_pages(pdf_path):
    """Extract text and metadata from every physical PDF page."""

    records = []

    with fitz.open(pdf_path) as doc:
        page_count = len(doc)
        document_id = pdf_path.stem

        for page_index, page in enumerate(doc):
            # Physical page numbers start at 1.
            page_number = page_index + 1

            text = page.get_text("text")
            text = text.replace("\x00", "").strip()

            record = {
                "document_id": document_id,
                "filename": pdf_path.name,
                "page_number": page_number,
                "total_pages": page_count,
                "text": text,
                "character_count": len(text),
                "word_count": len(text.split()),
                "extraction_status": (
                    "TEXT_EXTRACTED" if text else "EMPTY_TEXT"
                ),
            }

            records.append(record)

    return records


def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {PDF_DIR}"
        )

    all_records = []
    report_rows = []

    print(f"PDF directory: {PDF_DIR}")
    print(f"PDFs found: {len(pdf_files)}")
    print("\nStarting page-level text extraction...\n")

    for pdf_path in pdf_files:
        try:
            records = extract_pdf_pages(pdf_path)
            all_records.extend(records)

            extracted_pages = sum(
                r["extraction_status"] == "TEXT_EXTRACTED"
                for r in records
            )
            empty_pages = len(records) - extracted_pages

            report_rows.append({
                "filename": pdf_path.name,
                "page_count": len(records),
                "text_extracted_pages": extracted_pages,
                "empty_text_pages": empty_pages,
                "total_characters": sum(
                    r["character_count"] for r in records
                ),
                "status": (
                    "SUCCESS" if empty_pages == 0
                    else "REVIEW_REQUIRED"
                ),
            })

            print(
                f"{pdf_path.name}: "
                f"{len(records)} pages, "
                f"{extracted_pages} with text, "
                f"{empty_pages} empty"
            )

        except Exception as exc:
            report_rows.append({
                "filename": pdf_path.name,
                "page_count": "",
                "text_extracted_pages": "",
                "empty_text_pages": "",
                "total_characters": "",
                "status": f"ERROR: {exc}",
            })

            print(f"ERROR processing {pdf_path.name}: {exc}")

    # Write one JSON Lines record per PDF page.
    # UTF-8 preserves non-English text and symbols.
    with JSONL_PATH.open("w", encoding="utf-8") as file:
        for record in all_records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # Write a per-document extraction summary.
    fieldnames = [
        "filename",
        "page_count",
        "text_extracted_pages",
        "empty_text_pages",
        "total_characters",
        "status",
    ]

    with REPORT_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print("\nExtraction finished.")
    print(f"Page records: {len(all_records)}")
    print(f"JSONL output: {JSONL_PATH}")
    print(f"Summary report: {REPORT_PATH}")

    errors = [
        row for row in report_rows
        if str(row["status"]).startswith("ERROR:")
    ]
    empty_pages = sum(
        row["empty_text_pages"]
        for row in report_rows
        if isinstance(row["empty_text_pages"], int)
    )

    print(f"PDF processing errors: {len(errors)}")
    print(f"Pages with empty extracted text: {empty_pages}")


if __name__ == "__main__":
    main()
