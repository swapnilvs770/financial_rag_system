
from pathlib import Path
import fitz
import csv

PDF_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
REPORT_PATH = OUTPUT_DIR / "pdf_validation_report.csv"


def validate_pdf(pdf_path):
    result = {
        "filename": pdf_path.name,
        "size_kb": round(pdf_path.stat().st_size / 1024, 2),
        "pages": 0,
        "text_pages": 0,
        "total_text_chars": 0,
        "status": "",
    }

    try:
        with fitz.open(pdf_path) as doc:
            if doc.needs_pass:
                result["status"] = "PASSWORD_PROTECTED"
                return result

            result["pages"] = len(doc)

            if len(doc) == 0:
                result["status"] = "EMPTY_PDF"
                return result

            for page in doc:
                text = page.get_text("text").strip()

                if text:
                    result["text_pages"] += 1
                    result["total_text_chars"] += len(text)

            if result["total_text_chars"] == 0:
                result["status"] = "NO_EXTRACTABLE_TEXT"
            elif result["text_pages"] < result["pages"]:
                result["status"] = "PARTIAL_TEXT"
            else:
                result["status"] = "TEXT_EXTRACTABLE"

    except Exception as exc:
        result["status"] = f"ERROR: {type(exc).__name__}"

    return result


def main():
    if not PDF_DIR.exists():
        raise SystemExit(f"Folder not found: {PDF_DIR}")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise SystemExit("No PDF files found in data/raw.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [validate_pdf(path) for path in pdf_files]

    with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nPDFs checked: {len(results)}")
    print("-" * 75)

    for result in results:
        print(
            f"{result['filename']:<42} "
            f"Pages: {result['pages']:<4} "
            f"Status: {result['status']}"
        )

    print(f"\nReport saved to: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
