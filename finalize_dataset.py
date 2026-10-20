
from pathlib import Path
import csv
import fitz

PDF_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Page numbers are 1-based.
confirmed_blank_pages = {
    "Annual_Report_Amazon_2022.pdf": [88],
    "Annual_Report_Amazon_2023.pdf": [92],
    "Annual_Report_Amazon_2024.pdf": [91],
    "Annual_Report_Amazon_2025.pdf": [92],
    "Annual_Report_Ford_2025.pdf": [2, 210, 211],
    "Annual_Report_Renishaw_2025.pdf": [60],
}

cover_pages = {
    "Annual_Report_Infosys_2025.pdf": [1],
}

records = []

for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
    with fitz.open(pdf_path) as doc:
        blank_pages = []
        text_pages = 0

        for number, page in enumerate(doc, start=1):
            if page.get_text("text").strip():
                text_pages += 1
            else:
                blank_pages.append(number)

        confirmed = confirmed_blank_pages.get(pdf_path.name, [])
        covers = cover_pages.get(pdf_path.name, [])

        unresolved = sorted(
            set(blank_pages) - set(confirmed) - set(covers)
        )

        if unresolved:
            status = "REVIEW_REQUIRED"
        elif blank_pages:
            status = "VALIDATED_WITH_NON_TEXT_PAGES"
        else:
            status = "TEXT_EXTRACTABLE"

        records.append({
            "filename": pdf_path.name,
            "page_count": len(doc),
            "text_pages": text_pages,
            "blank_or_nontext_pages": ";".join(map(str, blank_pages)),
            "confirmed_blank_pages": ";".join(map(str, confirmed)),
            "cover_pages": ";".join(map(str, covers)),
            "unresolved_pages": ";".join(map(str, unresolved)),
            "status": status,
        })

report = OUTPUT_DIR / "final_dataset_inventory.csv"

with report.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print(f"PDFs inventoried: {len(records)}")
print(f"Inventory saved to: {report}")
print("\nStatus summary:")

for status in sorted({r["status"] for r in records}):
    count = sum(r["status"] == status for r in records)
    print(f"{status}: {count}")

unresolved = [
    r for r in records if r["status"] == "REVIEW_REQUIRED"
]

if unresolved:
    print("\nPages requiring further inspection:")
    for record in unresolved:
        print(record["filename"], record["unresolved_pages"])
