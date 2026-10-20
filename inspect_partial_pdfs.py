
from pathlib import Path
import fitz

PDF_DIR = Path("data/raw")

for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
    try:
        with fitz.open(pdf_path) as doc:
            missing_pages = []

            for page_number, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()

                if not text:
                    missing_pages.append(page_number)

            if missing_pages:
                print(f"\n{pdf_path.name}")
                print(f"Total pages: {len(doc)}")
                print(f"Pages without extractable text: {len(missing_pages)}")
                print(f"Page numbers: {missing_pages}")

    except Exception as exc:
        print(f"\nCould not inspect {pdf_path.name}: {exc}")
