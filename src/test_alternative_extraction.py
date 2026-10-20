
from pathlib import Path
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "raw" / "Annual_Report_Nvidia_2025.pdf"

# Physical PDF page 128 corresponds to zero-based index 127.
with pdfplumber.open(PDF_PATH) as pdf:
    page = pdf.pages[127]
    text = page.extract_text(layout=False)

    print("Page:", 128)
    print("Text extraction successful:", bool(text and text.strip()))
    print("-" * 70)
    print(text[:6000] if text else "No text extracted")
