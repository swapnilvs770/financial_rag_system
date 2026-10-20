
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "raw" / "Annual_Report_Nvidia_2025.pdf"

with fitz.open(PDF_PATH) as pdf:
    page = pdf[127]  # Physical page 128

    print("Page:", 128)
    print("\nFonts used on this page:")

    for font in page.get_fonts(full=True):
        print(font)

    print("\nText extraction sample:")
    print(repr(page.get_text("text")[:500]))

    print("\nText blocks:")
    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue

        for line in block["lines"]:
            for span in line["spans"]:
                print({
                    "font": span.get("font"),
                    "font_size": span.get("size"),
                    "text_sample": span.get("text", "")[:100],
                })
                if span.get("text"):
                    break
            else:
                continue
            break
        else:
            continue
        break
