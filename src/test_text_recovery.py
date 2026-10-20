
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "raw" / "Annual_Report_Nvidia_2025.pdf"

with fitz.open(PDF_PATH) as pdf:
    text = pdf[127].get_text("text")

print("ORIGINAL EXTRACTED TEXT")
print(repr(text[:500]))

print("\nSIMPLE LETTER-SHIFT TEST")
def shift_letters(value, shift):
    result = []

    for char in value:
        if "a" <= char <= "z":
            result.append(chr((ord(char) - ord("a") + shift) % 26 + ord("a")))
        elif "A" <= char <= "Z":
            result.append(chr((ord(char) - ord("A") + shift) % 26 + ord("A")))
        else:
            result.append(char)

    return "".join(result)

for shift in range(1, 26):
    candidate = shift_letters(text[:250], shift)
    if "The following table" in candidate:
        print(f"Potential shift: {shift}")
        print(candidate[:500])
