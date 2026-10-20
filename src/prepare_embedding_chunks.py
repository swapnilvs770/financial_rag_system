
import json
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"
EXCLUSIONS_FILE = ROOT / "data" / "processed" / "excluded_documents.json"
OUTPUT_FILE = ROOT / "data" / "processed" / "embedding_ready_chunks.jsonl"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 400
OVERLAP_TOKENS = 50


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    with EXCLUSIONS_FILE.open("r", encoding="utf-8") as f:
        exclusions = json.load(f)

    excluded_files = {
        item["filename"]
        for item in exclusions.get("excluded_documents", [])
    }

    total_input = 0
    excluded_count = 0
    split_count = 0
    output_count = 0

    with CHUNKS_FILE.open("r", encoding="utf-8") as source, \
         OUTPUT_FILE.open("w", encoding="utf-8") as destination:

        for line in source:
            chunk = json.loads(line)
            total_input += 1

            if chunk.get("filename") in excluded_files:
                excluded_count += 1
                continue

            text = chunk.get("text", "")
            if not text.strip():
                continue

            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=True,
            )

            offsets = encoded["offset_mapping"]
            token_count = len(offsets)

            if token_count <= MAX_TOKENS:
                destination.write(json.dumps(
                    chunk, ensure_ascii=False
                ) + "\n")
                output_count += 1
                continue

            split_count += 1
            segment_number = 1
            token_start = 0

            while token_start < token_count:
                token_end = min(
                    token_start + MAX_TOKENS, token_count
                )

                char_start = (
                    0 if token_start == 0
                    else offsets[token_start][0]
                )
                char_end = (
                    len(text) if token_end == token_count
                    else offsets[token_end - 1][1]
                )

                segment_text = text[char_start:char_end]

                if segment_text:
                    segment = chunk.copy()
                    segment["parent_chunk_id"] = chunk["chunk_id"]
                    segment["chunk_id"] = (
                        f'{chunk["chunk_id"]}_s{segment_number:03d}'
                    )
                    segment["segment_index"] = segment_number
                    segment["segment_token_count"] = (
                        token_end - token_start
                    )
                    segment["start_char"] = (
                        chunk["start_char"] + char_start
                    )
                    segment["end_char"] = (
                        chunk["start_char"] + char_end
                    )
                    segment["character_count"] = len(segment_text)
                    segment["word_count"] = len(segment_text.split())
                    segment["text"] = segment_text

                    destination.write(json.dumps(
                        segment, ensure_ascii=False
                    ) + "\n")
                    output_count += 1

                segment_number += 1

                if token_end == token_count:
                    break

                token_start = token_end - OVERLAP_TOKENS

    print(f"Input chunks: {total_input}")
    print(f"Excluded chunks' documents: {excluded_count}")
    print(f"Original chunks split: {split_count}")
    print(f"Output chunks: {output_count}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
