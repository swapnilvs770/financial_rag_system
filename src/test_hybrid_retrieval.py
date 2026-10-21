from pathlib import Path

code = r'''import contextlib
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FAISS_INDEX_PATH = EMBEDDINGS_DIR / "financial_rag.index"
DENSE_METADATA_PATH = EMBEDDINGS_DIR / "embedding_metadata.jsonl"
BM25_INDEX_PATH = EMBEDDINGS_DIR / "bm25_index.pkl"
BM25_METADATA_PATH = EMBEDDINGS_DIR / "bm25_metadata.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "embedding_ready_chunks.jsonl"

OUTPUT_DIR = PROCESSED_DIR / "retrieval_tests"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

RETRIEVAL_K = 50
FINAL_K = 10
DISPLAY_K = 5
RRF_K = 60

TOKEN_PATTERN = re.compile(
    r"[a-z]+(?:[./'-][a-z0-9]+)*|[0-9]+(?:\.[0-9]+)?"
)

# Query format: (ID, filename stem, question, BM25 expansion, target chunk IDs)
# The natural-language question is used for dense retrieval.
# The expanded text is used only for BM25 lexical retrieval.
TEST_QUERIES = [
    (
        "Q01",
        "employee_cost_ratio",
        "What percentage of TCS revenue went to employee costs in FY2025?",
        "Employee Cost / Total Revenue financial performance ratio FY 2025",
        ["Annual_Report_TCS_2025_p0096_c001_s001"],
    ),
    (
        "Q02",
        "revenue",
        "What was TCS's revenue in FY2025?",
        "Revenue from operations total revenue FY 2025 amount",
        ["Annual_Report_TCS_2025_p0181_c001", "Annual_Report_TCS_2025_p0070_c001"],
    ),
    (
        "Q03",
        "ebit_margin",
        "What was TCS's EBIT margin in FY2025?",
        "EBIT / Total Revenue operating margin FY 2025",
        ["Annual_Report_TCS_2025_p0096_c001_s001", "Annual_Report_TCS_2025_p0088_c002"],
    ),
    (
        "Q04",
        "profit_after_tax_ratio",
        "What percentage of TCS revenue was profit after tax in FY2025?",
        "Profit After Tax / Total Revenue PAT ratio FY 2025",
        ["Annual_Report_TCS_2025_p0096_c001_s002", "Annual_Report_TCS_2025_p0088_c002"],
    ),
    (
        "Q05",
        "revenue_growth",
        "How much did TCS revenue grow in FY2025?",
        "Revenue growth percentage FY 2025 FY 2024",
        ["Annual_Report_TCS_2025_p0096_c001_s002", "Annual_Report_TCS_2025_p0088_c002"],
    ),
    (
        "Q06",
        "employee_expenses",
        "How much did TCS spend on employee costs in FY2025?",
        "Employee cost employee benefit expenses FY 2025 amount",
        ["Annual_Report_TCS_2025_p0181_c001", "Annual_Report_TCS_2025_p0095_c001_s001"],
    ),
    (
        "Q07",
        "other_operating_costs",
        "What were TCS's other operating costs in FY2025?",
        "Other operating cost expenses FY 2025 amount",
        ["Annual_Report_TCS_2025_p0095_c001_s001", "Annual_Report_TCS_2025_p0181_c001"],
    ),
    (
        "Q08",
        "operating_cash_flow",
        "How much operating cash flow did TCS generate in FY2025?",
        "Operating cash flows cash generated FY 2025 amount",
        [],
    ),
    (
        "Q09",
        "debt_ratio",
        "Does TCS report debt excluding lease liabilities in its FY2025 ratio analysis?",
        "Debt excluding lease liabilities equity ratio FY 2025",
        ["Annual_Report_TCS_2025_p0096_c002_s001"],
    ),
    (
        "Q10",
        "earnings_per_share",
        "What was TCS's earnings per share in FY2025?",
        "Earnings per share EPS FY 2025",
        ["Annual_Report_TCS_2025_p0088_c002", "Annual_Report_TCS_2025_p0181_c001"],
    ),
    (
        "Q11",
        "capital_expenditure_ratio",
        "What percentage of TCS revenue was spent on capital expenditure in FY2025?",
        "Capital Expenditure / Total Revenue ratio FY 2025",
        ["Annual_Report_TCS_2025_p0096_c002_s001"],
    ),
    (
        "Q12",
        "historical_employee_cost_ratio",
        "How did TCS's employee cost ratio change between FY2024 and FY2025?",
        "Employee Cost / Total Revenue FY 2024 FY 2025 historical ratio",
        ["Annual_Report_TCS_2025_p0096_c001_s001"],
    ),
]


# --------------------------------------------------
# File and metadata utilities
# --------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load JSON Lines data and report the exact location of malformed rows."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc
    return records


def get_chunk_id(record: dict) -> str:
    chunk_id = record.get("chunk_id") or record.get("id")
    if not chunk_id:
        raise ValueError(
            "A metadata record has no 'chunk_id' or 'id' field. "
            f"Available fields: {list(record.keys())}"
        )
    return str(chunk_id)


def tokenize(text: str) -> list[str]:
    """Use the same tokenization pattern as the existing BM25 setup."""
    return TOKEN_PATTERN.findall(text.lower())


def get_chunk_text(record: dict) -> str:
    for field in ("text", "chunk_text", "content"):
        value = record.get(field)
        if isinstance(value, str):
            return value
    return ""


def get_source(record: dict) -> str:
    return str(
        record.get("source_file")
        or record.get("filename")
        or record.get("source")
        or record.get("document_name")
        or "Unknown source"
    )


def get_page(record: dict) -> str:
    for field in ("page_number", "page", "physical_page"):
        if record.get(field) is not None:
            return str(record[field])
    return "Unknown"


# --------------------------------------------------
# Load retrieval indexes
# --------------------------------------------------

def load_faiss_index():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_INDEX_PATH}\n"
            "Build the existing FAISS index before running these tests."
        )

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    metadata = load_jsonl(DENSE_METADATA_PATH)

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index and dense metadata counts do not match: "
            f"{index.ntotal} vectors vs {len(metadata)} metadata records."
        )
    return index, metadata


def load_bm25_index():
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {BM25_INDEX_PATH}\n"
            "Build the existing BM25 index before running these tests."
        )

    import pickle

    with BM25_INDEX_PATH.open("rb") as file:
        saved_index = pickle.load(file)

    metadata = load_jsonl(BM25_METADATA_PATH)

    # The project's existing pickle may store BM25 inside a dictionary.
    if isinstance(saved_index, dict):
        bm25 = next(
            (
                value
                for value in saved_index.values()
                if hasattr(value, "get_scores")
            ),
            None,
        )
        if bm25 is None:
            raise TypeError(
                "BM25 pickle is a dictionary, but no value exposes get_scores()."
            )
    else:
        bm25 = saved_index

    if not hasattr(bm25, "get_scores"):
        raise TypeError(
            f"Unsupported BM25 object type: {type(bm25).__name__}"
        )

    corpus = getattr(bm25, "corpus", None)
    if corpus is not None and len(corpus) != len(metadata):
        raise ValueError(
            "BM25 corpus and metadata counts do not match: "
            f"{len(corpus)} documents vs {len(metadata)} metadata records."
        )

    return bm25, metadata


# --------------------------------------------------
# Retrieval methods
# --------------------------------------------------

def dense_search(query, model, index, metadata, top_k=RETRIEVAL_K):
    """Retrieve candidates with normalized embeddings and FAISS inner product."""
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    search_k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vector, search_k)
    results = []

    for rank, (score, row_index) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        if row_index < 0:
            continue
        record = metadata[int(row_index)]
        results.append(
            {
                "chunk_id": get_chunk_id(record),
                "rank": rank,
                "score": float(score),
                "metadata": record,
            }
        )
    return results


def lexical_search(query, bm25, metadata, top_k=RETRIEVAL_K):
    """Retrieve candidates using BM25 lexical matching."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    search_k = min(top_k, len(scores))
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: (-float(scores[i]), i),
    )[:search_k]

    results = []
    for rank, row_index in enumerate(ranked_indices, start=1):
        score = float(scores[row_index])
        if score <= 0:
            continue
        record = metadata[row_index]
        results.append(
            {
                "chunk_id": get_chunk_id(record),
                "rank": rank,
                "score": score,
                "metadata": record,
            }
        )
    return results


def reciprocal_rank_fusion(dense_results, lexical_results, rrf_k=RRF_K):
    """Combine rankings using RRF; do not compare FAISS and BM25 score scales."""
    fused = {}

    for method, results in (
        ("dense", dense_results),
        ("bm25", lexical_results),
    ):
        for result in results:
            chunk_id = result["chunk_id"]
            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "metadata": result["metadata"],
                }

            fused[chunk_id]["rrf_score"] += 1.0 / (rrf_k + result["rank"])
            fused[chunk_id][f"{method}_rank"] = result["rank"]
            fused[chunk_id][f"{method}_score"] = result["score"]

    return sorted(
        fused.values(),
        key=lambda item: (-item["rrf_score"], item["chunk_id"]),
    )


# --------------------------------------------------
# Report formatting
# --------------------------------------------------

def format_retrieval_section(title, results, method):
    lines = ["", title, "-" * len(title)]
    for item in results[:DISPLAY_K]:
        if method == "dense":
            score_text = f"Similarity: {item['score']:.4f}"
        else:
            score_text = f"BM25: {item['score']:.4f}"
        lines.append(
            f"Rank {item['rank']:>2} | {score_text} | {item['chunk_id']}"
        )
        lines.append(f"  Source: {get_source(item['metadata'])}")
        lines.append(f"  Physical page: {get_page(item['metadata'])}")
    return lines


def format_hybrid_section(results, chunks_by_id):
    lines = ["", "Top hybrid retrieval results (RRF)", "-" * 40]

    for position, item in enumerate(results[:FINAL_K], start=1):
        record = chunks_by_id.get(item["chunk_id"], item["metadata"])
        dense_rank = item["dense_rank"] if item["dense_rank"] is not None else "-"
        bm25_rank = item["bm25_rank"] if item["bm25_rank"] is not None else "-"

        lines.extend(
            [
                "",
                "-" * 80,
                f"Hybrid rank: {position} | RRF score: {item['rrf_score']:.6f}",
                f"Dense rank: {dense_rank} | BM25 rank: {bm25_rank}",
                f"Chunk ID: {item['chunk_id']}",
                f"Source: {get_source(record)}",
                f"Physical page: {get_page(record)}",
            ]
        )

        evidence = get_chunk_text(record)
        if evidence:
            preview = evidence[:1200]
            if len(evidence) > 1200:
                preview += "..."
            lines.extend(["Evidence:", preview])
        else:
            lines.append("Evidence text is unavailable in the canonical chunk record.")

    return lines


def run_one_query(
    query_id,
    filename_stem,
    question,
    bm25_expansion,
    target_chunk_ids,
    model,
    faiss_index,
    dense_metadata,
    bm25,
    bm25_metadata,
    chunks_by_id,
):
    # Dense retrieval always uses the original question.
    dense_results = dense_search(
        question, model, faiss_index, dense_metadata
    )

    # BM25 uses the explicit expansion configured for this test.
    bm25_query = f"{question} {bm25_expansion}".strip()
    lexical_results = lexical_search(
        bm25_query, bm25, bm25_metadata
    )
    fused_results = reciprocal_rank_fusion(dense_results, lexical_results)

    # Replace index metadata with canonical chunk records for evidence.
    for item in fused_results:
        if item["chunk_id"] in chunks_by_id:
            item["metadata"] = chunks_by_id[item["chunk_id"]]

    dense_ranks = {item["chunk_id"]: item["rank"] for item in dense_results}
    bm25_ranks = {item["chunk_id"]: item["rank"] for item in lexical_results}
    hybrid_ranks = {
        item["chunk_id"]: rank
        for rank, item in enumerate(fused_results, start=1)
    }

    lines = [
        "=" * 100,
        f"{query_id}: {question}",
        "=" * 100,
        "",
        f"Original question: {question}",
        f"BM25 expansion: {bm25_expansion}",
        f"BM25 query used: {bm25_query}",
        f"Expected target chunk IDs: {', '.join(target_chunk_ids) if target_chunk_ids else 'Not configured'}",
        "",
        f"Candidate counts: dense={len(dense_results)}, BM25={len(lexical_results)}, hybrid={len(fused_results)}",
    ]

    lines.extend(format_retrieval_section(
        "Top dense retrieval results (FAISS)", dense_results, "dense"
    ))
    lines.extend(format_retrieval_section(
        "Top lexical retrieval results (BM25)", lexical_results, "bm25"
    ))
    lines.extend(format_hybrid_section(fused_results, chunks_by_id))

    lines.extend(["", "Target-chunk rank check", "-" * 24])
    if target_chunk_ids:
        for target_id in target_chunk_ids:
            lines.append(
                f"{target_id} | dense rank: {dense_ranks.get(target_id, '-')} "
                f"| BM25 rank: {bm25_ranks.get(target_id, '-')} "
                f"| hybrid rank: {hybrid_ranks.get(target_id, '-')}"
            )
    else:
        lines.append("No target chunk IDs configured; inspect retrieved evidence manually.")

    output_path = OUTPUT_DIR / f"{query_id}_{filename_stem}.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    best_hybrid = fused_results[0] if fused_results else None
    target_hybrid_ranks = [
        hybrid_ranks[target_id]
        for target_id in target_chunk_ids
        if target_id in hybrid_ranks
    ]

    summary = {
        "query_id": query_id,
        "filename": output_path.name,
        "question": question,
        "bm25_expansion": bm25_expansion,
        "top_hybrid_chunk": best_hybrid["chunk_id"] if best_hybrid else "None",
        "top_hybrid_source": get_source(best_hybrid["metadata"]) if best_hybrid else "None",
        "top_hybrid_page": get_page(best_hybrid["metadata"]) if best_hybrid else "None",
        "best_configured_target_hybrid_rank": min(target_hybrid_ranks) if target_hybrid_ranks else None,
        "target_found_in_hybrid_top_10": any(
            target_id in hybrid_ranks and hybrid_ranks[target_id] <= FINAL_K
            for target_id in target_chunk_ids
        ) if target_chunk_ids else None,
    }
    return output_path, summary


# --------------------------------------------------
# Main batch runner
# --------------------------------------------------

def main() -> int:
    required_files = [
        FAISS_INDEX_PATH,
        DENSE_METADATA_PATH,
        BM25_INDEX_PATH,
        BM25_METADATA_PATH,
        CHUNKS_PATH,
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        print("ERROR: Required files are missing:")
        for path in missing:
            print(f"  - {path}")
        print("\nBuild the existing indexes before running the batch test.")
        return 1

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("Loading FAISS index and metadata...")
        faiss_index, dense_metadata = load_faiss_index()

        print("Loading BM25 index and metadata...")
        bm25, bm25_metadata = load_bm25_index()

        print("Loading canonical embedding-ready chunks...")
        chunks = load_jsonl(CHUNKS_PATH)
        chunks_by_id = {}
        for chunk in chunks:
            chunk_id = get_chunk_id(chunk)
            if chunk_id in chunks_by_id:
                raise ValueError(
                    f"Duplicate chunk ID in embedding-ready chunks: {chunk_id}"
                )
            chunks_by_id[chunk_id] = chunk

        # Validate index metadata IDs against canonical chunk records.
        for label, records in (
            ("FAISS", dense_metadata),
            ("BM25", bm25_metadata),
        ):
            ids = [get_chunk_id(record) for record in records]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{label} metadata contains duplicate chunk IDs.")
            missing_ids = set(ids) - set(chunks_by_id)
            if missing_ids:
                example = next(iter(missing_ids))
                raise ValueError(
                    f"{label} metadata has {len(missing_ids)} chunk IDs absent "
                    f"from the canonical chunk file. Example: {example}"
                )

        if faiss_index.ntotal != len(chunks_by_id):
            print(
                "WARNING: FAISS vector count differs from canonical chunk count: "
                f"{faiss_index.ntotal} vs {len(chunks_by_id)}"
            )
        if len(bm25_metadata) != len(chunks_by_id):
            print(
                "WARNING: BM25 metadata count differs from canonical chunk count: "
                f"{len(bm25_metadata)} vs {len(chunks_by_id)}"
            )

        print(f"Embedding-ready chunks: {len(chunks_by_id)}")
        print(f"FAISS vectors: {faiss_index.ntotal}")
        print(f"BM25 documents: {len(bm25_metadata)}")
        print(f"Loading embedding model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)

        summaries = []
        total = len(TEST_QUERIES)

        for index, query_config in enumerate(TEST_QUERIES, start=1):
            query_id, filename_stem, question, expansion, targets = query_config
            print(f"[{index}/{total}] Running {query_id}: {question}")

            output_path, summary = run_one_query(
                query_id,
                filename_stem,
                question,
                expansion,
                targets,
                model,
                faiss_index,
                dense_metadata,
                bm25,
                bm25_metadata,
                chunks_by_id,
            )
            summaries.append(summary)
            print(f"    Saved: {output_path}")

        # Write a compact batch-level overview.
        summary_lines = [
            "FINANCIAL RAG — BATCH RETRIEVAL TEST SUMMARY",
            "=" * 100,
            f"Queries tested: {len(summaries)}",
            f"Model: {MODEL_NAME}",
            f"Retrieval candidates per method: {RETRIEVAL_K}",
            f"Hybrid results per query: {FINAL_K}",
            f"RRF_K: {RRF_K}",
            "",
            "Query-level results",
            "-" * 100,
        ]

        for item in summaries:
            target_rank = item["best_configured_target_hybrid_rank"]
            target_status = (
                str(target_rank) if target_rank is not None else "not configured/not retrieved"
            )
            found = item["target_found_in_hybrid_top_10"]
            found_text = (
                "yes" if found is True else "no" if found is False else "not evaluated"
            )
            summary_lines.extend(
                [
                    f"{item['query_id']} | {item['question']}",
                    f"  Report: {item['filename']}",
                    f"  Top hybrid chunk: {item['top_hybrid_chunk']}",
                    f"  Top source/page: {item['top_hybrid_source']} / {item['top_hybrid_page']}",
                    f"  Best configured target hybrid rank: {target_status}",
                    f"  Configured target in hybrid top {FINAL_K}: {found_text}",
                    "",
                ]
            )

        summary_path = OUTPUT_DIR / "batch_summary.txt"
        summary_path.write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8"
        )

        print("")
        print("=" * 80)
        print("BATCH RETRIEVAL TEST COMPLETED")
        print("=" * 80)
        print(f"Reports created: {len(summaries)}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Batch summary: {summary_path}")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Keep terminal output concise while preserving detailed per-query reports.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    console_log = OUTPUT_DIR / "batch_run_log.txt"
    with console_log.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file):
            exit_code = main()

    # Print the log after execution so PowerShell does not truncate long reports.
    if console_log.exists():
        print(console_log.read_text(encoding="utf-8"))
    print(f"Batch run log: {console_log}")
    raise SystemExit(exit_code)
'''

out = Path("src/test_hybrid_retrieval_batch.py")
out.write_text(code, encoding="utf-8")
print(f"Created {out} ({len(code.splitlines())} lines)")
