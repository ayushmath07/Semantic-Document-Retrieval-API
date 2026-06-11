"""Build FAISS, BM25, corpus CDFs, corpus LM, and doc lookup artifacts.

By default this indexes BEIR SciFact when ``data/scifact`` is present. Use
``--dataset sample`` to rebuild the original toy CS corpus.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --dataset sample
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.calibration import (  # noqa: E402
    build_corpus_cdfs,
    build_corpus_language_model,
    save_corpus_cdfs,
    save_corpus_lm,
)
from app.datasets import (  # noqa: E402
    SCIFACT_DIR,
    load_scifact_corpus,
    load_scifact_golden_set,
    scifact_available,
)
from app.retriever import INDEX_DIR, DocumentStore, chunk_text  # noqa: E402

SUPPORTED = {".txt", ".md", ".pdf"}
GOLDEN_SET_PATH = Path("eval/golden_set.json")
INDEX_METADATA_PATH = INDEX_DIR / "index_metadata.json"
MIN_CDF_QUERIES = 100


def choose_dataset(requested: str) -> str:
    if requested != "auto":
        return requested
    return "scifact" if scifact_available(SCIFACT_DIR) else "sample"


def load_sample_golden_queries() -> list[str]:
    if not GOLDEN_SET_PATH.exists():
        return []
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        golden = json.load(f)
    return [item["query"] for item in golden]


def load_dataset_queries(dataset: str) -> list[str]:
    if dataset == "scifact":
        return [item["query"] for item in load_scifact_golden_set(SCIFACT_DIR)]
    return load_sample_golden_queries()


def generate_pseudo_queries(store: DocumentStore, count: int) -> list[str]:
    """Generate pseudo-queries from random indexed chunks."""
    if not store.bm25_index.doc_metadata:
        return []

    all_texts = [m["text"] for m in store.bm25_index.doc_metadata]
    random.seed(42)
    selected = random.sample(all_texts, min(count, len(all_texts)))

    pseudo_queries = []
    for text in selected:
        words = text.split()
        if len(words) >= 5:
            start = random.randint(0, max(0, len(words) - 8))
            span = " ".join(words[start : start + random.randint(5, 8)])
            pseudo_queries.append(span)

    return pseudo_queries


def index_sample_documents(store: DocumentStore) -> dict[str, Any]:
    docs_dir = Path("data/sample_docs")
    if not docs_dir.exists():
        raise FileNotFoundError(f"{docs_dir} not found")

    files = sorted(f for f in docs_dir.iterdir() if f.suffix.lower() in SUPPORTED)
    print(f"Step 1: Indexing {len(files)} sample documents...\n")

    total_chunks = 0
    total_characters = 0
    for file_path in files:
        stats = store.add_file(file_path)
        total_chunks += stats["chunks"]
        total_characters += stats["characters"]
        print(
            f"  * {stats['filename']:40s} "
            f"{stats['chunks']:3d} chunks  {stats['characters']:6d} chars"
        )

    return {
        "dataset": "sample",
        "documents": len(files),
        "chunks": total_chunks,
        "characters": total_characters,
    }


def index_scifact_documents(store: DocumentStore) -> dict[str, Any]:
    records = load_scifact_corpus(SCIFACT_DIR)
    if not records:
        raise FileNotFoundError(f"No SciFact documents found under {SCIFACT_DIR}")

    print(f"Step 1: Indexing {len(records)} SciFact corpus documents...\n")
    documents = []
    total_characters = 0
    for record in records:
        total_characters += len(record["text"])
        documents.extend(chunk_text(record["text"], record["doc_id"]))

    stats = store.add_documents(documents)
    print(
        f"  Indexed {len(records)} source documents as "
        f"{stats['chunks']} chunks ({total_characters} chars)."
    )
    return {
        "dataset": "scifact",
        "documents": len(records),
        "chunks": stats["chunks"],
        "characters": total_characters,
    }


def build_calibration_artifacts(
    store: DocumentStore, dataset: str, total_chunks: int
) -> dict[str, Any]:
    print("\nStep 2: Building corpus-level CDFs...")

    golden_queries = load_dataset_queries(dataset)
    random.seed(42)
    cdf_golden_queries = golden_queries[: min(50, len(golden_queries))]
    n_pseudo = max(0, MIN_CDF_QUERIES - len(cdf_golden_queries))
    pseudo_queries = generate_pseudo_queries(store, n_pseudo)
    sample_queries = cdf_golden_queries + pseudo_queries

    print(
        f"  Sample queries: {len(cdf_golden_queries)} labeled + "
        f"{len(pseudo_queries)} pseudo = {len(sample_queries)} total"
    )

    cdf_bm25, cdf_dense = build_corpus_cdfs(
        bm25_score_fn=store.bm25_index.score_all,
        dense_score_fn=store._dense_score_all,
        sample_queries=sample_queries,
    )

    save_corpus_cdfs(cdf_bm25, cdf_dense, INDEX_DIR)
    print(f"  BM25 CDF: {len(cdf_bm25)} scores (shape: {cdf_bm25.shape})")
    print(f"  Dense CDF: {len(cdf_dense)} scores (shape: {cdf_dense.shape})")

    print("\nStep 3: Building corpus language model...")
    all_texts = [m["text"] for m in store.bm25_index.doc_metadata]
    term_freqs, total_terms = build_corpus_language_model(all_texts)
    save_corpus_lm(term_freqs, total_terms, INDEX_DIR / "corpus_lm.pkl")
    print(f"  Vocabulary size: {len(term_freqs)} unique terms")
    print(f"  Total terms: {total_terms}")

    return {
        "cdf_sample_queries": len(sample_queries),
        "cdf_scores_per_retriever": len(sample_queries) * total_chunks,
        "vocabulary_terms": len(term_freqs),
        "total_terms": total_terms,
    }


def write_index_metadata(metadata: dict[str, Any]) -> None:
    INDEX_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build retrieval artifacts.")
    parser.add_argument(
        "--dataset",
        choices=("auto", "sample", "scifact"),
        default="auto",
        help="Dataset to index. auto prefers SciFact when data/scifact exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = choose_dataset(args.dataset)
    store = DocumentStore()

    if dataset == "scifact":
        corpus_stats = index_scifact_documents(store)
    else:
        corpus_stats = index_sample_documents(store)

    print(f"\n  FAISS index: {store.vector_store.index.ntotal} vectors")
    print(f"  BM25 index: {len(store.bm25_index.tokenized_corpus)} documents")

    artifact_stats = build_calibration_artifacts(store, dataset, corpus_stats["chunks"])
    metadata = {**corpus_stats, **artifact_stats}
    write_index_metadata(metadata)

    print(f"\n{'=' * 60}")
    print("  BUILD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Dataset:          {metadata['dataset']}")
    print(f"  Documents:        {metadata['documents']}")
    print(f"  Chunks:           {metadata['chunks']}")
    print(f"  FAISS vectors:    {store.vector_store.index.ntotal}")
    print(f"  BM25 documents:   {len(store.bm25_index.tokenized_corpus)}")
    print(
        f"  CDF samples:      {metadata['cdf_sample_queries']} queries x {metadata['chunks']} docs"
    )
    print(f"  Vocabulary:       {metadata['vocabulary_terms']} terms")
    print(f"  Doc lookup:       {len(store.doc_lookup)} entries")
    print(f"  Metadata:         {INDEX_METADATA_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
