"""
Evaluate retrieval quality of the Semantic Document Retrieval API.

Loads the golden evaluation set, runs each query against the FAISS index,
and computes Precision@k, Recall@k, MRR, and latency statistics.

Usage:
    1. Index all sample docs first:  python scripts/build_index.py
    2. Run evaluation:               python scripts/evaluate.py
"""

import json
import statistics
import sys
from pathlib import Path

# Add project root to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retriever import DocumentStore  # noqa: E402


EVAL_FILE = Path("eval/golden_set.json")
TOP_K = 3


def load_golden_set() -> list[dict]:
    with open(EVAL_FILE, encoding="utf-8") as f:
        return json.load(f)


def evaluate(store: DocumentStore, golden_set: list[dict], top_k: int = TOP_K) -> dict:
    """Run all queries and compute retrieval metrics."""
    precisions: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    per_query: list[dict] = []

    for item in golden_set:
        query = item["query"]
        expected_sources = set(item["relevant_sources"])
        keywords = [kw.lower() for kw in item.get("relevant_keywords", [])]

        results, latency_ms = store.search(query, top_k=top_k)
        latencies.append(latency_ms)

        # Check which results are relevant (source matches OR keyword match)
        retrieved_sources = []
        relevant_count = 0
        first_relevant_rank = None

        for rank, r in enumerate(results, start=1):
            source = r["source"]
            text_lower = r["text"].lower()
            retrieved_sources.append(source)

            is_source_match = source in expected_sources
            is_keyword_match = any(kw in text_lower for kw in keywords)
            is_relevant = is_source_match and is_keyword_match

            if is_relevant:
                relevant_count += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank

        # Precision@k: fraction of retrieved results that are relevant
        precision = relevant_count / len(results) if results else 0.0
        precisions.append(precision)

        # Recall@k: fraction of expected sources found in results
        found_sources = expected_sources.intersection(set(retrieved_sources))
        recall = len(found_sources) / len(expected_sources) if expected_sources else 0.0
        recalls.append(recall)

        # Reciprocal Rank: 1/rank of first relevant result
        rr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0
        reciprocal_ranks.append(rr)

        per_query.append({
            "query": query,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "rr": round(rr, 3),
            "latency_ms": latency_ms,
            "retrieved": retrieved_sources,
            "expected": list(expected_sources),
        })

    metrics = {
        f"precision@{top_k}": round(statistics.mean(precisions), 4),
        f"recall@{top_k}": round(statistics.mean(recalls), 4),
        "mrr": round(statistics.mean(reciprocal_ranks), 4),
        "latency_p50_ms": round(statistics.median(latencies), 2),
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else 0,
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "total_queries": len(golden_set),
        "queries_with_results": sum(1 for p in per_query if p["retrieved"]),
    }

    return {"metrics": metrics, "per_query": per_query}


def main() -> None:
    print("Loading evaluation set...")
    golden_set = load_golden_set()
    print(f"  {len(golden_set)} queries loaded\n")

    print("Loading document store and FAISS index...")
    store = DocumentStore()
    store.load()

    if store.vector_store is None:
        print("ERROR: No FAISS index found. Run 'python scripts/build_index.py' first.")
        sys.exit(1)

    print(f"  Index loaded successfully\n")

    print(f"Running evaluation (top_k={TOP_K})...\n")
    results = evaluate(store, golden_set)

    # Print metrics summary
    print("=" * 60)
    print("  RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    m = results["metrics"]
    print(f"  Precision@{TOP_K}:       {m[f'precision@{TOP_K}']:.4f}")
    print(f"  Recall@{TOP_K}:          {m[f'recall@{TOP_K}']:.4f}")
    print(f"  MRR:                {m['mrr']:.4f}")
    print(f"  Latency (p50):      {m['latency_p50_ms']:.2f} ms")
    print(f"  Latency (p95):      {m['latency_p95_ms']:.2f} ms")
    print(f"  Latency (mean):     {m['latency_mean_ms']:.2f} ms")
    print(f"  Total queries:      {m['total_queries']}")
    print(f"  Queries w/ results: {m['queries_with_results']}")
    print("=" * 60)

    # Print per-query breakdown for failures
    failures = [q for q in results["per_query"] if q["precision"] == 0]
    if failures:
        print(f"\n  {len(failures)} queries with zero precision:")
        for f in failures:
            print(f"    - \"{f['query']}\"")
            print(f"      expected: {f['expected']}, got: {f['retrieved']}")

    # Save full results to file
    output_path = Path("eval/results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results saved to {output_path}")


if __name__ == "__main__":
    main()
