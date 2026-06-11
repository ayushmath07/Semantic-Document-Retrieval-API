"""Scientific evaluation for calibrated entropy-weighted hybrid retrieval.

The evaluator supports both the original toy corpus and BEIR SciFact. When the
current index metadata says ``dataset=scifact`` it evaluates the 300 SciFact
test claims with BEIR qrels from ``data/scifact/qrels/test.tsv``.

Usage:
    python scripts/build_index.py
    python scripts/evaluate.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.calibration import (  # noqa: E402
    CALIBRATION_METHODS,
    bootstrap_ci,
    clarity_score,
    compute_entropy,
    nqc,
    pearson_correlation,
    wig,
)
from app.datasets import SCIFACT_DIR, load_scifact_golden_set, scifact_available  # noqa: E402
from app.retriever import INDEX_DIR, RETRIEVAL_MODES, DocumentStore  # noqa: E402

EVAL_FILE = Path("eval/golden_set.json")
RESULTS_FILE = Path("eval/results.json")
INDEX_METADATA_FILE = INDEX_DIR / "index_metadata.json"
PRIMARY_K = 10
METRIC_KS = (3, 5)
RETRIEVAL_DEPTH = 20
BOOTSTRAP_RESAMPLES = 1000
CALIBRATION_STYLES = ("raw", "minmax", "zscore", "cdf")
H2_COMPARISONS = (
    ("hybrid_calibrated", "rrf"),
    ("hybrid_calibrated", "hybrid_fixed"),
    ("hybrid_calibrated_rerank", "hybrid_fixed_rerank"),
)


def load_index_metadata() -> dict[str, Any]:
    if not INDEX_METADATA_FILE.exists():
        return {"dataset": "scifact" if scifact_available(SCIFACT_DIR) else "sample"}
    with open(INDEX_METADATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_golden_set(dataset: str) -> list[dict[str, Any]]:
    if dataset == "scifact":
        return load_scifact_golden_set(SCIFACT_DIR)
    with open(EVAL_FILE, encoding="utf-8") as f:
        return json.load(f)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((pct / 100) * len(ordered)) - 1))
    return float(ordered[index])


def clean_float(value: float, digits: int = 6) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return round(float(value), digits)


def is_relevant_source(source: str, item: dict[str, Any]) -> bool:
    return str(source) in {str(s) for s in item.get("relevant_sources", [])}


def is_relevant(result: dict[str, Any], item: dict[str, Any]) -> bool:
    source = str(result.get("source", ""))
    if is_relevant_source(source, item):
        return True
    keywords = [kw.lower() for kw in item.get("relevant_keywords", [])]
    text = f"{source} {result.get('text', '')}".lower()
    return any(keyword in text for keyword in keywords)


def total_relevant_documents(item: dict[str, Any]) -> int:
    return max(1, len({str(s) for s in item.get("relevant_sources", [])}))


def dedupe_results_by_source(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the highest-ranked chunk for each source document."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for result in results:
        source = str(result.get("source", ""))
        if source in seen:
            continue
        seen.add(source)
        unique.append(result)
    return unique


def metric_bundle(
    results: list[dict[str, Any]],
    item: dict[str, Any],
    total_relevant: int,
) -> dict[str, float]:
    unique_results = dedupe_results_by_source(results)
    flags = [1 if is_relevant(result, item) else 0 for result in unique_results[:PRIMARY_K]]
    metrics: dict[str, float] = {}

    for k in METRIC_KS:
        relevant_at_k = sum(flags[:k])
        metrics[f"precision@{k}"] = relevant_at_k / k
        metrics[f"recall@{k}"] = relevant_at_k / total_relevant

    first_rank = next((rank for rank, flag in enumerate(flags, start=1) if flag), None)
    metrics["mrr"] = 1.0 / first_rank if first_rank else 0.0

    dcg = sum(flag / math.log2(rank + 1) for rank, flag in enumerate(flags, start=1))
    ideal_count = min(PRIMARY_K, total_relevant)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    metrics[f"ndcg@{PRIMARY_K}"] = dcg / idcg if idcg else 0.0
    return metrics


def aggregate_mode(rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = [
        "precision@3",
        "precision@5",
        "recall@3",
        "recall@5",
        f"ndcg@{PRIMARY_K}",
        "mrr",
    ]
    summary: dict[str, float] = {}
    for name in metric_names:
        values = [row["metrics"][name] for row in rows]
        summary[name] = clean_float(statistics.mean(values), 4)
        summary[f"{name}_std"] = (
            clean_float(statistics.stdev(values), 4) if len(values) > 1 else 0.0
        )

    latencies = [row["latency_ms"] for row in rows]
    summary.update(
        {
            "latency_mean_ms": clean_float(statistics.mean(latencies), 2),
            "latency_p50_ms": clean_float(statistics.median(latencies), 2),
            "latency_p95_ms": clean_float(percentile(latencies, 95), 2),
            "queries": len(rows),
        }
    )
    return summary


def calibrate_for_entropy(
    scores: np.ndarray,
    method: str,
    corpus_cdf: np.ndarray | None,
) -> np.ndarray:
    fn = CALIBRATION_METHODS[method]
    if method == "cdf":
        if corpus_cdf is None:
            return CALIBRATION_METHODS["minmax"](scores)
        return fn(scores, corpus_cdf=corpus_cdf)
    return fn(scores)


def entropy_debug_rows(
    store: DocumentStore,
    golden_set: list[dict[str, Any]],
    cdf_bm25: np.ndarray | None,
    cdf_dense: np.ndarray | None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in golden_set[:limit]:
        query = item["query"]
        sparse_scores = store.bm25_index.score_all(query)
        dense_scores = store._dense_score_all(query)
        rows.append(
            {
                "query_id": item.get("query_id"),
                "query": query,
                "top_raw_bm25_scores": [
                    clean_float(value, 4) for value in np.sort(sparse_scores)[::-1][:10].tolist()
                ],
                "sparse_entropy_by_calibration": {
                    method: clean_float(
                        compute_entropy(calibrate_for_entropy(sparse_scores, method, cdf_bm25))
                    )
                    for method in CALIBRATION_STYLES
                },
                "dense_entropy_by_calibration": {
                    method: clean_float(
                        compute_entropy(calibrate_for_entropy(dense_scores, method, cdf_dense))
                    )
                    for method in CALIBRATION_STYLES
                },
            }
        )
    return rows


def h1_entropy_analysis(
    store: DocumentStore,
    golden_set: list[dict[str, Any]],
    per_query_by_mode: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    cdf_bm25: np.ndarray | None = None
    cdf_dense: np.ndarray | None = None
    if store.corpus_cdfs is not None:
        cdf_bm25, cdf_dense = store.corpus_cdfs

    entropy_values: dict[str, dict[str, list[float]]] = {
        "sparse": {method: [] for method in CALIBRATION_STYLES},
        "dense": {method: [] for method in CALIBRATION_STYLES},
    }
    qpp_values: dict[str, list[float]] = {
        "clarity_sparse_top10": [],
        "nqc_sparse": [],
        "nqc_dense": [],
        "wig_sparse": [],
        "wig_dense": [],
    }

    for item in golden_set:
        query = item["query"]
        sparse_scores = store.bm25_index.score_all(query)
        dense_scores = store._dense_score_all(query)

        for method in CALIBRATION_STYLES:
            sparse_cal = calibrate_for_entropy(sparse_scores, method, cdf_bm25)
            dense_cal = calibrate_for_entropy(dense_scores, method, cdf_dense)
            entropy_values["sparse"][method].append(compute_entropy(sparse_cal))
            entropy_values["dense"][method].append(compute_entropy(dense_cal))

        sparse_top = store._search_sparse(query, top_k=PRIMARY_K)
        dense_top = store._search_dense(query, top_k=PRIMARY_K)
        sparse_top_scores = np.array([r["score"] for r in sparse_top], dtype=float)
        dense_top_scores = np.array([r["score"] for r in dense_top], dtype=float)
        sparse_mean = store.bm25_index.corpus_mean_score(query)
        dense_mean = float(np.mean(dense_scores)) if len(dense_scores) else 0.0
        query_len = max(1, len(query.split()))

        if store.corpus_lm is not None:
            term_freqs, total_terms = store.corpus_lm
            qpp_values["clarity_sparse_top10"].append(
                clarity_score([r["text"] for r in sparse_top], term_freqs, total_terms)
            )
        else:
            qpp_values["clarity_sparse_top10"].append(0.0)
        qpp_values["nqc_sparse"].append(nqc(sparse_top_scores, sparse_mean))
        qpp_values["nqc_dense"].append(nqc(dense_top_scores, dense_mean))
        qpp_values["wig_sparse"].append(wig(sparse_top_scores, sparse_mean, query_len))
        qpp_values["wig_dense"].append(wig(dense_top_scores, dense_mean, query_len))

    quality = {
        "sparse": {
            "mrr": [row["metrics"]["mrr"] for row in per_query_by_mode["sparse"]],
            f"ndcg@{PRIMARY_K}": [
                row["metrics"][f"ndcg@{PRIMARY_K}"] for row in per_query_by_mode["sparse"]
            ],
        },
        "dense": {
            "mrr": [row["metrics"]["mrr"] for row in per_query_by_mode["dense"]],
            f"ndcg@{PRIMARY_K}": [
                row["metrics"][f"ndcg@{PRIMARY_K}"] for row in per_query_by_mode["dense"]
            ],
        },
    }

    correlations: dict[str, Any] = {}
    for retriever in ("sparse", "dense"):
        correlations[retriever] = {}
        for method in CALIBRATION_STYLES:
            correlations[retriever][method] = {}
            for metric_name, quality_values in quality[retriever].items():
                corr = pearson_correlation(entropy_values[retriever][method], quality_values)
                correlations[retriever][method][metric_name] = {
                    "r": clean_float(corr["r"]),
                    "p_value": clean_float(corr["p_value"]),
                    "n": corr["n"],
                }

    qpp_correlations = {
        predictor: {
            f"hybrid_calibrated_{metric_name}": {
                "r": clean_float(corr["r"]),
                "p_value": clean_float(corr["p_value"]),
                "n": corr["n"],
            }
            for metric_name, values in {
                "mrr": [row["metrics"]["mrr"] for row in per_query_by_mode["hybrid_calibrated"]],
                f"ndcg@{PRIMARY_K}": [
                    row["metrics"][f"ndcg@{PRIMARY_K}"]
                    for row in per_query_by_mode["hybrid_calibrated"]
                ],
            }.items()
            for corr in [pearson_correlation(predictor_values, values)]
        }
        for predictor, predictor_values in qpp_values.items()
    }

    verdict: dict[str, Any] = {}
    for retriever in ("sparse", "dense"):
        verdict[retriever] = {}
        for metric_name in ("mrr", f"ndcg@{PRIMARY_K}"):
            ranked = sorted(
                CALIBRATION_STYLES,
                key=lambda method: correlations[retriever][method][metric_name]["r"],
            )
            verdict[retriever][metric_name] = {
                "best_negative_method": ranked[0],
                "cdf_is_best_negative": ranked[0] == "cdf",
            }

    return {
        "hypothesis": (
            "CDF-calibrated entropy has stronger negative Pearson correlation "
            "with per-query retrieval quality than raw, min-max, or z-score entropy."
        ),
        "entropy_correlations": correlations,
        "entropy_debug": entropy_debug_rows(store, golden_set, cdf_bm25, cdf_dense),
        "qpp_baseline_correlations": qpp_correlations,
        "verdict": verdict,
    }


def bootstrap_with_pvalue(values_a: list[float], values_b: list[float]) -> dict[str, float | bool]:
    ci = bootstrap_ci(values_a, values_b, n_resamples=BOOTSTRAP_RESAMPLES)
    diffs = np.array(values_a, dtype=float) - np.array(values_b, dtype=float)
    if len(diffs) == 0:
        p_value = 1.0
    else:
        rng = np.random.default_rng(seed=42)
        samples = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            indices = rng.integers(0, len(diffs), size=len(diffs))
            samples.append(float(np.mean(diffs[indices])))
        samples_arr = np.array(samples)
        p_less_or_equal_zero = float(np.mean(samples_arr <= 0))
        p_greater_or_equal_zero = float(np.mean(samples_arr >= 0))
        p_value = min(1.0, 2 * min(p_less_or_equal_zero, p_greater_or_equal_zero))
    return {
        "mean_diff": clean_float(ci["mean_diff"]),
        "ci_lower": clean_float(ci["ci_lower"]),
        "ci_upper": clean_float(ci["ci_upper"]),
        "p_value": clean_float(p_value),
        "significant_95": bool(ci["significant"]),
    }


def h2_significance(per_query_by_mode: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    metrics = (f"ndcg@{PRIMARY_K}", "precision@3", "precision@5", "mrr")
    results: dict[str, Any] = {}
    for mode_a, mode_b in H2_COMPARISONS:
        comparison_key = f"{mode_a}_vs_{mode_b}"
        results[comparison_key] = {}
        for metric_name in metrics:
            values_a = [row["metrics"][metric_name] for row in per_query_by_mode[mode_a]]
            values_b = [row["metrics"][metric_name] for row in per_query_by_mode[mode_b]]
            results[comparison_key][metric_name] = bootstrap_with_pvalue(values_a, values_b)
    return {
        "hypothesis": (
            "CDF-calibrated entropy fusion improves aggregate metrics over "
            "RRF and fixed-alpha fusion baselines."
        ),
        "paired_bootstrap": results,
        "resamples": BOOTSTRAP_RESAMPLES,
        "confidence": 0.95,
    }


def evaluate(
    store: DocumentStore,
    golden_set: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    modes = list(RETRIEVAL_MODES)
    per_query_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    relevant_totals = {item["query"]: total_relevant_documents(item) for item in golden_set}

    for mode in modes:
        print(f"  Evaluating mode: {mode}")
        for item in golden_set:
            query = item["query"]
            results, latency_ms, telemetry = store.search(query, top_k=RETRIEVAL_DEPTH, mode=mode)
            unique_results = dedupe_results_by_source(results)
            total_relevant = relevant_totals[query]
            metrics = metric_bundle(unique_results, item, total_relevant)
            per_query_by_mode[mode].append(
                {
                    "query_id": item.get("query_id"),
                    "query": query,
                    "query_type": item.get("query_type", "unknown"),
                    "metrics": {key: clean_float(value, 4) for key, value in metrics.items()},
                    "latency_ms": clean_float(latency_ms, 2),
                    "retrieved": [
                        {
                            "rank": rank,
                            "source": result.get("source"),
                            "chunk": result.get("chunk"),
                            "score": result.get("score"),
                            "relevant": is_relevant(result, item),
                        }
                        for rank, result in enumerate(unique_results[:PRIMARY_K], start=1)
                    ],
                    "expected_sources": item.get("relevant_sources", []),
                    "total_relevant_documents": total_relevant,
                    "telemetry": telemetry,
                }
            )

    mode_summaries = {mode: aggregate_mode(rows) for mode, rows in per_query_by_mode.items()}

    return {
        "metadata": {
            **metadata,
            "generated_at_unix": int(time.time()),
            "golden_queries": len(golden_set),
            "retrieval_modes": modes,
            "primary_metric": f"ndcg@{PRIMARY_K}",
            "candidate_depth": RETRIEVAL_DEPTH,
            "relevance_rule": "document-level source match; keywords only for toy fallback labels",
        },
        "aggregate_metrics": mode_summaries,
        "h1_distributional": h1_entropy_analysis(store, golden_set, per_query_by_mode),
        "h2_downstream": h2_significance(per_query_by_mode),
        "per_query": per_query_by_mode,
    }


def assert_store_ready(store: DocumentStore) -> None:
    missing = []
    if store.vector_store is None:
        missing.append("FAISS index")
    if store.bm25_index.bm25 is None:
        missing.append("BM25 index")
    if store.corpus_cdfs is None:
        missing.append("corpus CDFs")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing {joined}. Run 'python scripts/build_index.py' first.")


def print_summary(results: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print("RETRIEVAL ABLATION SUMMARY")
    print("=" * 88)
    header = (
        f"{'mode':30s} {'NDCG@10':>8s} {'std':>7s} {'MRR':>7s} "
        f"{'P@3':>7s} {'P@5':>7s} {'p95 ms':>8s}"
    )
    print(header)
    print("-" * len(header))
    for mode, metrics in results["aggregate_metrics"].items():
        print(
            f"{mode:30s} "
            f"{metrics[f'ndcg@{PRIMARY_K}']:8.4f} "
            f"{metrics[f'ndcg@{PRIMARY_K}_std']:7.4f} "
            f"{metrics['mrr']:7.4f} "
            f"{metrics['precision@3']:7.4f} "
            f"{metrics['precision@5']:7.4f} "
            f"{metrics['latency_p95_ms']:8.2f}"
        )

    print("\nH1 sparse entropy debug (first 3 queries):")
    for row in results["h1_distributional"]["entropy_debug"]:
        print(f"  {row.get('query_id') or '-'}: {row['query'][:90]}")
        print(f"    top BM25: {row['top_raw_bm25_scores'][:5]}")
        print(f"    sparse H: {row['sparse_entropy_by_calibration']}")

    print("\nH1 best negative entropy correlations:")
    verdict = results["h1_distributional"]["verdict"]
    for retriever, metrics in verdict.items():
        for metric_name, data in metrics.items():
            print(
                f"  {retriever:6s} {metric_name:7s}: "
                f"{data['best_negative_method']} "
                f"(cdf_best={data['cdf_is_best_negative']})"
            )

    print("\nH2 paired bootstrap NDCG@10:")
    comparisons = results["h2_downstream"]["paired_bootstrap"]
    for name, metrics in comparisons.items():
        ndcg = metrics[f"ndcg@{PRIMARY_K}"]
        print(
            f"  {name}: diff={ndcg['mean_diff']:.4f}, "
            f"95% CI [{ndcg['ci_lower']:.4f}, {ndcg['ci_upper']:.4f}], "
            f"p={ndcg['p_value']:.4f}"
        )


def main() -> None:
    print("Loading index metadata...")
    metadata = load_index_metadata()
    dataset = metadata.get("dataset", "sample")
    print(f"  Dataset: {dataset}")

    print("\nLoading evaluation set...")
    golden_set = load_golden_set(dataset)
    print(f"  {len(golden_set)} queries loaded")

    print("\nLoading retrieval artifacts...")
    store = DocumentStore()
    store.load()
    try:
        assert_store_ready(store)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print("  FAISS, BM25, CDF, corpus LM, and lookup artifacts loaded")

    print("\nRunning evaluation...")
    results = evaluate(store, golden_set, metadata)
    print_summary(results)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
