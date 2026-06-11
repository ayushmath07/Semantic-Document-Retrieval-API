"""Fusion strategies for combining sparse (BM25) and dense (FAISS) retrieval results.

Implements multiple fusion approaches for ablation comparison:
- Reciprocal Rank Fusion (RRF) — rank-based, score-agnostic baseline
- Linear combination — fixed α weighted sum
- Entropy-weighted fusion — adaptive α per query via calibrated entropy

All fusion methods produce a unified candidate list from two retriever outputs.
"""

from typing import Any

import numpy as np

from app.calibration import (
    CALIBRATION_METHODS,
    compute_alpha,
    compute_entropy,
)


def _merge_candidates(
    sparse_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge candidates from both retrievers, keyed by (source, chunk).

    Returns dict mapping key → {source, chunk, text, sparse_score, dense_score}.
    """
    merged: dict[str, dict[str, Any]] = {}

    for r in sparse_results:
        key = f"{r['source']}:{r['chunk']}"
        if key not in merged:
            merged[key] = {
                "source": r["source"],
                "chunk": r["chunk"],
                "text": r["text"],
                "sparse_score": r.get("score", 0.0),
                "dense_score": 0.0,
            }
        else:
            merged[key]["sparse_score"] = r.get("score", 0.0)

    for r in dense_results:
        key = f"{r['source']}:{r['chunk']}"
        if key not in merged:
            merged[key] = {
                "source": r["source"],
                "chunk": r["chunk"],
                "text": r["text"],
                "sparse_score": 0.0,
                "dense_score": r.get("score", 0.0),
            }
        else:
            merged[key]["dense_score"] = r.get("score", 0.0)

    return merged


def rrf_fuse(
    sparse_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    k: int = 60,
    top_k: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reciprocal Rank Fusion.

    RRF(d) = Σ 1/(k + rank(d)) across retriever lists.
    Score-agnostic, operates on ranks only.

    Returns (fused_results, metadata).
    """
    rrf_scores: dict[str, float] = {}
    doc_data: dict[str, dict[str, Any]] = {}

    # Process sparse results
    for rank, r in enumerate(sparse_results, start=1):
        key = f"{r['source']}:{r['chunk']}"
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in doc_data:
            doc_data[key] = {
                "source": r["source"],
                "chunk": r["chunk"],
                "text": r["text"],
            }

    # Process dense results
    for rank, r in enumerate(dense_results, start=1):
        key = f"{r['source']}:{r['chunk']}"
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in doc_data:
            doc_data[key] = {
                "source": r["source"],
                "chunk": r["chunk"],
                "text": r["text"],
            }

    # Sort by RRF score
    sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    results = []
    for key in sorted_keys:
        r = doc_data[key]
        r["score"] = round(rrf_scores[key], 6)
        results.append(r)

    if top_k is not None:
        results = results[:top_k]

    metadata = {"fusion_method": "rrf", "rrf_k": k}
    return results, metadata


def linear_fuse(
    sparse_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    alpha: float = 0.5,
    calibration: str = "minmax",
    corpus_cdfs: tuple[np.ndarray, np.ndarray] | None = None,
    top_k: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Linear combination with fixed alpha.

    fused_score = α * sparse_calibrated + (1-α) * dense_calibrated

    Args:
        alpha: Weight for sparse retriever (0=all dense, 1=all sparse).
        calibration: Score calibration method ('raw', 'minmax', 'zscore', 'cdf').
        corpus_cdfs: (cdf_bm25, cdf_dense) required when calibration='cdf'.
    """
    merged = _merge_candidates(sparse_results, dense_results)
    if not merged:
        return [], {
            "fusion_method": "linear",
            "alpha": alpha,
            "calibration": calibration,
        }

    sparse_scores = np.array([m["sparse_score"] for m in merged.values()])
    dense_scores = np.array([m["dense_score"] for m in merged.values()])

    effective_calibration = calibration
    if calibration == "cdf" and corpus_cdfs is None:
        effective_calibration = "minmax"

    cal_fn = CALIBRATION_METHODS[effective_calibration]
    if effective_calibration == "cdf" and corpus_cdfs is not None:
        cal_sparse = cal_fn(sparse_scores, corpus_cdf=corpus_cdfs[0])
        cal_dense = cal_fn(dense_scores, corpus_cdf=corpus_cdfs[1])
    else:
        cal_sparse = cal_fn(sparse_scores)
        cal_dense = cal_fn(dense_scores)

    fused_scores = alpha * cal_sparse + (1 - alpha) * cal_dense

    results = []
    for (_key, data), score in zip(merged.items(), fused_scores, strict=True):
        results.append(
            {
                "source": data["source"],
                "chunk": data["chunk"],
                "text": data["text"],
                "score": round(float(score), 6),
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    if top_k is not None:
        results = results[:top_k]

    metadata = {
        "fusion_method": "linear",
        "alpha": alpha,
        "calibration": effective_calibration,
        "requested_calibration": calibration,
    }
    return results, metadata


def entropy_fuse(
    sparse_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    calibration: str = "cdf",
    corpus_cdfs: tuple[np.ndarray, np.ndarray] | None = None,
    top_k: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Entropy-weighted adaptive fusion.

    Computes per-query alpha from calibrated score distribution entropy:
    α = H_dense / (H_dense + H_sparse + ε)

    When dense has high entropy (low confidence), α is large → more weight on BM25.
    """
    merged = _merge_candidates(sparse_results, dense_results)
    if not merged:
        return [], {
            "fusion_method": "entropy_weighted",
            "alpha": 0.5,
            "h_sparse": 0.0,
            "h_dense": 0.0,
            "calibration": calibration,
        }

    sparse_scores = np.array([m["sparse_score"] for m in merged.values()])
    dense_scores = np.array([m["dense_score"] for m in merged.values()])

    effective_calibration = calibration
    if calibration == "cdf" and corpus_cdfs is None:
        effective_calibration = "minmax"

    cal_fn = CALIBRATION_METHODS[effective_calibration]
    if effective_calibration == "cdf" and corpus_cdfs is not None:
        cal_sparse = cal_fn(sparse_scores, corpus_cdf=corpus_cdfs[0])
        cal_dense = cal_fn(dense_scores, corpus_cdf=corpus_cdfs[1])
    else:
        cal_sparse = cal_fn(sparse_scores)
        cal_dense = cal_fn(dense_scores)

    # Compute entropy of each retriever's calibrated scores
    h_sparse = compute_entropy(cal_sparse)
    h_dense = compute_entropy(cal_dense)

    # Adaptive alpha
    alpha = compute_alpha(h_dense, h_sparse)

    # Fuse
    fused_scores = alpha * cal_sparse + (1 - alpha) * cal_dense

    results = []
    for (_key, data), score in zip(merged.items(), fused_scores, strict=True):
        results.append(
            {
                "source": data["source"],
                "chunk": data["chunk"],
                "text": data["text"],
                "score": round(float(score), 6),
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    if top_k is not None:
        results = results[:top_k]

    metadata = {
        "fusion_method": "entropy_weighted",
        "alpha": round(alpha, 6),
        "h_sparse": round(h_sparse, 6),
        "h_dense": round(h_dense, 6),
        "calibration": effective_calibration,
        "requested_calibration": calibration,
    }
    return results, metadata
