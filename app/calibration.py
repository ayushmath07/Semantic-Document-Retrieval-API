"""Score calibration, entropy computation, and QPP predictors.

This is the research module. It addresses the cross-retriever score calibration
problem: BM25 produces unbounded term-frequency scores while dense retrievers
produce bounded similarity scores. Comparing entropy across these distributions
is meaningless without calibration to a common probability space.

Corpus-level CDFs are built offline during indexing (not per-query top-k, which
would normalize away the distributional signal by construction).

QPP baselines (Clarity Score, NQC, WIG) are included for head-to-head comparison
against calibrated entropy as a retrieval quality predictor.
"""

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Score Calibration Methods
# ---------------------------------------------------------------------------


def calibrate_raw(scores: np.ndarray, **_: Any) -> np.ndarray:
    """Identity calibration (baseline). No transformation."""
    return np.asarray(scores, dtype=float)


def calibrate_minmax(scores: np.ndarray, **_: Any) -> np.ndarray:
    """Min-max normalization to [0, 1]."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return scores
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-10:
        return np.full_like(scores, 0.5)
    return (scores - min_s) / (max_s - min_s)


def calibrate_zscore(scores: np.ndarray, **_: Any) -> np.ndarray:
    """Z-score normalization. Returns standardized scores (mean=0, std=1)."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return scores
    std_s = scores.std()
    if std_s < 1e-10:
        return np.zeros_like(scores)
    return (scores - scores.mean()) / std_s


def calibrate_cdf(scores: np.ndarray, corpus_cdf: np.ndarray) -> np.ndarray:
    """CDF quantile normalization using a corpus-level CDF.

    For each score, returns its percentile rank in the corpus distribution.
    The corpus_cdf is a sorted array of scores built offline during indexing.

    This maps any distribution to approximately uniform [0,1], but crucially
    the mapping is FIXED (corpus-level), so per-query variation is preserved.
    """
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0 or len(corpus_cdf) == 0:
        return scores
    calibrated = np.searchsorted(corpus_cdf, scores) / len(corpus_cdf)
    return calibrated


CALIBRATION_METHODS = {
    "raw": calibrate_raw,
    "minmax": calibrate_minmax,
    "zscore": calibrate_zscore,
    "cdf": calibrate_cdf,
}


# ---------------------------------------------------------------------------
# Entropy Computation
# ---------------------------------------------------------------------------


def compute_entropy(scores: np.ndarray, epsilon: float = 1e-10) -> float:
    """Shannon entropy of a score distribution.

    Normalizes scores to a probability distribution, then computes
    H = -sum(p * log2(p)).

    Edge cases:
    - Empty or single-element: returns 0.0
    - All equal: returns log2(n) (maximum entropy)
    - Negative values: uses softmax, which preserves z-score shape without
      subtracting away the affine transform being tested.
    """
    scores = np.asarray(scores, dtype=float)
    if len(scores) <= 1:
        return 0.0

    if np.allclose(scores, scores[0]):
        return math.log2(len(scores))

    if np.any(scores < 0):
        shifted = scores - np.max(scores)
        exp_scores = np.exp(np.clip(shifted, -745, 0))
        total = exp_scores.sum()
        if total < epsilon:
            return 0.0
        probs = exp_scores / total
    else:
        adjusted = scores + epsilon
        total = adjusted.sum()
        if total < epsilon:
            return math.log2(len(scores))
        probs = adjusted / total

    # Shannon entropy (base 2)
    entropy = -float(np.sum(probs * np.log2(probs + epsilon)))
    return entropy


def compute_alpha(h_dense: float, h_sparse: float, epsilon: float = 1e-8) -> float:
    """Compute entropy-weighted fusion weight.

    α = H_dense / (H_dense + H_sparse + ε)

    Bayesian precision-weighted mixing interpretation:
    If each retriever produces a likelihood estimate P(rel|d,q), the entropy
    of its calibrated score distribution approximates the inverse precision of
    that estimate. Weighting by inverse entropy is proportional to weighting by
    the precision (inverse variance) of each retriever's likelihood — a standard
    precision-weighted Bayesian combination.

    When dense has high entropy (low confidence), α is large → more weight on BM25.
    When sparse has high entropy (low confidence), α is small → more weight on dense.
    """
    return h_dense / (h_dense + h_sparse + epsilon)


# ---------------------------------------------------------------------------
# Corpus-Level CDF Construction (Offline)
# ---------------------------------------------------------------------------


def build_corpus_cdfs(
    bm25_score_fn: Any,
    dense_score_fn: Any,
    sample_queries: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build corpus-level CDFs for both retrievers.

    For each sample query, scores ALL documents with both retrievers.
    Pools all scores into sorted arrays (one per retriever).

    Args:
        bm25_score_fn: callable(query) → np.ndarray of BM25 scores for all docs
        dense_score_fn: callable(query) → np.ndarray of dense scores for all docs
        sample_queries: list of representative queries (golden set + pseudo-queries)

    Returns:
        (cdf_bm25, cdf_dense): sorted score arrays for CDF lookup
    """
    all_bm25_scores: list[float] = []
    all_dense_scores: list[float] = []

    for query in sample_queries:
        bm25_scores = bm25_score_fn(query)
        dense_scores = dense_score_fn(query)

        if len(bm25_scores) > 0:
            all_bm25_scores.extend(bm25_scores.tolist())
        if len(dense_scores) > 0:
            all_dense_scores.extend(dense_scores.tolist())

    cdf_bm25 = np.sort(np.array(all_bm25_scores))
    cdf_dense = np.sort(np.array(all_dense_scores))

    return cdf_bm25, cdf_dense


def save_corpus_cdfs(cdf_bm25: np.ndarray, cdf_dense: np.ndarray, directory: Path) -> None:
    """Save corpus-level CDFs to disk as .npy files."""
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "corpus_cdf_bm25.npy", cdf_bm25)
    np.save(directory / "corpus_cdf_dense.npy", cdf_dense)


def load_corpus_cdfs(directory: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load corpus-level CDFs from disk. Returns None if not found."""
    bm25_path = directory / "corpus_cdf_bm25.npy"
    dense_path = directory / "corpus_cdf_dense.npy"
    if not bm25_path.exists() or not dense_path.exists():
        return None
    return np.load(bm25_path), np.load(dense_path)


# ---------------------------------------------------------------------------
# QPP Baseline Predictors
# ---------------------------------------------------------------------------


def clarity_score(
    top_k_texts: list[str],
    corpus_term_freqs: Counter,
    corpus_total_terms: int,
) -> float:
    """Clarity Score: KL divergence of top-k document LM vs corpus LM.

    Cronen-Townsend, Zhou, Croft (2002). "Predicting Query Performance."

    KL(P(w|θ_D@k) || P(w|θ_C)) = Σ P(w|θ_D@k) * log2(P(w|θ_D@k) / P(w|θ_C))
    """
    # Build top-k language model
    top_k_freqs: Counter = Counter()
    top_k_total = 0
    for text in top_k_texts:
        tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
        top_k_freqs.update(tokens)
        top_k_total += len(tokens)

    if top_k_total == 0 or corpus_total_terms == 0:
        return 0.0

    # KL divergence (with Laplace smoothing for corpus model)
    kl = 0.0
    for word, count in top_k_freqs.items():
        p_w_topk = count / top_k_total
        # +1 Laplace smoothing to avoid division by zero
        p_w_corpus = (corpus_term_freqs.get(word, 0) + 1) / (
            corpus_total_terms + len(corpus_term_freqs)
        )
        if p_w_topk > 0 and p_w_corpus > 0:
            kl += p_w_topk * math.log2(p_w_topk / p_w_corpus)

    return kl


def nqc(top_k_scores: np.ndarray, corpus_mean_score: float) -> float:
    """Normalized Query Commitment.

    Shtok, Kurland, Carmel, Raiber, Markovits (2012).
    "Using Statistical Decision Theory and Relevance Models for
    Query-Performance Prediction."

    NQC = std(top-k scores) / |s(q, C)|

    The denominator s(q, C) is the retrieval score of a virtual corpus-level
    document. We use the mean retrieval score across all documents as a
    standard practical approximation.
    """
    scores = np.asarray(top_k_scores, dtype=float)
    if len(scores) == 0 or abs(corpus_mean_score) < 1e-10:
        return 0.0
    return float(np.std(scores) / abs(corpus_mean_score))


def wig(top_k_scores: np.ndarray, corpus_mean_score: float, query_len: int) -> float:
    """Weighted Information Gain.

    Zhou and Croft (2007). "Query Performance Prediction in Web Search
    Environments."

    WIG = mean(top-k scores - corpus_mean) / sqrt(|q|)
    """
    scores = np.asarray(top_k_scores, dtype=float)
    if len(scores) == 0 or query_len <= 0:
        return 0.0
    return float(np.mean(scores - corpus_mean_score) / math.sqrt(query_len))


# ---------------------------------------------------------------------------
# Corpus Language Model (for Clarity Score)
# ---------------------------------------------------------------------------


def build_corpus_language_model(
    all_texts: list[str],
) -> tuple[Counter, int]:
    """Build corpus-level language model (unigram term frequencies).

    Returns (term_freqs, total_terms).
    """
    freqs: Counter = Counter()
    total = 0
    for text in all_texts:
        tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
        freqs.update(tokens)
        total += len(tokens)
    return freqs, total


def save_corpus_lm(term_freqs: Counter, total_terms: int, path: Path) -> None:
    """Save corpus language model to disk."""
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"term_freqs": term_freqs, "total_terms": total_terms}, f)


def load_corpus_lm(path: Path) -> tuple[Counter, int] | None:
    """Load corpus language model from disk."""
    import pickle

    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)  # noqa: S301
    return data["term_freqs"], data["total_terms"]


# ---------------------------------------------------------------------------
# Statistical Analysis
# ---------------------------------------------------------------------------


def pearson_correlation(
    predictor_values: list[float], quality_values: list[float]
) -> dict[str, float]:
    """Compute Pearson correlation with p-value.

    Returns dict with 'r', 'p_value', 'n'.
    """
    n = len(predictor_values)
    if n < 3:
        return {"r": 0.0, "p_value": 1.0, "n": n}
    if np.allclose(predictor_values, predictor_values[0]) or np.allclose(
        quality_values, quality_values[0]
    ):
        return {"r": 0.0, "p_value": 1.0, "n": n}
    r, p = stats.pearsonr(predictor_values, quality_values)
    if np.isnan(r) or np.isnan(p):
        return {"r": 0.0, "p_value": 1.0, "n": n}
    return {"r": float(r), "p_value": float(p), "n": n}


def bootstrap_ci(
    values_a: list[float],
    values_b: list[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap 95% confidence interval for the difference in means.

    Used for paired significance testing in H2 ablation comparisons.
    """
    values_a = np.array(values_a)
    values_b = np.array(values_b)
    n = len(values_a)

    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "significant": False}

    diffs = []
    rng = np.random.default_rng(seed=42)
    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        diff = np.mean(values_a[indices]) - np.mean(values_b[indices])
        diffs.append(diff)

    diffs = np.sort(diffs)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    mean_diff = float(np.mean(values_a) - np.mean(values_b))

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": ci_lower > 0 or ci_upper < 0,
    }
