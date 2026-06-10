"""Cross-encoder reranking using sentence-transformers.

Provides high-precision re-scoring of (query, document) pairs using a
cross-encoder model. This is the final stage of the retrieval pipeline,
applied after fusion to re-sort candidates by cross-encoder relevance.

Model: cross-encoder/ms-marco-MiniLM-L6-v2 (22M parameters, fast inference).
"""

from typing import Any

import numpy as np


class CrossEncoderReranker:
    """Lazy-loaded cross-encoder for reranking retrieval results."""

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._model = None

    @property
    def model(self):
        """Lazy-load the cross-encoder model on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.MODEL_NAME)
        return self._model

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Re-score and re-sort candidates using cross-encoder.

        Args:
            query: The search query.
            candidates: List of result dicts with at least a 'text' field.
            top_k: If set, return only the top-k re-ranked results.

        Returns:
            Re-sorted candidates with 'cross_encoder_score' added to each.
        """
        if not candidates:
            return []

        # Build (query, document) pairs
        pairs = [(query, c["text"]) for c in candidates]

        # Score all pairs
        scores = self.model.predict(pairs)
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()

        # Attach scores and sort descending
        for candidate, score in zip(candidates, scores):
            candidate["cross_encoder_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked
