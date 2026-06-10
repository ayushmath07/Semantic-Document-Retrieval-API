"""BM25 sparse retrieval using rank_bm25.

Provides lexical keyword-based retrieval to complement dense FAISS search.
Supports full-corpus scoring for building corpus-level CDFs.
"""

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25Okapi-based sparse retrieval index with disk persistence."""

    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: list[list[str]] = []
        self.doc_metadata: list[dict[str, Any]] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [tok for tok in text.split() if len(tok) > 1]

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add documents to the BM25 index.

        Args:
            documents: List of dicts with keys 'text', 'source', 'chunk'.
        """
        for doc in documents:
            tokens = self.tokenize(doc["text"])
            self.tokenized_corpus.append(tokens)
            self.doc_metadata.append(
                {
                    "source": doc["source"],
                    "chunk": doc["chunk"],
                    "text": doc["text"],
                }
            )
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Retrieve top-k documents by BM25 score.

        Returns list of dicts with 'source', 'chunk', 'text', 'score'.
        """
        if self.bm25 is None:
            return []

        tokens = self.tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    {
                        "source": self.doc_metadata[idx]["source"],
                        "chunk": self.doc_metadata[idx]["chunk"],
                        "text": self.doc_metadata[idx]["text"],
                        "score": float(scores[idx]),
                    }
                )
        return results

    def score_all(self, query: str) -> np.ndarray:
        """Score ALL documents for a query. Used for corpus-level CDF construction."""
        if self.bm25 is None:
            return np.array([])
        tokens = self.tokenize(query)
        return self.bm25.get_scores(tokens)

    def corpus_mean_score(self, query: str) -> float:
        """Mean BM25 score across the entire corpus for a query.

        Approximation of s(q, C) from Shtok et al. (2012) for NQC computation.
        """
        scores = self.score_all(query)
        return float(np.mean(scores)) if len(scores) > 0 else 0.0

    def save(self, path: Path) -> None:
        """Persist tokenized corpus and metadata to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "tokenized_corpus": self.tokenized_corpus,
                    "doc_metadata": self.doc_metadata,
                },
                f,
            )

    def load(self, path: Path) -> None:
        """Load tokenized corpus and rebuild BM25 index."""
        if not path.exists():
            return
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self.tokenized_corpus = data["tokenized_corpus"]
        self.doc_metadata = data["doc_metadata"]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
