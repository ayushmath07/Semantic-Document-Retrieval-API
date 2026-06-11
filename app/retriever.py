"""Pipeline orchestrator for hybrid retrieval with cross-retriever score calibration.

Coordinates BM25 sparse retrieval, FAISS dense retrieval, score calibration,
fusion strategies, and cross-encoder reranking across 7 retrieval modes.

Modes:
    dense                   - FAISS only (original behavior)
    sparse                  - BM25 only
    rrf                     - Both → Reciprocal Rank Fusion
    hybrid_fixed            - Both → min-max calibration → α=0.5 linear fusion
    hybrid_calibrated       - Both → CDF calibration → entropy-weighted fusion
    hybrid_fixed_rerank     - hybrid_fixed + cross-encoder reranking
    hybrid_calibrated_rerank - hybrid_calibrated + cross-encoder reranking
"""

import contextlib
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader

from app.calibration import load_corpus_cdfs, load_corpus_lm
from app.fusion import entropy_fuse, linear_fuse, rrf_fuse
from app.reranker import CrossEncoderReranker
from app.sparse_retriever import BM25Index

BASE_DIR = Path(os.getenv("DATA_DIR", "data"))
UPLOAD_DIR = BASE_DIR / "uploads"
INDEX_DIR = BASE_DIR / "faiss_index"
BM25_INDEX_PATH = INDEX_DIR / "bm25_index.pkl"
DOC_LOOKUP_PATH = INDEX_DIR / "doc_lookup.pkl"
CORPUS_LM_PATH = INDEX_DIR / "corpus_lm.pkl"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Minimum relevance score for dense-only mode. Benchmark evaluation should not
# threshold top-k retrieval, so the default is disabled.
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.0"))

RETRIEVAL_MODES = {
    "dense": "FAISS dense retrieval only (original behavior)",
    "sparse": "BM25 sparse retrieval only",
    "rrf": "BM25 + FAISS → Reciprocal Rank Fusion (k=60)",
    "hybrid_fixed": "BM25 + FAISS → min-max calibration → α=0.5 linear fusion",
    "hybrid_calibrated": "BM25 + FAISS → CDF calibration → entropy-weighted fusion",
    "hybrid_fixed_rerank": "hybrid_fixed + cross-encoder reranking",
    "hybrid_calibrated_rerank": "hybrid_calibrated + cross-encoder reranking (full pipeline)",
}

# Number of candidates to retrieve from each retriever before fusion
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))


class DocumentStore:
    def __init__(self) -> None:
        self.vector_store: FAISS | None = None
        self._embeddings: HuggingFaceEmbeddings | None = None
        self.bm25_index = BM25Index()
        self._reranker: CrossEncoderReranker | None = None
        self.documents: set[str] = set()
        self.corpus_cdfs: tuple[np.ndarray, np.ndarray] | None = None
        self.corpus_lm: tuple[Any, int] | None = None
        self.doc_lookup: dict[str, str] = {}

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return self._embeddings

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def load(self) -> None:
        """Load all indices, CDFs, and lookup tables from disk."""
        if INDEX_DIR.exists():
            with contextlib.suppress(Exception):
                self.vector_store = FAISS.load_local(
                    str(INDEX_DIR),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )

        self.bm25_index.load(BM25_INDEX_PATH)
        self.documents = {str(m["source"]) for m in self.bm25_index.doc_metadata}

        # Load corpus-level CDFs
        self.corpus_cdfs = load_corpus_cdfs(INDEX_DIR)

        # Load corpus language model (for Clarity Score)
        self.corpus_lm = load_corpus_lm(CORPUS_LM_PATH)

        # Load document text lookup
        if DOC_LOOKUP_PATH.exists():
            with open(DOC_LOOKUP_PATH, "rb") as f:
                self.doc_lookup = pickle.load(f)  # noqa: S301

    def save(self) -> None:
        if self.vector_store is not None:
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(INDEX_DIR))

    def add_file(self, file_path: Path) -> dict[str, Any]:
        text = extract_text(file_path)
        if not text.strip():
            raise ValueError("No readable text found in the uploaded file.")

        chunks = chunk_text(text, file_path.name)
        stats = self.add_documents(chunks)
        return {"filename": file_path.name, **stats, "characters": len(text)}

    def add_documents(self, documents: list[Document]) -> dict[str, Any]:
        """Bulk-add already constructed LangChain documents."""
        if not documents:
            return {"chunks": 0}

        # Build FAISS index
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)

        # Build BM25 index
        bm25_docs = [
            {
                "text": doc.page_content,
                "source": doc.metadata["source"],
                "chunk": doc.metadata["chunk"],
            }
            for doc in documents
        ]
        self.bm25_index.add_documents(bm25_docs)
        self.bm25_index.save(BM25_INDEX_PATH)

        # Update document text lookup
        for doc in documents:
            key = f"{doc.metadata['source']}:{doc.metadata['chunk']}"
            self.doc_lookup[key] = doc.page_content
            self.documents.add(str(doc.metadata["source"]))
        DOC_LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DOC_LOOKUP_PATH, "wb") as f:
            pickle.dump(self.doc_lookup, f)

        self.save()
        return {"chunks": len(documents)}

    def _search_dense(self, query: str, top_k: int = CANDIDATE_K) -> list[dict[str, Any]]:
        """Dense retrieval via FAISS."""
        if self.vector_store is None:
            return []

        matches = self.vector_store.similarity_search_with_score(query, k=top_k)
        results = []
        for doc, distance in matches:
            score = 1 / (1 + float(distance))  # L2 to similarity
            results.append(
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk": doc.metadata.get("chunk", 0),
                    "text": doc.page_content,
                    "score": round(score, 6),
                }
            )
        return results

    def _search_sparse(self, query: str, top_k: int = CANDIDATE_K) -> list[dict[str, Any]]:
        """Sparse retrieval via BM25."""
        return self.bm25_index.search(query, top_k=top_k)

    def _dense_score_all(self, query: str) -> np.ndarray:
        """Score ALL documents with dense retriever. For CDF construction.

        Uses IndexFlatL2 exhaustive search — embeddings are already stored
        in FAISS, no re-encoding needed.
        """
        if self.vector_store is None:
            return np.array([])

        # Get all docs with scores (large k = all docs)
        n_docs = self.vector_store.index.ntotal
        if n_docs == 0:
            return np.array([])

        matches = self.vector_store.similarity_search_with_score(query, k=n_docs)
        # Convert L2 distances to similarity scores
        return np.array([1 / (1 + float(dist)) for _, dist in matches])

    def search(
        self, query: str, top_k: int = 3, mode: str = "hybrid_calibrated_rerank"
    ) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
        """Search using the specified retrieval mode.

        Returns (results, total_latency_ms, telemetry).
        """
        t_start = time.perf_counter()
        telemetry: dict[str, Any] = {"mode": mode}

        if mode == "dense":
            results = self._search_dense(query, top_k=top_k)
            # Apply legacy score filter for backward compatibility
            results = [r for r in results if r["score"] >= MIN_RELEVANCE_SCORE]

        elif mode == "sparse":
            results = self._search_sparse(query, top_k=top_k)

        elif mode == "rrf":
            sparse = self._search_sparse(query, top_k=CANDIDATE_K)
            dense = self._search_dense(query, top_k=CANDIDATE_K)
            results, meta = rrf_fuse(sparse, dense, top_k=top_k)
            telemetry.update(meta)

        elif mode == "hybrid_fixed":
            sparse = self._search_sparse(query, top_k=CANDIDATE_K)
            dense = self._search_dense(query, top_k=CANDIDATE_K)
            results, meta = linear_fuse(sparse, dense, alpha=0.5, calibration="minmax", top_k=top_k)
            telemetry.update(meta)

        elif mode == "hybrid_calibrated":
            sparse = self._search_sparse(query, top_k=CANDIDATE_K)
            dense = self._search_dense(query, top_k=CANDIDATE_K)
            results, meta = entropy_fuse(
                sparse,
                dense,
                calibration="cdf",
                corpus_cdfs=self.corpus_cdfs,
                top_k=top_k,
            )
            telemetry.update(meta)

        elif mode == "hybrid_fixed_rerank":
            sparse = self._search_sparse(query, top_k=CANDIDATE_K)
            dense = self._search_dense(query, top_k=CANDIDATE_K)
            fused, meta = linear_fuse(
                sparse,
                dense,
                alpha=0.5,
                calibration="minmax",
                top_k=CANDIDATE_K,
            )
            telemetry.update(meta)
            results = self.reranker.rerank(query, fused, top_k=top_k)
            telemetry["reranked"] = True

        elif mode == "hybrid_calibrated_rerank":
            sparse = self._search_sparse(query, top_k=CANDIDATE_K)
            dense = self._search_dense(query, top_k=CANDIDATE_K)
            fused, meta = entropy_fuse(
                sparse,
                dense,
                calibration="cdf",
                corpus_cdfs=self.corpus_cdfs,
                top_k=CANDIDATE_K,
            )
            telemetry.update(meta)
            results = self.reranker.rerank(query, fused, top_k=top_k)
            telemetry["reranked"] = True

        else:
            raise ValueError(f"Unknown retrieval mode: {mode}. Available: {list(RETRIEVAL_MODES)}")

        latency_ms = (time.perf_counter() - t_start) * 1000
        telemetry["latency_ms"] = round(latency_ms, 2)

        return results, round(latency_ms, 2), telemetry

    def answer(
        self, query: str, top_k: int = 3, mode: str = "hybrid_calibrated_rerank"
    ) -> dict[str, Any]:
        results, latency_ms, telemetry = self.search(query, top_k, mode)

        if not results:
            return {
                "answer": "I do not have any indexed documents yet. Upload a PDF or text file first.",
                "sources": [],
                "retrieval_latency_ms": latency_ms,
                "telemetry": telemetry,
            }

        # Build answer from top chunks
        answer_parts: list[str] = []
        for r in results[:3]:
            snippet = r["text"].strip()
            if len(snippet) > 300:
                snippet = snippet[:300].rsplit(" ", 1)[0] + "..."
            answer_parts.append(f"[{r['source']}] {snippet}")

        return {
            "answer": "\n\n".join(answer_parts),
            "sources": results,
            "retrieval_latency_ms": latency_ms,
            "telemetry": telemetry,
        }


# ---------------------------------------------------------------------------
# Text extraction and chunking (unchanged from original)
# ---------------------------------------------------------------------------


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError("Only PDF, TXT, and MD files are supported.")


def chunk_text(text: str, source: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    pieces = splitter.split_text(text)
    return [
        Document(page_content=piece, metadata={"source": source, "chunk": index + 1})
        for index, piece in enumerate(pieces)
    ]
