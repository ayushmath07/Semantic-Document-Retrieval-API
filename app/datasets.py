"""Dataset loaders for benchmark-style retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCIFACT_DIR = Path("data/scifact")


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def scifact_available(data_dir: Path = SCIFACT_DIR) -> bool:
    return (
        (data_dir / "corpus.jsonl").exists()
        and (data_dir / "queries.jsonl").exists()
        and (data_dir / "qrels" / "test.tsv").exists()
    )


def load_scifact_corpus(data_dir: Path = SCIFACT_DIR) -> list[dict[str, str]]:
    """Load SciFact corpus records as document dictionaries."""
    records: list[dict[str, str]] = []
    for item in iter_jsonl(data_dir / "corpus.jsonl"):
        title = item.get("title", "").strip()
        body = item.get("text", "").strip()
        text = f"{title}\n\n{body}".strip()
        if text:
            records.append(
                {
                    "doc_id": str(item["_id"]),
                    "title": title,
                    "text": text,
                }
            )
    return records


def load_scifact_queries(data_dir: Path = SCIFACT_DIR) -> dict[str, str]:
    return {
        str(item["_id"]): item.get("text", "").strip()
        for item in iter_jsonl(data_dir / "queries.jsonl")
        if item.get("text", "").strip()
    }


def load_scifact_qrels(data_dir: Path = SCIFACT_DIR, split: str = "test") -> dict[str, list[str]]:
    qrels_path = data_dir / "qrels" / f"{split}.tsv"
    qrels: dict[str, list[str]] = {}
    with open(qrels_path, encoding="utf-8") as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            query_id, corpus_id, score = parts
            if int(score) <= 0:
                continue
            qrels.setdefault(str(query_id), []).append(str(corpus_id))
    return qrels


def load_scifact_golden_set(
    data_dir: Path = SCIFACT_DIR, split: str = "test"
) -> list[dict[str, Any]]:
    """Return BEIR-style golden rows compatible with scripts/evaluate.py."""
    queries = load_scifact_queries(data_dir)
    qrels = load_scifact_qrels(data_dir, split=split)
    golden_set: list[dict[str, Any]] = []
    for query_id in sorted(qrels, key=lambda value: int(value) if value.isdigit() else value):
        query = queries.get(query_id)
        if not query:
            continue
        golden_set.append(
            {
                "query_id": query_id,
                "query": query,
                "relevant_sources": sorted(set(qrels[query_id])),
                "relevant_keywords": [],
                "query_type": "scifact",
            }
        )
    return golden_set
