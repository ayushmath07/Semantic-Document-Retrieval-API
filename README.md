# Calibrated Entropy-Weighted Hybrid Retrieval

A research-oriented hybrid retrieval system for testing whether score calibration makes retrieval uncertainty measurable and useful. The system combines BM25 sparse retrieval, FAISS dense retrieval, corpus-level CDF score calibration, entropy-weighted fusion, and optional cross-encoder reranking behind a FastAPI service.

The current benchmark is **BEIR SciFact**: 5,183 PubMed abstracts, 300 test claims, and document-level qrels.

## Research Question

Does calibrating BM25 and dense retrieval scores to a common probability space via corpus-level CDF normalization make entropy a more reliable predictor of per-query retrieval quality, and does that translate to better fusion?

## Hypotheses

**H1: Distributional signal.** Corpus-level CDF calibration should produce a stronger negative Pearson correlation between score-distribution entropy and per-query retrieval quality than raw, min-max, or z-score normalization.

**H2: Downstream retrieval.** Entropy-weighted fusion over CDF-calibrated scores should outperform Reciprocal Rank Fusion and fixed-alpha linear fusion on NDCG@10, P@3, P@5, and MRR.

## Method

SciFact documents are indexed as chunks, but evaluation is document-level: retrieved chunks are deduplicated by `source` document ID before metrics are computed against BEIR qrels.

Corpus-level CDFs are built offline:

```text
50 SciFact test claims + 50 pseudo queries from corpus chunks
        |
        v
score every query against all 17,243 chunks
        |
        v
sort 1,724,300 BM25 scores and 1,724,300 dense scores
        |
        v
corpus_cdf_bm25.npy and corpus_cdf_dense.npy
```

At query time, CDF calibration maps raw scores to corpus percentiles:

```text
calibrated_score = CDF_corpus(raw_score)
```

Entropy is computed over calibrated score mass:

```text
H = -sum_i p_i log2(p_i)
```

Adaptive fusion uses:

```text
alpha = H_dense / (H_dense + H_sparse + epsilon)
fused = alpha * sparse_score + (1 - alpha) * dense_score
```

If dense retrieval is high-entropy, alpha increases and the system trusts BM25 more. If sparse retrieval is high-entropy, alpha decreases and the system trusts dense retrieval more.

## Important Fixes

The first toy-corpus run exposed two evaluation bugs:

- Entropy was subtracting the minimum score before normalization. That made raw, min-max, and z-score entropy collapse under affine transforms. Entropy now treats nonnegative calibrated scores as probability mass directly and uses softmax only for negative-valued z-scores.
- Metrics were chunk-level. BEIR qrels are document-level, so evaluation now deduplicates retrieved chunks by source document ID.

Dense retrieval thresholding is also disabled by default (`MIN_RELEVANCE_SCORE=0.0`) because benchmark top-k retrieval should not filter results before metric computation.

## Retrieval Modes

| Mode | Description |
| --- | --- |
| `dense` | FAISS dense retrieval only |
| `sparse` | BM25 retrieval only |
| `rrf` | Reciprocal Rank Fusion over BM25 and FAISS |
| `hybrid_fixed` | BM25 + FAISS with min-max calibration and fixed alpha=0.5 |
| `hybrid_calibrated` | BM25 + FAISS with CDF calibration and entropy-weighted alpha |
| `hybrid_fixed_rerank` | Fixed hybrid followed by cross-encoder reranking |
| `hybrid_calibrated_rerank` | CDF entropy hybrid followed by cross-encoder reranking |

## SciFact Results

Generated with:

```bash
python scripts/build_index.py --dataset scifact
python scripts/evaluate.py
```

Full output is saved to `eval/results.json`.

### H1: Entropy Correlations

Pearson correlations below use entropy as the predictor and per-query retrieval quality as the target. A stronger negative value means higher entropy better predicts lower quality.

| Retriever | Calibration | r vs NDCG@10 | p-value | r vs MRR | p-value |
| --- | --- | ---: | ---: | ---: | ---: |
| Sparse | raw | 0.0033 | 0.9550 | 0.0057 | 0.9220 |
| Sparse | min-max | 0.0033 | 0.9550 | 0.0057 | 0.9220 |
| Sparse | z-score | -0.3759 | <0.000001 | -0.3855 | <0.000001 |
| Sparse | CDF | -0.0085 | 0.8834 | -0.0031 | 0.9575 |
| Dense | raw | 0.0865 | 0.1348 | 0.0636 | 0.2718 |
| Dense | min-max | 0.0053 | 0.9273 | -0.0164 | 0.7775 |
| Dense | z-score | -0.4065 | <0.000001 | -0.3970 | <0.000001 |
| Dense | CDF | -0.1151 | 0.0463 | -0.1100 | 0.0571 |

**H1 is not supported as originally stated.** CDF entropy is not the strongest quality predictor. Z-score entropy is the strongest negative predictor for both sparse and dense retrieval on SciFact.

The sparse raw and min-max rows are identical because BM25 all-document score vectors usually have minimum zero; min-max then only rescales the nonnegative vector, and probability normalization is scale-invariant. The z-score row now differs, confirming the entropy bug was fixed.

### H2: Retrieval Ablation

NDCG@10 is the primary metric. Standard deviation is per-query NDCG@10 variance.

| Mode | NDCG@10 | NDCG std | MRR | P@3 | P@5 | R@3 | R@5 | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | 0.6715 | 0.3885 | 0.6329 | 0.2466 | 0.1647 | 0.6855 | 0.7495 | 33.43 ms |
| `sparse` | 0.6151 | 0.4065 | 0.5804 | 0.2233 | 0.1520 | 0.6311 | 0.7089 | 129.53 ms |
| `rrf` | 0.7028 | 0.3737 | 0.6713 | 0.2555 | 0.1680 | 0.7180 | 0.7744 | 249.31 ms |
| `hybrid_fixed` | 0.6829 | 0.3866 | 0.6527 | 0.2522 | 0.1647 | 0.7080 | 0.7594 | 231.60 ms |
| `hybrid_calibrated` | 0.6981 | 0.3802 | 0.6672 | 0.2489 | 0.1660 | 0.6980 | 0.7661 | 240.71 ms |
| `hybrid_fixed_rerank` | 0.7006 | 0.3773 | 0.6653 | 0.2578 | 0.1700 | 0.7161 | 0.7714 | 2432.39 ms |
| `hybrid_calibrated_rerank` | 0.7071 | 0.3703 | 0.6719 | 0.2622 | 0.1720 | 0.7319 | 0.7838 | 2116.34 ms |

### Paired Bootstrap Tests

Each comparison uses 1,000 paired bootstrap resamples over the 300 SciFact test queries.

| Comparison | Metric | Mean diff | 95% CI | p-value | Significant |
| --- | --- | ---: | ---: | ---: | --- |
| `hybrid_calibrated` vs `rrf` | NDCG@10 | -0.0047 | [-0.0170, +0.0072] | 0.4260 | no |
| `hybrid_calibrated` vs `rrf` | P@3 | -0.0067 | [-0.0122, -0.0022] | 0.0020 | yes, worse |
| `hybrid_calibrated` vs `hybrid_fixed` | NDCG@10 | +0.0152 | [+0.0034, +0.0271] | 0.0120 | yes |
| `hybrid_calibrated` vs `hybrid_fixed` | MRR | +0.0145 | [+0.0020, +0.0267] | 0.0280 | yes |
| `hybrid_calibrated_rerank` vs `hybrid_fixed_rerank` | NDCG@10 | +0.0065 | [-0.0057, +0.0195] | 0.3140 | no |

**H2 is partially supported.** CDF entropy fusion significantly improves over fixed-alpha fusion on NDCG@10 and MRR. It does not beat RRF on NDCG@10, and it is significantly worse than RRF on P@3. With reranking, calibrated fusion has the highest absolute NDCG@10 but the improvement over fixed reranking is not statistically significant.

## Architecture

```text
app/
  datasets.py           SciFact corpus/query/qrels loaders
  sparse_retriever.py   BM25 index, search, and full-corpus scoring
  calibration.py        CDF transforms, entropy, QPP predictors, statistics
  fusion.py             RRF, fixed linear fusion, entropy-weighted fusion
  reranker.py           SentenceTransformers cross-encoder reranking
  retriever.py          Seven-mode retrieval orchestrator
  main.py               FastAPI application

scripts/
  build_index.py        Builds FAISS, BM25, CDFs, corpus LM, and doc lookup
  evaluate.py           Runs H1/H2 evaluation and bootstrap significance tests

eval/
  golden_set.json       Toy fallback labels
  results.json          Latest generated evaluation report
```

Runtime artifacts are stored under `data/faiss_index/`:

| Artifact | Purpose |
| --- | --- |
| `index.faiss`, `index.pkl` | Dense FAISS vector store |
| `bm25_index.pkl` | Tokenized BM25 corpus and metadata |
| `corpus_cdf_bm25.npy` | Corpus-level BM25 score CDF |
| `corpus_cdf_dense.npy` | Corpus-level dense score CDF |
| `corpus_lm.pkl` | Corpus unigram language model for Clarity Score |
| `doc_lookup.pkl` | `(source, chunk)` text lookup |
| `index_metadata.json` | Dataset and artifact build metadata |

## Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python scripts/build_index.py --dataset scifact
uvicorn app.main:app --reload
```

Open the API docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The first run may download SentenceTransformers models. Dense retrieval uses `sentence-transformers/all-MiniLM-L6-v2`; reranking uses `cross-encoder/ms-marco-MiniLM-L6-v2`.

## API

### List Modes

```http
GET /modes
```

### Query

```http
POST /query
```

```json
{
  "question": "0-dimensional biomaterials show inductive properties.",
  "top_k": 3,
  "mode": "hybrid_calibrated_rerank"
}
```

The response includes retrieved sources, latency, and mode-specific telemetry such as entropy values and adaptive alpha.

### Upload

```http
POST /upload
```

Upload `.pdf`, `.txt`, or `.md` files with multipart form-data key `file`.

## Run Evaluation

```bash
python scripts/build_index.py --dataset scifact
python scripts/evaluate.py
```

The evaluator writes:

- aggregate metrics for all seven modes
- NDCG@10 variance and per-query metrics
- H1 Pearson correlations with p-values
- entropy debug rows for the first three queries
- H2 paired bootstrap confidence intervals and p-values
- per-query rankings, relevance flags, and telemetry

## Run Tests

```bash
pytest tests/ -v
```

Current test coverage verifies health checks, upload validation, empty-index behavior, mode listing, mode validation, query telemetry, calibration entropy behavior, and document listing.

## Docker

```bash
docker compose up --build
```

API docs are available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Interpretation

The original toy CS corpus was useful for plumbing, but it was too easy and too small to validate the research claim. SciFact changes the story: calibration is not a universal win, RRF remains a strong baseline, and the clearest uncertainty signal comes from z-score entropy rather than CDF entropy. That is a stronger project outcome than a polished toy result because the system now produces falsifiable, benchmarked evidence.
