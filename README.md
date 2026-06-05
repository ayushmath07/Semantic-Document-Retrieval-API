# Semantic Document Retrieval API

An AI-powered document search API built with Python, FastAPI, LangChain, FAISS, and HuggingFace embeddings.

Upload PDF, TXT, or Markdown documents and query them with natural language. The system extracts text, splits it into semantically meaningful chunks, generates vector embeddings, and performs similarity search to return the most relevant passages.

## Architecture

<img width="2822" height="1472" alt="Gemini_Generated_Image_z2nb2ez2nb2ez2nb" src="https://github.com/user-attachments/assets/30ea86b0-ad9a-408b-8aa4-616744535e33" />

### Pipeline Details

1. **Text Extraction**: PDFs are parsed page-by-page with `pypdf`. TXT and MD files are read directly.
2. **Chunking**: `RecursiveCharacterTextSplitter` with `chunk_size=800` and `chunk_overlap=120`. The splitter tries to break on paragraph boundaries first, then sentences, then words, preserving semantic coherence.
3. **Embedding**: `all-MiniLM-L6-v2` from sentence-transformers (384-dimensional vectors). Configurable via the `EMBEDDING_MODEL` environment variable.
4. **Indexing**: FAISS `IndexFlatL2` for exact nearest-neighbor search. Index persists to disk and loads on startup.
5. **Retrieval**: `similarity_search_with_score` returns top-k results. L2 distance is normalized to a 0–1 relevance score via `1/(1+distance)`. Results below `MIN_RELEVANCE_SCORE` (default 0.30) are filtered out.
6. **Response**: Each result includes source document name, chunk number, relevance score, and the chunk text.

## Tech Stack

| Part | Tool |
| --- | --- |
| API | FastAPI (async, OpenAPI/Swagger docs) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (384-dim) |
| Vector Store | FAISS `IndexFlatL2` with disk persistence |
| PDF Parsing | pypdf |
| Container | Docker + Docker Compose |
| CI | GitHub Actions |
| Testing | pytest with FastAPI TestClient |

## Project Structure

```text
app/
  main.py            # FastAPI routes and request validation
  retriever.py       # Text extraction, chunking, embedding, FAISS search, scoring
data/
  sample_docs/       # 16 technical documents (~80 KB corpus)
  uploads/           # User-uploaded files
  faiss_index/       # Persisted FAISS index
eval/
  golden_set.json    # 40 labeled query-document pairs for evaluation
  results.json       # Latest evaluation results (auto-generated)
scripts/
  build_index.py     # Index all sample documents
  evaluate.py        # Run retrieval evaluation (Precision@k, MRR, latency)
tests/
  conftest.py        # Pytest fixtures with monkeypatching
  test_api.py        # 6 API tests (health, upload, query, validation, edge cases)
.github/
  workflows/ci.yml   # GitHub Actions CI pipeline
Dockerfile
docker-compose.yml
requirements.txt
```

## Run Locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open the API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

The first upload/query may take a moment while HuggingFace downloads the embedding model (~90 MB).

## Run With Docker

```bash
docker compose up --build
```

API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## API Endpoints

### Health Check

```http
GET /health
```

### Upload Document

```http
POST /upload
```

Upload a `.pdf`, `.txt`, or `.md` file using form-data with key `file`.

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@data/sample_docs/machine_learning.txt"
```

### Query Documents

```http
POST /query
```

Body:

```json
{
  "question": "What is supervised learning?",
  "top_k": 3
}
```

Response:

```json
{
  "question": "What is supervised learning?",
  "answer": "[machine_learning.txt] Supervised learning is the most common form of machine learning...",
  "sources": [
    {
      "source": "machine_learning.txt",
      "chunk": 2,
      "score": 0.7621,
      "text": "Supervised learning is the most common form..."
    }
  ],
  "retrieval_latency_ms": 3.45
}
```

### List Documents

```http
GET /documents
```

## Evaluation

The project includes a retrieval evaluation pipeline with 40 labeled queries.

### Build Index and Run Evaluation

```bash
python scripts/build_index.py
python scripts/evaluate.py
```

### Metrics

Evaluation on 17 technical documents (~80 KB, 126 chunks, 40 labeled queries):

| Metric | Value |
| --- | --- |
| Precision@3 | 0.5667 |
| Recall@3 | 1.0000 |
| MRR | 0.9167 |
| Latency (p50) | 15.85 ms |
| Latency (p95) | 18.23 ms |

Run `python scripts/evaluate.py` to generate actual numbers. Results are saved to `eval/results.json`.

## Run Tests

```bash
pytest tests/ -v
```

6 tests covering: health endpoint, upload + query flow, file type validation, empty index handling, input validation, and document listing.

## Configuration

| Environment Variable | Default | Description |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `DATA_DIR` | `data` | Base directory for uploads and index |
| `MIN_RELEVANCE_SCORE` | `0.30` | Minimum similarity score (0–1) to include in results |

## Design Decisions

- **No LLM generation**: The system returns retrieved chunks directly rather than passing them through an LLM. This keeps the project dependency-light (no API keys required) and focuses on retrieval quality, which is the foundation of any RAG system.
- **Score threshold**: Results below 0.30 relevance are filtered to avoid returning irrelevant content for off-topic queries.
- **Flat index**: `IndexFlatL2` provides exact nearest-neighbor search. For the current corpus size (~130 chunks), brute-force search is sub-5ms. For larger corpora, switching to `IndexIVFFlat` or `IndexHNSWFlat` would give sub-linear search time.
- **Configurable model**: The embedding model is set via environment variable so different models can be benchmarked without code changes.
