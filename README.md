# Semantic Document Retrieval API

A beginner-friendly AI document search API built with Python, FastAPI, LangChain, FAISS, HuggingFace embeddings, and Docker.

This is not trying to be a production RAG platform. It is a clean project that shows the core pipeline:

1. Upload a PDF, TXT, or Markdown file.
2. Extract the text.
3. Split the text into chunks.
4. Convert chunks into HuggingFace embeddings.
5. Store/search embeddings with FAISS.
6. Return a simple answer with source chunks.

## Tech Stack

| Part | Tool |
| --- | --- |
| API | FastAPI |
| Chunking | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector DB | FAISS |
| PDF parsing | pypdf |
| Container | Docker |

## Project Structure

```text
app/
  main.py        # FastAPI routes
  retriever.py   # PDF extraction, chunking, embeddings, FAISS search
data/
  uploads/       # Uploaded files
  faiss_index/   # Saved FAISS index
scripts/
  build_index.py # Optional script to index sample text files
tests/
  test_api.py
Dockerfile
docker-compose.yml
requirements.txt
```

## Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

The first upload/query may take a little time because HuggingFace downloads the embedding model.

## Run With Docker

```bash
docker compose up --build
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Health

```http
GET /health
```

### Upload Document

```http
POST /upload
```

Upload a `.pdf`, `.txt`, or `.md` file using form-data with the key `file`.

Example with curl:

```bash
curl -X POST "http://127.0.0.1:8000/upload" ^
  -F "file=@data/sample_docs/artificial_intelligence.txt"
```

### Ask a Question

```http
POST /query
```

Body:

```json
{
  "question": "What is artificial intelligence?",
  "top_k": 3
}
```

Example response:

```json
{
  "question": "What is artificial intelligence?",
  "answer": "Artificial intelligence is a field of computer science...",
  "sources": [
    {
      "source": "artificial_intelligence.txt",
      "chunk": 1,
      "score": 0.82,
      "text": "Artificial intelligence is..."
    }
  ]
}
```

### List Documents

```http
GET /documents
```

## Optional: Build Index From Sample Docs

```bash
python scripts/build_index.py
```

## Run Tests

```bash
pytest
```

## Resume Bullet

Semantic Document Retrieval API | Python, FastAPI, LangChain, FAISS, HuggingFace, Docker

- Built an AI-powered document Q&A API that extracts text from PDFs, chunks documents, creates HuggingFace embeddings, and stores them in a FAISS vector index.
- Added FastAPI endpoints for uploading documents, querying semantically similar chunks, returning source snippets, and running the app locally with Docker.
