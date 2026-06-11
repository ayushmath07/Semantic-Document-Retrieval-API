from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.retriever import RETRIEVAL_MODES, UPLOAD_DIR, DocumentStore

store = DocumentStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    store.load()
    yield


app = FastAPI(
    title="Semantic Document Retrieval API",
    description=(
        "Hybrid retrieval API combining BM25 sparse search, FAISS dense search, "
        "entropy-weighted adaptive fusion, and cross-encoder reranking. "
        "Supports 7 retrieval modes for ablation comparison."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What does the document say about AI?"])
    top_k: int = Field(default=3, ge=1, le=20)
    mode: str = Field(
        default="hybrid_calibrated_rerank",
        description="Retrieval mode. See /modes for available options.",
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": "Semantic Document Retrieval API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "modes": "/modes",
    }


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "index_loaded": store.vector_store is not None,
        "bm25_loaded": store.bm25_index.bm25 is not None,
        "corpus_cdfs_loaded": store.corpus_cdfs is not None,
        "documents": sorted(store.documents),
    }


@app.get("/modes")
async def list_modes() -> dict[str, object]:
    """List all available retrieval modes with descriptions."""
    return {
        "modes": RETRIEVAL_MODES,
        "default": "hybrid_calibrated_rerank",
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Upload must include a filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and MD files are supported.")

    safe_name = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_name
    content = await file.read()
    file_path.write_bytes(content)

    try:
        stats = store.add_file(file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Document indexed successfully.",
        **stats,
    }


@app.post("/query")
async def query_documents(request: QueryRequest) -> dict[str, object]:
    if request.mode not in RETRIEVAL_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode '{request.mode}'. Available: {list(RETRIEVAL_MODES)}",
        )

    result = store.answer(request.question, request.top_k, request.mode)
    return {
        "question": request.question,
        **result,
    }


@app.get("/documents")
async def list_documents() -> dict[str, object]:
    return {
        "documents": sorted(store.documents),
        "index_ready": store.vector_store is not None,
    }
