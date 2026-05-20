from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.retriever import UPLOAD_DIR, DocumentStore


store = DocumentStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    store.load()
    yield


app = FastAPI(
    title="Semantic Document Retrieval API",
    description="A simple FastAPI + LangChain + FAISS project for PDF/text semantic search.",
    version="0.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What does the document say about AI?"])
    top_k: int = Field(default=3, ge=1, le=8)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": "Semantic Document Retrieval API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "index_loaded": store.vector_store is not None,
        "documents": sorted(store.documents),
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
    result = store.answer(request.question, request.top_k)
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
