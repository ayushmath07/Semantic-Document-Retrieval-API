import os
from pathlib import Path
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader


BASE_DIR = Path(os.getenv("DATA_DIR", "data"))
UPLOAD_DIR = BASE_DIR / "uploads"
INDEX_DIR = BASE_DIR / "faiss_index"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class DocumentStore:
    def __init__(self) -> None:
        self.vector_store: FAISS | None = None
        self._embeddings: HuggingFaceEmbeddings | None = None
        self.documents: set[str] = set()

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return self._embeddings

    def load(self) -> None:
        if INDEX_DIR.exists():
            self.vector_store = FAISS.load_local(
                str(INDEX_DIR),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

    def save(self) -> None:
        if self.vector_store is not None:
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(INDEX_DIR))

    def add_file(self, file_path: Path) -> dict[str, Any]:
        text = extract_text(file_path)
        if not text.strip():
            raise ValueError("No readable text found in the uploaded file.")

        chunks = chunk_text(text, file_path.name)
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

        self.documents.add(file_path.name)
        self.save()
        return {"filename": file_path.name, "chunks": len(chunks), "characters": len(text)}

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if self.vector_store is None:
            return []

        matches = self.vector_store.similarity_search_with_score(query, k=top_k)
        results = []
        for doc, distance in matches:
            results.append(
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk": doc.metadata.get("chunk", 0),
                    "score": round(1 / (1 + float(distance)), 4),
                    "text": doc.page_content,
                }
            )
        return results

    def answer(self, query: str, top_k: int = 3) -> dict[str, Any]:
        results = self.search(query, top_k)
        if not results:
            return {
                "answer": "I do not have any indexed documents yet. Upload a PDF or text file first.",
                "sources": [],
            }

        best_lines = [result["text"].strip() for result in results[:2]]
        answer = " ".join(best_lines)
        return {
            "answer": answer,
            "sources": results,
        }


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
