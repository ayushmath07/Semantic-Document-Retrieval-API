def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_and_query_text_document(client):
    from app.main import store

    def fake_add_file(file_path):
        store.documents.add(file_path.name)
        return {"filename": file_path.name, "chunks": 1, "characters": 84}

    def fake_answer(question, top_k):
        return {
            "answer": "[notes.txt] Semantic search finds similar text chunks.",
            "sources": [
                {
                    "source": "notes.txt",
                    "chunk": 1,
                    "score": 0.91,
                    "text": "Semantic search finds similar text chunks.",
                }
            ],
            "retrieval_latency_ms": 1.23,
        }

    store.add_file = fake_add_file
    store.answer = fake_answer

    upload = client.post(
        "/upload",
        files={
            "file": (
                "notes.txt",
                "Python is used with FastAPI to build APIs. Semantic search finds similar text chunks.",
                "text/plain",
            )
        },
    )

    assert upload.status_code == 200
    assert upload.json()["chunks"] >= 1

    query = client.post("/query", json={"question": "What is semantic search?", "top_k": 2})

    assert query.status_code == 200
    data = query.json()
    assert "Semantic search" in data["answer"]
    assert data["sources"][0]["source"] == "notes.txt"
    assert "retrieval_latency_ms" in data


def test_upload_rejects_unsupported_file(client):
    response = client.post(
        "/upload",
        files={"file": ("data.csv", "col1,col2\na,b", "text/csv")},
    )
    assert response.status_code == 400
    assert "supported" in response.json()["detail"].lower()


def test_query_empty_index(client):
    response = client.post("/query", json={"question": "What is AI?", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == []


def test_query_validation(client):
    # Question too short (min_length=3)
    response = client.post("/query", json={"question": "hi", "top_k": 1})
    assert response.status_code == 422

    # top_k out of range
    response = client.post("/query", json={"question": "What is AI?", "top_k": 99})
    assert response.status_code == 422


def test_list_documents_empty(client):
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["documents"] == []
    assert data["index_ready"] is False
