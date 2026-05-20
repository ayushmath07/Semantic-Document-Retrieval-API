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
            "answer": "Semantic search finds similar text chunks.",
            "sources": [
                {
                    "source": "notes.txt",
                    "chunk": 1,
                    "score": 0.91,
                    "text": "Semantic search finds similar text chunks.",
                }
            ],
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
