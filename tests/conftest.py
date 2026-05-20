import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app.main as main
    import app.retriever as retriever

    upload_dir = tmp_path / "uploads"
    index_dir = tmp_path / "faiss_index"

    monkeypatch.setattr(retriever, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(retriever, "INDEX_DIR", index_dir)
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)

    main.store = retriever.DocumentStore()

    with TestClient(main.app) as test_client:
        yield test_client
