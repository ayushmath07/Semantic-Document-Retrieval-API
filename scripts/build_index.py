from pathlib import Path

from app.retriever import DocumentStore


def main() -> None:
    store = DocumentStore()
    docs_dir = Path("data/sample_docs")

    for file_path in docs_dir.glob("*.txt"):
        stats = store.add_file(file_path)
        print(f"Indexed {stats['filename']} ({stats['chunks']} chunks)")


if __name__ == "__main__":
    main()
