"""Index all sample documents in data/sample_docs/.

Supports .txt, .md, and .pdf files.

Usage:
    python scripts/build_index.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retriever import DocumentStore  # noqa: E402


SUPPORTED = {".txt", ".md", ".pdf"}


def main() -> None:
    store = DocumentStore()
    docs_dir = Path("data/sample_docs")

    if not docs_dir.exists():
        print(f"ERROR: {docs_dir} not found.")
        sys.exit(1)

    files = sorted(f for f in docs_dir.iterdir() if f.suffix.lower() in SUPPORTED)
    print(f"Found {len(files)} documents to index.\n")

    total_chunks = 0
    for file_path in files:
        stats = store.add_file(file_path)
        total_chunks += stats["chunks"]
        print(f"  * {stats['filename']:40s}  {stats['chunks']:3d} chunks  {stats['characters']:6d} chars")

    print(f"\nDone. Indexed {len(files)} documents, {total_chunks} total chunks.")


if __name__ == "__main__":
    main()
