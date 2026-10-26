"""Build the Chroma index from the Ultralytics docs."""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.chunking import chunk_file

DOCS_ROOT = Path("data/ultralytics-repo/docs/en")
INDEX_DIR = "index"
COLLECTION = "yolo_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EXCLUDE = {"reference"}  # auto-generated API pages, excluded from the base index


def main():
    files = [f for f in DOCS_ROOT.rglob("*.md") if not EXCLUDE & set(f.relative_to(DOCS_ROOT).parts)]
    chunks = [c for f in files for c in chunk_file(f, DOCS_ROOT)]
    sizes = sorted(len(c.text) for c in chunks)
    print(f"{len(files)} files -> {len(chunks)} chunks | median {sizes[len(sizes) // 2]} chars, max {sizes[-1]}")

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode([c.text for c in chunks], batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    client = chromadb.PersistentClient(path=INDEX_DIR)
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), 500):
        batch = chunks[i : i + 500]
        col.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=embeddings[i : i + 500].tolist(),
            metadatas=[{"path": c.path, "url": c.url, "heading": c.heading} for c in batch],
        )
    print(f"Indexed {col.count()} chunks into '{INDEX_DIR}/'")


if __name__ == "__main__":
    main()