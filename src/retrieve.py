"""Query the Chroma index."""

import sys

import chromadb
from sentence_transformers import SentenceTransformer

from src.ingest import COLLECTION, EMBED_MODEL, INDEX_DIR

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # recommended for bge query side


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.col = chromadb.PersistentClient(path=INDEX_DIR).get_collection(COLLECTION)

    def search(self, query: str, k: int = 5) -> list[dict]:
        emb = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=[emb], n_results=k)
        return [
            {"text": t, "url": m["url"], "heading": m["heading"], "score": 1 - d}
            for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]


if __name__ == "__main__":
    for r in Retriever().search(" ".join(sys.argv[1:])):
        print(f"{r['score']:.3f}  {r['heading']}\n       {r['url']}\n")