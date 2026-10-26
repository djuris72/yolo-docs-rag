"""Query the Chroma index."""

import sys

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.ingest import COLLECTION, EMBED_MODEL, INDEX_DIR

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # recommended for bge query side
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
POOL = 30  # candidates passed to the reranker


class Retriever:
    def __init__(self, rerank: bool = False):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.col = chromadb.PersistentClient(path=INDEX_DIR).get_collection(COLLECTION)
        self.reranker = CrossEncoder(RERANK_MODEL) if rerank else None

    def search(self, query: str, k: int = 5, exclude: tuple = ()) -> list[dict]:
        emb = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=[emb], n_results=100 if exclude else max(k, POOL))
        out = [
            {"text": t, "url": m["url"], "heading": m["heading"], "score": 1 - d}
            for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
        if exclude:
            out = [r for r in out if not any(f"/{e}/" in r["url"] for e in exclude)]
        if self.reranker:
            out = out[:POOL]
            scores = self.reranker.predict([(query, r["text"]) for r in out])
            for r, s in zip(out, scores):
                r["score"] = float(s)
            out.sort(key=lambda r: r["score"], reverse=True)
        return out[:k]


if __name__ == "__main__":
    for r in Retriever().search(" ".join(sys.argv[1:])):
        print(f"{r['score']:.3f}  {r['heading']}\n       {r['url']}\n")