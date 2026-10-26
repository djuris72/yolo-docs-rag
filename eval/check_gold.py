"""Check that every gold URL exists in the index."""

import json
from pathlib import Path

import chromadb

from src.ingest import COLLECTION, INDEX_DIR

col = chromadb.PersistentClient(path=INDEX_DIR).get_collection(COLLECTION)
urls = {m["url"] for m in col.get(include=["metadatas"])["metadatas"]}
for line in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines():
    q = json.loads(line)
    for g in q["gold"]:
        if g not in urls:
            print("MISSING", q["id"], g)
print("done,", len(urls), "pages in index")