"""Check that every gold URL exists in the index. Usage: python -m eval.check_gold [questions.jsonl]"""

import json
import sys
from pathlib import Path

import chromadb

from src.ingest import COLLECTION, INDEX_DIR

path = sys.argv[1] if len(sys.argv) > 1 else "eval/questions.jsonl"
col = chromadb.PersistentClient(path=INDEX_DIR).get_collection(COLLECTION)
urls = {m["url"] for m in col.get(include=["metadatas"])["metadatas"]}
missing = 0
for line in Path(path).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    q = json.loads(line)
    for g in q["gold"]:
        if g not in urls:
            missing += 1
            print("MISSING", q["id"], g)
print(f"done, {len(urls)} pages in index, {missing} missing gold URLs")
