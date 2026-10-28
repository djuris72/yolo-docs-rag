"""Print answers that failed the keyword check or were refused."""

import json
import sys

path = sys.argv[1]
for line in open(path, encoding="utf-8"):
    r = json.loads(line)
    if r["answerable"] and (r["correct"] is False or r["refused"]):
        print(f"=== {r['id']} refused={r['refused']} correct={r['correct']}")
        print(r["answer"][:600])
        print()
        