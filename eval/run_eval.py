"""Retrieval evaluation: page-level hit@k and MRR."""

import argparse
import json
from pathlib import Path

from src.retrieve import Retriever

K = 5
FETCH = 20  # chunks to fetch before collapsing to unique pages


def unique_pages(results):
    seen, pages = set(), []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            pages.append(r["url"])
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()
    print("exclude:", args.exclude, args.rerank or "none")

    qs = [json.loads(l) for l in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    qs = [q for q in qs if q["gold"]]
    ret = Retriever(rerank=args.rerank)
    hit1 = hitk = mrr = 0.0
    misses = []
    for q in qs:
        pages = unique_pages(ret.search(q["question"], k=FETCH, exclude=tuple(args.exclude)))
        rank = next((i + 1 for i, p in enumerate(pages) if p in q["gold"]), None)
        hit1 += rank == 1
        hitk += rank is not None and rank <= K
        mrr += 1 / rank if rank else 0
        if rank is None or rank > K:
            misses.append((q["id"], q["question"], rank))
    n = len(qs)
    print(f"n={n}  hit@1={hit1 / n:.2f}  hit@{K}={hitk / n:.2f}  MRR={mrr / n:.3f}")
    for m in misses:
        print("MISS", m)


if __name__ == "__main__":
    main()