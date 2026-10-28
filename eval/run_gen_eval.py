"""Generation evaluation: refusals, citations, keyword correctness, latency."""

import json
import re
import time
from pathlib import Path

from src.generate import MODEL, answer
from src.retrieve import Retriever

REFUSAL = "could not find this in the ultralytics documentation"


def keywords_ok(low: str, kw: list[str]) -> bool:
    """Every keyword must appear as a whole word; 'a|b' accepts either alternative."""
    return all(any(re.search(rf"\b{re.escape(a)}\b", low) for a in k.lower().split("|")) for k in kw)


def main():
    qs = [json.loads(l) for l in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ret = Retriever(rerank=True)
    rows = []
    for q in qs:
        t = time.time()
        out = answer(q["question"], ret)
        text = out["answer"]
        low = text.lower()
        kw = q.get("must_include")
        refused = REFUSAL in low and len(low) < 150
        rows.append(
            {
                "id": q["id"],
                "answerable": bool(q["gold"]),
                "refused": refused,
                "empty": len(re.sub(r"\[\d\]|\s", "", text)) < 20 and not refused,
                "cited": bool(re.search(r"\[\d\]", text)),
                "correct": keywords_ok(low, kw) if kw else None,
                "source_hit": any(g in out["sources"] for g in q["gold"]) if q["gold"] else None,
                "seconds": round(time.time() - t, 1),
                "answer": text,
                "headings": out["headings"],
            }
        )
        print(q["id"], "refused" if refused else "answered", f"{rows[-1]['seconds']}s")

    Path(f"eval/gen_{MODEL.replace(':', '_')}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    kw_rows = [r for r in ans if r["correct"] is not None]
    answered = [r for r in ans if not r["refused"]]
    print(f"\nmodel={MODEL}")
    print(f"answerable n={len(ans)}  false refusals={sum(r['refused'] for r in ans) / len(ans):.2f}")
    print(f"  source in context={sum(bool(r['source_hit']) for r in ans) / len(ans):.2f}")
    print(f"  cited [n]={sum(r['cited'] for r in answered) / max(1, len(answered)):.2f}")
    print(f"  citation-only answers={sum(r['empty'] for r in rows)}")
    print(f"  keyword correct={sum(bool(r['correct']) for r in kw_rows) / len(kw_rows):.2f} (n={len(kw_rows)})")
    print(f"unanswerable n={len(una)}  correct refusals={sum(r['refused'] for r in una) / len(una):.2f}")
    print(f"avg seconds/question={sum(r['seconds'] for r in rows) / len(rows):.1f}")


if __name__ == "__main__":
    main()