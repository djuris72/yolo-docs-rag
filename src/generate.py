"""Answer questions with retrieved context via a local Ollama model."""

import os
import sys

import requests

from src.retrieve import Retriever

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
K = 4
MAX_CHUNK_CHARS = 1500
EXCLUDE: tuple = ("yolov5", "platform")

SYSTEM = (
    "You answer questions about Ultralytics YOLO using ONLY the numbered context passages. "
    "Write a short answer (1-4 sentences or a brief code example) and put citations like [1] after the "
    "statements they support. Never reply with only a citation. "
    "If the context does not contain the answer, reply with exactly this sentence and nothing else: "
    "'I could not find this in the Ultralytics documentation.' "
    "Never add that sentence to an answer. Do not use outside knowledge."
)


def build_prompt(question: str, passages: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i}] ({p['url']})\n{p['text'][:MAX_CHUNK_CHARS]}" for i, p in enumerate(passages, 1))
    return f"Context:\n{ctx}\n\nQuestion: {question}"


def answer(question: str, retriever: Retriever) -> dict:
    passages = retriever.search(question, k=K, exclude=EXCLUDE)
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(question, passages)},
            ],
            "options": {"temperature": 0, "num_ctx": 4096},
        },
        timeout=600,
    )
    resp.raise_for_status()
    return {
        "answer": resp.json()["message"]["content"],
        "sources": [p["url"] for p in passages],
        "headings": [p["heading"] for p in passages],
    }


if __name__ == "__main__":
    out = answer(" ".join(sys.argv[1:]), Retriever(rerank=True))
    print(out["answer"], "\n\nSources:")
    for i, u in enumerate(out["sources"], 1):
        print(f"[{i}] {u}")