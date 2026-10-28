"""REST API for the RAG pipeline."""

from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generate import MODEL, answer
from src.retrieve import Retriever

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = Retriever(rerank=True)  # load embedding model, index and reranker once
    yield
    state.clear()


app = FastAPI(title="yolo-docs-rag", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Source(BaseModel):
    n: int  # matches the [n] citations in the answer
    url: str
    heading: str


class Answer(BaseModel):
    answer: str
    sources: list[Source]
    model: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


@app.post("/ask", response_model=Answer)
def ask(q: Question):
    try:
        out = answer(q.question, state["retriever"])
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"LLM backend unavailable: {e}") from e
    sources = [Source(n=i, url=u, heading=h) for i, (u, h) in enumerate(zip(out["sources"], out["headings"]), 1)]
    return Answer(answer=out["answer"], sources=sources, model=MODEL)
