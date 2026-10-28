"""API tests: the model and the retriever are replaced, so no Ollama or index is needed."""

import requests
from fastapi.testclient import TestClient

import src.api as api


def test_health():
    r = TestClient(api.app).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_question_is_validated():
    assert TestClient(api.app).post("/ask", json={"question": "a"}).status_code == 422


def test_sources_are_numbered_like_the_citations(monkeypatch):
    monkeypatch.setitem(api.state, "retriever", object())
    monkeypatch.setattr(
        api, "answer", lambda q, r: {"answer": "ok [1]", "sources": ["u1", "u2"], "headings": ["h1", "h2"]}
    )
    r = TestClient(api.app).post("/ask", json={"question": "How do I train a model?"})
    assert r.status_code == 200
    assert r.json()["sources"][1] == {"n": 2, "url": "u2", "heading": "h2"}


def test_llm_backend_down_returns_503(monkeypatch):
    def boom(q, r):
        raise requests.ConnectionError("no ollama")

    monkeypatch.setitem(api.state, "retriever", object())
    monkeypatch.setattr(api, "answer", boom)
    r = TestClient(api.app).post("/ask", json={"question": "How do I train a model?"})
    assert r.status_code == 503
