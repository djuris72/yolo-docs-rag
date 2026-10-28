# yolo-docs-rag

A retrieval-augmented question-answering system over the official [Ultralytics YOLO documentation](https://docs.ultralytics.com), built to be **measured**, not just demoed. The repo has the full pipeline (ingestion, retrieval, reranking, local-LLM generation), a REST API, a Docker image, unit tests and an evaluation harness that quantifies each design decision.

I use YOLO (it powers my [ANPR thesis project](https://github.com/djuris72/ANPR)), so I can judge answer correctness myself. That is why this documentation was the testbed.

> Full numbers, per-question notes and how to reproduce them are in [`results.md`](results.md).

## Headline results

| What | Result |
|---|---|
| Retrieval, hit@1 (31 questions) | 0.35 baseline → **0.71** with a filter and a cross-encoder reranker |
| Retrieval, hit@5 | 0.87 → **0.97** after fixing a chunking bug (see below) |
| Generation, `qwen2.5:7b`, prompt v2 | 0 false refusals, 5/5 correct refusals, keyword check 13/13, about 16 s per question on a GTX 1660 Super |
| Held-out set (10 questions, run once) | see [`results.md`](results.md) (filled in after the single run) |

The question set was used while developing the system, so the development numbers are optimistic. The held-out set exists to check that.

## Pipeline

```
docs/en/*.md ──► expand macros ──► heading-aware chunking ──► bge-small embeddings ──► Chroma (cosine)
                                                                                           │
question ──► query embedding ──► top-30 candidates ──► cross-encoder rerank ──► top-4 passages
                                                                                           │
                                          local LLM (Ollama, Qwen2.5) with citation + refusal prompt
```

| Stage | Choice | Notes |
|---|---|---|
| Source | `docs/en` of the `ultralytics/ultralytics` repo (Markdown) | 304 files after excluding the auto-generated `reference/` pages |
| Macro expansion | `{% include "macros/..." %}` is replaced by the included file | the argument tables (`imgsz`, `conf`, ...) live in these macros |
| Chunking | Split on `#`–`###` headings, heading path prepended, code blocks never split, large tables cut into small pieces that **repeat the header row** | 5,771 chunks, median 652 chars, max 3,096 |
| Embeddings | `BAAI/bge-small-en-v1.5` with the recommended query prefix | normalized, cosine similarity |
| Vector store | Chroma, persistent and local | no server needed |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top 30 | optional (`rerank=True`) |
| Generator | Ollama, `qwen2.5:3b` or `qwen2.5:7b`, temperature 0 | cites passages as `[n]`; must refuse when the context has no answer |

Every chunk keeps its public docs URL, so each answer comes with sources.

## A bug the evaluation found

The docs build their argument tables from MkDocs macros, for example `{% include "macros/predict-args.md" %}`. My first chunker treated that line as plain text, so **no argument table was in the index**. The symptom was one question ("keep detections above a certain certainty"): the 7B model correctly refused because nothing relevant was in its context, and the 3B model invented a parameter called `confidence_threshold` (the real one is `conf`). Page-level retrieval metrics did not show it, because the page `usage/cfg` was still found. Only reading the retrieved chunks did.

Fix: expand the includes before chunking, and cut the big tables into small pieces that repeat the header and have their column padding collapsed. Effect: baseline hit@5 went from 0.87 to 0.97, and the 7B keyword check from 12/13 to 13/13.

## Setup (Windows)

```bat
git clone https://github.com/djuris72/yolo-docs-rag.git
cd yolo-docs-rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
```

Get the documentation. It is not redistributed in this repo; it belongs to the AGPL-3.0 Ultralytics project.

```bat
git clone --depth 1 --filter=blob:none --sparse https://github.com/ultralytics/ultralytics data/ultralytics-repo
cd data/ultralytics-repo
git sparse-checkout set docs
cd ../..
```

Install [Ollama](https://ollama.com) and pull a model:

```bat
ollama pull qwen2.5:7b
```

Notes for Windows:

- `torch` may need the Microsoft Visual C++ Redistributable.
- If imports fail because `torchvision` or `torchaudio` do not match `torch`, uninstall them; this project does not use them.

## Usage

```bat
:: 1. Build the index (about 13 min on CPU)
python -m src.ingest

:: 2. Query the retriever
python -m src.retrieve How do I export a model to ONNX

:: 3. Ask a question end to end
set LLM_MODEL=qwen2.5:7b
python -m src.generate How do I resume an interrupted training run?
```

### REST API

```bat
uvicorn src.api:app --port 8000
```

```bat
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"How do I stop training early?\"}"
```

The response contains the answer, the model name and numbered sources. The `n` of each source matches the `[n]` citations in the answer text. If Ollama is not running, `/ask` returns HTTP 503 and `/health` still answers.

### Docker

The image holds the code only. The index and the model cache are mounted, so the image stays small and rebuilding the index does not need a rebuild of the image.

```bat
docker build -t yolo-docs-rag .
docker run --rm -p 8000:8000 ^
  -v "%cd%\index:/app/index" ^
  -v "D:\hf_cache:/app/.hf_cache" ^
  -e OLLAMA_URL=http://host.docker.internal:11434/api/chat ^
  -e LLM_MODEL=qwen2.5:7b ^
  yolo-docs-rag
```

Notes:

- The container reaches Ollama on the host through `host.docker.internal`. If the connection is refused, start Ollama with `OLLAMA_HOST=0.0.0.0`.
- The container uses CPU-only torch, so embedding and reranking run on CPU (the LLM still runs on the host GPU through Ollama).
- Pin `chromadb` and `sentence-transformers` in `requirements.txt` to the versions you built the index with. An index written by one Chroma version is not guaranteed to open in another.

### Tests

```bat
python -m pytest -q
```

11 tests: the chunker (heading paths, front matter, code blocks, table header repetition, macro expansion, padding) and the API (validation, source numbering, 503 on a missing LLM backend). The API tests replace the model and the retriever, so they need neither Ollama nor an index.

## Evaluation

```bat
python -m eval.check_gold
python -m eval.run_eval --exclude yolov5 platform --rerank
python -m eval.run_gen_eval
```

- `eval/questions.jsonl`: 36 development questions (31 answerable with page-level gold URLs, 5 that the docs do not cover). 13 have a `must_include` keyword that I checked against the live documentation.
- `eval/heldout.jsonl`: 10 questions on pages that no development question used (8 answerable, 2 unanswerable). They are run once, with the final configuration, and not used for tuning:

```bat
python -m eval.check_gold eval/heldout.jsonl
python -m eval.run_eval --questions eval/heldout.jsonl --exclude yolov5 platform --rerank
set LLM_MODEL=qwen2.5:7b
python -m eval.run_gen_eval --questions eval/heldout.jsonl
```

Retrieval metrics are computed over unique pages: hit@1, hit@5 and MRR. Generation metrics: false refusals, correct refusals, how often the answer cites `[n]`, a keyword check, and latency.

## Repository layout

```
src/
  chunking.py     macro expansion and heading-aware chunker
  ingest.py       build the Chroma index
  retrieve.py     vector search, optional filter and reranker
  generate.py     answer with a local Ollama model
  api.py          FastAPI service
eval/
  questions.jsonl development questions
  heldout.jsonl   held-out questions (run once)
  check_gold.py   verify that gold URLs exist in the index
  run_eval.py     retrieval metrics
  run_gen_eval.py generation metrics
tests/            pytest suite
Dockerfile
requirements.txt
results.md        all measurements and per-question notes
```

## Limitations

- Small evaluation sets: good for large effects, not for fine ranking differences. One question is about 3 points of hit@k.
- The development set was used to make design decisions (for example the `yolov5/` and `platform/` filter), so those numbers are optimistic. The held-out set covers only three pages, so it checks generalization to new pages but not breadth.
- `must_include` is a keyword check. It can pass a wrong answer that happens to contain the keyword, and it exists for only 13 of the 31 answerable questions.
- Gold labels and keywords reflect the docs at the time of writing; the documentation changes often (YOLO26 renamed or deprecated some arguments, and a new Apple format appeared).
- The exclusion filter removes YOLOv5 and Platform content from answers on purpose; without it, hit@1 is lower.
- Duplicate chunks: the same macro table is included by several pages, so the context may contain near-identical passages. I have not measured how often.

## Roadmap

- [x] Ingestion, chunking, vector search
- [x] Retrieval evaluation and experiments (filter, reranker)
- [x] Local-LLM generation with refusal prompt, prompt v1 vs v2
- [x] Chunking bug found and fixed through evaluation (macros)
- [x] FastAPI endpoint, Docker image, unit tests
- [ ] Held-out run (single run, results in `results.md`)
- [ ] Hybrid search (BM25 + vectors) and a stronger reranker
- [ ] Deduplicate near-identical passages before generation

## License

Code in this repository: add a `LICENSE` file (MIT is a common choice). The Ultralytics documentation is licensed by Ultralytics under AGPL-3.0 and is not included here.
