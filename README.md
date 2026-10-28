yolo-docs-rag

A retrieval-augmented question-answering system over the official Ultralytics YOLO documentation, built to be measured, not just demoed. The repo contains the full pipeline (ingestion, retrieval, reranking, local-LLM generation) and a small evaluation harness that quantifies each design decision.

Status: retrieval and its evaluation are done; generation works end to end and is being evaluated; REST API, Docker and tests are in progress (see Roadmap).

Why this project

Most RAG tutorials stop at "it returns an answer". The goal here is the part that usually gets skipped: a hand-written evaluation set, one-variable-at-a-time experiments, and an honest error analysis. The YOLO docs are a good testbed because I use the tool (it powers my ANPR thesis project), so I can judge answer correctness myself.

Pipeline
docs/en/*.md ──► heading-aware chunking ──► bge-small embeddings ──► Chroma (cosine)
                                                                         │
question ──► query embedding ──► top-30 candidates ──► cross-encoder rerank ──► top-4 passages
                                                                         │
                                       local LLM (Ollama, Qwen2.5) with citation + refusal prompt
Stage	Choice	Notes
Source	docs/en of the ultralytics/ultralytics repo (Markdown)	304 files after excluding the auto-generated reference/ pages
Chunking	Split on #–### headings; heading path is prepended to each chunk; fenced code blocks are never split	5,864 chunks, median 644 chars, max 3,124
Embeddings	BAAI/bge-small-en-v1.5 (+ recommended query prefix)	normalized, cosine similarity
Vector store	Chroma (persistent, local)	no server needed
Reranker	cross-encoder/ms-marco-MiniLM-L-6-v2 over the top 30	optional (rerank=True)
Generator	Ollama, qwen2.5:3b / qwen2.5:7b, temperature 0	cites passages as [n]; must refuse when the context has no answer

Every chunk keeps its public docs URL, so answers are returned with sources.

Evaluation

eval/questions.jsonl has 36 hand-written questions:

31 answerable questions with page-level gold labels (URLs). Page-level labels stay valid when chunking changes.
A mix of easy questions (close to a page title), paraphrased ones (e.g. "certainty" instead of conf), and multi-page questions.
5 unanswerable questions (audio, speech, text-to-image, Nintendo Switch, LLM training) used to test refusals.
For a subset, a must_include keyword that a correct answer has to contain (e.g. patience for early stopping). I checked each of these against the live documentation. It is a coarse, deterministic check; it does not prove an answer is fully correct.

Retrieval metrics are computed over unique pages: hit@1, hit@5 and MRR (n = 31).

Retrieval results
Experiment	hit@1	hit@5	MRR
Baseline (bge-small, no filter)	0.35	0.87	0.578
+ exclude yolov5/ and platform/ pages	0.52	0.94	0.690
+ cross-encoder reranker (no filter)	0.58	0.94	0.735
+ filter + reranker	0.71	0.94	0.802

How to read this:

The big jump (0.35 → 0.71 hit@1) is reliable. Small differences are not: with n = 31, one question is about 3 percentage points.
The reranker recovers most of what the manual filter does while still allowing questions about YOLOv5 and the Platform. The filter was chosen by looking at this same question set, so its gain is optimistic; a held-out set is planned (see Roadmap).
Remaining misses are consistent across all configurations:
q19 ("certainty" vs. the conf argument) is a vocabulary mismatch, and the argument sits inside a large settings table.
q28 ("export and then check accuracy of the exported model") needs two different pages and the reranker favors a chunk that answers only one half.
Generation results

Informal observations so far (single questions, not yet a measured result):

qwen2.5:3b answered the resume-training question correctly and refused the audio question with the exact refusal sentence.
It did not write [n] citations even though the prompt asks for them, so the sources list is currently attached by the code rather than by the model.

The generation evaluation (eval/run_gen_eval.py) reports false refusals, correct refusals, citation rate, keyword correctness and latency per model. Numbers will be added here when the run completes.

Model	False refusals	Correct refusals	Cites [n]	Keyword correct	s / question
qwen2.5:3b	TBD	TBD	TBD	TBD	TBD
qwen2.5:7b	TBD	TBD	TBD	TBD	TBD

Local speed on a GTX 1660 Super (6 GB): about 80–90 tokens/s for the 3B model (fully on GPU) and about 22 tokens/s for the 7B model (partly offloaded to CPU).

Setup (Windows)
bat
git clone https://github.com/djuris72/yolo-docs-rag.git
cd yolo-docs-rag
python -m venv venv
venv\Scripts\activate
pip install chromadb sentence-transformers requests fastapi uvicorn pytest

Get the documentation (not redistributed in this repo; it is part of the AGPL-3.0 Ultralytics project):

bat
git clone --depth 1 --filter=blob:none --sparse https://github.com/ultralytics/ultralytics data/ultralytics-repo
cd data/ultralytics-repo
git sparse-checkout set docs
cd ../..

Install Ollama and pull a model:

bat
ollama pull qwen2.5:3b

Notes for Windows:

On some machines torch needs the Microsoft Visual C++ Redistributable.
If import errors mention torchvision or torchaudio not matching torch, uninstall them; this project does not use them.
Usage
bat
:: 1. Build the index (about 12 min on CPU; much faster with a CUDA build of torch)
python -m src.ingest

:: 2. Query the retriever
python -m src.retrieve How do I export a model to ONNX

:: 3. Ask a question end to end
set LLM_MODEL=qwen2.5:3b
python -m src.generate How do I resume an interrupted training run?

:: 4. Evaluation
python -m eval.check_gold
python -m eval.run_eval --exclude yolov5 platform --rerank
python -m eval.run_gen_eval
Repository layout
src/
  chunking.py     heading-aware Markdown chunker
  ingest.py       build the Chroma index
  retrieve.py     vector search, optional filter and reranker
  generate.py     answer with a local Ollama model
eval/
  questions.jsonl hand-written eval set (gold URLs, must_include)
  check_gold.py   verifies gold URLs exist in the index
  run_eval.py     retrieval metrics (hit@k, MRR)
  run_gen_eval.py generation metrics (refusals, citations, keywords, latency)
Limitations
Small eval set (31 answerable questions): good for large effects, not for fine ranking differences.
must_include is keyword-based; it can miss a wrong answer that happens to contain the keyword.
The documentation changes often (YOLO26 renamed or deprecated some arguments, e.g. half/int8 became quantize). Gold URLs and keywords reflect the docs at the time of writing.
Chunking is tuned by inspection on one page structure; very large tables (such as the settings tables) are split by line and lose their header row.
The exclusion filter is a pragmatic trade-off: it removes YOLOv5 and Platform content from answers.
Roadmap
 Ingestion, chunking, vector search
 Retrieval evaluation and experiments (filter, reranker)
 Local-LLM generation with refusal prompt
 Generation evaluation for 3B vs 7B
 Held-out question set (10 questions) run once with the final configuration
 Hybrid search (BM25 + vectors) and error analysis of q19 / q28
 FastAPI endpoint and Docker image
 Unit tests for the chunker and retriever
License

Code in this repository: MIT (to be added). The Ultralytics documentation is licensed by Ultralytics under AGPL-3.0 and is not included here.