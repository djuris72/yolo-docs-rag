# Results

All numbers come from `eval/run_eval.py` and `eval/run_gen_eval.py` on a Ryzen 5 2600, 16 GB RAM and a GTX 1660 Super (6 GB). Embedding and reranking ran on CPU; the LLM ran through Ollama.

## How to read these numbers

- Development set: 36 questions, of which 31 are answerable (page-level gold URLs). With n = 31, one question is about 3 percentage points. A difference of one question is noise; only large jumps are reliable.
- This set was also used to make design decisions, so the numbers are optimistic. The held-out section at the bottom is the honest check.
- Metrics are computed over unique pages. hit@k: a gold page is among the first k distinct pages. MRR: mean reciprocal rank of the first gold page.

## Index history

| Index | Chunks | Median chars | Max chars | What changed |
|---|---|---|---|---|
| v0 | 5,750 | 629 | 19,565 | Heading-aware chunking; oversized blocks (large tables and code) kept whole |
| v1 | 5,864 | 644 | 3,124 | Oversized blocks split on line boundaries |
| v2 | 5,771 | 652 | 3,096 | MkDocs macro includes expanded; table pieces repeat the header and padding is collapsed |

`bge-small` reads at most 512 tokens (about 2,000 characters), so chunks of 19,565 characters in v0 were silently truncated. Index building takes about 13 minutes on CPU (12:45 for v2).

## Retrieval (31 answerable questions)

Cells are hit@1 / hit@5 / MRR.

| Experiment | Index v1 (no argument tables) | Index v2 (macros expanded) |
|---|---|---|
| Baseline (bge-small, no filter, no reranker) | 0.35 / 0.87 / 0.578 | 0.45 / 0.97 / 0.671 |
| + filter (exclude `yolov5/` and `platform/`) | 0.52 / 0.94 / 0.690 | not run |
| + cross-encoder reranker | 0.58 / 0.94 / 0.735 | 0.55 / 0.97 / 0.730 |
| + filter + reranker | 0.71 / 0.94 / 0.802 | **0.71 / 0.97 / 0.826** |

What this shows:

- The big effects are real: 0.35 → 0.71 hit@1 from the filter plus the reranker, and 0.87 → 0.97 hit@5 from fixing the index.
- The index fix improved recall (the right page is now found) but not rank at the top. The remaining weakness is ordering, not finding.
- The reranker recovers most of what the manual filter does while still allowing questions about YOLOv5 and the Platform. The filter was chosen by looking at this same question set, so its gain is optimistic.
- 0.58 → 0.55 for the reranker alone is one question and is not interpreted.

Remaining retrieval misses (index v2, filter + reranker): only `q28` ("export a model, then check the accuracy of the exported version"). It needs two different pages, and the reranker prefers a chunk that answers only one half.

## Generation (31 answerable + 5 unanswerable)

| Setup | False refusals | Correct refusals | Cites `[n]` | Keyword correct (n = 13) | s / question |
|---|---|---|---|---|---|
| qwen2.5:3b, prompt v1, index v1 | 0.03 | 1.00 | 0.13 | 0.92 | 9.3 |
| qwen2.5:3b, prompt v2, index v1 | 0.00 | 0.80 | 0.13 | 0.92 | 9.7 |
| qwen2.5:7b, prompt v1, index v1 | 0.06 | 1.00 | 0.83 | 0.62 | 17.2 |
| qwen2.5:7b, prompt v2, index v1 | 0.00 | 1.00 | 0.77 | 0.92 | 15.7 |
| qwen2.5:7b, prompt v2, index v2 | 0.00 | 1.00 | 0.77 | **1.00** | 16.4 |

Source in context (the right page is among the 4 passages the model saw): 0.94 on index v1, 0.97 on index v2. The 3B model was not re-run on index v2.

Caveats:

- The prompt v1 rows were scored with a looser checker (keywords as substrings, the refusal sentence accepted anywhere in the answer). They are not directly comparable with the v2 rows. I did not verify whether the answer filter (`EXCLUDE`) was set in the v1 runs, so the v1 and v2 difference may mix two changes.
- The keyword check covers 13 questions, so one question moves it by about 8 points. The 0.92 → 1.00 step is one question.
- 3B and 7B differ clearly in two things that rest on about 30 answers: the 7B model follows the citation instruction (77–83%) and the 3B model almost never does (13%), and 7B is about 1.7× slower.
- Latency is per question end to end: retrieval, reranking and generation.

Model speed measured with a short prompt (`ollama run --verbose`): `qwen2.5:3b` about 80–90 tokens/s fully on GPU, `qwen2.5:7b` about 22 tokens/s with part of the model on CPU.

## What the failures looked like

Reading the failed answers mattered more than the aggregate numbers.

| Question | What happened | Cause |
|---|---|---|
| q19 ("certainty") | 7B refused; 3B invented `confidence_threshold` | No argument table in the index (macro bug). Fixed in index v2 |
| q17, q18 | 7B answered with only `[1]` | Prompt v1 made the model treat the citation rule too literally. Prompt v2 forbids citation-only answers; 0 such answers afterwards |
| q01 | Correct answer followed by the refusal sentence, counted as a refusal | Checker flaw. Fixed by requiring a short answer to count as a refusal |
| q23 (Apple format) | 7B answered with the newer `coreai` / `.aimodel` format and was marked wrong | Not a hallucination: the docs have `integrations/coreai.md`. The gold set and the keyword were too narrow; fixed |
| q29 (CPU formats) | 7B named only ONNX | Real, partial miss (the docs also name OpenVINO) |
| q27, q34 (3B) | Refused an answerable question; answered an unanswerable one | Small-model behaviour; single instances, not interpreted further |

## Held-out set

10 questions (`eval/heldout.jsonl`): 8 answerable questions on three pages that no development question used (End-to-End Detection, Thread-Safe Inference, Triton Inference Server) and 2 questions the docs do not cover. Each answerable question was checked against the live pages. The set is run **once**, with the final configuration (index v2, filter + reranker, `qwen2.5:7b`, prompt v2), and not tuned afterwards.

Commands:

```bat
python -m eval.check_gold eval/heldout.jsonl
python -m eval.run_eval --questions eval/heldout.jsonl --exclude yolov5 platform --rerank
set LLM_MODEL=qwen2.5:7b
python -m eval.run_gen_eval --questions eval/heldout.jsonl
```

| Metric | Value |
|---|---|
| Retrieval hit@1 (n = 8) | 1.00 |
| Retrieval hit@5 | 1.00 |
| Retrieval MRR | 1.000 |
| False refusals | 0.00 |
| Correct refusals (n = 2) | 2/2 |
| Keyword correct (n = 8) | 8/8 |
| Cites `[n]` | 0.88 (7 of 8) |
| Source in context | 1.00 |
| Seconds / question | 16.1 |

All 8 answerable questions were retrieved at rank 1 and answered with the right keyword, and both unanswerable questions were refused. I read this as "no sign of overfitting to the development set", not as a measure of overall quality: the held-out pages have distinctive vocabulary, there are no argument-table or multi-page questions, and with n = 8 a perfect score is consistent with a true rate anywhere above roughly 63%.

## Changes to the evaluation itself

- Keyword check changed from substring to whole word, with `a|b` alternatives (so `conf` no longer passes on "confidence").
- A short answer containing the refusal sentence counts as a refusal; a longer answer containing it does not.
- Added a "citation-only answers" counter.
- Gold set: q23 now also accepts the `coreai` page and keyword; q24 also accepts the `litert` page.
