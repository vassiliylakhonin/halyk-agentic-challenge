# Document-grounded banking agent

Answers questions about a pack of banking documents. For each question it decides which
document version is operative, pulls the facts that are split across several files,
computes, and returns the answer with page-level references to the text it relied on.

Built as a competition submission for a banking document-agent challenge, and kept
reusable: the competition-specific parts live in one adapter file.

## What it does

1. **Ingest.** Reads PDF, images, XLSX, DOCX, CSV, JSON and plain text from one folder.
   PDFs and images are also passed to the model as bytes, so scans and complex tables do
   not depend on an OCR text layer being present.
2. **Index.** Extracts header facts per document (type, subject, contract number,
   effective date, version label, what it supersedes), then runs one cross-document pass
   that marks which version is operative. This verdict is attached to every answer prompt.
3. **Answer.** Per question: pick the documents, read them with citations enabled, work
   the problem, and state every arithmetic step as a literal expression.
4. **Check the arithmetic.** Every step the model states is re-evaluated in Python. A
   mismatch triggers one repair pass. The interpreter wins arguments about multiplication.
5. **Package.** `submission.json` is rewritten after every answer, so a valid file exists
   on disk from the first result onward.

## Why it is built this way

The scored criteria are correctness, computational accuracy, and evidence. Each maps to a
mechanism rather than to prompt wording:

| Criterion | Mechanism |
|---|---|
| Correct decision | Version resolution runs once, up front, and travels with every prompt. A pipeline that quotes a superseded addendum is confidently wrong. |
| Computational accuracy | `arith.py` recomputes the model's own stated expressions and repairs mismatches. |
| Evidence | Citations are requested natively, so references carry document, page, and the quoted text rather than a reconstruction. |

Citations and structured output cannot be requested in the same API call, so answering is
split: a grounded turn produces prose plus citations, a second turn converts that into the
typed record. That split is also where the arithmetic check fits.

## Worked example

`examples/` holds three documents describing one loan — a base agreement and two
amendments, one of them superseded — plus four questions that a pipeline without
version resolution gets wrong in a predictable way. See [examples/README.md](examples/README.md).

## Setup

```bash
uv venv && uv pip install -e .
cp .env.example .env    # then put your ANTHROPIC_API_KEY in .env
```

## Run

```bash
hac ingest                      # what got parsed, and which files have no text layer
hac index                       # build the document catalogue and version verdicts
hac run                         # answer everything, write submission.json
hac run --limit 5               # smoke test on the first five questions
hac package                     # rebuild submission.json from answers.jsonl, no API calls
hac validate                    # check every question id is present and non-empty
```

Paths and models come from `config.json`; `--docs`, `--questions` and `--out` override
them for one invocation.

`hac run` resumes by default: answers already in `out/answers.jsonl` are not recomputed.
Killing the process and restarting is safe and is the intended recovery path.

## Operating under a time limit

- `concurrency` in `config.json` is the throughput knob. Raise it until rate limits push
  back, then stop.
- `effort_answer` trades tokens for depth: `high` is the default, `medium` roughly halves
  cost, `xhigh` is for the hardest questions.
- `router_enabled: false` sends the whole corpus with every question and caches the
  document prefix. Cheaper and better when the pack is small enough to fit; wasteful when
  it is not.
- `out/run_report.json` lists errors, arithmetic flags, answers with no citations, and
  low-confidence answers. That is the queue to review by hand if time remains.

## Adapting to a different question or submission format

`src/hac/adapters/halyk_agentic.py` owns two functions and nothing else:

- `load_questions(path)` — already accepts a JSON list, an object keyed by id, JSONL, or
  one question per line
- `build_submission(results)` — the object written to `submission.json`

Everything upstream works on `(qid, question)` pairs. Changing the output shape does not
touch the pipeline.

## Limits

- No live retrieval. The agent reads the supplied pack and nothing else. If a rate or a
  day-count basis is not in the documents, the answer says so instead of supplying a
  market convention.
- Not legal, compliance, financial, or investment advice. Output is for review by a person
  who can check it against the source documents.
- The arithmetic check verifies that stated expressions evaluate to stated values. It does
  not verify that the expression was the right one to write.
- No scoring or evaluation harness against a ground truth set. Correctness on the open
  dataset has not been measured yet.

## Tests

```bash
pytest tests -q
```

The offline suite covers expression evaluation and its refusal to execute code, the
arithmetic mismatch path, every question-file shape the loader accepts, submission
ordering, and ingestion. It needs no API key.
