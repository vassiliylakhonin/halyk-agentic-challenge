# Worked example

Three documents describing one loan, two of them amendments, one of those
superseded. Four questions that a pipeline without version resolution answers
wrongly in a predictable way.

Run it:

```bash
hac --docs examples/docs ingest
hac --docs examples/docs index
hac --docs examples/docs --questions examples/questions.json --out examples/submission.json run
```

The first two commands need no API key beyond `index`; `ingest` is offline.

## What each question tests

| # | Tests |
|---|---|
| 1 | Version resolution. Addendum No. 2 governs from 1 February 2025, so the answer is 16.0 per cent — not the 14.5 in the base agreement, and not the 15.25 in the superseded Addendum No. 1. |
| 2 | Computation with a basis taken from the clause rather than from convention. 250,000,000 × 0.16 × 90 / 365. Every step is recomputed in Python before the answer is accepted. |
| 3 | Partial supersession. Addendum No. 2 restates the early-repayment clause: 30 days is no longer sufficient, and a 0.5 per cent fee now applies. |
| 4 | Reading a supersession chain backwards, and reporting a date the documents state rather than one inferred from the addendum's own signing date. |
