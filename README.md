# Covenant compliance from a document pack

Reads a pack of banking documents and a transaction ledger, and decides for every
financial covenant of every borrower whether it is met, what the constrained figure
actually is, and which transaction settles the matter.

Built for a covenant-testing challenge. The competition-specific shapes live in one
adapter; the rest is reusable.

## The split that the design turns on

A model reads. Python computes. Nothing else.

The model is asked for three things, each returning JSON against a schema:

| Job | In | Out |
|---|---|---|
| `read_clause` | one covenant clause | a metric specification |
| `read_supplement` | a borrower's non-contract documents | a list of ledger adjustments |
| `read_page` | a page image | its text |

A specification says what to compute, never a result:

```json
{"numerator": [{"cat": "capex"}], "denominator": [{"ebitda": {}}],
 "direction": "max", "threshold": 9.0}
```

Half of every cell's score decays with relative error and reaches zero at five per
cent. An interpreter does not misplace a decimal, so every total, ratio and verdict is
computed in Python from an agreed description of the metric.

## Results, and the gap between them

| | Open pack, 12 borrowers | Hidden pack, 27 borrowers |
|---|---|---|
| score | 0.9389 | **0.6139** |
| placing | — | 83rd of 160 |

The left column is the pack this code could be measured against while it was being
written. The right column is the one that counts, and the 32-point fall between them is
the honest headline of this repository. The winning entry scored 0.9636 on the hidden
pack, which is roughly what the left column promised and this code did not deliver.

Read [why the two columns differ](#why-the-columns-differ) before taking the left one as
evidence of anything.

Detail on the open pack, from a clean folder, one command, nothing hand-written for
those borrowers: 34 of 36 statuses correct, 7 of 9 evidence transactions, no empty
cells, nine minutes at three readings per clause. Filling every cell with the commonest
status for its clause number, without opening a document, scores 0.375.

## What the pack punishes, and what answers it

The dataset is built around traps. Each one is a mechanism here, not a prompt.

**A superseded edition of the agreement sits beside the current one.** Every borrower has
both. Editions are detected by the marker printed on their first page, and clauses are
read from the operative one.

**Counterparty names are decoys.** An antenna mast lease is paid to a company with
"Payroll Group" in its name. Categories come from the payment description alone; the
counterparty is never consulted for them. The agreements say the same from the other
side — related-party status follows the dossier, "not the payment narrative".

**Related-party payments look like operating costs.** A management advisory retainer paid
to a dossier-identified related party is on its own accounting line. Counting it as an
operating cost puts every ratio built on operating costs out by its amount, and through
EBITDA that reaches half of clause 6.1. Capital expenditure is the exception: an asset bought
from a related party is still capital expenditure.

**A holding sits just under the threshold.** Thresholds differ per borrower, from 20 to 40 per
cent, and are read from the dossier rather than assumed. Entity names are matched on a
normalised form, so "Aktau Holdings LLP" and "Aktau Holdings L.L.P." are one party while
"Aktau Terminal Properties LLP" is not.

**The auditor considers an adjustment and rejects it.** Applying it is the trap. Rejected
adjustments are recorded as ignored, not applied.

**A draft interim schedule carries the final report's reference number.** It is superseded
by the final agreed-upon-procedures report and must not be applied.

**An amount never reached the ledger export.** It is disclosed in a treasury memo, not by
the auditor. A missing amount is a defect in the data, applied whoever reports it; a
change of accounting line is a judgement, where the auditor's conclusion governs.

**A dossier exists only as a scan**, with the account number printed inside the image.
Pages below a text threshold are rendered and transcribed, and the fields derived from a
document's text are rebuilt afterwards. Skip that rebuild and the document stays
unlinked, and its borrower silently loses its related parties.

**Interest on a finance sublease reads as a lease payment.** It is a financing cost.

**One-off items below a materiality floor are not added back to EBITDA.**

## Running it

```bash
uv venv && uv pip install -e .
cp .env.example .env          # then put your API key in .env
```

```bash
hac solve \
  --docs-dir  path/to/documents \
  --ledger    path/to/ledger.csv \
  --template  path/to/submission_template.json \
  --out       submission.json \
  --samples   3
```

`--provider` selects `openai` or `anthropic`; the model name is discovered from the
account rather than hard-coded, so a renamed or retired model does not stop a timed run.

Before sending, a structural check against the template:

```bash
hac check --submission submission.json --template path/to/submission_template.json
```

And where a key exists, scoring with the organiser's own formula:

```bash
hac score --submission submission.json --key path/to/ground_truth.json --cells
```

## Behaviour under a deadline

The run is written for a fixed window with no second attempt.

- Every stage degrades rather than fails. A clause the model will not read falls back to
  a rule-based reader; a metric that cannot be computed still writes a cell. An empty
  cell and a wrong cell score the same, so no cell is left null.
- The submission is rewritten after every borrower. A process killed at minute eighty
  leaves a valid file behind.
- Reading a clause is a sampling process: two runs of the same pipeline differed by 0.03.
  Each clause and supplement is read `--samples` times and the agreed reading is used.
  Agreement is reported per borrower, so a shaky reading is visible rather than silent.
  Consensus ignores free-text fields, since two readings that agree on every adjustment
  but word the reasoning differently are the same reading.
- `out/specs_seen.json` and `out/adjustments_seen.json` record exactly what the model
  read, for review while there is still time to review it.

## Layout

```
src/hac/
  bank.py         ledger, document linking, clause extraction   (no API calls)
  kyc.py          related parties, ownership thresholds, name matching
  categories.py   ordered rules from payment description to accounting line
  spec.py         the metric language and its evaluator
  audit.py        auditor adjustments applied to the ledger
  extract.py      the three model jobs and their schemas
  providers.py    OpenAI and Anthropic behind one interface
  vote.py         consensus across repeated readings
  pipeline.py     choosing which document of a borrower to read from
  covenants.py    the rule-based clause reader used when the model path fails
  solve.py        pack in, submission out
  score.py        the organiser's scoring formula, reproduced
  cli.py          solve, score, check
```

Everything in the first four files runs without an API key, which is why most of the
work could be done and measured before a single call was made.

## Why the columns differ

The design principle at the top of this file is that a model reads and Python computes.
It was followed for arithmetic and broken everywhere else. Categories, dossier parsing,
agreement detection and account identifiers were all hand-written rules, derived from
twelve borrowers and validated on those same twelve.

That choice was made when the project had a five-dollar budget and no API key, and the
aim was to do as much as possible without a call. By the time a full run cost fifty
cents the constraint was gone, and the architecture built for it was not revisited.

An explicit argument was written down for why the rules were safe: the answer key
carried `seed: 42, version: v1`, so the hidden pack came from the same generator, and
reading its grammar by hand looked like learning the rules rather than fitting the data.
Same generator, different conventions. The hidden pack used a different account prefix,
put covenants under different clause numbers, wrote the client dossier three ways
instead of one, and named capital purchases with the verb on the other side of the noun.
Each was found and patched inside the three-hour window, which means the ones that were
not found are the ones that cost the score.

Rerunning the open pack after every change reported a steady score throughout. That
measurement can only detect a rule that broke. A rule with no coverage of a form absent
from the open pack looks healthy right up to the moment it matters.

The transferable lesson, stated for whoever reads this next: anything recognised from
free-text description belongs to the model, not to a keyword list. Saving the call costs
more than the call.

## Limits

- Two cells of the open pack are not solved. One covenant tests group capital expenditure
  drawn from the consolidated statements of a parent that has no document in the pack.
  One proportion sits within a rounding step of its threshold and lands on the wrong side.
- The evidence rule names the transaction an adjustment acted on, or the one whose removal
  changes the verdict. Where neither applies, the field is left empty rather than guessed.
- Not an audit, and not legal, financial or accounting advice. Output is for review by a
  person who can check it against the source documents.
- Accuracy on the published pack of twelve borrowers did not predict accuracy on the
  hidden pack of twenty-seven. It fell from 0.9389 to 0.6139. The rule-based layer is
  where that went, and it has not been rewritten since.

## Tests

```bash
pip install -e . pytest
```

```bash
pytest tests -q
```

Sixteen tests, no API key required: expression evaluation and its refusal to execute code,
the arithmetic mismatch path, entity-name normalisation, dossier thresholds, the
related-party exclusion, auditor adjustments and the materiality floor, consensus across
readings, the scoring formula, and the field rebuild after a page is recovered from an
image.

## License

[MIT](LICENSE)
