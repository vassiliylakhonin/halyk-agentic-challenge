# Halyk Agentic Challenge

A competition entry for testing loan covenants from banking documents and a transaction ledger. Models extract metric definitions and document adjustments; Python calculates amounts, ratios, and covenant statuses.

**Result:** 0.6139 on the hidden pack, placing 83rd of 160. The open-pack score was 0.9389. The gap exposes the limits of the document-parsing rules; these results do not establish production accuracy.

## How it works

| Stage | Responsibility |
|---|---|
| Read documents | Select agreement editions, link borrower records, and recover scanned text |
| Extract specifications | Ask a model for structured clause definitions and ledger adjustments |
| Compute | Evaluate the metric specification against the ledger in Python |
| Reconcile | Compare repeated readings and record agreement and extracted specifications |
| Export | Write the competition submission and check its structure against a template |

Model extraction can be wrong even when the arithmetic is exact. Some document recognition and categorisation still use rules fitted to the open pack. The pipeline's deadline-oriented fallbacks can produce a populated cell without reliable evidence; a complete submission is not a verified answer.

## Quick start

Python 3.11 or later, from a checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/hac demo
```

`hac demo` runs one explicitly synthetic capex covenant without an API key or private competition data.

For a real document pack, configure the provider key using [.env.example](.env.example), then run:

```bash
.venv/bin/hac solve \
  --docs-dir path/to/documents \
  --ledger path/to/ledger.csv \
  --template path/to/submission_template.json \
  --out submission.json \
  --samples 3
.venv/bin/hac check --submission submission.json --template path/to/submission_template.json
```

`--provider` selects OpenAI or Anthropic. A solve run sends document content to the configured model provider; use only material you are authorised to process there. Competition documents and answer keys are not bundled.

When an answer key is available, `hac score --submission submission.json --key path/to/ground_truth.json --cells` applies the organiser's scoring formula.

## Results and limitations

| Dataset | Borrowers | Score | Placing |
|---|---|---|---|
| Open pack | 12 | 0.9389 | — |
| Hidden pack | 27 | 0.6139 | 83 / 160 |

The hidden pack used document conventions that the open-pack rules did not cover. Repeatedly passing the open pack concealed those gaps. See [results and design notes](docs/results-and-design.md) for the detailed failure analysis, evidence-transaction limits, and deadline behaviour.

Outputs require human review against source documents. This is a competition implementation, not an audit or legal, financial, accounting, or compliance advice.

## Code and tests

The reusable components live under [src/hac/](src/hac/): metric specifications and evaluation, document and ledger linking, provider adapters, repeated-reading consensus, and submission scoring. [cli.py](src/hac/cli.py) defines the available commands.

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest tests -q
```

The tests run without an API key. They cover deterministic arithmetic, refusal to execute expressions as code, adjustments, entity matching, consensus, and scoring. Passing them does not validate model extraction on unseen documents.

[MIT license](LICENSE).
