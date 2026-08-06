"""Does the model reproduce the hand-derived reading of the open pack?

Runs the model path over the open dataset, scores the result against the
published key, and prints the difference against the specifications and
adjustments that were written by hand. That difference is the whole question:
the hand-written files cannot be carried into the private set, so a model path
that scores materially lower is the real state of the submission.

    python scripts/validate_open.py --provider openai
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hac import audit, providers  # noqa: E402
from hac.bank import (Ledger, extract_pdfs, link_documents,  # noqa: E402
                      pages_needing_ocr)
from hac.extract import read_clause, read_page, read_supplement  # noqa: E402
from hac.pipeline import agreement_for, kyc_for  # noqa: E402
from hac.score import score_submission  # noqa: E402
from hac.spec import Context, decisive_txn, evaluate_spec  # noqa: E402

OPEN = Path("data/open")
SUPPLEMENT_MARKER = "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ"


def supplement_text(docs, scenario: str) -> str:
    for d in docs:
        if d.scenario == scenario and SUPPLEMENT_MARKER in d.text:
            tail = d.text.split(SUPPLEMENT_MARKER, 1)[1]
            return tail.split("За аудитора")[0][:8000]
    return ""


def procedures_reports(docs, scenario: str) -> str:
    """Agreed-upon-procedures reports referenced from the supplement. Drafts are
    included on purpose: refusing them is part of what is being tested."""
    parts = []
    for d in docs:
        if d.scenario == scenario and "согласованных процедур" in d.text:
            parts.append(" ".join(d.text.split())[:6000])
        elif d.scenario == scenario and "ПРОМЕЖУТОЧНАЯ ВЕДОМОСТЬ" in d.text:
            parts.append(" ".join(d.text.split())[:6000])
    return "\n\n---\n\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--skip-ocr", action="store_true")
    args = ap.parse_args()

    for line in (Path(".env").read_text().splitlines() if Path(".env").exists() else []):
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            import os
            os.environ.setdefault(k.strip(), v.strip())

    usage = providers.Usage()
    chat = providers.build(args.provider, args.model, usage)
    print(f"provider={args.provider} model={chat.name}\n")

    ledger = Ledger.load(OPEN / "master_ledger_2025.csv")
    docs = link_documents(
        extract_pdfs(OPEN / "documents", "out/pdftext_model.json", None), ledger)

    # 1. pages with no text layer
    recovered = 0
    for d in docs:
        blanks = pages_needing_ocr(d.pages)
        if not blanks or args.skip_ocr:
            continue
        for n in blanks:
            try:
                import pypdfium2 as pdfium
                page = pdfium.PdfDocument(d.path)[n - 1]
                image = page.render(scale=2).to_pil()
                import io
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                d.pages[n - 1] = read_page(chat, buf.getvalue())
                recovered += 1
            except Exception as e:  # rendering is optional, extraction is not
                print(f"  page render failed for {d.doc_id} p{n}: "
                      f"{type(e).__name__}: {e}")
        d.text = "\n".join(d.pages)
    print(f"pages recovered by the model: {recovered}")

    template = json.loads((OPEN / "submission_template.json").read_text())
    by_scenario = ledger.by_scenario()
    specs: dict[str, dict] = {}
    adjustments: dict[str, dict] = {}

    submission = {"team": "VizierAI", "contact_email": "", "model": chat.name,
                  "answers": {}}

    for scenario, cells in template["answers"].items():
        agreement = agreement_for(scenario, docs)
        text = supplement_text(docs, scenario) + "\n\n" + procedures_reports(docs, scenario)
        try:
            block = read_supplement(chat, text) if text.strip() else {}
        except Exception as e:
            block = {}
            print(f"  {scenario}: supplement failed {type(e).__name__}: {e}")
        adjustments[scenario] = block

        txns = by_scenario.get(scenario, [])
        adj = audit.build(block, txns)
        ctx = Context(txns=txns, profile=kyc_for(scenario, docs),
                      overrides=adj.overrides)

        specs[scenario] = {}
        submission["answers"][scenario] = {}
        for covenant in cells:
            clause = agreement.clause(covenant) if agreement else ""
            try:
                spec = read_clause(chat, clause) if clause else {}
            except Exception as e:
                spec = {}
                print(f"  {scenario} {covenant}: clause failed "
                      f"{type(e).__name__}: {e}")
            specs[scenario][covenant] = spec

            if spec.get("numerator"):
                extra = []
                if covenant == "6.1" and adj.add_back_total():
                    extra.append({"const": adj.add_back_total()})
                if covenant == "6.1":
                    extra += [{"const": o["amount"]} for o in adj.off_ledger]
                run = dict(spec)
                if extra:
                    run["numerator"] = list(spec["numerator"]) + extra
                result = evaluate_spec(run, ctx)
                evidence = decisive_txn(run, ctx, result)
            else:
                result, evidence = None, None

            submission["answers"][scenario][covenant] = {
                "status": result.status if result else None,
                "actual": round(result.actual, 2)
                if result and result.actual is not None else None,
                "evidence_txn_id": evidence,
            }

    Path("out").mkdir(exist_ok=True)
    Path("out/specs_model.json").write_text(
        json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
    Path("out/adjustments_model.json").write_text(
        json.dumps(adjustments, ensure_ascii=False, indent=2), encoding="utf-8")
    Path("out/submission_model.json").write_text(
        json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")

    key = json.loads((OPEN / "ground_truth.json").read_text())
    result = score_submission(submission, key)
    print(f"\nmodel path: {result['mean']:.4f} "
          f"({result['total']:.2f}/{result['cells']}) | "
          f"status {result['status_correct']}/{result['cells']} | "
          f"evidence {result['evidence_correct']}/{result['evidence_cells']}")
    print(f"by covenant: {result['by_covenant']}")
    print(f"calls {usage.calls}, input {usage.input_tokens}, "
          f"output {usage.output_tokens}")

    print("\ncells below full marks:")
    for cell in result["detail"]:
        if cell["score"] < 1.0:
            print(f"  {cell['scenario']:4} {cell['covenant']}  "
                  f"{cell['score']:.2f}  {cell['note'][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
