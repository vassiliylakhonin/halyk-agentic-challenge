from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import Config


def _load_env() -> None:
    """Minimal .env support so the key never has to live in shell history."""
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def cmd_solve(cfg: Config, args) -> int:
    """A document pack in, a submission out."""
    from .solve import solve

    report = solve(
        args.docs_dir, args.ledger, args.template, args.out,
        provider=args.provider, model=args.llm_model, samples=args.samples,
        team=args.team or cfg.team,
        contact_email=args.contact_email or cfg.contact_email,
    )
    print(json.dumps(report.summary(), ensure_ascii=False, indent=2))

    out_dir = Path(args.trace_dir or cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adjustments_seen.json").write_text(
        json.dumps(report.adjustments, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "specs_seen.json").write_text(json.dumps(
        {f"{t.scenario}.{t.covenant}": {"source": t.source, "agreement": t.note,
                                        "spec": t.spec}
         for t in report.traces}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"what the model read: {out_dir}/specs_seen.json, "
          f"{out_dir}/adjustments_seen.json")
    return 0


def cmd_score(cfg: Config, args) -> int:
    """Score a submission against a key, using the organiser's formula."""
    from .score import load_and_score

    result = load_and_score(args.submission, args.key)
    detail = result.pop("detail")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.cells:
        print("\ncells below full marks:")
        for cell in detail:
            if cell["score"] < 1.0:
                print(f"  {cell['scenario']:4} {cell['covenant']}  "
                      f"{cell['score']:.2f}  {cell['note']}")
    return 0


def cmd_check(cfg: Config, args) -> int:
    """Structural check of a submission against the template, before sending it."""
    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    submission = json.loads(Path(args.submission).read_text(encoding="utf-8"))
    problems: list[str] = []

    for field in ("team", "contact_email", "model"):
        if not str(submission.get(field, "")).strip():
            problems.append(f"top-level field {field} is empty")
    if set(template) != set(submission):
        problems.append("top-level keys differ from the template")

    answers = submission.get("answers", {})
    for scenario, cells in template.get("answers", {}).items():
        if scenario not in answers:
            problems.append(f"{scenario}: missing")
            continue
        for covenant, shape in cells.items():
            cell = answers[scenario].get(covenant)
            if cell is None:
                problems.append(f"{scenario} {covenant}: missing")
                continue
            if set(cell) != set(shape):
                problems.append(f"{scenario} {covenant}: fields differ")
            if cell.get("status") not in ("COMPLIANT", "BREACH"):
                problems.append(f"{scenario} {covenant}: status {cell.get('status')!r}")
            actual = cell.get("actual")
            if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                problems.append(f"{scenario} {covenant}: actual is not a number")
            elif actual < 0:
                problems.append(f"{scenario} {covenant}: actual is negative")
            elif round(actual, 2) != actual:
                problems.append(f"{scenario} {covenant}: more than two decimals")

    extra = set(answers) - set(template.get("answers", {}))
    if extra:
        problems.append(f"scenarios not in the template: {sorted(extra)}")

    print(json.dumps({"ok": not problems, "problems": problems},
                     ensure_ascii=False, indent=2))
    return 0 if not problems else 1


def cmd_demo(cfg: Config, args) -> int:
    """Run the deterministic core on one synthetic covenant without API access."""
    from .bank import Txn
    from .spec import Context, decisive_txn, evaluate_spec

    txns = [
        Txn(
            "TXN-DEMO-1",
            "2026-01-15",
            "DEMO-1001",
            "Synthetic Equipment Vendor",
            "Purchase of production equipment",
            -1_250_000.0,
            "USD",
        )
    ]
    spec = {
        "numerator": [{"cat": "capex"}],
        "direction": "max",
        "threshold": 1_000_000.0,
    }
    result = evaluate_spec(spec, Context(txns=txns))
    payload = {
        "fixture": "synthetic; not customer or competition data",
        "status": result.status,
        "actual": result.actual,
        "threshold": spec["threshold"],
        "evidence_txn_id": decisive_txn(spec, Context(txns=txns), result),
        "model_called": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_env()
    p = argparse.ArgumentParser(
        prog="hac", description="Covenant compliance from a document pack")
    p.add_argument("-c", "--config", default="config.json")
    sub = p.add_subparsers(dest="cmd", required=True)

    sv = sub.add_parser("solve", help="document pack -> submission.json")
    sv.add_argument("--docs-dir", required=True)
    sv.add_argument("--ledger", required=True)
    sv.add_argument("--template", required=True)
    sv.add_argument("--out", default="submission.json")
    sv.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    sv.add_argument("--llm-model", default=None)
    sv.add_argument("--samples", type=int, default=3,
                    help="readings per clause; the agreed reading is used")
    sv.add_argument("--team", default=None)
    sv.add_argument("--contact-email", default=None)
    sv.add_argument("--trace-dir", default=None)
    sv.set_defaults(func=cmd_solve)

    sc = sub.add_parser("score", help="score a submission against a key")
    sc.add_argument("--submission", required=True)
    sc.add_argument("--key", required=True)
    sc.add_argument("--cells", action="store_true", help="list cells below full marks")
    sc.set_defaults(func=cmd_score)

    ck = sub.add_parser("check", help="structural check before sending")
    ck.add_argument("--submission", required=True)
    ck.add_argument("--template", required=True)
    ck.set_defaults(func=cmd_check)

    dm = sub.add_parser("demo", help="run one synthetic deterministic covenant check")
    dm.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    cfg = Config.load(args.config)
    return args.func(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
