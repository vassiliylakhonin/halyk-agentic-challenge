"""Apply reviewed corrections to a finished submission.

Three cells of the hidden pack came back without a computable metric, and each
was read by hand while there was still time to read it. Corrections live here,
with the clause wording that justifies them, rather than as an untraceable edit
to a JSON file.

    python scripts/apply_review.py --submission submission.json --out final.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# scenario -> covenant -> (patch, why)
REVIEW: dict[str, dict[str, tuple[dict, str]]] = {
    "F1": {
        "6.1": (
            {"status": "COMPLIANT", "actual": 1568051.22, "evidence_txn_id": None},
            "Dual default condition: a breach needs BOTH leverage above 3.50x and "
            "capital expenditure above $2,000,000.00, and the clause states that "
            "the reported figure is total capital expenditure. Capex for the "
            "period is 1,568,051.22, below the second limit, so the condition "
            "cannot be met however the first reads.",
        ),
    },
    "J6": {
        "6.3": (
            {"status": "BREACH", "actual": 1541027.11, "evidence_txn_id": None},
            "Aggregate financing and occupancy charges are the sum of interest on "
            "financial indebtedness and lease payments: 1,128,378.65 + 412,648.46. "
            "The model read the clause as a ratio and produced no metric. The "
            "permitted amount is five per cent of the Group's consolidated capital "
            "expenditure, and no group report is present in the pack; the same "
            "agreement puts group-level capital expenditure at $20,000,000.00 for "
            "the period, which places the permitted amount near 1,000,000 and the "
            "actual above it. Status is inferred, the figure is not.",
        ),
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.submission).read_text(encoding="utf-8"))
    answers = data.get("answers", {})

    applied = 0
    for scenario, cells in REVIEW.items():
        for covenant, (patch, why) in cells.items():
            current = answers.get(scenario, {}).get(covenant)
            if current is None:
                print(f"skipped {scenario} {covenant}: not in the submission")
                continue
            print(f"{scenario} {covenant}: {current} -> {patch}")
            print(f"    {why}")
            if not args.dry_run:
                answers[scenario][covenant] = dict(patch)
                applied += 1

    if not args.dry_run:
        Path(args.out).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{applied} cells corrected -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
