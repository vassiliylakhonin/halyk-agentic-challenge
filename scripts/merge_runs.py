"""Merge two complete runs into one submission.

Two runs of the pipeline over the same pack agree on most cells and disagree on
a few, because reading a clause is a sampling process. Where they disagree the
choice is made by rule, in this order:

1. A metric of exactly zero is almost always a failed reading, not a covenant
   that happens to sit at zero. The run that produced a number wins.
2. Otherwise, the run whose specification was unanimous across its own repeated
   readings wins over one that was only a majority.
3. Otherwise the later run wins, because it carries the fixes made between them.

Every decision is printed with the rule that made it, so the merge can be read
rather than trusted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def agreement_rank(note: str | None) -> int:
    note = (note or "").lower()
    if note.startswith("unanimous"):
        return 2
    if note.startswith("majority"):
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="earlier submission")
    ap.add_argument("--b", required=True, help="later submission")
    ap.add_argument("--specs-a", required=True)
    ap.add_argument("--specs-b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    sa = json.loads(Path(args.specs_a).read_text(encoding="utf-8"))
    sb = json.loads(Path(args.specs_b).read_text(encoding="utf-8"))

    merged = json.loads(json.dumps(b))          # start from the later run
    reasons: Counter[str] = Counter()

    for scenario, cells in b["answers"].items():
        for covenant in cells:
            cell_a = a["answers"].get(scenario, {}).get(covenant)
            cell_b = cells[covenant]
            if cell_a is None or cell_a == cell_b:
                reasons["identical"] += 1
                continue

            key = f"{scenario}.{covenant}"
            zero_a = cell_a.get("actual") in (0, 0.0)
            zero_b = cell_b.get("actual") in (0, 0.0)

            if zero_b and not zero_a:
                chosen, rule = cell_a, "zero metric in the later run"
            elif zero_a and not zero_b:
                chosen, rule = cell_b, "zero metric in the earlier run"
            else:
                rank_a = agreement_rank((sa.get(key) or {}).get("agreement"))
                rank_b = agreement_rank((sb.get(key) or {}).get("agreement"))
                if rank_a > rank_b:
                    chosen, rule = cell_a, "earlier run read it unanimously"
                elif rank_b > rank_a:
                    chosen, rule = cell_b, "later run read it unanimously"
                else:
                    chosen, rule = cell_b, "equal agreement, later run carries the fixes"

            merged["answers"][scenario][covenant] = dict(chosen)
            reasons[rule] += 1
            if chosen != cell_b or cell_a["status"] != cell_b["status"]:
                print(f"{key:9} {cell_a['status']:9}/{cell_a['actual']:>14} vs "
                      f"{cell_b['status']:9}/{cell_b['actual']:>14} -> "
                      f"{chosen['status']:9} ({rule})")

    Path(args.out).write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print("\n" + json.dumps(dict(reasons), ensure_ascii=False, indent=2))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
