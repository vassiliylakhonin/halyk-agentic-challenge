"""Majority merge across any number of runs. See merge_runs.py for the rules."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def rank(note: str | None) -> int:
    note = (note or "").lower()
    return 2 if note.startswith("unanimous") else 1 if note.startswith("majority") else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="submission.json:specs_seen.json pairs, earliest first")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = []
    for pair in args.runs:
        sub_path, spec_path = pair.split(":")
        runs.append((json.loads(Path(sub_path).read_text(encoding="utf-8")),
                     json.loads(Path(spec_path).read_text(encoding="utf-8")),
                     sub_path))

    base = json.loads(json.dumps(runs[-1][0]))
    reasons: Counter[str] = Counter()

    for scenario, cells in base["answers"].items():
        for covenant in cells:
            key = f"{scenario}.{covenant}"
            candidates = []
            for sub, specs, name in runs:
                cell = sub["answers"].get(scenario, {}).get(covenant)
                if cell and cell.get("status"):
                    candidates.append((cell, rank((specs.get(key) or {}).get("agreement")), name))
            if not candidates:
                reasons["no candidate"] += 1
                continue

            numeric = [c for c in candidates if c[0].get("actual") not in (0, 0.0)]
            pool = numeric or candidates
            if numeric and len(numeric) < len(candidates):
                reasons["dropped a zero metric"] += 1

            counts = Counter(c[0]["status"] for c in pool)
            top = counts.most_common()
            if len(top) > 1 and top[0][1] == top[1][1]:
                pool.sort(key=lambda c: c[1], reverse=True)
                chosen = pool[0][0]
                reasons["tie broken by agreement"] += 1
            else:
                status = top[0][0]
                agreeing = [c[0] for c in pool if c[0]["status"] == status]
                values = sorted(c["actual"] for c in agreeing
                                if isinstance(c.get("actual"), (int, float)))
                middle = len(values) // 2
                actual = (values[middle] if len(values) % 2
                          else (values[middle - 1] + values[middle]) / 2) if values else 0.0
                evidence = None
                named = [c["evidence_txn_id"] for c in agreeing if c.get("evidence_txn_id")]
                if named:
                    cand, hits = Counter(named).most_common(1)[0]
                    if hits > len(agreeing) / 2:
                        evidence = cand
                chosen = {"status": status, "actual": round(actual, 2),
                          "evidence_txn_id": evidence}
                reasons["majority" if len(counts) > 1 else "unanimous across runs"] += 1

            if chosen != cells[covenant]:
                print(f"{key:9} -> {chosen['status']:9} {chosen['actual']:>16}"
                      f"   (was {cells[covenant]['status']} {cells[covenant]['actual']})")
            base["answers"][scenario][covenant] = dict(chosen)

    Path(args.out).write_text(json.dumps(base, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print("\n" + json.dumps(dict(reasons), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
