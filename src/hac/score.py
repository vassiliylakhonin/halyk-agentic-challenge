"""Scorer that reproduces the published rules, so every change can be measured
against the open key instead of argued about.

Per cell, out of 1.0:
  status          0.50  exact "COMPLIANT" / "BREACH", else the whole cell is 0
  actual          0.30  linear decay with relative error, zero at 5%
  evidence_txn_id 0.20  exact match when the key names a transaction;
                        when the key is null those 0.20 decay with `actual`
                        on the same scale

The organiser weights cells by difficulty in the final tally. Those weights are
not published, so this reports the unweighted mean plus the per-cell detail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STATUSES = ("COMPLIANT", "BREACH")


def actual_fraction(got: Any, key: float | int | None) -> float:
    """Fraction of the `actual` points earned. Mirrors the published formula."""
    if key is None:
        return 0.0
    if isinstance(got, bool) or not isinstance(got, (int, float)):
        return 0.0
    if key == 0:
        return 1.0 if got == 0 else 0.0
    e = abs(float(got) - float(key)) / abs(float(key))
    return max(0.0, 1.0 - e / 0.05)


@dataclass
class CellScore:
    scenario: str
    covenant: str
    score: float
    status_ok: bool
    actual_frac: float
    evidence_ok: bool
    got: dict = field(default_factory=dict)
    key: dict = field(default_factory=dict)

    @property
    def note(self) -> str:
        if not self.status_ok:
            return f"status {self.got.get('status')!r} != {self.key.get('status')!r}"
        bits = []
        if self.actual_frac < 1.0:
            bits.append(f"actual {self.got.get('actual')} vs {self.key.get('actual')} "
                        f"({self.actual_frac:.0%} of the points)")
        if self.key.get("evidence_txn_id") and not self.evidence_ok:
            bits.append(f"evidence {self.got.get('evidence_txn_id')!r} vs "
                        f"{self.key.get('evidence_txn_id')!r}")
        return "; ".join(bits) or "exact"


def score_cell(got: dict | None, key: dict, scenario: str, covenant: str) -> CellScore:
    got = got or {}
    status_ok = got.get("status") in STATUSES and got.get("status") == key.get("status")
    if not status_ok:
        return CellScore(scenario, covenant, 0.0, False, 0.0, False, got, key)

    frac = actual_fraction(got.get("actual"), key.get("actual"))
    total = 0.50 + 0.30 * frac

    key_ev = key.get("evidence_txn_id")
    if key_ev:
        ev_ok = got.get("evidence_txn_id") == key_ev
        total += 0.20 if ev_ok else 0.0
    else:
        ev_ok = True           # nothing to match; points ride on `actual`
        total += 0.20 * frac

    return CellScore(scenario, covenant, round(total, 4), True, frac, ev_ok, got, key)


def score_submission(submission: dict, ground_truth: dict) -> dict:
    key_scenarios = ground_truth.get("scenarios", ground_truth)
    answers = submission.get("answers", {})

    cells: list[CellScore] = []
    for scenario, block in key_scenarios.items():
        covenants = block.get("covenants", block)
        for covenant, key in covenants.items():
            got = (answers.get(scenario) or {}).get(covenant)
            cells.append(score_cell(got, key, scenario, covenant))

    n = len(cells) or 1
    return {
        "cells": n,
        "total": round(sum(c.score for c in cells), 4),
        "mean": round(sum(c.score for c in cells) / n, 4),
        "status_correct": sum(1 for c in cells if c.status_ok),
        "actual_exact": sum(1 for c in cells if c.status_ok and c.actual_frac == 1.0),
        "evidence_correct": sum(1 for c in cells
                                if c.key.get("evidence_txn_id") and c.evidence_ok),
        "evidence_cells": sum(1 for c in cells if c.key.get("evidence_txn_id")),
        "by_covenant": {
            cov: round(sum(c.score for c in cells if c.covenant == cov)
                       / max(1, sum(1 for c in cells if c.covenant == cov)), 4)
            for cov in sorted({c.covenant for c in cells})
        },
        "detail": [
            {"scenario": c.scenario, "covenant": c.covenant,
             "score": c.score, "note": c.note}
            for c in sorted(cells, key=lambda c: (c.score, c.scenario))
        ],
    }


def load_and_score(submission_path: str | Path, ground_truth_path: str | Path) -> dict:
    sub = json.loads(Path(submission_path).read_text(encoding="utf-8"))
    gt = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    return score_submission(sub, gt)
