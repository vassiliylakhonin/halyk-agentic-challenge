"""Repeating the reading and taking the consensus.

Two runs of the same pipeline over the same pack differ by a few hundredths of a
point, because reading a clause is a sampling process. At three minutes and a
fraction of a cent per run, the fix is not a better prompt but more runs: read
each clause and each supplement several times and keep what the readings agree
on.

Consensus is taken on the specification, not on the answer, so the arithmetic
still happens once, in Python, over an agreed description of the metric.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Callable, TypeVar

T = TypeVar("T")


# Free-text fields explain a reading rather than drive it. Two runs that agree on
# every adjustment but word their reasoning differently are the same reading, so
# prose is left out of the identity.
PROSE_KEYS = ("ignored", "why", "reasoning", "note")


def _strip_prose(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_prose(v) for k, v in value.items() if k not in PROSE_KEYS}
    if isinstance(value, list):
        return [_strip_prose(v) for v in value]
    return value


def _key(value: Any) -> str:
    """A stable identity for a JSON structure, so near-identical readings group."""
    return json.dumps(_strip_prose(value), sort_keys=True, ensure_ascii=False)


def consensus(readings: list[T], *, minimum: int = 1) -> tuple[T | None, float]:
    """The most common reading and the share of runs that produced it."""
    usable = [r for r in readings if r]
    if not usable:
        return None, 0.0
    counts = Counter(_key(r) for r in usable)
    winner, hits = counts.most_common(1)[0]
    if hits < minimum:
        return None, 0.0
    for r in usable:
        if _key(r) == winner:
            return r, hits / len(readings)
    return None, 0.0


def repeat(fn: Callable[[], T], times: int, on_error: Callable[[Exception], None]
           | None = None) -> list[T]:
    out: list[T] = []
    for _ in range(max(1, times)):
        try:
            out.append(fn())
        except Exception as e:  # a failed sample is a missing vote, not a failure
            if on_error:
                on_error(e)
    return out


def spec_consensus(readings: list[dict]) -> tuple[dict | None, float, str]:
    """Consensus over covenant specifications.

    Falls back to the reading whose threshold and direction the majority agree
    on, since those two fields decide the verdict and therefore half the cell.
    """
    agreed, share = consensus(readings)
    if agreed is not None and share > 0.5:
        return agreed, share, "unanimous" if share == 1.0 else "majority"

    usable = [r for r in readings if r and r.get("numerator")]
    if not usable:
        return None, 0.0, "no usable reading"

    core = Counter((r.get("direction"), r.get("threshold")) for r in usable)
    (direction, threshold), hits = core.most_common(1)[0]
    for r in usable:
        if (r.get("direction"), r.get("threshold")) == (direction, threshold):
            return r, hits / len(readings), "agreed on threshold and direction only"
    return usable[0], 1 / len(readings), "no agreement, first reading kept"


def cell_consensus(cells: list[dict]) -> dict:
    """Consensus over finished cells, used when specifications disagree.

    Status is decided by majority. `actual` is the median of the runs that voted
    with the majority, which is robust to a single wild reading in a way the mean
    is not. Evidence is taken only when a majority names the same transaction.
    """
    usable = [c for c in cells if c and c.get("status")]
    if not usable:
        return {"status": None, "actual": None, "evidence_txn_id": None}

    status, _ = Counter(c["status"] for c in usable).most_common(1)[0]
    agreeing = [c for c in usable if c["status"] == status]

    values = sorted(c["actual"] for c in agreeing if isinstance(c.get("actual"), (int, float)))
    if values:
        middle = len(values) // 2
        actual = (values[middle] if len(values) % 2
                  else (values[middle - 1] + values[middle]) / 2)
    else:
        actual = None

    evidence = None
    named = [c["evidence_txn_id"] for c in agreeing if c.get("evidence_txn_id")]
    if named:
        candidate, hits = Counter(named).most_common(1)[0]
        if hits > len(agreeing) / 2:
            evidence = candidate

    return {
        "status": status,
        "actual": None if actual is None else round(actual, 2),
        "evidence_txn_id": evidence,
    }
