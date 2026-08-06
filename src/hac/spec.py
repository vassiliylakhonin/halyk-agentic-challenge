"""A small expression language for covenant metrics.

The division of labour: a model reads the clause and emits one of these
specifications; Python evaluates it. The model never multiplies or divides,
which is what protects the 0.30 of each cell that decays with relative error.

A specification is plain JSON, so it can be inspected, diffed, cached and
hand-written. Terms:

    {"cat": "capex"}                     total on a category, sign-corrected
    {"cat": "revenue", "quarter": 4}     restricted to a calendar quarter
    {"ebitda": {}}                       revenue minus operating costs
    {"related": {}}                      payments to related parties
    {"max": [term, term]}                the larger of the terms
    {"const": 918447.52}                 an amount disclosed outside the ledger

A metric is a list of terms (summed) over an optional denominator list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bank import Txn
from .categories import Category, classify
from .kyc import KycProfile, normalize_entity

INFLOW_CATEGORIES = {Category.REVENUE, Category.NON_OPERATING, Category.FINANCING}


@dataclass
class Context:
    txns: list[Txn]
    profile: KycProfile | None = None
    overrides: dict[str, dict] = field(default_factory=dict)  # txn_id -> patch

    def effective(self, t: Txn) -> tuple[float | None, Category]:
        """Amount and category after the auditor's adjustments are applied."""
        patch = self.overrides.get(t.txn_id, {})
        amount = patch.get("amount", t.amount)
        if "category" in patch:
            cat = Category(patch["category"])
        else:
            cat = classify(t.description, t.counterparty)[0]
        return amount, cat


def _quarter(date: str) -> int:
    try:
        month = int(date.split("-")[1])
    except (IndexError, ValueError):
        return 0
    return (month - 1) // 3 + 1


def category_total(ctx: Context, cat: Category, quarter: int | None = None
                   ) -> tuple[float, list[str]]:
    """Total on one accounting line.

    Payments to related parties sit on their own line and are not part of any
    other. The pack makes this visible: a management advisory retainer paid to a
    dossier-identified related party reads as an operating cost, and counting it
    as one puts every ratio built on operating costs out by its amount.
    """
    inflow = cat in INFLOW_CATEGORIES
    related = ctx.profile.related_keys if ctx.profile else set()
    total = 0.0
    ids: list[str] = []
    for t in ctx.txns:
        amount, c = ctx.effective(t)
        if amount is None or c is not cat:
            continue
        if (not inflow and related
                and normalize_entity(t.counterparty) in related):
            continue
        if inflow and amount <= 0:
            continue
        if not inflow and amount >= 0:
            continue
        if quarter and _quarter(t.date) != quarter:
            continue
        total += amount if inflow else -amount
        ids.append(t.txn_id)
    return total, ids


def related_total(ctx: Context) -> tuple[float, list[str]]:
    if ctx.profile is None:
        return 0.0, []
    keys = ctx.profile.related_keys
    total = 0.0
    ids: list[str] = []
    for t in ctx.txns:
        amount, _ = ctx.effective(t)
        if amount is None or amount >= 0:
            continue
        if normalize_entity(t.counterparty) in keys:
            total += -amount
            ids.append(t.txn_id)
    return total, ids


def eval_term(term: dict[str, Any], ctx: Context) -> tuple[float, list[str]]:
    value, ids = _eval_term(term, ctx)
    return (-value if term.get("negate") else value), ids


def _eval_term(term: dict[str, Any], ctx: Context) -> tuple[float, list[str]]:
    if "const" in term:
        return float(term["const"]), []
    if "cat" in term:
        return category_total(ctx, Category(term["cat"]), term.get("quarter"))
    if "related" in term:
        return related_total(ctx)
    if "ebitda" in term:
        rev, a = category_total(ctx, Category.REVENUE)
        opex, b = category_total(ctx, Category.OPEX)
        return rev - opex, a + b
    if "max" in term:
        best, best_ids = 0.0, []
        for sub in term["max"]:
            v, ids = eval_term(sub, ctx)
            if v > best:
                best, best_ids = v, ids
        return best, best_ids
    raise ValueError(f"unknown term: {term}")


def eval_terms(terms: list[dict], ctx: Context) -> tuple[float, list[str]]:
    total = 0.0
    ids: list[str] = []
    for term in terms:
        v, i = eval_term(term, ctx)
        total += v
        ids.extend(i)
    return total, ids


@dataclass
class Result:
    actual: float | None
    status: str | None
    contributors: list[str] = field(default_factory=list)
    note: str = ""


def evaluate_spec(spec: dict, ctx: Context) -> Result:
    """Compute one covenant. Returns None values when the spec cannot be applied."""
    numerator = spec.get("numerator")
    if not numerator:
        return Result(None, None, note="spec has no numerator")

    value, ids = eval_terms(numerator, ctx)
    denominator = spec.get("denominator")
    if denominator:
        den, den_ids = eval_terms(denominator, ctx)
        if den == 0:
            return Result(None, None, note="denominator is zero")
        value = value / den
        ids = ids + den_ids

    direction = spec.get("direction", "max")
    threshold = spec.get("threshold")

    # A springing test only applies once its trigger is met; until then the
    # borrower is compliant and the metric is still reported.
    springing = spec.get("springing")
    if springing:
        trigger, _ = eval_terms(springing["terms"], ctx)
        if trigger <= springing.get("above", 0):
            return Result(abs(value), "COMPLIANT", ids, note="springing test not triggered")

    if threshold is None:
        return Result(abs(value), None, ids, note="no threshold")

    # The clause states its limit to the same two decimals the submission asks
    # for, so the comparison is made at that precision. A ratio of 0.0442
    # against a 0.04x cap is reported as 0.04 and is not a breach.
    reported = round(abs(value), 2)
    signed = -reported if value < 0 else reported
    if direction == "max":
        status = "BREACH" if signed > threshold else "COMPLIANT"
    else:
        status = "BREACH" if signed < threshold else "COMPLIANT"
    return Result(abs(value), status, ids)


def decisive_txn(spec: dict, ctx: Context, result: Result) -> str | None:
    """The one transaction whose removal changes the verdict.

    The case defines evidence exactly that way: a line that merely contributes
    to a total is not evidence. So each contributing transaction is dropped in
    turn, and only an unambiguous single flip counts.
    """
    if result.status is None or not result.contributors:
        return None
    flips: list[str] = []
    for txn_id in dict.fromkeys(result.contributors):
        patched = dict(ctx.overrides)
        patched[txn_id] = {**patched.get(txn_id, {}), "amount": None}
        probe = evaluate_spec(spec, Context(ctx.txns, ctx.profile, patched))
        if probe.status and probe.status != result.status:
            flips.append(txn_id)
    return flips[0] if len(flips) == 1 else None


def load_specs(path: str | Path) -> dict[str, dict[str, dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}
