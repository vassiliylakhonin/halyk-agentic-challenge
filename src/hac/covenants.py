"""Turning a covenant clause into a number and a verdict.

A clause states a metric, a direction and a threshold. The families that repeat
across borrowers are handled here in Python; a clause that matches no family is
left for the model-backed path, which reads it and returns the same
specification this module produces by rule.

Nothing in here rounds before the end: the score decays with relative error, so
the only rounding that happens is the two decimals the submission asks for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .bank import Txn
from .categories import Category, category_from_clause, classify
from .kyc import KycProfile, normalize_entity

MONEY_RE = re.compile(r"\$\s*([\d\s,]+(?:\.\d{1,2})?)")
RATIO_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*x\b", re.I)
MIN_WORDS = ("не ниже", "не менее", "минимальн", "not less than", "minimum")
MAX_WORDS = ("не превыш", "максимальн", "не допускать, чтобы", "shall not exceed",
             "maximum", "ceiling")


def money(text: str) -> float | None:
    m = MONEY_RE.search(text)
    if not m:
        return None
    return float(re.sub(r"[\s,]", "", m.group(1)))


def ratio(text: str) -> float | None:
    m = RATIO_RE.search(text)
    return float(m.group(1).replace(",", ".")) if m else None


def direction(text: str) -> str:
    low = text.lower()
    i_min = min((low.find(w) for w in MIN_WORDS if low.find(w) >= 0), default=10**9)
    i_max = min((low.find(w) for w in MAX_WORDS if low.find(w) >= 0), default=10**9)
    return "min" if i_min < i_max else "max"


@dataclass
class Spec:
    """What a clause asks to be computed."""

    family: str
    direction: str                 # "min" | "max"
    threshold: float | None
    category: Category | None = None
    denominator: Category | None = None
    lines: list[Category] = field(default_factory=list)
    note: str = ""


@dataclass
class Outcome:
    actual: float | None
    status: str | None
    evidence_txn_id: str | None = None
    spec: Spec | None = None
    contributors: list[str] = field(default_factory=list)
    note: str = ""


def verdict(actual: float, spec: Spec) -> str:
    if spec.threshold is None:
        return "COMPLIANT"
    if spec.direction == "max":
        return "BREACH" if actual > spec.threshold else "COMPLIANT"
    return "BREACH" if actual < spec.threshold else "COMPLIANT"


# ---------------------------------------------------------------- recognising

def read_clause(text: str) -> Spec | None:
    """Rule-based reading of a covenant clause. None means the model path is needed."""
    low = text.lower()
    dirn = direction(text)

    if "связанным сторонам" in low or "related-party" in low or "аффилированн" in low:
        r = ratio(text)
        if r is not None and ("выручк" in low or "revenue" in low):
            return Spec(family="related_party_ratio", direction=dirn, threshold=r,
                        category=Category.REVENUE,
                        note="related-party payments over revenue")
        return Spec(family="related_party_abs", direction=dirn, threshold=money(text),
                    note="related-party payments, absolute")

    if "накладных расходов" in low and ("наибольш" in low or "по отдельности" in low):
        lines = []
        if "оплату труда" in low or "оплаты труда" in low:
            lines.append(Category.PAYROLL)
        if "коммунальн" in low:
            lines.append(Category.UTILITIES)
        return Spec(family="max_single_line", direction=dirn, threshold=money(text),
                    lines=lines, note="largest of the named overhead lines")

    cat = category_from_clause(text)
    if cat is Category.REVENUE and dirn == "min":
        return Spec(family="category_total", direction=dirn, threshold=money(text),
                    category=Category.REVENUE, note="revenue on a named line")
    if cat is not None and money(text) is not None:
        return Spec(family="category_total", direction=dirn, threshold=money(text),
                    category=cat, note=f"total on the {cat.value} line")
    return None


# ------------------------------------------------------------------ computing

def sum_category(txns: list[Txn], cat: Category, *, inflow: bool) -> tuple[float, list[str]]:
    total = 0.0
    ids: list[str] = []
    for t in txns:
        if t.amount is None:
            continue
        if inflow and t.amount <= 0:
            continue
        if not inflow and t.amount >= 0:
            continue
        if classify(t.description, t.counterparty)[0] is not cat:
            continue
        total += t.amount if inflow else -t.amount
        ids.append(t.txn_id)
    return total, ids


def sum_related_party(txns: list[Txn], profile: KycProfile | None) -> tuple[float, list[str]]:
    if profile is None:
        return 0.0, []
    keys = profile.related_keys
    total = 0.0
    ids: list[str] = []
    for t in txns:
        if t.amount is None or t.amount >= 0:
            continue
        if normalize_entity(t.counterparty) in keys:
            total += -t.amount
            ids.append(t.txn_id)
    return total, ids


def evaluate(spec: Spec, txns: list[Txn], profile: KycProfile | None) -> Outcome:
    if spec.family == "related_party_abs":
        total, ids = sum_related_party(txns, profile)
        return Outcome(actual=total, status=verdict(total, spec), spec=spec,
                       contributors=ids)

    if spec.family == "related_party_ratio":
        num, ids = sum_related_party(txns, profile)
        den, _ = sum_category(txns, Category.REVENUE, inflow=True)
        val = num / den if den else 0.0
        return Outcome(actual=val, status=verdict(val, spec), spec=spec,
                       contributors=ids,
                       note=f"{num:.2f} over {den:.2f}")

    if spec.family == "max_single_line":
        best = 0.0
        best_ids: list[str] = []
        for cat in spec.lines:
            total, ids = sum_category(txns, cat, inflow=False)
            if total > best:
                best, best_ids = total, ids
        return Outcome(actual=best, status=verdict(best, spec), spec=spec,
                       contributors=best_ids)

    if spec.family == "category_total":
        assert spec.category is not None
        inflow = spec.category is Category.REVENUE
        total, ids = sum_category(txns, spec.category, inflow=inflow)
        return Outcome(actual=total, status=verdict(total, spec), spec=spec,
                       contributors=ids)

    return Outcome(actual=None, status=None, spec=spec, note="no handler")
