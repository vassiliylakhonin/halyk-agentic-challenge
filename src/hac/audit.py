"""Applying the auditor's covenant supplement to the ledger.

The supplement is where the pack hides its decisive moves: a payment moved from
one line to another, an invoice pushed out of the period, an amount that never
reached the ledger dump, a rate taken from an actual settlement. It is also
where it hides its decoys - an adjustment the auditor considered and rejected,
and a draft schedule carrying the same reference number as the final report.

Adjustments are matched to a transaction by amount and counterparty, because the
supplement names those and not the transaction id. The transaction an adjustment
lands on is also the transaction that answers `evidence_txn_id`: removing it is
what changes the verdict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .bank import Txn
from .categories import Category
from .kyc import normalize_entity

CENT = 0.01


@dataclass
class Adjustments:
    overrides: dict[str, dict] = field(default_factory=dict)
    off_ledger: list[dict] = field(default_factory=list)
    addbacks: dict = field(default_factory=dict)
    touched: list[str] = field(default_factory=list)   # evidence candidates, in order
    notes: list[str] = field(default_factory=list)

    def add_back_total(self) -> float:
        if not self.addbacks:
            return 0.0
        floor = self.addbacks.get("materiality", 0.0)
        return sum(a for a in self.addbacks.get("items", []) if a >= floor)

    def off_ledger_total(self, category: Category) -> float:
        return sum(o["amount"] for o in self.off_ledger
                   if Category(o.get("category", "other")) is category)


def _match(txns: list[Txn], amount: float, counterparty: str | None) -> Txn | None:
    key = normalize_entity(counterparty) if counterparty else None
    for t in txns:
        if t.amount is None:
            continue
        if abs(abs(t.amount) - abs(amount)) > CENT:
            continue
        if key and normalize_entity(t.counterparty) != key:
            continue
        return t
    return None


def build(block: dict, txns: list[Txn]) -> Adjustments:
    adj = Adjustments(addbacks=block.get("addbacks", {}))

    for item in block.get("reclass", []):
        t = _match(txns, item["amount"], item.get("counterparty"))
        if t is None:
            adj.notes.append(f"reclassification not matched: {item}")
            continue
        adj.overrides.setdefault(t.txn_id, {})["category"] = item["to"]
        adj.touched.append(t.txn_id)

    for item in block.get("exclude", []):
        txn_id = item.get("txn_id")
        if txn_id is None:
            t = _match(txns, item.get("amount", 0), item.get("counterparty"))
            txn_id = t.txn_id if t else None
        if txn_id:
            adj.overrides.setdefault(txn_id, {})["amount"] = None
            adj.touched.append(txn_id)

    for item in block.get("amount_fix", []):
        adj.overrides.setdefault(item["txn_id"], {})["amount"] = item["amount"]
        adj.touched.append(item["txn_id"])

    for item in block.get("fx", []):
        rate = item["usd"] / item["foreign"] if item.get("foreign") else None
        if not rate:
            continue
        for t in txns:
            if t.currency == item["currency"] and t.amount is not None:
                adj.overrides.setdefault(t.txn_id, {})["amount"] = t.amount * rate
        adj.notes.append(f"{item['currency']} converted at {rate:.6f}")

    adj.off_ledger = list(block.get("off_ledger", []))
    return adj


def load(path: str | Path) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}
