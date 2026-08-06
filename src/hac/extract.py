"""What the model is asked to do, and nothing more.

Three jobs, each returning JSON against a schema:

  read_clause      a covenant clause -> a metric specification
  read_supplement  an auditor's covenant supplement -> adjustments
  read_page        a page image -> its text

The model never computes a total, a ratio or a verdict. It describes what should
be computed, and Python does it. That split is deliberate: half of every cell's
score rides on a number being right to within five per cent, and an interpreter
does not misplace a decimal.
"""

from __future__ import annotations

from typing import Any

from .categories import Category
from .providers import Chat

CATEGORIES = [c.value for c in Category]

TERM = {
    "type": "object",
    "properties": {
        "cat": {"type": ["string", "null"], "enum": CATEGORIES + [None],
                "description": "an accounting line to total"},
        "quarter": {"type": ["integer", "null"],
                    "description": "restrict to a calendar quarter, 1-4"},
        "negate": {"type": ["boolean", "null"],
                   "description": "subtract this term instead of adding it"},
        "const": {"type": ["number", "null"],
                  "description": "a fixed amount stated outside the ledger"},
        "related": {"type": ["boolean", "null"],
                    "description": "total of payments to related parties"},
        "ebitda": {"type": ["boolean", "null"],
                   "description": "revenue minus operating costs"},
        "max_of": {"type": ["array", "null"], "items": {"type": "string",
                                                        "enum": CATEGORIES},
                   "description": "the larger of these lines, taken singly"},
    },
    "additionalProperties": False,
}

SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "numerator": {"type": "array", "items": TERM},
        "denominator": {"type": ["array", "null"], "items": TERM},
        "direction": {"type": "string", "enum": ["min", "max"]},
        "threshold": {"type": ["number", "null"]},
        "springing_terms": {"type": ["array", "null"], "items": TERM,
                            "description": "trigger metric of a springing test"},
        "springing_above": {"type": ["number", "null"]},
        "reasoning": {"type": "string", "description": "one sentence"},
    },
    "required": ["numerator", "direction", "threshold", "reasoning"],
    "additionalProperties": False,
}

SPEC_SYSTEM = """You convert a financial covenant from a bank loan agreement into a \
specification that a program will evaluate against a transaction ledger.

Report only what the clause says.

- direction is "max" when the borrower must not exceed the limit, "min" when it must \
stay at or above it.
- threshold is the printed number: a money amount as a plain number, a ratio written \
"0.42x" as 0.42.
- numerator is the metric being limited. denominator is present only when the clause \
states a ratio of one aggregate to another.
- A clause that applies only once some trigger is exceeded is a springing test: put the \
trigger metric in springing_terms and its level in springing_above.
- "the larger of A and B, taken singly and not in aggregate" is max_of.
- EBITDA means revenue minus operating costs unless the clause defines it otherwise.
- Do not compute anything. Do not invent a category that the clause does not name."""

ADJ_SCHEMA = {
    "type": "object",
    "properties": {
        "reclass": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "counterparty": {"type": "string"},
                "to": {"type": "string", "enum": CATEGORIES},
                "why": {"type": "string"},
            },
            "required": ["amount", "to"], "additionalProperties": False}},
        "exclude": {"type": "array", "items": {
            "type": "object",
            "properties": {"txn_id": {"type": "string"}, "why": {"type": "string"}},
            "required": ["txn_id"], "additionalProperties": False}},
        "amount_fix": {"type": "array", "items": {
            "type": "object",
            "properties": {"txn_id": {"type": "string"}, "amount": {"type": "number"},
                           "why": {"type": "string"}},
            "required": ["txn_id", "amount"], "additionalProperties": False}},
        "off_ledger": {"type": "array", "items": {
            "type": "object",
            "properties": {"amount": {"type": "number"},
                           "category": {"type": "string", "enum": CATEGORIES},
                           "why": {"type": "string"}},
            "required": ["amount", "category"], "additionalProperties": False}},
        "fx": {"type": "array", "items": {
            "type": "object",
            "properties": {"currency": {"type": "string"}, "foreign": {"type": "number"},
                           "usd": {"type": "number"}, "why": {"type": "string"}},
            "required": ["currency", "foreign", "usd"], "additionalProperties": False}},
        "addbacks": {"type": ["object", "null"], "properties": {
            "materiality": {"type": "number"},
            "items": {"type": "array", "items": {"type": "number"}}},
            "additionalProperties": False},
        "ignored": {"type": "array", "items": {"type": "string"},
                    "description": "adjustments deliberately not applied, and why"},
    },
    "additionalProperties": False,
}

ADJ_SYSTEM = """You read an auditor's covenant-compliance supplement and list the \
adjustments that must be applied to a transaction ledger before covenants are tested.

Apply only what the auditor concluded. The supplements contain deliberate distractors:

- An adjustment the auditor considered and then rejected must NOT be applied. Wording \
such as "первоначальная классификация сохраняется" or "корректировка не производилась" \
means no adjustment. Record it in `ignored` instead.
- A draft or interim schedule is superseded by the final agreed-upon-procedures report, \
even when both carry the same reference number. Wording such as "ПРОЕКТ", \
"ПРОМЕЖУТОЧНАЯ ВЕДОМОСТЬ" or "не является окончательной позицией" means ignore it.
- Boilerplate accounting policy is not an adjustment.

Kinds of adjustment:
- reclass: an amount moved from one accounting line to another.
- exclude: a transaction pushed outside the covenant period, for example an invoice \
whose services fall in the next year.
- amount_fix: a transaction whose amount is missing or wrong in the ledger dump, with \
the corrected amount. An expense is negative.
- off_ledger: an obligation disclosed by the auditor that is not booked as any \
transaction.
- fx: a foreign-currency amount and the dollar amount that actually settled it.
- addbacks: one-off items added back to EBITDA, with the materiality floor below which \
an item is not added back. List every item; the floor is applied later."""

OCR_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string",
                 "description": "the page transcribed, tables as one row per line"},
    },
    "required": ["text"], "additionalProperties": False,
}

OCR_SYSTEM = """Transcribe the page exactly. Keep every number, percentage and entity \
name as printed. Render a table as one row per line, cells separated by spaces, with \
the column header line kept. Do not summarise, translate or explain."""


def _clean(term: dict[str, Any]) -> dict[str, Any] | None:
    """Model output -> the internal term form, dropping nulls."""
    out: dict[str, Any] = {}
    if term.get("max_of"):
        out["max"] = [{"cat": c} for c in term["max_of"]]
    elif term.get("const") is not None:
        out["const"] = term["const"]
    elif term.get("related"):
        out["related"] = {}
    elif term.get("ebitda"):
        out["ebitda"] = {}
    elif term.get("cat"):
        out["cat"] = term["cat"]
        if term.get("quarter"):
            out["quarter"] = term["quarter"]
    else:
        return None
    if term.get("negate"):
        out["negate"] = True
    return out


def read_clause(chat: Chat, clause_text: str) -> dict[str, Any]:
    raw = chat.json(system=SPEC_SYSTEM, user=clause_text, schema=SPEC_SCHEMA)
    spec: dict[str, Any] = {
        "numerator": [t for t in (_clean(x) for x in raw.get("numerator", [])) if t],
        "direction": raw.get("direction", "max"),
        "threshold": raw.get("threshold"),
        "reasoning": raw.get("reasoning", ""),
    }
    denominator = [t for t in (_clean(x) for x in (raw.get("denominator") or [])) if t]
    if denominator:
        spec["denominator"] = denominator
    trigger = [t for t in (_clean(x) for x in (raw.get("springing_terms") or [])) if t]
    if trigger and raw.get("springing_above") is not None:
        spec["springing"] = {"terms": trigger, "above": raw["springing_above"]}
    return spec


def read_supplement(chat: Chat, supplement_text: str) -> dict[str, Any]:
    raw = chat.json(system=ADJ_SYSTEM, user=supplement_text, schema=ADJ_SCHEMA)
    block = {k: raw.get(k) or [] for k in
             ("reclass", "exclude", "amount_fix", "off_ledger", "fx")}
    if raw.get("addbacks") and raw["addbacks"].get("items"):
        block["addbacks"] = raw["addbacks"]
    block["ignored"] = raw.get("ignored") or []
    return {k: v for k, v in block.items() if v}


def read_page(chat: Chat, image: bytes, media_type: str = "image/png") -> str:
    raw = chat.json(system=OCR_SYSTEM, user="Transcribe this page.",
                    schema=OCR_SCHEMA, images=[(media_type, image)])
    return raw.get("text", "")
