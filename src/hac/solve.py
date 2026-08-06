"""The competition-day path: a folder in, a submission out.

Written for a three-hour window with no second attempt, so the order of concerns
is: never crash, never leave a cell empty, never lose work already done.

  * every stage is wrapped - a document that will not parse, a clause the model
    will not read, a supplement that returns nonsense, all degrade to the
    rule-based reading rather than to nothing
  * the submission is rewritten to disk after every borrower, so a process that
    dies at minute eighty leaves a valid file behind
  * an empty cell and a wrong cell score the same, so a cell is never left null
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import audit, providers
from .bank import (Ledger, PdfDoc, extract_pdfs, link_documents,
                   pages_needing_ocr, refresh_from_text)
from .categories import Category
from .covenants import read_clause as rule_read_clause
from .extract import read_clause, read_page, read_supplement
from .kyc import is_kyc, parse_kyc
from .pipeline import agreement_for
from .spec import Context, decisive_txn, evaluate_spec
from .vote import consensus, repeat, spec_consensus

SUPPLEMENT_MARKER = "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ"
PROCEDURES_MARKERS = ("согласованных процедур", "ПРОМЕЖУТОЧНАЯ ВЕДОМОСТЬ")

# Used only when everything else has failed, so that no cell goes out empty.
LAST_RESORT = {"6.1": "BREACH", "6.2": "COMPLIANT", "6.3": "COMPLIANT"}


@dataclass
class Trace:
    scenario: str
    covenant: str
    source: str = ""          # model | rules | fallback
    spec: dict | None = None
    note: str = ""


@dataclass
class Report:
    traces: list[Trace] = field(default_factory=list)
    adjustments: dict = field(default_factory=dict)
    ocr_pages: int = 0
    failures: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> dict:
        by_source: dict[str, int] = {}
        for t in self.traces:
            by_source[t.source] = by_source.get(t.source, 0) + 1
        return {
            "cells": len(self.traces),
            "by_source": by_source,
            "ocr_pages": self.ocr_pages,
            "failures": self.failures,
            "seconds": round(self.seconds, 1),
        }


def render_page(path: str, page_number: int) -> bytes | None:
    try:
        import io

        import pypdfium2 as pdfium

        page = pdfium.PdfDocument(path)[page_number - 1]
        buffer = io.BytesIO()
        page.render(scale=2).to_pil().save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:
        return None


def recover_pages(chat: providers.Chat, docs: list[PdfDoc], report: Report) -> None:
    for doc in docs:
        blanks = pages_needing_ocr(doc.pages)
        if not blanks:
            continue
        for n in blanks:
            image = render_page(doc.path, n)
            if image is None:
                continue
            try:
                doc.pages[n - 1] = read_page(chat, image)
                report.ocr_pages += 1
            except Exception as e:
                report.failures.append(f"ocr {doc.doc_id} p{n}: {type(e).__name__}")
        refresh_from_text(doc)


def auditor_text(docs: list[PdfDoc], scenario: str) -> str:
    """Everything about a borrower except the agreements themselves.

    Adjustments do not only live in the auditor's supplement. A treasury memo
    carries an amount that never reached the ledger dump, and a covenant can
    point at it explicitly. Anything that is not the contract is short, so all
    of it goes to the reader rather than a guessed subset - a document type that
    turns up for the first time on competition day is then already covered.
    """
    parts: list[str] = []
    for d in docs:
        if d.scenario != scenario or d.superseded:
            continue
        if SUPPLEMENT_MARKER in d.text:
            tail = d.text.split(SUPPLEMENT_MARKER, 1)[1]
            parts.append(f"<<< {d.doc_id} · covenant supplement >>>\n"
                         + tail.split("За аудитора")[0][:8000])
        elif d.has_covenants and d.n_pages > 10:
            continue                      # the agreement is read clause by clause
        else:
            parts.append(f"<<< {d.doc_id} >>>\n" + " ".join(d.text.split())[:6000])
    return "\n\n---\n\n".join(parts)


def kyc_profile(docs: list[PdfDoc], scenario: str):
    for d in docs:
        if d.scenario == scenario and is_kyc(d.text):
            try:
                return parse_kyc(d.text)
            except Exception:
                return None
    return None


def solve(
    docs_dir: str | Path,
    ledger_path: str | Path,
    template_path: str | Path,
    out_path: str | Path,
    *,
    provider: str = "openai",
    model: str | None = None,
    samples: int = 1,
    team: str = "",
    contact_email: str = "",
    text_cache: str | Path | None = None,
) -> Report:
    started = time.monotonic()
    report = Report()

    usage = providers.Usage()
    try:
        chat: providers.Chat | None = providers.build(provider, model, usage)
        model_name = chat.name
    except Exception as e:
        chat, model_name = None, "rules-only"
        report.failures.append(f"no model access: {type(e).__name__}: {e}")

    ledger = Ledger.load(ledger_path)
    docs = link_documents(extract_pdfs(docs_dir, text_cache, None), ledger)
    if chat is not None:
        recover_pages(chat, docs, report)
        # Linking is redone: an account number can live inside a scanned page.
        docs = link_documents(docs, ledger)

    template = json.loads(Path(template_path).read_text(encoding="utf-8"))
    by_scenario = ledger.by_scenario()

    submission: dict[str, Any] = {
        "team": team,
        "contact_email": contact_email,
        "model": model_name,
        "answers": {},
    }

    def flush() -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")

    for scenario, cells in template.get("answers", {}).items():
        agreement = agreement_for(scenario, docs)
        txns = by_scenario.get(scenario, [])

        block: dict = {}
        if chat is not None:
            text = auditor_text(docs, scenario)
            if text.strip():
                readings = repeat(
                    lambda: read_supplement(chat, text), samples,
                    lambda e: report.failures.append(
                        f"{scenario} supplement: {type(e).__name__}: {e}"))
                agreed, share = consensus(readings)
                block = agreed or (readings[0] if readings else {})
                if readings and share < 1.0:
                    report.failures.append(
                        f"{scenario} supplement agreed by {share:.0%} of readings")

        adj = audit.build(block, txns)
        report.adjustments[scenario] = block
        ctx = Context(txns=txns, profile=kyc_profile(docs, scenario),
                      overrides=adj.overrides)

        answers: dict[str, dict] = {}
        for covenant in cells:
            trace = Trace(scenario=scenario, covenant=covenant)
            clause = agreement.clause(covenant) if agreement else ""
            spec: dict | None = None

            if chat is not None and clause:
                readings = repeat(
                    lambda: read_clause(chat, clause), samples,
                    lambda e: report.failures.append(
                        f"{scenario} {covenant} clause: {type(e).__name__}: {e}"))
                candidate, share, how = spec_consensus(readings)
                if candidate and candidate.get("numerator"):
                    spec, trace.source = candidate, "model"
                    trace.note = f"{how} ({share:.0%})"

            if spec is None and clause:
                fallback = rule_read_clause(clause)
                if fallback is not None:
                    spec = _from_rule_spec(fallback)
                    trace.source = "rules"

            result = None
            if spec:
                run = dict(spec)
                extra: list[dict] = []
                if adj.add_back_total():
                    extra.append({"const": adj.add_back_total()})
                extra += [{"const": o["amount"]} for o in adj.off_ledger]
                if extra and covenant == "6.1":
                    run["numerator"] = list(spec["numerator"]) + extra
                try:
                    result = evaluate_spec(run, ctx)
                    evidence = decisive_txn(run, ctx, result)
                except Exception as e:
                    result, evidence = None, None
                    report.failures.append(
                        f"{scenario} {covenant} evaluate: {type(e).__name__}: {e}")
            else:
                evidence = None

            if result is None or result.status is None:
                trace.source = trace.source or "fallback"
                trace.note = "no metric could be computed"
                answers[covenant] = {
                    "status": LAST_RESORT.get(covenant, "COMPLIANT"),
                    "actual": 0.0,
                    "evidence_txn_id": None,
                }
            else:
                answers[covenant] = {
                    "status": result.status,
                    "actual": round(result.actual, 2)
                    if result.actual is not None else 0.0,
                    "evidence_txn_id": evidence,
                }
            trace.spec = spec
            report.traces.append(trace)

        submission["answers"][scenario] = answers
        flush()

    flush()
    report.seconds = time.monotonic() - started
    return report


def _from_rule_spec(spec) -> dict:
    """The rule reader's dataclass, expressed in the JSON specification form."""
    numerator: list[dict] = []
    denominator: list[dict] = []
    if spec.family in ("related_party_abs", "related_party_ratio"):
        numerator = [{"related": {}}]
        if spec.family == "related_party_ratio":
            denominator = [{"cat": Category.REVENUE.value}]
    elif spec.family == "max_single_line":
        numerator = [{"max": [{"cat": c.value} for c in spec.lines]}]
    elif spec.family == "category_total" and spec.category is not None:
        numerator = [{"cat": spec.category.value}]
    out: dict[str, Any] = {
        "numerator": numerator,
        "direction": spec.direction,
        "threshold": spec.threshold,
    }
    if denominator:
        out["denominator"] = denominator
    return out
