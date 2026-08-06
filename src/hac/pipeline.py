"""End to end: pack in, submission out."""

from __future__ import annotations

import json
from pathlib import Path

from .bank import Ledger, PdfDoc, agreements_by_scenario, extract_pdfs, link_documents
from .covenants import Outcome, evaluate, read_clause
from .kyc import KycProfile, is_kyc, parse_kyc


def agreement_for(scenario: str, docs: list[PdfDoc]) -> PdfDoc | None:
    """The operative agreement: not a superseded edition, and long enough to be
    the contract rather than a dossier that merely cites a clause number."""
    ag = agreements_by_scenario(docs).get(scenario, {"current": []})
    current = sorted(ag["current"], key=lambda d: -d.n_pages)
    return current[0] if current else None


def kyc_for(scenario: str, docs: list[PdfDoc]) -> KycProfile | None:
    hits = [d for d in docs if d.scenario == scenario and is_kyc(d.text)]
    return parse_kyc(hits[0].text) if hits else None


def run_pack(
    docs_dir: str | Path,
    ledger_path: str | Path,
    template_path: str | Path,
    *,
    text_cache: str | Path | None = None,
    ocr_cache: str | Path | None = None,
    team: str = "",
    contact_email: str = "",
    model: str = "",
) -> tuple[dict, dict[tuple[str, str], Outcome]]:
    ledger = Ledger.load(ledger_path)
    docs = link_documents(extract_pdfs(docs_dir, text_cache, ocr_cache), ledger)
    template = json.loads(Path(template_path).read_text(encoding="utf-8"))
    by_scenario = ledger.by_scenario()

    submission = {
        "team": team,
        "contact_email": contact_email,
        "model": model,
        "answers": {},
    }
    outcomes: dict[tuple[str, str], Outcome] = {}

    for scenario, cells in template["answers"].items():
        agreement = agreement_for(scenario, docs)
        profile = kyc_for(scenario, docs)
        txns = by_scenario.get(scenario, [])
        answers: dict[str, dict] = {}

        for covenant in cells:
            clause = agreement.clause(covenant) if agreement else ""
            spec = read_clause(clause) if clause else None
            out = (evaluate(spec, txns, profile) if spec
                   else Outcome(actual=None, status=None, note="clause not read"))
            outcomes[(scenario, covenant)] = out
            answers[covenant] = {
                "status": out.status,
                "actual": None if out.actual is None else round(out.actual, 2),
                "evidence_txn_id": out.evidence_txn_id,
            }
        submission["answers"][scenario] = answers

    return submission, outcomes
