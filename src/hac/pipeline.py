"""Choosing which document of a borrower to read from."""

from __future__ import annotations

from .bank import Ledger, PdfDoc, agreements_by_scenario, extract_pdfs, link_documents
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
