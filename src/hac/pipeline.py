"""Choosing which document of a borrower to read from."""

from __future__ import annotations

from .bank import PdfDoc, agreements_by_scenario, clause_numbers
from .kyc import KycProfile, is_kyc, parse_kyc


def agreement_for(scenario: str, docs: list[PdfDoc],
                  wanted: set[str] | None = None) -> PdfDoc | None:
    """The operative agreement: not a superseded edition, and the document that
    prints most of the clauses being asked about rather than a dossier that
    merely cites a clause number."""
    ag = agreements_by_scenario(docs, wanted).get(scenario, {"current": []})
    if not ag["current"]:
        return None
    if wanted:
        return max(ag["current"],
                   key=lambda d: (len(clause_numbers(d.text) & wanted), d.n_pages))
    return max(ag["current"], key=lambda d: d.n_pages)


def kyc_for(scenario: str, docs: list[PdfDoc]) -> KycProfile | None:
    hits = [d for d in docs if d.scenario == scenario and is_kyc(d.text)]
    return parse_kyc(hits[0].text) if hits else None
