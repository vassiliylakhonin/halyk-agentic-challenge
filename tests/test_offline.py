"""Everything here runs without an API key. These are the parts that must not
break under time pressure on competition day."""

from __future__ import annotations

import pytest

from hac import audit
from hac.audit import Adjustments
from hac.bank import PdfDoc, Txn, parse_amount, refresh_from_text
from hac.categories import Category, classify
from hac.kyc import KycProfile, Owner, normalize_entity, parse_kyc
from hac.score import score_cell
from hac.spec import Context, category_total, evaluate_spec, related_total
from hac.vote import cell_consensus, consensus, spec_consensus


# ----------------------------------------------------------------- ledger

def test_dirty_amounts_parse_or_report_themselves_missing():
    assert parse_amount("-366837.86") == pytest.approx(-366837.86)
    assert parse_amount("(1 234.50)") == pytest.approx(-1234.50)
    assert parse_amount("1,234,567.89") == pytest.approx(1234567.89)
    assert parse_amount("") is None
    assert parse_amount("n/a") is None


# ---------------------------------------------------------------- dossier

def test_entity_normalisation_collapses_legal_forms():
    n = normalize_entity
    assert n("Atyrau Holding Group L.L.P.") == n("Atyrau Holding Group LLP")
    assert n("Aktau Holdings LLP") == n("Aktau Holdings L.L.P.")
    assert n("Ertis Capital, LLP") == n("Ertis Capital LLP")
    assert n("Hartley Building Services (Turkistan point)") == n("Hartley Building Services")
    assert n("Aktau Holdings LLP") != n("Aktau Terminal Properties LLP")


def test_kyc_reads_threshold_and_excludes_holdings_below_it():
    text = (
        "Досье «Знай своего клиента» (KYC)\n"
        "Организация Доля голосующих прав\n"
        "Aktau Holdings LLP 34.5%\n"
        "Kaspi Marine Engineering LLP 18.7%\n"
        "Ural Crane Works LLP 6.2%\n"
        "Организации, в которых Группа владеет 20.0% и более голосующих прав, "
        "признаются связанными сторонами для целей Договора.\n"
        "Идентификация и проверка сведений\n"
    )
    profile = parse_kyc(text)
    assert profile.threshold == 20.0
    assert len(profile.owners) == 3
    assert [o.name for o in profile.related] == ["Aktau Holdings LLP"]


# ------------------------------------------------------------- categories

def test_description_decides_the_category_and_the_counterparty_never_does():
    assert classify("Antenna mast lease — Pavlodar block",
                    "Bridgeport Payroll Group")[0] is Category.LEASE
    assert classify("Social tax remittance", "Bridgeport Payroll Group")[0] is Category.TAX


def test_the_planted_category_traps():
    assert classify("Purchase of grain conveyor equipment")[0] is Category.CAPEX
    assert classify("Flood remediation and silo repair works")[0] is Category.OPEX
    assert classify("Capitalised interest charge 2025")[0] is Category.INTEREST
    assert classify("Interest on finance sublease")[0] is Category.INTEREST
    assert classify("Outdoor marketing site hire")[0] is Category.MARKETING
    assert classify("Payroll advance recovered from staff")[0] is Category.NON_OPERATING
    assert classify("Term loan facility drawdown")[0] is Category.FINANCING


# --------------------------------------------------- related parties, lines

def _context() -> Context:
    txns = [
        Txn("T1", "2025-02-01", "A", "Plant Services LLP",
            "Plant operating and maintenance expenses", -6166592.66, "USD"),
        Txn("T2", "2025-03-01", "A", "Ertis Capital LLP",
            "Management advisory retainer", -307018.08, "USD"),
        Txn("T3", "2025-04-01", "A", "Ertis Capital LLP",
            "Purchase of crusher equipment", -1000000.0, "USD"),
    ]
    profile = KycProfile(threshold=20.0, owners=[Owner("Ertis Capital LLP", 31.4)])
    return Context(txns=txns, profile=profile)


def test_related_party_payments_leave_the_operating_lines_alone():
    ctx = _context()
    opex, _ = category_total(ctx, Category.OPEX)
    related, _ = related_total(ctx)
    assert opex == pytest.approx(6166592.66)
    assert related == pytest.approx(1307018.08)


def test_capital_expenditure_keeps_related_party_assets():
    capex, _ = category_total(_context(), Category.CAPEX)
    assert capex == pytest.approx(1000000.0)


# -------------------------------------------------------------- evaluator

def test_a_springing_test_does_not_apply_until_it_is_triggered():
    txns = [Txn("T1", "2025-01-01", "A", "Bank", "Term loan facility drawdown",
                1000.0, "USD")]
    spec = {"numerator": [{"cat": "financing"}], "direction": "max", "threshold": 0.5,
            "springing": {"terms": [{"cat": "financing"}], "above": 5000.0}}
    result = evaluate_spec(spec, Context(txns=txns))
    assert result.status == "COMPLIANT"
    assert "not triggered" in result.note


def test_a_quarter_restriction_selects_only_that_quarter():
    txns = [
        Txn("T1", "2025-03-31", "A", "X", "Handling sales settlement", 100.0, "USD"),
        Txn("T2", "2025-11-30", "A", "X", "Handling sales settlement", 400.0, "USD"),
    ]
    spec = {"numerator": [{"cat": "revenue", "quarter": 4}],
            "direction": "min", "threshold": 350.0}
    result = evaluate_spec(spec, Context(txns=txns))
    assert result.actual == pytest.approx(400.0)
    assert result.status == "COMPLIANT"


# --------------------------------------------------- auditor adjustments

def test_audit_reclassification_moves_a_line_and_marks_it():
    txns = [
        Txn("TXN-X-1", "2025-03-01", "ACC-1", "Irtysh Advisory Bureau",
            "Advisory retainer", -592296.10, "USD"),
        Txn("TXN-X-2", "2025-04-01", "ACC-1", "Other LLP", "Office rent", -1000.0, "USD"),
    ]
    adj = audit.build(
        {"reclass": [{"amount": 592296.10, "counterparty": "Irtysh Advisory Bureau",
                      "to": "interest"}]}, txns)
    assert adj.overrides["TXN-X-1"]["category"] == "interest"
    assert adj.touched == ["TXN-X-1"]


def test_addbacks_respect_the_materiality_floor():
    adj = Adjustments(addbacks={"materiality": 300000.0,
                                "items": [251338.94, 342905.28, 481247.63]})
    assert adj.add_back_total() == pytest.approx(824152.91)


# --------------------------------------------------------------- consensus

def test_consensus_ignores_wording_and_counts_substance():
    a = {"reclass": [{"amount": 1.0, "to": "opex", "why": "one wording"}],
         "ignored": ["a long explanation"]}
    b = {"reclass": [{"amount": 1.0, "to": "opex", "why": "another wording"}],
         "ignored": ["a different explanation"]}
    c = {"reclass": [{"amount": 2.0, "to": "tax"}], "ignored": []}
    agreed, share = consensus([a, b, c])
    assert agreed["reclass"][0]["amount"] == 1.0
    assert share == pytest.approx(2 / 3)


def test_spec_consensus_falls_back_to_threshold_and_direction():
    a = {"numerator": [{"cat": "capex"}], "direction": "max", "threshold": 0.42}
    b = {"numerator": [{"cat": "opex"}], "direction": "max", "threshold": 0.42}
    c = {"numerator": [{"cat": "lease"}], "direction": "min", "threshold": 1.0}
    spec, share, how = spec_consensus([a, b, c])
    assert spec["threshold"] == 0.42 and spec["direction"] == "max"
    assert "threshold and direction" in how


def test_cell_consensus_takes_the_median_of_the_majority():
    out = cell_consensus([
        {"status": "BREACH", "actual": 1.70, "evidence_txn_id": "T1"},
        {"status": "BREACH", "actual": 1.68, "evidence_txn_id": "T1"},
        {"status": "COMPLIANT", "actual": 9.0, "evidence_txn_id": None},
    ])
    assert out["status"] == "BREACH"
    assert out["actual"] == pytest.approx(1.69)
    assert out["evidence_txn_id"] == "T1"


# ----------------------------------------------------------------- scoring

def test_scorer_matches_the_published_formula():
    key = {"status": "BREACH", "actual": 100.0, "evidence_txn_id": None}
    assert score_cell({"status": "COMPLIANT", "actual": 100.0}, key, "S", "6.1").score == 0.0
    assert score_cell({"status": "BREACH", "actual": 100.0}, key, "S", "6.1").score == 1.0
    half = score_cell({"status": "BREACH", "actual": 102.5}, key, "S", "6.1").score
    assert half == pytest.approx(0.5 + 0.30 * 0.5 + 0.20 * 0.5)
    assert score_cell({"status": "BREACH", "actual": 105.0}, key, "S", "6.1").score == 0.5


# ---------------------------------------------------------- recovered pages

def test_recovered_pages_restore_the_derived_fields():
    doc = PdfDoc(doc_id="d", path="d.pdf", n_pages=1, text="", pages=[""])
    assert doc.accounts == []
    doc.pages[0] = "Досье KYC · Счёт ACC-7806 · пункт 6.1"
    refresh_from_text(doc)
    assert doc.accounts == ["ACC-7806"]
    assert doc.has_covenants is True
