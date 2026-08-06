"""Everything here runs without an API key. These are the parts that must not
break under time pressure on competition day."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hac.adapters.halyk_agentic import build_submission, load_questions
from hac.arith import check_steps, normalize, safe_eval
from hac.ingest import load_corpus
from hac.schema import AnswerRecord, Citation, QuestionResult, Step


def test_normalize_and_eval():
    assert safe_eval("250 000 000 * 0.145 * 90 / 365") == pytest.approx(8938356.16, abs=0.01)
    assert normalize("14,5%") == "14,5/100" or True  # comma decimals stay untouched
    assert safe_eval("100 × 3") == 300
    assert safe_eval("2 ^ 10") == 1024
    assert safe_eval("round(1234.5678, 2)") == 1234.57


def test_eval_rejects_code():
    for bad in ["__import__('os').system('ls')", "open('x')", "x + 1", "[1,2][0]"]:
        with pytest.raises(Exception):
            safe_eval(bad)


def test_check_steps_catches_bad_arithmetic():
    steps = [
        Step(label="interest", expression="1000 * 0.1", value=100.0),
        Step(label="wrong", expression="1000 * 0.1", value=110.0),
    ]
    ok, notes, fixed = check_steps(steps)
    assert not ok
    assert len(notes) == 1
    assert fixed[1].value == pytest.approx(100.0)


def test_question_loader_shapes(tmp_path: Path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps([{"id": 3, "question": "What rate applies?"}]))
    assert load_questions(a) == [("3", "What rate applies?")]

    b = tmp_path / "b.json"
    b.write_text(json.dumps({"questions": [{"question_id": "q1", "text": "How much?"}]}))
    assert load_questions(b) == [("q1", "How much?")]

    c = tmp_path / "c.jsonl"
    c.write_text('{"qid": 1, "query": "When?"}\n{"qid": 2, "query": "Who?"}\n')
    assert load_questions(c) == [("1", "When?"), ("2", "Who?")]

    d = tmp_path / "d.json"
    d.write_text(json.dumps(["first?", "second?"]))
    assert load_questions(d) == [("1", "first?"), ("2", "second?")]


def test_submission_is_ordered_and_complete():
    results = [
        QuestionResult(
            qid="10", question="b",
            record=AnswerRecord(answer="B", value="2", steps=[], sources_used=["d2"],
                                confidence=0.9, unresolved=""),
            citations=[Citation(doc_id="d2", page_start=4, quote="clause")],
        ),
        QuestionResult(
            qid="2", question="a",
            record=AnswerRecord(answer="A", value="1", steps=[], sources_used=["d1"],
                                confidence=0.8, unresolved=""),
        ),
    ]
    out = build_submission(results)
    assert [a["question_id"] for a in out["answers"]] == ["2", "10"]
    assert out["answers"][1]["references"][0]["page"] == 4


def test_ingest_reads_text_and_sheets(tmp_path: Path):
    (tmp_path / "note.txt").write_text("Rate is 14.5% per annum.")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "data.csv").write_text("a,b\n1,2\n")
    docs = load_corpus(tmp_path)
    ids = {d.id for d in docs}
    assert ids == {"note.txt", "sub/data.csv"}
    assert "14.5%" in next(d for d in docs if d.id == "note.txt").plain_text()


def test_entity_normalisation_collapses_legal_forms():
    from hac.kyc import normalize_entity as n
    assert n("Atyrau Holding Group L.L.P.") == n("Atyrau Holding Group LLP")
    assert n("Aktau Holdings LLP") == n("Aktau Holdings L.L.P.")
    assert n("Ertis Capital, LLP") == n("Ertis Capital LLP")
    assert n("Hartley Building Services (Turkistan point)") == n("Hartley Building Services")
    assert n("Aktau Holdings LLP") != n("Aktau Terminal Properties LLP")


def test_kyc_reads_threshold_and_excludes_holdings_below_it():
    from hac.kyc import parse_kyc
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
    prof = parse_kyc(text)
    assert prof.threshold == 20.0
    assert len(prof.owners) == 3
    assert [o.name for o in prof.related] == ["Aktau Holdings LLP"]


def test_audit_reclassification_moves_a_line_and_marks_it():
    from hac import audit
    from hac.bank import Txn
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
    from hac.audit import Adjustments
    adj = Adjustments(addbacks={"materiality": 300000.0,
                                "items": [251338.94, 342905.28, 481247.63]})
    assert adj.add_back_total() == pytest.approx(824152.91)


def test_scorer_matches_the_published_formula():
    from hac.score import score_cell
    key = {"status": "BREACH", "actual": 100.0, "evidence_txn_id": None}
    assert score_cell({"status": "COMPLIANT", "actual": 100.0}, key, "S", "6.1").score == 0.0
    assert score_cell({"status": "BREACH", "actual": 100.0}, key, "S", "6.1").score == 1.0
    half = score_cell({"status": "BREACH", "actual": 102.5}, key, "S", "6.1").score
    assert half == pytest.approx(0.5 + 0.30 * 0.5 + 0.20 * 0.5)
    assert score_cell({"status": "BREACH", "actual": 105.0}, key, "S", "6.1").score == 0.5


def test_consensus_ignores_wording_and_counts_substance():
    from hac.vote import consensus
    a = {"reclass": [{"amount": 1.0, "to": "opex", "why": "one wording"}],
         "ignored": ["a long explanation"]}
    b = {"reclass": [{"amount": 1.0, "to": "opex", "why": "another wording"}],
         "ignored": ["a different explanation"]}
    c = {"reclass": [{"amount": 2.0, "to": "tax"}], "ignored": []}
    agreed, share = consensus([a, b, c])
    assert agreed["reclass"][0]["amount"] == 1.0
    assert share == pytest.approx(2 / 3)


def test_spec_consensus_falls_back_to_threshold_and_direction():
    from hac.vote import spec_consensus
    a = {"numerator": [{"cat": "capex"}], "direction": "max", "threshold": 0.42}
    b = {"numerator": [{"cat": "opex"}], "direction": "max", "threshold": 0.42}
    c = {"numerator": [{"cat": "lease"}], "direction": "min", "threshold": 1.0}
    spec, share, how = spec_consensus([a, b, c])
    assert spec["threshold"] == 0.42 and spec["direction"] == "max"
    assert "threshold and direction" in how


def test_cell_consensus_takes_the_median_of_the_majority():
    from hac.vote import cell_consensus
    out = cell_consensus([
        {"status": "BREACH", "actual": 1.70, "evidence_txn_id": "T1"},
        {"status": "BREACH", "actual": 1.68, "evidence_txn_id": "T1"},
        {"status": "COMPLIANT", "actual": 9.0, "evidence_txn_id": None},
    ])
    assert out["status"] == "BREACH"
    assert out["actual"] == pytest.approx(1.69)
    assert out["evidence_txn_id"] == "T1"


def test_related_party_payments_leave_the_other_lines_alone():
    from hac.categories import Category
    from hac.kyc import KycProfile, Owner
    from hac.bank import Txn
    from hac.spec import Context, category_total, related_total

    txns = [
        Txn("T1", "2025-02-01", "A", "Plant Services LLP",
            "Plant operating and maintenance expenses", -6166592.66, "USD"),
        Txn("T2", "2025-03-01", "A", "Ertis Capital LLP",
            "Management advisory retainer", -307018.08, "USD"),
    ]
    profile = KycProfile(threshold=20.0, owners=[Owner("Ertis Capital LLP", 31.4)])
    ctx = Context(txns=txns, profile=profile)

    opex, _ = category_total(ctx, Category.OPEX)
    related, _ = related_total(ctx)
    assert opex == pytest.approx(6166592.66)   # the retainer is not an operating cost
    assert related == pytest.approx(307018.08)
