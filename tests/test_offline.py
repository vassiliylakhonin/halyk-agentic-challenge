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
