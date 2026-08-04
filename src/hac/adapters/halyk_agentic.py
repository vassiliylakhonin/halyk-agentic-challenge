"""Competition format adapter — the only file that should need editing when the
organiser publishes the exact input and submission shapes.

Everything upstream works on (qid, question) pairs and QuestionResult objects.
This module owns two mappings and nothing else:

    load_questions(path)        -> [(qid, question_text), ...]
    build_submission(results)   -> the object serialised to submission.json

The loader already accepts the shapes such packs normally arrive in (a JSON list,
a JSON object keyed by id, JSONL, or one question per line). Verify it against
the open dataset on 6 August and freeze it; the private set will use the same
shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schema import QuestionResult

ID_KEYS = ("id", "qid", "question_id", "question_number", "number", "index", "no")
TEXT_KEYS = ("question", "text", "query", "prompt", "q", "question_text")


def _pick(d: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _from_obj(obj: Any, fallback_id: int) -> tuple[str, str] | None:
    if isinstance(obj, str):
        return str(fallback_id), obj
    if isinstance(obj, dict):
        text = _pick(obj, TEXT_KEYS)
        if text is None:
            return None
        qid = _pick(obj, ID_KEYS)
        return str(qid if qid is not None else fallback_id), str(text)
    return None


def load_questions(path: str | Path) -> list[tuple[str, str]]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    out: list[tuple[str, str]] = []

    if p.suffix.lower() == ".jsonl":
        for i, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if line:
                item = _from_obj(json.loads(line), i)
                if item:
                    out.append(item)
        return out

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [(str(i), line.strip())
                for i, line in enumerate(raw.splitlines(), start=1) if line.strip()]

    if isinstance(data, dict):
        for key in ("questions", "data", "items", "rows"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            # object keyed by question id
            for i, (k, v) in enumerate(data.items(), start=1):
                item = _from_obj(v, i)
                if item:
                    out.append((str(k), item[1]))
            return out

    if isinstance(data, list):
        for i, obj in enumerate(data, start=1):
            item = _from_obj(obj, i)
            if item:
                out.append(item)
    return out


def build_submission(results: list[QuestionResult]) -> Any:
    """Internal results -> the file the organiser scores.

    Shape below is the defensible default: the answer, the single extracted value,
    and page-level references. Replace the key names with the published ones once
    they exist; keep `references`, since evidence is a scored criterion.
    """
    answers = []
    for r in sorted(results, key=lambda x: _sort_key(x.qid)):
        rec = r.record
        refs = [
            {
                "document": c.doc_id,
                "page": c.page_start,
                "quote": c.quote,
            }
            for c in r.citations[:12]
        ]
        answers.append({
            "question_id": r.qid,
            "question": r.question,
            "answer": (rec.answer if rec else "") or "",
            "value": (rec.value if rec else "") or "",
            "reasoning_steps": [
                {"label": s.label, "expression": s.expression, "value": s.value}
                for s in (rec.steps if rec else [])
            ],
            "references": refs,
            "confidence": (rec.confidence if rec else 0.0),
        })
    return {"answers": answers}


def _sort_key(qid: str):
    try:
        return (0, int(qid), "")
    except (TypeError, ValueError):
        return (1, 0, str(qid))


def write_submission(results: list[QuestionResult], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(build_submission(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
