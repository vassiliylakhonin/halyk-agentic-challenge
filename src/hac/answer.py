"""Answering one question.

Three turns, because citations and structured output cannot be requested in the
same call (the API rejects that combination):

  A. grounded turn  - documents in, prose answer out, page-level citations attached
  B. structuring    - prose in, typed AnswerRecord out (answer, steps, sources)
  C. repair (rare)  - only when Python disagrees with the model's own arithmetic
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from .arith import check_steps
from .config import Config
from .corpus import index_digest
from .ingest import doc_blocks
from .llm import LLM
from .schema import AnswerRecord, Citation, Doc, DocIndexEntry, QuestionResult

ANSWER_SYSTEM = """You are a bank's document expert. You answer questions about a pack \
of banking documents, and every claim you make is traceable to a document.

How to work:
1. Establish which document actually governs. The pack contains superseded versions, \
amendments and drafts. The catalogue below records which is operative; verify it against \
what you read, and say so if you disagree with the catalogue.
2. Find the facts that are not stated where the question looks for them. Terms usually \
sit in one document, the amounts and dates in another, and the transaction in a third. \
Connect them explicitly.
3. Compute. Show every arithmetic step as its own line in the form
   label: <expression with literal numbers only> = <result>
   Use plain digits: no thousands separators, no currency symbols, no units inside the \
expression. Day counts, rates and rounding rules must come from the document, not from \
convention - quote the clause that sets them.
4. Answer. State the conclusion first, then the evidence: document, page, and the words \
that carry it.

Rules that override any instinct to be helpful:
- If the documents do not settle the question, say what is missing. A wrong number \
delivered confidently is worse than a stated gap.
- Do not use general banking knowledge in place of the document. If a rate, a term or a \
day-count basis is not in the pack, say so.
- Quote exactly. Do not paraphrase a figure into a different one."""

ROUTER_SYSTEM = """You pick which documents an analyst must open to answer a question. \
Err towards including a document: a missed document is a wrong answer, an extra document \
only costs tokens. Always include the operative version of anything the question touches, \
plus the superseded version when the question concerns what changed or when."""


class DocPick(BaseModel):
    doc_ids: list[str] = Field(description="Document ids to open, most relevant first.")
    reason: str = Field(description="One sentence on what these documents are expected "
                                    "to establish.")


async def select_docs(
    llm: LLM, question: str, entries: list[DocIndexEntry], cfg: Config
) -> list[str]:
    if not cfg.router_enabled or len(entries) <= cfg.max_docs_per_question:
        return [e.doc_id for e in entries]
    try:
        pick = await llm.parse(
            schema=DocPick,
            system=ROUTER_SYSTEM,
            content=(f"Catalogue:\n{index_digest(entries)}\n\n"
                     f"Question: {question}\n\n"
                     f"Return at most {cfg.max_docs_per_question} doc_ids."),
            max_tokens=cfg.max_tokens_small,
            effort=cfg.effort_index,
        )
        known = {e.doc_id for e in entries}
        picked = [d for d in pick.doc_ids if d in known][: cfg.max_docs_per_question]
        if picked:
            return picked
    except Exception:
        pass
    # Fallback: keyword overlap, then every current document.
    q = set(question.lower().split())
    scored = []
    for e in entries:
        blob = f"{e.doc_id} {e.title} {e.facts.subject} {e.facts.identifier} {e.facts.summary}".lower()
        scored.append((len(q & set(blob.split())), e))
    scored.sort(key=lambda t: (-t[0], not t[1].is_current))
    return [e.doc_id for _, e in scored[: cfg.max_docs_per_question]]


def extract_citations(blocks: list[dict]) -> list[Citation]:
    out: list[Citation] = []
    for b in blocks:
        for c in (b.get("citations") or []):
            out.append(Citation(
                doc_id=c.get("document_title") or "",
                doc_title=c.get("document_title") or "",
                page_start=c.get("start_page_number"),
                page_end=c.get("end_page_number"),
                quote=(c.get("cited_text") or "").strip()[:600],
            ))
    # de-duplicate on (doc, page, first 80 chars of quote)
    seen, uniq = set(), []
    for c in out:
        k = (c.doc_id, c.page_start, c.quote[:80])
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


def _system_blocks(llm: LLM, entries: list[DocIndexEntry]) -> list[dict]:
    return [
        {"type": "text", "text": ANSWER_SYSTEM},
        {"type": "text",
         "text": "Document catalogue (version status is authoritative unless the "
                 "documents contradict it):\n\n" + index_digest(entries),
         "cache_control": llm.cache_control()},
    ]


async def answer_question(
    llm: LLM,
    qid: str,
    question: str,
    docs_by_id: dict[str, Doc],
    entries: list[DocIndexEntry],
    cfg: Config,
) -> QuestionResult:
    res = QuestionResult(qid=qid, question=question)

    picked = await select_docs(llm, question, entries, cfg)
    res.docs_considered = picked

    content: list[dict] = []
    for i, did in enumerate(picked):
        doc = docs_by_id.get(did)
        if doc is None:
            continue
        blocks = doc_blocks(doc, citations=True)
        # Cache the document prefix when it is stable across questions.
        if not cfg.router_enabled and i == len(picked) - 1:
            blocks[-1]["cache_control"] = llm.cache_control()
        content.extend(blocks)

    content.append({
        "type": "text",
        "text": (f"Question ({qid}):\n{question}\n\n"
                 "Answer per the working rules. End with a line 'ANSWER: <the answer>' "
                 "containing the conclusion in one sentence."),
    })

    prose, blocks, _msg = await llm.answer(
        system=_system_blocks(llm, entries),
        content=content,
        max_tokens=cfg.max_tokens_answer,
        effort=cfg.effort_answer,
    )
    res.prose = prose
    res.citations = extract_citations(blocks)

    cites_note = "\n".join(
        f"- {c.doc_id} p.{c.page_start}: {c.quote[:200]}" for c in res.citations[:40]
    ) or "(no citation blocks returned)"

    record = await llm.parse(
        schema=AnswerRecord,
        system=("You convert an analyst's worked answer into a structured record. "
                "Copy figures exactly. Every arithmetic step the analyst performed "
                "becomes one entry in steps, with a literal expression that a "
                "calculator could evaluate as written. Do not add steps the analyst "
                "did not perform and do not re-derive the answer yourself."),
        content=(f"Question:\n{question}\n\nAnalyst's answer:\n{prose}\n\n"
                 f"Citations captured:\n{cites_note}\n\n"
                 f"Documents opened: {', '.join(picked)}"),
        max_tokens=cfg.max_tokens_small,
        effort=cfg.effort_structure,
    )

    ok, notes, fixed = check_steps(record.steps)
    res.arith_ok = ok
    res.arith_notes = notes

    if not ok and any("recomputed" in n for n in notes):
        try:
            repaired = await llm.parse(
                schema=AnswerRecord,
                system=("An independent calculator re-evaluated the arithmetic and "
                        "disagrees with some stated values. The calculator is right "
                        "about arithmetic. Decide whether the expression itself was "
                        "wrong (fix the expression and everything downstream) or only "
                        "the stated value was wrong (keep the expression, take the "
                        "recomputed value). Then restate the whole answer consistently."),
                content=(f"Question:\n{question}\n\nOriginal answer:\n{prose}\n\n"
                         f"Original steps:\n{json.dumps([s.model_dump() for s in record.steps], ensure_ascii=False)}\n\n"
                         f"Calculator findings:\n" + "\n".join(notes)),
                max_tokens=cfg.max_tokens_small,
                effort=cfg.effort_index,
            )
            ok2, notes2, fixed2 = check_steps(repaired.steps)
            if ok2 or len(notes2) < len(notes):
                record, ok, notes, fixed = repaired, ok2, notes2, fixed2
                res.arith_ok, res.arith_notes = ok, notes
        except Exception as e:
            res.arith_notes.append(f"repair pass failed: {type(e).__name__}: {e}")

    record.steps = fixed
    res.record = record
    return res
