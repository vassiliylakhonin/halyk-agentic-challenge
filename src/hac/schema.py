"""Internal data model. Kept separate from the competition submission format,
which lives in adapters/ and can be rewritten in minutes once it is published."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DocKind = Literal["pdf", "image", "sheet", "docx", "text"]


class Page(BaseModel):
    n: int
    text: str = ""


class Doc(BaseModel):
    id: str
    path: str
    kind: DocKind
    title: str
    media_type: str | None = None
    n_pages: int = 1
    pages: list[Page] = Field(default_factory=list)
    bytes_len: int = 0

    def plain_text(self, max_chars: int | None = None) -> str:
        parts = [f"[[page {p.n}]]\n{p.text}" for p in self.pages if p.text.strip()]
        s = "\n\n".join(parts)
        return s if max_chars is None else s[:max_chars]


class DocFacts(BaseModel):
    """Header facts extracted once per document. Drives version resolution."""

    doc_type: str = Field(description="Kind of banking document, e.g. loan agreement, "
                                      "addendum, tariff schedule, statement, policy, invoice.")
    subject: str = Field(description="Entity or contract the document is about "
                                     "(borrower, counterparty, account, contract number). "
                                     "Empty string if not stated.")
    identifier: str = Field(description="Document/contract number as printed. Empty string if none.")
    effective_date: str = Field(description="Effective or signing date, ISO 8601 (YYYY-MM-DD) "
                                            "when stated. Empty string if not stated.")
    version_label: str = Field(description="Revision or version marker as printed "
                                           "(e.g. 'Addendum No. 2'). Empty string if none.")
    supersedes: str = Field(description="Identifier of the document this one replaces or amends, "
                                        "as stated in the text. Empty string if not stated.")
    key_numbers: list[str] = Field(description="Up to 10 salient figures with their labels, "
                                               "e.g. 'rate 14.5% p.a.', 'principal 250,000,000 KZT'.")
    summary: str = Field(description="Two sentences: what this document establishes and what "
                                     "another document would need it for.")


class DocIndexEntry(BaseModel):
    doc_id: str
    title: str
    kind: DocKind
    n_pages: int
    facts: DocFacts
    is_current: bool = True
    superseded_by: str | None = None
    note: str = ""


class Citation(BaseModel):
    doc_id: str = ""
    doc_title: str = ""
    page_start: int | None = None
    page_end: int | None = None
    quote: str = ""


class Step(BaseModel):
    """One arithmetic step, stated so it can be recomputed deterministically."""

    label: str = Field(description="What this step computes.")
    expression: str = Field(description="Pure arithmetic expression using literal numbers only, "
                                        "e.g. '250000000 * 0.145 * 90 / 365'. No variables, no units, "
                                        "no currency symbols, no thousands separators.")
    value: float = Field(description="The numeric result of the expression.")


class AnswerRecord(BaseModel):
    """Structured form of one answer. Pass B produces this from pass A prose."""

    answer: str = Field(description="The final answer, self-contained, no preamble.")
    value: str = Field(description="The answer reduced to a single number, date, name or "
                                   "yes/no when the question admits one. Empty string otherwise.")
    steps: list[Step] = Field(description="Every arithmetic step performed, in order. "
                                          "Empty list if the question needed no computation.")
    sources_used: list[str] = Field(description="doc_id values that the answer actually relies on.")
    confidence: float = Field(description="0.0 to 1.0. Below 0.5 means the documents did not "
                                          "settle the question.")
    unresolved: str = Field(description="What was missing or contradictory, if anything. "
                                        "Empty string when the answer is fully supported.")


class QuestionResult(BaseModel):
    qid: str
    question: str
    record: AnswerRecord | None = None
    citations: list[Citation] = Field(default_factory=list)
    prose: str = ""
    arith_ok: bool = True
    arith_notes: list[str] = Field(default_factory=list)
    docs_considered: list[str] = Field(default_factory=list)
    error: str = ""
    usage: dict = Field(default_factory=dict)
    elapsed_s: float = 0.0
