"""Corpus index: per-document header facts, then a cross-document pass that
decides which version of each document is the operative one.

"Which version is in force" is a scored requirement of the challenge and it is
also the failure mode a naive RAG pipeline hits first: it quotes a superseded
addendum with perfect confidence. So it is resolved once, up front, and the
verdict travels with every answer prompt.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel, Field

from .ingest import doc_blocks
from .llm import LLM
from .schema import Doc, DocFacts, DocIndexEntry

FACTS_SYSTEM = (
    "You extract header facts from banking documents. Report only what the document "
    "states. Never infer a date, a version or an identifier that is not printed. "
    "Leave a field as an empty string when the document does not state it."
)

VERSION_SYSTEM = (
    "You are reconciling a pack of banking documents into a version history. "
    "Several documents may describe the same contract, tariff or policy at different "
    "points in time. Decide which document is operative and which has been replaced.\n"
    "Rules:\n"
    "- Group by subject and identifier, not by filename.\n"
    "- An amendment or addendum supersedes only the clauses it names; the base "
    "document stays operative for everything else. Mark both as current in that case "
    "and say so in the note.\n"
    "- A full restatement supersedes the prior document entirely.\n"
    "- If the evidence does not settle the order, mark every candidate current and "
    "say what is missing in the note. Guessing an order is worse than admitting it."
)


class VersionVerdict(BaseModel):
    doc_id: str
    is_current: bool = Field(description="True if this document is operative for at "
                                         "least some of its clauses.")
    superseded_by: str = Field(description="doc_id that replaces it, or empty string.")
    note: str = Field(description="One sentence of reasoning, or the missing evidence.")


class VersionReport(BaseModel):
    verdicts: list[VersionVerdict]


async def _facts_for(llm: LLM, doc: Doc, effort: str, max_tokens: int) -> DocFacts:
    text = doc.plain_text(20_000)
    if text.strip():
        content: list[dict] | str = (
            f"Document id: {doc.id}\nFilename: {doc.title}\nPages: {doc.n_pages}\n\n"
            f"{text}"
        )
    else:
        # No text layer (scan or photo). Let the model read the bytes.
        content = [
            {"type": "text", "text": f"Document id: {doc.id}\nFilename: {doc.title}"},
            *doc_blocks(doc, citations=False),
        ]
    return await llm.parse(
        schema=DocFacts,
        system=FACTS_SYSTEM,
        content=content,
        max_tokens=max_tokens,
        effort=effort,
    )


async def build_index(
    llm: LLM,
    docs: list[Doc],
    *,
    effort: str = "medium",
    max_tokens: int = 8000,
    concurrency: int = 8,
) -> list[DocIndexEntry]:
    sem = asyncio.Semaphore(concurrency)

    async def one(doc: Doc) -> DocIndexEntry:
        async with sem:
            try:
                facts = await _facts_for(llm, doc, effort, max_tokens)
            except Exception as e:  # never let one bad file sink the index
                facts = DocFacts(
                    doc_type="unknown", subject="", identifier="", effective_date="",
                    version_label="", supersedes="", key_numbers=[],
                    summary=f"Facts extraction failed: {type(e).__name__}: {e}",
                )
            return DocIndexEntry(
                doc_id=doc.id, title=doc.title, kind=doc.kind,
                n_pages=doc.n_pages, facts=facts,
            )

    entries = await asyncio.gather(*(one(d) for d in docs))
    return await resolve_versions(llm, list(entries), effort=effort, max_tokens=max_tokens)


async def resolve_versions(
    llm: LLM,
    entries: list[DocIndexEntry],
    *,
    effort: str = "medium",
    max_tokens: int = 8000,
) -> list[DocIndexEntry]:
    if len(entries) < 2:
        return entries
    compact = [
        {
            "doc_id": e.doc_id,
            "doc_type": e.facts.doc_type,
            "subject": e.facts.subject,
            "identifier": e.facts.identifier,
            "effective_date": e.facts.effective_date,
            "version_label": e.facts.version_label,
            "supersedes": e.facts.supersedes,
            "summary": e.facts.summary,
        }
        for e in entries
    ]
    try:
        report = await llm.parse(
            schema=VersionReport,
            system=VERSION_SYSTEM,
            content="Documents:\n" + json.dumps(compact, ensure_ascii=False, indent=1)
                    + "\n\nReturn one verdict per doc_id, all of them.",
            max_tokens=max_tokens,
            effort=effort,
        )
    except Exception:
        return entries

    by_id = {v.doc_id: v for v in report.verdicts}
    for e in entries:
        v = by_id.get(e.doc_id)
        if v is not None:
            e.is_current = v.is_current
            e.superseded_by = v.superseded_by or None
            e.note = v.note
    return entries


def save_index(entries: list[DocIndexEntry], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps([e.model_dump() for e in entries], ensure_ascii=False, indent=2)
    )


def load_index(path: str | Path) -> list[DocIndexEntry]:
    data = json.loads(Path(path).read_text())
    return [DocIndexEntry.model_validate(d) for d in data]


def index_digest(entries: list[DocIndexEntry]) -> str:
    """Compact catalogue that goes into every answer prompt, cached."""
    lines = []
    for e in entries:
        status = "CURRENT" if e.is_current else f"SUPERSEDED by {e.superseded_by or '?'}"
        f = e.facts
        bits = [
            f"- {e.doc_id} [{status}] type={f.doc_type or '?'}",
            f"  subject={f.subject or '?'} id={f.identifier or '?'} "
            f"effective={f.effective_date or '?'} version={f.version_label or '?'}",
            f"  {f.summary}",
        ]
        if f.key_numbers:
            bits.append("  figures: " + "; ".join(f.key_numbers[:10]))
        if e.note:
            bits.append(f"  version note: {e.note}")
        lines.append("\n".join(bits))
    return "\n".join(lines)
