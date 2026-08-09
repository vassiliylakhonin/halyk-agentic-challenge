"""Deterministic layer for the covenant task.

Everything here runs without an API call: text extraction, linking documents to
borrowers, spotting superseded editions, locating covenant clauses, and slicing
the ledger. The model is needed only for judgement that cannot be regexed, so
the cheaper this layer is, the more budget is left for the parts that need it.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Account identifiers are not one fixed prefix: the packs use ACC-7801 and
# TELE-4471 alike. The ledger is the authority on which ones exist, so a
# document is linked by finding an id the ledger actually knows.
ACC_RE = re.compile(r"\b[A-Z]{2,8}-\d{3,6}\b")
TXN_RE = re.compile(r"\bTXN-[A-Z0-9]+-\d+\b")

# Printed on the front of every replaced edition in the open pack.
SUPERSEDED_MARKERS = (
    "НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ",
    "Заменена и изложена",
    "SUPERSEDED",
    "NO LONGER IN FORCE",
)

# Clauses are printed as "Пункт 6.1", "Clause 6.1" or a bare "6.1", and PDF
# extraction leaves non-breaking spaces between the word and the number.
CLAUSE_RE = re.compile(
    r"(?:^|\n)\s*(?:Пункт|Clause|Section|Статья)?[\s ]*(\d+\.\d+)[\s ]*[.\)]?[\s ]+",
    re.M,
)


def parse_amount(raw: str) -> float | None:
    """The ledger is deliberately dirty: blanks, spaces, parentheses for
    negatives, currency symbols and comma decimals all appear. None means the
    row states no amount, which is a fact about the row, not a parse failure."""
    s = (raw or "").strip()
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    s = re.sub(r"[^\d,.\-+]", "", s)
    if "," in s and "." in s:
        s = s.replace(",", "") if s.rfind(".") > s.rfind(",") else s.replace(".", "").replace(",", ".")
    elif s.count(",") == 1 and len(s.split(",")[-1]) in (1, 2):
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


@dataclass
class Txn:
    txn_id: str
    date: str
    account_id: str
    counterparty: str
    description: str
    amount: float | None
    currency: str

    @property
    def scenario(self) -> str:
        parts = self.txn_id.split("-")
        return parts[1] if len(parts) > 2 else ""

    @property
    def is_outflow(self) -> bool:
        return self.amount is not None and self.amount < 0


@dataclass
class Ledger:
    txns: list[Txn]

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        return cls([
            Txn(
                txn_id=r["txn_id"], date=r["date"], account_id=r["account_id"],
                counterparty=r["counterparty"], description=r["description"],
                amount=parse_amount(r.get("amount", "")), currency=r["currency"],
            )
            for r in rows
        ])

    def by_scenario(self) -> dict[str, list[Txn]]:
        out: dict[str, list[Txn]] = defaultdict(list)
        for t in self.txns:
            out[t.scenario].append(t)
        return dict(out)

    def account_to_scenario(self) -> dict[str, str]:
        """Each account_id belongs to exactly one scenario, per the case notes."""
        seen: dict[str, set[str]] = defaultdict(set)
        for t in self.txns:
            seen[t.account_id].add(t.scenario)
        return {acc: sorted(scen)[0] for acc, scen in seen.items() if len(scen) == 1}

    def scenarios(self, template_keys: set[str] | None = None) -> list[str]:
        found = {t.scenario for t in self.txns}
        return sorted(found & template_keys) if template_keys else sorted(found)


@dataclass
class PdfDoc:
    doc_id: str
    path: str
    n_pages: int
    text: str
    pages: list[str] = field(default_factory=list)
    accounts: list[str] = field(default_factory=list)
    scenario: str = ""
    superseded: bool = False
    has_covenants: bool = False

    def clause(self, number: str) -> str:
        """Text of a numbered clause, from its heading to the next one."""
        hits = list(CLAUSE_RE.finditer(self.text))
        for i, m in enumerate(hits):
            if m.group(1) == number:
                end = hits[i + 1].start() if i + 1 < len(hits) else len(self.text)
                return self.text[m.start():end].strip()
        return ""


MIN_PAGE_CHARS = 40


def load_ocr_cache(path: str | Path | None) -> dict[str, dict[str, str]]:
    """Text recovered from pages that carry no text layer. The run-time vision
    path writes this file; it is read here so downstream code never has to care
    whether a page arrived as text or as an image."""
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def pages_needing_ocr(pages: list[str]) -> list[int]:
    """1-based page numbers whose extracted text is too thin to be real."""
    return [i for i, t in enumerate(pages, start=1) if len(t.strip()) < MIN_PAGE_CHARS]


def extract_pdfs(
    docs_dir: str | Path,
    cache_path: str | Path | None = None,
    ocr_cache_path: str | Path | None = None,
) -> list[PdfDoc]:
    """Extract every PDF once and cache the text, so reruns are instant."""
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        return [PdfDoc(**d) for d in json.loads(cache.read_text(encoding="utf-8"))]

    from pypdf import PdfReader

    ocr = load_ocr_cache(ocr_cache_path)
    docs: list[PdfDoc] = []
    for p in sorted(Path(docs_dir).glob("*.pdf")):
        try:
            reader = PdfReader(str(p))
            pages = [(pg.extract_text() or "") for pg in reader.pages]
        except Exception:
            pages = []
        recovered = ocr.get(p.stem, {})
        for n in pages_needing_ocr(pages):
            if str(n) in recovered:
                pages[n - 1] = recovered[str(n)]
        text = "\n".join(pages)
        docs.append(PdfDoc(
            doc_id=p.stem, path=str(p), n_pages=len(pages) or 1,
            text=text, pages=pages,
            accounts=sorted(set(ACC_RE.findall(text))),
            superseded=any(m in text for m in SUPERSEDED_MARKERS),
            has_covenants=bool(re.search(r"\b6\.[123]\b", text)),
        ))

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([d.__dict__ for d in docs], ensure_ascii=False),
                         encoding="utf-8")
    return docs


def refresh_from_text(doc: PdfDoc) -> PdfDoc:
    """Recompute the fields derived from a document's text.

    Needed after pages are recovered from images: an account number, a
    supersession marker or a covenant clause can live inside a scan, and until
    the derived fields are rebuilt the document stays unlinked and invisible.
    """
    doc.text = "\n".join(doc.pages)
    doc.accounts = sorted(set(ACC_RE.findall(doc.text)))
    doc.superseded = any(m in doc.text for m in SUPERSEDED_MARKERS)
    doc.has_covenants = bool(re.search(r"\b6\.[123]\b", doc.text))
    return doc


def link_documents(docs: list[PdfDoc], ledger: Ledger) -> list[PdfDoc]:
    """Attach a scenario to every document that names an account we can place."""
    acc2scen = ledger.account_to_scenario()
    known = set(acc2scen)
    for d in docs:
        found = {a for a in d.accounts if a in known}
        if not found:
            # The candidate regex can miss an unusual shape; fall back to looking
            # for the ledger's own identifiers verbatim.
            found = {a for a in known if a in d.text}
        scens = {acc2scen[a] for a in found}
        d.scenario = sorted(scens)[0] if len(scens) == 1 else ""
    return docs


def clause_numbers(text: str) -> set[str]:
    """Clause numbers a document prints as headings."""
    return {m.group(1) for m in CLAUSE_RE.finditer(text)}


def agreements_by_scenario(
    docs: list[PdfDoc], wanted: set[str] | None = None
) -> dict[str, dict[str, list[PdfDoc]]]:
    """Per scenario, the credit agreements split into current and superseded.

    Which clause numbers carry the covenants is a property of the pack, not a
    constant: one borrower prints them under 6.1 and another under 5.1. So a
    document counts as the agreement when it prints the clauses the template
    asks about, and `wanted` comes from the template.
    """
    out: dict[str, dict[str, list[PdfDoc]]] = defaultdict(
        lambda: {"current": [], "superseded": []})
    for d in docs:
        if not d.scenario:
            continue
        found = clause_numbers(d.text)
        hit = bool(found & wanted) if wanted else d.has_covenants
        if hit:
            out[d.scenario]["superseded" if d.superseded else "current"].append(d)
    return dict(out)
