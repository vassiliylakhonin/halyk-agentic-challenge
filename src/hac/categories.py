"""Classifying ledger lines into the accounting categories the covenants name.

The ledger ships without a category column on purpose, and the descriptions are
written to punish pattern matching:

  * "Purchase of grain conveyor equipment" is capital expenditure, while
    "Flood remediation and silo repair works" is repair and stays in operating
  * "Capitalised interest charge" reads like capex and is interest
  * "Outdoor marketing site hire" reads like a lease and is marketing
  * most inflows are refunds, rebates and clearing sweeps, not revenue

Rules are ordered: the first matching rule wins, so the specific traps above are
placed ahead of the generic keyword that would otherwise catch them. Anything
unmatched comes back as OTHER, which is what the model-backed fallback is for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    REVENUE = "revenue"
    CAPEX = "capex"
    OPEX = "opex"
    LEASE = "lease"
    PAYROLL = "payroll"
    UTILITIES = "utilities"
    TELECOM = "telecom"
    MARKETING = "marketing"
    TAX = "tax"
    INTEREST = "interest"
    INSURANCE = "insurance"
    FINANCING = "financing"
    NON_OPERATING = "non_operating"
    OTHER = "other"


# Russian labels as they appear inside covenant clauses, mapped to a category.
CLAUSE_LABELS: dict[str, Category] = {
    "выручка": Category.REVENUE,
    "капитальные затраты": Category.CAPEX,
    "капитальных затрат": Category.CAPEX,
    "операционные расходы": Category.OPEX,
    "операционных расходов": Category.OPEX,
    "арендные платежи": Category.LEASE,
    "арендных платежей": Category.LEASE,
    "оплату труда": Category.PAYROLL,
    "оплаты труда": Category.PAYROLL,
    "персонал": Category.PAYROLL,
    "коммунальные услуги": Category.UTILITIES,
    "коммунальных услуг": Category.UTILITIES,
    "маркетинг": Category.MARKETING,
    "налог": Category.TAX,
    "проценты": Category.INTEREST,
    "страхован": Category.INSURANCE,
}


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern
    category: Category
    why: str


def _r(expr: str, cat: Category, why: str) -> Rule:
    return Rule(re.compile(expr, re.I), cat, why)


# Order is the logic. Traps first, generic keywords last.
RULES: list[Rule] = [
    # --- money coming back, never revenue ------------------------------------
    _r(r"\b(refund|rebate|returned|return of|credit received|credited|recover(y|ed)|"
       r"reimbursement|sweep back|clearing account sweep|drawback|"
       r"unused .*budget|overpayment)\b", Category.NON_OPERATING,
       "a return of money already spent is not income"),
    _r(r"\b(interest income|income on treasury|lease incentive received|"
       r"insurance broker rebate|experience refund)\b", Category.NON_OPERATING,
       "financial or incentive income, outside the operating revenue line"),

    # --- financing proceeds, ahead of the generic interest rule ---------------
    _r(r"\b(drawdown|draw-down|drawn under|proceeds of|facility utilisation)\b",
       Category.FINANCING, "money drawn under a facility, not an expense"),

    # --- traps that look like another category -------------------------------
    _r(r"\bcapitalised interest\b", Category.INTEREST,
       "capitalised interest is a financing cost, not capital expenditure"),
    _r(r"\binterest\b.{0,24}\b(lease|sublease|hire purchase)\b", Category.INTEREST,
       "interest on a finance lease is a financing cost, not a lease payment"),
    _r(r"\bmarketing\b.*\b(site hire|hire)\b|\b(outdoor|billboard) marketing\b",
       Category.MARKETING, "site hire bought for advertising is marketing"),
    _r(r"\b(repair|remediation|cleaning|clearance|overhaul|refurbish\w*|"
       r"servicing|maintenance)\b", Category.OPEX,
       "repair and upkeep stay in operating costs"),

    # --- capital expenditure -------------------------------------------------
    _r(r"\b(purchase|acquisition|procurement) of\b.*\b(equipment|machinery|plant|"
       r"vehicle|fleet|conveyor|crane|line|system|installation|asset)\b",
       Category.CAPEX, "acquisition of an asset"),
    _r(r"\b(construction|erection|installation) of\b|\bcapital (works|project|"
       r"expenditure|programme)\b|\bnew build\b", Category.CAPEX,
       "construction and capital programmes"),

    # --- named operating lines ----------------------------------------------
    _r(r"\bpayroll\b|\bstaff (wages|salaries)\b|\bsalary\b|\bwage\b|"
       r"\bsocial (contribution|insurance contribution)\b", Category.PAYROLL,
       "labour cost"),
    _r(r"\b(marketing|advertis\w+|ad campaign|media buy|sponsorship|exhibition|"
       r"newsletter|collateral|press insertion|promotional|point-of-sale|"
       r"brand|livery)\b", Category.MARKETING, "marketing line"),
    _r(r"\b(lease|rent|tenancy|ground lease|sublease)\b", Category.LEASE,
       "lease or rent"),
    _r(r"\b(electricity|power|water|gas|heating|utility|utilities|metering|"
       r"network capacity|waste|sewer)\b", Category.UTILITIES, "utility supply"),
    _r(r"\btelecom\w*|\bleased line\b|\bmobile fleet plan\b|\bconnectivity\b",
       Category.TELECOM, "communications"),
    _r(r"\b(tax|vat|levy|duty|excise|customs)\b", Category.TAX, "tax or levy"),
    _r(r"\b(interest|coupon|overdraft|revolver)\b", Category.INTEREST,
       "financing cost"),
    _r(r"\b(insurance|indemnity|bond|underwrit\w+)\b", Category.INSURANCE,
       "insurance premium"),

    # --- revenue -------------------------------------------------------------
    _r(r"\b(sales|revenue|settlement|invoice|billing|turnover|handling|"
       r"stevedoring|freight|haulage|throughput|tariff income)\b",
       Category.REVENUE, "operating sales"),

    # --- everything else that is clearly an operating cost -------------------
    _r(r"\b(operating costs?|professional fees?|advisory|retainer|consult\w+|"
       r"legal|arbitration|audit fee|subscription|logistics|transport|"
       r"security|catering|training|travel)\b", Category.OPEX, "operating cost"),
]


def classify(description: str, counterparty: str = "") -> tuple[Category, str]:
    """Category comes from the payment description alone.

    Counterparty names are decoys: an antenna mast lease is paid to a company
    with "Payroll Group" in its name, an equipment purchase to "... Realty".
    The agreements say the same thing from the other side - related-party status
    follows the dossier, not the payment narrative - so the two fields are kept
    strictly apart. `counterparty` is accepted and ignored so callers read
    naturally.
    """
    text = description
    for rule in RULES:
        if rule.pattern.search(text):
            return rule.category, rule.why
    return Category.OTHER, "no rule matched"


def category_from_clause(clause_text: str) -> Category | None:
    """Which accounting line a covenant clause is written about."""
    low = clause_text.lower()
    best: tuple[int, Category] | None = None
    for label, cat in CLAUSE_LABELS.items():
        i = low.find(f"«{label}")
        if i < 0:
            i = low.find(label)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, cat)
    return best[1] if best else None
