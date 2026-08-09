"""Related parties from the KYC dossier.

The dossier lists counterparty entities with a voting-rights percentage and
states the threshold above which an entity counts as related. The pack plants
three kinds of decoy around it:

  * the same entity spelled differently in the ledger ("LLP" vs "L.L.P.")
  * an entity just under the threshold (18.7 per cent against a 20 per cent test)
  * a similarly named entity that is not in the dossier at all

So matching is done on a normalised name, and the threshold is read from the
document rather than assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Legal-form tokens, compared after punctuation is stripped and dotted
# abbreviations are collapsed ("l.l.p." arrives here as "llp").
LEGAL_FORM_TOKENS = {
    "llp", "llc", "jsc", "ltd", "limited", "inc", "lp", "plc", "co",
    "company", "corp", "corporation", "gmbh", "sa", "bv", "nv",
    "тоо", "ао", "оао", "зао", "пао",
}

OWNER_ROW_RE = re.compile(
    r"([A-ZА-ЯЁ][^%\n]{2,70}?)\s+(\d{1,3}(?:[.,]\d{1,2})?)\s*%"
)
THRESHOLD_RE = re.compile(
    r"(\d{1,3}(?:[.,]\d{1,2})?)\s*%\s*и\s*более[^.]{0,80}?связанным[и]?\s+сторон",
    re.I | re.S,
)
THRESHOLD_RE_EN = re.compile(
    r"(\d{1,3}(?:[.,]\d{1,2})?)\s*(?:per cent|%)\s+or\s+more[^.]{0,80}?related part",
    re.I | re.S,
)

PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")


def normalize_entity(name: str) -> str:
    """Collapse spelling variants so a ledger counterparty can be matched to a
    dossier entity: "Atyrau Holding Group L.L.P." and "Atyrau Holding Group LLP"
    have to land on the same key. Drops the site qualifier the ledger appends.

    Punctuation goes first, so a dotted abbreviation arrives as separate letters
    and can be collapsed back into its undotted form before the legal-form
    tokens are removed.
    """
    s = (name or "").strip().lower()
    s = PAREN_RE.sub("", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # "l l p" -> "llp", "j s c" -> "jsc"
    s = re.sub(r"\b(?:(?:[a-z]\s){1,4}[a-z])\b",
               lambda m: m.group(0).replace(" ", ""), s)
    tokens = [t for t in s.split() if t not in LEGAL_FORM_TOKENS]
    return " ".join(tokens).strip()


@dataclass
class Owner:
    name: str
    share: float

    @property
    def key(self) -> str:
        return normalize_entity(self.name)


# The dossier comes in two forms. One prints an ownership table and a threshold.
# The other declares status per counterparty by name, and then the percentage
# never appears: "Контрагент «Altyn Capital L.L.P.» классифицирован как
# АФФИЛИРОВАННОЕ ЛИЦО Заёмщика". Both are read; a named declaration is decisive
# on its own, because it states the conclusion the threshold exists to reach.
def _loose(word: str) -> str:
    """A pattern tolerant of the stray spaces PDF extraction leaves inside a
    word: the pack yields "ОГР АНИЧЕННОЙ" and "У становленные" routinely."""
    return r"\s*".join(re.escape(ch) for ch in word)


_STATUSES = {
    "unrestricted": ("НЕОГРАНИЧЕННОЙ ДОЧЕРНЕЙ", "НЕОГРАНИЧЕННАЯ ДОЧЕРНЯЯ",
                     "UNRESTRICTED SUBSIDIARY"),
    "restricted": ("ОГРАНИЧЕННОЙ ДОЧЕРНЕЙ", "ОГРАНИЧЕННАЯ ДОЧЕРНЯЯ",
                   "RESTRICTED SUBSIDIARY"),
    "related": ("АФФИЛИРОВАННОЕ ЛИЦО", "АФФИЛИРОВАННЫМ ЛИЦОМ",
                "СВЯЗАННОЙ СТОРОНОЙ", "СВЯЗАННАЯ СТОРОНА",
                "RELATED PARTY", "AFFILIATE"),
}

# Longest first, and "неограниченной" before "ограниченной", so a negation is
# never read as its opposite.
_ALL_STATUSES = [(kind, phrase)
                 for kind in ("unrestricted", "restricted", "related")
                 for phrase in _STATUSES[kind]]

# The name is anchored to the word that introduces it, so a statute cited in
# quotation marks later in the dossier is never read as a counterparty.
RECORD_RE = re.compile(
    r"(?:Контрагент|Counterparty|Организация|Entity)\s*[«\"]([^»\"]{3,70})[»\"]"
    r"(.{0,90}?)(" + "|".join(_loose(p) for _, p in _ALL_STATUSES) + r")",
    re.I | re.S,
)


def _status_kind(matched: str) -> str:
    squashed = re.sub(r"\s+", "", matched).upper()
    for kind, phrase in _ALL_STATUSES:
        if squashed.startswith(re.sub(r"\s+", "", phrase).upper()):
            return kind
    return ""


@dataclass
class KycProfile:
    threshold: float
    owners: list[Owner]
    declared_related: list[str] = field(default_factory=list)
    declared_unrestricted: list[str] = field(default_factory=list)

    @property
    def related(self) -> list[Owner]:
        return [o for o in self.owners if o.share >= self.threshold]

    @property
    def related_keys(self) -> set[str]:
        keys = {o.key for o in self.related if o.key}
        keys |= {normalize_entity(n) for n in self.declared_related}
        return {k for k in keys if k}

    @property
    def unrestricted_keys(self) -> set[str]:
        return {normalize_entity(n) for n in self.declared_unrestricted
                if normalize_entity(n)}


def _num(s: str) -> float:
    return float(s.replace(",", "."))


TABLE_START = ("Доля голосующих прав", "Voting rights", "Доля участия")
TABLE_END = ("Организации, в которых", "Entities in which", "Идентификация и проверка",
             "Identification and verification")

# Splits the table region into "<name> <share>%" pairs without letting the
# column header or the threshold sentence be read as an entity.
ROW_SPLIT_RE = re.compile(r"(.+?)\s+(\d{1,3}(?:[.,]\d{1,2})?)\s*%")


def parse_kyc(text: str, default_threshold: float = 20.0) -> KycProfile:
    flat = " ".join(text.split())

    threshold = default_threshold
    m = THRESHOLD_RE.search(flat) or THRESHOLD_RE_EN.search(flat)
    if m:
        threshold = _num(m.group(1))

    declared_related: list[str] = []
    declared_unrestricted: list[str] = []
    for m in RECORD_RE.finditer(flat):
        name = m.group(1).strip()
        kind = _status_kind(m.group(3))
        if kind == "unrestricted":
            declared_unrestricted.append(name)
        elif kind == "related":
            declared_related.append(name)

    region = ""
    for start in TABLE_START:
        i = flat.find(start)
        if i >= 0:
            region = flat[i + len(start):]
            break
    if not region:
        return KycProfile(threshold=threshold, owners=[],
                          declared_related=declared_related,
                          declared_unrestricted=declared_unrestricted)
    for end in TABLE_END:
        j = region.find(end)
        if j >= 0:
            region = region[:j]
            break

    owners: list[Owner] = []
    seen: set[str] = set()
    for m in ROW_SPLIT_RE.finditer(region):
        name = m.group(1).strip(" .,;·—-")
        key = normalize_entity(name)
        if not key or key in seen or len(name) < 3:
            continue
        seen.add(key)
        owners.append(Owner(name=name, share=_num(m.group(2))))
    return KycProfile(threshold=threshold, owners=owners,
                      declared_related=declared_related,
                      declared_unrestricted=declared_unrestricted)


NONE_FOUND = ("Связанные стороны среди контрагентов не выявлены",
              "No related parties were identified")


def is_kyc(text: str) -> bool:
    """The pack also contains internal notes that merely mention KYC. The
    dossier is the one that prints the ownership table, declares a
    counterparty's status by name, or states that there are none - the third
    form matters because a dossier saying "none found" is an answer, while a
    dossier that was never recognised is a silent loss."""
    named = ("Знай своего клиента" in text or "Know Your Customer" in text)
    flat = " ".join(text.split())
    return named and (any(s in text for s in TABLE_START)
                      or bool(RECORD_RE.search(flat))
                      or any(n in flat for n in NONE_FOUND))
