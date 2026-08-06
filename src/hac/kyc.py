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
from dataclasses import dataclass

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


@dataclass
class KycProfile:
    threshold: float
    owners: list[Owner]

    @property
    def related(self) -> list[Owner]:
        return [o for o in self.owners if o.share >= self.threshold]

    @property
    def related_keys(self) -> set[str]:
        return {o.key for o in self.related if o.key}


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

    region = ""
    for start in TABLE_START:
        i = flat.find(start)
        if i >= 0:
            region = flat[i + len(start):]
            break
    if not region:
        return KycProfile(threshold=threshold, owners=[])
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
    return KycProfile(threshold=threshold, owners=owners)


def is_kyc(text: str) -> bool:
    """The pack also contains internal notes that merely mention KYC. The
    dossier itself is the one with the beneficial-ownership table."""
    return ("Знай своего клиента" in text or "Know Your Customer" in text) and any(
        s in text for s in TABLE_START
    )
