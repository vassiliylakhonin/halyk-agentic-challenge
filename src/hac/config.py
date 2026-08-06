from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Defaults for a run. Every field can be overridden on the command line,
    so a competition day never depends on a file being edited correctly."""

    # Model access
    provider: str = "openai"
    model: str | None = None          # None asks the account what it has
    samples: int = 3                  # readings per clause; the agreed one is used

    # Submission identity, written into the top-level fields of the file
    team: str = ""
    contact_email: str = ""

    # Where traces of what the model read are written
    out_dir: str = "out"

    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        cfg = cls()
        if path and Path(path).exists():
            data = json.loads(Path(path).read_text())
            known = set(cfg.__dataclass_fields__)
            for k, v in data.items():
                if k in known:
                    setattr(cfg, k, v)
                else:
                    cfg.extra[k] = v
        return cfg

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
