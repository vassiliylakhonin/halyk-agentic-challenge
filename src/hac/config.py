from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Models. One model, different effort per stage.
    model: str = "claude-opus-5"
    effort_answer: str = "high"       # low | medium | high | xhigh | max
    effort_index: str = "medium"
    effort_structure: str = "low"

    max_tokens_answer: int = 32000
    max_tokens_small: int = 8000

    # Retrieval
    max_docs_per_question: int = 6
    router_enabled: bool = True       # False = send the whole corpus every time
    inline_text_char_budget: int = 400_000  # cap on text-only corpus sent per question

    # Concurrency and resilience
    concurrency: int = 8
    per_question_timeout_s: float = 420.0
    max_attempts: int = 3

    # Prompt caching
    cache_ttl: str = "1h"             # "5m" or "1h"

    # Refusal fallback (Opus 5). Degrades to the plain endpoint on any 400.
    server_side_fallback: bool = True

    # Hard spend ceiling in USD for one run. 0 disables the check. When the
    # estimated spend crosses it, questions still in the queue are skipped and
    # the partial submission on disk stays valid.
    budget_usd: float = 0.0

    # Paths
    docs_dir: str = "data/private/docs"
    questions_file: str = "data/private/questions.json"
    out_dir: str = "out"
    index_file: str = "out/index.json"
    answers_file: str = "out/answers.jsonl"
    submission_file: str = "submission.json"

    # Which adapter maps internal answers -> the competition file
    adapter: str = "hac.adapters.halyk_agentic"

    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        cfg = cls()
        if path and Path(path).exists():
            data = json.loads(Path(path).read_text())
            known = {f for f in cfg.__dataclass_fields__}
            for k, v in data.items():
                if k in known:
                    setattr(cfg, k, v)
                else:
                    cfg.extra[k] = v
        return cfg

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
