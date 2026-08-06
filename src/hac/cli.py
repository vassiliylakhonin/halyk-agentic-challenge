from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path

from .config import Config
from .corpus import build_index, index_digest, load_index, save_index
from .ingest import load_corpus
from .llm import LLM
from .runner import run as run_pipeline


def _load_env() -> None:
    """Minimal .env support so the key never has to live in shell history."""
    p = Path(".env")
    if not p.exists():
        return
    import os
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def cmd_ingest(cfg: Config, args) -> int:
    docs = load_corpus(cfg.docs_dir)
    kinds: dict[str, int] = {}
    pages = 0
    no_text = []
    for d in docs:
        kinds[d.kind] = kinds.get(d.kind, 0) + 1
        pages += d.n_pages
        if not d.plain_text().strip():
            no_text.append(d.id)
    print(json.dumps({
        "documents": len(docs),
        "by_kind": kinds,
        "pages_or_chunks": pages,
        "no_text_layer": no_text,
    }, ensure_ascii=False, indent=2))
    if no_text:
        print("\nThe files above have no extractable text. They will be sent to the "
              "model as images/PDF bytes, which is the intended OCR path.", file=sys.stderr)
    return 0


def cmd_index(cfg: Config, args) -> int:
    docs = load_corpus(cfg.docs_dir)
    llm = LLM(cfg.model, cache_ttl=cfg.cache_ttl, max_attempts=cfg.max_attempts)
    entries = asyncio.run(build_index(
        llm, docs, effort=cfg.effort_index,
        max_tokens=cfg.max_tokens_small, concurrency=cfg.concurrency))
    save_index(entries, cfg.index_file)
    print(index_digest(entries))
    print(f"\nindex written to {cfg.index_file}; "
          f"estimated cost so far ${llm.usage.cost_usd(cfg.cache_ttl):.2f}")
    return 0


def cmd_run(cfg: Config, args) -> int:
    adapter = importlib.import_module(cfg.adapter)
    questions = adapter.load_questions(cfg.questions_file)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("no questions loaded — check questions_file and the adapter", file=sys.stderr)
        return 2
    docs = load_corpus(cfg.docs_dir)

    if Path(cfg.index_file).exists() and not args.reindex:
        entries = load_index(cfg.index_file)
        known = {e.doc_id for e in entries}
        missing = [d for d in docs if d.id not in known]
        if missing:
            print(f"{len(missing)} documents are not in the index; rebuilding", file=sys.stderr)
            entries = None  # type: ignore[assignment]
    else:
        entries = None  # type: ignore[assignment]

    if entries is None:
        llm = LLM(cfg.model, cache_ttl=cfg.cache_ttl, max_attempts=cfg.max_attempts)
        entries = asyncio.run(build_index(
            llm, docs, effort=cfg.effort_index,
            max_tokens=cfg.max_tokens_small, concurrency=cfg.concurrency))
        save_index(entries, cfg.index_file)

    print(f"{len(docs)} documents, {len(questions)} questions, "
          f"concurrency {cfg.concurrency}, model {cfg.model}")
    asyncio.run(run_pipeline(cfg, questions, docs, entries, resume=not args.no_resume))
    return 0


def cmd_solve(cfg: Config, args) -> int:
    """Competition-day entry point: a document pack in, a submission out."""
    from .solve import solve
    report = solve(
        args.docs_dir, args.ledger, args.template, args.out or cfg.submission_file,
        provider=args.provider, model=args.llm_model,
        team=cfg.team, contact_email=cfg.contact_email,
        text_cache=args.text_cache,
    )
    print(json.dumps(report.summary(), ensure_ascii=False, indent=2))
    return 0


def cmd_package(cfg: Config, args) -> int:
    """Rebuild submission.json from answers.jsonl without calling the API."""
    from .runner import load_done
    adapter = importlib.import_module(cfg.adapter)
    done = load_done(cfg.answers_file)
    if not done:
        print("answers.jsonl is empty", file=sys.stderr)
        return 2
    adapter.write_submission(list(done.values()), cfg.submission_file)
    print(f"{len(done)} answers -> {cfg.submission_file}")
    return 0


def cmd_validate(cfg: Config, args) -> int:
    adapter = importlib.import_module(cfg.adapter)
    questions = adapter.load_questions(cfg.questions_file)
    sub_path = Path(cfg.submission_file)
    if not sub_path.exists():
        print(f"{sub_path} does not exist — run `hac run` or `hac package` first",
              file=sys.stderr)
        return 2
    data = json.loads(sub_path.read_text(encoding="utf-8"))
    answers = data.get("answers", data) if isinstance(data, dict) else data
    ids = {str(a.get("question_id")) for a in answers}
    missing = [qid for qid, _ in questions if qid not in ids]
    empty = [a.get("question_id") for a in answers if not (a.get("answer") or "").strip()]
    print(json.dumps({
        "questions": len(questions),
        "answers_in_file": len(answers),
        "missing_question_ids": missing,
        "empty_answers": empty,
        "ok": not missing and not empty,
    }, ensure_ascii=False, indent=2))
    return 0 if not missing and not empty else 1


def main(argv: list[str] | None = None) -> int:
    _load_env()
    p = argparse.ArgumentParser(prog="hac", description="Document-grounded banking agent")
    p.add_argument("-c", "--config", default="config.json")
    p.add_argument("--docs", help="override docs_dir")
    p.add_argument("--questions", help="override questions_file")
    p.add_argument("--out", help="override submission_file")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="scan the document folder and report what was parsed")
    ix = sub.add_parser("index", help="build the document catalogue and version verdicts")
    ix.set_defaults(func=cmd_index)

    r = sub.add_parser("run", help="answer every question and write submission.json")
    r.add_argument("--limit", type=int, default=0, help="answer only the first N questions")
    r.add_argument("--reindex", action="store_true")
    r.add_argument("--no-resume", action="store_true")
    r.set_defaults(func=cmd_run)

    sv = sub.add_parser("solve", help="document pack -> submission.json")
    sv.add_argument("--docs-dir", required=True)
    sv.add_argument("--ledger", required=True)
    sv.add_argument("--template", required=True)
    sv.add_argument("--out", default=None)
    sv.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    sv.add_argument("--llm-model", default=None)
    sv.add_argument("--text-cache", default=None)
    sv.set_defaults(func=cmd_solve)

    sub.add_parser("package", help="rebuild submission.json from answers.jsonl"
                   ).set_defaults(func=cmd_package)
    sub.add_parser("validate", help="check submission.json covers every question"
                   ).set_defaults(func=cmd_validate)

    args = p.parse_args(argv)
    cfg = Config.load(args.config)
    if args.docs:
        cfg.docs_dir = args.docs
    if args.questions:
        cfg.questions_file = args.questions
    if args.out:
        cfg.submission_file = args.out

    func = getattr(args, "func", cmd_ingest)
    return func(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
