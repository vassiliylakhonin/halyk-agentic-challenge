"""Orchestration for the scored window.

Design constraints, in order of importance:
  1. A valid submission.json exists on disk at all times, from the first answer on.
  2. Nothing is computed twice. answers.jsonl is append-only; a rerun resumes.
  3. One bad question cannot take down the run. Failures are recorded and skipped.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from pathlib import Path

from .config import Config
from .answer import answer_question
from .llm import LLM
from .schema import Doc, DocIndexEntry, QuestionResult


def load_done(path: str | Path) -> dict[str, QuestionResult]:
    p = Path(path)
    if not p.exists():
        return {}
    done: dict[str, QuestionResult] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = QuestionResult.model_validate_json(line)
        except Exception:
            continue
        if not r.error:
            done[r.qid] = r
    return done


def append_result(path: str | Path, result: QuestionResult) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(result.model_dump_json() + "\n")


async def run(
    cfg: Config,
    questions: list[tuple[str, str]],
    docs: list[Doc],
    entries: list[DocIndexEntry],
    *,
    resume: bool = True,
) -> list[QuestionResult]:
    adapter = importlib.import_module(cfg.adapter)
    llm = LLM(cfg.model, cache_ttl=cfg.cache_ttl, max_attempts=cfg.max_attempts)
    docs_by_id = {d.id: d for d in docs}

    done = load_done(cfg.answers_file) if resume else {}
    if done:
        print(f"resuming: {len(done)} answers already on disk")
    pending = [(qid, q) for qid, q in questions if qid not in done]

    results: dict[str, QuestionResult] = dict(done)
    sem = asyncio.Semaphore(cfg.concurrency)
    write_lock = asyncio.Lock()
    started = time.monotonic()
    counter = {"n": 0}

    async def one(qid: str, question: str) -> None:
        t0 = time.monotonic()
        if cfg.budget_usd and llm.usage.cost_usd(cfg.cache_ttl) >= cfg.budget_usd:
            res = QuestionResult(qid=qid, question=question,
                                 error=f"skipped: budget ceiling ${cfg.budget_usd} reached")
            async with write_lock:
                results[qid] = res
                append_result(cfg.answers_file, res)
            return
        try:
            async with sem:
                res = await asyncio.wait_for(
                    answer_question(llm, qid, question, docs_by_id, entries, cfg),
                    timeout=cfg.per_question_timeout_s,
                )
        except asyncio.TimeoutError:
            res = QuestionResult(qid=qid, question=question,
                                 error=f"timeout after {cfg.per_question_timeout_s}s")
        except Exception as e:
            res = QuestionResult(qid=qid, question=question,
                                 error=f"{type(e).__name__}: {e}")
        res.elapsed_s = round(time.monotonic() - t0, 1)

        async with write_lock:
            results[qid] = res
            append_result(cfg.answers_file, res)
            counter["n"] += 1
            adapter.write_submission(list(results.values()), cfg.submission_file)
            flag = "!" if (res.error or not res.arith_ok) else " "
            print(f"[{counter['n']}/{len(pending)}]{flag} q={qid} {res.elapsed_s}s "
                  f"docs={len(res.docs_considered)} cites={len(res.citations)}"
                  + (f" ERROR {res.error}" if res.error else ""))

    await asyncio.gather(*(one(qid, q) for qid, q in pending))

    ordered = [results[qid] for qid, _ in questions if qid in results]
    adapter.write_submission(ordered, cfg.submission_file)

    elapsed = time.monotonic() - started
    report = {
        "questions": len(questions),
        "answered": sum(1 for r in ordered if not r.error),
        "errors": [r.qid for r in ordered if r.error],
        "arith_flagged": [r.qid for r in ordered if not r.arith_ok],
        "no_citations": [r.qid for r in ordered if not r.citations and not r.error],
        "low_confidence": [r.qid for r in ordered
                           if r.record and r.record.confidence < 0.5],
        "wall_clock_s": round(elapsed, 1),
        "usage": llm.usage.as_dict(),
        "estimated_cost_usd": round(llm.usage.cost_usd(cfg.cache_ttl), 2),
    }
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.out_dir, "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    return ordered
