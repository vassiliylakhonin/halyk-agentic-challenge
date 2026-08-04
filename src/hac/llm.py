"""Anthropic client wrapper: streaming answers with citations, structured parsing,
usage accounting, and a client-side refusal fallback.

Deliberately stays on the plain (non-beta) Messages API on the hot path. During a
three-hour scored window, fewer beta surfaces means fewer ways to eat a 400 at
minute one. Refusal handling is done client-side: on stop_reason == "refusal" the
same request is replayed on FALLBACK_MODEL.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Type, TypeVar

import anthropic
from pydantic import BaseModel

FALLBACK_MODEL = "claude-opus-4-8"

# USD per million tokens. Update if pricing moves; only used for the cost report.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

T = TypeVar("T", bound=BaseModel)


class RefusedError(RuntimeError):
    """Both the primary and the fallback model declined the request."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0
    per_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, model: str, u: Any) -> None:
        self.calls += 1
        it = getattr(u, "input_tokens", 0) or 0
        ot = getattr(u, "output_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        self.input_tokens += it
        self.output_tokens += ot
        self.cache_write_tokens += cw
        self.cache_read_tokens += cr
        m = self.per_model.setdefault(model, {"in": 0, "out": 0, "cw": 0, "cr": 0, "calls": 0})
        m["in"] += it
        m["out"] += ot
        m["cw"] += cw
        m["cr"] += cr
        m["calls"] += 1

    def cost_usd(self, cache_ttl: str = "1h") -> float:
        write_mult = 2.0 if cache_ttl == "1h" else 1.25
        total = 0.0
        for model, m in self.per_model.items():
            pin, pout = PRICES.get(model, (5.0, 25.0))
            total += m["in"] / 1e6 * pin
            total += m["cw"] / 1e6 * pin * write_mult
            total += m["cr"] / 1e6 * pin * 0.1
            total += m["out"] / 1e6 * pout
        return total

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "per_model": self.per_model,
        }


class LLM:
    def __init__(self, model: str, cache_ttl: str = "1h", max_attempts: int = 3):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it (or put it in .env and "
                "source it) before running the pipeline."
            )
        self.client = anthropic.AsyncAnthropic(max_retries=4, timeout=600.0)
        self.model = model
        self.cache_ttl = cache_ttl
        self.max_attempts = max_attempts
        self.usage = Usage()

    def cache_control(self) -> dict:
        return {"type": "ephemeral", "ttl": self.cache_ttl}

    # ------------------------------------------------------------------ text

    async def answer(
        self,
        *,
        system: list[dict],
        content: list[dict],
        max_tokens: int,
        effort: str,
    ) -> tuple[str, list[dict], Any]:
        """Streamed answer turn. Returns (prose, raw_content_blocks, message)."""
        last: Exception | None = None
        for model in (self.model, FALLBACK_MODEL):
            for attempt in range(self.max_attempts):
                try:
                    async with self.client.messages.stream(
                        model=model,
                        max_tokens=max_tokens,
                        system=system,
                        messages=[{"role": "user", "content": content}],
                        output_config={"effort": effort},
                    ) as stream:
                        msg = await stream.get_final_message()
                    self.usage.add(model, msg.usage)
                    if msg.stop_reason == "refusal":
                        break  # try the fallback model
                    blocks = [b.model_dump() for b in msg.content]
                    prose = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                    return prose, blocks, msg
                except (anthropic.APIConnectionError, anthropic.InternalServerError) as e:
                    last = e
                    await asyncio.sleep(2 * (attempt + 1))
                except anthropic.RateLimitError as e:
                    last = e
                    await asyncio.sleep(5 * (attempt + 1))
        if last is not None:
            raise last
        raise RefusedError("both the primary and the fallback model refused this request")

    # ------------------------------------------------------------ structured

    async def parse(
        self,
        *,
        schema: Type[T],
        system: str,
        content: list[dict] | str,
        max_tokens: int,
        effort: str,
    ) -> T:
        """Structured turn via output_config.format. Never combine with citations."""
        messages = [{"role": "user", "content": content}]
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                resp = await self.client.messages.parse(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                    output_format=schema,
                    output_config={"effort": effort},
                )
                self.usage.add(self.model, resp.usage)
                if resp.stop_reason == "refusal":
                    raise RefusedError("structuring turn refused")
                parsed = getattr(resp, "parsed_output", None)
                if parsed is None:
                    raise ValueError("structured output came back empty")
                return parsed
            except (anthropic.APIConnectionError, anthropic.InternalServerError,
                    anthropic.RateLimitError, ValueError) as e:
                last = e
                await asyncio.sleep(3 * (attempt + 1))
        raise last or RuntimeError("structured call failed")
