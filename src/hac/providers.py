"""Model access, behind one small interface.

The pipeline asks a model for three things only: a covenant specification, a set
of auditor adjustments, and the text of a page that arrived as an image. Each
returns JSON against a schema. Keeping that surface narrow is what makes the
provider swappable, and it is why switching from one vendor to another costs an
afternoon rather than a rewrite.

Model names are not hard-coded. The account is asked what it actually has, and
the first match from a preference list is used, so a name that was renamed or
retired does not take the run down at 11:00 on competition day.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

# Ordered by preference. Matching is by prefix, so a dated snapshot such as
# "gpt-5.1-2026-03-01" satisfies the "gpt-5.1" entry.
OPENAI_PREFERENCE = ("gpt-5.2", "gpt-5.1", "gpt-5", "o4", "gpt-4.1", "gpt-4o")
ANTHROPIC_PREFERENCE = ("claude-opus-5", "claude-sonnet-5", "claude-opus-4-8",
                        "claude-haiku-4-5")


class Chat(Protocol):
    name: str

    def json(self, *, system: str, user: str, schema: dict,
             images: list[tuple[str, bytes]] | None = None) -> dict:
        ...


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def add(self, model: str, i: int, o: int) -> None:
        self.calls += 1
        self.input_tokens += i
        self.output_tokens += o
        self.by_model[model] = self.by_model.get(model, 0) + 1


def _pick(available: list[str], preference: tuple[str, ...]) -> str | None:
    for want in preference:
        matches = sorted(m for m in available if m.startswith(want))
        if matches:
            return matches[0]
    return None


class OpenAIChat:
    """Chat Completions with a JSON-schema response format."""

    def __init__(self, model: str | None = None, usage: Usage | None = None):
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Put it in .env in the repository root."
            )
        self.client = OpenAI()
        self.usage = usage or Usage()
        self.name = model or self._discover()

    def _discover(self) -> str:
        try:
            available = [m.id for m in self.client.models.list().data]
        except Exception as e:
            raise RuntimeError(f"could not list models: {type(e).__name__}: {e}") from e
        chosen = _pick(available, OPENAI_PREFERENCE)
        if chosen is None:
            raise RuntimeError(
                "none of the preferred models are available on this account; "
                f"it offers {sorted(available)[:20]}"
            )
        return chosen

    def json(self, *, system: str, user: str, schema: dict,
             images: list[tuple[str, bytes]] | None = None) -> dict:
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for media_type, blob in images or []:
            data = base64.standard_b64encode(blob).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            })

        response = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema, "strict": False},
            },
        )
        u = getattr(response, "usage", None)
        self.usage.add(self.name,
                       getattr(u, "prompt_tokens", 0) or 0,
                       getattr(u, "completion_tokens", 0) or 0)
        text = response.choices[0].message.content or "{}"
        return json.loads(text)


class AnthropicChat:
    """The same interface over the Messages API, kept so the pipeline is not
    tied to one vendor. Unused while the account in play is OpenAI."""

    def __init__(self, model: str | None = None, usage: Usage | None = None):
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic()
        self.usage = usage or Usage()
        self.name = model or ANTHROPIC_PREFERENCE[0]

    def json(self, *, system: str, user: str, schema: dict,
             images: list[tuple[str, bytes]] | None = None) -> dict:
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for media_type, blob in images or []:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type,
                           "data": base64.standard_b64encode(blob).decode("ascii")},
            })
        response = self.client.messages.create(
            model=self.name,
            max_tokens=8000,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        u = response.usage
        self.usage.add(self.name, u.input_tokens or 0, u.output_tokens or 0)
        text = "".join(b.text for b in response.content if b.type == "text")
        return json.loads(text or "{}")


def build(provider: str = "openai", model: str | None = None,
          usage: Usage | None = None) -> Chat:
    if provider == "openai":
        return OpenAIChat(model, usage)
    if provider == "anthropic":
        return AnthropicChat(model, usage)
    raise ValueError(f"unknown provider: {provider}")
