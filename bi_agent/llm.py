"""Thin wrapper around the Anthropic Messages API that only returns validated models.

Two entry points:

* :meth:`LLM.structured` — forces a tool call whose input schema is the pydantic model,
  parses it with ``extra="forbid"``, runs an optional semantic check, and retries a
  bounded number of times with the validation errors fed back.
* :meth:`LLM.researched` — same, but the model may first use Anthropic's server-side
  ``web_search`` tool. Every URL the search returned is captured so the caller can reject
  findings whose sources were never actually retrieved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import LLMOutputError

T = TypeVar("T", bound=BaseModel)
SemanticCheck = Callable[[Any], list[str]]

DEFAULT_MODEL = "claude-sonnet-4-5"


@dataclass(frozen=True)
class SearchHit:
    url: str
    title: str
    page_age: str | None = None


def _attr(block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _block_to_dict(block: Any) -> dict:
    if isinstance(block, dict):
        return block
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    return {k: v for k, v in vars(block).items() if v is not None}


def _format_errors(exc: ValidationError) -> list[str]:
    out = []
    for e in exc.errors():
        loc = ".".join(str(x) for x in e.get("loc", ()))
        out.append(f"{loc}: {e.get('msg')}")
    return out


class LLM:
    def __init__(
        self,
        client: Any,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 16_000,
        max_attempts: int = 3,
        max_search_uses: int = 8,
        max_research_turns: int = 4,
    ) -> None:
        if max_attempts < 1 or max_research_turns < 1:
            raise ValueError("max_attempts and max_research_turns must be >= 1")
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.max_search_uses = max_search_uses
        self.max_research_turns = max_research_turns
        self.calls = 0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _tool(name: str, schema: type[BaseModel], description: str) -> dict:
        return {"name": name, "description": description, "input_schema": schema.model_json_schema()}

    def _create(self, **kwargs: Any) -> Any:
        self.calls += 1
        return self.client.messages.create(model=self.model, max_tokens=self.max_tokens, **kwargs)

    @staticmethod
    def _find_tool_use(response: Any, name: str) -> Any | None:
        for block in _attr(response, "content", []) or []:
            if _attr(block, "type") == "tool_use" and _attr(block, "name") == name:
                return block
        return None

    @staticmethod
    def _collect_hits(response: Any) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for block in _attr(response, "content", []) or []:
            if _attr(block, "type") != "web_search_tool_result":
                continue
            content = _attr(block, "content", [])
            if not isinstance(content, list):
                continue  # error object, e.g. max_uses_exceeded
            for r in content:
                if _attr(r, "type") == "web_search_result" and _attr(r, "url"):
                    hits.append(SearchHit(url=_attr(r, "url"), title=_attr(r, "title", "") or "",
                                          page_age=_attr(r, "page_age")))
        return hits

    def _parse(self, schema: type[T], payload: Any, check: SemanticCheck | None) -> tuple[T | None, list[str]]:
        try:
            obj = schema.model_validate(payload)
        except ValidationError as exc:
            return None, _format_errors(exc)
        errors = check(obj) if check else []
        return (obj, []) if not errors else (None, errors)

    # ------------------------------------------------------------------ structured
    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        tool_name: str = "submit",
        tool_description: str = "Submit the completed, fully populated result.",
        semantic_check: SemanticCheck | None = None,
    ) -> T:
        tool = self._tool(tool_name, schema, tool_description)
        messages: list[dict] = [{"role": "user", "content": user}]
        last_errors: list[str] = []
        for _attempt in range(self.max_attempts):
            response = self._create(
                system=system, messages=messages, tools=[tool],
                tool_choice={"type": "tool", "name": tool_name},
            )
            block = self._find_tool_use(response, tool_name)
            if block is None:
                last_errors = ["model did not call the output tool"]
                messages = messages + [{"role": "user", "content": f"Call the {tool_name} tool now."}]
                continue
            obj, last_errors = self._parse(schema, _attr(block, "input"), semantic_check)
            if obj is not None:
                return obj
            messages = messages + [
                {"role": "assistant", "content": [_block_to_dict(b) for b in _attr(response, "content")]},
                {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": _attr(block, "id"), "is_error": True,
                    "content": "Validation failed. Fix every item and call the tool again:\n"
                    + "\n".join(f"- {e}" for e in last_errors[:40]),
                }]},
            ]
        raise LLMOutputError(
            f"{tool_name}: output failed validation after {self.max_attempts} attempts", last_errors
        )

    # ------------------------------------------------------------------ researched
    def researched(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        tool_name: str = "submit_findings",
        tool_description: str = "Submit findings with the exact source URLs returned by web_search.",
        allowed_domains: list[str] | None = None,
    ) -> tuple[T, list[SearchHit]]:
        submit = self._tool(tool_name, schema, tool_description)
        search: dict = {"type": "web_search_20250305", "name": "web_search", "max_uses": self.max_search_uses}
        if allowed_domains:
            search["allowed_domains"] = allowed_domains
        messages: list[dict] = [{"role": "user", "content": user}]
        hits: list[SearchHit] = []
        last_errors: list[str] = []
        force = False
        for _turn in range(self.max_research_turns):
            kwargs: dict = dict(system=system, messages=messages, tools=[search, submit])
            if force:
                kwargs["tool_choice"] = {"type": "tool", "name": tool_name}
            response = self._create(**kwargs)
            hits.extend(self._collect_hits(response))
            content = [_block_to_dict(b) for b in _attr(response, "content", []) or []]
            block = self._find_tool_use(response, tool_name)
            if block is None:
                if _attr(response, "stop_reason") == "pause_turn":
                    messages = messages + [{"role": "assistant", "content": content}]
                    continue
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"Searching is over. Call {tool_name} now with what you found."},
                ]
                force = True
                last_errors = ["model did not call the findings tool"]
                continue
            obj, last_errors = self._parse(schema, _attr(block, "input"), None)
            if obj is not None:
                return obj, hits
            messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": _attr(block, "id"), "is_error": True,
                    "content": "Validation failed. Fix every item and call the tool again:\n"
                    + "\n".join(f"- {e}" for e in last_errors[:40]),
                }]},
            ]
            force = True
        raise LLMOutputError(
            f"{tool_name}: no valid findings after {self.max_research_turns} turns", last_errors
        )


def build_client(api_key: str | None = None) -> Any:
    """Construct the real Anthropic client. Imported lazily so tests never need the SDK network path."""
    import anthropic

    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def pretty(obj: BaseModel) -> str:
    return json.dumps(obj.model_dump(mode="json"), indent=1, ensure_ascii=False)
