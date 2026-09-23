"""Thin wrapper around the Anthropic Messages API that only returns validated models.

Two entry points:

* :meth:`LLM.structured` — forces a tool call whose input schema is the pydantic model,
  parses it with ``extra="forbid"``, runs an optional semantic check, and retries a
  bounded number of times with the validation errors fed back.
* :meth:`LLM.researched` — same, but the model may first use Anthropic's server-side
  ``web_search`` tool. Every URL the search returned is captured so the caller can reject
  findings whose sources were never actually retrieved.

Every request uses automatic prompt caching, so retries and research continuations re-read
their unchanged prefix at the cached rate instead of paying for it again. Token usage and
web-search counts are recorded per call (:attr:`LLM.usage`) so a run's cost can be measured.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import LLMOutputError

T = TypeVar("T", bound=BaseModel)
SemanticCheck = Callable[[Any], list[str]]
Repair = Callable[[Any], tuple[Any, list[str]]]
log = logging.getLogger("bi_agent")

DEFAULT_MODEL = "claude-sonnet-5"

# USD per million tokens (input, output) and per web search, for the cost estimate in usage.json.
# Cache writes bill at 1.25x input, cache reads at 0.1x input.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
}
PRICE_PER_SEARCH = 10.0 / 1000
# Models that predate the dynamic-filtering web search tool (web_search_20260209).
_BASIC_SEARCH_MODELS = re.compile(r"haiku|claude-3|-4-[015]\b|-4-[015]-|sonnet-4-5|opus-4-5")


def search_tool_type(model: str) -> str:
    return "web_search_20250305" if _BASIC_SEARCH_MODELS.search(model) else "web_search_20260209"


@dataclass
class Usage:
    """Token and search counts for one API call."""

    label: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    web_search_requests: int = 0
    stop_reason: str | None = None

    @classmethod
    def from_response(cls, label: str, response: Any) -> "Usage":
        u = _attr(response, "usage")
        server = _attr(u, "server_tool_use") if u is not None else None

        def num(obj: Any, name: str) -> int:
            v = _attr(obj, name, 0) if obj is not None else 0
            return v if isinstance(v, int) else 0

        return cls(
            label=label, input_tokens=num(u, "input_tokens"), output_tokens=num(u, "output_tokens"),
            cache_creation_input_tokens=num(u, "cache_creation_input_tokens"),
            cache_read_input_tokens=num(u, "cache_read_input_tokens"),
            web_search_requests=num(server, "web_search_requests"),
            stop_reason=_attr(response, "stop_reason"),
        )

    def cost(self, model: str) -> float | None:
        price = PRICES.get(model)
        if price is None:
            return None
        inp, out = price
        return (
            self.input_tokens * inp + self.cache_creation_input_tokens * inp * 1.25
            + self.cache_read_input_tokens * inp * 0.1 + self.output_tokens * out
        ) / 1_000_000 + self.web_search_requests * PRICE_PER_SEARCH


@dataclass
class UsageLog:
    model: str
    calls: list[Usage] = field(default_factory=list)

    def summary(self) -> dict:
        keys = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                "web_search_requests")
        total = {k: sum(getattr(c, k) for c in self.calls) for k in keys}
        costs = [c.cost(self.model) for c in self.calls]
        total["calls"] = len(self.calls)
        total["estimated_cost_usd"] = round(sum(costs), 4) if None not in costs else None
        return total


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
        max_tokens: int = 64_000,
        max_attempts: int = 3,
        max_search_uses: int = 10,
        max_research_turns: int = 4,
        thinking: dict | None = None,
    ) -> None:
        if max_attempts < 1 or max_research_turns < 1:
            raise ValueError("max_attempts and max_research_turns must be >= 1")
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.max_search_uses = max_search_uses
        self.max_research_turns = max_research_turns
        # Thinking is off by default: the output is a forced tool call validated in code, and the
        # previous default model ran without thinking. Pass {"type": "adaptive"} to turn it on.
        self.thinking = thinking or {"type": "disabled"}
        self.calls = 0
        self.usage = UsageLog(model)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _tool(name: str, schema: type[BaseModel], description: str) -> dict:
        return {"name": name, "description": description, "input_schema": schema.model_json_schema()}

    def _create(self, label: str, **kwargs: Any) -> Any:
        """One API call. Streamed, because the SDK refuses non-streaming requests whose
        ``max_tokens`` could take over ten minutes; the full message is assembled before returning.

        A response cut off by ``max_tokens`` raises at once: its tool input is truncated, and
        asking again with the same limit would pay for the same truncation again.
        """
        self.calls += 1
        with self.client.messages.stream(
            model=self.model, max_tokens=self.max_tokens, thinking=self.thinking,
            cache_control={"type": "ephemeral"},  # caches the longest unchanged prefix automatically
            **kwargs,
        ) as stream:
            response = stream.get_final_message()
        usage = Usage.from_response(label, response)
        self.usage.calls.append(usage)
        if usage.stop_reason == "max_tokens":
            raise LLMOutputError(
                f"{label}: output was cut off at max_tokens={self.max_tokens}; re-run with a higher --max-tokens",
                [f"stop_reason max_tokens after {usage.output_tokens} output tokens"],
            )
        return response

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
        repair: Repair | None = None,
        shared_tools: list[tuple[str, type[BaseModel], str]] | None = None,
    ) -> T:
        """Force ``tool_name`` and return its validated input.

        ``repair`` runs on the raw tool input before validation and may fix mechanical problems
        (for example citations of evidence ids that do not exist) without another model call;
        the notes it returns are logged. ``shared_tools`` declares extra tools so that several
        stages send an identical tool list and can share a cached prefix; only ``tool_name`` is
        ever forced. ``system`` may be a string or a list of text blocks.
        """
        tool = self._tool(tool_name, schema, tool_description)
        tools = [tool]
        if shared_tools:
            tools = [self._tool(n, sc, d) for n, sc, d in shared_tools]
            if tool_name not in {t["name"] for t in tools}:
                tools.append(tool)
        messages: list[dict] = [{"role": "user", "content": user}]
        last_errors: list[str] = []
        for _attempt in range(self.max_attempts):
            response = self._create(
                tool_name, system=system, messages=messages, tools=tools,
                tool_choice={"type": "tool", "name": tool_name},
            )
            block = self._find_tool_use(response, tool_name)
            if block is None:
                last_errors = ["model did not call the output tool"]
                messages = messages + [{"role": "user", "content": f"Call the {tool_name} tool now."}]
                continue
            payload = _attr(block, "input")
            if repair is not None:
                payload, notes = repair(payload)
                for note in notes:
                    log.warning("%s: repaired %s", tool_name, note)
            obj, last_errors = self._parse(schema, payload, semantic_check)
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
        search: dict = {"type": search_tool_type(self.model), "name": "web_search", "max_uses": self.max_search_uses}
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
            response = self._create(tool_name, **kwargs)
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


def is_transient(exc: BaseException) -> bool:
    """True for failures worth skipping past (the next call may succeed); False for ones that will
    repeat on every call (no credit, invalid key, unknown model, malformed request)."""
    import anthropic

    if isinstance(exc, (LLMOutputError, anthropic.APIConnectionError)):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def build_client(api_key: str | None = None) -> Any:
    """Construct the real Anthropic client. Imported lazily so tests never need the SDK network path."""
    import anthropic

    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def pretty(obj: BaseModel) -> str:
    """Compact JSON for prompts: indentation costs tokens and tells the model nothing."""
    return json.dumps(obj.model_dump(mode="json", exclude_none=True), separators=(",", ":"), ensure_ascii=False)
