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
import uuid
from datetime import datetime, timezone
from pathlib import Path
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


def fetch_tool_type(model: str) -> str:
    return "web_fetch_20250910" if _BASIC_SEARCH_MODELS.search(model) else "web_fetch_20260209"


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
    retrieval_method: str = "web_search"
    retrieved_at: str = ""
    content: str | None = None
    requested_url: str | None = None
    content_limitation: str | None = "search result text unavailable"


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


def _decode_json_strings(node: Any) -> Any:
    """Undo a known glitch in large tool inputs: a list or object sent as a JSON-encoded string
    (``"findings": "[{...}]"``). Strings that do not parse as a JSON list/object are left alone.
    Parsing is lenient about raw line breaks inside strings, which such payloads often contain."""
    if isinstance(node, dict):
        return {k: _decode_json_strings(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_decode_json_strings(v) for v in node]
    if isinstance(node, str) and node.lstrip()[:1] in ("[", "{"):
        try:
            value = json.loads(node, strict=False)
        except ValueError:
            return node
        return _decode_json_strings(value) if isinstance(value, (list, dict)) else node
    return node


def _keys(payload: Any) -> str:
    return ", ".join(list(payload)[:8]) if isinstance(payload, dict) else type(payload).__name__


def _at(node: Any, loc: tuple) -> tuple[Any, Any] | None:
    """(container, key) for a validation-error location, or None if the path is gone."""
    for part in loc[:-1]:
        try:
            node = node[part]
        except (KeyError, IndexError, TypeError):
            return None
    return (node, loc[-1]) if isinstance(node, (dict, list)) else None


def _normalize(schema: type[BaseModel], payload: Any, errors: list[dict]) -> tuple[Any, list[str]]:
    """Mechanical repairs for structural slips in a large tool input, so they cost no extra call.

    - fields wrapped in one outer key (``{"identity": {...fields...}}``) are unwrapped;
    - a duplicated key the API renamed (``website_subject_2``) is dropped when the original exists,
      renamed back when it does not;
    - a string over its length limit is cut at the last word boundary before the limit.
    Each repair is returned as a note. Anything else is left for the model to fix.
    """
    import copy

    payload = copy.deepcopy(payload)
    notes: list[str] = []
    fields = set(schema.model_fields)
    if isinstance(payload, dict) and not fields & set(payload):
        inner = [v for v in payload.values() if isinstance(v, dict) and len(fields & set(v)) >= len(fields) / 2]
        if len(inner) == 1:
            notes.append(f"unwrapped fields sent inside {next(k for k, v in payload.items() if v is inner[0])!r}")
            return inner[0], notes
    for e in errors:
        loc, kind = tuple(e.get("loc", ())), e.get("type")
        found = _at(payload, loc) if loc else None
        if found is None:
            continue
        parent, key = found
        if kind == "extra_forbidden" and isinstance(parent, dict) and isinstance(key, str):
            base = re.sub(r"_\d+$", "", key)
            if base != key and key in parent:
                if base in parent:
                    notes.append(f"dropped duplicate key {key!r}")
                    parent.pop(key)
                else:
                    notes.append(f"renamed {key!r} to {base!r}")
                    parent[base] = parent.pop(key)
        elif kind == "string_too_long" and isinstance(parent[key], str):
            limit = (e.get("ctx") or {}).get("max_length")
            if limit:
                cut = parent[key][: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
                notes.append(f"shortened {'.'.join(map(str, loc))} from {len(parent[key])} to {len(cut)} characters")
                parent[key] = cut
    return payload, notes


class LLM:
    def __init__(
        self,
        client: Any,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 64_000,
        max_attempts: int = 3,
        max_search_uses: int = 10,
        max_fetch_uses: int = 3,
        max_fetch_tokens: int = 20_000,
        max_research_turns: int = 4,
        thinking: dict | None = None,
        run_budget: dict | None = None,
        input_token_allowance: int = 1_000_000,
    ) -> None:
        if max_attempts < 1 or max_research_turns < 1:
            raise ValueError("max_attempts and max_research_turns must be >= 1")
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.max_search_uses = max_search_uses
        self.max_fetch_uses = max_fetch_uses
        self.max_fetch_tokens = max_fetch_tokens
        self.max_research_turns = max_research_turns
        # Thinking is off by default: the output is a forced tool call validated in code, and the
        # previous default model ran without thinking. Pass {"type": "adaptive"} to turn it on.
        self.thinking = thinking or {"type": "disabled"}
        self.calls = 0
        self.usage = UsageLog(model)
        self.accounting = None
        self.audit = None
        self.run_budget = run_budget or {}
        self.input_token_allowance = input_token_allowance
        self.debug_dir: Path | None = None  # where rejected tool inputs are saved, for diagnosis

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
        call_id = self.accounting.reserve(label, kwargs) if self.accounting else None
        self.calls += 1
        try:
            with self.client.messages.stream(
                model=self.model, max_tokens=self.max_tokens, thinking=self.thinking,
                cache_control={"type": "ephemeral"}, **kwargs,
            ) as stream:
                response = stream.get_final_message()
        except BaseException as exc:
            if self.accounting:
                self.accounting.finish(call_id, error=f"{type(exc).__name__}: final usage unavailable")
            raise
        usage = Usage.from_response(label, response)
        self.usage.calls.append(usage)
        if self.accounting:
            self.accounting.finish(call_id, usage if _attr(response, "usage") is not None else None)
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
        """URLs the server tools actually retrieved: search results and successfully fetched pages."""
        hits: list[SearchHit] = []
        stamp = datetime.now(timezone.utc).isoformat()
        requests = {_attr(b, "id"): _attr(_attr(b, "input", {}), "url")
                    for b in _attr(response, "content", []) or [] if _attr(b, "type") == "server_tool_use"}
        for block in _attr(response, "content", []) or []:
            if _attr(block, "type") == "web_fetch_tool_result":
                result = _attr(block, "content")
                if _attr(result, "type") == "web_fetch_result" and _attr(result, "url"):
                    doc = _attr(result, "content")
                    hits.append(SearchHit(url=_attr(result, "url"), title=_attr(doc, "title", "") or "",
                                          retrieval_method="web_fetch", retrieved_at=_attr(result, "retrieved_at", stamp),
                                          requested_url=requests.get(_attr(block, "tool_use_id")),
                                          content=(_attr(_attr(doc, "source"), "data")
                                                   if _attr(_attr(doc, "source"), "type") == "text" else None),
                                          content_limitation=(None if _attr(_attr(doc, "source"), "type") == "text"
                                                              else "document text unavailable")))
                continue  # anything else is an error object (url_not_accessible, too many uses ...)
            if _attr(block, "type") != "web_search_tool_result":
                continue
            content = _attr(block, "content", [])
            if not isinstance(content, list):
                continue  # error object, e.g. max_uses_exceeded
            for r in content:
                if _attr(r, "type") == "web_search_result" and _attr(r, "url"):
                    hits.append(SearchHit(url=_attr(r, "url"), title=_attr(r, "title", "") or "",
                                          page_age=_attr(r, "page_age"), retrieved_at=stamp,
                                          content=_attr(r, "text") or _attr(r, "snippet"),
                                          content_limitation=None if (_attr(r, "text") or _attr(r, "snippet"))
                                          else "search discovery only; encrypted/unavailable source text"))
        return hits

    def _save_rejected(self, tool_name: str, payload: Any, errors: list[str]) -> None:
        if self.debug_dir is None:
            return
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            path = self.debug_dir / f"{tool_name}-{uuid.uuid4().hex}.json"
            path.write_text(json.dumps({"errors": errors, "input": payload}, indent=1, ensure_ascii=False,
                                       default=str), encoding="utf-8")
        except OSError as exc:  # diagnosis must never break a run
            log.debug("could not save rejected input: %s", exc)

    @staticmethod
    def _retry_messages(response: Any, block: Any, tool_name: str, errors: list[str]) -> list[dict]:
        """The assistant turn plus a user turn answering *every* tool call in it.

        The API rejects a conversation in which any ``tool_use`` block lacks a ``tool_result`` in
        the next message, and a reply can hold several calls (the same tool twice, or a second
        declared tool), so each one gets a result: the validation errors for the call that was
        read, a short note for the others.
        """
        results = []
        for b in _attr(response, "content", []) or []:
            if _attr(b, "type") != "tool_use":
                continue
            if _attr(b, "id") == _attr(block, "id"):
                text = "Validation failed. Fix every item and call the tool again:\n" + "\n".join(
                    f"- {e}" for e in errors[:40])
            else:
                text = f"Ignored: only one call to {tool_name} is expected, with the complete result."
            results.append({"type": "tool_result", "tool_use_id": _attr(b, "id"), "is_error": True, "content": text})
        return [
            {"role": "assistant", "content": [_block_to_dict(b) for b in _attr(response, "content")]},
            {"role": "user", "content": results},
        ]

    def _parse(self, schema: type[T], payload: Any, check: SemanticCheck | None) -> tuple[T | None, list[str]]:
        """Validate ``payload``; on failure try the mechanical repairs (JSON strings, wrapped or
        duplicated keys, over-long strings) a few times before giving the errors back."""
        obj = None
        current = payload
        for _round in range(3):
            try:
                obj = schema.model_validate(current)
                break
            except ValidationError as exc:
                decoded = _decode_json_strings(current)
                fixed, notes = _normalize(schema, decoded, exc.errors())
                if decoded != current:
                    notes.insert(0, "decoded list/object fields sent as JSON strings")
                if not notes:
                    return None, _format_errors(exc)
                log.info("repaired the tool input without another call: %s", "; ".join(notes))
                if self.audit:
                    self.audit(current, fixed, notes)
                current = fixed
        if obj is None:
            try:
                obj = schema.model_validate(current)
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
                tool_choice={"type": "tool", "name": tool_name, "disable_parallel_tool_use": True},
            )
            block = self._find_tool_use(response, tool_name)
            if block is None:
                last_errors = ["model did not call the output tool"]
                messages = messages + [{"role": "user", "content": f"Call the {tool_name} tool now."}]
                continue
            payload = _attr(block, "input")
            if repair is not None:
                original = payload
                payload, notes = repair(payload)
                if notes and self.audit:
                    self.audit(original, payload, notes)
                if notes:
                    log.info("%s: %d mechanical fixes applied without another model call (-v lists them)",
                             tool_name, len(notes))
                for note in notes:
                    log.debug("%s: repaired %s", tool_name, note)
            obj, last_errors = self._parse(schema, payload, semantic_check)
            if obj is not None:
                return obj
            self._save_rejected(tool_name, payload, last_errors)
            log.info("%s: output failed validation (%d problems, first: %s; top-level keys: %s); asking again",
                     tool_name, len(last_errors), last_errors[0] if last_errors else "?", _keys(payload))
            messages = messages + self._retry_messages(response, block, tool_name, last_errors)
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
        tool_description: str = "Submit findings with the exact source URLs returned by web_search or web_fetch.",
        allowed_domains: list[str] | None = None,
        max_search_uses: int | None = None,
    ) -> tuple[T, list[SearchHit]]:
        submit = self._tool(tool_name, schema, tool_description)
        search: dict = {"type": search_tool_type(self.model), "name": "web_search",
                        "max_uses": max_search_uses or self.max_search_uses}
        if allowed_domains:
            search["allowed_domains"] = allowed_domains
        tools: list[dict] = [search]
        if self.max_fetch_uses:
            # web_fetch reads a page the search surfaced (a filing, an annual report) in full.
            fetch: dict = {"type": fetch_tool_type(self.model), "name": "web_fetch",
                           "max_uses": self.max_fetch_uses, "max_content_tokens": self.max_fetch_tokens}
            if allowed_domains:
                fetch["allowed_domains"] = allowed_domains
            tools.append(fetch)
        tools.append(submit)
        messages: list[dict] = [{"role": "user", "content": user}]
        hits: list[SearchHit] = []
        last_errors: list[str] = []
        force = False
        for _turn in range(self.max_research_turns):
            kwargs: dict = dict(system=system, messages=messages, tools=tools)
            if force and search_tool_type(self.model) == "web_search_20250305":
                # Forcing is only safe with the basic search tool. The dynamic-filtering version
                # (web_search_20260209) runs code on the server (programmatic tool calling), which the
                # API refuses to combine with forced or single-call tool_choice; there the prompt asks
                # for the call and every extra call is answered by _retry_messages.
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
            self._save_rejected(tool_name, _attr(block, "input"), last_errors)
            log.info("%s: output failed validation (%d problems, first: %s; top-level keys: %s); asking again",
                     tool_name, len(last_errors), last_errors[0] if last_errors else "?", _keys(_attr(block, "input")))
            messages = messages + self._retry_messages(response, block, tool_name, last_errors)
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

    return anthropic.Anthropic(api_key=api_key, max_retries=0) if api_key else anthropic.Anthropic(max_retries=0)


def pretty(obj: BaseModel) -> str:
    """Compact JSON for prompts: indentation costs tokens and tells the model nothing."""
    return json.dumps(obj.model_dump(mode="json", exclude_none=True), separators=(",", ":"), ensure_ascii=False)
