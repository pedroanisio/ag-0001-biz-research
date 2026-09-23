from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, Field

from bi_agent.errors import LLMOutputError
from bi_agent.llm import LLM, SearchHit, _block_to_dict, pretty
from tests.conftest import response, scripted, search_result_block, text_block, tool_use


class Out(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    count: int


def test_structured_returns_validated_model_and_forces_tool():
    client = scripted([response(tool_use("submit", {"name": "a", "count": 1}))])
    llm = LLM(client, model="m")
    out = llm.structured(system="s", user="u", schema=Out)
    assert out == Out(name="a", count=1)
    call = client.messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "submit", "disable_parallel_tool_use": True}
    assert call["tools"][0]["input_schema"]["type"] == "object"
    assert llm.calls == 1


def test_structured_retries_with_errors_then_succeeds():
    client = scripted([
        response(tool_use("submit", {"name": "", "count": "x", "extra": 1}, "tu_9")),
        response(tool_use("submit", {"name": "ok", "count": 2})),
    ])
    out = LLM(client, model="m", max_attempts=2).structured(system="s", user="u", schema=Out)
    assert out.count == 2
    second = client.messages.calls[1]["messages"]
    assert second[1]["role"] == "assistant"
    result = second[2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "tu_9" and result["is_error"]
    assert "extra" in result["content"] and "count" in result["content"]


def test_structured_semantic_check_feeds_back_and_bounds():
    client = scripted([response(tool_use("submit", {"name": "a", "count": 1}))] * 3)
    llm = LLM(client, model="m", max_attempts=3)
    with pytest.raises(LLMOutputError) as exc:
        llm.structured(system="s", user="u", schema=Out, semantic_check=lambda o: ["count must be 2"])
    assert exc.value.errors == ["count must be 2"] and llm.calls == 3


def test_structured_handles_missing_tool_call():
    client = scripted([response(text_block("chatter")), response(tool_use("submit", {"name": "a", "count": 1}))])
    assert LLM(client, model="m", max_attempts=2).structured(system="s", user="u", schema=Out).name == "a"
    assert "Call the submit tool now" in client.messages.calls[1]["messages"][-1]["content"]


def test_structured_exhausts_attempts_without_tool_call():
    client = scripted([response(text_block("no")), response(text_block("no"))])
    with pytest.raises(LLMOutputError, match="after 2 attempts"):
        LLM(client, model="m", max_attempts=2).structured(system="s", user="u", schema=Out)


def test_constructor_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        LLM(object(), max_attempts=0)
    with pytest.raises(ValueError):
        LLM(object(), max_research_turns=0)


def test_researched_collects_hits_and_result():
    client = scripted([
        response(search_result_block(["https://a.test/1", "https://b.test/2"]), tool_use("submit_findings", {"name": "a", "count": 1}))
    ])
    llm = LLM(client, model="m", max_search_uses=3)
    out, hits = llm.researched(system="s", user="u", schema=Out, allowed_domains=["a.test"])
    assert out.name == "a"
    assert hits == [SearchHit("https://a.test/1", "Title of https://a.test/1", "2025-01-01"),
                    SearchHit("https://b.test/2", "Title of https://b.test/2", "2025-01-01")]
    tools = client.messages.calls[0]["tools"]
    assert tools[0]["type"] == "web_search_20260209" and tools[0]["max_uses"] == 3
    assert tools[0]["allowed_domains"] == ["a.test"]
    assert "tool_choice" not in client.messages.calls[0]


def test_researched_forces_tool_after_text_only_turn_and_pause():
    client = scripted([
        response(search_result_block(["https://a.test/1"]), text_block("thinking"), stop_reason="pause_turn"),
        response(text_block("done searching"), stop_reason="end_turn"),
        response(tool_use("submit_findings", {"name": "a", "count": 1})),
    ])
    llm = LLM(client, model="claude-sonnet-4-5", max_research_turns=4)  # basic web search: forcing is allowed
    out, hits = llm.researched(system="s", user="u", schema=Out)
    assert out.name == "a" and [h.url for h in hits] == ["https://a.test/1"]
    assert "tool_choice" not in client.messages.calls[1]
    assert client.messages.calls[2]["tool_choice"] == {"type": "tool", "name": "submit_findings"}


def test_researched_never_forces_the_tool_with_dynamic_filtering_search():
    # web_search_20260209 uses programmatic tool calling; the API rejects forced/single-call tool_choice with it
    client = scripted([
        response(text_block("done searching"), stop_reason="end_turn"),
        response(tool_use("submit_findings", {"name": "", "count": 1}, "tu_1")),
        response(tool_use("submit_findings", {"name": "a", "count": 1})),
    ])
    out, _ = LLM(client, model="claude-sonnet-5").researched(system="s", user="u", schema=Out)
    assert out.name == "a"
    assert all("tool_choice" not in c for c in client.messages.calls)
    assert "Call submit_findings now" in client.messages.calls[1]["messages"][-1]["content"]


def test_list_sent_as_json_string_is_decoded_without_another_call():
    class Many(BaseModel):
        model_config = ConfigDict(extra="forbid")
        items: list[Out]

    client = scripted([response(tool_use("submit", {"items": '[{"name": "a", "count": 1}]'}))])
    llm = LLM(client, model="m")
    assert llm.structured(system="s", user="u", schema=Many).items[0].name == "a" and llm.calls == 1
    bad = scripted([response(tool_use("submit", {"items": "[not json"}))] * 2)
    with pytest.raises(LLMOutputError):
        LLM(bad, model="m", max_attempts=2).structured(system="s", user="u", schema=Many)
    still_bad = scripted([response(tool_use("submit", {"items": '[{"name": ""}]'}))])
    with pytest.raises(LLMOutputError):
        LLM(still_bad, model="m", max_attempts=1).structured(system="s", user="u", schema=Many)


def test_researched_ignores_search_error_blocks():
    from types import SimpleNamespace

    err = SimpleNamespace(type="web_search_tool_result", content=SimpleNamespace(type="web_search_tool_result_error",
                                                                                  error_code="max_uses_exceeded"))
    client = scripted([response(err, tool_use("submit_findings", {"name": "a", "count": 1}))])
    _, hits = LLM(client, model="m").researched(system="s", user="u", schema=Out)
    assert hits == []


def test_researched_retries_validation_then_exhausts():
    bad = response(tool_use("submit_findings", {"name": "", "count": 1}, "tu_2"))
    client = scripted([bad, bad])
    with pytest.raises(LLMOutputError, match="no valid findings after 2 turns") as exc:
        LLM(client, model="m", max_research_turns=2).researched(system="s", user="u", schema=Out)
    assert any("name" in e for e in exc.value.errors)
    assert client.messages.calls[1]["messages"][-1]["content"][0]["tool_use_id"] == "tu_2"


def test_block_to_dict_handles_dicts_models_and_namespaces():
    class B(BaseModel):
        type: str = "text"
        text: str = "hi"
        cache: None = None

    assert _block_to_dict({"a": 1}) == {"a": 1}
    assert _block_to_dict(B()) == {"type": "text", "text": "hi"}
    assert _block_to_dict(text_block("x")) == {"type": "text", "text": "x"}
    assert pretty(Out(name="a", count=1)) == '{"name":"a","count":1}'


def test_search_tool_version_follows_model():
    from bi_agent.llm import search_tool_type

    assert search_tool_type("claude-sonnet-5") == "web_search_20260209"
    assert search_tool_type("claude-opus-4-8") == "web_search_20260209"
    for old in ("claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1", "claude-sonnet-4-0"):
        assert search_tool_type(old) == "web_search_20250305", old


def test_usage_is_recorded_and_priced():
    from types import SimpleNamespace

    usage = SimpleNamespace(input_tokens=1000, output_tokens=500, cache_creation_input_tokens=2000,
                            cache_read_input_tokens=10_000, server_tool_use=SimpleNamespace(web_search_requests=3))
    reply = SimpleNamespace(content=[tool_use("submit", {"name": "a", "count": 1})], stop_reason="tool_use", usage=usage)
    llm = LLM(scripted([reply]), model="claude-sonnet-5")
    llm.structured(system="s", user="u", schema=Out)
    total = llm.usage.summary()
    assert total["calls"] == 1 and total["cache_read_input_tokens"] == 10_000 and total["web_search_requests"] == 3
    # 1000*2 + 2000*2.5 + 10000*0.2 + 500*10 per million, plus 3 searches at $0.01
    assert total["estimated_cost_usd"] == round((2000 + 5000 + 2000 + 5000) / 1e6 + 0.03, 4)
    assert LLM(object(), model="unknown-model").usage.summary()["estimated_cost_usd"] == 0


def test_unknown_model_has_no_cost_estimate():
    from bi_agent.llm import Usage, UsageLog

    assert UsageLog("x", [Usage("a", input_tokens=5)]).summary()["estimated_cost_usd"] is None


def test_truncated_response_fails_at_once_without_retry():
    client = scripted([response(tool_use("submit", {"name": "a"}), stop_reason="max_tokens")] * 3)
    llm = LLM(client, model="m", max_tokens=100, max_attempts=3)
    with pytest.raises(LLMOutputError, match="cut off at max_tokens=100; re-run with a higher --max-tokens"):
        llm.structured(system="s", user="u", schema=Out)
    assert llm.calls == 1 and llm.usage.calls[0].stop_reason == "max_tokens"


def test_is_transient_classifies_errors():
    import anthropic
    import httpx

    from bi_agent.llm import is_transient

    req = httpx.Request("POST", "https://x.test")

    def status(code):
        return anthropic.APIStatusError("e", response=httpx.Response(code, request=req), body=None)

    assert is_transient(status(429)) and is_transient(status(529))
    assert not is_transient(status(400)) and not is_transient(status(401))
    assert is_transient(anthropic.APIConnectionError(request=req))
    assert is_transient(LLMOutputError("x", []))
    assert not is_transient(RuntimeError("bug"))


def test_researched_accepts_fetched_pages_as_hits_and_ignores_fetch_errors():
    from types import SimpleNamespace

    fetched = SimpleNamespace(type="web_fetch_tool_result", content=SimpleNamespace(
        type="web_fetch_result", url="https://filings.test/annual.pdf", retrieved_at="2026-09-01",
        content=SimpleNamespace(type="document", title="Annual report")))
    failed = SimpleNamespace(type="web_fetch_tool_result", content=SimpleNamespace(
        type="web_fetch_tool_error", error_code="url_not_accessible"))
    client = scripted([response(fetched, failed, tool_use("submit_findings", {"name": "a", "count": 1}))])
    _, hits = LLM(client, model="claude-sonnet-4-5", max_fetch_uses=2).researched(system="s", user="u", schema=Out)
    assert hits == [SearchHit("https://filings.test/annual.pdf", "Annual report", "2026-09-01")]
    tools = {t["name"]: t for t in client.messages.calls[0]["tools"]}
    assert tools["web_fetch"]["type"] == "web_fetch_20250910" and tools["web_fetch"]["max_uses"] == 2


def test_researched_without_fetch():
    client = scripted([response(tool_use("submit_findings", {"name": "a", "count": 1}))])
    LLM(client, model="m", max_fetch_uses=0).researched(system="s", user="u", schema=Out, max_search_uses=7)
    tools = client.messages.calls[0]["tools"]
    assert [t["name"] for t in tools] == ["web_search", "submit_findings"] and tools[0]["max_uses"] == 7


def _answers_every_tool_use(messages: list[dict]) -> bool:
    """The API rule behind the pbgas 400: each assistant tool_use needs a tool_result right after."""
    for i, m in enumerate(messages):
        if m["role"] != "assistant" or not isinstance(m["content"], list):
            continue
        ids = {b["id"] for b in m["content"] if b.get("type") == "tool_use"}
        nxt = messages[i + 1]["content"] if i + 1 < len(messages) else []
        answered = {b["tool_use_id"] for b in nxt if isinstance(b, dict) and b.get("type") == "tool_result"}
        if not ids <= answered:
            return False
    return True


def test_retry_answers_every_tool_call_in_the_rejected_reply():
    # the model called the output tool twice (and a second declared tool) in one reply; the first call is invalid
    client = scripted([
        response(tool_use("submit", {"name": "", "count": 1}, "tu_1"), tool_use("submit", {"name": "b", "count": 2}, "tu_2"),
                 tool_use("other", {"x": 1}, "tu_3")),
        response(tool_use("submit", {"name": "ok", "count": 3}, "tu_4")),
    ])
    out = LLM(client, model="m", max_attempts=2).structured(
        system="s", user="u", schema=Out, shared_tools=[("submit", Out, "d"), ("other", Out, "d")])
    assert out.count == 3
    retry = client.messages.calls[1]["messages"]
    assert _answers_every_tool_use(retry)
    results = {b["tool_use_id"]: b["content"] for b in retry[2]["content"]}
    assert "Validation failed" in results["tu_1"] and "Ignored" in results["tu_2"] and "Ignored" in results["tu_3"]


def test_research_retry_answers_every_tool_call():
    client = scripted([
        response(search_result_block(["https://a.test/1"]), tool_use("submit_findings", {"name": "", "count": 1}, "tu_1"),
                 tool_use("submit_findings", {"name": "x", "count": 1}, "tu_2")),
        response(tool_use("submit_findings", {"name": "a", "count": 1}, "tu_3")),
    ])
    out, _ = LLM(client, model="m").researched(system="s", user="u", schema=Out)
    assert out.name == "a" and _answers_every_tool_use(client.messages.calls[1]["messages"])
