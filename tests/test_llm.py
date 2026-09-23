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
    assert call["tool_choice"] == {"type": "tool", "name": "submit"}
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
    assert tools[0]["type"] == "web_search_20250305" and tools[0]["max_uses"] == 3
    assert tools[0]["allowed_domains"] == ["a.test"]
    assert "tool_choice" not in client.messages.calls[0]


def test_researched_forces_tool_after_text_only_turn_and_pause():
    client = scripted([
        response(search_result_block(["https://a.test/1"]), text_block("thinking"), stop_reason="pause_turn"),
        response(text_block("done searching"), stop_reason="end_turn"),
        response(tool_use("submit_findings", {"name": "a", "count": 1})),
    ])
    llm = LLM(client, model="m", max_research_turns=4)
    out, hits = llm.researched(system="s", user="u", schema=Out)
    assert out.name == "a" and [h.url for h in hits] == ["https://a.test/1"]
    assert "tool_choice" not in client.messages.calls[1]
    assert client.messages.calls[2]["tool_choice"] == {"type": "tool", "name": "submit_findings"}


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
    assert '"name": "a"' in pretty(Out(name="a", count=1))
