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
    assert [h.url for h in hits] == ["https://a.test/1", "https://b.test/2"]
    assert all(h.retrieved_at and h.content and h.retrieval_method == "web_search" for h in hits)
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
    assert len(hits) == 1 and hits[0].url == "https://filings.test/annual.pdf"
    assert hits[0].retrieved_at == "2026-09-01" and hits[0].retrieval_method == "web_fetch"
    assert hits[0].content is None and hits[0].content_limitation
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


class Card(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(max_length=20)
    count: int
    note: str = ""


@pytest.mark.parametrize("payload,expected", [
    ({"card": {"name": "a", "count": 1}}, Card(name="a", count=1)),                        # wrapped in an outer key
    ({"name": "a", "count": 1, "count_2": 1}, Card(name="a", count=1)),                   # duplicated key
    ({"name": "a", "note_2": "n", "count": 1}, Card(name="a", count=1, note="n")),        # renamed key
    ({"name": "a very long name that goes on", "count": 1}, Card(name="a very long name…", count=1)),
])
def test_structural_slips_are_repaired_without_another_call(payload, expected):
    client = scripted([response(tool_use("submit", payload))])
    llm = LLM(client, model="m", max_attempts=1)
    assert llm.structured(system="s", user="u", schema=Card) == expected and llm.calls == 1


def test_json_string_with_raw_line_breaks_is_decoded():
    class Many(BaseModel):
        model_config = ConfigDict(extra="forbid")
        items: list[Card]

    client = scripted([response(tool_use("submit", {"items": '[{"name": "a", "count": 1, "note": "line\nbreak"}]'}))])
    assert LLM(client, model="m", max_attempts=1).structured(system="s", user="u", schema=Many).items[0].note == "line\nbreak"


def test_rejected_inputs_are_saved_for_diagnosis(tmp_path):
    client = scripted([response(tool_use("submit", {"nope": 1})), response(tool_use("submit", {"name": "a", "count": 1}))])
    llm = LLM(client, model="m", max_attempts=2)
    llm.debug_dir = tmp_path / "debug"
    llm.structured(system="s", user="u", schema=Card)
    saved = list((tmp_path / "debug").glob("submit-*.json"))
    assert len(saved) == 1 and '"nope": 1' in saved[0].read_text()



def test_json_string_with_key_equals_slip_is_decoded():
    from bi_agent.llm import _decode_json_strings

    broken = '{"value": "x", "classification="company_claim", "evidence_ids": ["E001"]}'
    assert _decode_json_strings({"a": broken})["a"] == {"value": "x", "classification": "company_claim",
                                                         "evidence_ids": ["E001"]}
    assert _decode_json_strings({"a": "{not json"})["a"] == "{not json"


class _BreakingMessages:
    """A fake client whose stream breaks mid-response ``breaks`` times, then succeeds."""

    def __init__(self, breaks: int, exc: BaseException, reply) -> None:
        self.breaks, self.exc, self.reply, self.calls = breaks, exc, reply, []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        outer = self

        class _Stream:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return None

            def get_final_message(self):
                if len(outer.calls) <= outer.breaks:
                    raise outer.exc
                return outer.reply

        return _Stream()


def _dropped() -> BaseException:
    try:
        import httpx2 as h
    except ImportError:
        import httpx as h
    return h.RemoteProtocolError("peer closed connection without sending complete message body")


def test_stream_that_breaks_mid_response_is_retried():
    from types import SimpleNamespace

    msgs = _BreakingMessages(2, _dropped(), response(tool_use("submit", {"name": "a", "count": 1})))
    llm = LLM(SimpleNamespace(messages=msgs), model="m")
    waits = []
    llm._sleep = waits.append
    assert llm.structured(system="s", user="u", schema=Out).name == "a"
    assert len(msgs.calls) == 3 and waits == [5.0, 30.0]


def test_stream_retries_are_bounded_and_permanent_errors_are_not_retried():
    from types import SimpleNamespace

    import anthropic
    import httpx

    msgs = _BreakingMessages(5, _dropped(), None)
    llm = LLM(SimpleNamespace(messages=msgs), model="m")
    llm._sleep = lambda s: None
    with pytest.raises(Exception, match="peer closed"):
        llm.structured(system="s", user="u", schema=Out)
    assert len(msgs.calls) == 3
    req = httpx.Request("POST", "https://x.test")
    no_credit = anthropic.BadRequestError("400", response=httpx.Response(400, request=req), body=None)
    msgs = _BreakingMessages(5, no_credit, None)
    llm = LLM(SimpleNamespace(messages=msgs), model="m")
    with pytest.raises(anthropic.BadRequestError):
        llm.structured(system="s", user="u", schema=Out)
    assert len(msgs.calls) == 1



def test_single_object_sent_for_a_list_is_wrapped():
    class Many(BaseModel):
        model_config = ConfigDict(extra="forbid")
        items: list[Card]

    client = scripted([response(tool_use("submit", {"items": {"name": "a", "count": 1}}))])
    assert LLM(client, model="m", max_attempts=1).structured(system="s", user="u", schema=Many).items[0].name == "a"



def test_whole_output_packed_into_one_field_is_unwrapped():
    import json as _json

    packed = {"count": _json.dumps({"name": "a", "count": 1})}  # every field, as a string inside "count"
    client = scripted([response(tool_use("submit", packed))])
    assert LLM(client, model="m", max_attempts=1).structured(system="s", user="u", schema=Card) == Card(name="a", count=1)


def test_premises_on_an_output_without_that_field_are_dropped():
    client = scripted([response(tool_use("submit", {"name": "a", "count": 1, "premises": [{"evidence_id": "E1"}]}))])
    assert LLM(client, model="m", max_attempts=1).structured(system="s", user="u", schema=Card) == Card(name="a", count=1)
