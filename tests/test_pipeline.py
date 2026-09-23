from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from bi_agent import cli, pipeline
from bi_agent.crawler import Crawler
from bi_agent.errors import LLMOutputError, StageError
from bi_agent.llm import LLM, SearchHit
from bi_agent.models import (
    Analysis,
    Classification,
    EvidenceLedger,
    ExternalFindings,
    Identity,
    Narrative,
    RawFindings,
    SiteSignals,
    SourceKind,
    SourceType,
)
from tests.conftest import (
    SITE,
    analysis_payload,
    raw_findings_payload,
    response,
    search_result_block,
    stage_router,
    third_party_id,
    tool_use,
)


@pytest.fixture
def store(tmp_path: Path) -> pipeline.RunStore:
    return pipeline.RunStore(tmp_path / "run")


def test_stage_crawl_writes_pages_and_first_party_ledger(store, http_client):
    pages = pipeline.stage_crawl(store, SITE, Crawler(http_client, max_pages=5))
    assert len(pages) == 5
    ledger = store.ledger()
    assert len(ledger) == 5 and all(e.source_type is SourceType.FIRST_PARTY for e in ledger)
    assert store.meta()["pages"] == 5 and store.meta()["start"] == SITE + "/"
    assert ledger.get("E001").url == SITE + "/"
    assert "Acme sells" in ledger.get("E001").excerpt


def test_store_raises_before_inputs_exist(store):
    with pytest.raises(StageError):
        store.ledger()
    with pytest.raises(StageError):
        store.load_model("identity.json", Identity)
    with pytest.raises(StageError):
        store.load_json("run.json")


def test_pages_block_respects_budget_and_orders_by_score(store, http_client):
    pipeline.stage_crawl(store, SITE, Crawler(http_client, max_pages=6))
    ledger = store.ledger()
    block = pipeline._pages_block(store, ledger)
    assert block.index("[E001]") < block.index("/pricing")
    assert "JSON-LD" in block
    small = pipeline._pages_block(store, ledger, budget=700)
    assert 1 <= small.count("=== [") < block.count("=== [")


def test_verify_findings_rejects_unsearched_sources_and_reclassifies():
    ledger = EvidenceLedger()
    raw = RawFindings.model_validate(raw_findings_payload())
    raw.findings.append(raw.findings[0].model_copy(update={
        "classification": Classification.VERIFIED_FACT,
        "sources": [raw.findings[0].sources[0].model_copy(update={"url": SITE + "/about"})],
    }))
    raw.findings.append(raw.findings[0].model_copy(update={"classification": Classification.COMPANY_CLAIM}))
    hits = [SearchHit("https://news.test/acme-raises", "Acme raises")]
    accepted, rejected = pipeline.verify_findings(raw, hits, ledger, "acme-widgets.test", "2026-01-01T00:00:00+00:00")
    assert len(rejected) == 2 and any("made-up.test" in r for r in rejected)
    by_topic = {(f.topic.value, f.classification.value) for f in accepted}
    assert ("funding", "third_party_claim") in by_topic
    assert ("financials", "unknown") in by_topic
    assert ("funding", "company_claim") in by_topic  # verified_fact with only a first-party source is downgraded
    assert ("funding", "third_party_claim") in by_topic  # company_claim with only third-party evidence is re-labelled
    assert len(ledger) == 2
    assert ledger.get("E001").source_type is SourceType.THIRD_PARTY and ledger.get("E001").publisher == "News Test"
    assert ledger.get("E002").source_type is SourceType.FIRST_PARTY


def test_verify_findings_fills_title_and_publisher_from_hit():
    ledger = EvidenceLedger()
    raw = RawFindings.model_validate({"findings": [{
        "topic": "news", "statement": "s", "classification": "third_party_claim",
        "sources": [{"url": "https://x.test/p", "title": "", "publisher": "", "excerpt": "e",
                     "source_kind": "news"}]}], "not_found": []})
    accepted, _ = pipeline.verify_findings(raw, [SearchHit("https://x.test/p", "Hit title")], ledger, "acme.test", "t")
    assert accepted and ledger.get("E001").title == "Hit title" and ledger.get("E001").publisher == "x.test"


GROUPS = {"funding": {"funding": "rounds"}, "news": {"news": "press"}}


def _server_error() -> Exception:
    import anthropic
    import httpx

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.InternalServerError("overloaded", response=httpx.Response(500, request=req), body=None)


def _no_credit() -> Exception:
    import anthropic
    import httpx

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": "Your credit balance is too low"}}
    return anthropic.BadRequestError("400", response=httpx.Response(400, request=req, json=body), body=body)


def _run_to(store, http_client, stage: str, client=None) -> LLM:
    llm = LLM(client or stage_router(), model="m")
    pipeline.stage_crawl(store, SITE, Crawler(http_client, max_pages=6))
    if stage == "crawl":
        return llm
    pipeline.stage_identify(store, llm)
    if stage == "identify":
        return llm
    pipeline.stage_signals(store, llm)
    if stage == "signals":
        return llm
    pipeline.stage_research(store, llm, groups=GROUPS)
    if stage == "research":
        return llm
    pipeline.stage_resolve(store, llm)
    if stage == "resolve":
        return llm
    pipeline.stage_analyze(store, llm)
    if stage == "analyze":
        return llm
    pipeline.stage_narrate(store, llm)
    return llm


def test_identify_and_signals_stage_outputs(store, http_client):
    _run_to(store, http_client, "signals")
    ident = store.load_model("identity.json", Identity)
    assert ident.company_name.value == "Acme Widgets"
    sig = store.load_model("signals.json", SiteSignals)
    assert sig.offerings[0].name == "Monitor"


def test_identify_repairs_unknown_evidence_without_another_call(store, http_client, caplog):
    from tests.conftest import identity_payload

    bad = identity_payload()
    bad["company_name"]["evidence_ids"] = ["E999"]
    client = stage_router({"submit_identity": lambda kw: response(tool_use("submit_identity", bad))})
    caplog.set_level(logging.DEBUG, logger="bi_agent")
    llm = _run_to(store, http_client, "identify", client)
    ident = store.load_model("identity.json", Identity)
    assert ident.company_name.evidence_ids == [] and ident.company_name.classification is Classification.ANALYTICAL_INFERENCE
    assert llm.calls == 1
    assert any("unknown evidence id E999" in r.message for r in caplog.records)


def test_identify_and_signals_share_a_cacheable_prefix(store, http_client):
    llm = _run_to(store, http_client, "signals")
    first, second = llm.client.messages.calls[:2]
    assert first["system"] == second["system"] and first["tools"] == second["tools"]
    assert first["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert first["tool_choice"]["name"] == "submit_identity" and second["tool_choice"]["name"] == "submit_site_signals"
    assert all(c["cache_control"] == {"type": "ephemeral"} and c["thinking"] == {"type": "disabled"}
               for c in llm.client.messages.calls)


def test_research_stage_extends_ledger_and_survives_topic_failure(store, http_client):
    calls = {"n": 0}

    def flaky(kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _server_error()
        return response(search_result_block(["https://news.test/acme-raises"]),
                        tool_use("submit_findings", raw_findings_payload()))

    _run_to(store, http_client, "research", stage_router({"submit_findings": flaky}))
    res = store.load_model("findings.json", ExternalFindings)
    assert any("research call failed" in x for x in res.not_found)
    assert any("earnings reports" in x for x in res.not_found)
    assert len(res.findings) == 2 and res.rejected
    ledger = store.ledger()
    assert any(e.source_type is SourceType.THIRD_PARTY for e in ledger)
    assert not store.exists("research.partial.json")
    usage = store.load_json("usage.json")
    # the failed call never reached usage; "news" and one follow-up round did, and the follow-up
    # found nothing new, so research stopped there
    assert usage["stages"]["research"]["calls"] == 2 and "identify" in usage["stages"]


def test_research_stops_on_permanent_error_and_resumes(store, http_client):
    calls = {"n": 0}

    def second_group_has_no_credit(kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise _no_credit()
        return response(search_result_block(["https://news.test/acme-raises"]),
                        tool_use("submit_findings", raw_findings_payload()))

    import anthropic

    llm = _run_to(store, http_client, "signals", stage_router({"submit_findings": second_group_has_no_credit}))
    with pytest.raises(anthropic.BadRequestError):
        pipeline.stage_research(store, llm, groups=GROUPS)
    assert calls["n"] == 2  # stopped at once, no call for the groups after it
    assert store.load_json("research.partial.json")["done"] == ["funding"]
    with pytest.raises(StageError, match="research is unfinished .*funding.*re-run the research stage"):
        pipeline.stage_resolve(store, llm)
    res = pipeline.stage_research(store, llm, groups=GROUPS)  # resumes: "news", then one follow-up round
    assert calls["n"] == 4
    assert len(res.findings) == 2  # the same statements found again are not duplicated
    assert not store.exists("research.partial.json")


def test_research_fails_when_every_group_fails(store, http_client):
    def down(kw):
        raise _server_error()

    llm = _run_to(store, http_client, "signals", stage_router({"submit_findings": down}))
    with pytest.raises(StageError, match="every topic group"):
        pipeline.stage_research(store, llm, groups=GROUPS)


def test_resolve_refines_identity_with_external_evidence(store, http_client):
    from tests.conftest import identity_payload

    def resolved(kw):
        p = identity_payload()
        if "EXTERNAL FINDINGS" in kw["messages"][0]["content"]:  # the resolve call, not identify
            assert "Acme raised a Series A" in kw["messages"][0]["content"]
            p["legal_name"] = {"value": "Acme Widgets Ltda", "classification": "verified_fact",
                               "evidence_ids": [third_party_id(kw)]}
        return response(tool_use("submit_identity", p))

    llm = _run_to(store, http_client, "resolve", stage_router({"submit_identity": resolved}))
    site = store.load_model("identity.site.json", Identity)
    final = store.load_model("identity.json", Identity)
    assert site.legal_name.value is None or site.legal_name.value != "Acme Widgets Ltda"
    assert final.legal_name.value == "Acme Widgets Ltda"
    # one news source cannot verify a fact: the claim is kept as a third-party claim
    assert final.legal_name.classification is Classification.THIRD_PARTY_CLAIM
    assert "resolve" in store.load_json("usage.json")["stages"]
    pipeline.stage_analyze(store, llm)
    pipeline.stage_narrate(store, llm)
    assert "| Legal entity | Acme Widgets Ltda *(Third-party claim)*" in pipeline.stage_report(store)
    # re-running starts again from the website identity, and old runs without identity.site.json still work
    store.path("identity.site.json").unlink()
    pipeline.stage_resolve(store, llm)
    assert store.exists("identity.site.json")


def test_analyze_and_narrate_then_report(store, http_client):
    _run_to(store, http_client, "narrate")
    analysis = store.load_model("analysis.json", Analysis)
    assert len(analysis.strategic) == 10
    narrative = store.load_model("narrative.json", Narrative)
    assert len(narrative.executive_summary) == 5
    md = pipeline.stage_report(store)
    assert md == store.path("report.md").read_text()
    for i in range(1, 22):
        assert f"\n## {i}. " in md, f"section {i} missing"
    assert "| Company | Acme Widgets *(Company claim)* [E001] |" in md
    assert "| Rival Co | direct |" in md
    assert "Series A raised *(Third-party claim)*" in md
    assert "| E001 |" in md and "news.test/acme-raises" in md
    assert "Discarded during verification" in md
    assert "Research gap: funding: earnings reports" in md
    assert "No credible sourced market-size figure" in md
    assert "Identity uncertainties" in md


def test_report_labels_offerings_judgments_and_access_dates(store, http_client):
    _run_to(store, http_client, "narrate")
    ledger = store.ledger()
    ev = ledger.get("E001")
    ev_dict = [e.model_dump(mode="json") for e in ledger]
    ev_dict[0]["retrieved_at"] = "2020-02-03T04:05:06+00:00"
    store.save_json("evidence.json", ev_dict)
    md = pipeline.stage_report(store)
    assert "| Monitor | core product | Factory operators |" in md
    strategic = md.split("### Strategic analysis")[1].split("### Business maturity")[0]
    assert "analytical judgments" in strategic
    assert strategic.count("*(Analytical inference)*") == 10
    assert f"| {ev.id} |" in md and "| 2020-02-03 |" in md.split("## 21. Sources")[1]


def test_offering_kind_is_required_for_the_model_but_old_runs_still_load():
    from bi_agent.models import Offering, OfferingKind

    assert "kind" in Offering.model_json_schema()["required"]
    legacy = {"name": "X", "target_customer": "t", "problem_solved": "p", "key_capabilities": "k",
              "business_benefit": "b", "monetization": "m", "classification": "company_claim", "evidence_ids": ["E001"]}
    assert Offering.model_validate(legacy).kind is OfferingKind.OTHER


def test_report_renders_sizing_and_omits_uncited_sources(store, http_client):
    def with_sizing(kw):
        p = analysis_payload(third_party_id(kw))
        p["market"]["sizing"] = [{"metric": "TAM", "value": "$2B", "year": "2024", "methodology": "bottom-up",
                                  "limitations": "vendor estimate", "evidence_ids": ["E001"]}]
        return response(tool_use("submit_analysis", p))

    client = stage_router({"submit_analysis": with_sizing})
    _run_to(store, http_client, "narrate", client)
    md = pipeline.stage_report(store)
    assert "| TAM | $2B | 2024 | bottom-up | vendor estimate | [E001] |" in md
    sources = md.split("## 21. Sources")[1]
    assert "[E001]" not in sources.split("|---")[0]
    assert "| E006 |" not in sources  # a crawled page nobody cited is not listed


def test_analyze_downgrades_unsupported_classification_without_retry(store, http_client):
    def bad_fact(kw):
        p = analysis_payload(third_party_id(kw))
        p["swot"]["strengths"][0]["classification"] = "verified_fact"  # cites only first-party E001
        p["swot"]["strengths"][0]["statement"] += " [E999]"
        return response(tool_use("submit_analysis", p))

    llm = _run_to(store, http_client, "analyze", stage_router({"submit_analysis": bad_fact}))
    strength = store.load_model("analysis.json", Analysis).swot.strengths[0]
    assert strength.classification is Classification.COMPANY_CLAIM and "[E999]" not in strength.statement
    assert [(c.get("tool_choice") or {}).get("name") for c in llm.client.messages.calls].count("submit_analysis") == 1


def test_analyze_structural_failure_surfaces_errors(store, http_client):
    def missing_question(kw):
        p = analysis_payload(third_party_id(kw))
        p["strategic"][0]["question"] = "Something else?"
        return response(tool_use("submit_analysis", p))

    client = stage_router({"submit_analysis": missing_question})
    with pytest.raises(LLMOutputError) as exc:
        _run_to(store, http_client, "analyze", client)
    assert any("strategic answers must cover" in e for e in exc.value.errors)


def test_run_all_produces_report(store, http_client, caplog):
    caplog.set_level(logging.INFO, logger="bi_agent")
    md = pipeline.run_all(store, SITE, Crawler(http_client, max_pages=4), LLM(stage_router(), model="m"))
    assert md.startswith("# Company Intelligence Report: Acme Widgets")
    messages = [r.getMessage() for r in caplog.records]
    for i, (name, _) in enumerate(pipeline.STAGES, 1):
        assert any(m.startswith(f"[{i}/{len(pipeline.STAGES)}] {name}: done in") for m in messages)
    assert any(m.startswith("research 1/") and "searching" in m for m in messages)
    assert json.loads(store.path("analysis.json").read_text())["open_questions"]


# --------------------------------------------------------------------------- CLI


def test_cli_run_and_stage_by_stage(tmp_path, http_client, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    out = str(tmp_path / "r")
    factory = lambda: stage_router()  # noqa: E731
    common = ["--out", out, "--max-pages", "4", "--delay", "0"]
    assert cli.main(common + ["crawl", "--url", SITE], client_factory=factory, http_client=http_client) == 0
    assert "crawled 4 pages" in capsys.readouterr().out
    for stage in ("identify", "signals", "research", "resolve", "analyze", "narrate", "report"):
        assert cli.main(common + [stage], client_factory=factory, http_client=http_client) == 0, stage
    assert "report written" in capsys.readouterr().out
    assert (tmp_path / "r" / "report.md").exists()
    assert (tmp_path / "r" / "report.pdf").read_bytes().startswith(b"%PDF")
    assert cli.main(common + ["-v", "run", "--url", SITE], client_factory=factory, http_client=http_client) == 0


def test_cli_requires_api_key_for_llm_stages(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main(["--out", str(tmp_path), "identify"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_cli_reports_api_errors_without_traceback(tmp_path, http_client, monkeypatch, capsys):
    import anthropic
    import httpx

    def broke(_kw):
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        body = {"type": "error", "error": {"type": "invalid_request_error", "message": "Your credit balance is too low"}}
        raise anthropic.BadRequestError("400", response=httpx.Response(400, request=req, json=body), body=body)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    rc = cli.main(["--out", str(tmp_path / "r"), "--max-pages", "2", "--delay", "0", "run", "--url", SITE],
                  client_factory=lambda: stage_router({"submit_identity": broke}), http_client=http_client)
    err = capsys.readouterr().err
    assert rc == 3 and "400: Your credit balance is too low" in err and "re-run the failed stage" in err


def test_cli_reports_stage_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main(["--out", str(tmp_path), "report"]) == 2
    assert "missing" in capsys.readouterr().err


def test_cli_prints_llm_output_errors(tmp_path, http_client, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    bad = {"submit_identity": lambda kw: response(tool_use("submit_identity", {"nonsense": True}))}
    rc = cli.main(["--out", str(tmp_path), "--delay", "0", "run", "--url", SITE],
                  client_factory=lambda: stage_router(bad), http_client=http_client)
    assert rc == 2
    err = capsys.readouterr().err
    assert "failed validation" in err and "nonsense" in err


def test_cli_make_crawler_defaults(monkeypatch):
    args = cli.build_parser().parse_args(["--out", "x", "crawl", "--url", "u"])
    crawler = cli.make_crawler(args)
    assert crawler.max_pages == 60 and crawler.delay == 0.5


# --------------------------------------------------------------------------- follow-up research, thin sites


def _findings(n: int, prefix: str, url: str = "https://news.test/acme-raises") -> dict:
    return {"findings": [{"topic": "corporate", "statement": f"{prefix} finding {i}", "classification": "third_party_claim",
                          "sources": [{"url": url, "title": "t", "publisher": "p", "excerpt": "e", "source_kind": "news"}]}
                         for i in range(n)], "not_found": []}


def test_followup_rounds_continue_while_productive_and_stop_at_diminishing_returns(store, http_client):
    answers = iter([_findings(2, "group-a"), _findings(2, "group-b"), _findings(5, "round-1"), _findings(1, "round-2")])
    prompts_seen = []

    def research(kw):
        prompts_seen.append(kw["messages"][0]["content"])
        return response(search_result_block(["https://news.test/acme-raises"]),
                        tool_use("submit_findings", next(answers)))

    llm = _run_to(store, http_client, "signals", stage_router({"submit_findings": research}))
    res = pipeline.stage_research(store, llm, groups=GROUPS, followup_rounds=5)
    assert len(prompts_seen) == 4  # 2 groups, round 1 (5 new, continue), round 2 (1 new, stop)
    assert "Follow-up task" in prompts_seen[2] and "group-a finding 0" in prompts_seen[2]
    assert "round-1 finding 4" in prompts_seen[3]
    assert len(res.findings) == 10
    tools = {t["name"]: t for t in llm.client.messages.calls[-1]["tools"]}
    assert tools["web_fetch"]["type"] == "web_fetch_20260209" and tools["web_fetch"]["max_uses"] == 3


def test_thin_site_gets_more_searches_and_a_note_in_the_report(store, http_client):
    llm = _run_to(store, http_client, "signals")
    meta = store.meta()
    meta.update(thin_site=True, site_chars=900, js_rendered_pages=3)
    store.save_json("run.json", meta)
    pipeline.stage_research(store, llm, groups=GROUPS, followup_rounds=0)
    research_calls = [c for c in llm.client.messages.calls if any(t["name"] == "web_search" for t in c.get("tools", []))]
    assert {t["max_uses"] for c in research_calls for t in c["tools"] if t["name"] == "web_search"} == {15}
    assert len(research_calls) == 3  # two groups plus the one extra follow-up round for thin sites
    assert "little crawlable text" in research_calls[0]["messages"][0]["content"]
    pipeline.stage_resolve(store, llm)
    pipeline.stage_analyze(store, llm)
    pipeline.stage_narrate(store, llm)
    md = pipeline.stage_report(store)
    assert "> The website yielded little crawlable text (900 characters; 3 pages" in md


def test_crawl_records_site_profile(store, http_client):
    pipeline.stage_crawl(store, SITE, Crawler(http_client, max_pages=6))
    meta = store.meta()
    assert meta["site_chars"] > 0 and meta["js_rendered_pages"] == 0 and meta["thin_site"] is True
    assert all(e.source_kind is SourceKind.OFFICIAL_COMPANY for e in store.ledger())


def test_research_searches_under_related_names_and_records_source_kinds(store, http_client):
    from tests.conftest import identity_payload

    def product_site(kw):
        p = identity_payload()
        p["brands"] = {"value": "WidgetCloud", "classification": "company_claim", "evidence_ids": ["E001"]}
        p["website_subject"] = {"value": "the product WidgetCloud of Acme Widgets", "classification": "company_claim",
                                "evidence_ids": ["E001"]}
        return response(tool_use("submit_identity", p))

    llm = _run_to(store, http_client, "research", stage_router({"submit_identity": product_site}))
    first_research = next(c for c in llm.client.messages.calls if any(t["name"] == "web_search" for t in c["tools"]))
    content = first_research["messages"][0]["content"]
    assert "search these too): WidgetCloud" in content and "the product WidgetCloud of Acme Widgets" in content
    news = next(e for e in store.ledger() if e.url == "https://news.test/acme-raises")
    assert news.source_kind is SourceKind.NEWS
    pipeline.stage_resolve(store, llm)
    pipeline.stage_analyze(store, llm)
    pipeline.stage_narrate(store, llm)
    md = pipeline.stage_report(store)
    assert "| Website represents | the product WidgetCloud of Acme Widgets *(Company claim)*" in md
    assert "| news |" in md.split("## 21. Sources")[1]


def test_verify_findings_lowers_implausible_primary_record_claims():
    ledger = EvidenceLedger()
    raw = RawFindings.model_validate({"findings": [
        {"topic": "corporate", "statement": "Registered as Acme Ltda.", "classification": "verified_fact",
         "sources": [{"url": "https://cnpj-lookup.test/123", "title": "t", "publisher": "p", "excerpt": "e",
                      "source_kind": "government_regulatory"}]},
        {"topic": "corporate", "statement": "Registered in the state registry.", "classification": "verified_fact",
         "sources": [{"url": "https://www.gov.br/receita/123", "title": "t", "publisher": "p", "excerpt": "e",
                      "source_kind": "government_regulatory"}]},
    ], "not_found": []})
    hits = [SearchHit("https://cnpj-lookup.test/123", "t"), SearchHit("https://www.gov.br/receita/123", "t")]
    accepted, _ = pipeline.verify_findings(raw, hits, ledger, "acme.test", "t")
    assert accepted[0].classification is Classification.THIRD_PARTY_CLAIM  # a lookup site is not the registry
    assert ledger.get(accepted[0].evidence_ids[0]).source_kind is SourceKind.DATABASE_AGGREGATOR
    assert accepted[1].classification is Classification.VERIFIED_FACT
