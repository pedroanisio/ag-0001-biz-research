"""Shared fixtures: a fake in-memory website, a fake Anthropic client, and valid stage payloads."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Callable

import httpx
import pytest

from bi_agent.models import MATURITY_DIMENSIONS, STRATEGIC_QUESTIONS

SITE = "https://www.acme-widgets.test"

PAGES: dict[str, str] = {
    "/": """<html lang="en"><head><title>Acme Widgets</title>
        <meta name="description" content="Acme sells industrial widget monitoring software.">
        <script type="application/ld+json">{"@type":"Organization","name":"Acme Widgets Inc","foundingDate":"2015"}</script>
        <script>var x = 1;</script><style>body{}</style></head>
        <body><nav><a href="/about">About</a><a href="/pricing">Pricing</a><a href="/products/monitor">Product</a>
        <a href="/careers">Careers</a><a href="/private/secret">Secret</a><a href="https://other.test/x">Off-site</a>
        <a href="/brochure.pdf">PDF</a><a href="mailto:a@b.c">mail</a><a href="#top">top</a>
        <a href="/about?utm_source=x">About dup</a></nav>
        <main>Acme Widgets builds monitoring software for factories.</main></body></html>""",
    "/about": "<html><head><title>About Acme</title></head><body>Founded in 2015 in Austin, Texas by Jane Doe.</body></html>",
    "/pricing": "<html><head><title>Pricing</title></head><body>Starter $49/month. Enterprise: contact sales.</body></html>",
    "/products/monitor": "<html><head><title>Monitor</title></head><body>Real-time widget telemetry with API access.</body></html>",
    "/careers": "<html><head><title>Careers</title></head><body>Hiring: Enterprise Account Executive, ML Engineer.</body></html>",
    "/private/secret": "<html><body>should be blocked by robots</body></html>",
    "/from-sitemap": "<html><head><title>Sitemap page</title></head><body>Only discoverable via sitemap.</body></html>",
    "/robots.txt": "User-agent: *\nDisallow: /private/\nSitemap: https://www.acme-widgets.test/sitemap.xml\n",
    "/sitemap.xml": """<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://www.acme-widgets.test/from-sitemap</loc></url>
        <url><loc>https://www.acme-widgets.test/image.png</loc></url></urlset>""",
}


def site_handler(request: httpx.Request) -> httpx.Response:
    if request.url.host == "other.test":
        return httpx.Response(200, text="<html><body>off-site</body></html>", headers={"content-type": "text/html"})
    if request.url.host != "www.acme-widgets.test":
        return httpx.Response(404)
    path = request.url.path
    if path == "/redirect":
        return httpx.Response(302, headers={"location": "/about"})
    if path == "/leave":
        return httpx.Response(302, headers={"location": "https://other.test/landing"})
    if path == "/image.png":
        return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})
    if path == "/boom":
        raise httpx.ConnectError("boom")
    if path == "/big":
        return httpx.Response(200, content=b"<html>" + b"x" * 3_100_000, headers={"content-type": "text/html"})
    if path in PAGES:
        ctype = "text/plain" if path == "/robots.txt" else ("application/xml" if path.endswith(".xml") else "text/html")
        return httpx.Response(200, text=PAGES[path], headers={"content-type": ctype})
    return httpx.Response(404, text="nope", headers={"content-type": "text/html"})


@pytest.fixture
def http_client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(site_handler))


# --------------------------------------------------------------------------- fake Anthropic client


def tool_use(name: str, payload: dict, block_id: str = "tu_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, id=block_id, input=payload)


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def search_result_block(urls: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        type="web_search_tool_result", tool_use_id="srvtoolu_1",
        content=[SimpleNamespace(type="web_search_result", url=u, title=f"Title of {u}", page_age="2025-01-01") for u in urls],
    )


def response(*blocks: Any, stop_reason: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


class FakeMessages:
    def __init__(self, handler: Callable[[dict], Any]) -> None:
        self.handler = handler
        self.calls: list[dict] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.handler(kwargs)


class FakeClient:
    def __init__(self, handler: Callable[[dict], Any]) -> None:
        self.messages = FakeMessages(handler)


def scripted(responses: list[Any]) -> FakeClient:
    """Return the responses in order; raise if called more times than scripted."""
    queue = list(responses)

    def handler(_: dict) -> Any:
        if not queue:
            raise AssertionError("fake client called more times than scripted")
        return queue.pop(0)

    return FakeClient(handler)


# --------------------------------------------------------------------------- valid payload builders


def claim(statement: str, cls: str = "company_claim", ids: list[str] | None = None) -> dict:
    return {"statement": statement, "classification": cls, "evidence_ids": ids if ids is not None else ["E001"]}


def attr(value: str | None, cls: str = "company_claim", ids: list[str] | None = None) -> dict:
    if value is None:
        return {"value": None, "classification": "unknown", "evidence_ids": []}
    return {"value": value, "classification": cls, "evidence_ids": ids if ids is not None else ["E001"]}


def identity_payload() -> dict:
    fields = [
        "company_name", "legal_name", "parent_company", "subsidiaries", "brands", "headquarters",
        "countries_of_operation", "founding_year", "founders", "leadership", "ownership_structure",
        "public_private_status", "stock_ticker", "primary_industry", "adjacent_industries",
    ]
    d = {f: attr(None) for f in fields}
    d["company_name"] = attr("Acme Widgets")
    d["headquarters"] = attr("Austin, Texas")
    d["founding_year"] = attr("2015")
    d["identity_uncertainties"] = ["Legal entity name not confirmed in a registry."]
    return d


def signals_payload() -> dict:
    lists = [
        "customer_segments", "target_industries", "use_cases", "value_propositions", "pricing_model",
        "sales_and_distribution", "partnerships_and_integrations", "technology", "ip_regulatory_certifications",
        "geography", "named_customers", "positioning_and_language", "strategic_priorities", "careers_signals",
    ]
    d: dict = {k: [claim(f"{k} signal")] for k in lists}
    d["offerings"] = [{
        "name": "Monitor", "target_customer": "Factory operators", "problem_solved": "Unplanned downtime",
        "key_capabilities": "Real-time telemetry, API", "business_benefit": "Less downtime",
        "monetization": "Subscription from $49/month", "classification": "company_claim", "evidence_ids": ["E001"],
    }]
    return d


def raw_findings_payload(url: str = "https://news.test/acme-raises") -> dict:
    return {
        "findings": [
            {"topic": "funding", "statement": "Acme raised a Series A.", "classification": "third_party_claim",
             "sources": [{"url": url, "title": "Acme raises", "publisher": "News Test",
                          "excerpt": "Acme raised $5M.", "published": "2024-03-01"}]},
            {"topic": "financials", "statement": "Revenue is not disclosed.", "classification": "unknown", "sources": []},
            {"topic": "news", "statement": "Fabricated claim.", "classification": "verified_fact",
             "sources": [{"url": "https://made-up.test/nothing", "title": "x", "publisher": "x", "excerpt": "x"}]},
        ],
        "not_found": ["earnings reports"],
    }


def analysis_payload(third_party_id: str = "E001") -> dict:
    c = lambda s: claim(s)  # noqa: E731
    inf = lambda s: claim(s, "analytical_inference", []) # noqa: E731
    return {
        "business_model": {
            "customer_type": c("B2B"), "ideal_customer_profile": c("Mid-size manufacturers"),
            "buyer_user_decision_maker": inf("Plant manager buys, technicians use"),
            "revenue_model": [c("Subscription")], "go_to_market": [c("Direct sales plus self-serve")],
        },
        "pains": [{"kind": "operational", "description": "Unplanned downtime", "consequence_if_unsolved": "Lost output",
                   "evidence_ids": ["E001"]}],
        "market": {
            "primary_market": inf("Industrial IoT monitoring"), "adjacent_markets": [inf("Predictive maintenance")],
            "maturity": inf("Growth stage"), "structural_trends": [inf("Sensor cost decline")],
            "technological_shifts": [inf("Edge ML")], "regulatory_influences": [], "customer_behavior_changes": [],
            "barriers_to_entry": [inf("Integration effort")], "switching_costs": inf("Moderate"),
            "commoditization_risk": inf("Medium"), "consolidation_dynamics": inf("Active"),
            "sizing": [],
        },
        "competitors": [{"name": "Rival Co", "category": "direct", "offering": "Monitoring", "target_segment": "Factories",
                         "business_model": "Subscription", "key_strength": "Scale", "key_difference": "Hardware bundle",
                         "classification": "analytical_inference", "evidence_ids": []}],
        "differentiation": [{"dimension": "technology", "claimed": "Real-time", "observable": "API documented",
                             "reproducibility": "moderate", "evidence_ids": ["E001"]}],
        "technology": [c("Public API")], "commercial_signals": [claim("Series A raised", "third_party_claim", [third_party_id])],
        "financials": [claim("No revenue disclosed", "unknown", [])], "organization": [inf("Hiring sales and ML")],
        "swot": {"strengths": [c("Clear pricing")], "weaknesses": [inf("Small team")],
                 "opportunities": [inf("Adjacent predictive maintenance")], "threats": [inf("Incumbent bundling")]},
        "strategic": [{"question": q, "answer": f"Answer to: {q}", "evidence_ids": []} for q in STRATEGIC_QUESTIONS],
        "maturity": [{"dimension": d, "evidence": "Some evidence", "evidence_ids": ["E001"]} for d in MATURITY_DIMENSIONS],
        "red_flags": [], "opportunities": [{"kind": "partnership", "description": "OEM channel", "rationale": "Hardware gap",
                                            "evidence_ids": []}],
        "analyst_observations": [inf("Sales hiring suggests an enterprise push")],
        "open_questions": ["Who are the largest customers?"],
    }


def narrative_payload() -> dict:
    keys = [
        "what_the_company_does", "problems_it_solves", "products_and_services", "customer_segments_and_use_cases",
        "business_model_and_monetization", "go_to_market", "technology_and_ip", "market_landscape",
        "competitive_landscape", "differentiation_and_defensibility", "customers_partnerships_ecosystem",
        "financial_and_funding", "growth_and_traction", "risks_and_red_flags", "strategic_opportunities",
        "analyst_observations",
    ]
    d = {k: [f"Paragraph about {k} [E001]."] for k in keys}
    d["executive_summary"] = [f"Summary paragraph {i} [E001]." for i in range(5)]
    return d


def third_party_id(kwargs: dict) -> str:
    """Find a third-party evidence id in the ledger text of an analyze/narrate prompt (E001 if none)."""
    for line in kwargs["messages"][0]["content"].splitlines():
        if "(third_party)" in line:
            return line.split("]")[0].strip("[")
    return "E001"


def stage_router(overrides: dict[str, Callable[[dict], Any]] | None = None) -> FakeClient:
    """A fake client that answers each stage's tool by name with a valid payload."""
    overrides = overrides or {}

    def handler(kwargs: dict) -> Any:
        names = [t["name"] for t in kwargs.get("tools", [])]
        for n in names:
            if n in overrides:
                return overrides[n](kwargs)
        if "submit_identity" in names:
            return response(tool_use("submit_identity", identity_payload()))
        if "submit_site_signals" in names:
            return response(tool_use("submit_site_signals", signals_payload()))
        if "submit_findings" in names:
            return response(search_result_block(["https://news.test/acme-raises"]),
                            tool_use("submit_findings", raw_findings_payload()))
        if "submit_analysis" in names:
            return response(tool_use("submit_analysis", analysis_payload(third_party_id(kwargs))))
        if "submit_narrative" in names:
            return response(tool_use("submit_narrative", narrative_payload()))
        raise AssertionError(f"unexpected tools {names}")

    return FakeClient(handler)


def dumps(o: Any) -> str:
    return json.dumps(o)
