"""Render the Company Intelligence Report (Markdown) from validated stage outputs.

The renderer adds no facts. It lays out narrative prose, structured tables, and the sources
list, and marks every claim with its classification.
"""

from __future__ import annotations

import re

from .models import (
    Analysis,
    Attr,
    Claim,
    CompetitorCategory,
    EvidenceLedger,
    ExternalFindings,
    Identity,
    Narrative,
    SiteSignals,
)

LABEL = {
    "verified_fact": "Verified fact",
    "company_claim": "Company claim",
    "third_party_claim": "Third-party claim",
    "analytical_inference": "Analytical inference",
    "unknown": "Unknown",
}
_ID_RE = re.compile(r"E\d{3,}")


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _cites(ids: list[str]) -> str:
    return " ".join(f"[{i}]" for i in ids)


def _claim_line(c: Claim) -> str:
    return f"- {c.statement} *({LABEL[c.classification.value]})* {_cites(c.evidence_ids)}".rstrip()


def _claims(items: list[Claim], empty: str = "Nothing established from the evidence gathered.") -> str:
    return "\n".join(_claim_line(c) for c in items) if items else f"_{empty}_"


def _attr(a: Attr) -> str:
    if a.value is None:
        return "Unknown"
    return f"{_cell(a.value)} *({LABEL[a.classification.value]})* {_cites(a.evidence_ids)}".rstrip()


def _paras(paragraphs: list[str]) -> str:
    return "\n\n".join(p.strip() for p in paragraphs)


def _cited_ids(*texts: str) -> set[str]:
    found: set[str] = set()
    for t in texts:
        found.update(_ID_RE.findall(t))
    return found


def render_report(
    *,
    meta: dict,
    identity: Identity,
    signals: SiteSignals,
    findings: ExternalFindings,
    analysis: Analysis,
    narrative: Narrative,
    ledger: EvidenceLedger,
    access_date: str,
) -> str:
    n = narrative
    a = analysis
    name = identity.company_name.value or meta.get("start", meta.get("url", "the company"))
    out: list[str] = []
    w = out.append

    w(f"# Company Intelligence Report: {name}\n")
    w(f"Subject URL: {meta.get('url')}  \nReport generated: {access_date}  \n"
      f"Evidence items: {len(ledger)} (website pages crawled: {meta.get('pages', 0)})\n")
    w("Classification key: every claim carries one of *Verified fact*, *Company claim*, "
      "*Third-party claim*, *Analytical inference*, or *Unknown*. Bracketed ids such as [E012] "
      "resolve to the Sources section.\n")

    w("## 1. Executive Summary\n")
    w(_paras(n.executive_summary) + "\n")

    w("## 2. Company Snapshot\n")
    w("| Field | Value |\n|---|---|")
    rows = [
        ("Company", _attr(identity.company_name)), ("Legal entity", _attr(identity.legal_name)),
        ("Website", meta.get("start", meta.get("url", ""))), ("Headquarters", _attr(identity.headquarters)),
        ("Founded", _attr(identity.founding_year)), ("Founders", _attr(identity.founders)),
        ("Ownership", _attr(identity.ownership_structure)),
        ("Public / private", _attr(identity.public_private_status)), ("Ticker", _attr(identity.stock_ticker)),
        ("Parent company", _attr(identity.parent_company)), ("Subsidiaries", _attr(identity.subsidiaries)),
        ("Brands", _attr(identity.brands)), ("Leadership", _attr(identity.leadership)),
        ("Industry", _attr(identity.primary_industry)), ("Adjacent industries", _attr(identity.adjacent_industries)),
        ("Core market", _cell(a.market.primary_market.statement) + f" {_cites(a.market.primary_market.evidence_ids)}"),
        ("Business model", "; ".join(_cell(c.statement) for c in a.business_model.revenue_model)),
        ("Customer type", _cell(a.business_model.customer_type.statement)),
        ("Geographic presence", _attr(identity.countries_of_operation)),
    ]
    for k, v in rows:
        w(f"| {k} | {v} |")
    if identity.identity_uncertainties:
        w("\n**Identity uncertainties**\n")
        for u in identity.identity_uncertainties:
            w(f"- {u}")
    w("")

    w("## 3. What the Company Does\n")
    w(_paras(n.what_the_company_does) + "\n")

    w("## 4. Problems It Solves\n")
    w(_paras(n.problems_it_solves) + "\n")
    w("| Pain type | Problem | Consequence if unsolved | Evidence |\n|---|---|---|---|")
    for p in a.pains:
        w(f"| {p.kind.value} | {_cell(p.description)} | {_cell(p.consequence_if_unsolved)} | {_cites(p.evidence_ids)} |")
    w("")

    w("## 5. Products and Services\n")
    w(_paras(n.products_and_services) + "\n")
    w("| Offering | Target Customer | Problem Solved | Key Capabilities | Business Benefit | Monetization | Basis |\n"
      "|---|---|---|---|---|---|---|")
    for o in signals.offerings:
        w(f"| {_cell(o.name)} | {_cell(o.target_customer)} | {_cell(o.problem_solved)} | {_cell(o.key_capabilities)} | "
          f"{_cell(o.business_benefit)} | {_cell(o.monetization)} | {LABEL[o.classification.value]} {_cites(o.evidence_ids)} |")
    if not signals.offerings:
        w("| — | — | — | — | — | — | No offerings could be extracted from the website |")
    w("")

    w("## 6. Customer Segments and Use Cases\n")
    w(_paras(n.customer_segments_and_use_cases) + "\n")
    w("**Segments**\n\n" + _claims(signals.customer_segments) + "\n")
    w("**Target industries**\n\n" + _claims(signals.target_industries) + "\n")
    w("**Use cases**\n\n" + _claims(signals.use_cases) + "\n")

    w("## 7. Business Model and Monetization\n")
    w(_paras(n.business_model_and_monetization) + "\n")
    bm = a.business_model
    w("**Customer type**\n\n" + _claim_line(bm.customer_type) + "\n")
    w("**Ideal customer profile**\n\n" + _claim_line(bm.ideal_customer_profile) + "\n")
    w("**Buyer, user and economic decision maker**\n\n" + _claim_line(bm.buyer_user_decision_maker) + "\n")
    w("**Revenue model**\n\n" + _claims(bm.revenue_model) + "\n")
    w("**Pricing signals from the website**\n\n" + _claims(signals.pricing_model, "No pricing information published.") + "\n")

    w("## 8. Go-to-Market Strategy\n")
    w(_paras(n.go_to_market) + "\n")
    w(_claims(bm.go_to_market) + "\n")
    w("**Sales and distribution signals from the website**\n\n" + _claims(signals.sales_and_distribution) + "\n")

    w("## 9. Technology and Intellectual Property\n")
    w(_paras(n.technology_and_ip) + "\n")
    w("**Observed technology signals**\n\n" + _claims(a.technology + signals.technology) + "\n")
    w("**IP, regulatory and certification claims**\n\n" + _claims(signals.ip_regulatory_certifications) + "\n")

    w("## 10. Market Landscape\n")
    w(_paras(n.market_landscape) + "\n")
    m = a.market
    w("**Primary market**\n\n" + _claim_line(m.primary_market) + "\n")
    w("**Adjacent markets**\n\n" + _claims(m.adjacent_markets) + "\n")
    w("**Market maturity**\n\n" + _claim_line(m.maturity) + "\n")
    w("**Structural trends**\n\n" + _claims(m.structural_trends) + "\n")
    w("**Technological shifts**\n\n" + _claims(m.technological_shifts) + "\n")
    w("**Regulatory influences**\n\n" + _claims(m.regulatory_influences) + "\n")
    w("**Customer behaviour changes**\n\n" + _claims(m.customer_behavior_changes) + "\n")
    w("**Barriers to entry**\n\n" + _claims(m.barriers_to_entry) + "\n")
    w("**Switching costs**\n\n" + _claim_line(m.switching_costs) + "\n")
    w("**Commoditization risk**\n\n" + _claim_line(m.commoditization_risk) + "\n")
    w("**Consolidation dynamics**\n\n" + _claim_line(m.consolidation_dynamics) + "\n")
    w("**Market sizing**\n")
    if m.sizing:
        w("| Metric | Value | Year | Methodology | Limitations | Source |\n|---|---|---|---|---|---|")
        for s in m.sizing:
            w(f"| {s.metric} | {_cell(s.value)} | {s.year} | {_cell(s.methodology)} | {_cell(s.limitations)} | {_cites(s.evidence_ids)} |")
    else:
        w("_No credible sourced market-size figure was found; none is estimated here._")
    w("")

    w("## 11. Competitive Landscape\n")
    w(_paras(n.competitive_landscape) + "\n")
    w("| Company | Category | Offering | Target Segment | Business Model | Key Strength | Key Difference | Basis |\n"
      "|---|---|---|---|---|---|---|---|")
    order = {c: i for i, c in enumerate(CompetitorCategory)}
    for c in sorted(a.competitors, key=lambda x: order[x.category]):
        w(f"| {_cell(c.name)} | {c.category.value.replace('_', ' ')} | {_cell(c.offering)} | {_cell(c.target_segment)} | "
          f"{_cell(c.business_model)} | {_cell(c.key_strength)} | {_cell(c.key_difference)} | "
          f"{LABEL[c.classification.value]} {_cites(c.evidence_ids)} |")
    w("")

    w("## 12. Differentiation and Defensibility\n")
    w(_paras(n.differentiation_and_defensibility) + "\n")
    w("| Dimension | Claimed differentiation | Observable differentiation | Reproducibility | Evidence |\n|---|---|---|---|---|")
    for d in a.differentiation:
        w(f"| {_cell(d.dimension)} | {_cell(d.claimed)} | {_cell(d.observable)} | {d.reproducibility.value} | {_cites(d.evidence_ids)} |")
    w("")

    w("## 13. Customers, Partnerships and Ecosystem\n")
    w(_paras(n.customers_partnerships_ecosystem) + "\n")
    w("**Named customers and case studies**\n\n" + _claims(signals.named_customers, "No named customers found.") + "\n")
    w("**Partnerships and integrations**\n\n" + _claims(signals.partnerships_and_integrations, "No partnerships or integrations found.") + "\n")
    w("**Geography**\n\n" + _claims(signals.geography) + "\n")

    w("## 14. Financial and Funding Information\n")
    w(_paras(n.financial_and_funding) + "\n")
    w(_claims(a.financials, "No financial or funding information is publicly available; nothing is estimated here.") + "\n")

    w("## 15. Growth and Traction Signals\n")
    w(_paras(n.growth_and_traction) + "\n")
    w("**Commercial signals**\n\n" + _claims(a.commercial_signals) + "\n")
    w("**Organization and talent**\n\n" + _claims(a.organization) + "\n")
    w("**Hiring signals from the careers pages**\n\n" + _claims(signals.careers_signals, "No careers page content found.") + "\n")

    w("## 16. SWOT\n")
    for title, items in (("Strengths", a.swot.strengths), ("Weaknesses", a.swot.weaknesses),
                         ("Opportunities", a.swot.opportunities), ("Threats", a.swot.threats)):
        w(f"### {title}\n\n" + _claims(items) + "\n")

    w("### Strategic analysis\n")
    for s in a.strategic:
        w(f"**{s.question}**\n\n{s.answer} {_cites(s.evidence_ids)}\n".replace(" \n", "\n"))

    w("### Business maturity\n")
    w("| Dimension | Evidence |\n|---|---|")
    for r in a.maturity:
        w(f"| {r.dimension} | {_cell(r.evidence)} {_cites(r.evidence_ids)} |")
    w("")

    w("## 17. Risks and Red Flags\n")
    w(_paras(n.risks_and_red_flags) + "\n")
    w(_claims(a.red_flags, "No red flags were identified in the evidence gathered; absence of evidence is not evidence of absence.") + "\n")

    w("## 18. Strategic Opportunities\n")
    w(_paras(n.strategic_opportunities) + "\n")
    w("| Type | Opportunity | Why it exists | Evidence |\n|---|---|---|---|")
    for o in a.opportunities:
        w(f"| {_cell(o.kind)} | {_cell(o.description)} | {_cell(o.rationale)} | {_cites(o.evidence_ids)} |")
    w("")

    w("## 19. Analyst Observations\n")
    w(_paras(n.analyst_observations) + "\n")
    w(_claims(a.analyst_observations) + "\n")

    w("## 20. Open Questions\n")
    questions = list(a.open_questions) + [f"Research gap: {x}" for x in findings.not_found]
    w("\n".join(f"- {q}" for q in questions) if questions else "_None._")
    w("")

    w("## 21. Sources\n")
    body = "\n".join(out)
    used = _cited_ids(body)
    w("| Id | Title | Publisher | URL | Published | Type | Accessed |\n|---|---|---|---|---|---|---|")
    for e in ledger:
        if e.id in used:
            w(f"| {e.id} | {_cell(e.title)} | {_cell(e.publisher)} | {e.url} | {e.published or 'n/a'} | "
              f"{e.source_type.value.replace('_', ' ')} | {access_date} |")
    if findings.rejected:
        w("\n**Discarded during verification** (sources the research model cited that were never returned by search):\n")
        for r in findings.rejected:
            w(f"- {r}")
    w("")
    return "\n".join(out)
