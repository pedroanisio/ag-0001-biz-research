"""Render the Company Intelligence Report (Markdown) from validated stage outputs.

The renderer adds no facts. It lays out narrative prose, structured tables, and the sources
list, and marks every claim with its classification. Every heading, label and fixed sentence
comes from :mod:`bi_agent.i18n`, so the report reads in the run's language.
"""

from __future__ import annotations

import re

from .i18n import Translator
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

_ID_RE = re.compile(r"E\d{3,}")


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _cites(ids: list[str]) -> str:
    return " ".join(f"[{i}]" for i in ids)


def _paras(paragraphs: list[str]) -> str:
    return "\n\n".join(p.strip() for p in paragraphs)


def _cited_ids(*texts: str) -> set[str]:
    found: set[str] = set()
    for t in texts:
        found.update(_ID_RE.findall(t))
    return found


def _head(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |\n|" + "---|" * len(cells)


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
    lang: str = "en",
) -> str:
    t = Translator(lang)
    n = narrative
    a = analysis

    def label(cls: object) -> str:
        return t(getattr(cls, "value", cls))

    def claim_line(c: Claim) -> str:
        return f"- {c.statement} *({label(c.classification)})* {_cites(c.evidence_ids)}".rstrip()

    def claims(items: list[Claim], empty: str = "nothing") -> str:
        return "\n".join(claim_line(c) for c in items) if items else f"_{t(empty)}_"

    def attr(x: Attr) -> str:
        if x.value is None:
            return t("unknown")
        return f"{_cell(x.value)} *({label(x.classification)})* {_cites(x.evidence_ids)}".rstrip()

    def section(num: int, paras: list[str] | None = None) -> None:
        w(f"## {num}. {t(f's{num}')}\n")
        if paras is not None:
            w(_paras(paras) + "\n")

    def sub(key: str, content: str) -> None:
        w(f"**{t(key)}**\n\n{content}\n")

    name = identity.company_name.value or meta.get("start", meta.get("url", t("the_company")))
    out: list[str] = []
    w = out.append

    w(f"# {t('title', name=name)}\n")
    w(f"{t('subject_url')}: {meta.get('url')}  \n{t('generated')}: {access_date}  \n"
      f"{t('evidence_items', n=len(ledger), pages=meta.get('pages', 0))}\n")
    labels = [f"*{t(k)}*" for k in ("verified_fact", "company_claim", "third_party_claim", "analytical_inference", "unknown")]
    w(t("key", labels=", ".join(labels[:-1]) + f" {t('or')} " + labels[-1]) + "\n")

    section(1, n.executive_summary)

    section(2)
    w(_head(t("field"), t("value")))
    rows = [
        ("company", attr(identity.company_name)), ("legal_entity", attr(identity.legal_name)),
        ("website", meta.get("start", meta.get("url", ""))), ("headquarters", attr(identity.headquarters)),
        ("founded", attr(identity.founding_year)), ("founders", attr(identity.founders)),
        ("ownership", attr(identity.ownership_structure)),
        ("public_private", attr(identity.public_private_status)), ("ticker", attr(identity.stock_ticker)),
        ("parent", attr(identity.parent_company)), ("subsidiaries", attr(identity.subsidiaries)),
        ("brands", attr(identity.brands)), ("leadership", attr(identity.leadership)),
        ("industry", attr(identity.primary_industry)), ("adjacent_industries", attr(identity.adjacent_industries)),
        ("core_market", _cell(a.market.primary_market.statement) + f" {_cites(a.market.primary_market.evidence_ids)}"),
        ("business_model", "; ".join(_cell(c.statement) for c in a.business_model.revenue_model)),
        ("customer_type", _cell(a.business_model.customer_type.statement)),
        ("geo_presence", attr(identity.countries_of_operation)),
    ]
    for k, v in rows:
        w(f"| {t(k)} | {v} |")
    if identity.identity_uncertainties:
        w(f"\n**{t('identity_uncertainties')}**\n")
        for u in identity.identity_uncertainties:
            w(f"- {u}")
    w("")

    section(3, n.what_the_company_does)

    section(4, n.problems_it_solves)
    w(_head(t("pain_type"), t("problem"), t("consequence"), t("evidence")))
    for p in a.pains:
        w(f"| {t('pain.' + p.kind.value)} | {_cell(p.description)} | {_cell(p.consequence_if_unsolved)} | {_cites(p.evidence_ids)} |")
    w("")

    section(5, n.products_and_services)
    w(_head(t("offering"), t("type"), t("target_customer"), t("problem_solved"), t("capabilities"), t("benefit"),
            t("monetization"), t("basis")))
    for o in signals.offerings:
        w(f"| {_cell(o.name)} | {t('kind.' + o.kind.value)} | {_cell(o.target_customer)} | {_cell(o.problem_solved)} | {_cell(o.key_capabilities)} | "
          f"{_cell(o.business_benefit)} | {_cell(o.monetization)} | {label(o.classification)} {_cites(o.evidence_ids)} |")
    if not signals.offerings:
        w(f"| — | — | — | — | — | — | — | {t('no_offerings')} |")
    w("")

    section(6, n.customer_segments_and_use_cases)
    sub("segments", claims(signals.customer_segments))
    sub("target_industries", claims(signals.target_industries))
    sub("use_cases", claims(signals.use_cases))

    section(7, n.business_model_and_monetization)
    bm = a.business_model
    sub("customer_type", claim_line(bm.customer_type))
    sub("icp", claim_line(bm.ideal_customer_profile))
    sub("buyer_user", claim_line(bm.buyer_user_decision_maker))
    sub("revenue_model", claims(bm.revenue_model))
    sub("pricing_signals", claims(signals.pricing_model, "no_pricing"))

    section(8, n.go_to_market)
    w(claims(bm.go_to_market) + "\n")
    sub("sales_signals", claims(signals.sales_and_distribution))

    section(9, n.technology_and_ip)
    sub("tech_signals", claims(a.technology + signals.technology))
    sub("ip_claims", claims(signals.ip_regulatory_certifications))

    section(10, n.market_landscape)
    m = a.market
    sub("primary_market", claim_line(m.primary_market))
    sub("adjacent_markets", claims(m.adjacent_markets))
    sub("market_maturity", claim_line(m.maturity))
    sub("trends", claims(m.structural_trends))
    sub("tech_shifts", claims(m.technological_shifts))
    sub("regulatory", claims(m.regulatory_influences))
    sub("behaviour", claims(m.customer_behavior_changes))
    sub("barriers", claims(m.barriers_to_entry))
    sub("switching", claim_line(m.switching_costs))
    sub("commoditization", claim_line(m.commoditization_risk))
    sub("consolidation", claim_line(m.consolidation_dynamics))
    w(f"**{t('sizing')}**\n")
    if m.sizing:
        w(_head(t("metric"), t("value"), t("year"), t("methodology"), t("limitations"), t("source")))
        for s in m.sizing:
            w(f"| {s.metric} | {_cell(s.value)} | {s.year} | {_cell(s.methodology)} | {_cell(s.limitations)} | {_cites(s.evidence_ids)} |")
    else:
        w(f"_{t('no_sizing')}_")
    w("")

    section(11, n.competitive_landscape)
    w(_head(t("company"), t("category"), t("offering"), t("target_segment"), t("business_model"),
            t("strength"), t("difference"), t("basis")))
    order = {c: i for i, c in enumerate(CompetitorCategory)}
    for c in sorted(a.competitors, key=lambda x: order[x.category]):
        w(f"| {_cell(c.name)} | {t('cat.' + c.category.value)} | {_cell(c.offering)} | {_cell(c.target_segment)} | "
          f"{_cell(c.business_model)} | {_cell(c.key_strength)} | {_cell(c.key_difference)} | "
          f"{label(c.classification)} {_cites(c.evidence_ids)} |")
    w("")

    section(12, n.differentiation_and_defensibility)
    w(_head(t("dimension"), t("claimed_diff"), t("observable_diff"), t("reproducibility"), t("evidence")))
    for d in a.differentiation:
        w(f"| {_cell(d.dimension)} | {_cell(d.claimed)} | {_cell(d.observable)} | {t('repro.' + d.reproducibility.value)} | {_cites(d.evidence_ids)} |")
    w("")

    section(13, n.customers_partnerships_ecosystem)
    sub("named_customers", claims(signals.named_customers, "no_customers"))
    sub("partnerships", claims(signals.partnerships_and_integrations, "no_partnerships"))
    sub("geography", claims(signals.geography))

    section(14, n.financial_and_funding)
    w(claims(a.financials, "no_financials") + "\n")

    section(15, n.growth_and_traction)
    sub("commercial_signals", claims(a.commercial_signals))
    sub("organization", claims(a.organization))
    sub("hiring", claims(signals.careers_signals, "no_careers"))

    section(16)
    for key, items in (("strengths", a.swot.strengths), ("weaknesses", a.swot.weaknesses),
                       ("opportunities", a.swot.opportunities), ("threats", a.swot.threats)):
        w(f"### {t(key)}\n\n" + claims(items) + "\n")

    w(f"### {t('strategic_analysis')}\n")
    w(f"_{t('strategic_note')}_\n")
    for s in a.strategic:
        w(f"**{t.question(s.question)}**\n\n{s.answer} *({t('analytical_inference')})* {_cites(s.evidence_ids)}\n"
          .replace(" \n", "\n"))

    w(f"### {t('business_maturity')}\n")
    w(_head(t("dimension"), t("evidence")))
    for r in a.maturity:
        w(f"| {t.dimension(r.dimension)} | {_cell(r.evidence)} {_cites(r.evidence_ids)} |")
    w("")

    section(17, n.risks_and_red_flags)
    w(claims(a.red_flags, "no_red_flags") + "\n")

    section(18, n.strategic_opportunities)
    w(_head(t("type"), t("opportunity"), t("why_exists"), t("evidence")))
    for o in a.opportunities:
        w(f"| {_cell(o.kind)} | {_cell(o.description)} | {_cell(o.rationale)} | {_cites(o.evidence_ids)} |")
    w("")

    section(19, n.analyst_observations)
    w(claims(a.analyst_observations) + "\n")

    section(20)
    questions = list(a.open_questions) + [t("research_gap", gap=x) for x in findings.not_found]
    w("\n".join(f"- {q}" for q in questions) if questions else f"_{t('none')}_")
    w("")

    section(21)
    body = "\n".join(out)
    used = _cited_ids(body)
    w(_head(t("id"), t("title_col"), t("publisher"), t("url"), t("published"), t("type"), t("accessed")))
    for e in ledger:
        if e.id in used:
            accessed = (e.retrieved_at or "")[:10] or access_date  # when the pipeline fetched it
            w(f"| {e.id} | {_cell(e.title)} | {_cell(e.publisher)} | {e.url} | {e.published or t('n_a')} | "
              f"{t(e.source_type.value)} | {accessed} |")
    if findings.rejected:
        w(f"\n{t('discarded')}\n")
        for r in findings.rejected:
            w(f"- {r}")
    w("")
    return "\n".join(out)
