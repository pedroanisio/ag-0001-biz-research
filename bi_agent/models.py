"""Typed contracts for every stage boundary.

Every model rejects unknown fields (``extra="forbid"``). Cross-model rules that a JSON
schema cannot express (evidence ids must resolve, classifications must match the
kind of evidence cited) live in :func:`check_refs` and :func:`check_classifications`.
"""

from __future__ import annotations

import json
import hashlib
import re
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Literal

from .urls import normalize_url, hostname, registrable_domain

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EVIDENCE_ID = re.compile(r"^E\d{3,}$")
CITATION = re.compile(r"\[(E\d{3,})\]")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Classification(str, Enum):
    VERIFIED_FACT = "verified_fact"
    COMPANY_CLAIM = "company_claim"
    THIRD_PARTY_CLAIM = "third_party_claim"
    ANALYTICAL_INFERENCE = "analytical_inference"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    FIRST_PARTY = "first_party"
    THIRD_PARTY = "third_party"


class SourceKind(str, Enum):
    """What a source is, in the order of preference the research brief sets (tier 1 is best)."""

    GOVERNMENT_REGULATORY = "government_regulatory"
    COMPANY_FILING = "company_filing"
    OFFICIAL_COMPANY = "official_company"
    INVESTOR_DISCLOSURE = "investor_disclosure"
    PARTNER_CUSTOMER = "partner_customer"
    INDUSTRY_PUBLICATION = "industry_publication"
    NEWS = "news"
    DATABASE_AGGREGATOR = "database_aggregator"
    FORUM_SOCIAL = "forum_social"

    @property
    def tier(self) -> int:
        return list(SourceKind).index(self) + 1


# A record from a registry, regulator or statutory filing establishes a fact on its own.
PRIMARY_RECORD_KINDS = {SourceKind.GOVERNMENT_REGULATORY, SourceKind.COMPANY_FILING}
# Sources that cannot count as independent confirmation: the company speaking for itself, and
# forums/social media, which the brief allows only as supporting evidence.
NOT_INDEPENDENT_KINDS = {SourceKind.OFFICIAL_COMPANY, SourceKind.INVESTOR_DISCLOSURE, SourceKind.FORUM_SOCIAL}


# --------------------------------------------------------------------------- evidence


class Evidence(Strict):
    id: str
    source_type: SourceType
    url: str
    title: str = Field(max_length=300)
    publisher: str = Field(max_length=200)
    excerpt: str = Field(max_length=2000)
    published: str | None = None
    retrieved_at: str
    source_kind: SourceKind | None = None
    retrieval_method: Literal["crawl", "web_search", "web_fetch"] | None = None  # crawl, web_search, web_fetch; None means legacy/unproven
    requested_url: str | None = None
    final_url: str | None = None
    aliases: list[str] = Field(default_factory=list)  # only observed redirects
    content: str | None = None  # source text, never a generated summary
    content_limitation: str | None = None
    publisher_group: str | None = None  # configured/observed ownership, never model-supplied
    original_url: str | None = None


    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not EVIDENCE_ID.match(v):
            raise ValueError(f"evidence id {v!r} must match E### pattern")
        return v

    @field_validator("url")
    @classmethod
    def _url_format(cls, v: str) -> str:
        return normalize_url(v)



class EvidenceLedger:
    """Append-only registry of evidence items, keyed by id and de-duplicated by URL."""

    def __init__(self, items: list[Evidence] | None = None) -> None:
        self._items: dict[str, Evidence] = {}
        self._by_url: dict[str, str] = {}
        for item in items or []:
            self._items[item.id] = item
            for url in [item.url, *item.aliases]:
                self._by_url[normalize_url(url)] = item.id

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Evidence]:
        return iter(self._items.values())

    def has(self, evidence_id: str) -> bool:
        return evidence_id in self._items

    def get(self, evidence_id: str) -> Evidence:
        return self._items[evidence_id]

    def id_for_url(self, url: str) -> str | None:
        return self._by_url.get(normalize_url(url))

    def add(
        self,
        *,
        source_type: SourceType,
        url: str,
        title: str,
        publisher: str,
        excerpt: str,
        retrieved_at: str,
        published: str | None = None,
        source_kind: SourceKind | None = None,
        **retrieval: Any,
    ) -> Evidence:
        existing = self.id_for_url(url)
        if existing is not None:
            item = self._items[existing]
            # A document fetch can strengthen a discovery record without changing its id.
            if retrieval.get("content") and (not item.content or len(retrieval["content"]) > len(item.content)):
                for key, value in retrieval.items():
                    setattr(item, key, value)
                item.retrieved_at = retrieved_at
                item.source_kind = source_kind
            for alias in item.aliases:
                self._by_url[normalize_url(alias)] = item.id
            return item
        new_id = f"E{max((int(x[1:]) for x in self._items), default=0) + 1:03d}"
        item = Evidence(
            id=new_id,
            source_type=source_type,
            url=url,
            title=title[:300],
            publisher=publisher[:200],
            excerpt=excerpt[:2000],
            published=published,
            retrieved_at=retrieved_at,
            source_kind=source_kind, **retrieval,
        )
        self._items[new_id] = item
        for alias in [url, *item.aliases]:
            self._by_url[normalize_url(alias)] = new_id
        return item

    def index_text(self) -> str:
        """Compact id → source listing for prompts."""
        return "\n".join(
            f"[{e.id}] ({e.source_type.value}{', ' + e.source_kind.value if e.source_kind else ''}) "
            f"{e.title} — {e.publisher} — {e.url} — SOURCE TEXT: {json.dumps(e.content or 'unavailable; discovery only', ensure_ascii=False)}"
            for e in self
        )

    def save(self, path: Path) -> None:
        from .store import atomic_write, encoded
        atomic_write(path, encoded([e.model_dump(mode="json") for e in self]))

    @classmethod
    def load(cls, path: Path) -> "EvidenceLedger":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("evidence file must contain a list")
        items = [Evidence.model_validate(x) for x in raw]
        for e in items:  # re-check labels saved under older, looser rules
            if e.source_kind is not None:
                e.source_kind = plausible_source_kind(e.source_kind, e.url, e.source_type is SourceType.FIRST_PARTY, e.content, e.title)
        return cls(items)


def normalized_text(text: str) -> str:
    return " ".join(text.split()).casefold().strip(" .")


def passage_supported(statement: str, evidence: Evidence, passage: str | None = None) -> bool:
    """Conservative extractive support: no guessed entailment from a matching URL.

    Paraphrases must be regenerated as source excerpts or explicit inferences. This is a
    provenance check, not a guarantee that a publisher's assertion is true.
    """
    if not evidence.retrieval_method or not evidence.content or not normalized_text(statement):
        return False
    quote = normalized_text(passage or statement)
    return (normalized_text(statement) in quote and quote in normalized_text(evidence.content))


# Ownership grouping is configured in code, never trusted from a model-supplied publisher label.
PUBLISHER_OWNERS = {"reuters.com": "thomson-reuters", "reutersconnect.com": "thomson-reuters",
                    "wsj.com": "news-corp", "barrons.com": "news-corp", "marketwatch.com": "news-corp"}


def independence_key(e: Evidence) -> str:
    if e.publisher_group:
        return e.publisher_group
    content = (e.content or "").casefold()
    for marker, owner in (("(reuters)", "thomson-reuters"), ("source: reuters", "thomson-reuters"),
                          ("(ap)", "associated-press"), ("source: associated press", "associated-press")):
        if marker in content:
            return owner
    domain = registrable_domain(e.original_url or e.url)
    return PUBLISHER_OWNERS.get(domain, domain)


def verified_fact_supported(items: list[Evidence], statement: str = "") -> bool:
    items = [e for e in items if passage_supported(statement, e)]
    if any(e.retrieval_method != "web_search" and plausible_source_kind(e.source_kind, e.url, e.source_type is SourceType.FIRST_PARTY,
                                e.content, e.title) in PRIMARY_RECORD_KINDS for e in items if e.source_kind):
        return True
    return len(independent_groups(items, statement)) >= 2


def independent_groups(items: list[Evidence], statement: str) -> set[str]:
    items = [e for e in items if passage_supported(statement, e)]
    independent: set[str] = set()
    contents: set[str] = set()
    publishers: set[str] = set()
    for e in items:
        if e.source_type is not SourceType.THIRD_PARTY or e.source_kind in NOT_INDEPENDENT_KINDS:
            continue
        domain = independence_key(e)
        content = normalized_text(e.content or "")
        publisher = normalized_text(e.publisher)
        if content in contents or publisher in publishers:
            continue
        independent.add(domain)
        contents.add(content)
        publishers.add(publisher)
    return independent


def supported_classification(cls: str, items: list[Evidence], statement: str = "") -> str:
    """The strongest classification the cited evidence supports.

    verified_fact without enough independent confirmation becomes third_party_claim (or
    company_claim when only the company says so); company_claim needs a first-party source,
    third_party_claim a third-party one; a sourced claim with no evidence remains invalid until regenerated.
    """
    sourced = {Classification.VERIFIED_FACT.value, Classification.COMPANY_CLAIM.value,
               Classification.THIRD_PARTY_CLAIM.value}
    if cls not in sourced:
        return cls
    if not items:
        return cls  # leave unsupported facts invalid; never turn them into inferences
    kinds = {e.source_type for e in items}
    first, third = SourceType.FIRST_PARTY in kinds, SourceType.THIRD_PARTY in kinds
    if cls == Classification.VERIFIED_FACT.value and not verified_fact_supported(items, statement):
        return Classification.THIRD_PARTY_CLAIM.value if third else Classification.COMPANY_CLAIM.value
    if cls == Classification.COMPANY_CLAIM.value and not first:
        return Classification.THIRD_PARTY_CLAIM.value
    if cls == Classification.THIRD_PARTY_CLAIM.value and not third:
        return Classification.COMPANY_CLAIM.value
    return cls



# --------------------------------------------------------------------------- claims


class SupportingPassage(Strict):
    field: str = "statement"
    evidence_id: str
    passage: str = Field(min_length=1, max_length=2000)


class Claim(Strict):
    supporting_passages: list[SupportingPassage] = Field(default_factory=list)
    premises: list[SupportingPassage] = Field(default_factory=list)
    statement: str = Field(min_length=1, max_length=1200)
    classification: Classification
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("evidence_ids")
    @classmethod
    def _ids(cls, v: list[str]) -> list[str]:
        for x in v:
            if not EVIDENCE_ID.match(x):
                raise ValueError(f"evidence id {x!r} must match E### pattern")
        return v

    @model_validator(mode="after")
    def _needs_evidence(self) -> "Claim":
        needs = {
            Classification.VERIFIED_FACT,
            Classification.COMPANY_CLAIM,
            Classification.THIRD_PARTY_CLAIM,
        }
        if self.classification in needs and not self.evidence_ids:
            raise ValueError(
                f"classification {self.classification.value} requires at least one evidence id"
            )
        return self


class Attr(Strict):
    """A single identity attribute with provenance. ``value`` is None when unknown."""
    supporting_passages: list[SupportingPassage] = Field(default_factory=list)

    premises: list[SupportingPassage] = Field(default_factory=list)
    value: str | None = Field(default=None, max_length=1000)  # resolve merges website and registry detail
    classification: Classification = Classification.UNKNOWN
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def _consistent(self) -> "Attr":
        if self.value is None and self.classification != Classification.UNKNOWN:
            raise ValueError("an attribute with no value must be classified unknown")
        if self.value is not None and self.classification == Classification.UNKNOWN:
            raise ValueError("an attribute with a value cannot be classified unknown")
        if self.classification in {
            Classification.VERIFIED_FACT,
            Classification.COMPANY_CLAIM,
            Classification.THIRD_PARTY_CLAIM,
        } and not self.evidence_ids:
            raise ValueError("sourced classification requires evidence ids")
        return self


# --------------------------------------------------------------------------- stage outputs


def _require_website_subject(schema: dict) -> None:
    schema.setdefault("required", [])
    if "website_subject" not in schema["required"]:
        schema["required"].append("website_subject")


class Identity(Strict):
    # website_subject is required in the schema the model fills; the default lets older runs load.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, json_schema_extra=_require_website_subject)

    website_subject: Attr = Field(default_factory=Attr)  # the company itself, or a product/brand of it
    company_name: Attr
    legal_name: Attr
    parent_company: Attr
    subsidiaries: Attr
    brands: Attr
    headquarters: Attr
    countries_of_operation: Attr
    founding_year: Attr
    founders: Attr
    leadership: Attr
    ownership_structure: Attr
    public_private_status: Attr
    stock_ticker: Attr
    primary_industry: Attr
    adjacent_industries: Attr
    identity_uncertainties: list[str] = Field(default_factory=list, max_length=10)


class OfferingKind(str, Enum):
    CORE_PRODUCT = "core_product"
    SECONDARY_PRODUCT = "secondary_product"
    SERVICE = "service"
    PROFESSIONAL_SERVICES = "professional_services"
    SUBSCRIPTION = "subscription"
    PLATFORM = "platform"
    API = "api"
    SOFTWARE = "software"
    HARDWARE = "hardware"
    DATA_PRODUCT = "data_product"
    MARKETPLACE = "marketplace"
    LICENSING = "licensing"
    OTHER = "other"


def _require_kind(schema: dict) -> None:
    schema.setdefault("required", [])
    if "kind" not in schema["required"]:
        schema["required"].append("kind")


class Offering(Strict):
    supporting_passages: list[SupportingPassage] = Field(default_factory=list)
    premises: list[SupportingPassage] = Field(default_factory=list)
    # ``kind`` is required in the schema the model fills; the default only lets runs saved
    # before the field existed still load.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, json_schema_extra=_require_kind)

    name: str = Field(max_length=200)
    kind: OfferingKind = OfferingKind.OTHER
    target_customer: str = Field(max_length=400)
    problem_solved: str = Field(max_length=600)
    key_capabilities: str = Field(max_length=800)
    business_benefit: str = Field(max_length=600)
    monetization: str = Field(max_length=400)
    classification: Classification
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)


class SiteSignals(Strict):
    offerings: list[Offering] = Field(max_length=25)
    customer_segments: list[Claim] = Field(max_length=20)
    target_industries: list[Claim] = Field(max_length=20)
    use_cases: list[Claim] = Field(max_length=25)
    value_propositions: list[Claim] = Field(max_length=15)
    pricing_model: list[Claim] = Field(max_length=10)
    sales_and_distribution: list[Claim] = Field(max_length=10)
    partnerships_and_integrations: list[Claim] = Field(max_length=30)
    technology: list[Claim] = Field(max_length=25)
    ip_regulatory_certifications: list[Claim] = Field(max_length=20)
    geography: list[Claim] = Field(max_length=15)
    named_customers: list[Claim] = Field(max_length=40)
    positioning_and_language: list[Claim] = Field(max_length=15)
    strategic_priorities: list[Claim] = Field(max_length=15)
    careers_signals: list[Claim] = Field(max_length=20)


class ResearchTopic(str, Enum):
    CORPORATE = "corporate"
    FUNDING = "funding"
    FINANCIALS = "financials"
    LEADERSHIP = "leadership"
    CUSTOMERS = "customers"
    COMPETITORS = "competitors"
    REVIEWS = "reviews"
    HIRING = "hiring"
    TECHNOLOGY = "technology"
    REGULATORY = "regulatory"
    NEWS = "news"
    MARKET = "market"


class SourceRef(Strict):
    passage: str | None = Field(default=None, max_length=2000)
    url: str = Field(max_length=1000)
    title: str = Field(max_length=300)
    publisher: str = Field(max_length=200)
    excerpt: str = Field(max_length=1500)
    published: str | None = Field(default=None, max_length=40)
    source_kind: SourceKind


class RawFinding(Strict):
    entity: str = ""
    time_scope: str = ""
    contradicts: list[str] = Field(default_factory=list)
    premises: list[SupportingPassage] = Field(default_factory=list)
    """What the research model returns before sources are verified against search hits."""

    topic: ResearchTopic
    statement: str = Field(min_length=1, max_length=1200)
    classification: Classification
    sources: list[SourceRef] = Field(max_length=6)


class GapResolution(Strict):
    gap: str
    topic: ResearchTopic
    statement: str
    entity: str = ""
    time_scope: str = ""


class RawFindings(Strict):
    resolved_gaps: list[GapResolution] = Field(default_factory=list)
    findings: list[RawFinding] = Field(max_length=60)
    not_found: list[str] = Field(default_factory=list, max_length=20)


class Finding(Claim):
    entity: str = ""
    time_scope: str = ""
    contradicts: list[str] = Field(default_factory=list)
    supporting_passages: list[SupportingPassage] = Field(default_factory=list)
    topic: ResearchTopic
    statement: str = Field(min_length=1, max_length=12800)
    classification: Classification
    evidence_ids: list[str] = Field(default_factory=list, max_length=128)


class ExternalFindings(Strict):
    incomplete_groups: dict[str, dict] = Field(default_factory=dict)
    findings: list[Finding]
    not_found: list[str]
    rejected: list[str] = Field(default_factory=list)


class PainKind(str, Enum):
    FUNCTIONAL = "functional"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    TECHNICAL = "technical"
    REGULATORY = "regulatory"
    STRATEGIC = "strategic"


class AnalyticalRow(Strict):
    classification: Classification = Classification.ANALYTICAL_INFERENCE
    premises: list[SupportingPassage] = Field(default_factory=list)
    supporting_passages: list[SupportingPassage] = Field(default_factory=list)


class Pain(AnalyticalRow):
    kind: PainKind
    description: str = Field(max_length=600)
    consequence_if_unsolved: str = Field(max_length=600)
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)


class MarketSize(AnalyticalRow):
    classification: Classification = Classification.THIRD_PARTY_CLAIM
    metric: str = Field(pattern=r"^(TAM|SAM|SOM|growth)$")
    value: str = Field(max_length=200)
    year: str = Field(max_length=20)
    methodology: str = Field(min_length=1, max_length=600)
    limitations: str = Field(min_length=1, max_length=600)
    evidence_ids: list[str] = Field(min_length=1, max_length=6)


class Market(Strict):
    primary_market: Claim
    adjacent_markets: list[Claim] = Field(max_length=10)
    maturity: Claim
    structural_trends: list[Claim] = Field(max_length=12)
    technological_shifts: list[Claim] = Field(max_length=10)
    regulatory_influences: list[Claim] = Field(max_length=10)
    customer_behavior_changes: list[Claim] = Field(max_length=10)
    barriers_to_entry: list[Claim] = Field(max_length=10)
    switching_costs: Claim
    commoditization_risk: Claim
    consolidation_dynamics: Claim
    sizing: list[MarketSize] = Field(max_length=8)


class CompetitorCategory(str, Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    INCUMBENT = "incumbent"
    EMERGING = "emerging"
    INTERNAL_ALTERNATIVE = "internal_alternative"


class Competitor(Strict):
    supporting_passages: list[SupportingPassage] = Field(default_factory=list)
    premises: list[SupportingPassage] = Field(default_factory=list)
    name: str = Field(max_length=200)
    category: CompetitorCategory
    offering: str = Field(max_length=400)
    target_segment: str = Field(max_length=300)
    business_model: str = Field(max_length=300)
    key_strength: str = Field(max_length=400)
    key_difference: str = Field(max_length=400)
    classification: Classification
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)


class Reproducibility(str, Enum):
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    UNKNOWN = "unknown"


class Differentiator(AnalyticalRow):
    dimension: str = Field(max_length=100)
    claimed: str = Field(max_length=600)
    observable: str = Field(max_length=600)
    reproducibility: Reproducibility
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)


class BusinessModel(Strict):
    customer_type: Claim
    ideal_customer_profile: Claim
    buyer_user_decision_maker: Claim
    revenue_model: list[Claim] = Field(min_length=1, max_length=10)
    go_to_market: list[Claim] = Field(min_length=1, max_length=10)


class Swot(Strict):
    strengths: list[Claim] = Field(min_length=1, max_length=10)
    weaknesses: list[Claim] = Field(min_length=1, max_length=10)
    opportunities: list[Claim] = Field(min_length=1, max_length=10)
    threats: list[Claim] = Field(min_length=1, max_length=10)


STRATEGIC_QUESTIONS: tuple[str, ...] = (
    "What appears to be the company's strongest competitive asset?",
    "What is easiest for competitors to replicate?",
    "What is hardest to replicate?",
    "What could accelerate its growth?",
    "What could constrain its growth?",
    "What could disrupt the company?",
    "What adjacent markets could it enter?",
    "What partnerships would make strategic sense?",
    "What capabilities might it acquire rather than build?",
    "What would materially increase or decrease the company's strategic value?",
)

MATURITY_DIMENSIONS: tuple[str, ...] = (
    "Product maturity",
    "Commercial maturity",
    "Market maturity",
    "Technology maturity",
    "Operational maturity",
    "International maturity",
    "Partner ecosystem",
    "Brand maturity",
)


class StrategicAnswer(AnalyticalRow):
    question: str = Field(max_length=200)
    answer: str = Field(min_length=1, max_length=1500)
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)


class MaturityRow(AnalyticalRow):
    dimension: str = Field(max_length=60)
    evidence: str = Field(min_length=1, max_length=800)
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)


class Opportunity(AnalyticalRow):
    kind: str = Field(max_length=80)
    description: str = Field(max_length=600)
    rationale: str = Field(max_length=800)
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)


class Analysis(Strict):
    business_model: BusinessModel
    pains: list[Pain] = Field(min_length=1, max_length=18)
    market: Market
    competitors: list[Competitor] = Field(min_length=1, max_length=25)
    differentiation: list[Differentiator] = Field(min_length=1, max_length=16)
    technology: list[Claim] = Field(max_length=20)
    commercial_signals: list[Claim] = Field(max_length=25)
    financials: list[Claim] = Field(max_length=15)
    organization: list[Claim] = Field(max_length=15)
    swot: Swot
    strategic: list[StrategicAnswer] = Field(min_length=10, max_length=10)
    maturity: list[MaturityRow] = Field(min_length=8, max_length=8)
    red_flags: list[Claim] = Field(max_length=15)
    opportunities: list[Opportunity] = Field(max_length=12)
    analyst_observations: list[Claim] = Field(max_length=12)
    open_questions: list[str] = Field(max_length=20)

    @model_validator(mode="after")
    def _fixed_sets(self) -> "Analysis":
        got_q = {s.question for s in self.strategic}
        if got_q != set(STRATEGIC_QUESTIONS):
            raise ValueError(
                "strategic answers must cover exactly these questions: "
                + " | ".join(STRATEGIC_QUESTIONS)
            )
        got_d = {m.dimension for m in self.maturity}
        if got_d != set(MATURITY_DIMENSIONS):
            raise ValueError(
                "maturity rows must cover exactly these dimensions: " + " | ".join(MATURITY_DIMENSIONS)
            )
        for obs in self.analyst_observations:
            if obs.classification != Classification.ANALYTICAL_INFERENCE:
                raise ValueError("analyst observations must be classified analytical_inference")
        return self


class NarrativeStatement(Strict):
    supporting_passages: list[SupportingPassage] = Field(default_factory=list)
    claim_id: str
    statement: str
    classification: Classification
    evidence_ids: list[str]
    premise_claim_ids: list[str] = Field(default_factory=list)


class Narrative(Strict):
    """Factual prose selects validated claims; inference retains explicit claim premises."""

    executive_summary: list[NarrativeStatement] = Field(min_length=5, max_length=10)
    what_the_company_does: list[NarrativeStatement] = Field(min_length=1, max_length=8)
    problems_it_solves: list[NarrativeStatement] = Field(min_length=1, max_length=8)
    products_and_services: list[NarrativeStatement] = Field(min_length=1, max_length=8)
    customer_segments_and_use_cases: list[NarrativeStatement] = Field(min_length=1, max_length=8)
    business_model_and_monetization: list[NarrativeStatement] = Field(min_length=1, max_length=8)
    go_to_market: list[NarrativeStatement] = Field(min_length=1, max_length=8)
    technology_and_ip: list[NarrativeStatement] = Field(min_length=1, max_length=8)
    market_landscape: list[NarrativeStatement] = Field(min_length=1, max_length=8)
    competitive_landscape: list[NarrativeStatement] = Field(min_length=1, max_length=8)
    differentiation_and_defensibility: list[NarrativeStatement] = Field(min_length=1, max_length=8)
    customers_partnerships_ecosystem: list[NarrativeStatement] = Field(min_length=1, max_length=8)
    financial_and_funding: list[NarrativeStatement] = Field(min_length=1, max_length=6)
    growth_and_traction: list[NarrativeStatement] = Field(min_length=1, max_length=6)
    risks_and_red_flags: list[NarrativeStatement] = Field(min_length=1, max_length=8)
    strategic_opportunities: list[NarrativeStatement] = Field(min_length=1, max_length=8)
    analyst_observations: list[NarrativeStatement] = Field(min_length=1, max_length=8)


# --------------------------------------------------------------------------- semantic checks


def _walk(obj: Any, path: str = "$") -> Iterator[tuple[str, Any]]:
    if isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            yield from _walk(getattr(obj, name), f"{path}.{name}")
    elif isinstance(obj, list):
        for i, x in enumerate(obj):
            yield from _walk(x, f"{path}[{i}]")
    else:
        yield path, obj


def check_refs(obj: BaseModel, ledger: EvidenceLedger) -> list[str]:
    """Return every evidence reference (list field or inline citation) that does not resolve."""
    errors: list[str] = []
    for path, value in _walk(obj):
        if path.endswith("]") and ".evidence_ids[" in path and isinstance(value, str):
            if not ledger.has(value):
                errors.append(f"{path}: unknown evidence id {value}")
        elif isinstance(value, str):
            for cited in CITATION.findall(value):
                if not ledger.has(cited):
                    errors.append(f"{path}: unknown citation [{cited}]")
    return errors


def check_classifications(obj: BaseModel, ledger: EvidenceLedger) -> list[str]:
    """Reject classifications that the cited evidence cannot support.

    verified_fact needs a primary record or two independent third-party sources
    (:func:`verified_fact_supported`); company_claim needs at least one first-party source;
    third_party_claim needs at least one third-party source.
    """
    errors: list[str] = []
    for model, path in _models(obj):
        cls = getattr(model, "classification", None)
        ids = getattr(model, "evidence_ids", None)
        if cls is None or ids is None:
            continue
        items = [ledger.get(i) for i in ids if ledger.has(i)]
        kinds = {e.source_type for e in items}
        if cls == Classification.VERIFIED_FACT and not verified_fact_supported(items, getattr(model, "statement", getattr(model, "value", "")) or ""):
            errors.append(f"{path}: verified_fact requires a primary record (government/regulatory or filing) "
                          "or two independent third_party sources on different domains")
        elif cls == Classification.COMPANY_CLAIM and SourceType.FIRST_PARTY not in kinds:
            errors.append(f"{path}: company_claim requires first_party evidence")
        elif cls == Classification.THIRD_PARTY_CLAIM and SourceType.THIRD_PARTY not in kinds:
            errors.append(f"{path}: third_party_claim requires third_party evidence")
    return errors


def _models(obj: Any, path: str = "$") -> Iterator[tuple[BaseModel, str]]:
    if isinstance(obj, BaseModel):
        yield obj, path
        for name in type(obj).model_fields:
            yield from _models(getattr(obj, name), f"{path}.{name}")
    elif isinstance(obj, list):
        for i, x in enumerate(obj):
            yield from _models(x, f"{path}[{i}]")


def semantic_errors(obj: BaseModel, ledger: EvidenceLedger) -> list[str]:
    errors = check_refs(obj, ledger) + check_classifications(obj, ledger)
    for model, path in _models(obj):
        ids = getattr(model, "evidence_ids", [])
        cls = getattr(model, "classification", None)
        items = [ledger.get(i) for i in ids if ledger.has(i)]
        for support in getattr(model, "supporting_passages", []):
            supported_value = getattr(model, support.field, None)
            if (not isinstance(supported_value, str) or support.evidence_id not in ids
                    or not ledger.has(support.evidence_id)
                    or not passage_supported(supported_value, ledger.get(support.evidence_id), support.passage)):
                errors.append(f"{path}: unsupported quoted passage")
        if cls in {Classification.COMPANY_CLAIM, Classification.THIRD_PARTY_CLAIM, Classification.VERIFIED_FACT}:
            texts = factual_texts(model)
            for item in items:
                if not any(passage_supported(text, item) for text in texts):
                    errors.append(f"{path}: citation {item.id} has no support for this claim")
            for text in texts:
                if not any(passage_supported(text, e) for e in items):
                    errors.append(f"{path}: no retrieved passage supports {text!r}; use an exact source excerpt")
        if cls == Classification.ANALYTICAL_INFERENCE and not isinstance(model, NarrativeStatement):
            premises = getattr(model, "premises", [])
            if not premises:
                errors.append(f"{path}: analytical inference requires explicit retrieved premises")
            for premise in premises:
                if not ledger.has(premise.evidence_id) or not passage_supported(premise.passage, ledger.get(premise.evidence_id)):
                    errors.append(f"{path}: unsupported inference premise")
    return errors


def factual_texts(model: BaseModel) -> list[str]:
    if hasattr(model, "statement"):
        return [model.statement]
    if isinstance(model, Attr):
        return [model.value] if model.value else []
    # Structured tables must support every factual cell, not just the row's label.
    return [v for k, v in model.model_dump().items() if isinstance(v, str)
            and k not in {"classification", "kind", "category", "question", "dimension", "metric", "reproducibility"}]


def claim_catalog(**artifacts: BaseModel) -> dict[str, dict]:
    result = {}
    for name, obj in artifacts.items():
        for model, path in _models(obj):
            if not hasattr(model, "classification") or isinstance(model, NarrativeStatement):
                continue
            for text in factual_texts(model):
                key = "C" + hashlib.sha256(f"{name}:{path}:{text}".encode()).hexdigest()[:16]
                result[key] = {"statement": text, "classification": model.classification.value,
                               "evidence_ids": getattr(model, "evidence_ids", []),
                               "premises": [p.model_dump() for p in getattr(model, "premises", [])]}
    return result


def narrative_errors(obj: Narrative, ledger: EvidenceLedger, catalog: dict[str, dict]) -> list[str]:
    errors = semantic_errors(obj, ledger)
    for row, path in _models(obj):
        if not isinstance(row, NarrativeStatement):
            continue
        source = catalog.get(row.claim_id)
        if source is None:
            errors.append(f"{path}: unknown validated claim {row.claim_id}")
            continue
        if (row.statement != source["statement"] or row.classification.value != source["classification"]
                or set(row.evidence_ids) != set(source["evidence_ids"])):
            errors.append(f"{path}: prose must preserve the statement, classification and evidence of its claim")
        if row.classification == Classification.ANALYTICAL_INFERENCE:
            if not row.premise_claim_ids or any(x not in catalog or catalog[x]["classification"] in
                    {"unknown", "analytical_inference"} for x in row.premise_claim_ids):
                errors.append(f"{path}: inference needs validated factual premise claim IDs")
    return errors


# --------------------------------------------------------------------------- mechanical repair

def repair_refs(payload: Any, ledger: EvidenceLedger) -> tuple[Any, list[str]]:
    """Remove unresolved list references and lower unsupported classifications.

    Inline assertions remain unchanged so validation can reject them. Losing the last
    citation never changes a factual assertion into an inference. Notes are persisted by
    the LLM boundary in audit.json with original and repaired values.
    """
    notes: list[str] = []

    def walk(node: Any, path: str) -> Any:
        if isinstance(node, dict):
            out = {k: walk(v, f"{path}.{k}") for k, v in node.items()}
            if out.get("classification") == "analytical_inference" and isinstance(out.get("premises"), list):
                premises_ids = [p.get("evidence_id") for p in out["premises"]
                                if isinstance(p, dict) and isinstance(p.get("evidence_id"), str)
                                and ledger.has(p["evidence_id"])]
                if premises_ids and isinstance(out.get("evidence_ids", []), list):
                    combined = list(dict.fromkeys([*(i for i in out.get("evidence_ids", []) if isinstance(i, str)), *premises_ids]))
                    if combined != out.get("evidence_ids"):
                        out["evidence_ids"] = combined
                        notes.append(f"{path}: attached inference premise references")
            ids = out.get("evidence_ids")
            if isinstance(ids, list):
                kept = [i for i in ids if isinstance(i, str) and ledger.has(i)]
                for i in ids:
                    if i not in kept:
                        notes.append(f"{path}: removed unknown evidence id {i}")
                out["evidence_ids"] = kept
                cls = out.get("classification")
                if isinstance(cls, str) and cls in {"verified_fact", "company_claim", "third_party_claim"} and not out.get("supporting_passages"):
                    supports = []
                    fields = ([("statement", out["statement"])] if "statement" in out else
                              [("value", out["value"])] if "value" in out and "metric" not in out else
                              [(k, v) for k, v in out.items() if isinstance(v, str) and k not in {"classification", "kind", "category", "question", "dimension", "metric", "reproducibility"}])
                    for field, text in fields:
                        if not isinstance(text, str):
                            continue
                        for evidence_id in kept:
                            if passage_supported(text, ledger.get(evidence_id)):
                                supports.append({"evidence_id": evidence_id, "field": field, "passage": text})
                    if supports:
                        out["supporting_passages"] = supports
                        notes.append(f"{path}: attached exact retrieved supporting passages")
                if isinstance(cls, str):
                    new = supported_classification(cls, [ledger.get(i) for i in kept], out.get("statement", out.get("value", "")) or "")
                    if "value" in out and out["value"] is None:
                        new = Classification.UNKNOWN.value  # an attribute with no value can only be unknown
                    if new != cls:
                        notes.append(f"{path}: classification {cls} -> {new} (supported by the cited evidence)")
                        out["classification"] = new
            return out
        if isinstance(node, list):
            return [walk(x, f"{path}[{i}]") for i, x in enumerate(node)]
        return node

    return walk(payload, "$"), notes


# --------------------------------------------------------------------------- source-kind sanity

# Hosts that are governments, regulators, courts, official registries or patent/trademark offices.
TRUSTED_REGISTRY_DOMAINS = frozenset({
    "sec.gov", "gov.uk", "gov.br", "gouv.fr", "gob.es", "gov.au", "govt.nz", "admin.ch",
    "bund.de", "europa.eu", "wipo.int", "epo.org", "handelsregister.de", "unternehmensregister.de",
    "bundesanzeiger.de", "infogreffe.fr", "inpi.fr", "bodacc.fr", "boe.es", "registradores.org",
    "cnmv.es", "cmvm.pt", "amf-france.org", "consob.it", "zefix.ch", "shab.ch", "justiz.gv.at",
})
_RECORD = re.compile(r"\b(registration number|company number|cnpj|registered office|incorporated|registry record|patent number|form (10-k|10-q|20-f|8-k)|annual report)\b", re.I)
_FILING = re.compile(r"\b(form (10-k|10-q|20-f|8-k)|statutory (accounts|filing)|annual report|audited financial statements)\b", re.I)


# News, press and blog sections of official hosts publish announcements, not records.
_NEWS_PATH = re.compile(
    r"/(noticias?|news|imprensa|press|press-releases?|actualites|nachrichten|aktuelles|pressemitteilungen"
    r"|prensa|comunicados?|blog|artigos?|articles?)(/|$|-|\?)",
    re.IGNORECASE,
)


def plausible_source_kind(kind: SourceKind, url: str, on_site: bool,
                          content: str | None = None, title: str = "") -> SourceKind:
    host = hostname(url)
    official = any(host == d or host.endswith("." + d) for d in TRUSTED_REGISTRY_DOMAINS)
    document = title + "\n" + (content or "")
    if kind in PRIMARY_RECORD_KINDS:
        if _NEWS_PATH.search(url):
            return SourceKind.OFFICIAL_COMPANY if on_site else SourceKind.NEWS
        document_url = re.search(r"(?:\.pdf(?:$|[?#])|/(?:filings?|annual-reports?|10-k|10-q|20-f|8-k)(?:[/?.-]|$))", url, re.I)
        filing_title = _FILING.search(title)
        if (kind == SourceKind.COMPANY_FILING and content and _FILING.search(content)
                and (document_url or filing_title) and (official or on_site)):
            return kind
        if kind == SourceKind.GOVERNMENT_REGULATORY and official and content and _RECORD.search(document):
            return kind
        return SourceKind.OFFICIAL_COMPANY if on_site else SourceKind.DATABASE_AGGREGATOR
    if on_site:
        return kind if kind == SourceKind.INVESTOR_DISCLOSURE else SourceKind.OFFICIAL_COMPANY
    return kind
