"""Typed contracts for every stage boundary.

Every model rejects unknown fields (``extra="forbid"``). Cross-model rules that a JSON
schema cannot express (evidence ids must resolve, classifications must match the
kind of evidence cited) live in :func:`check_refs` and :func:`check_classifications`.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

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

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not EVIDENCE_ID.match(v):
            raise ValueError(f"evidence id {v!r} must match E### pattern")
        return v

    @field_validator("url")
    @classmethod
    def _url_format(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"url {v!r} must be absolute http(s)")
        return v


class EvidenceLedger:
    """Append-only registry of evidence items, keyed by id and de-duplicated by URL."""

    def __init__(self, items: list[Evidence] | None = None) -> None:
        self._items: dict[str, Evidence] = {}
        self._by_url: dict[str, str] = {}
        for item in items or []:
            self._items[item.id] = item
            self._by_url[normalize_url(item.url)] = item.id

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
    ) -> Evidence:
        existing = self.id_for_url(url)
        if existing is not None:
            return self._items[existing]
        new_id = f"E{len(self._items) + 1:03d}"
        item = Evidence(
            id=new_id,
            source_type=source_type,
            url=url,
            title=title[:300],
            publisher=publisher[:200],
            excerpt=excerpt[:2000],
            published=published,
            retrieved_at=retrieved_at,
        )
        self._items[new_id] = item
        self._by_url[normalize_url(url)] = new_id
        return item

    def index_text(self) -> str:
        """Compact id → source listing for prompts."""
        return "\n".join(
            f"[{e.id}] ({e.source_type.value}) {e.title} — {e.publisher} — {e.url}" for e in self
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps([e.model_dump(mode="json") for e in self], indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EvidenceLedger":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("evidence file must contain a list")
        return cls([Evidence.model_validate(x) for x in raw])


def normalize_url(url: str) -> str:
    u = url.strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    u = u.split("#", 1)[0]
    return u.rstrip("/")


# --------------------------------------------------------------------------- claims


class Claim(Strict):
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

    value: str | None = Field(default=None, max_length=500)
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


class Identity(Strict):
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


class Offering(Strict):
    name: str = Field(max_length=200)
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
    url: str = Field(max_length=1000)
    title: str = Field(max_length=300)
    publisher: str = Field(max_length=200)
    excerpt: str = Field(max_length=1500)
    published: str | None = Field(default=None, max_length=40)


class RawFinding(Strict):
    """What the research model returns before sources are verified against search hits."""

    topic: ResearchTopic
    statement: str = Field(min_length=1, max_length=1200)
    classification: Classification
    sources: list[SourceRef] = Field(max_length=6)


class RawFindings(Strict):
    findings: list[RawFinding] = Field(max_length=60)
    not_found: list[str] = Field(default_factory=list, max_length=20)


class Finding(Strict):
    topic: ResearchTopic
    statement: str = Field(min_length=1, max_length=1200)
    classification: Classification
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)


class ExternalFindings(Strict):
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


class Pain(Strict):
    kind: PainKind
    description: str = Field(max_length=600)
    consequence_if_unsolved: str = Field(max_length=600)
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)


class MarketSize(Strict):
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


class Differentiator(Strict):
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


class StrategicAnswer(Strict):
    question: str = Field(max_length=200)
    answer: str = Field(min_length=1, max_length=1500)
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)


class MaturityRow(Strict):
    dimension: str = Field(max_length=60)
    evidence: str = Field(min_length=1, max_length=800)
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)


class Opportunity(Strict):
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


class Narrative(Strict):
    """Prose sections. Citations are inline ``[E###]`` markers that must resolve."""

    executive_summary: list[str] = Field(min_length=5, max_length=10)
    what_the_company_does: list[str] = Field(min_length=1, max_length=8)
    problems_it_solves: list[str] = Field(min_length=1, max_length=8)
    products_and_services: list[str] = Field(min_length=1, max_length=8)
    customer_segments_and_use_cases: list[str] = Field(min_length=1, max_length=8)
    business_model_and_monetization: list[str] = Field(min_length=1, max_length=8)
    go_to_market: list[str] = Field(min_length=1, max_length=8)
    technology_and_ip: list[str] = Field(min_length=1, max_length=8)
    market_landscape: list[str] = Field(min_length=1, max_length=8)
    competitive_landscape: list[str] = Field(min_length=1, max_length=8)
    differentiation_and_defensibility: list[str] = Field(min_length=1, max_length=8)
    customers_partnerships_ecosystem: list[str] = Field(min_length=1, max_length=8)
    financial_and_funding: list[str] = Field(min_length=1, max_length=6)
    growth_and_traction: list[str] = Field(min_length=1, max_length=6)
    risks_and_red_flags: list[str] = Field(min_length=1, max_length=8)
    strategic_opportunities: list[str] = Field(min_length=1, max_length=8)
    analyst_observations: list[str] = Field(min_length=1, max_length=8)


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

    verified_fact needs at least one third-party source; company_claim needs at least one
    first-party source; third_party_claim needs at least one third-party source.
    """
    errors: list[str] = []
    for model, path in _models(obj):
        cls = getattr(model, "classification", None)
        ids = getattr(model, "evidence_ids", None)
        if cls is None or ids is None:
            continue
        kinds = {ledger.get(i).source_type for i in ids if ledger.has(i)}
        if cls == Classification.VERIFIED_FACT and SourceType.THIRD_PARTY not in kinds:
            errors.append(f"{path}: verified_fact requires independent (third_party) evidence")
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
    return check_refs(obj, ledger) + check_classifications(obj, ledger)


# --------------------------------------------------------------------------- mechanical repair

_SOURCED = {c.value for c in (Classification.VERIFIED_FACT, Classification.COMPANY_CLAIM, Classification.THIRD_PARTY_CLAIM)}


def _supported(cls: str, kinds: set[SourceType]) -> str:
    """The strongest classification the cited evidence supports (same rules as the research stage)."""
    if cls not in _SOURCED:
        return cls
    if not kinds:
        return Classification.ANALYTICAL_INFERENCE.value
    first, third = SourceType.FIRST_PARTY in kinds, SourceType.THIRD_PARTY in kinds
    if cls == Classification.VERIFIED_FACT.value and not third:
        return Classification.COMPANY_CLAIM.value
    if cls == Classification.COMPANY_CLAIM.value and not first:
        return Classification.THIRD_PARTY_CLAIM.value
    if cls == Classification.THIRD_PARTY_CLAIM.value and not third:
        return Classification.COMPANY_CLAIM.value
    return cls


def repair_refs(payload: Any, ledger: EvidenceLedger) -> tuple[Any, list[str]]:
    """Fix, without another model call, the reference errors :func:`semantic_errors` would reject.

    Evidence ids and inline ``[E###]`` citations that are not in the ledger are removed, and a
    classification the remaining evidence cannot support is lowered to the one it does support
    (a claim left with no evidence becomes an analytical inference). Every change is returned as
    a note so it can be logged. Structural problems are left for validation to report.
    """
    notes: list[str] = []

    def fix_text(text: str, path: str) -> str:
        def sub(m: re.Match) -> str:
            if ledger.has(m.group(1)):
                return m.group(0)
            notes.append(f"{path}: removed unknown citation [{m.group(1)}]")
            return ""

        fixed = CITATION.sub(sub, text)
        return re.sub(r"[ \t]{2,}", " ", re.sub(r"\s+([.,;:])", r"\1", fixed)) if fixed != text else text

    def walk(node: Any, path: str) -> Any:
        if isinstance(node, dict):
            out = {k: walk(v, f"{path}.{k}") for k, v in node.items()}
            ids = out.get("evidence_ids")
            if isinstance(ids, list):
                kept = [i for i in ids if isinstance(i, str) and ledger.has(i)]
                for i in ids:
                    if i not in kept:
                        notes.append(f"{path}: removed unknown evidence id {i}")
                out["evidence_ids"] = kept
                cls = out.get("classification")
                if isinstance(cls, str):
                    new = _supported(cls, {ledger.get(i).source_type for i in kept})
                    if "value" in out and out["value"] is None and new != cls:
                        new = Classification.UNKNOWN.value  # an attribute with no value can only be unknown
                    if new != cls:
                        notes.append(f"{path}: classification {cls} -> {new} (supported by the cited evidence)")
                        out["classification"] = new
            return out
        if isinstance(node, list):
            return [walk(x, f"{path}[{i}]") for i, x in enumerate(node)]
        if isinstance(node, str):
            return fix_text(node, path)
        return node

    return walk(payload, "$"), notes
