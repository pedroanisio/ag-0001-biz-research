from __future__ import annotations

import pytest
from pydantic import ValidationError

from bi_agent.models import (
    SourceKind,
    Analysis,
    Attr,
    Claim,
    Classification,
    Evidence,
    EvidenceLedger,
    Narrative,
    SourceType,
    check_classifications,
    check_refs,
    normalize_url,
    semantic_errors,
)
from tests.conftest import analysis_payload, narrative_payload


def _ledger() -> EvidenceLedger:
    led = EvidenceLedger()
    led.add(source_type=SourceType.FIRST_PARTY, url="https://acme.test/", title="Home", publisher="acme.test",
            excerpt="x", retrieved_at="2026-01-01T00:00:00+00:00")
    led.add(source_type=SourceType.THIRD_PARTY, url="https://news.test/a", title="News", publisher="News",
            excerpt="y", retrieved_at="2026-01-01T00:00:00+00:00", retrieval_method="web_fetch", content="s. News content")
    return led


def test_evidence_rejects_bad_id_and_url():
    with pytest.raises(ValidationError):
        Evidence(id="X1", source_type="first_party", url="https://a.test", title="t", publisher="p", excerpt="e",
                 retrieved_at="now")
    with pytest.raises(ValidationError):
        Evidence(id="E001", source_type="first_party", url="ftp://a.test", title="t", publisher="p", excerpt="e",
                 retrieved_at="now")


def test_evidence_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Evidence(id="E001", source_type="first_party", url="https://a.test", title="t", publisher="p", excerpt="e",
                 retrieved_at="now", extra_field="nope")


def test_ledger_assigns_ids_and_dedups_by_url(tmp_path):
    led = _ledger()
    dup = led.add(source_type=SourceType.THIRD_PARTY, url="http://WWW.news.test/a/", title="dup", publisher="p",
                  excerpt="z", retrieved_at="t")
    assert dup.id == "E003" and len(led) == 3
    assert led.id_for_url("https://news.test/a#frag") == "E002"
    assert led.has("E001") and not led.has("E009")
    assert "[E001] (first_party) Home" in led.index_text()
    led.save(tmp_path / "e.json")
    again = EvidenceLedger.load(tmp_path / "e.json")
    assert [e.id for e in again] == ["E001", "E002", "E003"]


def test_ledger_load_rejects_non_list(tmp_path):
    (tmp_path / "e.json").write_text('{"a": 1}')
    with pytest.raises(ValueError):
        EvidenceLedger.load(tmp_path / "e.json")


def test_normalize_url():
    assert normalize_url("HTTPS://www.Example.com/path/#x") == "https://www.example.com/path/"


@pytest.mark.parametrize("cls", ["verified_fact", "company_claim", "third_party_claim"])
def test_claim_sourced_classification_requires_evidence(cls):
    with pytest.raises(ValidationError):
        Claim(statement="s", classification=cls, evidence_ids=[])
    assert Claim(statement="s", classification=cls, evidence_ids=["E001"]).evidence_ids == ["E001"]


def test_claim_unknown_and_inference_need_no_evidence():
    assert Claim(statement="s", classification="unknown").evidence_ids == []
    assert Claim(statement="s", classification="analytical_inference").classification is Classification.ANALYTICAL_INFERENCE


def test_claim_rejects_bad_evidence_id():
    with pytest.raises(ValidationError):
        Claim(statement="s", classification="company_claim", evidence_ids=["bogus"])


def test_attr_consistency_rules():
    with pytest.raises(ValidationError):
        Attr(value=None, classification="company_claim", evidence_ids=["E001"])
    with pytest.raises(ValidationError):
        Attr(value="x", classification="unknown")
    with pytest.raises(ValidationError):
        Attr(value="x", classification="company_claim", evidence_ids=[])
    assert Attr(value="x", classification="analytical_inference").value == "x"


def test_check_refs_flags_unknown_ids_and_citations():
    led = _ledger()
    c = Claim(statement="cites [E077] inline", classification="company_claim", evidence_ids=["E001", "E099"])
    errors = check_refs(c, led)
    assert any("E099" in e for e in errors) and any("[E077]" in e for e in errors)
    assert check_refs(Claim(statement="ok [E002]", classification="company_claim", evidence_ids=["E001"]), led) == []


def test_check_classifications_matches_source_type():
    led = _ledger()
    bad_fact = Claim(statement="s", classification="verified_fact", evidence_ids=["E001"])  # first-party only
    bad_company = Claim(statement="s", classification="company_claim", evidence_ids=["E002"])  # third-party only
    bad_third = Claim(statement="s", classification="third_party_claim", evidence_ids=["E001"])
    one_source_fact = Claim(statement="s", classification="verified_fact", evidence_ids=["E001", "E002"])
    assert check_classifications(bad_fact, led) and check_classifications(bad_company, led)
    assert check_classifications(bad_third, led)
    assert any("two independent" in e for e in check_classifications(one_source_fact, led))
    led.add(source_type=SourceType.THIRD_PARTY, url="https://other.test/b", title="Other", publisher="Other",
            excerpt="z", retrieved_at="r", source_kind=SourceKind.NEWS, retrieval_method="web_fetch", content="s. Other content") # E003
    led.add(source_type=SourceType.THIRD_PARTY, url="https://www.gov.br/receita/x", title="Registry",
            publisher="Receita", excerpt="r", retrieved_at="r", source_kind=SourceKind.GOVERNMENT_REGULATORY, retrieval_method="web_fetch", content="s. Company number 123") # E004
    led.add(source_type=SourceType.THIRD_PARTY, url="https://news.test/other", title="Same host", publisher="News",
            excerpt="z", retrieved_at="r", source_kind=SourceKind.NEWS, retrieval_method="web_fetch", content="s. Same news publisher") # E005
    led.add(source_type=SourceType.THIRD_PARTY, url="https://forum.test/t", title="Forum", publisher="Forum",
            excerpt="z", retrieved_at="r", source_kind=SourceKind.FORUM_SOCIAL, retrieval_method="web_fetch", content="s. Forum statement") # E006
    two_sources = Claim(statement="s", classification="verified_fact", evidence_ids=["E002", "E003"])
    registry = Claim(statement="s", classification="verified_fact", evidence_ids=["E004"])
    same_domain = Claim(statement="s", classification="verified_fact", evidence_ids=["E002", "E005"])
    with_forum = Claim(statement="s", classification="verified_fact", evidence_ids=["E002", "E006"])
    assert check_classifications(two_sources, led) == [] and semantic_errors(registry, led) == []
    assert check_classifications(same_domain, led) and check_classifications(with_forum, led)


def test_analysis_requires_exact_strategic_questions_and_maturity_dimensions():
    p = analysis_payload()
    p["strategic"][0]["question"] = "A different question?"
    with pytest.raises(ValidationError, match="strategic answers must cover"):
        Analysis.model_validate(p)
    p = analysis_payload()
    p["maturity"][0]["dimension"] = "Vibes maturity"
    with pytest.raises(ValidationError, match="maturity rows must cover"):
        Analysis.model_validate(p)
    p = analysis_payload()
    p["analyst_observations"][0] = {"statement": "s", "classification": "company_claim", "evidence_ids": ["E001"]}
    with pytest.raises(ValidationError, match="analytical_inference"):
        Analysis.model_validate(p)
    assert Analysis.model_validate(analysis_payload()).swot.strengths


def test_market_sizing_requires_source_and_methodology():
    p = analysis_payload()
    p["market"]["sizing"] = [{"metric": "TAM", "value": "$1B", "year": "2025", "methodology": "", "limitations": "vendor",
                              "evidence_ids": ["E001"]}]
    with pytest.raises(ValidationError):
        Analysis.model_validate(p)
    p["market"]["sizing"] = [{"metric": "TAM", "value": "$1B", "year": "2025", "methodology": "top-down",
                              "limitations": "vendor", "evidence_ids": []}]
    with pytest.raises(ValidationError):
        Analysis.model_validate(p)


def test_narrative_executive_summary_bounds():
    p = narrative_payload()
    p["executive_summary"] = ["one", "two"]
    with pytest.raises(ValidationError):
        Narrative.model_validate(p)
    assert len(Narrative.model_validate(narrative_payload()).executive_summary) == 5


def test_repair_refs_drops_unknown_ids_and_lowers_classifications():
    from bi_agent.models import repair_refs

    ledger = EvidenceLedger()
    first = ledger.add(source_type=SourceType.FIRST_PARTY, url="https://a.test/", title="t", publisher="p",
                       excerpt="e", retrieved_at="r").id
    third = ledger.add(source_type=SourceType.THIRD_PARTY, url="https://n.test/", title="t", publisher="p",
                       excerpt="e", retrieved_at="r").id
    registry = ledger.add(source_type=SourceType.THIRD_PARTY, url="https://www.infogreffe.fr/x", title="t",
                          publisher="p", excerpt="e", retrieved_at="r", source_kind=SourceKind.GOVERNMENT_REGULATORY).id
    payload = {
        "a": {"classification": "verified_fact", "evidence_ids": [first, "E999"], "statement": f"x [{first}] [E999]."},
        "b": {"classification": "company_claim", "evidence_ids": [third]},
        "c": {"classification": "third_party_claim", "evidence_ids": [first]},
        "d": {"classification": "company_claim", "evidence_ids": ["E998"]},
        "e": {"value": None, "classification": "company_claim", "evidence_ids": ["E997"]},
        "f": {"classification": "verified_fact", "evidence_ids": [third]},
        "h": {"classification": "verified_fact", "evidence_ids": [registry]},
        "g": {"classification": "unknown", "evidence_ids": []},
        "n": 3,
    }
    fixed, notes = repair_refs(payload, ledger)
    assert fixed["a"] == {"classification": "company_claim", "evidence_ids": [first], "statement": f"x [{first}] [E999]."}
    assert fixed["b"]["classification"] == "third_party_claim"
    assert fixed["c"]["classification"] == "company_claim"
    assert fixed["d"] == {"classification": "company_claim", "evidence_ids": []}
    assert fixed["e"]["classification"] == "unknown"
    assert fixed["f"]["classification"] == "third_party_claim"  # one independent source is not verification
    assert fixed["h"]["classification"] == "third_party_claim" and fixed["g"]["classification"] == "unknown"
    assert fixed["n"] == 3
    assert any("E999" in n for n in notes)
    assert repair_refs({"s": "clean [E001]"}, EvidenceLedger()) == ({"s": "clean [E001]"}, [])



def test_government_press_releases_are_news_not_primary_records(tmp_path):
    from bi_agent.models import plausible_source_kind

    gov = SourceKind.GOVERNMENT_REGULATORY
    assert plausible_source_kind(gov, "https://paraiba.pb.gov.br/noticias/pbgas-conecta", False) is SourceKind.NEWS
    assert plausible_source_kind(gov, "https://www.sec.gov/news/press-release/2024-1", False) is SourceKind.NEWS
    assert plausible_source_kind(gov, "https://www.gov.br/receitafederal/pt-br/cnpj", False, "CNPJ 123") is gov
    led = EvidenceLedger()
    led.add(source_type=SourceType.THIRD_PARTY, url="https://paraiba.pb.gov.br/noticias/x", title="t", publisher="p",
            excerpt="e", retrieved_at="r", source_kind=gov)  # saved under the older rule
    led.save(tmp_path / "e.json")
    assert EvidenceLedger.load(tmp_path / "e.json").get("E001").source_kind is SourceKind.NEWS
