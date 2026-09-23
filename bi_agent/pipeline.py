"""Pipeline stages. Each stage reads its inputs from the run directory, writes one JSON
artifact, and can be executed on its own. ``run_all`` chains them.

Run directory layout::

    run.json        start url, timestamps, site language and report language
    pages.json      crawled pages
    evidence.json   evidence ledger (first- and third-party)
    identity.json   Identity
    signals.json    SiteSignals
    findings.json   ExternalFindings
    analysis.json   Analysis
    narrative.json  Narrative
    report.md       rendered report
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel

from . import prompts
from .crawler import Crawler, Page, canonical, registrable_host, same_site
from .errors import StageError
from .i18n import DEFAULT_LANG, SUPPORTED, detect_site_lang, normalize_lang
from .llm import LLM, SearchHit, pretty
from .models import (
    Analysis,
    Classification,
    EvidenceLedger,
    ExternalFindings,
    Finding,
    Identity,
    Narrative,
    RawFindings,
    SiteSignals,
    SourceType,
    check_refs,
    normalize_url,
    semantic_errors,
)
from .report import render_report

log = logging.getLogger("bi_agent")
T = TypeVar("T", bound=BaseModel)

PAGE_CHARS_FOR_LLM = 6_000
TOTAL_CHARS_FOR_LLM = 260_000


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunStore:
    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.dir / name

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def save_model(self, name: str, obj: BaseModel) -> None:
        self.path(name).write_text(json.dumps(obj.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

    def load_model(self, name: str, schema: type[T]) -> T:
        if not self.exists(name):
            raise StageError(f"missing {name}; run the earlier stage first")
        return schema.model_validate(json.loads(self.path(name).read_text(encoding="utf-8")))

    def save_json(self, name: str, data: Any) -> None:
        self.path(name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_json(self, name: str) -> Any:
        if not self.exists(name):
            raise StageError(f"missing {name}; run the earlier stage first")
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def ledger(self) -> EvidenceLedger:
        if not self.exists("evidence.json"):
            raise StageError("missing evidence.json; run the crawl stage first")
        return EvidenceLedger.load(self.path("evidence.json"))

    def save_ledger(self, ledger: EvidenceLedger) -> None:
        ledger.save(self.path("evidence.json"))

    def meta(self) -> dict:
        return self.load_json("run.json")

    def site_lang(self) -> str:
        return self.meta().get("site_lang", DEFAULT_LANG)

    def lang(self) -> str:
        """Language the report is written in: ``--lang`` if given, else the site's language."""
        meta = self.meta()
        return meta.get("lang") or meta.get("site_lang", DEFAULT_LANG)

    def set_lang(self, lang: str) -> None:
        code = normalize_lang(lang)
        if code is None:
            raise StageError(f"unsupported language {lang!r}; choose from {', '.join(SUPPORTED)}")
        meta = self.meta()
        meta["lang"] = code
        self.save_json("run.json", meta)


# --------------------------------------------------------------------------- stage: crawl


def stage_crawl(store: RunStore, url: str, crawler: Crawler, lang: str | None = None) -> list[Page]:
    pages = crawler.crawl(url)
    site_lang = detect_site_lang(pages)
    ledger = EvidenceLedger()
    for p in pages:
        ledger.add(
            source_type=SourceType.FIRST_PARTY, url=p.url, title=p.title or p.url,
            publisher=urlparse(p.url).hostname or "company website",
            excerpt=(p.description + "\n" + p.text)[:1500], retrieved_at=p.fetched_at,
        )
    store.save_json("run.json", {"url": url, "start": pages[0].url if pages else url, "started_at": now_iso(),
                                 "pages": len(pages), "site_lang": site_lang, "lang": None})
    if lang:
        store.set_lang(lang)
    store.save_json("pages.json", [p.to_dict() for p in pages])
    store.save_ledger(ledger)
    log.info("crawled %d pages, %d evidence items, site language %s", len(pages), len(ledger), site_lang)
    return pages


def _pages_block(store: RunStore, ledger: EvidenceLedger, budget: int = TOTAL_CHARS_FOR_LLM) -> str:
    pages = [Page.from_dict(d) for d in store.load_json("pages.json")]
    pages.sort(key=lambda p: -p.score)
    parts: list[str] = []
    used = 0
    for p in pages:
        eid = ledger.id_for_url(p.url)
        if eid is None:
            continue
        body = p.text[:PAGE_CHARS_FOR_LLM]
        chunk = f"\n=== [{eid}] {p.url}\nTitle: {p.title}\nDescription: {p.description}\n"
        if p.json_ld:
            chunk += "JSON-LD: " + json.dumps(p.json_ld)[:1500] + "\n"
        chunk += body + "\n"
        if used + len(chunk) > budget:
            break
        parts.append(chunk)
        used += len(chunk)
    return "".join(parts)


# --------------------------------------------------------------------------- stage: identify


def stage_identify(store: RunStore, llm: LLM) -> Identity:
    ledger = store.ledger()
    meta = store.meta()
    user = f"Start URL: {meta['url']}\n\nCrawled pages (each with its evidence id):\n{_pages_block(store, ledger, 120_000)}"
    identity = llm.structured(
        system=prompts.localized(prompts.IDENTIFY_SYSTEM, store.lang()), user=user, schema=Identity, tool_name="submit_identity",
        semantic_check=lambda o: semantic_errors(o, ledger),
    )
    store.save_model("identity.json", identity)
    return identity


# --------------------------------------------------------------------------- stage: signals


def stage_signals(store: RunStore, llm: LLM) -> SiteSignals:
    ledger = store.ledger()
    identity = store.load_model("identity.json", Identity)
    user = (
        f"Company: {identity.company_name.value or 'unknown'}\n\nCrawled pages (each with its evidence id):\n"
        f"{_pages_block(store, ledger)}"
    )
    signals = llm.structured(
        system=prompts.localized(prompts.SIGNALS_SYSTEM, store.lang()), user=user, schema=SiteSignals, tool_name="submit_site_signals",
        semantic_check=lambda o: semantic_errors(o, ledger),
    )
    store.save_model("signals.json", signals)
    return signals


# --------------------------------------------------------------------------- stage: research


def verify_findings(
    raw: RawFindings, hits: list[SearchHit], ledger: EvidenceLedger, site_host: str, retrieved_at: str
) -> tuple[list[Finding], list[str]]:
    """Convert raw findings into ledger-backed findings.

    A source is accepted only if its URL was returned by web_search in this run or belongs to the
    company's own site. Findings left with no accepted source are rejected (returned as messages),
    except ``unknown`` findings, which need no source.
    """
    hit_urls = {normalize_url(h.url): h for h in hits}
    accepted: list[Finding] = []
    rejected: list[str] = []
    for f in raw.findings:
        ids: list[str] = []
        for s in f.sources:
            key = normalize_url(s.url)
            on_site = same_site(s.url, site_host) if s.url.startswith("http") else False
            if key not in hit_urls and not on_site:
                rejected.append(f"[{f.topic.value}] dropped source not returned by search: {s.url}")
                continue
            ev = ledger.add(
                source_type=SourceType.FIRST_PARTY if on_site else SourceType.THIRD_PARTY,
                url=s.url, title=s.title or (hit_urls[key].title if key in hit_urls else s.url),
                publisher=s.publisher or (urlparse(s.url).hostname or "unknown"),
                excerpt=s.excerpt, published=s.published, retrieved_at=retrieved_at,
            )
            if ev.id not in ids:
                ids.append(ev.id)
        cls = f.classification
        if not ids and cls != Classification.UNKNOWN:
            rejected.append(f"[{f.topic.value}] dropped finding with no verifiable source: {f.statement[:120]}")
            continue
        kinds = {ledger.get(i).source_type for i in ids}
        if cls == Classification.VERIFIED_FACT and SourceType.THIRD_PARTY not in kinds:
            cls = Classification.COMPANY_CLAIM  # only the company says so
        if cls == Classification.THIRD_PARTY_CLAIM and SourceType.THIRD_PARTY not in kinds:
            cls = Classification.COMPANY_CLAIM
        if cls == Classification.COMPANY_CLAIM and SourceType.FIRST_PARTY not in kinds:
            cls = Classification.THIRD_PARTY_CLAIM
        accepted.append(Finding(topic=f.topic, statement=f.statement, classification=cls, evidence_ids=ids))
    return accepted, rejected


def stage_research(store: RunStore, llm: LLM, topics: dict[str, str] | None = None) -> ExternalFindings:
    ledger = store.ledger()
    identity = store.load_model("identity.json", Identity)
    signals = store.load_model("signals.json", SiteSignals)
    meta = store.meta()
    site_host = registrable_host(urlparse(canonical(meta["start"])).hostname or "")
    company = identity.company_name.value or site_host
    known_lines = [f"- offering: {o.name}: {o.problem_solved}" for o in signals.offerings[:8]]
    known_lines += [f"- segment: {c.statement}" for c in signals.customer_segments[:5]]
    if identity.headquarters.value:
        known_lines.append(f"- headquarters: {identity.headquarters.value}")
    known = "\n".join(known_lines) or "- nothing extracted from the website"
    all_findings: list[Finding] = []
    not_found: list[str] = []
    rejected: list[str] = []
    system = prompts.localized(prompts.RESEARCH_SYSTEM, store.lang())
    for topic, guidance in (topics or prompts.RESEARCH_TOPICS).items():
        user = prompts.research_user_prompt(company, meta["start"], topic, guidance, known, store.site_lang())
        try:
            raw, hits = llm.researched(system=system, user=user, schema=RawFindings)
        except Exception as exc:  # noqa: BLE001 - one failed topic must not sink the run
            log.warning("research topic %s failed: %s", topic, exc)
            not_found.append(f"{topic}: research call failed ({type(exc).__name__})")
            continue
        acc, rej = verify_findings(raw, hits, ledger, site_host, now_iso())
        all_findings.extend(acc)
        rejected.extend(rej)
        not_found.extend(f"{topic}: {x}" for x in raw.not_found)
        log.info("topic %s: %d findings accepted, %d rejected", topic, len(acc), len(rej))
    result = ExternalFindings(findings=all_findings, not_found=not_found, rejected=rejected)
    store.save_ledger(ledger)
    store.save_model("findings.json", result)
    return result


# --------------------------------------------------------------------------- stage: analyze


def stage_analyze(store: RunStore, llm: LLM) -> Analysis:
    ledger = store.ledger()
    identity = store.load_model("identity.json", Identity)
    signals = store.load_model("signals.json", SiteSignals)
    findings = store.load_model("findings.json", ExternalFindings)
    user = (
        "EVIDENCE LEDGER (only these ids may be cited):\n" + ledger.index_text()
        + "\n\nIDENTITY:\n" + pretty(identity)
        + "\n\nWEBSITE SIGNALS:\n" + pretty(signals)
        + "\n\nEXTERNAL FINDINGS:\n" + pretty(findings)
    )
    analysis = llm.structured(
        system=prompts.localized(prompts.ANALYZE_SYSTEM, store.lang()), user=user, schema=Analysis, tool_name="submit_analysis",
        semantic_check=lambda o: semantic_errors(o, ledger),
    )
    store.save_model("analysis.json", analysis)
    return analysis


# --------------------------------------------------------------------------- stage: narrate


def stage_narrate(store: RunStore, llm: LLM) -> Narrative:
    ledger = store.ledger()
    identity = store.load_model("identity.json", Identity)
    analysis = store.load_model("analysis.json", Analysis)
    findings = store.load_model("findings.json", ExternalFindings)
    user = (
        "EVIDENCE LEDGER (only these ids may be cited):\n" + ledger.index_text()
        + "\n\nIDENTITY:\n" + pretty(identity)
        + "\n\nANALYSIS:\n" + pretty(analysis)
        + "\n\nEXTERNAL FINDINGS:\n" + pretty(findings)
    )
    narrative = llm.structured(
        system=prompts.localized(prompts.NARRATE_SYSTEM, store.lang()), user=user, schema=Narrative, tool_name="submit_narrative",
        semantic_check=lambda o: check_refs(o, ledger),
    )
    store.save_model("narrative.json", narrative)
    return narrative


# --------------------------------------------------------------------------- stage: report


def stage_report(store: RunStore) -> str:
    ledger = store.ledger()
    md = render_report(
        meta=store.meta(),
        identity=store.load_model("identity.json", Identity),
        signals=store.load_model("signals.json", SiteSignals),
        findings=store.load_model("findings.json", ExternalFindings),
        analysis=store.load_model("analysis.json", Analysis),
        narrative=store.load_model("narrative.json", Narrative),
        ledger=ledger,
        access_date=now_iso()[:10],
        lang=store.lang(),
    )
    store.path("report.md").write_text(md, encoding="utf-8")
    return md


def run_all(store: RunStore, url: str, crawler: Crawler, llm: LLM, lang: str | None = None) -> str:
    stage_crawl(store, url, crawler, lang)
    stage_identify(store, llm)
    stage_signals(store, llm)
    stage_research(store, llm)
    stage_analyze(store, llm)
    stage_narrate(store, llm)
    return stage_report(store)
