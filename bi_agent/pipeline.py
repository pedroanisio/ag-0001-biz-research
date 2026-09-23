"""Pipeline stages. Each stage reads its inputs from the run directory, writes one JSON
artifact, and can be executed on its own. ``run_all`` chains them.

Run directory layout::

    run.json        start url, timestamps, site language and report language
    pages.json      crawled pages
    evidence.json   evidence ledger (first- and third-party)
    identity.site.json  Identity from the website alone (identify)
    identity.json   Identity: the website identity, refined by external research once resolve runs
    signals.json    SiteSignals
    findings.json   ExternalFindings
    analysis.json   Analysis
    narrative.json  Narrative
    report.md       rendered report
    usage.json      tokens, web searches and estimated cost per stage
    research.partial.json  research progress, present only while research is unfinished
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel

from . import prompts
from .crawler import Crawler, Page, canonical, registrable_host, same_site
from .errors import StageError
from .i18n import DEFAULT_LANG, SUPPORTED, detect_site_lang, normalize_lang
from .llm import LLM, SearchHit, is_transient, pretty
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
    repair_refs,
    semantic_errors,
)
from .report import render_report

log = logging.getLogger("bi_agent")
T = TypeVar("T", bound=BaseModel)

PAGE_CHARS_FOR_LLM = 6_000
TOTAL_CHARS_FOR_LLM = 260_000
IDENTIFY_CHARS_FOR_LLM = 120_000  # identify reads the highest-priority pages only


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


@contextmanager
def metered(store: RunStore, llm: LLM, stage: str) -> Iterator[None]:
    """Record the API usage of one stage in usage.json, also when the stage fails."""
    start = len(llm.usage.calls)
    try:
        yield
    finally:
        calls = llm.usage.calls[start:]
        if calls:
            data = store.load_json("usage.json") if store.exists("usage.json") else {"stages": {}}
            stage_log = type(llm.usage)(llm.model, calls)
            data["model"] = llm.model
            data["stages"][stage] = stage_log.summary()
            data["stages"][stage]["per_call"] = [c.__dict__ for c in calls]
            totals: dict[str, Any] = {}
            for summary in data["stages"].values():
                for k, v in summary.items():
                    if k == "per_call":
                        continue
                    if v is None or totals.get(k, 0) is None:
                        totals[k] = None
                    else:
                        totals[k] = round(totals.get(k, 0) + v, 4)
            data["total"] = totals
            store.save_json("usage.json", data)


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
    return "".join(_page_chunks(store, ledger, budget))


def _pages_split(store: RunStore, ledger: EvidenceLedger) -> tuple[str, str]:
    """(highest-priority pages up to the identify budget, the remaining pages up to the total budget)."""
    head: list[str] = []
    tail: list[str] = []
    used = 0
    for chunk in _page_chunks(store, ledger, TOTAL_CHARS_FOR_LLM):
        (head if not tail and used + len(chunk) <= IDENTIFY_CHARS_FOR_LLM else tail).append(chunk)
        used += len(chunk)
    return "".join(head), "".join(tail)


def _site_system(store: RunStore, ledger: EvidenceLedger) -> tuple[list[dict], str]:
    """System prompt shared by identify and signals, and the pages only signals reads.

    The pages live in the system prompt, marked for caching, because the two calls force
    different tools and a ``tool_choice`` change invalidates the cached messages but not the
    cached system prompt. Both calls also declare the same two tools (see ``SITE_TOOLS``).
    """
    head, tail = _pages_split(store, ledger)
    meta = store.meta()
    system = [
        {"type": "text", "text": prompts.localized(prompts.ANALYST_ROLE, store.lang())},
        {"type": "text", "cache_control": {"type": "ephemeral"},
         "text": f"Start URL: {meta['url']}\n\nCrawled pages (each with its evidence id):\n{head}"},
    ]
    return system, tail


SITE_TOOLS = [
    ("submit_identity", Identity, "Submit the completed, fully populated result."),
    ("submit_site_signals", SiteSignals, "Submit the completed, fully populated result."),
]


def _page_chunks(store: RunStore, ledger: EvidenceLedger, budget: int) -> list[str]:
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
    return parts


# --------------------------------------------------------------------------- stage: identify


def stage_identify(store: RunStore, llm: LLM) -> Identity:
    ledger = store.ledger()
    system, _tail = _site_system(store, ledger)
    with metered(store, llm, "identify"):
        identity = llm.structured(
            system=system, user=prompts.IDENTIFY_TASK, schema=Identity, tool_name="submit_identity",
            shared_tools=SITE_TOOLS, repair=lambda p: repair_refs(p, ledger),
            semantic_check=lambda o: semantic_errors(o, ledger),
        )
    store.save_model("identity.site.json", identity)
    store.save_model("identity.json", identity)
    return identity


# --------------------------------------------------------------------------- stage: signals


def stage_signals(store: RunStore, llm: LLM) -> SiteSignals:
    ledger = store.ledger()
    identity = store.load_model("identity.json", Identity)
    system, tail = _site_system(store, ledger)
    user = f"Company: {identity.company_name.value or 'unknown'}\n\n"
    if tail:
        user += f"More crawled pages (each with its evidence id):\n{tail}\n\n"
    user += prompts.SIGNALS_TASK
    with metered(store, llm, "signals"):
        signals = llm.structured(
            system=system, user=user, schema=SiteSignals, tool_name="submit_site_signals",
            shared_tools=SITE_TOOLS, repair=lambda p: repair_refs(p, ledger),
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


RESEARCH_PROGRESS = "research.partial.json"


def stage_research(
    store: RunStore, llm: LLM, groups: dict[str, dict[str, str]] | None = None
) -> ExternalFindings:
    """Research each topic group with web_search, checkpointing after every group.

    A group that fails with a transient error (rate limit, server error, network, invalid output)
    is recorded in not_found and the next group runs. Any other API error (no credit, bad key,
    unknown model) stops the stage at once: every later call would fail the same way. Progress is
    saved after each group, so re-running the stage resumes where it stopped instead of paying for
    finished groups again.
    """
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
    groups = groups or prompts.research_groups()

    progress = store.load_json(RESEARCH_PROGRESS) if store.exists(RESEARCH_PROGRESS) else {
        "done": [], "failed": [], "findings": [], "not_found": [], "rejected": []}
    if progress["done"]:
        log.info("resuming research: %s already done", ", ".join(progress["done"]))
    system = prompts.localized(prompts.RESEARCH_SYSTEM, store.lang())
    with metered(store, llm, "research"):
        for group, topics in groups.items():
            if group in progress["done"]:
                continue
            user = prompts.research_user_prompt(
                company, meta["start"], topics, known, store.site_lang(), llm.max_search_uses)
            try:
                raw, hits = llm.researched(system=system, user=user, schema=RawFindings)
            except Exception as exc:  # noqa: BLE001 - classified below
                if not is_transient(exc):
                    raise
                log.warning("research group %s failed: %s", group, exc)
                progress["failed"].append(group)
                progress["not_found"].append(f"{group}: research call failed ({type(exc).__name__})")
                continue
            acc, rej = verify_findings(raw, hits, ledger, site_host, now_iso())
            progress["findings"].extend(f.model_dump(mode="json") for f in acc)
            progress["rejected"].extend(rej)
            progress["not_found"].extend(f"{group}: {x}" for x in raw.not_found)
            progress["done"].append(group)
            store.save_ledger(ledger)
            store.save_json(RESEARCH_PROGRESS, progress)
            log.info("group %s (%s): %d findings accepted, %d rejected", group, ", ".join(topics), len(acc), len(rej))
    if not any(g in progress["done"] for g in groups):
        raise StageError("research failed for every topic group; nothing to analyze (re-run the research stage)")
    result = ExternalFindings(
        findings=[Finding.model_validate(f) for f in progress["findings"]],
        not_found=progress["not_found"], rejected=progress["rejected"],
    )
    store.save_ledger(ledger)
    store.save_model("findings.json", result)
    store.path(RESEARCH_PROGRESS).unlink(missing_ok=True)
    return result


# --------------------------------------------------------------------------- stage: resolve

IDENTITY_TOPICS = {"corporate", "funding", "leadership", "financials", "regulatory", "news"}


def stage_resolve(store: RunStore, llm: LLM) -> Identity:
    """Refine the website identity with what external research found (registries, filings, press).

    Always starts from identity.site.json, so running it again gives the same starting point; runs
    made before this stage existed fall back to identity.json.
    """
    ledger = store.ledger()
    site_file = "identity.site.json" if store.exists("identity.site.json") else "identity.json"
    site = store.load_model(site_file, Identity)
    if site_file == "identity.json":
        store.save_model("identity.site.json", site)
    findings = store.load_model("findings.json", ExternalFindings)
    relevant = [f for f in findings.findings if f.topic.value in IDENTITY_TOPICS]
    cited = {i for f in relevant for i in f.evidence_ids}
    cited |= {i for attr in site.model_dump().values() if isinstance(attr, dict) for i in attr.get("evidence_ids", [])}
    index = "\n".join(line for line, e in zip(ledger.index_text().splitlines(), ledger) if e.id in cited)
    user = (
        "EVIDENCE (ids you may cite):\n" + index
        + "\n\nIDENTITY FROM THE WEBSITE:\n" + pretty(site)
        + "\n\nEXTERNAL FINDINGS:\n" + json.dumps([f.model_dump(mode="json") for f in relevant],
                                                   separators=(",", ":"), ensure_ascii=False)
    )
    with metered(store, llm, "resolve"):
        identity = llm.structured(
            system=prompts.localized(prompts.RESOLVE_SYSTEM, store.lang()), user=user, schema=Identity,
            tool_name="submit_identity", repair=lambda p: repair_refs(p, ledger),
            semantic_check=lambda o: semantic_errors(o, ledger),
        )
    store.save_model("identity.json", identity)
    changed = [k for k in Identity.model_fields if getattr(site, k) != getattr(identity, k)]
    log.info("identity resolved with external research; changed: %s", ", ".join(changed) or "nothing")
    return identity


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
    with metered(store, llm, "analyze"):
        analysis = llm.structured(
            system=prompts.localized(prompts.ANALYZE_SYSTEM, store.lang()), user=user, schema=Analysis,
            tool_name="submit_analysis", repair=lambda p: repair_refs(p, ledger),
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
    with metered(store, llm, "narrate"):
        narrative = llm.structured(
            system=prompts.localized(prompts.NARRATE_SYSTEM, store.lang()), user=user, schema=Narrative,
            tool_name="submit_narrative", repair=lambda p: repair_refs(p, ledger),
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
    stage_resolve(store, llm)
    stage_analyze(store, llm)
    stage_narrate(store, llm)
    return stage_report(store)
