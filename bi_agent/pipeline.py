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
    report.pdf      the same report, typeset for reading and sharing
    usage.json      tokens, web searches and estimated cost per stage
    research.partial.json  research progress, present only while research is unfinished
"""

from __future__ import annotations

import json
import logging
import time
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
    SourceKind,
    SourceType,
    check_refs,
    plausible_source_kind,
    supported_classification,
    normalize_url,
    repair_refs,
    semantic_errors,
)
from .pdf import render_pdf
from .report import render_report

log = logging.getLogger("bi_agent")
T = TypeVar("T", bound=BaseModel)

PAGE_CHARS_FOR_LLM = 6_000
TOTAL_CHARS_FOR_LLM = 260_000
# A site below either threshold gives the analysis little to work with, so research is expanded.
THIN_SITE_CHARS = 20_000
THIN_SITE_PAGES = 5
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
            source_kind=SourceKind.OFFICIAL_COMPANY,
        )
    site_chars = sum(len(p.text) for p in pages)
    js_pages = sum(1 for p in pages if p.js_rendered)
    thin = site_chars < THIN_SITE_CHARS or len(pages) < THIN_SITE_PAGES
    store.save_json("run.json", {"url": url, "start": pages[0].url if pages else url, "started_at": now_iso(),
                                 "pages": len(pages), "site_lang": site_lang, "lang": None,
                                 "site_chars": site_chars, "js_rendered_pages": js_pages, "thin_site": thin})
    if js_pages:
        log.warning("%d of %d pages look JavaScript-rendered (almost no text without running scripts); the "
                    "crawler does not run JavaScript, so their content is missing", js_pages, len(pages))
    if thin:
        log.warning("the website yielded little text (%d characters from %d pages); external research will be "
                    "expanded", site_chars, len(pages))
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

    A source is accepted only if its URL was returned by web_search or web_fetch in this run or
    belongs to the company's own site. Findings left with no accepted source are rejected (returned
    as messages), except ``unknown`` findings, which need no source. Classifications are lowered to
    what the sources support (:func:`models.supported_classification`): a verified fact needs a
    primary record or two independent sources.
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
            kind = plausible_source_kind(s.source_kind, s.url, on_site)
            if kind is not s.source_kind:
                log.info("source kind of %s lowered from %s to %s", s.url, s.source_kind.value, kind.value)
            ev = ledger.add(
                source_type=SourceType.FIRST_PARTY if on_site else SourceType.THIRD_PARTY,
                url=s.url, title=s.title or (hit_urls[key].title if key in hit_urls else s.url),
                publisher=s.publisher or (urlparse(s.url).hostname or "unknown"),
                excerpt=s.excerpt, published=s.published, retrieved_at=retrieved_at, source_kind=kind,
            )
            if ev.source_kind is None:  # an item saved before source kinds existed
                ev.source_kind = kind
            if ev.id not in ids:
                ids.append(ev.id)
        cls = f.classification
        if not ids and cls != Classification.UNKNOWN:
            rejected.append(f"[{f.topic.value}] dropped finding with no verifiable source: {f.statement[:120]}")
            continue
        if ids:
            cls = Classification(supported_classification(cls.value, [ledger.get(i) for i in ids]))
        accepted.append(Finding(topic=f.topic, statement=f.statement, classification=cls, evidence_ids=ids))
    return accepted, rejected


RESEARCH_PROGRESS = "research.partial.json"
FOLLOWUP_ROUNDS = 2
MIN_NEW_FINDINGS = 3  # a follow-up round adding fewer new findings than this ends the research
THIN_SITE_SEARCH_FACTOR = 1.5


def _statement_key(text: str) -> str:
    return " ".join(text.lower().split())


def _research_call(
    store: RunStore, llm: LLM, progress: dict, name: str, system: str, user: str, ledger: EvidenceLedger,
    site_host: str, max_searches: int, label: str = "",
) -> int | None:
    """Run one research call and fold it into ``progress``; returns the number of new findings,
    or None when the call failed transiently. Permanent API errors propagate. ``label`` says where
    the call sits in the stage and what it covers."""
    label = label or name
    log.info("research %s: searching (up to %d web searches) ...", label, max_searches)
    start = time.monotonic()
    try:
        raw, hits = llm.researched(system=system, user=user, schema=RawFindings, max_search_uses=max_searches)
    except Exception as exc:  # noqa: BLE001 - classified below
        if not is_transient(exc):
            raise
        log.warning("research %s: failed after %s, continuing: %s", label, _elapsed(start), exc)
        progress["failed"].append(name)
        progress["not_found"].append(f"{name}: research call failed ({type(exc).__name__})")
        return None
    acc, rej = verify_findings(raw, hits, ledger, site_host, now_iso())
    if name in progress["failed"]:  # an earlier attempt failed; this one succeeded
        progress["failed"].remove(name)
        progress["not_found"] = [x for x in progress["not_found"] if not x.startswith(f"{name}: research call failed")]
    seen = {_statement_key(f["statement"]) for f in progress["findings"]}
    new = [f for f in acc if _statement_key(f.statement) not in seen]
    progress["findings"].extend(f.model_dump(mode="json") for f in new)
    progress["rejected"].extend(rej)
    progress["not_found"].extend(f"{name}: {x}" for x in raw.not_found)
    progress["done"].append(name)
    store.save_ledger(ledger)
    store.save_json(RESEARCH_PROGRESS, progress)
    log.info("research %s: %d new findings, %d rejected (%s; %d findings in total)",
             label, len(new), len(rej), _elapsed(start), len(progress["findings"]))
    return len(new)


def stage_research(
    store: RunStore, llm: LLM, groups: dict[str, dict[str, str]] | None = None,
    followup_rounds: int = FOLLOWUP_ROUNDS,
) -> ExternalFindings:
    """Research the company beyond its website, then follow the leads that research surfaces.

    1. One call per topic group (web_search plus web_fetch for reading a filing or report in full).
    2. Up to ``followup_rounds`` follow-up calls. Each is given what is known so far and investigates
       the entities discovered (parent, investors, founders, key competitors), contradictions,
       single-source claims and gaps. The rounds stop early once one adds fewer than
       MIN_NEW_FINDINGS new findings, the point of diminishing returns.

    When the website yielded little text (thin or JavaScript-rendered), every call gets more
    searches and one more follow-up round is allowed.

    A call that fails with a transient error (rate limit, server error, network, invalid output) is
    recorded in not_found and research continues. Any other API error (no credit, bad key, unknown
    model) stops the stage at once: every later call would fail the same way. Progress is saved
    after each call, so re-running the stage resumes where it stopped.
    """
    ledger = store.ledger()
    identity = store.load_model("identity.json", Identity)
    signals = store.load_model("signals.json", SiteSignals)
    meta = store.meta()
    site_host = registrable_host(urlparse(canonical(meta["start"])).hostname or "")
    company = identity.company_name.value or site_host
    other_names = [v for v in (identity.legal_name.value, identity.brands.value, identity.parent_company.value)
                   if v and v != company]
    known_lines = [f"- offering: {o.name}: {o.problem_solved}" for o in signals.offerings[:8]]
    known_lines += [f"- segment: {c.statement}" for c in signals.customer_segments[:5]]
    if identity.headquarters.value:
        known_lines.append(f"- headquarters: {identity.headquarters.value}")
    if identity.website_subject.value:
        known_lines.append(f"- the website represents: {identity.website_subject.value}")
    known = "\n".join(known_lines) or "- nothing extracted from the website"
    groups = groups or prompts.research_groups()
    thin = bool(meta.get("thin_site"))
    searches = round(llm.max_search_uses * THIN_SITE_SEARCH_FACTOR) if thin else llm.max_search_uses
    rounds = followup_rounds + 1 if thin else followup_rounds

    progress = store.load_json(RESEARCH_PROGRESS) if store.exists(RESEARCH_PROGRESS) else {
        "done": [], "failed": [], "findings": [], "not_found": [], "rejected": [], "stopped": False}
    progress.setdefault("stopped", False)
    if progress["done"]:
        log.info("resuming research: %s already done", ", ".join(progress["done"]))
    system = prompts.localized(prompts.RESEARCH_SYSTEM, store.lang())
    total = len(groups) + rounds
    log.info("research plan: %d topic groups (%s), then up to %d follow-up rounds; at most %d calls",
             len(groups), ", ".join(groups), rounds, total)
    with metered(store, llm, "research"):
        for i, (group, topics) in enumerate(groups.items(), 1):
            if group in progress["done"]:
                continue
            user = prompts.research_user_prompt(
                company, meta["start"], topics, known, store.site_lang(), searches,
                other_names=other_names, thin_site=thin)
            label = f"{i}/{total} {group} ({', '.join(topics)})"
            _research_call(store, llm, progress, group, system, user, ledger, site_host, searches, label)
        if not any(g in progress["done"] for g in groups):
            raise StageError("research failed for every topic group; nothing to analyze (re-run the research stage)")
        for r in range(1, rounds + 1):
            name = f"followup-{r}"
            if progress["stopped"]:
                break
            if name in progress["done"]:
                continue
            user = prompts.followup_user_prompt(
                company, meta["start"], progress["findings"], progress["not_found"], known, store.site_lang(),
                searches, other_names=other_names)
            label = f"{len(groups) + r}/{total} {name} (following leads, gaps and contradictions)"
            new = _research_call(store, llm, progress, name, system, user, ledger, site_host, searches, label)
            if new is not None and new < MIN_NEW_FINDINGS:
                progress["stopped"] = True
                store.save_json(RESEARCH_PROGRESS, progress)
                log.info("research stopped after %s: only %d new findings (diminishing returns); "
                         "skipping the remaining follow-up rounds", name, new)
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
    render_pdf(md, store.path("report.pdf"), lang=store.lang())
    return md


# (name, what it does) in run order; run_all numbers them so the log shows where the run is.
STAGES: list[tuple[str, str]] = [
    ("crawl", "crawling the website"),
    ("identify", "identifying the company from the website"),
    ("signals", "extracting offerings and customer segments"),
    ("research", "researching beyond the website (web search)"),
    ("resolve", "refining the identity with external evidence"),
    ("analyze", "analyzing"),
    ("narrate", "writing the narrative"),
    ("report", "rendering the report"),
]


def _elapsed(start: float) -> str:
    seconds = round(time.monotonic() - start)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


@contextmanager
def announced(store: RunStore, name: str) -> Iterator[None]:
    """Log the start and end of one stage of ``run_all``: its position, elapsed time, cost so far."""
    index = next(i for i, (n, _) in enumerate(STAGES, 1) if n == name)
    tag = f"[{index}/{len(STAGES)}] {name}"
    log.info("%s: %s ...", tag, dict(STAGES)[name])
    start = time.monotonic()
    try:
        yield
    except BaseException:
        log.error("%s: failed after %s", tag, _elapsed(start))
        raise
    cost = store.load_json("usage.json").get("total", {}).get("estimated_cost_usd") \
        if store.exists("usage.json") else None
    later = [n for n, _ in STAGES[index:]]
    log.info("%s: done in %s%s; %s", tag, _elapsed(start),
             f", run cost so far ~${cost:.2f}" if cost is not None else "",
             f"next: {', '.join(later)}" if later else "all stages done")


def run_all(
    store: RunStore, url: str, crawler: Crawler, llm: LLM, lang: str | None = None,
    followup_rounds: int = FOLLOWUP_ROUNDS,
) -> str:
    steps = {
        "crawl": lambda: stage_crawl(store, url, crawler, lang),
        "identify": lambda: stage_identify(store, llm),
        "signals": lambda: stage_signals(store, llm),
        "research": lambda: stage_research(store, llm, followup_rounds=followup_rounds),
        "resolve": lambda: stage_resolve(store, llm),
        "analyze": lambda: stage_analyze(store, llm),
        "narrate": lambda: stage_narrate(store, llm),
        "report": lambda: stage_report(store),
    }
    md = ""
    for name, _ in STAGES:
        with announced(store, name):
            md = steps[name]()
    return md
