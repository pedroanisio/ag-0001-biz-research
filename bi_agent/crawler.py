"""Bounded, robots-respecting crawler for one company website.

Pages are prioritised by path keywords that signal business content (about, pricing,
products, careers, investors ...). Every loop is bounded by ``max_pages`` and
``max_fetches``; the crawler never follows off-site links.
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib import robotparser
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from .errors import CrawlError

PRIORITY_TERMS: dict[str, int] = {
    "about": 10, "company": 8, "who-we-are": 8, "our-story": 8, "mission": 6,
    "product": 10, "solution": 9, "service": 9, "platform": 8, "feature": 6,
    "industr": 7, "customer": 8, "case-stud": 8, "case_stud": 8, "success": 5, "testimonial": 5,
    "pricing": 10, "plans": 8, "partner": 7, "integration": 7, "marketplace": 6,
    "developer": 6, "api": 6, "docs": 5, "documentation": 5, "resource": 3,
    "blog": 2, "news": 4, "press": 5, "media": 3, "career": 7, "jobs": 7, "join": 4,
    "investor": 9, "ir": 4, "security": 5, "compliance": 5, "trust": 5, "privacy": 3,
    "terms": 3, "legal": 3, "contact": 4, "team": 6, "leadership": 8, "management": 6,
    "faq": 4, "why": 4, "compare": 6, "vs": 5, "enterprise": 6,
}
SKIP_EXTENSIONS = (
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".css", ".js",
    ".zip", ".mp4", ".mp3", ".woff", ".woff2", ".ttf", ".xml", ".json", ".rss", ".atom",
)
TRACKING_PARAMS = re.compile(r"^(utm_|fbclid|gclid|mc_|ref$)")
DEFAULT_UA = "bi-agent/1.0 (+business research crawler; respects robots.txt)"


@dataclass
class Page:
    url: str
    status: int
    title: str
    description: str
    text: str
    links: list[str] = field(default_factory=list)
    json_ld: list[dict] = field(default_factory=list)
    lang: str | None = None
    fetched_at: str = ""
    score: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "Page":
        return cls(**d)


def canonical(url: str) -> str:
    """Lower-case host, strip fragment, tracking params, default ports and trailing slash."""
    p = urlparse(url.strip())
    host = (p.hostname or "").lower()
    if p.port and p.port not in (80, 443):
        host = f"{host}:{p.port}"
    query = "&".join(
        kv for kv in p.query.split("&") if kv and not TRACKING_PARAMS.match(kv.split("=", 1)[0])
    )
    path = re.sub(r"/+$", "", p.path) or "/"
    return urlunparse((p.scheme.lower() or "https", host, path, "", query, ""))


def registrable_host(host: str) -> str:
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def same_site(url: str, root_host: str) -> bool:
    h = registrable_host(urlparse(url).hostname or "")
    return h == root_host or h.endswith("." + root_host)


FEED_SECTIONS = ("news", "blog", "press", "resources", "media", "articles", "posts", "events")


def score_path(url: str, depth: int) -> int:
    """Higher is fetched first. Section pages outrank individual posts under feed sections."""
    segments = [s for s in urlparse(url).path.lower().split("/") if s]
    score = 10 if not segments else 0
    for i, seg in enumerate(segments):
        matched = sum(w for term, w in PRIORITY_TERMS.items() if term in seg)
        score += matched if i == 0 else matched // 3  # a keyword in a deep slug is weak evidence
    if len(segments) > 1 and segments[0] in FEED_SECTIONS:
        score -= 8
    score -= 2 * len(segments)  # shallower pages first
    score -= 3 * depth
    return score


def extract(url: str, html: str, status: int = 200) -> Page:
    """Pure HTML → Page extraction (no network). Text is capped at 30k chars."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "template"]):
        if tag.name == "script" and tag.get("type") == "application/ld+json":
            continue
        tag.decompose()
    json_ld: list[dict] = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            parsed = json.loads(tag.string or "")
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            json_ld.append(parsed)
        elif isinstance(parsed, list):
            json_ld.extend(x for x in parsed if isinstance(x, dict))
        tag.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    if soup.title:
        soup.title.decompose()  # the title is captured separately; keep it out of the body text
    desc = ""
    for attrs in ({"name": "description"}, {"property": "og:description"}):
        m = soup.find("meta", attrs=attrs)
        if m and m.get("content"):
            desc = str(m["content"]).strip()
            break
    lang = soup.html.get("lang") if soup.html and soup.html.get("lang") else None
    links: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = str(a["href"]).strip()
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        absolute = canonical(urljoin(url, href))
        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    text = re.sub(r"[ \t\r\f\v]+", " ", soup.get_text("\n"))
    text = re.sub(r"\n\s*\n+", "\n", text).strip()[:30_000]
    return Page(
        url=canonical(url), status=status, title=title[:300], description=desc[:1000],
        text=text, links=links, json_ld=json_ld[:10], lang=str(lang) if lang else None,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def parse_sitemap(xml_text: str) -> tuple[list[str], list[str]]:
    """Return (page urls, nested sitemap urls). Tolerates malformed XML by returning empty lists."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return [], []
    pages, nested = [], []
    is_index = root.tag.rsplit("}", 1)[-1] == "sitemapindex"
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag != "loc" or not el.text:
            continue
        (nested if is_index else pages).append(el.text.strip())
    return pages, nested


class Crawler:
    def __init__(
        self,
        client: httpx.Client,
        *,
        max_pages: int = 60,
        max_fetches: int | None = None,
        delay_seconds: float = 0.0,
        user_agent: str = DEFAULT_UA,
        sleep=time.sleep,
    ) -> None:
        if max_pages < 1:
            raise ValueError("max_pages must be >= 1")
        self.client = client
        self.max_pages = max_pages
        self.max_fetches = max_fetches or max_pages * 3
        self.delay = delay_seconds
        self.ua = user_agent
        self._sleep = sleep

    # ------------------------------------------------------------------ fetching
    def _get(self, url: str) -> httpx.Response | None:
        try:
            return self.client.get(
                url, headers={"User-Agent": self.ua, "Accept": "text/html,application/xhtml+xml,*/*;q=0.5"},
                follow_redirects=True, timeout=20.0,
            )
        except httpx.HTTPError:
            return None

    def _robots(self, root: str) -> robotparser.RobotFileParser:
        rp = robotparser.RobotFileParser()
        resp = self._get(urljoin(root, "/robots.txt"))
        if resp is not None and resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:
            rp.parse([])  # no robots → everything allowed
        return rp

    def _sitemap_urls(self, root: str, rp: robotparser.RobotFileParser) -> list[str]:
        candidates = list(rp.site_maps() or []) or [urljoin(root, "/sitemap.xml")]
        found: list[str] = []
        visited = 0
        while candidates and visited < 5:
            sm = candidates.pop(0)
            visited += 1
            resp = self._get(sm)
            if resp is None or resp.status_code != 200:
                continue
            pages, nested = parse_sitemap(resp.text)
            found.extend(pages)
            candidates.extend(nested)
        return found

    # ------------------------------------------------------------------ crawl loop
    def crawl(self, start_url: str) -> list[Page]:
        if not start_url.startswith(("http://", "https://")):
            start_url = "https://" + start_url
        start = canonical(start_url)
        first = self._get(start)
        if first is None or first.status_code >= 400:
            raise CrawlError(f"start URL {start} unreachable (status={getattr(first, 'status_code', None)})")
        start = canonical(str(first.url))
        parsed = urlparse(start)
        root = f"{parsed.scheme}://{parsed.netloc}/"
        root_host = registrable_host(parsed.hostname or "")
        rp = self._robots(root)

        pages: list[Page] = []
        seen: set[str] = {start}
        recorded: set[str] = set()  # final URLs after redirects, so one page is never stored twice
        frontier: list[tuple[int, int, str]] = []  # (score, depth, url)

        def push(url: str, depth: int) -> None:
            if url in seen or not same_site(url, root_host):
                return
            if urlparse(url).path.lower().endswith(SKIP_EXTENSIONS):
                return
            seen.add(url)
            frontier.append((score_path(url, depth), depth, url))

        home = self._accept(first, start, depth=0, root_host=root_host, recorded=recorded)
        if home is not None:
            pages.append(home)
            for link in home.links:
                push(link, 1)
        for sm_url in self._sitemap_urls(root, rp):
            push(canonical(sm_url), 1)

        fetches = 1
        while frontier and len(pages) < self.max_pages and fetches < self.max_fetches:
            frontier.sort(key=lambda t: (-t[0], t[1]))
            score, depth, url = frontier.pop(0)
            if not rp.can_fetch(self.ua, url):
                continue
            if self.delay:
                self._sleep(self.delay)
            resp = self._get(url)
            fetches += 1
            if resp is None:
                continue
            page = self._accept(resp, url, depth=depth, root_host=root_host, recorded=recorded)
            if page is None:
                continue
            page.score = score
            pages.append(page)
            for link in page.links:
                push(link, depth + 1)
        return pages

    def _accept(
        self, resp: httpx.Response, url: str, *, depth: int, root_host: str, recorded: set[str]
    ) -> Page | None:
        """Turn a response into a Page, or None when it is not an on-site HTML page we have not stored yet."""
        ctype = resp.headers.get("content-type", "")
        if resp.status_code != 200 or "html" not in ctype:
            return None
        if len(resp.content) > 3_000_000:
            return None
        final = canonical(str(resp.url)) if str(resp.url) else url
        if not same_site(final, root_host) or final in recorded:  # redirect went off-site or to a stored page
            return None
        recorded.add(final)
        page = extract(final, resp.text, resp.status_code)
        page.score = score_path(page.url, depth)
        return page
