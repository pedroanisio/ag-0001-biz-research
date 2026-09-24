"""Bounded, robots-respecting crawler for one company website.

Pages are prioritised by path keywords that signal business content (about, pricing,
products, careers, investors ...) in English, Portuguese, French, German and Spanish. Every loop is bounded by ``max_pages`` and
``max_fetches``; the crawler never follows off-site links.
"""

from __future__ import annotations

import json
import ipaddress
import socket
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib import robotparser
from urllib.parse import unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .urls import normalize_url, origin
from .errors import CrawlError
from .i18n import guess_text_lang, normalize_lang

# Business-content concepts and the path words that signal them in en / pt-br / fr / de / es.
# Paths are lower-cased, percent-decoded and stripped of accents before matching, so "preços",
# "pre%C3%A7os" and "precos" all hit. A concept counts once per path segment however many of its
# words match. Words of three letters or fewer must equal a whole hyphen/underscore token, so "ri"
# (relações com investidores) matches "/ri" but not "/pricing".
PRIORITY_CONCEPTS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (10, ("about", "sobre", "quem-somos", "a-propos", "apropos", "qui-sommes", "ueber-uns", "uber-uns",
          "acerca", "quienes-somos", "nosotros")),
    (8, ("company", "empresa", "entreprise", "societe", "unternehmen", "institucional", "institutionnel",
         "who-we-are", "our-story", "nossa-historia", "historia", "histoire", "geschichte")),
    (6, ("mission", "missao", "mision", "leitbild", "valores", "valeurs", "werte")),
    (10, ("product", "produto", "produit", "produkt", "producto")),
    (9, ("solution", "solucao", "solucoes", "solucion", "loesung", "losung")),
    (9, ("service", "servico", "servicio", "leistung")),
    (8, ("platform", "plataforma", "plateforme", "plattform")),
    (6, ("feature", "funcionalidade", "funcionalidad", "fonctionnalite", "funktion", "recursos")),
    (7, ("industr", "setor", "sector", "secteur", "branche")),
    (8, ("customer", "client", "kunde", "referenc", "referenz")),
    (8, ("case-stud", "case_stud", "casos", "cas-client", "etude-de-cas", "etudes-de-cas", "fallstudie")),
    (5, ("success", "sucesso", "exito", "erfolg")),
    (5, ("testimonial", "depoimento", "testimonio", "temoignage", "kundenstimme")),
    (10, ("pricing", "price", "preco", "precio", "prix", "tarif", "preise")),
    (8, ("plans", "planos", "planes", "offres", "abonnement")),
    (8, ("portfolio", "portafolio", "portefeuille", "investimentos", "investments", "inversiones")),
    (7, ("partner", "parceir", "partenaire", "alianza", "aliados", "socios")),
    (7, ("integration", "integrac", "integracion")),
    (6, ("marketplace",)),
    (6, ("developer", "desenvolvedor", "desarrollador", "developpeur", "entwickler")),
    (6, ("api",)),
    (5, ("docs", "documentation", "documentacao", "documentacion", "dokumentation")),
    (3, ("resource", "ressource", "materiais", "materiales")),
    (2, ("blog",)),
    (4, ("news", "noticia", "novidade", "actualite", "actualidad", "aktuell", "nachricht", "neuigkeit")),
    (5, ("press", "imprensa", "prensa", "presse")),
    (3, ("media", "midia", "medios", "medien")),
    (7, ("career", "carreira", "carrera", "carriere", "karriere", "jobs", "vagas", "trabalhe", "empleo",
         "emploi", "recrutement", "stellen")),
    (4, ("join",)),
    (9, ("investor", "investidor", "inversor", "inversionista", "investisseur")),
    (4, ("ir", "ri")),
    (5, ("security", "seguranca", "seguridad", "securite", "sicherheit")),
    (5, ("compliance", "conformidade", "conformite", "cumplimiento")),
    (5, ("trust", "confianca", "confianza", "confiance", "vertrauen")),
    (3, ("privacy", "privacidade", "privacidad", "confidentialite", "datenschutz", "lgpd", "rgpd", "dsgvo", "gdpr")),
    (3, ("terms", "termos", "terminos", "conditions", "cgu", "cgv", "agb", "nutzungsbedingungen")),
    # legal notices name the legal entity (Impressum and mentions légales are mandatory in DE/FR)
    (6, ("legal", "juridico", "mentions-legales", "impressum", "aviso-legal", "imprint")),
    (4, ("contact", "contato", "contacto", "kontakt", "fale-conosco")),
    (6, ("team", "equipe", "equipo")),
    (8, ("leadership", "lideranca", "liderazgo", "fuehrung", "direction")),
    (6, ("management", "gestao", "gestion", "diretoria", "directorio", "vorstand", "geschaeftsfuehrung")),
    (4, ("faq", "perguntas-frequentes", "preguntas-frecuentes", "duvidas", "haeufige-fragen")),
    (4, ("why", "por-que", "porque", "pourquoi", "warum")),
    (6, ("compare", "comparar", "comparatif", "comparacao", "comparacion", "vergleich")),
    (5, ("vs",)),
    (6, ("enterprise", "corporativo", "grandes-empresas")),
)
FEED_SECTIONS = frozenset((
    "news", "blog", "press", "resources", "media", "articles", "posts", "events",
    "noticias", "novidades", "imprensa", "artigos", "eventos", "materiais", "recursos", "midia",
    "actualites", "presse", "evenements", "ressources",
    "aktuelles", "nachrichten", "beitraege", "veranstaltungen", "medien", "artikel",
    "actualidad", "prensa", "articulos", "medios",
))
# First path segments that select a language version of the site ("/en/", "/pt-br/", "/de_DE/").
LOCALE_SEGMENT = re.compile(
    r"^(en|pt|fr|de|es|it|nl|ja|zh|ko|ru|pl|sv|da|no|nb|fi|tr|ar|he|cs)([-_][a-z]{2,4})?$"
)
SKIP_EXTENSIONS = (
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".css", ".js",
    ".zip", ".mp4", ".mp3", ".woff", ".woff2", ".ttf", ".xml", ".json", ".rss", ".atom",
)
JS_SHELL_MAX_TEXT = 300  # below this much server-rendered text, a script-heavy page is an app shell
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
    requested_url: str = ""
    redirect_chain: list[str] = field(default_factory=list)
    js_rendered: bool = False  # the HTML is an app shell whose content only appears after JavaScript runs

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "Page":
        return cls(**d)


def canonical(url: str) -> str:
    return normalize_url(url)


def registrable_host(host: str) -> str:
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def same_site(url: str, root_host: str) -> bool:
    h = registrable_host(urlparse(url).hostname or "")
    return h == root_host or h.endswith("." + root_host)


def _normalize_segment(seg: str) -> str:
    seg = unquote(seg).lower()
    seg = unicodedata.normalize("NFKD", seg)
    return "".join(ch for ch in seg if not unicodedata.combining(ch))


def _segment_score(seg: str) -> int:
    tokens = set(re.split(r"[-_.]+", seg))
    total = 0
    for weight, words in PRIORITY_CONCEPTS:
        if any((w in tokens) if len(w) <= 3 else (w in seg) for w in words):
            total += weight
    return total


def locale_of(url: str) -> str | None:
    """The language subtag of a locale prefix such as ``/pt-br/``, or None."""
    segments = [s for s in urlparse(url).path.lower().split("/") if s]
    if segments and (m := LOCALE_SEGMENT.match(segments[0])):
        return m.group(1)
    return None


def score_path(url: str, depth: int, site_lang: str | None = None) -> int:
    """Higher is fetched first. Section pages outrank individual posts under feed sections.

    A leading locale segment is ignored for keyword scoring; when ``site_lang`` is known, pages
    under a different locale prefix (another language version of the same site) are pushed back.
    """
    segments = [_normalize_segment(s) for s in urlparse(url).path.split("/") if s]
    penalty = 0
    if segments and (m := LOCALE_SEGMENT.match(segments[0])):
        if site_lang and m.group(1) != site_lang:
            penalty = 12
        segments = segments[1:]
    score = 10 if not segments else 0
    for i, seg in enumerate(segments):
        matched = _segment_score(seg)
        score += matched if i == 0 else matched // 3  # a keyword in a deep slug is weak evidence
    if len(segments) > 1 and segments[0] in FEED_SECTIONS:
        score -= 8
    score -= 2 * len(segments)  # shallower pages first
    score -= 3 * depth
    return score - penalty


def extract(url: str, html: str, status: int = 200) -> Page:
    """Pure HTML → Page extraction (no network). Text is capped at 30k chars."""
    soup = BeautifulSoup(html, "html.parser")
    scripts = len([s for s in soup.find_all("script") if s.get("type") != "application/ld+json"])
    app_root = soup.find(id=re.compile(r"^(root|app|__next|__nuxt|___gatsby|svelte)$")) is not None
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
        try:
            absolute = canonical(urljoin(url, href))
        except ValueError:
            continue
        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    for tag in soup.find_all("nav") + soup.find_all(attrs={"role": "navigation"}):
        tag.decompose()  # after link extraction: menus lead to pages but say nothing about the business
    text = re.sub(r"[ \t\r\f\v]+", " ", soup.get_text("\n"))
    text = re.sub(r"\n\s*\n+", "\n", text).strip()[:30_000]
    js_rendered = len(text) < JS_SHELL_MAX_TEXT and (scripts >= 3 or app_root)
    return Page(
        url=canonical(url), status=status, title=title[:300], description=desc[:1000],
        text=text, links=links, json_ld=json_ld[:10], lang=str(lang) if lang else None,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), js_rendered=js_rendered,
    )


BOILERPLATE_SHARE = 0.5  # a line on at least half the pages is site chrome (menus, footers, banners)
BOILERPLATE_MIN_PAGES = 4


def strip_boilerplate(pages: list[Page]) -> list[Page]:
    """Remove lines that repeat across most pages (menus outside <nav>, footers, cookie banners).

    The home page keeps its copy, so footer facts such as the legal entity or registration number
    are still read once. Only short lines count as boilerplate; a long paragraph quoted on many pages
    is content. Sites with fewer than BOILERPLATE_MIN_PAGES pages are left alone.
    """
    if len(pages) < BOILERPLATE_MIN_PAGES:
        return pages
    seen: dict[str, int] = {}
    for p in pages:
        for line in {ln.strip() for ln in p.text.splitlines() if ln.strip()}:
            seen[line] = seen.get(line, 0) + 1
    threshold = max(BOILERPLATE_MIN_PAGES, BOILERPLATE_SHARE * len(pages))
    chrome = {ln for ln, n in seen.items() if n >= threshold and len(ln) <= 200}
    for p in pages[1:]:
        p.text = "\n".join(ln for ln in p.text.splitlines() if ln.strip() not in chrome)
    return pages


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
        allow_private: bool = False,
        resolver=None,
        max_bytes: int = 3_000_000,
    ) -> None:
        if max_pages < 1:
            raise ValueError("max_pages must be >= 1")
        self.client = client
        self.max_pages = max_pages
        if max_fetches is not None and max_fetches < 1:
            raise ValueError("max_fetches must be >= 1")
        if max_bytes < 1 or delay_seconds < 0:
            raise ValueError("max_bytes must be positive and delay_seconds nonnegative")
        self.max_fetches = max_fetches if max_fetches is not None else max(10, max_pages * 3)
        self.delay = delay_seconds
        self.ua = user_agent
        self._sleep = sleep
        self.allow_private = allow_private
        self.resolver = resolver or socket.getaddrinfo
        self.max_bytes = max_bytes
        self._requests = 0
        self._policies: dict[str, robotparser.RobotFileParser] = {}
        self._root_host = ""
        self._start_origin = ""
        self._last_request: dict[str, float] = {}

    # ------------------------------------------------------------------ fetching
    def _permitted(self, url: str) -> bool:
        try:
            url = canonical(url)
            p = urlparse(url)
            if self._root_host and not same_site(url, self._root_host):
                return False
            # Subdomains may use standard HTTP(S) ports; a custom port is confined to the
            # explicitly supplied origin. Redirects cannot silently open another service.
            if p.port not in (None, 80 if p.scheme == "http" else 443) and origin(url) != self._start_origin:
                return False
            if not self.allow_private:
                addresses = self.resolver(p.hostname, p.port or (443 if p.scheme == "https" else 80), type=socket.SOCK_STREAM)
                if not addresses or any(not ipaddress.ip_address(x[4][0]).is_global for x in addresses):
                    return False
            return True
        except (ValueError, OSError):
            return False

    def _get(self, url: str, *, robots: bool = False, initial: bool = False) -> httpx.Response | None:
        requested = url
        chain: list[str] = []
        for _hop in range(6):
            if not self._permitted(url) or self._requests >= self.max_fetches:
                return None
            url = canonical(url)
            site = origin(url)
            if not robots:
                rp = self._robots(site)
                if not rp.can_fetch(self.ua, url) or self._requests >= self.max_fetches:
                    return None
            delay = self.delay
            rp = self._policies.get(site)
            if rp is not None:
                delay = max(delay, rp.crawl_delay(self.ua) or 0)
            if site in self._last_request and delay:
                self._sleep(max(0, delay - (time.monotonic() - self._last_request[site])))
            self._requests += 1
            self._last_request[site] = time.monotonic()
            try:
                with self.client.stream("GET", url, headers={"User-Agent": self.ua},
                                        follow_redirects=False, timeout=20.0) as resp:
                    if resp.status_code in (301, 302, 303, 307, 308):
                        target = canonical(urljoin(url, resp.headers.get("location", "")))
                        # Robots redirects are restricted to the same origin. Pages can
                        # canonicalize only inside the explicitly selected site boundary.
                        if initial and (urlparse(target).hostname or "") not in {
                                self._root_host, "www." + self._root_host, urlparse(requested).hostname}:
                            return None
                        if robots and origin(target) != site:
                            return None
                        if urlparse(url).scheme == "https" and urlparse(target).scheme != "https":
                            return None
                        chain.append(url)
                        url = target
                        continue
                    body = bytearray()
                    for chunk in resp.iter_bytes(chunk_size=16_384):
                        if len(body) + len(chunk) > self.max_bytes:
                            return None
                        body.extend(chunk)
                    headers = dict(resp.headers)
                    headers.pop("content-encoding", None)  # iter_bytes already decoded it
                    headers.pop("content-length", None)
                    result = httpx.Response(resp.status_code, headers=headers, content=bytes(body), request=resp.request)
                    result.extensions.update(requested_url=requested, redirect_chain=chain)
                    return result
            except (httpx.HTTPError, ValueError):
                return None
        return None

    def _robots(self, root: str) -> robotparser.RobotFileParser:
        site = origin(root)
        if site not in self._policies:
            rp = robotparser.RobotFileParser()
            resp = self._get(site + "/robots.txt", robots=True)
            if resp is not None and resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            elif resp is not None and resp.status_code in (404, 410):
                rp.parse([])
            else:
                rp.parse(["User-agent: *", "Disallow: /"])  # unavailable policy: fail closed
            self._policies[site] = rp
        return self._policies[site]

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
        if "://" not in start_url:
            start_url = "https://" + start_url
        try:
            start = canonical(start_url)
        except ValueError as exc:
            raise CrawlError(str(exc)) from exc
        self._requests = 0
        self._policies.clear()
        self._last_request.clear()
        self._root_host = registrable_host(urlparse(start).hostname or "")
        self._start_origin = origin(start)
        first = self._get(start, initial=True)
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
        site_lang: str | None = locale_of(start)  # primary subtag of the version we landed on

        def push(url: str, depth: int) -> None:
            if url in seen or not same_site(url, root_host):
                return
            if urlparse(url).path.lower().endswith(SKIP_EXTENSIONS):
                return
            seen.add(url)
            frontier.append((score_path(url, depth, site_lang), depth, url))

        home = self._accept(first, start, depth=0, root_host=root_host, recorded=recorded)
        if home is not None:
            pages.append(home)
            if site_lang is None:
                lang = guess_text_lang(home.text) or normalize_lang(home.lang)
                site_lang = lang.split("-")[0] if lang else None
            for link in home.links:
                push(link, 1)
        for sm_url in self._sitemap_urls(root, rp):
            try:
                push(canonical(sm_url), 1)
            except ValueError:
                continue

        while frontier and len(pages) < self.max_pages and self._requests < self.max_fetches:
            frontier.sort(key=lambda t: (-t[0], t[1]))
            score, depth, url = frontier.pop(0)
            resp = self._get(url)
            if resp is None:
                continue
            page = self._accept(resp, url, depth=depth, root_host=root_host, recorded=recorded)
            if page is None:
                continue
            page.score = score
            pages.append(page)
            for link in page.links:
                push(link, depth + 1)
        return strip_boilerplate(pages)

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
        page.requested_url = resp.extensions.get("requested_url", url)
        page.redirect_chain = resp.extensions.get("redirect_chain", [])
        page.score = score_path(page.url, depth)
        return page
