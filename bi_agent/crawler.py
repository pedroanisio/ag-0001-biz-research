"""Bounded, robots-respecting crawler for one company website.

Pages are prioritised by path keywords that signal business content (about, pricing,
products, careers, investors ...) in English, Portuguese, French, German and Spanish. Every loop is bounded by ``max_pages`` and
``max_fetches``; the crawler never follows off-site links.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib import robotparser
from urllib.parse import unquote, urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

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
    (3, ("legal", "juridico", "mentions-legales", "impressum", "aviso-legal")),
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
