from __future__ import annotations

import httpx
import pytest

from bi_agent.crawler import Crawler, canonical, extract, parse_sitemap, same_site, score_path
from bi_agent.errors import CrawlError
from tests.conftest import PAGES, SITE, site_handler


def test_canonical_strips_tracking_fragment_and_slash():
    assert canonical("HTTPS://WWW.Acme.test/About/?utm_source=x&b=1#frag") == "https://www.acme.test/About/?utm_source=x&b=1"
    assert canonical("https://acme.test") == "https://acme.test/"
    assert canonical("https://acme.test:8443/x/") == "https://acme.test:8443/x/"


def test_same_site_accepts_subdomains_only():
    assert same_site("https://docs.acme.test/x", "acme.test")
    assert same_site("https://www.acme.test/x", "acme.test")
    assert not same_site("https://acme.test.evil.com/x", "acme.test")
    assert not same_site("https://other.test/x", "acme.test")


def test_score_prefers_business_pages_and_shallow_depth():
    assert score_path("https://a.test/pricing", 1) > score_path("https://a.test/blog/2020/post", 3)
    assert score_path("https://a.test/", 0) > score_path("https://a.test/privacy", 1)
    # a keyword-rich news slug must not outrank a real section page
    assert score_path("https://a.test/company/leadership", 1) > score_path(
        "https://a.test/news/enterprise-ai-services-partner-company", 1)
    assert score_path("https://a.test/news", 1) > score_path("https://a.test/news/some-post", 1)


def test_extract_pulls_metadata_links_and_json_ld():
    page = extract(SITE + "/", PAGES["/"])
    assert page.title == "Acme Widgets"
    assert page.description.startswith("Acme sells")
    assert page.lang == "en"
    assert page.json_ld[0]["name"] == "Acme Widgets Inc"
    assert "var x" not in page.text and "body{}" not in page.text
    assert "monitoring software for factories" in page.text
    assert SITE + "/about" in page.links
    assert "https://other.test/x" in page.links  # extraction keeps links; the crawler filters them
    assert not any(l.startswith("mailto") or "#top" in l for l in page.links)
    assert page.links.count(SITE + "/about") == 1  # utm duplicate collapsed


def test_extract_tolerates_bad_json_ld_and_missing_head():
    page = extract("https://a.test/", '<html><body><script type="application/ld+json">{bad</script>hi</body></html>')
    assert page.json_ld == [] and page.title == "" and page.text == "hi"
    page = extract("https://a.test/", '<html><body><script type="application/ld+json">[{"a":1}, 3]</script></body></html>')
    assert page.json_ld == [{"a": 1}]


def test_parse_sitemap_handles_index_and_garbage():
    pages, nested = parse_sitemap(PAGES["/sitemap.xml"])
    assert SITE + "/from-sitemap" in pages and nested == []
    pages, nested = parse_sitemap(
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><sitemap><loc>https://a/s1.xml</loc></sitemap></sitemapindex>'
    )
    assert pages == [] and nested == ["https://a/s1.xml"]
    assert parse_sitemap("<<not xml") == ([], [])


def test_crawl_respects_robots_sitemap_domain_and_extensions(http_client):
    pages = Crawler(http_client, max_pages=20).crawl("www.acme-widgets.test")
    urls = {p.url for p in pages}
    assert SITE + "/" in urls
    assert SITE + "/from-sitemap" in urls  # discovered only through sitemap
    assert SITE + "/private/secret" not in urls  # robots disallow
    assert not any("other.test" in u for u in urls)  # off-site filtered
    assert not any(u.endswith((".pdf", ".png")) for u in urls)
    assert pages[0].url == SITE + "/"
    assert all(p.status == 200 for p in pages)


def test_crawl_orders_by_priority_and_bounds_pages(http_client):
    pages = Crawler(http_client, max_pages=3).crawl(SITE)
    assert len(pages) == 3
    assert pages[0].url == SITE + "/"
    # pricing and product/about outrank careers/sitemap-only page given the priority terms
    assert {pages[1].url, pages[2].url} <= {SITE + "/pricing", SITE + "/products/monitor", SITE + "/about"}


def test_crawl_max_fetches_bounds_loop(http_client):
    pages = Crawler(http_client, max_pages=50, max_fetches=2).crawl(SITE)
    assert len(pages) <= 2


def test_crawl_delay_uses_injected_sleep(http_client):
    slept: list[float] = []
    Crawler(http_client, max_pages=3, delay_seconds=0.25, sleep=slept.append).crawl(SITE)
    assert slept and all(0 <= s <= 0.25 for s in slept)


def test_crawl_unreachable_start_raises():
    client = httpx.Client(transport=httpx.MockTransport(site_handler))
    with pytest.raises(CrawlError):
        Crawler(client).crawl("https://www.acme-widgets.test/missing")
    with pytest.raises(CrawlError):
        Crawler(client).crawl("https://www.acme-widgets.test/boom")


def test_crawl_skips_network_errors_non_html_and_oversized(http_client):
    html = ('<html><body><a href="/boom">b</a><a href="/big">big</a><a href="/image.png">i</a>'
            '<a href="/redirect">r</a><a href="/about">a</a><a href="/leave">l</a></body></html>')

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text=html, headers={"content-type": "text/html"})
        return site_handler(request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    pages = Crawler(client, max_pages=10).crawl(SITE)
    urls = {p.url for p in pages}
    assert SITE + "/about" in urls  # followed the redirect and recorded the final URL
    assert [p.url for p in pages].count(SITE + "/about") == 1  # /redirect and /about are one page
    assert not any("other.test" in u for u in urls)  # a redirect that leaves the site is dropped
    assert SITE + "/big" not in urls and SITE + "/boom" not in urls


def test_crawl_without_robots_or_sitemap():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text="<html><title>t</title><body>hello</body></html>",
                                  headers={"content-type": "text/html"})
        return httpx.Response(404)

    pages = Crawler(httpx.Client(transport=httpx.MockTransport(handler)), max_pages=5).crawl("https://solo.test")
    assert len(pages) == 1 and pages[0].text == "hello"


def test_crawler_rejects_zero_pages(http_client):
    with pytest.raises(ValueError):
        Crawler(http_client, max_pages=0)


def test_page_round_trip(http_client):
    page = Crawler(http_client, max_pages=1).crawl(SITE)[0]
    from bi_agent.crawler import Page

    assert Page.from_dict(page.to_dict()) == page


def test_extract_flags_javascript_app_shells():
    shell = extract("https://a.test/", "<html><body><div id='root'></div><script src='a.js'></script></body></html>")
    assert shell.js_rendered
    many_scripts = extract("https://a.test/", "<html><body>Hi<script>1</script><script>2</script><script>3</script></body></html>")
    assert many_scripts.js_rendered
    rendered = extract("https://a.test/", "<html><body><div id='root'>" + "Real content. " * 40 + "</div><script></script></body></html>")
    assert not rendered.js_rendered



def test_extract_keeps_menu_links_but_drops_menu_text():
    page = extract("https://a.test/", "<html><body><nav><a href='/sobre'>Quem somos</a></nav>"
                                      "<div role='navigation'><a href='/precos'>Preços</a></div><p>Gás natural.</p></body></html>")
    assert "https://a.test/sobre" in page.links and "https://a.test/precos" in page.links
    assert "Quem somos" not in page.text and "Preços" not in page.text and "Gás natural." in page.text


def test_strip_boilerplate_keeps_the_home_copy_and_long_lines():
    from bi_agent.crawler import Page, strip_boilerplate

    footer = "PBGÁS · CNPJ 00.000.000/0001-00 · 0800 281 0197"
    quote = "x" * 250
    pages = [Page(url=f"https://a.test/{i}", status=200, title="", description="",
                  text=f"{footer}\nContent of page {i}\n{quote}") for i in range(5)]
    strip_boilerplate(pages)
    assert footer in pages[0].text  # the home page keeps it once
    assert all(footer not in p.text and f"Content of page {i}" in p.text for i, p in enumerate(pages) if i)
    assert all(quote in p.text for p in pages)  # long repeated text is content, not chrome
    few = [Page(url="u", status=200, title="", description="", text=footer) for _ in range(3)]
    assert all(p.text == footer for p in strip_boilerplate(few))
