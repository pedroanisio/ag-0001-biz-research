from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from bi_agent import cli, pipeline, prompts
from bi_agent.crawler import Crawler, Page, locale_of, score_path
from bi_agent.errors import StageError
from bi_agent.i18n import (
    MATURITY_DIMENSION_TEXT,
    STRATEGIC_QUESTION_TEXT,
    STRINGS,
    SUPPORTED,
    Translator,
    detect_site_lang,
    guess_text_lang,
    normalize_lang,
)
from bi_agent.llm import LLM
from bi_agent.models import MATURITY_DIMENSIONS, STRATEGIC_QUESTIONS
from tests.conftest import SITE, stage_router

SAMPLES = {
    "en": "We help our customers with the tools they need. Our platform is fast and you can start today. "
          "This is the best way for your team to work with data from all of our partners and to grow.",
    "pt-br": "A Esfera é uma gestora de investimentos com foco em tecnologia. Nossa equipe apoia empreendedores "
             "com capital e também com experiência. Você encontra aqui as empresas do nosso portfólio, pelo "
             "trabalho que fazemos ao lado dos fundadores, da ideia ao crescimento, e mais sobre a sua história.",
    "fr": "Nous aidons les entreprises avec une plateforme simple. Notre équipe est dans votre ville et nous "
          "sommes à votre écoute pour vous accompagner au quotidien. Les clients sont au cœur de notre projet "
          "et du service que nous proposons sur le marché, avec les partenaires et pour vous.",
    "de": "Wir helfen Unternehmen mit einer Plattform für die Zukunft. Unsere Kunden sind mit der Lösung "
          "zufrieden und die Daten sind sicher. Das ist nicht ein Produkt wie jedes andere, sie ist für den "
          "Mittelstand gebaut und auf die Bedürfnisse der Kunden zu geschnitten, mit Liebe und für Sie.",
    "es": "Ayudamos a las empresas con una plataforma sencilla. Nuestra misión es simple y también es clara: "
          "el cliente es lo más importante para nuestro equipo. Por eso sus datos están seguros con el servicio "
          "del que hablamos al principio, y su negocio crece más con nuestro apoyo y por nuestra red.",
}


# --------------------------------------------------------------------------- strings


def test_every_string_has_all_languages_with_matching_placeholders():
    fields = re.compile(r"\{(\w+)\}")
    for key, row in STRINGS.items():
        assert len(row) == len(SUPPORTED), key
        assert all(x.strip() for x in row), key
        assert {tuple(sorted(fields.findall(x))) for x in row} == {tuple(sorted(fields.findall(row[0])))}, key


def test_fixed_keys_are_translated_for_every_language():
    assert set(STRATEGIC_QUESTION_TEXT) == set(STRATEGIC_QUESTIONS)
    assert set(MATURITY_DIMENSION_TEXT) == set(MATURITY_DIMENSIONS)
    for row in list(STRATEGIC_QUESTION_TEXT.values()) + list(MATURITY_DIMENSION_TEXT.values()):
        assert len(row) == len(SUPPORTED) - 1 and all(row)


def test_translator_falls_back_and_formats():
    assert Translator("pt-BR")("s1") == "Sumário Executivo"
    assert Translator("xx").lang == "en"
    assert Translator("de")("missing-key") == "missing-key"
    assert Translator("fr")("research_gap", gap="x") == "Lacune de recherche : x"
    assert Translator("es").question(STRATEGIC_QUESTIONS[0]).startswith("¿")
    assert Translator("en").dimension("Brand maturity") == "Brand maturity"
    assert Translator("de").dimension("not a dimension") == "not a dimension"


# --------------------------------------------------------------------------- detection


@pytest.mark.parametrize(
    "tag,expected",
    [("pt-BR", "pt-br"), ("pt_PT", "pt-br"), ("en-US", "en"), ("DE", "de"), ("es-419", "es"),
     ("fr-CA", "fr"), ("it", None), ("", None), (None, None)],
)
def test_normalize_lang(tag, expected):
    assert normalize_lang(tag) == expected


@pytest.mark.parametrize("lang", SUPPORTED)
def test_guess_text_lang(lang):
    assert guess_text_lang(SAMPLES[lang] * 2) == lang


def test_guess_text_lang_abstains_on_thin_or_mixed_text():
    assert guess_text_lang("Esfera") is None
    assert guess_text_lang(SAMPLES["en"] + SAMPLES["pt-br"] + SAMPLES["de"]) is None


def test_site_lang_prefers_text_over_template_lang_attribute():
    pages = [Page(url="u", status=200, title="", description="", text=SAMPLES["pt-br"] * 2, lang="en-US")]
    assert detect_site_lang(pages) == "pt-br"


def test_site_lang_falls_back_to_html_lang_then_english():
    thin = [Page(url="u", status=200, title="", description="", text="Esfera", lang="de-DE"),
            Page(url="v", status=200, title="", description="", text="", lang="fr")]
    assert detect_site_lang(thin) == "de"  # home page outweighs one other page
    assert detect_site_lang([Page(url="u", status=200, title="", description="", text="")]) == "en"


# --------------------------------------------------------------------------- crawler scoring


@pytest.mark.parametrize(
    "business,feed",
    [("/sobre", "/blog/2021/post"), ("/quem-somos", "/noticias/uma-noticia"), ("/a-propos", "/actualites/x"),
     ("/ueber-uns", "/aktuelles/x"), ("/quienes-somos", "/noticias/y"), ("/precos", "/blog"),
     ("/carreiras", "/eventos/z"), ("/portfolio", "/imprensa/x")],
)
def test_business_pages_outrank_feeds_in_every_language(business, feed):
    assert score_path("https://a.test" + business, 1) > score_path("https://a.test" + feed, 1)


def test_accents_and_percent_encoding_match_the_plain_word():
    plain = score_path("https://a.test/precos", 1)
    assert score_path("https://a.test/pre%C3%A7os", 1) == plain
    assert score_path("https://a.test/preços", 1) == plain
    assert score_path("https://a.test/über-uns", 1) == score_path("https://a.test/uber-uns", 1)
    assert score_path("https://a.test/precos", 1) > score_path("https://a.test/xyz", 1)


def test_short_words_need_a_whole_token():
    assert score_path("https://a.test/ri", 1) > score_path("https://a.test/rz", 1)
    assert score_path("https://a.test/pricing", 1) == score_path("https://a.test/prices", 1)  # no "ri" bonus


def test_locale_prefix_is_ignored_and_other_locales_are_pushed_back():
    assert score_path("https://a.test/pt-br/sobre", 1) == score_path("https://a.test/sobre", 1)
    assert score_path("https://a.test/pt-br", 1, "pt") == score_path("https://a.test/", 1, "pt")
    assert score_path("https://a.test/en/about", 1, "pt") < score_path("https://a.test/pt/sobre", 1, "pt")
    assert locale_of("https://a.test/de_DE/x") == "de" and locale_of("https://a.test/sobre") is None


PT_SITE = "https://www.esfera.test"
PT_BODY = "<p>" + SAMPLES["pt-br"] + "</p>"
PT_PAGES = {
    "/": f"""<html lang="en-US"><head><title>Esfera</title></head><body>
        <a href="/pt-br/sobre">Sobre</a><a href="/en/about">About</a><a href="/pt-br/noticias/a">Notícia</a>
        <a href="/pt-br/pre%C3%A7os">Preços</a>{PT_BODY}</body></html>""",
    "/pt-br/sobre": f"<html lang='pt-BR'><head><title>Sobre</title></head><body>{PT_BODY}</body></html>",
    "/pt-br/preços": f"<html lang='pt-BR'><head><title>Preços</title></head><body>{PT_BODY}</body></html>",
    "/pt-br/noticias/a": f"<html lang='pt-BR'><head><title>N</title></head><body>{PT_BODY}</body></html>",
    "/en/about": "<html lang='en'><head><title>About</title></head><body>About us</body></html>",
}


@pytest.fixture
def pt_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in PT_PAGES:
            return httpx.Response(200, text=PT_PAGES[path], headers={"content-type": "text/html; charset=utf-8"})
        return httpx.Response(404, text="", headers={"content-type": "text/html"})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_crawl_of_portuguese_site_fetches_business_pages_first(pt_client):
    pages = Crawler(pt_client, max_pages=3).crawl(PT_SITE)
    urls = [p.url for p in pages]
    assert urls[1:] == [PT_SITE + "/pt-br/sobre", PT_SITE + "/pt-br/pre%C3%A7os"]


# --------------------------------------------------------------------------- pipeline and prompts


@pytest.fixture
def store(tmp_path: Path) -> pipeline.RunStore:
    return pipeline.RunStore(tmp_path / "run")


def test_crawl_records_site_language_and_prompts_follow_it(store, pt_client):
    pipeline.stage_crawl(store, PT_SITE, Crawler(pt_client, max_pages=3))
    assert store.meta()["site_lang"] == "pt-br" and store.lang() == "pt-br"
    from tests.conftest import identity_payload, response, tool_use
    payload = identity_payload()
    for value in payload.values():
        if isinstance(value, dict):
            value.update(value=None, classification="unknown", evidence_ids=[])
    payload["company_name"].update(value="Esfera", classification="company_claim", evidence_ids=["E001"])
    client = stage_router({"submit_identity": lambda kw: response(tool_use("submit_identity", payload))})
    pipeline.stage_identify(store, LLM(client, model="m"))
    assert "Brazilian Portuguese" in client.messages.calls[0]["system"][0]["text"]


def test_research_prompt_names_language_and_local_sources():
    text = prompts.research_user_prompt("Esfera", PT_SITE, {"corporate": "g"}, "- x", "pt-br", 10)
    assert "Brazilian Portuguese" in text and "Receita Federal" in text and "and in English" in text
    assert "- corporate: g" in text and "at most 10 searches" in text
    assert "both in" not in prompts.research_user_prompt("Acme", SITE, {"corporate": "g"}, "- x")
    assert set(prompts.LOCAL_SOURCES) == set(SUPPORTED)


def test_set_lang_rejects_unsupported(store, pt_client):
    pipeline.stage_crawl(store, PT_SITE, Crawler(pt_client, max_pages=2), lang="de")
    assert store.lang() == "de" and store.site_lang() == "pt-br"
    with pytest.raises(StageError):
        store.set_lang("it")


@pytest.mark.parametrize("lang,summary,question", [
    ("pt-br", "## 1. Sumário Executivo", "**Qual parece ser o ativo competitivo mais forte da empresa?**"),
    ("fr", "## 1. Synthèse", "**Quel semble être l'atout concurrentiel le plus solide de l'entreprise ?**"),
    ("de", "## 1. Zusammenfassung", "**Was scheint der stärkste Wettbewerbsvorteil des Unternehmens zu sein?**"),
    ("es", "## 1. Resumen ejecutivo", "**¿Cuál parece ser el activo competitivo más sólido de la empresa?**"),
])
def test_cli_lang_override_translates_the_report(tmp_path, http_client, monkeypatch, lang, summary, question):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    out = str(tmp_path / "r")
    common = ["--out", out, "--max-pages", "4", "--delay", "0"]
    assert cli.main(common + ["--lang", lang, "run", "--url", SITE],
                    client_factory=stage_router, http_client=http_client) == 0
    md = (tmp_path / "r" / "report.md").read_text(encoding="utf-8")
    assert summary in md and question in md
    assert "Executive Summary" not in md and "Classification key" not in md
    # A language change invalidates model outputs; regenerate with the new configuration.
    assert cli.main(common + ["--lang", "en", "report"], client_factory=stage_router, http_client=http_client) == 2
    assert cli.main(common + ["--lang", "en", "run", "--url", SITE], client_factory=stage_router, http_client=http_client) == 0
    assert "## 1. Executive Summary" in (tmp_path / "r" / "report.md").read_text(encoding="utf-8")
