from bi_agent import pdf
from bi_agent.i18n import Translator

MD = """# Company Intelligence Report: Acme Widgets

Subject URL: https://acme.test  
Report generated: 2026-09-23

> The website yielded little text.

## 1. Executive Summary

Acme sells widgets [E001] → fast.

**Segments**

- SMEs *(Company claim)* [E001]
- see https://acme.test/a_b_c *(Analytical inference)* [E999]

## 21. Sources

| Id | Title | URL |
|---|---|---|
| E001 | Home \\| Acme | https://acme.test/ |
"""


def test_parse_markdown_recognises_the_report_subset():
    kinds = [b.kind for b in pdf.parse_markdown(MD)]
    assert kinds == ["title", "para", "quote", "h1", "para", "label", "bullets", "h1", "table"]
    blocks = pdf.parse_markdown(MD)
    assert blocks[1].text == "Subject URL: https://acme.test\nReport generated: 2026-09-23"
    assert blocks[-1].rows == [["Id", "Title", "URL"], ["E001", "Home \\| Acme", "https://acme.test/"]]


def test_inline_markup_colours_labels_and_links_only_known_citations():
    inline = pdf.Inline(Translator("en"), targets={"E001"})
    out = inline("see https://acme.test/a_b_c *(Company claim)* [E001] [E999] <x> & **b**")
    assert '<a href="https://acme.test/a_b_c"' in out and "<i>b_c" not in out
    assert f'color="{pdf.CLASS_COLORS["company_claim"]}"' in out
    assert '<a href="#E001"' in out and '<a href="#E999"' not in out and "[E999]" in out
    assert "&lt;x&gt; &amp; <b>b</b>" in out


def test_encodable_keeps_accents_and_replaces_what_helvetica_cannot_draw():
    assert pdf._encodable("Relatório — “ok” → ≈") == "Relatório — “ok” -> ~"


def test_render_pdf_writes_a_linked_document(tmp_path):
    path = pdf.render_pdf(MD, tmp_path / "r.pdf", lang="en")
    data = path.read_bytes()
    assert data.startswith(b"%PDF")
    assert b"/Outlines" in data  # section bookmarks
    assert b"/Link" in data  # citation and URL links
