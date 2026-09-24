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
    assert '<a href="#E001"' in out and '<a href="#E999"' not in out
    assert 'E001</a>, E999]' in out  # adjacent citations are one bracket, each id linked only if it has a row
    assert "&lt;x&gt; &amp; <b>b</b>" in out


def test_encodable_keeps_accents_and_replaces_what_helvetica_cannot_draw(monkeypatch):
    monkeypatch.setattr(pdf, "TTF", False)  # the built-in Helvetica fallback
    assert pdf._encodable("Relatório — “ok” → ≈") == "Relatório — “ok” -> ~"
    monkeypatch.setattr(pdf, "TTF", True)  # an embedded Unicode font draws everything
    assert pdf._encodable("→ ≈") == "→ ≈"


def test_render_pdf_writes_a_linked_document(tmp_path):
    path = pdf.render_pdf(MD, tmp_path / "r.pdf", lang="en")
    data = path.read_bytes()
    assert data.startswith(b"%PDF")
    assert b"/Outlines" in data  # section bookmarks
    assert b"/Link" in data  # citation and URL links


WIDE = """# Company Intelligence Report: Acme

## 2. Company Snapshot

| Field | Value |
|---|---|
| Headquarters | Porto Alegre, Brazil *(Verified fact)* [E001] |
| Founded | Unknown |

## 5. Products and Services

| Offering | Type | Target Customer | Problem Solved | Key Capabilities | Business Benefit | Monetization | Basis |
|---|---|---|---|---|---|---|---|
| Monitor | core product | Plants | Downtime | Telemetry | Uptime | Subscription | Company claim [E001] [E002] |
| — | — | — | — | — | — | — | No offerings could be extracted from the website |

## 21. Sources

| Id | Title | Publisher | URL | Published | Type | Source kind | Accessed |
|---|---|---|---|---|---|---|---|
| E001 | Home | acme.test | https://acme.test/ | n/a | first party | company material | 2026-09-23 |
| E002 | News | news.test | https://news.test/a | n/a | third party | news | 2026-09-23 |
"""


def test_wide_tables_become_cards_and_sources_drop_empty_columns():
    t = Translator("en")
    blocks = pdf.parse_markdown(WIDE)
    products = next(b for b in blocks if b.kind == "table" and len(b.rows[0]) == 8)
    cards = pdf._cards(products, pdf._styles(), pdf.Inline(t, {"E001"}), t)
    assert len([c for c in cards if c.__class__.__name__ == "KeepTogether"]) == 1
    sources = next(b for b in blocks if b.kind == "table" and b.rows[0][0] == "Id")
    lean, caption = pdf._lean_sources(sources, t)
    assert lean.rows[0] == ["Id", "Title", "Publisher", "URL", "Source kind"]
    assert lean.rows[2][-1] == "news · third party" and caption == "All sources accessed on 2026-09-23."


def test_cover_facts_and_evidence_counts_come_from_the_report():
    t = Translator("en")
    blocks = pdf.parse_markdown(WIDE)
    assert pdf._key_facts(blocks, t) == [("Headquarters", "Porto Alegre, Brazil")]
    counts = pdf._evidence_counts(blocks, pdf.Inline(t))
    assert counts["verified_fact"] == 1 and counts["company_claim"] == 1


def test_basis_cells_and_tags_do_not_break_inside():
    inline = pdf.Inline(Translator("pt-br"), {"E001"})
    out = inline.basis("Afirmação da empresa [E001]")
    assert "AFIRMAÇÃO&nbsp;DA&nbsp;EMPRESA" in out and 'href="#E001"' in out
    assert inline.basis("plain text") == "plain text"


def test_wide_report_renders(tmp_path):
    assert pdf.render_pdf(WIDE, tmp_path / "w.pdf").read_bytes().startswith(b"%PDF")



def test_citation_groups_are_sorted_and_deduplicated():
    out = pdf.Inline(Translator("en"))("x [E101][E100]. y [E040][E139] [E098][E040]")
    assert "[E100, E101]" in out and "[E040, E098, E139]" in out


def test_prompts_keep_customer_pains_and_prose_honest():
    from bi_agent import prompts

    assert "CUSTOMERS have" in prompts.ANALYZE_SYSTEM and "not customer\n  pains" in prompts.ANALYZE_SYSTEM
    assert "complete sentences" in prompts.NARRATE_SYSTEM and "must agree with the structured analysis" in prompts.NARRATE_SYSTEM


CONSULTING = """# Company Intelligence Report: Acme

## Key messages

- Acme wins on distribution [E001]
- Pricing is opaque [E002]

## 3. What the Company Does

**Takeaway:** Acme sells widgets to factories [E001]

Acme builds monitoring software [E001].

**Segments**

- Factories *(Company claim)* [E001]

## 10. Market Landscape

**Market sizing**

| Metric | Value | Year | Methodology | Limitations | Source |
|---|---|---|---|---|---|
| TAM | USD 2.13 billion (2024) | 2024 | top-down | vendor | [E002] |

## 11. Competitive Landscape

**Competitive positioning**

Breadth: offering range  
Price: price level

| Company | Breadth | Price | Rationale | Basis |
|---|---|---|---|---|
| Acme | High | Medium | r | Analytical inference [E001] |
| Rival | Low | High | r | Analytical inference [E002] |

## 16. SWOT

### Strengths

- Brand *(Company claim)* [E001]

### Weaknesses

- Cost *(Company claim)* [E001]

### Opportunities

- Exports *(Analytical inference)* [E002]

### Threats

- Rivals *(Third-party claim)* [E002]

### Business maturity

| Dimension | Level | Evidence |
|---|---|---|
| Product maturity | Established | Shipping since 2015 [E001] |
| Brand maturity | Not rated | Unknown |

## 21. Sources

| Id | Title | URL |
|---|---|---|
| E001 | Home | https://acme.test/ |
| E002 | News | https://news.test/a |
"""


def test_consulting_layout_elements_are_built():
    story, _ = pdf.build_story(CONSULTING, "en")
    kinds = {type(f).__name__ for f in story}
    assert {"PositioningMap", "SizingChart"} <= kinds
    texts = " ".join(getattr(f, "text", "") for f in story if hasattr(f, "text"))
    assert "Key messages" in texts and "EXHIBIT 1" in texts and "Appendix: evidence detail" in texts
    assert "Acme sells widgets to factories" in texts  # the action title
    heads = [f for f in story if isinstance(f, pdf.SectionHeading)]
    assert any(h.plain == "3. What the Company Does" for h in heads)  # the TOC keeps the topic


def test_money_parsing_and_harvey_ranks():
    assert pdf._money("USD 2.13 billion (2024)") == ("USD", 2.13e9)
    assert pdf._money("R$ 9,6 milhões") == ("BRL", 9.6e6)
    assert pdf._money("not a number") is None
    assert pdf._money_label("USD", 2.13e9) == "USD 2.1 bn"


def test_consulting_layout_renders(tmp_path):
    assert pdf.render_pdf(CONSULTING, tmp_path / "c.pdf").read_bytes().startswith(b"%PDF")
