# bi-agent

An evidence-driven company research fleet. Given one URL, it crawls the company's site, researches the company beyond the site, and produces a 21-section Company Intelligence Report in which every claim is classified (verified fact, company claim, third-party claim, analytical inference, unknown) and traceable to a source the pipeline actually retrieved.

## Install

```
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

Python 3.10+. Off-site research uses Anthropic's server-side `web_search` tool, so no second search vendor is needed.

## Run

```
bi-agent --out runs/acme run --url https://acme.example
```

The fleet is eight scripts sharing one run directory, so every stage can be run, inspected and re-run on its own:

```
bi-agent --out runs/acme crawl --url https://acme.example   # pages.json, evidence.json, run.json
bi-agent --out runs/acme identify                            # identity.site.json (website only)
bi-agent --out runs/acme signals                             # signals.json
bi-agent --out runs/acme research                            # findings.json, extends evidence.json (resumable)
bi-agent --out runs/acme resolve                             # identity.json: website identity refined by research
bi-agent --out runs/acme analyze                             # analysis.json
bi-agent --out runs/acme narrate                             # narrative.json
bi-agent --out runs/acme report                              # report.pdf (and report.md, its source)
```

The PDF is typeset with reportlab from `report.md`: cover page, table of contents and PDF bookmarks, colour-coded classification labels, and `[E012]` citations that link to their row in the Sources table. Re-running `report` regenerates both files from the saved stage outputs without any API call.

Options: `--model` (default `claude-sonnet-5`, or `BI_AGENT_MODEL`), `--lang` (see below), `--max-pages` (60), `--max-tokens` per response (64k; responses are streamed, and a truncated response stops the stage instead of being retried), `--max-search-uses` per research call (10), `--max-fetch-uses` per research call (3, 0 disables `web_fetch`), `--followup-rounds` (2), `--delay` between page fetches (0.5 s), `-v`. Global options go before the stage name, and `--url` goes after it.

## Languages

Supported: English (`en`), Brazilian Portuguese (`pt-br`), French (`fr`), German (`de`) and Spanish (`es`).

- **Detection.** The crawl stage works out the site's language from the page text (a stop-word vote), falling back to `<html lang>` and then to English. The text comes first because many templates ship `lang="en-US"` over Portuguese or Spanish content. The result is saved as `site_lang` in `run.json`.
- **Crawling.** Page-priority keywords cover all five languages (`/sobre`, `/quem-somos`, `/a-propos`, `/ueber-uns`, `/quienes-somos`, `/precos`, `/carreiras` …). Paths are matched without accents or percent-encoding, so `/preços` counts as `/precos`. A locale prefix such as `/pt-br/` is ignored for scoring, and other language versions of the same site (`/en/…` on a Portuguese site) are fetched last.
- **Research.** The model searches in the site's language and in English, and gets that language's local registries, press, review and job sites (Receita Federal / Reclame Aqui, Infogreffe / Pappers, Handelsregister / Kununu, BORME / CNMV …).
- **Report language.** The report is written in the site's language by default. `--lang` overrides it, and it can be passed to any stage: `bi-agent --out runs/acme --lang de report` re-renders an existing run's headings and labels. Prose the model already wrote stays in its original language until `narrate` (and the stages before it) are re-run. The model always returns the strategic questions, maturity dimensions and enum values as English keys, and the renderer translates them (`bi_agent/i18n.py`).

## How the evidence discipline is enforced

Every model output crosses a typed boundary with five controls:

1. **Typed parse that rejects unknown fields.** Each stage forces a tool call whose input schema is a pydantic model with `extra="forbid"` (`bi_agent/models.py`).
2. **Semantic validation the schema cannot express.** Every `evidence_ids` entry and every inline `[E###]` citation must resolve in the evidence ledger; `verified_fact` requires a primary record (government, regulator, registry or statutory filing) or two independent third-party sources on different domains (the company's own material and forums never count as independent); `company_claim` requires a first-party source; the ten strategic questions and eight maturity dimensions must appear with their exact text; market sizing requires a source, year, methodology and limitations (`models.semantic_errors`).
3. **Defined failure path with a typed error.** Reference errors are repaired in code before validation, with no extra model call: evidence ids and `[E###]` citations missing from the ledger are removed, and a classification the remaining evidence cannot support is lowered (`verified_fact` backed only by the company becomes `company_claim`; a claim left with no evidence becomes `analytical_inference`). These are the same rules the research stage applies to findings, and every repair is logged as a warning (`models.repair_refs`). Everything else (missing questions, wrong types, unknown fields) is fed back to the model for a bounded number of attempts, then `LLMOutputError` carries the error list to the CLI (exit code 2). An Anthropic API error ends the run with exit code 3 and a one-line message.
4. **Adversarial tests.** `tests/` feed unknown fields, fabricated evidence ids, mis-classified claims, unsearched source URLs, text-only model turns, search error blocks, off-site redirects and robots-blocked paths, and assert the failure behaviour. 120 tests, 98 % branch coverage, no network.
5. **Loop bounds owned by ordinary code.** `max_pages`, `max_fetches`, `max_attempts`, `max_research_turns`, `max_search_uses`, five sitemaps, twelve research topics in five groups.

The research stage adds two more guards:

- **Retrieved sources only.** A finding's source URL is accepted only if `web_search` returned it or `web_fetch` retrieved it in the same call (or it belongs to the company's own site). Findings whose sources the model reconstructed are discarded and listed at the end of the report under "Discarded during verification".
- **Source kinds.** Every source is labelled with the brief's source hierarchy: government/regulator, statutory filing, company material, investor disclosure, partner/customer, industry publication, news, database/aggregator, forum/social. The label is shown in the Sources table and in the evidence index the model sees. A label that would let a single source verify a fact (a primary record) is kept only for official hosts (`.gov`, `.gouv`, `.gob`, Companies House, Handelsregister, Infogreffe, CVM, SEC …). A CNPJ-lookup or registry-copy site becomes a database/aggregator.

## Research depth

- **Topic groups, then follow-up rounds.** After the five topic groups, up to `--followup-rounds` (2) follow-up calls are each given everything found so far. They investigate the entities discovered (parent company, owners and investors, founders, major customers and partners, key competitors), contradictions, single-source claims and recorded gaps. Research stops early once a round adds fewer than 3 new findings: that is the point of diminishing returns.
- **Reading, not just searching.** Research calls also have `web_fetch` (up to `--max-fetch-uses`, 3 per call, 20k tokens per page), so a registry entry, filing or annual report can be read in full.
- **Thin or JavaScript-rendered websites.** The crawler flags pages that are app shells (almost no text, script-driven). Under 20k characters or 5 pages the site counts as thin: every research call gets 50 % more searches, one extra follow-up round is allowed, the prompts tell the model to rely on external sources, and the report says so at the top. The crawler does not run JavaScript.
- **Product or company.** `identify` records what the website represents (`website_subject`: the company itself, or a product or brand of a named organisation), using legal notices, terms, copyright lines and registration numbers. Legal-notice pages (Impressum, mentions légales, aviso legal) are fetched early for this. Research searches under the related names too (legal name, brands, parent), and `analyze` covers both the product and the organisation when they differ.

## Layout

```
bi_agent/models.py    typed contracts, evidence ledger, semantic checks
bi_agent/crawler.py   bounded robots-respecting crawler with page prioritisation
bi_agent/llm.py       Anthropic wrapper: forced structured output, bounded retries, search-hit capture
bi_agent/prompts.py   the analyst brief per stage (the research spec lives here)
bi_agent/pipeline.py  stages and the run directory
bi_agent/report.py    Markdown renderer (adds no facts)
bi_agent/pdf.py       typesets report.md as report.pdf (reportlab)
bi_agent/i18n.py      supported languages, site-language detection, translated report strings
bi_agent/cli.py       subcommands
```

## Cost and size

One full run is 5 structured calls (plus any validation retries) and 6–7 research calls (5 topic groups plus 1–2 follow-up rounds; one more on thin sites), each with up to 10 searches and 3 page fetches, so at most 70 searches (105 on a thin site). Page text sent to the model is capped at 260k characters (6k per page).

What keeps the bill down:

- **Measured, not guessed.** Every call's tokens, cache reads and writes and web searches are written to `usage.json` per stage, with an estimated cost. The CLI prints the running total after each stage.
- **Prompt caching on every request.** Research continuations and validation retries re-read their unchanged prefix at 10 % of the input price. `identify` and `signals` send an identical system prompt (analyst role plus the highest-priority pages) and the same two tools, so `signals` reads those pages from the cache.
- **Five research groups instead of twelve topics.** Topics that share queries (corporate, funding and leadership, for example) are researched in one call.
- **Mechanical errors are fixed in code.** A wrong citation no longer costs a full regeneration of the analysis (see control 3).
- **Stop on errors that will repeat.** No credit, an invalid key or an unknown model stops the research stage at once instead of failing every remaining group and then paying for `analyze` and `narrate` on empty data. Rate limits, server errors and invalid output skip only that group.
- **Resumable research.** Progress is saved after each group in `research.partial.json`, so re-running `research` after a failure pays only for the groups that had not finished.
- **Compact JSON in prompts** and thinking disabled by default (the output is a forced tool call validated in code). On models that support it, research uses `web_search_20260209`, which filters search results before they enter the context. Older models get `web_search_20250305` automatically.
