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

The fleet is seven scripts sharing one run directory, so every stage can be run, inspected and re-run on its own:

```
bi-agent --out runs/acme crawl --url https://acme.example   # pages.json, evidence.json, run.json
bi-agent --out runs/acme identify                            # identity.json
bi-agent --out runs/acme signals                             # signals.json
bi-agent --out runs/acme research                            # findings.json, extends evidence.json
bi-agent --out runs/acme analyze                             # analysis.json
bi-agent --out runs/acme narrate                             # narrative.json
bi-agent --out runs/acme report                              # report.md
```

Options: `--model` (default `claude-sonnet-4-5`, or `BI_AGENT_MODEL`), `--lang` (see below), `--max-pages` (60), `--max-search-uses` per research topic (8), `--delay` between page fetches (0.5 s), `-v`. Global options go before the stage name, and `--url` goes after it.

## Languages

Supported: English (`en`), Brazilian Portuguese (`pt-br`), French (`fr`), German (`de`) and Spanish (`es`).

- **Detection.** The crawl stage works out the site's language from the page text (a stop-word vote), falling back to `<html lang>` and then to English. The text comes first because many templates ship `lang="en-US"` over Portuguese or Spanish content. The result is saved as `site_lang` in `run.json`.
- **Crawling.** Page-priority keywords cover all five languages (`/sobre`, `/quem-somos`, `/a-propos`, `/ueber-uns`, `/quienes-somos`, `/precos`, `/carreiras` …). Paths are matched without accents or percent-encoding, so `/preços` counts as `/precos`. A locale prefix such as `/pt-br/` is ignored for scoring, and other language versions of the same site (`/en/…` on a Portuguese site) are fetched last.
- **Research.** The model searches in the site's language and in English, and gets that language's local registries, press, review and job sites (Receita Federal / Reclame Aqui, Infogreffe / Pappers, Handelsregister / Kununu, BORME / CNMV …).
- **Report language.** The report is written in the site's language by default. `--lang` overrides it, and it can be passed to any stage: `bi-agent --out runs/acme --lang de report` re-renders an existing run's headings and labels. Prose the model already wrote stays in its original language until `narrate` (and the stages before it) are re-run. The model always returns the strategic questions, maturity dimensions and enum values as English keys, and the renderer translates them (`bi_agent/i18n.py`).

## How the evidence discipline is enforced

Every model output crosses a typed boundary with five controls:

1. **Typed parse that rejects unknown fields.** Each stage forces a tool call whose input schema is a pydantic model with `extra="forbid"` (`bi_agent/models.py`).
2. **Semantic validation the schema cannot express.** Every `evidence_ids` entry and every inline `[E###]` citation must resolve in the evidence ledger; `verified_fact` requires at least one third-party source; `company_claim` requires a first-party source; the ten strategic questions and eight maturity dimensions must appear with their exact text; market sizing requires a source, year, methodology and limitations (`models.semantic_errors`).
3. **Defined failure path with a typed error.** Validation errors are fed back to the model for a bounded number of attempts, then `LLMOutputError` carries the error list to the CLI (exit code 2).
4. **Adversarial tests.** `tests/` feed unknown fields, fabricated evidence ids, mis-classified claims, unsearched source URLs, text-only model turns, search error blocks, off-site redirects and robots-blocked paths, and assert the failure behaviour. 98 tests, 98.6 % branch coverage, no network.
5. **Loop bounds owned by ordinary code.** `max_pages`, `max_fetches`, `max_attempts`, `max_research_turns`, `max_search_uses`, five sitemaps, twelve research topics.

The research stage adds one more guard: a finding's source URL is accepted only if `web_search` returned that URL in the same call (or it belongs to the company's own site). Findings whose sources the model reconstructed are discarded and listed at the end of the report under "Discarded during verification".

## Layout

```
bi_agent/models.py    typed contracts, evidence ledger, semantic checks
bi_agent/crawler.py   bounded robots-respecting crawler with page prioritisation
bi_agent/llm.py       Anthropic wrapper: forced structured output, bounded retries, search-hit capture
bi_agent/prompts.py   the analyst brief per stage (the research spec lives here)
bi_agent/pipeline.py  stages and the run directory
bi_agent/report.py    Markdown renderer (adds no facts)
bi_agent/i18n.py      supported languages, site-language detection, translated report strings
bi_agent/cli.py       subcommands
```

## Cost and size

One full run is roughly 6 structured calls plus 12 research calls with up to 8 searches each. Sizing per run: page text sent to the model is capped at 260k characters (6k per page).
