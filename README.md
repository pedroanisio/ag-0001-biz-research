# bi-agent

An evidence-driven company research pipeline. It crawls one company's website, researches external sources, and produces a 21-section Markdown/PDF report with classifications and citations.

## Install and run

```sh
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...
bi-agent --out runs/acme run --url https://acme.example
```

Python 3.10+ on a platform with `fcntl` writer locks. Research uses Anthropic's server-side search and fetch tools. Tests use mock HTTP transports and scripted model responses; they incur no API charges.

Global options go **before** the stage name; `--url` goes after `crawl` or `run`.

```sh
bi-agent --out runs/acme crawl --url https://acme.example
bi-agent --out runs/acme identify
bi-agent --out runs/acme signals
bi-agent --out runs/acme research
bi-agent --out runs/acme resolve
bi-agent --out runs/acme analyze
bi-agent --out runs/acme narrate
bi-agent --out runs/acme report
bi-agent --out runs/acme status
```

`report` generates `report.md` and `report.pdf` from validated saved artifacts without API calls. PDF citations link to the source table. `status` shows completed, stale, incomplete, and runnable stages.

## Evidence and claim contracts

A company's own domain is not retrieval proof. Accepted sources must have a saved crawl record or a successful search/fetch result. The ledger stores retrieval method, time, requested/final URLs, observed redirect aliases, available source text, and content limitations separately from model-written excerpts. Fetch error blocks never create retrieval records. Search results with unavailable or encrypted text remain discovery records and cannot support facts until text is fetched.

Factual statements use short source excerpts. The verifier checks that the statement and its supporting passage occur in the saved source text; URL equality alone is insufficient. This conservative extractive contract does not adjudicate arbitrary paraphrases or translations: such factual output must be regenerated as an excerpt. Inferences remain available with explicit quoted premises. The check establishes what the source says, not whether the publisher is truthful.

Verified facts require a supporting primary document or two independent third-party sources supporting that particular statement. Official registries use parsed domain boundaries and an explicit trust list. Filings need document-level indicators; an ordinary company page or government press release cannot gain primary-record status from a model label. Search snippets alone cannot establish a primary document. Independence uses the bundled Public Suffix List (including private suffixes), known publisher ownership, original-source metadata when available, duplicate content, and recognized syndication attribution. Unknown ownership and unmarked syndication may still require analyst review. The configured ownership groups are based on [News Corp's Dow Jones description](https://newscorp.com/?company=dow-jones) and [Thomson Reuters' news products](https://www.thomsonreuters.com/en/products-services/news-media).

Narrative entries reference stable IDs in a validated claim catalog saved as `claims.json` and preserve each selected statement, classification, and evidence. Inferences additionally identify factual premise claims. The renderer supplies classification labels and citations. Removing the last citation cannot turn a factual assertion into an inference. Unresolved inline assertions are rejected. Every repair performed by a pipeline model call records its original value, repaired value, and reasons in `audit.json`; research verification decisions are recorded there too. Saved inputs are validated again before narrative/report generation.

Retrieved pages appear as delimited source data in user messages, never in system instructions. The system prompt states that embedded roles and instructions cannot change the task, tools, or evidence rules. The analyst system prefix and tool definitions remain shared/cacheable; page caching across forced tool changes is not assumed. The adversarial regression fixture checks this request layout and rejection of unsupported output; it is not proof that prompt injection is eliminated.

## Safe saved runs

`manifest.json` is the atomic commit point. It references immutable, content-addressed files in `objects/`. Each stage artifact records its run ID, schema version, producing stage, model/prompt/configuration identifiers, and input hashes. `commits/` retains manifest history. Top-level JSON, Markdown and PDF files are readable projections; stages read the committed objects, so editing a projection does not change pipeline inputs.

Each stage holds a nonblocking per-run writer lock. Files are written with temporary files, `fsync`, and atomic replacement; a manifest is published after its related objects exist. Interrupted publication leaves the prior committed state readable. Re-crawling creates a new run revision and removes old artifact projections; earlier immutable objects and commit history remain available for inspection. Changes to upstream data mark dependent artifacts stale. Unversioned legacy directories cannot be resumed or rendered: start with a fresh crawl. There is no automatic migration that assumes old evidence is trustworthy.

Research retains `research.partial.json`, including after publishing partial findings. Every group has pending/completed/failed status, attempts and failure history. Each outcome commits progress together with the ledger. Transient failures get at most two attempts per stage invocation; permanent API errors stop immediately. Restarting research calls only unfinished groups, and rejects checkpoints with incompatible inputs or configuration. Reports distinguish operational failures from completed searches that found no reliable evidence.

Follow-ups upsert claims by topic, entity, time scope and normalized statement. They merge supporting evidence and reassess classification, deduplicate within each response, and preserve conflicting statements separately. Progress counts supported new facts, independent corroboration and supported gap resolutions. A round with fewer than three improvements ends the follow-up loop.

## Crawl policy

All pages, robots files, sitemaps and redirect hops share one request budget and fetch path. The crawler checks HTTP(S) URL validity, site/origin permissions, destination addresses, per-origin robots rules and pacing before issuing requests. Redirects cannot leave the selected site or downgrade HTTPS. Initial canonical-host redirects are limited to the supplied host and its root/`www` counterpart. Nonstandard ports are confined to the explicitly supplied origin. Robots redirects stay on their own origin. Missing robots files (404/410) allow crawling; unavailable/error policies fail closed.

Loopback, private and link-local destinations are rejected by default. `--allow-private` explicitly enables crawling internal sites. Decoded response bodies, including compressed responses, are streamed with a 3 MB cap. The crawler does not execute JavaScript; thin sites receive more research searches and one extra follow-up round.

One URL identity policy applies throughout: normalize scheme/host and the scheme's default port, preserve path/query case, trailing slash and non-default ports, strip fragments, and link aliases only through observed redirects. Tracking parameters are preserved by default; the URL utility exposes their removal as an explicit option.

## Usage, budgets and options

Each API call reserves an allowance in `usage.json` **before** sending the request, then immediately persists its final usage. Records include run/stage/attempt/call IDs, model, token/cache/search counts, and pricing assumptions. Totals are derived from the cumulative history; retries and stage restarts never replace earlier charges. Interrupted calls retain their reservation with unavailable final usage. Unknown prices stay unknown. Estimates use the configured price table and are not vendor invoices.

```sh
bi-agent --out runs/acme --budget-tokens 8000000 --budget-searches 80 --budget-usd 40 run --url https://acme.example
```

Budgets persist across restarts; an explicit budget option revises its saved limit. New calls are refused when completed charges plus outstanding reservations plus the next allowance exceed a limit. The token allowance reserves one million input tokens (or the request's UTF-8 byte length if larger), plus the output limit; cost reservations use the higher cache-write input rate and maximum allowed searches. This intentionally leaves headroom for server tool content. Budget refusal preserves completed work. Unknown pricing prevents admission under a cost budget. The SDK's hidden retries are disabled so every attempted request is recorded.

Other options: `--model` (or `BI_AGENT_MODEL`, default `claude-sonnet-5`), `--max-pages` (60), `--max-tokens` per response (64000), `--max-search-uses` per research call (10), `--max-fetch-uses` (3; zero disables fetch), `--followup-rounds` (2), `--delay` (0.5 seconds), `--lang`, and `-v`. Model responses are streamed; truncated structured output fails immediately.

Languages: English (`en`), Brazilian Portuguese (`pt-br`), French (`fr`), German (`de`), Spanish (`es`). Crawl priorities and research sources cover all five. Language is detected from page text, falling back to HTML language metadata. `--lang` overrides report/analysis language; factual source excerpts retain their original wording. Changing language invalidates model-dependent outputs, which must be regenerated.

## Development

```sh
python -m pytest --cov=bi_agent --cov-report=term-missing
```

The coverage gate is 90% combined statement/branch coverage. `tests/test_improvements.py` contains the acceptance regressions for `improvements-01.md`, alongside the existing crawler, model, pipeline, multilingual and PDF tests.

Main modules: `models.py` (contracts/provenance), `urls.py` (resource identity), `crawler.py`, `llm.py`, `accounting.py`, `store.py` (atomic persistence), `pipeline.py`, `prompts.py`, `report.py`, `pdf.py`, `i18n.py`, and `cli.py`.
