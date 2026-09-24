"""Command-line entry point. Each stage is a subcommand; ``run`` chains them all.

    bi-agent run --url https://example.com --out runs/example
    bi-agent crawl --url https://example.com --out runs/example
    bi-agent identify --out runs/example
    bi-agent signals --out runs/example
    bi-agent research --out runs/example
    bi-agent resolve --out runs/example
    bi-agent analyze --out runs/example
    bi-agent narrate --out runs/example
    bi-agent report --out runs/example
"""

from __future__ import annotations

import argparse
import logging
import json
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

import anthropic
import httpx

from . import pipeline
from .crawler import Crawler
from .errors import BiAgentError
from .i18n import SUPPORTED
from .llm import DEFAULT_MODEL, LLM, build_client

STAGES_NEEDING_LLM = {"identify", "signals", "research", "resolve", "analyze", "narrate", "run"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bi-agent", description="Company intelligence research fleet")
    p.add_argument("--out", required=True, help="run directory (created if missing)")
    p.add_argument("--model", default=os.environ.get("BI_AGENT_MODEL", DEFAULT_MODEL))
    p.add_argument("--max-pages", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=64_000, help="output token limit per model response")
    p.add_argument("--max-search-uses", type=int, default=10, help="web_search calls per research call")
    p.add_argument("--max-fetch-uses", type=int, default=3, help="web_fetch calls per research call (0 disables)")
    p.add_argument("--followup-rounds", type=int, default=pipeline.FOLLOWUP_ROUNDS,
                   help="follow-up research rounds after the topic groups (stops earlier at diminishing returns)")
    p.add_argument("--budget-tokens", type=int, help="cumulative run token budget")
    p.add_argument("--budget-searches", type=int, help="cumulative run search budget")
    p.add_argument("--budget-usd", type=float, help="cumulative estimated cost budget; requires known pricing")
    p.add_argument("--allow-private", action="store_true", help="explicitly allow private-network crawling")
    p.add_argument("--delay", type=float, default=0.5, help="seconds between page fetches")
    p.add_argument("--lang", choices=SUPPORTED, default=None,
                   help="report language (default: the website's language, detected at crawl)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="stage", required=True)
    for name in ("run", "crawl"):
        s = sub.add_parser(name)
        s.add_argument("--url", required=True)
    for name in ("identify", "signals", "research", "resolve", "analyze", "narrate", "report", "status"):
        sub.add_parser(name)
    return p


def make_llm(args: argparse.Namespace, client_factory: Callable[[], object] = build_client) -> LLM:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise BiAgentError("ANTHROPIC_API_KEY is not set; the LLM stages cannot run")
    return LLM(client_factory(), model=args.model, max_search_uses=args.max_search_uses,
               max_fetch_uses=args.max_fetch_uses, max_tokens=args.max_tokens,
               run_budget={"tokens": args.budget_tokens, "searches": args.budget_searches, "cost_usd": args.budget_usd})


def make_crawler(args: argparse.Namespace, http_client: httpx.Client | None = None) -> Crawler:
    return Crawler(http_client or httpx.Client(), max_pages=args.max_pages, delay_seconds=args.delay, allow_private=args.allow_private)


def _print_usage(store: pipeline.RunStore) -> None:
    if not store.exists("usage.json"):
        return
    total = store.load_json("usage.json").get("total", {})
    cost = total.get("estimated_cost_usd")
    print(f"API usage so far: {total.get('calls', 0)} calls, {total.get('input_tokens', 0)} input + "
          f"{total.get('cache_read_input_tokens', 0)} cached + {total.get('cache_creation_input_tokens', 0)} "
          f"cache-write tokens, {total.get('output_tokens', 0)} output tokens, "
          f"{total.get('web_search_requests', 0)} searches"
          + (f", ~${cost:.2f}" if cost is not None else "") + f" (details in {store.path('usage.json')})")


def _transport_errors() -> tuple[type[BaseException], ...]:
    found = []
    for module in ("httpx2", "httpx"):
        try:
            found.append(__import__(module).TransportError)
        except (ImportError, AttributeError):
            pass
    return tuple(found)


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[], object] = build_client,
    http_client: httpx.Client | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    store = pipeline.RunStore(Path(args.out))
    try:
        llm = make_llm(args, client_factory) if args.stage in STAGES_NEEDING_LLM else None
        if llm is not None:
            llm.debug_dir = store.path("debug")  # rejected tool inputs, for diagnosing validation failures
        if args.lang and args.stage not in ("run", "crawl"):
            store.set_lang(args.lang)
        if args.stage == "run":
            md = pipeline.run_all(store, args.url, make_crawler(args, http_client), llm, args.lang,
                                  followup_rounds=args.followup_rounds)
            print(f"report written to {store.path('report.pdf')} (Markdown: {store.path('report.md')})")
        elif args.stage == "crawl":
            pages = pipeline.stage_crawl(store, args.url, make_crawler(args, http_client), args.lang)
            print(f"crawled {len(pages)} pages into {store.dir} (site language: {store.site_lang()}, "
                  f"report language: {store.lang()})")
        elif args.stage == "status":
            print(json.dumps(store.status(), indent=2))
        elif args.stage == "identify":
            ident = pipeline.stage_identify(store, llm)
            print(f"identity: {ident.company_name.value or 'unknown'}")
        elif args.stage == "signals":
            sig = pipeline.stage_signals(store, llm)
            print(f"site signals: {len(sig.offerings)} offerings")
        elif args.stage == "research":
            res = pipeline.stage_research(store, llm, followup_rounds=args.followup_rounds)
            print(f"research: {len(res.findings)} findings, {len(res.rejected)} rejected, "
                  f"{len(res.incomplete_groups)} unfinished groups (resume with research)")
        elif args.stage == "resolve":
            ident = pipeline.stage_resolve(store, llm)
            print(f"identity: {ident.company_name.value or 'unknown'} (legal name: {ident.legal_name.value or 'unknown'})")
        elif args.stage == "analyze":
            pipeline.stage_analyze(store, llm)
            print("analysis written")
        elif args.stage == "narrate":
            pipeline.stage_narrate(store, llm)
            print("narrative written")
        else:
            md = pipeline.stage_report(store)
            print(f"report written to {store.path('report.pdf')} (Markdown: {store.path('report.md')})")
        if llm is not None:
            _print_usage(store)
    except (anthropic.APIConnectionError, *_transport_errors()) as exc:
        _print_usage(store)
        print(f"error: connection to the Anthropic API failed after retries: {type(exc).__name__}: {str(exc)[:200]}",
              file=sys.stderr)
        print(f"  finished stages are saved in {store.dir}; re-run the failed stage and the ones after it",
              file=sys.stderr)
        return 3
    except anthropic.APIStatusError as exc:
        _print_usage(store)
        message = exc.body.get("error", {}).get("message") if isinstance(exc.body, dict) else None
        print(f"error: Anthropic API returned {exc.status_code}: {message or exc}", file=sys.stderr)
        print(f"  finished stages are saved in {store.dir}; re-run the failed stage and the ones after it",
              file=sys.stderr)
        return 3
    except (BiAgentError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        for e in getattr(exc, "errors", [])[:20]:
            print(f"  - {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
