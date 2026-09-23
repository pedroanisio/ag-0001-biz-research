"""Command-line entry point. Each stage is a subcommand; ``run`` chains them all.

    bi-agent run --url https://example.com --out runs/example
    bi-agent crawl --url https://example.com --out runs/example
    bi-agent identify --out runs/example
    bi-agent signals --out runs/example
    bi-agent research --out runs/example
    bi-agent analyze --out runs/example
    bi-agent narrate --out runs/example
    bi-agent report --out runs/example
"""

from __future__ import annotations

import argparse
import logging
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

STAGES_NEEDING_LLM = {"identify", "signals", "research", "analyze", "narrate", "run"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bi-agent", description="Company intelligence research fleet")
    p.add_argument("--out", required=True, help="run directory (created if missing)")
    p.add_argument("--model", default=os.environ.get("BI_AGENT_MODEL", DEFAULT_MODEL))
    p.add_argument("--max-pages", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=64_000, help="output token limit per model response")
    p.add_argument("--max-search-uses", type=int, default=10, help="web_search calls per research topic group")
    p.add_argument("--delay", type=float, default=0.5, help="seconds between page fetches")
    p.add_argument("--lang", choices=SUPPORTED, default=None,
                   help="report language (default: the website's language, detected at crawl)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="stage", required=True)
    for name in ("run", "crawl"):
        s = sub.add_parser(name)
        s.add_argument("--url", required=True)
    for name in ("identify", "signals", "research", "analyze", "narrate", "report"):
        sub.add_parser(name)
    return p


def make_llm(args: argparse.Namespace, client_factory: Callable[[], object] = build_client) -> LLM:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise BiAgentError("ANTHROPIC_API_KEY is not set; the LLM stages cannot run")
    return LLM(client_factory(), model=args.model, max_search_uses=args.max_search_uses, max_tokens=args.max_tokens)


def make_crawler(args: argparse.Namespace, http_client: httpx.Client | None = None) -> Crawler:
    return Crawler(http_client or httpx.Client(), max_pages=args.max_pages, delay_seconds=args.delay)


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


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[], object] = build_client,
    http_client: httpx.Client | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    store = pipeline.RunStore(Path(args.out))
    try:
        llm = make_llm(args, client_factory) if args.stage in STAGES_NEEDING_LLM else None
        if args.lang and args.stage not in ("run", "crawl"):
            store.set_lang(args.lang)
        if args.stage == "run":
            md = pipeline.run_all(store, args.url, make_crawler(args, http_client), llm, args.lang)
            print(f"report written to {store.path('report.md')} ({len(md)} chars)")
        elif args.stage == "crawl":
            pages = pipeline.stage_crawl(store, args.url, make_crawler(args, http_client), args.lang)
            print(f"crawled {len(pages)} pages into {store.dir} (site language: {store.site_lang()}, "
                  f"report language: {store.lang()})")
        elif args.stage == "identify":
            ident = pipeline.stage_identify(store, llm)
            print(f"identity: {ident.company_name.value or 'unknown'}")
        elif args.stage == "signals":
            sig = pipeline.stage_signals(store, llm)
            print(f"site signals: {len(sig.offerings)} offerings")
        elif args.stage == "research":
            res = pipeline.stage_research(store, llm)
            print(f"research: {len(res.findings)} findings, {len(res.rejected)} rejected")
        elif args.stage == "analyze":
            pipeline.stage_analyze(store, llm)
            print("analysis written")
        elif args.stage == "narrate":
            pipeline.stage_narrate(store, llm)
            print("narrative written")
        else:
            md = pipeline.stage_report(store)
            print(f"report written to {store.path('report.md')} ({len(md)} chars)")
        if llm is not None:
            _print_usage(store)
    except anthropic.APIStatusError as exc:
        _print_usage(store)
        message = exc.body.get("error", {}).get("message") if isinstance(exc.body, dict) else None
        print(f"error: Anthropic API returned {exc.status_code}: {message or exc}", file=sys.stderr)
        print(f"  finished stages are saved in {store.dir}; re-run the failed stage and the ones after it",
              file=sys.stderr)
        return 3
    except BiAgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        for e in getattr(exc, "errors", [])[:20]:
            print(f"  - {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
