"""Resource identity shared by retrieval and evidence. Aliases require observed redirects."""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit, urlunsplit

import tldextract

_suffix = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)
_tracking = re.compile(r"^(utm_|fbclid$|gclid$|mc_)")


def normalize_url(url: str, *, strip_tracking: bool = False) -> str:
    try:
        if any(ord(c) < 33 for c in url.strip()) or re.search(r"%(?![0-9a-fA-F]{2})", url):
            raise ValueError("invalid characters")
        p = urlsplit(url.strip())
        if p.scheme.lower() not in {"http", "https"} or not p.hostname or p.username or p.password:
            raise ValueError("absolute HTTP(S) URL without credentials required")
        host = p.hostname.encode("idna").decode("ascii").lower()
        if ":" in host:
            host = f"[{ipaddress.IPv6Address(host)}]"
        elif not re.fullmatch(r"[a-z0-9.-]+", host) or ".." in host:
            raise ValueError("invalid hostname")
        port = p.port
        if port is not None and port != {"http": 80, "https": 443}[p.scheme.lower()]:
            host += f":{port}"
        query = p.query
        if strip_tracking:
            query = "&".join(x for x in query.split("&") if not _tracking.match(x.split("=", 1)[0]))
        return urlunsplit((p.scheme.lower(), host, p.path or "/", query, ""))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"invalid URL {url!r}: {exc}") from exc


def hostname(url: str) -> str:
    return urlsplit(normalize_url(url)).hostname or ""


def registrable_domain(url: str) -> str:
    host = hostname(url)
    parts = _suffix(host)
    # Reserved test domains have no PSL suffix. Keep the last two labels in fixtures.
    return parts.top_domain_under_public_suffix or ".".join(host.split(".")[-2:])


def origin(url: str) -> str:
    p = urlsplit(normalize_url(url))
    return f"{p.scheme}://{p.netloc}"
