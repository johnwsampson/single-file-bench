#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp", "httpx", "beautifulsoup4", "html2text"]
# ///
"""Web operations: search, fetch, engines. SearXNG backend with HTML fallback.

Usage:
    sft_web.py search "query" [options]
    sft_web.py fetch URL
    sft_web.py engines [--category CAT]
    sft_web.py mcp-stdio
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from html import unescape
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

import html2text
import httpx
from bs4 import BeautifulSoup

# =============================================================================
# LOGGING
# =============================================================================
_LEVELS = {"TRACE": 5, "DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40, "FATAL": 50}
_THRESHOLD = _LEVELS.get(os.environ.get("SFB_LOG_LEVEL", "INFO"), 20)
_LOG_DIR = os.environ.get("SFB_LOG_DIR", "")
_SCRIPT = Path(__file__).stem
_LOG = (
    Path(_LOG_DIR) / f"{_SCRIPT}_log.tsv"
    if _LOG_DIR
    else Path(__file__).parent / f"{_SCRIPT}_log.tsv"
)
_HEADER = "#timestamp\tscript\tlevel\tevent\tmessage\tdetail\tmetrics\ttrace\n"


def _log(
    level: str,
    event: str,
    msg: str,
    *,
    detail: str = "",
    metrics: str = "",
    trace: str = "",
):
    """Append TSV log line. Logging never crashes the main flow."""
    if _LEVELS.get(level, 20) < _THRESHOLD:
        return
    try:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        write_header = not _LOG.exists()
        with open(_LOG, "a") as f:
            if write_header:
                f.write(_HEADER)
            f.write(f"{ts}\t{_SCRIPT}\t{level}\t{event}\t{msg}\t{detail}\t{metrics}\t{trace}\n")
    except Exception:
        pass


# =============================================================================
# CONFIGURATION
# =============================================================================
EXPOSED = ["search", "fetch", "engines"]  # CLI + MCP — both interfaces

SEARXNG_BASE = "http://localhost:8888"

SEARCH_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
FETCH_TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0)

CONFIG = {
    "searxng_url": f"{SEARXNG_BASE}/search",
    "searxng_config_url": f"{SEARXNG_BASE}/config",
    "search_max_count": 50,
    "fetch_user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "fetch_max_content": 500_000,
    "fetch_max_redirects": 5,
}

# Lazy-initialized shared HTTP client
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    """Lazy-init shared HTTP client with connection pooling."""
    global _client
    if _client is None:
        _client = httpx.Client(
            timeout=SEARCH_TIMEOUT,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _client


# =============================================================================
# INVARIANTS — what we know to be true about valid requests
# =============================================================================


def _assert_fetchable_url(url: str):
    """URL is public HTTP(S) with a host that resolves to a public address."""
    import socket

    parsed = urlparse(url)
    assert parsed.scheme in ("http", "https"), (
        f"URL scheme is http or https (got {parsed.scheme})"
    )
    assert parsed.hostname, "URL has a hostname"
    # Our SearXNG instance is always reachable
    if parsed.hostname == "localhost":
        return
    # Resolve hostname and verify the address is public
    try:
        addr = ip_address(parsed.hostname)
    except ValueError:
        # Hostname, not IP — resolve it
        resolved = socket.getaddrinfo(parsed.hostname, None, type=socket.SOCK_STREAM)
        assert resolved, f"{parsed.hostname} resolves to an address"
        addr = ip_address(resolved[0][4][0])
    assert not (addr.is_private or addr.is_loopback or addr.is_link_local), (
        f"{parsed.hostname} resolves to a public address (got {addr})"
    )


# =============================================================================
# CORE FUNCTIONS — assert invariants, let exceptions propagate
# =============================================================================


def _parse_html_results(html_text: str, count: int = 10) -> list[dict]:
    """Parse SearXNG HTML response. HTML is external — defensive skipping is legitimate here."""
    soup = BeautifulSoup(html_text, "html.parser")
    articles = soup.find_all("article", class_=re.compile(r"result"), limit=count)

    results = []
    for art in articles:
        h3 = art.find("h3")
        if not h3:
            continue
        link = h3.find("a", href=True)
        if not link:
            continue

        url = unescape(link["href"])
        title = link.get_text(strip=True)

        snippet = ""
        content_p = art.find("p", class_="content")
        if content_p:
            snippet = content_p.get_text(strip=True)

        engines_list = []
        eng_div = art.find("div", class_="engines")
        if eng_div:
            engines_list = [
                span.get_text(strip=True) for span in eng_div.find_all("span")
            ]

        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "engines": engines_list,
            }
        )

    return results


def search_impl(
    query: str,
    count: int = 10,
    engines: str = None,
    category: str = None,
    time_range: str = None,
    pageno: int = None,
    language: str = None,
    safesearch: int = None,
    format: str = "json",
) -> tuple[list | str, dict]:
    """Search via SearXNG. Returns (results, metrics). Raises on failure."""
    _log("INFO", "search_start", f"query={query}")
    start = time.monotonic()
    client = _get_client()

    params = {"q": query, "format": format}
    if engines:
        params["engines"] = engines
    if category:
        params["categories"] = category
    if time_range:
        params["time_range"] = time_range
    if pageno:
        params["pageno"] = pageno
    if language:
        params["language"] = language
    if safesearch is not None:
        params["safesearch"] = safesearch

    resp = client.get(CONFIG["searxng_url"], params=params, timeout=SEARCH_TIMEOUT)

    # SearXNG returns 403 when JSON format is disabled — known behavior, fall back to HTML
    if resp.status_code == 403 and format == "json":
        _log("WARN", "search_fallback", "JSON 403, trying HTML")
        params["format"] = "html"
        resp = client.get(CONFIG["searxng_url"], params=params, timeout=SEARCH_TIMEOUT)
        resp.raise_for_status()

        results = _parse_html_results(resp.text, count)
        latency_ms = (time.monotonic() - start) * 1000
        _log(
            "INFO",
            "search_complete",
            f"found {len(results)} results",
            detail="format=html_fallback",
        )
        return results, {
            "query": query,
            "results": len(results),
            "latency_ms": round(latency_ms, 2),
            "format": "html_fallback",
            "status": "success",
        }

    resp.raise_for_status()
    latency_ms = (time.monotonic() - start) * 1000

    # Raw formats return as-is
    if format in ("csv", "rss"):
        _log("INFO", "search_complete", f"raw {format} response")
        return resp.text, {
            "query": query,
            "format": format,
            "latency_ms": round(latency_ms, 2),
            "status": "success",
        }

    # SearXNG JSON contract: response is application/json with a "results" array
    content_type = resp.headers.get("content-type", "")
    assert "application/json" in content_type, (
        f"Asked SearXNG for JSON, got {content_type}"
    )

    data = resp.json()
    assert "results" in data, (
        f"SearXNG JSON missing 'results' key, got: {list(data.keys())}"
    )

    results = [
        {
            "title": item["title"],  # SearXNG guarantees
            "url": item["url"],  # SearXNG guarantees
            "snippet": item.get("content", ""),  # optional per engine
            "engines": item["engines"],  # SearXNG guarantees
        }
        for item in data["results"][:count]
    ]

    _log("INFO", "search_complete", f"found {len(results)} results")
    return results, {
        "query": query,
        "results": len(results),
        "latency_ms": round(latency_ms, 2),
        "format": "json",
        "status": "success",
    }


def engines_impl(category: str = None, enabled_only: bool = True) -> tuple[dict, dict]:
    """Fetch available engines. Asserts SearXNG config contract."""
    _log("INFO", "engines_start", f"category={category}")
    start = time.monotonic()
    client = _get_client()

    resp = client.get(CONFIG["searxng_config_url"], timeout=SEARCH_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    # SearXNG config contract
    categories = data["categories"]
    raw_engines = data["engines"]

    latency_ms = (time.monotonic() - start) * 1000

    engines = []
    for eng in raw_engines:
        if enabled_only and not eng.get("enabled", False):
            continue
        if category and category not in eng.get("categories", []):
            continue
        engines.append(
            {
                "name": eng["name"],  # SearXNG guarantees
                "shortcut": eng["shortcut"],  # SearXNG guarantees
                "categories": eng["categories"],  # SearXNG guarantees
                "enabled": eng["enabled"],  # SearXNG guarantees
                "paging": eng.get("paging", False),
                "time_range": eng.get("time_range_support", False),
                "safesearch": eng.get("safesearch", False),
                "language": eng.get("language_support", False),
            }
        )

    _log("INFO", "engines_complete", f"showing {len(engines)} of {len(raw_engines)}")
    return {"categories": categories, "engines": engines}, {
        "total_engines": len(raw_engines),
        "shown": len(engines),
        "latency_ms": round(latency_ms, 2),
        "status": "success",
    }


FETCH_FORMATS = {"markdown", "text", "html"}


def fetch_impl(url: str, format: str = "markdown") -> tuple[dict, dict]:
    """Fetch URL and return content in requested format. Validates input, asserts invariants, raises on failure.

    Args:
        url: The URL to fetch content from
        format: Output format — "markdown" (default), "text", or "html"
    """
    _assert_fetchable_url(url)
    assert format in FETCH_FORMATS, (
        f"format is one of {sorted(FETCH_FORMATS)} (got {format!r})"
    )

    _log("INFO", "fetch_start", f"url={url} format={format}")
    start = time.monotonic()
    client = _get_client()
    headers = {"User-Agent": CONFIG["fetch_user_agent"]}
    max_bytes = CONFIG["fetch_max_content"]

    with client.stream(
        "GET",
        url,
        headers=headers,
        timeout=FETCH_TIMEOUT,
        follow_redirects=True,
        extensions={"max_redirects": CONFIG["fetch_max_redirects"]},
    ) as resp:
        resp.raise_for_status()

        # Content fits in our budget
        cl = resp.headers.get("content-length")
        assert not cl or int(cl) <= max_bytes, (
            f"Content-Length {cl} fits in {max_bytes} byte budget"
        )

        # Streaming confirms the budget holds
        chunks = []
        total = 0
        for chunk in resp.iter_bytes(chunk_size=8192):
            total += len(chunk)
            assert total <= max_bytes, (
                f"Download fits in {max_bytes} byte budget (at {total})"
            )
            chunks.append(chunk)

        content = b"".join(chunks)

    text = content.decode(resp.encoding or "utf-8", errors="replace")
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(
        ["script", "style", "nav", "footer", "aside", "iframe", "noscript"]
    ):
        tag.decompose()

    # Title extraction — HTML is external, fallback is legitimate
    title = url
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    if format == "html":
        out = str(soup).strip()
    elif format == "text":
        out = soup.get_text(separator="\n", strip=True)
    else:
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        out = h.handle(str(soup)).strip()

    latency_ms = (time.monotonic() - start) * 1000
    _log("INFO", "fetch_complete", f"chars={len(out)}", detail=f"url={url} format={format}")
    return (
        {"url": url, "title": title, "content": out, "format": format},
        {
            "latency_ms": round(latency_ms, 2),
            "chars": len(out),
            "format": format,
            "status": "success",
        },
    )


# =============================================================================
# CLI INTERFACE
# =============================================================================
def main():
    _log("INFO", "start", "Command started")
    import argparse

    parser = argparse.ArgumentParser(
        description="Web operations: search, fetch, engines"
    )
    # -V (capital) for version: lowercase -v reserved for future --verbose flag alignment
    parser.add_argument("-V", "--version", action="version", version="4.2.0")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # mcp-stdio
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")

    # search
    p_search = subparsers.add_parser("search", help="Web search via SearXNG")
    p_search.add_argument("query", nargs="?", help="Search query (or pipe via stdin)")
    p_search.add_argument("-c", "--count", type=int, default=10, help="Max results")
    p_search.add_argument("-e", "--engines", help="Specific engines (comma-separated)")
    p_search.add_argument("-C", "--category", help="Category filter")
    p_search.add_argument(
        "-t",
        "--time-range",
        choices=["day", "week", "month", "year"],
        help="Recency filter",
    )
    p_search.add_argument("-p", "--pageno", type=int, default=1, help="Page number")
    p_search.add_argument("-l", "--language", help="Language code (e.g. en, de, all)")
    p_search.add_argument(
        "-s",
        "--safesearch",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Safe search (0/1/2)",
    )
    p_search.add_argument(
        "-f",
        "--format",
        choices=["json", "csv", "rss"],
        default="json",
        help="Output format",
    )

    # engines
    p_engines = subparsers.add_parser(
        "engines", help="List SearXNG engines and categories"
    )
    p_engines.add_argument("-C", "--category", help="Filter by category")
    p_engines.add_argument(
        "-a", "--all", action="store_true", help="Include disabled engines"
    )

    # fetch
    p_fetch = subparsers.add_parser("fetch", help="Fetch URL content")
    p_fetch.add_argument(
        "url", nargs="?", help="URL to fetch (or pipe via stdin, one per line)"
    )
    p_fetch.add_argument(
        "-f", "--format", default="markdown", choices=sorted(FETCH_FORMATS),
        help="Output format (default: markdown)"
    )

    args = parser.parse_args()

    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "search":
            query = args.query
            if not query and not sys.stdin.isatty():
                query = sys.stdin.read().strip()
            assert query, "query required (positional argument or stdin)"
            results, metrics = search_impl(
                query,
                args.count,
                args.engines or None,
                args.category or None,
                args.time_range or None,
                args.pageno if args.pageno > 1 else None,
                args.language or None,
                args.safesearch or None,
                args.format,
            )
            if (
                args.format in ("csv", "rss")
                and metrics.get("format") != "html_fallback"
            ):
                print(results)
            else:
                print(json.dumps({"results": results, "metrics": metrics}, indent=2))
        elif args.command == "engines":
            result, metrics = engines_impl(
                args.category or None,
                not args.all,
            )
            cats = result["categories"]
            engs = result["engines"]
            print(f"Categories ({len(cats)}): {', '.join(cats)}")
            print(f"\nEngines ({len(engs)}):")
            for eng in engs:
                flags = []
                if eng["paging"]:
                    flags.append("paging")
                if eng["time_range"]:
                    flags.append("time_range")
                if eng["safesearch"]:
                    flags.append("safesearch")
                if eng["language"]:
                    flags.append("language")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                cat_str = ", ".join(eng["categories"])
                status = "on" if eng["enabled"] else "OFF"
                print(
                    f"  {eng['shortcut']:8s} {eng['name']:30s} ({cat_str}) {status}{flag_str}"
                )
        elif args.command == "fetch":
            urls = [args.url] if args.url else []
            if not urls and not sys.stdin.isatty():
                urls = [line.strip() for line in sys.stdin if line.strip()]
            assert urls, "url required (positional argument or stdin, one per line)"
            for url in urls:
                result, metrics = fetch_impl(url, format=args.format)
                print(
                    f"# {result['title']}\n\nSource: {result['url']}\n\n{result['content']}"
                )
        else:
            parser.print_help()
    except (AssertionError, Exception) as e:
        _log("ERROR", args.command or "unknown", str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# FASTMCP SERVER — exceptions propagate, FastMCP handles them
# =============================================================================
def _run_mcp():
    from fastmcp import FastMCP

    mcp = FastMCP("web")

    @mcp.tool()
    def search(
        query: str,
        count: int = 10,
        engines: str = "",
        category: str = "",
        time_range: str = "",
        pageno: int = 1,
        language: str = "",
        safesearch: int = 0,
        format: str = "json",
    ) -> str:
        """Web search via SearXNG (aggregates Google, Brave, DDG, etc).

        Falls back to HTML parsing if JSON format not enabled on instance.

        Args:
            query: Search query
            count: Max results (default 10)
            engines: Specific engines comma-separated (e.g. "google,brave") or empty for all
            category: Category filter (general, images, videos, news, files, it, science, social media)
            time_range: Recency filter (day, week, month, year)
            pageno: Page number (default 1)
            language: Language code (e.g. "en", "de", "all")
            safesearch: Safe search level (0=none, 1=moderate, 2=strict)
            format: Output format (json, csv, rss) - default json
        """
        results, metrics = search_impl(
            query,
            count,
            engines or None,
            category or None,
            time_range or None,
            pageno if pageno > 1 else None,
            language or None,
            safesearch or None,
            format,
        )
        if format in ("csv", "rss") and metrics.get("format") != "html_fallback":
            return json.dumps({"raw": results, "metrics": metrics}, indent=2)
        return json.dumps({"results": results, "metrics": metrics}, indent=2)

    @mcp.tool()
    def engines(category: str = "", enabled_only: bool = True) -> str:
        """List available SearXNG engines and categories.

        Args:
            category: Filter engines by category (e.g. "general", "news")
            enabled_only: Only show enabled engines (default True)
        """
        result, metrics = engines_impl(
            category or None,
            enabled_only,
        )
        return json.dumps({"data": result, "metrics": metrics}, indent=2)

    @mcp.tool()
    def fetch(url: str, format: str = "markdown") -> str:
        """Fetch webpage and convert to clean Markdown.

        Removes scripts, styles, nav, footer. Returns readable content.

        Args:
            url: The URL to fetch content from
            format: Output format — "markdown" (default), "text", or "html"
        """
        result, metrics = fetch_impl(url, format=format)
        return f"# {result['title']}\n\nSource: {result['url']}\n\n{result['content']}"

    print("web MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
