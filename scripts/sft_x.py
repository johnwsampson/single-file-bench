#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp", "xai-sdk"]
# ///
"""X/Twitter search and web search via xAI Grok API.

Usage:
    sft_x.py search "query" [options]
    sft_x.py web-search "query" [options]
    sft_x.py user handle [options]
    sft_x.py post url [options]
    sft_x.py mcp-stdio
"""

import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# LOGGING
# =============================================================================
_LEVELS = {"TRACE": 5, "DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40, "FATAL": 50}
_THRESHOLD = _LEVELS.get(os.environ.get("SFA_LOG_LEVEL", "INFO"), 20)
_LOG_DIR = os.environ.get("SFA_LOG_DIR", "")
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
EXPOSED = ["search", "web-search", "user", "post"]  # CLI + MCP — both interfaces

CONFIG = {
    "default_model": "grok-4-1-fast",
    "request_timeout_seconds": 60.0,
}

# Auto-load secrets from script directory
_secrets_path = Path(__file__).parent / ".secrets"
if _secrets_path.is_file():
    with open(_secrets_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line.startswith("export ") and "=" in _line:
                _key, _val = _line[7:].split("=", 1)
                os.environ.setdefault(_key, _val)

XAI_API_KEY = os.getenv("XAI_API_KEY", "")


# =============================================================================
# CORE FUNCTIONS
# =============================================================================


def _build_params(**kwargs):
    """Build params dict, excluding None values."""
    return {k: v for k, v in kwargs.items() if v is not None}


def _extract_usage(response):
    """Extract usage info from response.

    Reasoning tokens are optional (only present for reasoning models).
    All other fields are guaranteed by xAI SDK response contract.
    """
    assert hasattr(response, "usage"), (
        f"Response has usage attribute (got {dir(response)})"
    )
    if not response.usage:
        return {}

    usage = response.usage
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": getattr(
            usage, "reasoning_tokens", 0
        ),  # Optional for non-reasoning models
        "total_tokens": usage.total_tokens,
    }


@contextmanager
def _grok_client():
    """Context manager for Grok client lifecycle."""
    from xai_sdk import Client

    assert XAI_API_KEY, "XAI_API_KEY is set in environment"
    client = Client(api_key=XAI_API_KEY)
    try:
        yield client
    finally:
        client.close()


def _x_search_impl(
    query: str,
    model: str = "grok-4-1-fast",
    handles: str | None = None,
    exclude_handles: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    include_citations: bool = True,
) -> dict:
    """Search X/Twitter via Grok.

    CLI: search
    MCP: x_search

    Propagates exceptions — caller handles display.
    """
    from xai_sdk.tools import x_search as xai_x_search
    from xai_sdk.chat import user

    # Parse handles
    allowed_handles = handles.split(",") if handles else None
    excluded_handles = exclude_handles.split(",") if exclude_handles else None

    assert not (allowed_handles and excluded_handles), (
        "Only one of handles or exclude_handles can be specified (got both)"
    )

    # Parse dates (format: YYYY-MM-DD)
    parsed_from = datetime.strptime(from_date, "%Y-%m-%d") if from_date else None
    parsed_to = datetime.strptime(to_date, "%Y-%m-%d") if to_date else None

    start_ms = time.time() * 1000
    _log(
        "INFO",
        "x_search_start",
        f"query={query}",
        detail=f"model={model} handles={handles}",
    )

    with _grok_client() as client:
        tool_params = _build_params(
            allowed_x_handles=allowed_handles,
            excluded_x_handles=excluded_handles,
            from_date=parsed_from,
            to_date=parsed_to,
        )

        include_options = ["inline_citations"] if include_citations else []

        chat_params = {"model": model, "tools": [xai_x_search(**tool_params)]}
        if include_options:
            chat_params["include"] = include_options

        chat = client.chat.create(**chat_params)
        chat.append(user(query))
        response = chat.sample()

        latency_ms = time.time() * 1000 - start_ms

        # xAI SDK guarantees response.content exists
        assert hasattr(response, "content"), (
            f"Response has content attribute (got {dir(response)})"
        )

        result = {
            "content": response.content,
            "citations": list(response.citations) if response.citations else [],
            "metrics": {
                "query": query,
                "model": model,
                "latency_ms": round(latency_ms, 2),
                "status": "success",
                "usage": _extract_usage(response),
            },
        }

        # inline_citations are optional — only present if requested
        if (
            include_citations
            and hasattr(response, "inline_citations")
            and response.inline_citations
        ):
            result["inline_citations"] = [
                {
                    "id": c.id,
                    "url": c.x_citation.url
                    if c.HasField("x_citation")
                    else getattr(c, "web_citation", {}).get(
                        "url", ""
                    ),  # web_citation fallback for mixed results
                }
                for c in response.inline_citations
            ]

        _log(
            "INFO",
            "x_search_complete",
            f"query={query}",
            detail=f"model={model}",
            metrics=f"latency_ms={round(latency_ms, 2)} status=success",
        )
        return result


def _web_search_impl(
    query: str,
    model: str = "grok-4-1-fast",
    domains: str | None = None,
    exclude_domains: str | None = None,
    include_citations: bool = True,
) -> dict:
    """Web search via Grok.

    CLI: web-search
    MCP: grok_web_search

    Propagates exceptions — caller handles display.
    """
    from xai_sdk.tools import web_search as xai_web_search
    from xai_sdk.chat import user

    # Parse domains
    allowed_domains = domains.split(",") if domains else None
    excluded_domains = exclude_domains.split(",") if exclude_domains else None

    assert not (allowed_domains and excluded_domains), (
        "Only one of domains or exclude_domains can be specified (got both)"
    )

    start_ms = time.time() * 1000
    _log(
        "INFO",
        "web_search_start",
        f"query={query}",
        detail=f"model={model} domains={domains}",
    )

    with _grok_client() as client:
        tool_params = _build_params(
            allowed_domains=allowed_domains,
            excluded_domains=excluded_domains,
        )

        include_options = ["inline_citations"] if include_citations else []

        chat_params = {"model": model, "tools": [xai_web_search(**tool_params)]}
        if include_options:
            chat_params["include"] = include_options

        chat = client.chat.create(**chat_params)
        chat.append(user(query))
        response = chat.sample()

        latency_ms = time.time() * 1000 - start_ms

        assert hasattr(response, "content"), (
            f"Response has content attribute (got {dir(response)})"
        )

        result = {
            "content": response.content,
            "citations": list(response.citations) if response.citations else [],
            "metrics": {
                "query": query,
                "model": model,
                "latency_ms": round(latency_ms, 2),
                "status": "success",
                "usage": _extract_usage(response),
            },
        }

        if (
            include_citations
            and hasattr(response, "inline_citations")
            and response.inline_citations
        ):
            result["inline_citations"] = [
                {
                    "id": c.id,
                    "url": c.web_citation.url if c.HasField("web_citation") else "",
                }
                for c in response.inline_citations
            ]

        _log(
            "INFO",
            "web_search_complete",
            f"query={query}",
            detail=f"model={model}",
            metrics=f"latency_ms={round(latency_ms, 2)} status=success",
        )
        return result


def _x_user_impl(
    handle: str,
    model: str = "grok-4-1-fast",
    count: int = 10,
) -> dict:
    """Get recent posts from a specific X user.

    CLI: user
    MCP: x_user

    Propagates exceptions — caller handles display.
    """
    from xai_sdk.tools import x_search as xai_x_search
    from xai_sdk.chat import user

    # Normalize handle (remove @ if present)
    handle = handle.lstrip("@")

    start_ms = time.time() * 1000
    _log(
        "INFO",
        "x_user_start",
        f"handle=@{handle}",
        detail=f"model={model} count={count}",
    )

    with _grok_client() as client:
        tool_params = _build_params(allowed_x_handles=[handle])

        chat_params = {
            "model": model,
            "tools": [xai_x_search(**tool_params)],
            "include": ["inline_citations"],
        }

        chat = client.chat.create(**chat_params)
        query = f"Get the {count} most recent posts from @{handle}. List each post with its date and content."
        chat.append(user(query))
        response = chat.sample()

        latency_ms = time.time() * 1000 - start_ms

        assert hasattr(response, "content"), (
            f"Response has content attribute (got {dir(response)})"
        )

        result = {
            "handle": handle,
            "content": response.content,
            "citations": list(response.citations) if response.citations else [],
            "metrics": {
                "model": model,
                "latency_ms": round(latency_ms, 2),
                "status": "success",
                "usage": _extract_usage(response),
            },
        }

        if hasattr(response, "inline_citations") and response.inline_citations:
            result["posts"] = [
                {
                    "id": c.id,
                    "url": c.x_citation.url if c.HasField("x_citation") else "",
                }
                for c in response.inline_citations
            ]

        _log(
            "INFO",
            "x_user_complete",
            f"handle=@{handle}",
            detail=f"model={model}",
            metrics=f"latency_ms={round(latency_ms, 2)} status=success",
        )
        return result


def _x_post_impl(
    url: str,
    model: str = "grok-4-1-fast",
) -> dict:
    """Get a specific X post by URL.

    CLI: post
    MCP: x_post

    Propagates exceptions — caller handles display.
    """
    from xai_sdk.tools import x_search as xai_x_search
    from xai_sdk.chat import user

    start_ms = time.time() * 1000
    _log("INFO", "x_post_start", f"url={url}", detail=f"model={model}")

    with _grok_client() as client:
        chat_params = {
            "model": model,
            "tools": [xai_x_search()],
            "include": ["inline_citations"],
        }

        chat = client.chat.create(**chat_params)
        query = f"Find and quote the exact content of this X post: {url}. Include the author, date, and full text."
        chat.append(user(query))
        response = chat.sample()

        latency_ms = time.time() * 1000 - start_ms

        assert hasattr(response, "content"), (
            f"Response has content attribute (got {dir(response)})"
        )

        result = {
            "url": url,
            "content": response.content,
            "citations": list(response.citations) if response.citations else [],
            "metrics": {
                "model": model,
                "latency_ms": round(latency_ms, 2),
                "status": "success",
                "usage": _extract_usage(response),
            },
        }

        if hasattr(response, "inline_citations") and response.inline_citations:
            result["inline_citations"] = [
                {
                    "id": c.id,
                    "url": c.x_citation.url if c.HasField("x_citation") else "",
                }
                for c in response.inline_citations
            ]

        _log(
            "INFO",
            "x_post_complete",
            f"url={url}",
            detail=f"model={model}",
            metrics=f"latency_ms={round(latency_ms, 2)} status=success",
        )
        return result


# =============================================================================
# CLI INTERFACE
# =============================================================================
def main():
    import argparse

    _log("INFO", "start", "Command started")

    parser = argparse.ArgumentParser(description="X/Twitter search via xAI Grok API")
    # -V (capital) for version: lowercase -v reserved for future --verbose flag alignment
    parser.add_argument("-V", "--version", action="version", version="2.1.0")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # mcp-stdio
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")

    # CLI for _x_search_impl
    p_search = subparsers.add_parser("search", help="Search X/Twitter")
    p_search.add_argument("query", nargs="?", help="Search query (or pipe via stdin)")
    p_search.add_argument("-m", "--model", default="grok-4-1-fast", help="Model")
    p_search.add_argument("-H", "--handles", help="Include handles (comma-separated)")
    p_search.add_argument(
        "-X", "--exclude-handles", help="Exclude handles (comma-separated)"
    )
    p_search.add_argument(
        "-f", "--from", dest="from_date", help="From date (YYYY-MM-DD)"
    )
    p_search.add_argument("-t", "--to", dest="to_date", help="To date (YYYY-MM-DD)")

    # CLI for _web_search_impl
    p_web = subparsers.add_parser("web-search", help="Web search via Grok")
    p_web.add_argument("query", nargs="?", help="Search query (or pipe via stdin)")
    p_web.add_argument("-m", "--model", default="grok-4-1-fast", help="Model")
    p_web.add_argument("-d", "--domains", help="Include domains (comma-separated)")
    p_web.add_argument(
        "-X", "--exclude-domains", help="Exclude domains (comma-separated)"
    )

    # CLI for _x_user_impl
    p_user = subparsers.add_parser("user", help="Get recent posts from X user")
    p_user.add_argument(
        "handle", nargs="?", help="X handle (with or without @, or pipe via stdin)"
    )
    p_user.add_argument("-m", "--model", default="grok-4-1-fast", help="Model")
    p_user.add_argument("-c", "--count", type=int, default=10, help="Number of posts")

    # CLI for _x_post_impl
    p_post = subparsers.add_parser("post", help="Get specific X post by URL")
    p_post.add_argument("url", nargs="?", help="X post URL (or pipe via stdin)")
    p_post.add_argument("-m", "--model", default="grok-4-1-fast", help="Model")

    args = parser.parse_args()

    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "search":
            query = args.query
            if not query and not sys.stdin.isatty():
                query = sys.stdin.read().strip()
            assert query, "query required (positional argument or stdin)"
            result = _x_search_impl(
                query,
                args.model,
                args.handles or None,
                args.exclude_handles or None,
                args.from_date or None,
                args.to_date or None,
            )
            print(json.dumps(result, indent=2))
        elif args.command == "web-search":
            query = args.query
            if not query and not sys.stdin.isatty():
                query = sys.stdin.read().strip()
            assert query, "query required (positional argument or stdin)"
            result = _web_search_impl(
                query,
                args.model,
                args.domains or None,
                args.exclude_domains or None,
            )
            print(json.dumps(result, indent=2))
        elif args.command == "user":
            handle = args.handle
            if not handle and not sys.stdin.isatty():
                handle = sys.stdin.read().strip()
            assert handle, "handle required (positional argument or stdin)"
            result = _x_user_impl(handle, args.model, args.count)
            print(json.dumps(result, indent=2))
        elif args.command == "post":
            url = args.url
            if not url and not sys.stdin.isatty():
                url = sys.stdin.read().strip()
            assert url, "url required (positional argument or stdin)"
            result = _x_post_impl(url, args.model)
            print(json.dumps(result, indent=2))
        else:
            parser.print_help()
    except (AssertionError, Exception) as e:
        _log("ERROR", args.command or "unknown", str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# FASTMCP SERVER (lazy — only loaded when mcp-stdio is invoked)
# =============================================================================
def _run_mcp():
    from fastmcp import FastMCP

    mcp = FastMCP("x")

    # MCP for _x_search_impl
    @mcp.tool()
    def search(
        query: str,
        model: str = "grok-4-1-fast",
        handles: str = "",
        exclude_handles: str = "",
        from_date: str = "",
        to_date: str = "",
    ) -> str:
        """Search X/Twitter via Grok's live search.

        Args:
            query: Search query (natural language)
            model: Grok model (default grok-4-1-fast)
            handles: Comma-separated X handles to include (max 10)
            exclude_handles: Comma-separated X handles to exclude (max 10)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns synthesized answer with citations from X posts.
        """
        result = _x_search_impl(
            query,
            model,
            handles if handles else None,
            exclude_handles if exclude_handles else None,
            from_date if from_date else None,
            to_date if to_date else None,
        )
        return json.dumps(result, indent=2)

    # MCP for _web_search_impl
    @mcp.tool()
    def web_search(
        query: str,
        model: str = "grok-4-1-fast",
        domains: str = "",
        exclude_domains: str = "",
    ) -> str:
        """Web search via Grok (alternative to SearXNG with AI synthesis).

        Args:
            query: Search query (natural language)
            model: Grok model (default grok-4-1-fast)
            domains: Comma-separated domains to include (max 5)
            exclude_domains: Comma-separated domains to exclude (max 5)

        Returns synthesized answer with citations.
        """
        result = _web_search_impl(
            query,
            model,
            domains if domains else None,
            exclude_domains if exclude_domains else None,
        )
        return json.dumps(result, indent=2)

    # MCP for _x_user_impl
    @mcp.tool()
    def user(
        handle: str,
        model: str = "grok-4-1-fast",
        count: int = 10,
    ) -> str:
        """Get recent posts from a specific X user.

        Args:
            handle: X handle (with or without @)
            model: Grok model (default grok-4-1-fast)
            count: Number of recent posts to fetch (default 10)

        Returns recent posts with dates, content, and URLs.
        """
        result = _x_user_impl(handle, model, count)
        return json.dumps(result, indent=2)

    # MCP for _x_post_impl
    @mcp.tool()
    def post(
        url: str,
        model: str = "grok-4-1-fast",
    ) -> str:
        """Get a specific X post by URL.

        Args:
            url: Full X/Twitter post URL
            model: Grok model (default grok-4-1-fast)

        Returns the post content, author, and date.
        """
        result = _x_post_impl(url, model)
        return json.dumps(result, indent=2)

    print("x MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
