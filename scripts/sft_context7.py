#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.25.0",
#     "fastmcp",
# ]
# ///
"""Fetch up-to-date library documentation via Context7 API.

Search for libraries and retrieve version-specific documentation.
Useful for getting accurate API references for frameworks like Next.js, React, etc.

Usage:
    sft_context7.py search "nextjs"
    sft_context7.py docs /vercel/next.js "middleware authentication"
    sft_context7.py mcp-stdio

Environment:
    CONTEXT7_API_KEY - Required API key for Context7
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
EXPOSED = ["search", "docs"]

CONFIG = {
    "version": "1.0.0",
    "base_url": "https://context7.com/api/v2",
    "timeout_seconds": 30.0,
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def _get_api_key() -> str:
    """Get API key from .secrets file or environment."""
    key = os.environ.get("CONTEXT7_API_KEY")
    if not key:
        secrets_path = Path(__file__).parent / ".secrets"
        if secrets_path.exists():
            for line in secrets_path.read_text().splitlines():
                if line.startswith("export CONTEXT7_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise Exception("Missing CONTEXT7_API_KEY — set in environment or scripts/.secrets")
    return key


def _search_impl(query: str, topic: str | None = None) -> tuple[dict, dict]:
    """Search for libraries by name. CLI: search, MCP: search."""
    import httpx
    
    start_ms = time.time() * 1000
    api_key = _get_api_key()
    
    params = {"libraryName": query}
    if topic:
        params["query"] = topic
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        resp = httpx.get(
            f"{CONFIG['base_url']}/libs/search",
            params=params,
            headers=headers,
            timeout=CONFIG["timeout_seconds"],
        )
        resp.raise_for_status()
        data = resp.json()
        
        latency_ms = time.time() * 1000 - start_ms
        results = data.get("results", [])
        
        metrics = {
            "provider": "context7",
            "query": query,
            "result_count": len(results),
            "latency_ms": round(latency_ms, 2),
            "status": "success",
        }
        
        return {"results": results}, metrics
        
    except Exception as e:
        latency_ms = time.time() * 1000 - start_ms
        metrics = {
            "provider": "context7",
            "query": query,
            "latency_ms": round(latency_ms, 2),
            "status": "error",
            "error": str(e),
        }
        return {"error": str(e)}, metrics


def _docs_impl(library_id: str, query: str, tokens: int | None = None) -> tuple[dict, dict]:
    """Get documentation for a library. CLI: docs, MCP: docs."""
    import httpx
    
    start_ms = time.time() * 1000
    api_key = _get_api_key()
    
    params = {
        "libraryId": library_id,
        "query": query,
    }
    if tokens:
        params["tokens"] = tokens
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        resp = httpx.get(
            f"{CONFIG['base_url']}/context",
            params=params,
            headers=headers,
            timeout=CONFIG["timeout_seconds"],
        )
        
        # Handle special status codes
        if resp.status_code == 202:
            return {
                "status": "indexing",
                "message": "Library is being indexed, try again later",
            }, {
                "provider": "context7",
                "library_id": library_id,
                "status": "indexing",
            }
        elif resp.status_code == 301:
            redirect = resp.json().get("redirectUrl", "unknown")
            return {"status": "redirect", "message": f"Library moved to: {redirect}"}, {
                "provider": "context7",
                "library_id": library_id,
                "status": "redirect",
            }
        
        resp.raise_for_status()
        content = resp.text
        
        latency_ms = time.time() * 1000 - start_ms
        
        metrics = {
            "provider": "context7",
            "library_id": library_id,
            "query": query,
            "content_length": len(content),
            "latency_ms": round(latency_ms, 2),
            "status": "success",
        }
        
        return {"content": content}, metrics
        
    except Exception as e:
        latency_ms = time.time() * 1000 - start_ms
        metrics = {
            "provider": "context7",
            "library_id": library_id,
            "latency_ms": round(latency_ms, 2),
            "status": "error",
            "error": str(e),
        }
        return {"error": str(e)}, metrics


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch library documentation via Context7 API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sft_context7.py search "nextjs"
  sft_context7.py search "react" --topic hooks
  sft_context7.py docs /vercel/next.js "middleware authentication"
  sft_context7.py docs /facebook/react "useEffect" --tokens 2000
        """,
    )
    
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {CONFIG['version']}",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # search
    p_search = subparsers.add_parser("search", help="Search for libraries")
    p_search.add_argument("query", help="Library name to search for")
    p_search.add_argument("-t", "--topic", help="Topic to filter results")
    p_search.add_argument("-o", "--output", help="Output file")
    p_search.add_argument("-q", "--quiet", action="store_true", help="Suppress metrics")
    
    # docs
    p_docs = subparsers.add_parser("docs", help="Get library documentation")
    p_docs.add_argument("library_id", help="Library ID (e.g., /vercel/next.js)")
    p_docs.add_argument("query", help="Documentation query")
    p_docs.add_argument("-n", "--tokens", type=int, help="Max tokens in response")
    p_docs.add_argument("-o", "--output", help="Output file")
    p_docs.add_argument("-q", "--quiet", action="store_true", help="Suppress metrics")
    
    # MCP server
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")
    
    args = parser.parse_args()
    
    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "search":
            query = args.query
            if not query and not sys.stdin.isatty():
                query = sys.stdin.read().strip()
            assert query, "query required (positional argument or stdin)"
            
            result, metrics = _search_impl(query, args.topic)
            
            if not args.quiet:
                status = metrics.get("status", "unknown")
                if status == "success":
                    print(
                        f"[{_SCRIPT}] {metrics['latency_ms']:.0f}ms, {metrics['result_count']} results",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[{_SCRIPT}] ERROR: {metrics.get('error', 'unknown')}",
                        file=sys.stderr,
                    )
            
            if "error" in result:
                output_text = f"Error: {result['error']}"
            else:
                lines = []
                for lib in result.get("results", []):
                    lib_id = lib.get("id", "unknown")
                    title = lib.get("title", lib.get("name", "unknown"))
                    desc = lib.get("description", "")[:100]
                    lines.append(f"- **{title}** (`{lib_id}`)")
                    if desc:
                        lines.append(f"  {desc}")
                output_text = "\n".join(lines) if lines else "No results found."
            
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(output_text)
                if not args.quiet:
                    print(f"Output written to {args.output}", file=sys.stderr)
            else:
                print(output_text)
            
            _log("INFO", "search", f"query={query} results={metrics.get('result_count', 0)}", metrics=json.dumps(metrics))
            
        elif args.command == "docs":
            result, metrics = _docs_impl(args.library_id, args.query, args.tokens)
            
            if not args.quiet:
                status = metrics.get("status", "unknown")
                if status == "success":
                    print(
                        f"[{_SCRIPT}] {metrics['latency_ms']:.0f}ms, {metrics['content_length']} chars",
                        file=sys.stderr,
                    )
                elif status in ("indexing", "redirect"):
                    print(f"[{_SCRIPT}] {result.get('message', status)}", file=sys.stderr)
                else:
                    print(
                        f"[{_SCRIPT}] ERROR: {metrics.get('error', 'unknown')}",
                        file=sys.stderr,
                    )
            
            if "error" in result:
                output_text = f"Error: {result['error']}"
            elif "status" in result:
                output_text = result.get("message", result["status"])
            else:
                output_text = result.get("content", "")
            
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(output_text)
                if not args.quiet:
                    print(f"Output written to {args.output}", file=sys.stderr)
            else:
                print(output_text)
            
            _log("INFO", "docs", f"library={args.library_id} status={metrics.get('status')}", metrics=json.dumps(metrics))
        else:
            parser.print_help()
    except AssertionError as e:
        _log("ERROR", "contract_violation", str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        _log("ERROR", "runtime_error", str(e))
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# FASTMCP SERVER
# =============================================================================

def _run_mcp():
    """Run as MCP server."""
    from fastmcp import FastMCP
    
    mcp = FastMCP("context7")
    
    @mcp.tool()
    def search(query: str, topic: str = "") -> str:
        """Search for libraries by name.
        
        Args:
            query: Library name to search for (e.g., "nextjs", "react")
            topic: Optional topic to filter results
        
        Returns:
            JSON with library results and metadata
        """
        result, metrics = _search_impl(query, topic if topic else None)
        return json.dumps({"result": result, "metrics": metrics}, indent=2)
    
    @mcp.tool()
    def docs(library_id: str, query: str, tokens: int = 0) -> str:
        """Get documentation context for a library.
        
        Args:
            library_id: Library ID (e.g., "/vercel/next.js", "/facebook/react")
            query: Documentation query (e.g., "middleware authentication")
            tokens: Max tokens in response (0 = default)
        
        Returns:
            JSON with documentation content and metadata
        """
        tokens_val = tokens if tokens > 0 else None
        result, metrics = _docs_impl(library_id, query, tokens_val)
        return json.dumps({"result": result, "metrics": metrics}, indent=2)
    
    print("Context7 MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
