#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.25.0",
#     "fastmcp",
# ]
# ///
"""Search code across GitHub repositories via Grep.app.

Searches across 1M+ GitHub repositories using the Grep.app MCP service.
Supports regex patterns, language filtering, and repository filtering.

Usage:
    sft_grepcode.py search "pattern" [options]
    sft_grepcode.py mcp-stdio
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


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
EXPOSED = ["search"]  # CLI + MCP — both interfaces

CONFIG = {
    "endpoint": "https://mcp.grep.app",
    "timeout_seconds": 30.0,
    "version": "1.0.0",
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def _search_impl(
    query: str,
    use_regexp: bool = False,
    match_case: bool = False,
    match_whole_words: bool = False,
    repo: str | None = None,
    path: str | None = None,
    languages: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Search GitHub repos via Grep.app MCP server.
    
    CLI: search
    MCP: search
    
    Returns (result_dict, metrics_dict).
    """
    start_ms = time.time() * 1000
    
    # Build tool arguments
    tool_args = {"query": query}
    if use_regexp:
        tool_args["useRegexp"] = True
    if match_case:
        tool_args["matchCase"] = True
    if match_whole_words:
        tool_args["matchWholeWords"] = True
    if repo:
        tool_args["repo"] = repo
    if path:
        tool_args["path"] = path
    if languages:
        tool_args["language"] = languages
    
    # MCP JSON-RPC request
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "searchGitHub",
            "arguments": tool_args,
        },
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    
    try:
        resp = httpx.post(
            CONFIG["endpoint"],
            headers=headers,
            json=payload,
            timeout=CONFIG["timeout_seconds"],
        )
        resp.raise_for_status()
        
        # Parse SSE response
        raw = resp.text
        result_data = None
        
        for line in raw.strip().split("\n"):
            if line.startswith("data: "):
                json_str = line[6:]
                data = json.loads(json_str)
                if "result" in data:
                    result_data = data["result"]
                elif "error" in data:
                    raise Exception(data["error"].get("message", "Unknown MCP error"))
        
        if not result_data:
            raise Exception("No result in MCP response")
        
        # Extract content from MCP tool result
        content = result_data.get("content", [])
        text_content = ""
        for item in content:
            if item.get("type") == "text":
                text_content = item.get("text", "")
                break
        
        latency_ms = time.time() * 1000 - start_ms
        
        # Parse the text content into structured results
        results = _parse_grep_results(text_content)
        
        metrics = {
            "provider": "grep.app",
            "query": query,
            "use_regexp": use_regexp,
            "languages": languages or [],
            "repo": repo,
            "result_count": len(results),
            "latency_ms": round(latency_ms, 2),
            "status": "success",
            "cost_usd": 0.0,
        }
        
        return {"results": results, "raw": text_content}, metrics
    
    except Exception as e:
        latency_ms = time.time() * 1000 - start_ms
        metrics = {
            "provider": "grep.app",
            "query": query,
            "latency_ms": round(latency_ms, 2),
            "status": "error",
            "error": str(e),
        }
        return {"error": str(e)}, metrics


def _parse_grep_results(text: str) -> list[dict[str, Any]]:
    """Parse grep.app text results into structured data."""
    results = []
    current = {}
    in_snippets = False
    snippet_lines = []
    
    for line in text.split("\n"):
        line_stripped = line.strip()
        
        if line_stripped.startswith("Repository:"):
            # Save previous result
            if current:
                if snippet_lines:
                    current["snippets"] = "\n".join(snippet_lines)
                results.append(current)
            current = {"repository": line_stripped[12:].strip()}
            snippet_lines = []
            in_snippets = False
        
        elif line_stripped.startswith("Path:"):
            current["path"] = line_stripped[5:].strip()
        
        elif line_stripped.startswith("URL:"):
            current["url"] = line_stripped[4:].strip()
        
        elif line_stripped.startswith("License:"):
            current["license"] = line_stripped[8:].strip()
        
        elif line_stripped.startswith("Snippets:"):
            in_snippets = True
        
        elif line_stripped.startswith("--- Snippet"):
            if snippet_lines:
                snippet_lines.append("")  # Separator between snippets
            snippet_lines.append(line_stripped)
        
        elif in_snippets and line:
            snippet_lines.append(line)
    
    # Don't forget the last result
    if current:
        if snippet_lines:
            current["snippets"] = "\n".join(snippet_lines)
        results.append(current)
    
    return results


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Search GitHub repositories via Grep.app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sft_repo_grep.py search "def main" --language python
  sft_repo_grep.py search "useState" --repo facebook/react
  sft_repo_grep.py search "TODO.*FIXME" --regex
        """,
    )
    
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {CONFIG['version']}",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search GitHub repos")
    search_parser.add_argument("query", help="Code pattern to search for")
    search_parser.add_argument(
        "-r", "--regex", action="store_true", help="Treat query as regex"
    )
    search_parser.add_argument(
        "-c", "--case", action="store_true", help="Case sensitive search"
    )
    search_parser.add_argument(
        "-w", "--whole-words", action="store_true", help="Match whole words only"
    )
    search_parser.add_argument(
        "-R", "--repo", help="Filter by repository (e.g., 'vercel/', 'facebook/react')"
    )
    search_parser.add_argument(
        "-p", "--path", help="Filter by file path (e.g., '/components/', 'README.md')"
    )
    search_parser.add_argument(
        "-l", "--language", action="append", help="Filter by language (can repeat)"
    )
    search_parser.add_argument(
        "-o", "--output", help="Output file (default: stdout)"
    )
    search_parser.add_argument(
        "-j", "--json", action="store_true", help="Output as JSON"
    )
    search_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress metrics output"
    )
    
    # MCP server command
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")
    
    args = parser.parse_args()
    
    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "search":
            # Get query from stdin if not provided and stdin is piped
            query = args.query
            if not query and not sys.stdin.isatty():
                query = sys.stdin.read().strip()
            
            assert query, "query required (positional argument or stdin)"
            
            result, metrics = _search_impl(
                query=query,
                use_regexp=args.regex,
                match_case=args.case,
                match_whole_words=args.whole_words,
                repo=args.repo,
                path=args.path,
                languages=args.language,
            )
            
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
            
            if args.json:
                output_text = json.dumps(result, indent=2)
            elif "error" in result:
                output_text = f"Error: {result['error']}"
            else:
                output_text = result.get("raw", "")
            
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(output_text)
                if not args.quiet:
                    print(f"Output written to {args.output}", file=sys.stderr)
            else:
                print(output_text)
            
            _log("INFO", "search", f"query='{query}' results={metrics.get('result_count', 0)}", metrics=json.dumps(metrics))
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
    
    mcp = FastMCP("repo_grep")
    
    @mcp.tool()
    def search(
        query: str,
        use_regexp: bool = False,
        match_case: bool = False,
        match_whole_words: bool = False,
        repo: str = "",
        path: str = "",
        languages: str = "",
    ) -> str:
        """Search GitHub repositories via Grep.app for code patterns.
        
        Searches across 1M+ GitHub repositories using the Grep.app MCP service.
        
        Args:
            query: Code pattern to search for (required)
            use_regexp: Treat query as regex pattern (default: False)
            match_case: Enable case-sensitive search (default: False)
            match_whole_words: Match whole words only (default: False)
            repo: Filter by repository (e.g., "facebook/react", "vercel/")
            path: Filter by file path (e.g., "/components/", "README.md")
            languages: Comma-separated language filters (e.g., "python,typescript")
        
        Returns:
            JSON string with results and metadata
        """
        lang_list = [l.strip() for l in languages.split(",") if l.strip()] if languages else None
        
        result, metrics = _search_impl(
            query=query,
            use_regexp=use_regexp,
            match_case=match_case,
            match_whole_words=match_whole_words,
            repo=repo if repo else None,
            path=path if path else None,
            languages=lang_list,
        )
        
        response = {
            "results": result.get("results", []),
            "metrics": metrics,
        }
        
        if "error" in result:
            response["error"] = result["error"]
        
        return json.dumps(response, indent=2)
    
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
