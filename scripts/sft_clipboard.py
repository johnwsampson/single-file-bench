#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp"]
# ///
"""Clipboard operations — read and write system clipboard text.

Universal interchange between agents and human. Supports macOS (pbcopy/pbpaste),
Linux X11 (xclip, xsel), and Wayland (wl-clipboard).

Usage:
    sft_clipboard.py read
    sft_clipboard.py write "text to copy"
    echo "piped text" | sft_clipboard.py write
    sft_clipboard.py mcp-stdio
"""

import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# LOGGING (TSV format — see Principle 6)
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
EXPOSED = ["read", "write"]  # CLI + MCP — both interfaces

CLIPBOARD_TIMEOUT_SECONDS = 5


# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def _get_clip_tool() -> tuple[list[str], list[str]]:
    """Detect available clipboard tool. Returns (read_cmd, write_cmd).

    Priority: macOS pbcopy/pbpaste > xclip > xsel > wl-clipboard.
    """
    if platform.system() == "Darwin":
        assert shutil.which("pbpaste"), "macOS pbpaste on PATH"
        return ["pbpaste"], ["pbcopy"]

    if shutil.which("xclip"):
        return (
            ["xclip", "-selection", "clipboard", "-o"],
            ["xclip", "-selection", "clipboard", "-i"],
        )
    if shutil.which("xsel"):
        return (
            ["xsel", "--clipboard", "--output"],
            ["xsel", "--clipboard", "--input"],
        )
    if shutil.which("wl-paste"):
        return ["wl-paste"], ["wl-copy"]

    assert False, "Clipboard tool available (install xclip, xsel, or wl-clipboard)"


def _read_impl() -> tuple[str, dict]:
    """Read current clipboard contents.

    CLI: read
    MCP: read

    Returns (clipboard_text, metrics).
    """
    start_ms = time.time() * 1000
    read_cmd, _ = _get_clip_tool()
    result = subprocess.run(
        read_cmd, capture_output=True, text=True, timeout=CLIPBOARD_TIMEOUT_SECONDS
    )
    assert result.returncode == 0, (
        f"Clipboard read succeeds (exit {result.returncode}: {result.stderr.strip()})"
    )
    text = result.stdout
    latency_ms = round(time.time() * 1000 - start_ms, 2)
    _log(
        "INFO",
        "read_complete",
        f"Read {len(text)} chars from clipboard",
        detail=f"chars={len(text)}",
        metrics=f"latency_ms={latency_ms} status=success",
    )
    return text, {"status": "success", "chars": len(text), "latency_ms": latency_ms}


def _write_impl(text: str) -> tuple[str, dict]:
    """Write text to system clipboard.

    CLI: write
    MCP: write

    Args:
        text: Content to write to clipboard.

    Returns (confirmation_message, metrics).
    """
    start_ms = time.time() * 1000
    _, write_cmd = _get_clip_tool()
    result = subprocess.run(
        write_cmd,
        input=text,
        capture_output=True,
        text=True,
        timeout=CLIPBOARD_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, (
        f"Clipboard write succeeds (exit {result.returncode}: {result.stderr.strip()})"
    )
    msg = f"Wrote {len(text)} chars to clipboard"
    latency_ms = round(time.time() * 1000 - start_ms, 2)
    _log(
        "INFO",
        "write_complete",
        msg,
        detail=f"chars={len(text)}",
        metrics=f"latency_ms={latency_ms} status=success",
    )
    return msg, {"status": "success", "chars": len(text), "latency_ms": latency_ms}


# =============================================================================
# CLI INTERFACE
# =============================================================================
def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Clipboard operations — read/write system clipboard text"
    )
    parser.add_argument("-V", "--version", action="version", version="1.1.0")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # mcp-stdio
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")

    # CLI for _read_impl
    subparsers.add_parser("read", help="Read clipboard contents")

    # CLI for _write_impl
    p_write = subparsers.add_parser("write", help="Write text to clipboard")
    p_write.add_argument("text", nargs="?", help="Text to write (or stdin if omitted)")

    args = parser.parse_args()

    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "read":
            text, metrics = _read_impl()
            print(text, end="")
        elif args.command == "write":
            text = args.text
            if not text and not sys.stdin.isatty():
                text = sys.stdin.read()
            assert text is not None, "text required (positional argument or stdin)"
            msg, metrics = _write_impl(text)
            print(json.dumps({"message": msg, "metrics": metrics}, indent=2))
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
    import json

    from fastmcp import FastMCP

    mcp = FastMCP("clip")

    # MCP for _read_impl
    @mcp.tool()
    def read() -> str:
        """Read current clipboard contents.

        Returns the text currently in the system clipboard.

        Args:
            (no arguments)
        """
        text, metrics = _read_impl()
        return text

    # MCP for _write_impl
    @mcp.tool()
    def write(text: str) -> str:
        """Write text to system clipboard.

        Args:
            text: Content to write to clipboard
        """
        msg, metrics = _write_impl(text)
        return json.dumps({"message": msg, "metrics": metrics}, indent=2)

    print("clip MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
