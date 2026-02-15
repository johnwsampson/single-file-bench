#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp"]
# ///
"""Read-only file operations: read, inspect, search within files."""

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
EXPOSED = [
    "read",
    "head",
    "tail",
    "cat",
    "list_dir",
    "tree",
    "stats",
    "lc",
    "wc",
    "read_line",
    "read_string",
]

CONFIG = {
    "display_char_limit": 25000,
    "context_lines": 3,
}

SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".exe",
    ".bin",
    ".png",
    ".jpg",
    ".gif",
    ".ico",
    ".zip",
    ".tar",
    ".gz",
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================


# --- Path helpers ---


def _normalize_path(path_str: str) -> Path:
    """Normalize a path string to a resolved Path object."""
    if not path_str:
        return Path.cwd()
    if (
        sys.platform == "win32"
        and path_str.startswith("/")
        and len(path_str) > 2
        and path_str[2] == "/"
    ):
        drive = path_str[1]
        rest = path_str[2:]
        path_str = f"{drive}:{rest}"
    return Path(path_str).expanduser().resolve()


# --- Read functions ---


def _read_impl(
    file_path: str,
    start_line: int = 1,
    end_line: int = -1,
    raw: bool = False,
) -> tuple[bool, str, str]:
    """Read file contents with optional line range.

    Returns (success, content, error).
    """
    start_ms = time.time() * 1000
    path = _normalize_path(file_path)
    if not path.exists():
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "WARN",
            "read",
            file_path,
            metrics=f"latency_ms={latency_ms} status=error",
        )
        return False, "", f"File not found: {file_path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total_lines = len(lines)
        if end_line == -1:
            end_line = total_lines
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, end_line)
        content_lines = lines[start_idx:end_idx]
        if raw:
            result = "".join(content_lines)
        else:
            numbered = [
                f"{i + start_idx + 1:4}: {line}" for i, line in enumerate(content_lines)
            ]
            result = "".join(numbered)
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "read",
            str(path),
            metrics=f"latency_ms={latency_ms} status=success lines={len(content_lines)}",
        )
        return True, result, ""
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "read",
            str(path),
            detail=str(e),
            metrics=f"latency_ms={latency_ms} status=error",
        )
        return False, "", str(e)


def _head_impl(file_path: str, lines: int = 10) -> str:
    """Show first N lines of file.

    Returns formatted string with line numbers and header.
    """
    start_ms = time.time() * 1000
    path = _normalize_path(file_path)
    assert path.exists(), f"File not found: {file_path}"
    assert not path.is_dir(), f"Cannot read directory: {file_path}"
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        file_lines = content.splitlines()
        selected = file_lines[:lines]
        numbered = [f"{i + 1:>4}: {line}" for i, line in enumerate(selected)]
        header = (
            f"=== {path} (first {min(lines, len(file_lines))}"
            f" of {len(file_lines)} lines) ==="
        )
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "head",
            str(path),
            metrics=f"latency_ms={latency_ms} status=success lines={len(selected)}",
        )
        return header + "\n" + "\n".join(numbered)
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "head",
            str(path),
            detail=str(e),
            metrics=f"latency_ms={latency_ms} status=error",
        )
        return f"Error reading file: {e}"


def _tail_impl(file_path: str, lines: int = 10) -> str:
    """Show last N lines of file.

    Returns formatted string with line numbers and header.
    """
    start_ms = time.time() * 1000
    path = _normalize_path(file_path)
    assert path.exists(), f"File not found: {file_path}"
    assert not path.is_dir(), f"Cannot read directory: {file_path}"
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        file_lines = content.splitlines()
        total = len(file_lines)
        start_idx = max(0, total - lines)
        selected = file_lines[start_idx:]
        numbered = [
            f"{start_idx + i + 1:>4}: {line}" for i, line in enumerate(selected)
        ]
        header = f"=== {path} (last {len(selected)} of {total} lines) ==="
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "tail",
            str(path),
            metrics=f"latency_ms={latency_ms} status=success lines={len(selected)}",
        )
        return header + "\n" + "\n".join(numbered)
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "tail",
            str(path),
            detail=str(e),
            metrics=f"latency_ms={latency_ms} status=error",
        )
        return f"Error reading file: {e}"


def _cat_impl(
    file_path: str,
    directory: bool = False,
    extensions: str | None = None,
    recursive: bool = True,
) -> tuple[bool, str, str]:
    """Read file raw, or concatenate all files in a directory.

    When directory=False (default), reads a single file without line numbers.
    When directory=True, reads all files in the directory, concatenated.

    Returns (success, content, error).
    """
    start_ms = time.time() * 1000
    path = _normalize_path(file_path)

    if not directory:
        # Single file mode
        if not path.exists():
            latency_ms = round(time.time() * 1000 - start_ms, 2)
            _log(
                "WARN",
                "cat",
                file_path,
                metrics=f"latency_ms={latency_ms} status=error",
            )
            return False, "", f"File not found: {file_path}"
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            latency_ms = round(time.time() * 1000 - start_ms, 2)
            _log(
                "INFO",
                "cat",
                str(path),
                metrics=f"latency_ms={latency_ms} status=success chars={len(content)}",
            )
            return True, content, ""
        except Exception as e:
            latency_ms = round(time.time() * 1000 - start_ms, 2)
            _log(
                "ERROR",
                "cat",
                str(path),
                detail=str(e),
                metrics=f"latency_ms={latency_ms} status=error",
            )
            return False, "", str(e)

    # Directory mode
    if not path.exists():
        return False, "", f"Directory not found: {file_path}"
    if not path.is_dir():
        return False, "", f"Not a directory: {file_path}"

    ext_list: list[str] | None = None
    if extensions:
        ext_list = [e.strip().lower() for e in extensions.split(",")]
        ext_list = [e if e.startswith(".") else f".{e}" for e in ext_list]

    files_to_read: list[Path] = []
    if recursive:
        for fp in path.rglob("*"):
            if fp.is_file() and fp.suffix.lower() not in SKIP_SUFFIXES:
                if ext_list is None or fp.suffix.lower() in ext_list:
                    files_to_read.append(fp)
    else:
        for fp in path.iterdir():
            if fp.is_file() and fp.suffix.lower() not in SKIP_SUFFIXES:
                if ext_list is None or fp.suffix.lower() in ext_list:
                    files_to_read.append(fp)

    files_to_read.sort()
    output_parts: list[str] = []
    total_chars = 0
    display_limit = CONFIG["display_char_limit"]

    for fp in files_to_read:
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
            rel_path = fp.relative_to(path)
            file_header = f"=== {rel_path} ==="
            file_output = f"{file_header}\n{content}\n"

            if total_chars + len(file_output) > display_limit:
                remaining = len(files_to_read) - len(output_parts)
                output_parts.append(f"\n... truncated ({remaining} files remaining)")
                break

            output_parts.append(file_output)
            total_chars += len(file_output)
        except Exception:
            continue

    if not output_parts:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "cat_dir",
            str(path),
            metrics=f"latency_ms={latency_ms} status=success files=0",
        )
        return True, f"No readable files found in {file_path}", ""

    latency_ms = round(time.time() * 1000 - start_ms, 2)
    _log(
        "INFO",
        "cat_dir",
        str(path),
        metrics=f"latency_ms={latency_ms} status=success files={len(output_parts)}",
    )
    return True, "\n".join(output_parts), ""


def _list_dir_impl(dir_path: str = ".") -> tuple[bool, str, str]:
    """List directory contents with [DIR] prefixes.

    Returns (success, listing, error).
    """
    start_ms = time.time() * 1000
    path = _normalize_path(dir_path)
    if not path.exists():
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "WARN",
            "list_dir",
            dir_path,
            metrics=f"latency_ms={latency_ms} status=error",
        )
        return False, "", f"Directory not found: {dir_path}"
    if not path.is_dir():
        return False, "", f"Not a directory: {dir_path}"
    try:
        items = sorted(os.listdir(path))
        out: list[str] = []
        for item in items:
            item_path = path / item
            prefix = "[DIR] " if item_path.is_dir() else "      "
            out.append(f"{prefix}{item}")
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "list_dir",
            str(path),
            metrics=f"latency_ms={latency_ms} status=success items={len(items)}",
        )
        return True, "\n".join(out), ""
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "list_dir",
            str(path),
            detail=str(e),
            metrics=f"latency_ms={latency_ms} status=error",
        )
        return False, "", str(e)


def _tree_impl(dir_path: str = ".", max_depth: int = 2) -> tuple[bool, str, str]:
    """Build directory tree as JSON, up to max_depth levels.

    Returns (success, tree_json, error).
    """
    start_ms = time.time() * 1000
    root_path = _normalize_path(dir_path)
    if not root_path.exists():
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "WARN",
            "tree",
            dir_path,
            metrics=f"latency_ms={latency_ms} status=error",
        )
        return False, "", f"Path not found: {dir_path}"

    def _build_tree(dp: Path, current_depth: int) -> dict[str, Any] | str:
        if current_depth > max_depth:
            return "..."
        tree: dict[str, Any] = {}
        try:
            items = sorted(
                os.listdir(dp),
                key=lambda x: (not os.path.isdir(dp / x), x.lower()),
            )
            for item in items:
                if item.startswith("."):
                    continue
                item_path = dp / item
                if item_path.is_dir():
                    tree[item + "/"] = _build_tree(item_path, current_depth + 1)
                else:
                    tree[item] = "file"
        except Exception as e:
            return f"Error: {e}"
        return tree

    try:
        tree_data = {root_path.name + "/": _build_tree(root_path, 1)}
        result = json.dumps(tree_data, indent=2)
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "tree",
            str(root_path),
            metrics=f"latency_ms={latency_ms} status=success",
        )
        return True, result, ""
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "tree",
            str(root_path),
            detail=str(e),
            metrics=f"latency_ms={latency_ms} status=error",
        )
        return False, "", str(e)


def _stats_impl(file_path: str) -> tuple[bool, str, str]:
    """Get file or directory metadata (size, dates, permissions).

    Returns (success, stats_json, error).
    """
    start_ms = time.time() * 1000
    path = _normalize_path(file_path)
    if not path.exists():
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "WARN",
            "stats",
            file_path,
            metrics=f"latency_ms={latency_ms} status=error",
        )
        return False, "", f"Path not found: {file_path}"
    try:
        stat = path.stat()
        data = {
            "path": str(path),
            "type": "directory" if path.is_dir() else "file",
            "size_bytes": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:],
        }
        result = json.dumps(data, indent=2)
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "stats",
            str(path),
            metrics=f"latency_ms={latency_ms} status=success",
        )
        return True, result, ""
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "stats",
            str(path),
            detail=str(e),
            metrics=f"latency_ms={latency_ms} status=error",
        )
        return False, "", str(e)


def _lc_impl(file_path: str) -> str:
    """Count lines in a file.

    Returns line count as string, or error message.
    """
    start_ms = time.time() * 1000
    path = _normalize_path(file_path)
    assert path.exists(), f"File not found: {file_path}"
    assert not path.is_dir(), f"Cannot count lines in directory: {file_path}"
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        count = len(content.splitlines())
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "lc",
            str(path),
            metrics=f"latency_ms={latency_ms} status=success count={count}",
        )
        return str(count)
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "lc",
            str(path),
            detail=str(e),
            metrics=f"latency_ms={latency_ms} status=error",
        )
        return f"Error: {e}"


def _wc_impl(file_path: str) -> str:
    """Count lines, words, and characters in a file.

    Returns formatted count string, or error message.
    """
    start_ms = time.time() * 1000
    path = _normalize_path(file_path)
    assert path.exists(), f"File not found: {file_path}"
    assert not path.is_dir(), f"Cannot count in directory: {file_path}"
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        line_count = len(content.splitlines())
        word_count = len(content.split())
        char_count = len(content)
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "wc",
            str(path),
            metrics=(
                f"latency_ms={latency_ms} status=success"
                f" lines={line_count} words={word_count} chars={char_count}"
            ),
        )
        return (
            f"{line_count:>8} lines\n"
            f"{word_count:>8} words\n"
            f"{char_count:>8} chars\n"
            f"{path}"
        )
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "wc",
            str(path),
            detail=str(e),
            metrics=f"latency_ms={latency_ms} status=error",
        )
        return f"Error: {e}"


def _read_line_impl(file_path: str, line_no: int, context: int = 3) -> str:
    """Show specific line with surrounding context lines.

    Returns formatted output with markers on the target line.
    """
    start_ms = time.time() * 1000
    path = _normalize_path(file_path)
    assert path.exists(), f"File not found: {file_path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        assert 1 <= line_no <= total, f"Line {line_no} out of range (1-{total})"
        start = max(1, line_no - context)
        end = min(total, line_no + context)
        out = [f"{path}:{line_no}", ""]
        for i in range(start, end + 1):
            marker = ">>>" if i == line_no else "   "
            out.append(f"{marker} {i:>4}|{lines[i - 1].rstrip()}")
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "read_line",
            f"{path}:{line_no}",
            metrics=f"latency_ms={latency_ms} status=success",
        )
        return "\n".join(out)
    except AssertionError:
        raise
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "read_line",
            f"{path}:{line_no}",
            detail=str(e),
            metrics=f"latency_ms={latency_ms} status=error",
        )
        return f"Error reading line: {e}"


def _read_string_impl(file_path: str, pattern: str) -> str:
    """Show lines in file matching a text pattern.

    Returns formatted matches with line numbers.
    """
    start_ms = time.time() * 1000
    path = _normalize_path(file_path)
    assert path.exists(), f"File not found: {file_path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        matches: list[str] = []
        for i, line in enumerate(lines, 1):
            if pattern in line:
                matches.append(f"{i:>4}: {line.rstrip()}")
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        if not matches:
            _log(
                "INFO",
                "read_string",
                f"{path} pattern={pattern}",
                metrics=f"latency_ms={latency_ms} status=success matches=0",
            )
            return f"No matches for '{pattern}' in {file_path}"
        _log(
            "INFO",
            "read_string",
            f"{path} pattern={pattern}",
            metrics=f"latency_ms={latency_ms} status=success matches={len(matches)}",
        )
        return f"{path}: {len(matches)} match(es)\n\n" + "\n".join(matches)
    except AssertionError:
        raise
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "read_string",
            f"{path} pattern={pattern}",
            detail=str(e),
            metrics=f"latency_ms={latency_ms} status=error",
        )
        return f"Error searching file: {e}"


# =============================================================================
# CLI INTERFACE
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Read-only file operations: read, inspect, search within files"
    )
    # -V (capital) for version: lowercase -v reserved for future --verbose flag alignment
    parser.add_argument("-V", "--version", action="version", version="1.1.0")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # --- mcp-stdio ---
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")

    # --- read ---
    p_read = subparsers.add_parser("read", help="Read file with line numbers")
    p_read.add_argument("file_path", nargs="?", default=None)
    p_read.add_argument("-s", "--start", type=int, default=1)
    p_read.add_argument("-e", "--end", type=int, default=-1)
    p_read.add_argument("-r", "--raw", action="store_true", help="No line numbers")

    # --- head ---
    p_head = subparsers.add_parser("head", help="First N lines of file")
    p_head.add_argument("file_path", nargs="?", default=None)
    p_head.add_argument("-n", "--lines", type=int, default=10)

    # --- tail ---
    p_tail = subparsers.add_parser("tail", help="Last N lines of file")
    p_tail.add_argument("file_path", nargs="?", default=None)
    p_tail.add_argument("-n", "--lines", type=int, default=10)

    # --- cat ---
    p_cat = subparsers.add_parser("cat", help="Raw read file or concatenate directory")
    p_cat.add_argument("file_path", nargs="?", default=None)
    p_cat.add_argument(
        "-d",
        "--dir",
        action="store_true",
        dest="directory",
        help="Read all files in directory",
    )
    p_cat.add_argument(
        "-x",
        "--ext",
        dest="extensions",
        default=None,
        help="Filter by extensions (comma-separated)",
    )
    p_cat.add_argument(
        "-R",
        "--recursive",
        action="store_true",
        default=True,
        help="Include subdirectories (default: True, use -R to be explicit)",
    )

    # --- list / list-dir ---
    p_list = subparsers.add_parser("list", help="List directory contents")
    p_list.add_argument("path", nargs="?", default=".")
    p_list_dir = subparsers.add_parser(
        "list-dir", help="List directory contents (alias)"
    )
    p_list_dir.add_argument("path", nargs="?", default=".")

    # --- tree ---
    p_tree = subparsers.add_parser("tree", help="Directory tree")
    p_tree.add_argument("path", nargs="?", default=".")
    p_tree.add_argument("-d", "--depth", type=int, default=2)

    # --- stats ---
    p_stats = subparsers.add_parser("stats", help="File/directory metadata")
    p_stats.add_argument("path", nargs="?", default=None)

    # --- lc ---
    p_lc = subparsers.add_parser("lc", help="Line count")
    p_lc.add_argument("path", nargs="?", default=None)

    # --- wc ---
    p_wc = subparsers.add_parser("wc", help="Word/char count")
    p_wc.add_argument("path", nargs="?", default=None)

    # --- read-line ---
    p_rl = subparsers.add_parser("read-line", help="Show specific line with context")
    p_rl.add_argument("file_path", nargs="?", default=None)
    p_rl.add_argument("line_no", nargs="?", type=int, default=None)
    p_rl.add_argument("-c", "--context", type=int, default=3)

    # --- read-string ---
    p_rs = subparsers.add_parser("read-string", help="Show lines matching pattern")
    p_rs.add_argument("file_path", nargs="?", default=None)
    p_rs.add_argument("pattern", nargs="?", default=None)

    args = parser.parse_args()

    # --- Dispatch ---
    try:
        if args.command == "mcp-stdio":
            _run_mcp()

        elif args.command == "read":
            if not args.file_path and not sys.stdin.isatty():
                args.file_path = sys.stdin.read().strip()
            assert args.file_path, (
                "file_path required. Usage: sfa_file_read.py read <file>"
            )
            success, content, error = _read_impl(
                args.file_path, args.start, args.end, args.raw
            )
            if success:
                print(content, end="")
            else:
                print(f"Error: {error}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "head":
            if not args.file_path and not sys.stdin.isatty():
                args.file_path = sys.stdin.read().strip()
            assert args.file_path, (
                "file_path required. Usage: sfa_file_read.py head <file>"
            )
            print(_head_impl(args.file_path, args.lines))

        elif args.command == "tail":
            if not args.file_path and not sys.stdin.isatty():
                args.file_path = sys.stdin.read().strip()
            assert args.file_path, (
                "file_path required. Usage: sfa_file_read.py tail <file>"
            )
            print(_tail_impl(args.file_path, args.lines))

        elif args.command == "cat":
            if not args.file_path and not sys.stdin.isatty():
                args.file_path = sys.stdin.read().strip()
            assert args.file_path, (
                "file_path required. Usage: sfa_file_read.py cat <file>"
            )
            success, content, error = _cat_impl(
                args.file_path, args.directory, args.extensions, args.recursive
            )
            if success:
                print(content, end="")
            else:
                print(f"Error: {error}", file=sys.stderr)
                sys.exit(1)

        elif args.command in ("list", "list-dir"):
            path = args.path
            if path == "." and not sys.stdin.isatty():
                stdin_val = sys.stdin.read().strip()
                if stdin_val:
                    path = stdin_val
            success, listing, error = _list_dir_impl(path)
            if success:
                print(listing)
            else:
                print(f"Error: {error}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "tree":
            path = args.path
            if path == "." and not sys.stdin.isatty():
                stdin_val = sys.stdin.read().strip()
                if stdin_val:
                    path = stdin_val
            success, tree_out, error = _tree_impl(path, args.depth)
            if success:
                print(tree_out)
            else:
                print(f"Error: {error}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "stats":
            if not args.path and not sys.stdin.isatty():
                args.path = sys.stdin.read().strip()
            assert args.path, "path required. Usage: sfa_file_read.py stats <path>"
            success, stats_out, error = _stats_impl(args.path)
            if success:
                print(stats_out)
            else:
                print(f"Error: {error}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "lc":
            if not args.path and not sys.stdin.isatty():
                args.path = sys.stdin.read().strip()
            assert args.path, "path required. Usage: sfa_file_read.py lc <path>"
            print(_lc_impl(args.path))

        elif args.command == "wc":
            if not args.path and not sys.stdin.isatty():
                args.path = sys.stdin.read().strip()
            assert args.path, "path required. Usage: sfa_file_read.py wc <path>"
            print(_wc_impl(args.path))

        elif args.command == "read-line":
            if not args.file_path and not sys.stdin.isatty():
                args.file_path = sys.stdin.read().strip()
            assert args.file_path, (
                "file_path required. Usage: sfa_file_read.py read-line <file> <line_no>"
            )
            assert args.line_no is not None, (
                "line_no required. Usage: sfa_file_read.py read-line <file> <line_no>"
            )
            print(_read_line_impl(args.file_path, args.line_no, args.context))

        elif args.command == "read-string":
            if not args.file_path and not sys.stdin.isatty():
                args.file_path = sys.stdin.read().strip()
            assert args.file_path, (
                "file_path required. Usage: sfa_file_read.py read-string <file> <pattern>"
            )
            assert args.pattern, (
                "pattern required. Usage: sfa_file_read.py read-string <file> <pattern>"
            )
            print(_read_string_impl(args.file_path, args.pattern))

        else:
            parser.print_help()
    except (AssertionError, Exception) as e:
        _log("ERROR", args.command or "unknown", str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# FASTMCP SERVER
# =============================================================================
def _run_mcp():
    from fastmcp import FastMCP

    mcp = FastMCP("fr")

    @mcp.tool()
    def read(path: str, start: int = 1, end: int = -1, raw: bool = False) -> str:
        """Read file contents with optional line range.

        Args:
            path: File path to read
            start: Start line (1-indexed, default: 1)
            end: End line (default: -1 for entire file)
            raw: If true, omit line numbers (default: false)
        """
        success, content, error = _read_impl(path, start, end, raw)
        return content if success else f"ERROR: {error}"

    @mcp.tool()
    def head(path: str, lines: int = 10) -> str:
        """Show first N lines of a file.

        Args:
            path: File path to read
            lines: Number of lines to show (default: 10)
        """
        return _head_impl(path, lines)

    @mcp.tool()
    def tail(path: str, lines: int = 10) -> str:
        """Show last N lines of a file.

        Args:
            path: File path to read
            lines: Number of lines to show (default: 10)
        """
        return _tail_impl(path, lines)

    @mcp.tool()
    def cat(path: str) -> str:
        """Read file contents raw (no line numbers).

        Args:
            path: File path to read
        """
        success, content, error = _cat_impl(path)
        return content if success else f"ERROR: {error}"

    @mcp.tool()
    def list_dir(path: str = ".") -> str:
        """List directory contents.

        Args:
            path: Directory path (default: current directory)
        """
        success, listing, error = _list_dir_impl(path)
        return listing if success else f"ERROR: {error}"

    @mcp.tool()
    def tree(path: str = ".", depth: int = 2) -> str:
        """Show directory tree as JSON.

        Args:
            path: Directory path (default: current directory)
            depth: Maximum depth to traverse (default: 2)
        """
        success, tree_out, error = _tree_impl(path, depth)
        return tree_out if success else f"ERROR: {error}"

    @mcp.tool()
    def stats(path: str) -> str:
        """Get file or directory metadata (size, dates, permissions).

        Args:
            path: File or directory path
        """
        success, stats_out, error = _stats_impl(path)
        return stats_out if success else f"ERROR: {error}"

    @mcp.tool()
    def lc(path: str) -> str:
        """Count lines in a file.

        Args:
            path: File path to count
        """
        return _lc_impl(path)

    @mcp.tool()
    def wc(path: str) -> str:
        """Count lines, words, and characters in a file.

        Args:
            path: File path to count
        """
        return _wc_impl(path)

    @mcp.tool()
    def read_line(path: str, line_no: int, context: int = 3) -> str:
        """Show specific line with surrounding context.

        Args:
            path: File path to read
            line_no: Line number to show (1-indexed)
            context: Number of context lines above and below (default: 3)
        """
        return _read_line_impl(path, line_no, context)

    @mcp.tool()
    def read_string(path: str, pattern: str) -> str:
        """Show lines in file matching a text pattern.

        Args:
            path: File path to search
            pattern: Text pattern to match
        """
        return _read_string_impl(path, pattern)

    print("fr MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
