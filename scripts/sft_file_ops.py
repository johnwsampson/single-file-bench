#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp"]
# ///
"""File operations — find, patch, check workflow with recon tools.

RULE: Never read a file for editing.

Workflow:
    1. find  — find the patch target (search with context)
    2. dry   — dry run the patch (preview what will change)
    3. patch — execute the patch
    4. check — confirm the patch landed

Usage:
    sft_file_ops.py find <pattern> [path] [--context N] [--include glob]
    sft_file_ops.py dry <file_glob> <start> <end> <old> <new>
    sft_file_ops.py patch <file_glob> <start> <end> <old> <new>
    sft_file_ops.py check <file> [--expected text]
    sft_file_ops.py mcp-stdio
"""

import fnmatch
import json
import os
import re
import shutil
import subprocess
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
EXPOSED = [
    "find",
    "dry",
    "patch",
    "check",
    "scout",
    "search",
    "read",
    "tail",
    "cat",
    "tree",
    "view",
    "count",
    "locate",
    "write",
    "create",
    "append",
    "cp",
    "mv",
    "rm",
    "recover",
    "splice",
]

BACKUP_DIR = Path.home() / ".sfb" / "backups"
FO_SPLICE_FILE = Path(os.environ.get("FO_SPLICE_FILE", "/tmp/fo_splice.tmp"))

SKIP_SUFFIXES = {
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin",
    ".png", ".jpg", ".gif", ".ico", ".zip", ".tar", ".gz",
}

GREP_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".yml", ".yaml",
    ".js", ".ts", ".html", ".css", ".sh", ".toml",
}


# =============================================================================
# PATH HELPERS
# =============================================================================


def _normalize_path(path_str: str) -> Path:
    """Normalize a path string to a resolved Path object."""
    if not path_str:
        return Path.cwd()
    return Path(path_str).expanduser().resolve()


def _get_tool_path(tool_name: str, env_var: str) -> str | None:
    """Get path to external tool, checking env override first."""
    override = os.environ.get(env_var)
    if override and os.path.exists(override):
        return override
    return shutil.which(tool_name)


def _run_command(args: list[str], cwd: Path | None = None, timeout: int = 30) -> str:
    """Run external command and return stdout."""
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        _log("WARN", "timeout", f"{timeout}s, cmd={' '.join(str(a) for a in args[:3])}")
        return ""
    except subprocess.CalledProcessError:
        return ""


# =============================================================================
# BACKUP HELPERS (git-aware)
# =============================================================================


def _create_backup(file_path: Path) -> Path | None:
    """Create backup of file before modification. Returns backup path or None."""
    if not file_path.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"{file_path.name}.{ts}.backup"
    shutil.copy2(file_path, backup_path)
    return backup_path


def _is_git_clean(file_path: Path) -> bool:
    """Check if file matches git HEAD."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(file_path)],
            cwd=file_path.parent,
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _backup_if_dirty(file_path: Path) -> Path | None:
    """Backup file only if it has uncommitted changes."""
    if file_path.exists() and not _is_git_clean(file_path):
        return _create_backup(file_path)
    return None


# =============================================================================
# GLOB RESOLUTION
# =============================================================================


def _resolve_glob(file_glob: str) -> list[Path]:
    """Resolve a file glob to a list of existing file paths."""
    path = Path(file_glob).expanduser()
    # If it's an exact path, return it directly
    if path.exists() and path.is_file():
        return [path.resolve()]
    # Try as glob from cwd
    results = sorted(Path.cwd().glob(file_glob))
    return [p.resolve() for p in results if p.is_file()]


# =============================================================================
# CORE FUNCTIONS
# =============================================================================
# WORKFLOW — find, dry, patch, check


def _find_target_impl(
    pattern: str,
    path: str = ".",
    context: int = 0,
    include: str = "",
) -> str:
    """Find patch target — search with context to locate the text to edit.

    Returns matching lines with optional surrounding context.
    This is step 1: find the exact text you'll replace.
    """
    start_ms = time.time() * 1000
    root_path = _normalize_path(path)

    if root_path.is_file():
        cwd_path = root_path.parent
        target_arg: str | None = root_path.name
    else:
        cwd_path = root_path
        target_arg = None

    results: list[str] = []

    # Strategy 1: ripgrep
    rg_path = _get_tool_path("rg", "SFB_RG_PATH")
    if rg_path:
        cmd = [rg_path, "--line-number", "--no-heading", pattern]
        if include:
            cmd.extend(["--glob", include])
        if context > 0:
            cmd.extend(["-C", str(context)])
        if target_arg:
            cmd.append(target_arg)
        output = _run_command(cmd, cwd=cwd_path)
        if output:
            results = output.splitlines()
            latency_ms = round(time.time() * 1000 - start_ms, 2)
            _log("INFO", "find", f"rg: {pattern}", metrics=f"latency_ms={latency_ms} matches={len(results)}")
            return "\n".join(results)

    # Strategy 2: git grep
    git_path = _get_tool_path("git", "SFB_GIT_PATH")
    if git_path and (cwd_path / ".git").exists():
        cmd = [git_path, "grep", "-n", pattern]
        if context > 0:
            cmd.extend(["-C", str(context)])
        if target_arg:
            cmd.extend(["--", target_arg])
        elif include:
            cmd.extend(["--", include])
        output = _run_command(cmd, cwd=cwd_path)
        if output:
            results = output.splitlines()
            latency_ms = round(time.time() * 1000 - start_ms, 2)
            _log("INFO", "find", f"git: {pattern}", metrics=f"latency_ms={latency_ms} matches={len(results)}")
            return "\n".join(results)

    # Strategy 3: Python fallback
    EXCLUDE_DIRS = {
        ".git", "__pycache__", ".scripts_hold", "node_modules",
        "venv", ".venv", "env", "build", "dist",
    }
    if root_path.is_file():
        files_to_search = [root_path]
    else:
        files_to_search = []
        for root, dirs, files in os.walk(root_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
            for filename in files:
                if include and not fnmatch.fnmatch(filename, include):
                    continue
                if Path(filename).suffix in GREP_EXTENSIONS:
                    files_to_search.append(Path(root) / filename)

    for file_path in files_to_search:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_lines = f.readlines()
            for i, line in enumerate(file_lines):
                if pattern in line:
                    if context > 0:
                        start_idx = max(0, i - context)
                        end_idx = min(len(file_lines), i + context + 1)
                        for j in range(start_idx, end_idx):
                            prefix = f"{file_path}:{j + 1}:"
                            marker = ">" if j == i else " "
                            results.append(f"{marker}{prefix}{file_lines[j].rstrip()}")
                        results.append("--")
                    else:
                        results.append(f"{file_path}:{i + 1}:{line.rstrip()}")
        except Exception:
            continue

    latency_ms = round(time.time() * 1000 - start_ms, 2)
    _log("INFO", "find", f"python: {pattern}", metrics=f"latency_ms={latency_ms} matches={len(results)}")

    if not results:
        return f"No matches for '{pattern}'"
    return "\n".join(results)


def _patch_impl(
    file_glob: str,
    context_start: int,
    context_end: int,
    old: str,
    new: str,
    dry_run: bool = False,
) -> str:
    """Patch files — find old string within context range, replace with new.

    When dry_run=True, shows preview without modifying files.
    Supports glob patterns to patch across multiple files.

    Args:
        file_glob: File path or glob pattern
        context_start: Start line of target context (1-based)
        context_end: End line of target context (inclusive)
        old: Text to find within the context range
        new: Replacement text
        dry_run: Preview only, don't modify
    """
    start_ms = time.time() * 1000
    files = _resolve_glob(file_glob)
    assert files, f"No files match: {file_glob}"

    all_results: list[str] = []
    action = "DRY" if dry_run else "PATCH"

    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            all_results.append(f"[FAIL] {file_path}: {e}")
            continue

        lines = content.splitlines(keepends=True)
        total = len(lines)
        start_idx = max(0, context_start - 1)
        end_idx = min(total, context_end)

        # Extract the context range
        context_text = "".join(lines[start_idx:end_idx])

        if old not in context_text:
            all_results.append(f"[SKIP] {file_path}: old string not found in lines {context_start}-{context_end}")
            continue

        # Count matches within context
        match_count = context_text.count(old)
        new_context = context_text.replace(old, new)

        if dry_run:
            # Show diff preview
            out = [
                f"--- {file_path}",
                f"+++ {file_path}",
                f"@@ lines {context_start}-{context_end} ({match_count} match(es)) @@",
            ]
            # Show removed lines
            for line in context_text.splitlines():
                out.append(f"-{line}")
            # Show added lines
            for line in new_context.splitlines():
                out.append(f"+{line}")
            out.append(f"[DRY] {match_count} replacement(s) ready. Run patch to apply.")
            all_results.append("\n".join(out))
        else:
            # Execute the patch
            backup_path = _backup_if_dirty(file_path)
            new_lines = lines[:start_idx] + [new_context] + lines[end_idx:]
            new_content = "".join(new_lines)
            file_path.write_text(new_content, encoding="utf-8")

            # BDA: verify write
            verify = file_path.read_text(encoding="utf-8")
            if new in verify:
                status = "verified"
            else:
                status = "WRITE OK but new text not found in readback"

            out = [f"[OK] {file_path}: {match_count} replacement(s) [{status}]"]
            if backup_path:
                out.append(f"     Backup: {backup_path}")

            old_snippet = old[:40] + "..." if len(old) > 40 else old
            new_snippet = new[:40] + "..." if len(new) > 40 else new
            out.append(f"     '{old_snippet}' -> '{new_snippet}'")
            all_results.append("\n".join(out))

    latency_ms = round(time.time() * 1000 - start_ms, 2)
    _log("INFO", action.lower(), f"{file_glob} L{context_start}-{context_end}",
         metrics=f"latency_ms={latency_ms} files={len(files)}")

    return "\n\n".join(all_results)


def _splice_impl(
    file_path: str,
    start_line: int = 0,
    end_line: int = 0,
    start_str: str = "",
    stop_str: str = "",
    dry_run: bool = False,
    replace_all: bool = False,
) -> str:
    """Splice content from FO_SPLICE_FILE into a target file.

    Three targeting modes from two optional pairs:
    - Lines only: replace everything between start_line and end_line (inclusive)
    - Strings only: find start_str..stop_str anchors, replace between them (inclusive)
    - Both: find anchors only within the line range

    New content always comes from FO_SPLICE_FILE (set via env var).
    Write content there first (e.g. via heredoc), then call splice.

    Args:
        file_path: Target file to splice into
        start_line: Start line of target range (1-based, 0 = not set)
        end_line: End line of target range (inclusive, 0 = not set)
        start_str: Start anchor string (empty = not set)
        stop_str: Stop anchor string (empty = not set)
        dry_run: Preview only, don't modify
        replace_all: Allow multiple anchor matches (strings-only mode)
    """
    start_ms = time.time() * 1000
    action = "DRY-SPLICE" if dry_run else "SPLICE"
    fpath = _normalize_path(file_path)
    assert fpath.is_file(), f"Target file not found: {fpath}"

    # Read new content from splice file
    assert FO_SPLICE_FILE.is_file(), f"Splice file not found: {FO_SPLICE_FILE}. Write content there first."
    new_content = FO_SPLICE_FILE.read_text(encoding="utf-8")

    # Read target file
    content = fpath.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines(keepends=True)
    total = len(lines)

    has_lines = start_line > 0 and end_line > 0
    has_strings = bool(start_str) and bool(stop_str)
    assert has_lines or has_strings, "Must provide --start/-s and --end/-e (lines), or --start-str and --stop-str (anchors), or both"

    replacements: list[tuple[int, int]] = []  # list of (start_idx, end_idx) to replace

    if has_lines and not has_strings:
        # Mode 1: Lines only — replace the range
        s_idx = max(0, start_line - 1)
        e_idx = min(total, end_line)
        assert s_idx < e_idx, f"Invalid line range: {start_line}-{end_line}"
        replacements.append((s_idx, e_idx))

    elif has_strings and not has_lines:
        # Mode 2: Strings only — find all anchor pairs in whole file
        search_text = content
        offset = 0
        while True:
            s_pos = search_text.find(start_str, offset)
            if s_pos == -1:
                break
            e_pos = search_text.find(stop_str, s_pos + len(start_str))
            if e_pos == -1:
                break
            e_pos += len(stop_str)
            # Convert char positions to line indices
            s_line = content[:s_pos].count("\n")
            # Find the end of the line containing stop_str
            end_of_stop = search_text.find("\n", e_pos)
            if end_of_stop == -1:
                e_line = total
            else:
                e_line = content[:end_of_stop].count("\n") + 1
            replacements.append((s_line, e_line))
            offset = e_pos
        assert replacements, f"Anchor pair not found: start=\"{start_str}\" stop=\"{stop_str}\""
        if len(replacements) > 1 and not replace_all:
            assert False, f"Multiple anchor matches ({len(replacements)}). Use --all to replace all, or add line range to narrow scope."

    else:
        # Mode 3: Both — find anchors within line range
        s_idx = max(0, start_line - 1)
        e_idx = min(total, end_line)
        assert s_idx < e_idx, f"Invalid line range: {start_line}-{end_line}"
        scoped_text = "".join(lines[s_idx:e_idx])
        s_pos = scoped_text.find(start_str)
        assert s_pos != -1, f"Start anchor \"{start_str}\" not found in lines {start_line}-{end_line}"
        e_pos = scoped_text.find(stop_str, s_pos + len(start_str))
        assert e_pos != -1, f"Stop anchor \"{stop_str}\" not found after start anchor in lines {start_line}-{end_line}"
        e_pos += len(stop_str)
        # Convert to absolute line indices within the scoped range
        rel_s_line = scoped_text[:s_pos].count("\n")
        end_of_stop = scoped_text.find("\n", e_pos)
        if end_of_stop == -1:
            rel_e_line = e_idx - s_idx
        else:
            rel_e_line = scoped_text[:end_of_stop].count("\n") + 1
        replacements.append((s_idx + rel_s_line, s_idx + rel_e_line))

    # Ensure new content ends with newline for clean splicing
    if new_content and not new_content.endswith("\n"):
        new_content += "\n"

    # Apply replacements in reverse order (so indices stay valid)
    replacements.sort(reverse=True)

    if dry_run:
        out = [f"--- {fpath}", f"+++ {fpath}"]
        for s_idx, e_idx in sorted(replacements):
            old_text = "".join(lines[s_idx:e_idx])
            out.append(f"@@ lines {s_idx+1}-{e_idx} ({len(replacements)} replacement(s)) @@")
            for line in old_text.splitlines():
                out.append(f"-{line}")
            for line in new_content.splitlines():
                out.append(f"+{line}")
        out.append(f"[DRY-SPLICE] {len(replacements)} region(s) ready. Run without --dry to apply.")
        result_text = "\n".join(out)
    else:
        backup_path = _backup_if_dirty(fpath)
        for s_idx, e_idx in replacements:
            new_lines_list = new_content.splitlines(keepends=True)
            lines = lines[:s_idx] + new_lines_list + lines[e_idx:]
        new_file_content = "".join(lines)
        fpath.write_text(new_file_content, encoding="utf-8")

        # BDA: verify
        verify = fpath.read_text(encoding="utf-8")
        # Check that new content appears (strip trailing newline for matching)
        check_str = new_content.rstrip("\n")
        if check_str in verify:
            status = "verified"
        else:
            status = "WRITE OK but new content not found in readback"

        out = [f"[OK] {fpath}: {len(replacements)} region(s) spliced [{status}]"]
        if backup_path:
            out.append(f"     Backup: {backup_path}")
        for s_idx, e_idx in sorted(replacements, reverse=True):
            out.append(f"     Replaced lines {s_idx+1}-{e_idx}")
        result_text = "\n".join(out)

    latency_ms = round(time.time() * 1000 - start_ms, 2)
    _log("INFO", action.lower(), f"{file_path} regions={len(replacements)}",
         metrics=f"latency_ms={latency_ms}")

    return result_text


def _check_impl(
    file_path: str,
    context_start: int = 0,
    context_end: int = 0,
    expected: str = "",
) -> str:
    """Check patch outcome — verify expected content at location.

    Without expected text, shows the content at the location.
    With expected text, confirms presence or reports mismatch.
    """
    start_ms = time.time() * 1000
    path = _normalize_path(file_path)
    assert path.exists(), f"File not found: {file_path}"
    assert path.is_file(), f"Not a file: {file_path}"

    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    total = len(lines)

    # If no line range specified, check whole file
    if context_start == 0 and context_end == 0:
        if expected:
            if expected in content:
                for i, line in enumerate(lines, 1):
                    if expected in line:
                        latency_ms = round(time.time() * 1000 - start_ms, 2)
                        _log("INFO", "check", f"{path} expected=found", metrics=f"latency_ms={latency_ms}")
                        return f"[OK] '{expected}' found in {path}\n>>> {i:>4}|{line}"
            latency_ms = round(time.time() * 1000 - start_ms, 2)
            _log("WARN", "check", f"{path} expected=not_found", metrics=f"latency_ms={latency_ms}")
            return f"[FAIL] '{expected}' not found in {path}"
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log("INFO", "check", str(path), metrics=f"latency_ms={latency_ms}")
        return f"[OK] {path}\n  Lines: {total}\n  Size: {path.stat().st_size} bytes"

    # Line range specified
    start_idx = max(0, context_start - 1)
    end_idx = min(total, context_end)
    target_lines = lines[start_idx:end_idx]
    target_content = "\n".join(target_lines)

    if expected:
        if expected in target_content:
            out = [f"[OK] '{expected}' found at {path}:{context_start}-{context_end}", ""]
            for i, line in enumerate(target_lines, context_start):
                marker = ">>>" if expected in line else "   "
                out.append(f"{marker} {i:>4}|{line}")
            latency_ms = round(time.time() * 1000 - start_ms, 2)
            _log("INFO", "check", f"{path}:{context_start}-{context_end} expected=found", metrics=f"latency_ms={latency_ms}")
            return "\n".join(out)
        else:
            out = [f"[FAIL] '{expected}' not in {path}:{context_start}-{context_end}", "", "Actual:"]
            for i, line in enumerate(target_lines, context_start):
                out.append(f"    {i:>4}|{line}")
            latency_ms = round(time.time() * 1000 - start_ms, 2)
            _log("WARN", "check", f"{path}:{context_start}-{context_end} expected=not_found", metrics=f"latency_ms={latency_ms}")
            return "\n".join(out)

    # No expected — just show the content
    out = [f"[OK] {path}:{context_start}-{context_end}", ""]
    for i, line in enumerate(target_lines, context_start):
        out.append(f"    {i:>4}|{line}")
    latency_ms = round(time.time() * 1000 - start_ms, 2)
    _log("INFO", "check", f"{path}:{context_start}-{context_end}", metrics=f"latency_ms={latency_ms}")
    return "\n".join(out)


# =============================================================================
# RECON: Non-destructive intelligence gathering
# =============================================================================


def _search_impl(
    pattern: str,
    path: str = ".",
    context: int = 0,
    include: str = "",
) -> str:
    """Cross-codebase content search. Alias for find with search semantics."""
    return _find_target_impl(pattern, path, context=context, include=include)


def _scout_impl(coords: str, pattern: str = "") -> str:
    """Unified code intelligence — structural recon via pattern routing.

    Patterns: functions, classes, imports, docstrings, comments, todos,
    decorators, types, errors, tests, main, header, exports, section:MARKER
    Without pattern: shows file structure overview.
    """
    path = _normalize_path(coords.split(":")[0])
    assert path.exists(), f"Path not found: {coords}"

    if not pattern:
        if path.is_dir():
            return f"Directory: {path}\nUse a pattern to search, or specify a file."
        return _scout_file_structure(path)

    lower = pattern.lower()
    if path.is_dir():
        return f"{lower} mode requires a file path"

    dispatch = {
        "functions": _scout_functions, "funcs": _scout_functions, "defs": _scout_functions,
        "classes": _scout_classes, "class": _scout_classes,
        "imports": _scout_imports, "import": _scout_imports,
        "docstrings": _scout_docstrings, "docs": _scout_docstrings,
        "comments": _scout_comments, "comment": _scout_comments,
        "todos": _scout_todos, "todo": _scout_todos, "fixme": _scout_todos,
        "decorators": _scout_decorators, "deco": _scout_decorators,
        "types": _scout_types, "typing": _scout_types,
        "errors": _scout_errors, "exceptions": _scout_errors,
        "tests": _scout_tests, "test": _scout_tests,
        "main": _scout_main, "entry": _scout_main,
        "header": _scout_header, "preamble": _scout_header,
        "exports": _scout_exports, "__all__": _scout_exports,
    }

    if lower in dispatch:
        return dispatch[lower](path)
    if lower.startswith("section:"):
        parts = pattern.split(":", 2)
        end_marker = parts[2] if len(parts) >= 3 else None
        return _scout_section(path, parts[1], end_marker)

    # Fallback: treat as regex search within file
    return _scout_pattern(path, pattern)


def _scout_file_structure(path: Path) -> str:
    """Show file structure with function/class definitions."""
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    out = [f"{path}", f"  Lines: {len(lines)}"]
    defs = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "async def ")):
            indent = len(line) - len(line.lstrip())
            name = stripped.split("(")[0].split(":")[0]
            defs.append((i, indent, name))
    if defs:
        out.append("")
        for line_num, indent, name in defs:
            prefix = "  " * (indent // 4)
            out.append(f"  {prefix}{line_num}: {name}")
    else:
        out.append("  (No classes/functions detected)")
    return "\n".join(out)


def _scout_functions(path: Path) -> str:
    """Extract function/method definitions."""
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    funcs: list[str] = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            indent = len(line) - len(line.lstrip())
            sig = stripped.split("(")[0].replace("def ", "").replace("async ", "").strip()
            prefix = "  " * (indent // 4)
            funcs.append(f"{i:>4}: {prefix}{sig}")
    if funcs:
        return f"{path}\n  Functions: {len(funcs)}\n\n" + "\n".join(funcs)
    return f"{path}\n  (No functions found)"


def _scout_classes(path: Path) -> str:
    """Extract class definitions with methods."""
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    classes: list[tuple[int, str, list[tuple[int, str]]]] = []
    current_class: str | None = None
    current_indent = 0
    current_line = 0
    current_methods: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if stripped.startswith("class "):
            if current_class:
                classes.append((current_line, current_class, current_methods))
                current_methods = []
            class_part = stripped[6:]
            current_class = class_part.split("(")[0].rstrip(":")
            if "(" in class_part:
                inheritance = class_part.split("(")[1].rstrip("):").strip()
                if inheritance:
                    current_class = f"{current_class}({inheritance})"
            current_indent = indent
            current_line = i
        elif current_class and (stripped.startswith("def ") or stripped.startswith("async def ")):
            if indent > current_indent:
                method = stripped.split("(")[0].replace("def ", "").replace("async ", "").strip()
                current_methods.append((i, method))
    if current_class:
        classes.append((current_line, current_class, current_methods))
    if classes:
        out = [f"{path}", f"  Classes: {len(classes)}", ""]
        for ln, name, methods in classes:
            out.append(f"{ln:>4}: class {name}")
            for ml, mn in methods:
                out.append(f"{ml:>4}:   .{mn}")
        return "\n".join(out)
    return f"{path}\n  (No classes found)"


def _scout_imports(path: Path) -> str:
    """Extract import statements."""
    content = path.read_text(encoding="utf-8", errors="replace")
    imports: list[str] = []
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(f"{i:>4}: {stripped}")
    if imports:
        return f"{path}\n  Imports: {len(imports)}\n\n" + "\n".join(imports)
    return f"{path}\n  (No imports found)"


def _scout_docstrings(path: Path) -> str:
    """Extract docstrings."""
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    in_doc = False
    doc_start = 0
    doc_lines: list[str] = []
    docstrings: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines, 1):
        if '"""' in line or "'''" in line:
            if not in_doc:
                in_doc = True
                doc_start = i
                doc_lines = [line.strip()]
                quote = '"""' if '"""' in line else "'''"
                if line.count(quote) >= 2:
                    in_doc = False
                    docstrings.append((doc_start, i, doc_lines[0]))
                    doc_lines = []
            else:
                doc_lines.append(line.strip())
                summary = ""
                for dl in doc_lines:
                    clean = dl.strip().strip('"').strip("'").strip()
                    if clean:
                        summary = clean[:60] + ("..." if len(clean) > 60 else "")
                        break
                docstrings.append((doc_start, i, summary))
                in_doc = False
                doc_lines = []
        elif in_doc:
            doc_lines.append(line.strip())
    if docstrings:
        out = [f"{path}", f"  Docstrings: {len(docstrings)}", ""]
        for start, end, summary in docstrings:
            out.append(f"{start:>4}-{end}: {summary}" if start != end else f"{start:>4}: {summary}")
        return "\n".join(out)
    return f"{path}\n  (No docstrings found)"


def _scout_comments(path: Path) -> str:
    """Extract comment lines."""
    content = path.read_text(encoding="utf-8", errors="replace")
    comments: list[str] = []
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            comments.append(f"{i:>4}: {stripped[:70]}")
    if comments:
        return f"{path}\n  Comments: {len(comments)}\n\n" + "\n".join(comments)
    return f"{path}\n  (No comments found)"


def _scout_todos(path: Path) -> str:
    """Extract TODO, FIXME, HACK, XXX markers."""
    content = path.read_text(encoding="utf-8", errors="replace")
    marker_re = re.compile(r"\b(TODO|FIXME|HACK|XXX|NOTE|BUG)[\s:\(\-]", re.IGNORECASE)
    todos: list[str] = []
    for i, line in enumerate(content.splitlines(), 1):
        m = marker_re.search(line)
        if m:
            todos.append(f"{i:>4}: [{m.group(1).upper()}] {line[m.start():].strip()[:80]}")
    if todos:
        return f"{path}\n  Markers: {len(todos)}\n\n" + "\n".join(todos)
    return f"{path}\n  (No TODO/FIXME markers found)"


def _scout_decorators(path: Path) -> str:
    """Extract @decorator lines."""
    content = path.read_text(encoding="utf-8", errors="replace")
    decos: list[str] = []
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("@") and not stripped.startswith("@@"):
            decos.append(f"{i:>4}: {stripped}")
    if decos:
        return f"{path}\n  Decorators: {len(decos)}\n\n" + "\n".join(decos)
    return f"{path}\n  (No decorators found)"


def _scout_types(path: Path) -> str:
    """Extract type hints, TypedDict, dataclass, Protocol."""
    content = path.read_text(encoding="utf-8", errors="replace")
    items: list[str] = []
    checks = [
        ("type alias", lambda s: s.startswith("type ") or ": TypeAlias" in s),
        ("TypedDict", lambda s: "TypedDict" in s),
        ("dataclass", lambda s: "@dataclass" in s),
        ("Protocol", lambda s: "(Protocol)" in s or "(Protocol," in s),
        ("annotation", lambda s: ") ->" in s),
    ]
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        for ptype, check in checks:
            if check(stripped):
                items.append(f"{i:>4}: [{ptype}] {stripped[:70]}")
                break
    if items:
        return f"{path}\n  Type definitions: {len(items)}\n\n" + "\n".join(items)
    return f"{path}\n  (No type definitions found)"


def _scout_errors(path: Path) -> str:
    """Extract try/except/raise statements."""
    content = path.read_text(encoding="utf-8", errors="replace")
    items: list[str] = []
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("try:"):
            items.append(f"{i:>4}: [try] {stripped}")
        elif stripped.startswith("except") and (":" in stripped or stripped == "except"):
            items.append(f"{i:>4}: [except] {stripped[:60]}")
        elif stripped.startswith("raise ") or stripped == "raise":
            items.append(f"{i:>4}: [raise] {stripped[:60]}")
    if items:
        return f"{path}\n  Error handling: {len(items)}\n\n" + "\n".join(items)
    return f"{path}\n  (No error handling found)"


def _scout_tests(path: Path) -> str:
    """Extract test functions."""
    content = path.read_text(encoding="utf-8", errors="replace")
    tests: list[str] = []
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            name = stripped.replace("async def ", "def ").replace("def ", "").split("(")[0]
            if name.startswith("test_") or name.endswith("_test"):
                tests.append(f"{i:>4}: {name}")
        elif stripped.startswith("class ") and "Test" in stripped:
            tests.append(f"{i:>4}: {stripped.split('(')[0].split(':')[0]}")
    if tests:
        return f"{path}\n  Tests: {len(tests)}\n\n" + "\n".join(tests)
    return f"{path}\n  (No tests found)"


def _scout_main(path: Path) -> str:
    """Find if __name__ == '__main__' block."""
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    for i, line in enumerate(lines, 1):
        if "__name__" in line and "__main__" in line and "if" in line:
            out = [f"{path}", f"  Main block at line {i}", ""]
            for j in range(i, min(i + 15, len(lines) + 1)):
                out.append(f"{j:>4}|{lines[j - 1]}")
                if j > i and lines[j - 1].strip() and not lines[j - 1].startswith((" ", "\t")):
                    break
            return "\n".join(out)
    return f"{path}\n  (No main block found)"


def _scout_header(path: Path) -> str:
    """Extract module docstring / file header."""
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    header: list[str] = []
    in_doc = False
    doc_char: str | None = None
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if i <= 2 and (stripped.startswith("#!") or stripped.startswith("# -*-")):
            header.append(f"{i:>4}|{line}")
            continue
        if not in_doc:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                doc_char = stripped[:3]
                in_doc = True
                header.append(f"{i:>4}|{line}")
                if stripped.count(doc_char) >= 2:
                    break
            elif stripped.startswith("#"):
                header.append(f"{i:>4}|{line}")
            elif stripped:
                break
        else:
            header.append(f"{i:>4}|{line}")
            if doc_char and doc_char in line and not line.strip().startswith(doc_char):
                break
    if header:
        return f"{path}\n  Header: {len(header)} lines\n\n" + "\n".join(header)
    return f"{path}\n  (No header found)"


def _scout_exports(path: Path) -> str:
    """Extract __all__ or module.exports."""
    content = path.read_text(encoding="utf-8", errors="replace")
    exports: list[str] = []
    in_all = False
    depth = 0
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if "__all__" in stripped and "=" in stripped:
            in_all = True
            depth = stripped.count("[") - stripped.count("]")
            exports.append(f"{i:>4}: {stripped[:70]}")
            if depth <= 0:
                in_all = False
        elif in_all:
            depth += stripped.count("[") - stripped.count("]")
            exports.append(f"{i:>4}: {stripped[:70]}")
            if depth <= 0:
                in_all = False
    if exports:
        return f"{path}\n  Exports: {len(exports)} lines\n\n" + "\n".join(exports)
    return f"{path}\n  (No exports found)"


def _scout_section(path: Path, start_marker: str, end_marker: str | None = None) -> str:
    """Extract content between markers."""
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    start_idx: int | None = None
    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i
            break
    assert start_idx is not None, f"Start marker not found: {start_marker}"
    end_idx = len(lines)
    if end_marker:
        for i in range(start_idx + 1, len(lines)):
            if end_marker in lines[i]:
                end_idx = i + 1
                break
        else:
            return f"End marker not found: {end_marker}"
    selected = lines[start_idx:end_idx]
    out = [f"{path}:{start_idx + 1}-{end_idx}", f"  Section: {len(selected)} lines", ""]
    for i, line in enumerate(selected, start=start_idx + 1):
        out.append(f"{i:>4}|{line}")
    return "\n".join(out)


def _scout_pattern(path: Path, pattern: str) -> str:
    """Search for regex pattern in file."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"
    content = path.read_text(encoding="utf-8", errors="replace")
    results: list[str] = []
    for i, line in enumerate(content.splitlines(), 1):
        if regex.search(line):
            results.append(f"  {path}:{i}")
            results.append(f"    {line.strip()}")
    if results:
        return f"Found {len(results) // 2} matches for '{pattern}':\n\n" + "\n".join(results)
    return f"No matches for '{pattern}' in {path}"


def _read_impl(path: str, start: int = 1, end: int = -1) -> str:
    """Read file with line numbers and range selection."""
    file_path = _normalize_path(path)
    assert file_path.exists(), f"File not found: {path}"
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    total = len(lines)
    if end == -1:
        end = total
    start_idx = max(0, start - 1)
    end_idx = min(total, end)
    numbered = [f"{i + start_idx + 1:4}: {line}" for i, line in enumerate(lines[start_idx:end_idx])]
    return "".join(numbered)


def _tail_impl(path: str, lines: int = 10) -> str:
    """Show last N lines of a file."""
    file_path = _normalize_path(path)
    assert file_path.exists(), f"File not found: {path}"
    content = file_path.read_text(encoding="utf-8", errors="replace")
    file_lines = content.splitlines()
    total = len(file_lines)
    start_idx = max(0, total - lines)
    selected = file_lines[start_idx:]
    numbered = [f"{start_idx + i + 1:>4}: {line}" for i, line in enumerate(selected)]
    header = f"=== {file_path} (last {len(selected)} of {total} lines) ==="
    return header + "\n" + "\n".join(numbered)


def _cat_impl(path: str, directory: bool = False, extensions: str = "") -> str:
    """Read file raw or concatenate directory contents."""
    file_path = _normalize_path(path)
    if not directory:
        assert file_path.exists(), f"File not found: {path}"
        return file_path.read_text(encoding="utf-8", errors="replace")
    # Directory mode
    assert file_path.is_dir(), f"Not a directory: {path}"
    ext_list = None
    if extensions:
        ext_list = [e.strip().lower() if e.startswith(".") else f".{e.strip().lower()}" for e in extensions.split(",")]
    parts: list[str] = []
    total_chars = 0
    files = sorted(file_path.rglob("*"))
    for fp in files:
        if not fp.is_file() or fp.suffix.lower() in SKIP_SUFFIXES:
            continue
        if ext_list and fp.suffix.lower() not in ext_list:
            continue
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
            rel = fp.relative_to(file_path)
            entry = f"=== {rel} ===\n{content}\n"
            if total_chars + len(entry) > 25000:
                parts.append(f"\n... truncated ({len(files) - len(parts)} files remaining)")
                break
            parts.append(entry)
            total_chars += len(entry)
        except Exception:
            continue
    return "\n".join(parts) if parts else f"No readable files in {path}"


def _tree_impl(path: str = ".", depth: int = 2) -> str:
    """Directory tree as JSON."""
    root = _normalize_path(path)
    assert root.exists(), f"Path not found: {path}"

    def build(dp: Path, d: int) -> dict[str, Any] | str:
        if d > depth:
            return "..."
        tree: dict[str, Any] = {}
        try:
            items = sorted(os.listdir(dp), key=lambda x: (not os.path.isdir(dp / x), x.lower()))
            for item in items:
                if item.startswith("."):
                    continue
                ip = dp / item
                tree[item + "/" if ip.is_dir() else item] = build(ip, d + 1) if ip.is_dir() else "file"
        except Exception as e:
            return f"Error: {e}"
        return tree

    return json.dumps({root.name + "/": build(root, 1)}, indent=2)


def _view_impl(file_path: str) -> str:
    """AST-based Python script overview."""
    import ast as ast_mod

    path = _normalize_path(file_path)
    assert path.suffix == ".py", f"view requires a .py file (got {path.suffix})"
    assert path.exists(), f"File not found: {file_path}"

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    try:
        tree = ast_mod.parse(content)
    except SyntaxError:
        # Regex fallback
        out = [f"{path.name} (AST parse failed — regex fallback)"]
        for line in lines:
            if line.strip().startswith("EXPOSED"):
                out.append(f"\n{line.strip()}")
                break
        out.append("\nFUNCTIONS:")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                out.append(f"  {i}: {stripped.rstrip(':')}")
        return "\n".join(out)

    out: list[str] = []
    module_doc = ast_mod.get_docstring(tree)
    first_line = module_doc.split("\n")[0].strip() if module_doc else ""
    out.append(f"{path.name}" + (f" — {first_line}" if first_line else ""))

    # Version
    version: str | None = None
    for line in lines:
        stripped = line.strip()
        if 'version="' in stripped and ('"-V"' in stripped or '"--version"' in stripped):
            m = re.search(r'version="([^"]+)"', stripped)
            if m:
                version = m.group(1)
                break
    if version:
        out[0] = f"{path.name} v{version}" + (f" — {first_line}" if first_line else "")

    # MCP name
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or "in line" in stripped:
            continue
        if "FastMCP(" in line:
            m = re.search(r'=\s*FastMCP\(["\']([^"\']+)', line)
            if m:
                out.append(f"MCP: {m.group(1)}")
                break

    # EXPOSED
    for line in lines:
        if line.strip().startswith("EXPOSED"):
            out.append(f"\n{line.strip()}")
            break

    # Functions with signatures
    out.append("\nFUNCTIONS:")
    for node in ast_mod.iter_child_nodes(tree):
        if isinstance(node, (ast_mod.FunctionDef, ast_mod.AsyncFunctionDef)):
            args_list: list[str] = []
            offset = len(node.args.args) - len(node.args.defaults)
            for i, arg in enumerate(node.args.args):
                ann = f": {ast_mod.unparse(arg.annotation)}" if arg.annotation else ""
                didx = i - offset
                default = f" = {ast_mod.unparse(node.args.defaults[didx])}" if 0 <= didx < len(node.args.defaults) else ""
                args_list.append(f"{arg.arg}{ann}{default}")
            returns = f" -> {ast_mod.unparse(node.returns)}" if node.returns else ""
            out.append(f"  {node.name}({', '.join(args_list)}){returns}")
            for stmt in node.body:
                if isinstance(stmt, ast_mod.Assert):
                    test_str = ast_mod.unparse(stmt.test)
                    msg_str = f" — {ast_mod.unparse(stmt.msg)}" if stmt.msg else ""
                    out.append(f"    assert {test_str}{msg_str}")
                elif isinstance(stmt, ast_mod.Expr) and isinstance(stmt.value, ast_mod.Constant):
                    continue
                else:
                    break

    # CONFIG
    for node in ast_mod.iter_child_nodes(tree):
        if isinstance(node, ast_mod.Assign):
            for target in node.targets:
                if isinstance(target, ast_mod.Name) and target.id == "CONFIG":
                    out.append("\nCONFIG:")
                    if isinstance(node.value, ast_mod.Dict):
                        for key, val in zip(node.value.keys, node.value.values):
                            if key:
                                out.append(f"  {ast_mod.unparse(key)}: {ast_mod.unparse(val)}")

    # PEP 723 deps
    in_block = False
    for line in lines:
        if line.strip() == "# /// script":
            in_block = True
        elif line.strip() == "# ///" and in_block:
            break
        elif in_block and "dependencies" in line:
            out.append(f"\nDEPENDENCIES: {line.strip().lstrip('# ')}")

    return "\n".join(out)


def _count_impl(pattern: str, path: str = ".") -> str:
    """Count matches of pattern across files."""
    root_path = _normalize_path(path)
    rg_path = _get_tool_path("rg", "SFB_RG_PATH")
    total = 0
    if rg_path:
        cmd = [rg_path, "-c", pattern]
        cwd = root_path.parent if root_path.is_file() else root_path
        if root_path.is_file():
            cmd.append(root_path.name)
        output = _run_command(cmd, cwd=cwd)
        for line in output.splitlines():
            parts = line.rsplit(":", 1)
            if len(parts) == 2 and parts[1].isdigit():
                total += int(parts[1])
            elif len(parts) == 1 and parts[0].isdigit():
                total += int(parts[0])
    else:
        result = _find_target_impl(pattern, path)
        total = result.count("\n") + (1 if result and not result.startswith("No matches") else 0)
    return str(total)


def _locate_impl(pattern: str, path: str = ".") -> str:
    """Find files matching glob pattern."""
    root_path = _normalize_path(path)

    # Strategy 1: ripgrep --files
    rg_path = shutil.which("rg")
    if rg_path:
        cmd = [rg_path, "--files", "--glob", pattern]
        try:
            output = subprocess.check_output(
                cmd, cwd=root_path, text=True, errors="replace", timeout=10,
            ).strip()
            if output:
                results = [str(root_path / p) for p in output.splitlines()]
                return json.dumps(results[:1000], indent=2)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass

    # Strategy 2: git ls-files
    git_path = shutil.which("git")
    if git_path and (root_path / ".git").exists():
        try:
            output = subprocess.check_output(
                [git_path, "ls-files", pattern],
                cwd=root_path, text=True, errors="replace", timeout=10,
            ).strip()
            if output:
                results = [str(root_path / p) for p in output.splitlines()]
                return json.dumps(results[:1000], indent=2)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass

    # Strategy 3: glob
    results = [str(p) for p in root_path.glob(pattern) if p.is_file()]
    return json.dumps(sorted(results)[:1000], indent=2) if results else "No files found"


# =============================================================================
# OPS: Destructive operations with git-aware backup
# =============================================================================


def _write_impl(file_path: str, content: str) -> str:
    """Write content to file with backup and BDA verification."""
    path = _normalize_path(file_path)
    is_new = not path.exists()
    backup_path = _backup_if_dirty(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    # BDA
    written = path.read_text(encoding="utf-8")
    assert written == content, f"BDA failed: content mismatch in {path}"

    size = len(content.encode("utf-8"))
    if is_new:
        return f"Created {path} ({size} bytes) [verified]"
    backup_line = f"\n     Backup: {backup_path}" if backup_path else ""
    return f"Wrote {path} ({size} bytes) [verified]{backup_line}"


def _create_impl(file_path: str, content: str = "") -> str:
    """Create new file. Fails if exists."""
    path = _normalize_path(file_path)
    assert not path.exists(), f"File already exists: {file_path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    written = path.read_text(encoding="utf-8")
    assert written == content, f"BDA failed: content mismatch in {path}"
    return f"Created {path} ({len(content.encode('utf-8'))} bytes) [verified]"


def _append_impl(file_path: str, content: str) -> str:
    """Append content to existing file."""
    path = _normalize_path(file_path)
    assert path.exists(), f"File not found: {file_path}"
    backup_path = _backup_if_dirty(path)
    with open(path, "a", encoding="utf-8") as f:
        if not content.startswith("\n"):
            f.write("\n")
        f.write(content)
    backup_line = f"\n     Backup: {backup_path}" if backup_path else ""
    return f"[OK] Appended to {path}{backup_line}"


def _cp_impl(src: str, dest: str) -> str:
    """Copy file or directory with backup."""
    src_path = _normalize_path(src)
    dest_path = _normalize_path(dest)
    assert src_path.exists(), f"Source not found: {src}"
    backup_path = _backup_if_dirty(dest_path) if dest_path.exists() else None
    if src_path.is_dir():
        shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
    else:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)
    result = f"[OK] Copied: {src_path} -> {dest_path}"
    if backup_path:
        result += f"\n     Backup: {backup_path}"
    return result


def _mv_impl(src: str, dest: str) -> str:
    """Move/rename with backup."""
    src_path = _normalize_path(src)
    dest_path = _normalize_path(dest)
    assert src_path.exists(), f"Source not found: {src}"
    backup_path = _backup_if_dirty(dest_path) if dest_path.exists() else None
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_path), str(dest_path))
    result = f"[OK] Moved: {src_path} -> {dest_path}"
    if backup_path:
        result += f"\n     Backup: {backup_path}"
    return result


def _rm_impl(file_path: str) -> str:
    """Delete with backup."""
    import tarfile
    path = _normalize_path(file_path)
    assert path.exists(), f"Not found: {file_path}"
    if path.is_dir():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"{path.name}.{ts}.tar.gz"
        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(path, arcname=path.name)
        shutil.rmtree(path)
        return f"[OK] Deleted directory: {path}\n     Backup: {backup_path}"
    backup_path = _backup_if_dirty(path)
    path.unlink()
    result = f"[OK] Deleted: {path}"
    if backup_path:
        result += f"\n     Backup: {backup_path}"
    return result


def _recover_impl(
    action: str,
    backup_name: str = "",
    target_path: str = "",
    pattern: str = "",
    confirm_delete: bool = False,
) -> str:
    """Backup management: list, assess, restore, clear."""
    action = action.lower()
    if action == "list":
        if not BACKUP_DIR.exists():
            return "No backup directory found"
        backups = sorted(BACKUP_DIR.glob("*.backup"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pattern:
            backups = [b for b in backups if pattern.lower() in b.name.lower()]
        if not backups:
            return "No backups found"
        out = [f"Backups: {len(backups)}", f"Location: {BACKUP_DIR}", ""]
        for b in backups:
            out.append(f"  {b.name} ({b.stat().st_size} bytes)")
        return "\n".join(out)
    elif action == "assess":
        assert backup_name, "assess requires backup_name"
        bp = BACKUP_DIR / (backup_name if backup_name.endswith(".backup") else f"{backup_name}.backup")
        assert bp.exists(), f"Backup not found: {backup_name}"
        content = bp.read_text(encoding="utf-8", errors="replace")
        blines = content.splitlines()
        out = [f"Backup: {bp.name}", f"Size: {bp.stat().st_size} bytes", f"Lines: {len(blines)}", "", "Preview (first 20):"]
        for i, line in enumerate(blines[:20], 1):
            out.append(f"{i:>4}|{line}")
        if len(blines) > 20:
            out.append(f"  ... ({len(blines) - 20} more)")
        return "\n".join(out)
    elif action in ("restore", "strike"):
        assert backup_name, "restore requires backup_name"
        assert target_path, "restore requires target_path"
        bp = BACKUP_DIR / (backup_name if backup_name.endswith(".backup") else f"{backup_name}.backup")
        assert bp.exists(), f"Backup not found: {backup_name}"
        target = Path(target_path).resolve()
        current_backup = _backup_if_dirty(target) if target.exists() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bp, target)
        out = [f"[OK] Restored {target} from {backup_name}"]
        if current_backup:
            out.append(f"     Previous backed up to: {current_backup}")
        return "\n".join(out)
    elif action == "clear":
        if not BACKUP_DIR.exists():
            return "No backup directory found"
        backups = list(BACKUP_DIR.glob("*.backup"))
        if pattern:
            backups = [b for b in backups if pattern.lower() in b.name.lower()]
        if not backups:
            return "No backups to clear"
        if not confirm_delete:
            total_size = sum(b.stat().st_size for b in backups)
            return f"[DRY] Would delete {len(backups)} backup(s) ({total_size:,} bytes). Use confirm_delete=True."
        deleted = 0
        for b in backups:
            try:
                b.unlink()
                deleted += 1
            except Exception:
                pass
        return f"[OK] Deleted {deleted} backup(s)"
    return f"Unknown action: {action}. Use: list, assess, restore, clear"


# =============================================================================
# CLI INTERFACE
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="File operations — find, patch, check workflow with recon tools"
    )
    parser.add_argument("-V", "--version", action="version", version="1.0.0")
    sub = parser.add_subparsers(dest="command", help="Commands")

    # --- mcp-stdio ---
    sub.add_parser("mcp-stdio", help="Run as MCP server")

    # --- Workflow: find ---
    p_find = sub.add_parser("find", help="Find patch target (search with context)")
    p_find.add_argument("pattern")
    p_find.add_argument("path", nargs="?", default=".")
    p_find.add_argument("-C", "--context", type=int, default=0)
    p_find.add_argument("-g", "--include", default="")

    # --- Workflow: dry ---
    p_dry = sub.add_parser("dry", help="Dry run patch (preview)")
    p_dry.add_argument("file_glob")
    p_dry.add_argument("context_start", type=int)
    p_dry.add_argument("context_end", type=int)
    p_dry.add_argument("old")
    p_dry.add_argument("new")

    # --- Workflow: patch ---
    p_patch = sub.add_parser("patch", help="Execute patch")
    p_patch.add_argument("file_glob")
    p_patch.add_argument("context_start", type=int)
    p_patch.add_argument("context_end", type=int)
    p_patch.add_argument("old")
    p_patch.add_argument("new")

    # --- Workflow: check ---
    p_check = sub.add_parser("check", help="Confirm patch outcome")
    p_check.add_argument("file_path")
    p_check.add_argument("-s", "--start", type=int, default=0)
    p_check.add_argument("-e", "--end", type=int, default=0)
    p_check.add_argument("-x", "--expected", default="")

    # --- Recon: scout ---
    p_scout = sub.add_parser("scout", help="Code intelligence")
    p_scout.add_argument("coords")
    p_scout.add_argument("pattern", nargs="?", default="")

    # --- Recon: search ---
    p_search = sub.add_parser("search", help="Cross-codebase content search")
    p_search.add_argument("pattern")
    p_search.add_argument("path", nargs="?", default=".")
    p_search.add_argument("-C", "--context", type=int, default=0)
    p_search.add_argument("-g", "--include", default="")

    # --- Recon: read ---
    p_read = sub.add_parser("read", help="Read file with line numbers")
    p_read.add_argument("path")
    p_read.add_argument("-s", "--start", type=int, default=1)
    p_read.add_argument("-e", "--end", type=int, default=-1)

    # --- Recon: tail ---
    p_tail = sub.add_parser("tail", help="Last N lines")
    p_tail.add_argument("path")
    p_tail.add_argument("-n", "--lines", type=int, default=10)

    # --- Recon: cat ---
    p_cat = sub.add_parser("cat", help="Raw read or directory concat")
    p_cat.add_argument("path")
    p_cat.add_argument("-d", "--dir", action="store_true", dest="directory")
    p_cat.add_argument("-x", "--ext", dest="extensions", default="")

    # --- Recon: tree ---
    p_tree = sub.add_parser("tree", help="Directory structure")
    p_tree.add_argument("path", nargs="?", default=".")
    p_tree.add_argument("-d", "--depth", type=int, default=2)

    # --- Recon: view ---
    p_view = sub.add_parser("view", help="Python script overview (AST)")
    p_view.add_argument("file_path")

    # --- Recon: count ---
    p_count = sub.add_parser("count", help="Count pattern matches")
    p_count.add_argument("pattern")
    p_count.add_argument("path", nargs="?", default=".")

    # --- Recon: locate ---
    p_locate = sub.add_parser("locate", help="Find files by glob pattern")
    p_locate.add_argument("pattern")
    p_locate.add_argument("path", nargs="?", default=".")

    # --- Ops: write ---
    p_write = sub.add_parser("write", help="Write file (backup + BDA)")
    p_write.add_argument("file_path")
    p_write.add_argument("content", nargs="?", default=None)

    # --- Ops: create ---
    p_create = sub.add_parser("create", help="Create new file")
    p_create.add_argument("file_path")
    p_create.add_argument("content", nargs="?", default="")

    # --- Ops: append ---
    p_append = sub.add_parser("append", help="Append to file")
    p_append.add_argument("file_path")
    p_append.add_argument("content", nargs="?", default=None)

    # --- Ops: cp ---
    p_cp = sub.add_parser("cp", help="Copy with backup")
    p_cp.add_argument("src")
    p_cp.add_argument("dest")

    # --- Ops: mv ---
    p_mv = sub.add_parser("mv", help="Move with backup")
    p_mv.add_argument("src")
    p_mv.add_argument("dest")

    # --- Ops: rm ---
    p_rm = sub.add_parser("rm", help="Delete with backup")
    p_rm.add_argument("path")

    # --- Ops: splice ---
    p_splice = sub.add_parser("splice", help="Splice content from scratch file into target")
    p_splice.add_argument("file_path", help="Target file")
    p_splice.add_argument("-s", "--start", type=int, default=0, dest="start_line", help="Start line (1-based)")
    p_splice.add_argument("-e", "--end", type=int, default=0, dest="end_line", help="End line (inclusive)")
    p_splice.add_argument("-S", "--start-str", default="", help="Start anchor string")
    p_splice.add_argument("-E", "--stop-str", default="", help="Stop anchor string")
    p_splice.add_argument("-n", "--dry", action="store_true", help="Dry run (preview)")
    p_splice.add_argument("-a", "--all", action="store_true", dest="replace_all", help="Replace all anchor matches")

    # --- Ops: recover ---
    p_recover = sub.add_parser("recover", help="Backup management")
    p_recover.add_argument("action", choices=["list", "assess", "restore", "clear"])
    p_recover.add_argument("-b", "--backup", default="")
    p_recover.add_argument("-t", "--target", default="")
    p_recover.add_argument("-p", "--pattern", default="")
    p_recover.add_argument("-y", "--confirm", action="store_true")

    args = parser.parse_args()

    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "find":
            print(_find_target_impl(args.pattern, args.path, context=args.context, include=args.include))
        elif args.command == "dry":
            print(_patch_impl(args.file_glob, args.context_start, args.context_end, args.old, args.new, dry_run=True))
        elif args.command == "patch":
            print(_patch_impl(args.file_glob, args.context_start, args.context_end, args.old, args.new))
        elif args.command == "check":
            print(_check_impl(args.file_path, args.start, args.end, args.expected))
        elif args.command == "scout":
            print(_scout_impl(args.coords, args.pattern))
        elif args.command == "search":
            print(_search_impl(args.pattern, args.path, context=args.context, include=args.include))
        elif args.command == "read":
            print(_read_impl(args.path, args.start, args.end))
        elif args.command == "tail":
            print(_tail_impl(args.path, args.lines))
        elif args.command == "cat":
            print(_cat_impl(args.path, directory=args.directory, extensions=args.extensions))
        elif args.command == "tree":
            print(_tree_impl(args.path, args.depth))
        elif args.command == "view":
            print(_view_impl(args.file_path))
        elif args.command == "count":
            print(_count_impl(args.pattern, args.path))
        elif args.command == "locate":
            print(_locate_impl(args.pattern, args.path))
        elif args.command == "write":
            content = args.content
            if not content and not sys.stdin.isatty():
                content = sys.stdin.read()
            assert content is not None, "content required"
            print(_write_impl(args.file_path, content))
        elif args.command == "create":
            print(_create_impl(args.file_path, args.content))
        elif args.command == "append":
            content = args.content
            if not content and not sys.stdin.isatty():
                content = sys.stdin.read()
            assert content is not None, "content required"
            print(_append_impl(args.file_path, content))
        elif args.command == "cp":
            print(_cp_impl(args.src, args.dest))
        elif args.command == "mv":
            print(_mv_impl(args.src, args.dest))
        elif args.command == "rm":
            print(_rm_impl(args.path))
        elif args.command == "splice":
            print(_splice_impl(
                args.file_path,
                start_line=args.start_line,
                end_line=args.end_line,
                start_str=args.start_str,
                stop_str=args.stop_str,
                dry_run=args.dry,
                replace_all=args.replace_all,
            ))
        elif args.command == "recover":
            print(_recover_impl(args.action, backup_name=args.backup, target_path=args.target, pattern=args.pattern, confirm_delete=args.confirm))
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

    mcp = FastMCP("fo")

    # --- Workflow: the 4-step edit flow ---

    @mcp.tool()
    def find(pattern: str, path: str = ".", context: int = 0, include: str = "") -> str:
        """Find patch target — search with context to locate text for editing.

        Step 1: Find the exact text you'll replace. Use context to get surrounding lines.

        Args:
            pattern: Text or regex to search for
            path: File or directory to search (default: current directory)
            context: Lines of surrounding context to include (key for targeting)
            include: File glob filter (e.g. "*.py")
        """
        return _find_target_impl(pattern, path, context=context, include=include)

    @mcp.tool()
    def dry(file_glob: str, context_start: int, context_end: int, old: str, new: str) -> str:
        """Dry run the patch — preview what will change without modifying files.

        Step 2: See the diff before committing. Same args as patch.

        Args:
            file_glob: File path or glob pattern (e.g. "scripts/sft_*.py")
            context_start: Start line of target context (1-based)
            context_end: End line of target context (inclusive)
            old: Text to find within the context range
            new: Replacement text (empty string = delete)
        """
        return _patch_impl(file_glob, context_start, context_end, old, new, dry_run=True)

    @mcp.tool()
    def patch(file_glob: str, context_start: int, context_end: int, old: str, new: str) -> str:
        """Execute the patch — find old text in context range, replace with new.

        Step 3: Apply the change. Git-aware backup on dirty files.
        Update: old->new. Delete: new="". Insert: old="anchor", new="anchor\\nnew_content".

        Args:
            file_glob: File path or glob pattern (e.g. "scripts/sft_*.py")
            context_start: Start line of target context (1-based)
            context_end: End line of target context (inclusive)
            old: Text to find within the context range
            new: Replacement text (empty string = delete)
        """
        return _patch_impl(file_glob, context_start, context_end, old, new)

    @mcp.tool()
    def check(file_path: str, context_start: int = 0, context_end: int = 0, expected: str = "") -> str:
        """Confirm the patch landed — verify expected content at location.

        Step 4: Post-patch verification. Without expected, shows content at location.

        Args:
            file_path: File to check
            context_start: Start line to check (0 = whole file)
            context_end: End line to check (0 = whole file)
            expected: Text that should be present (empty = just show content)
        """
        return _check_impl(file_path, context_start, context_end, expected)

    # --- Recon: non-destructive intelligence gathering ---

    @mcp.tool()
    def scout(coords: str, pattern: str = "") -> str:
        """Code intelligence — structural recon via pattern routing.

        Patterns: functions, classes, imports, docstrings, comments, todos,
        decorators, types, errors, tests, main, header, exports, section:MARKER
        Without pattern: shows file structure overview. With regex: searches file.

        Args:
            coords: File path (optionally with :line or :start-end)
            pattern: Extraction pattern or regex
        """
        return _scout_impl(coords, pattern)

    @mcp.tool()
    def search(pattern: str, path: str = ".", context: int = 0, include: str = "") -> str:
        """Cross-codebase content search with context.

        Args:
            pattern: Text or regex to search for
            path: Directory or file to search (default: current directory)
            context: Lines of surrounding context (default: 0)
            include: File glob filter (e.g. "*.py")
        """
        return _search_impl(pattern, path, context=context, include=include)

    @mcp.tool()
    def read(path: str, start: int = 1, end: int = -1) -> str:
        """Read file with line numbers and optional range.

        Args:
            path: File path to read
            start: Start line (1-indexed, default: 1)
            end: End line (default: -1 for entire file)
        """
        return _read_impl(path, start, end)

    @mcp.tool()
    def tail(path: str, lines: int = 10) -> str:
        """Show last N lines of a file.

        Args:
            path: File path
            lines: Number of lines (default: 10)
        """
        return _tail_impl(path, lines)

    @mcp.tool()
    def cat(path: str, directory: bool = False, extensions: str = "") -> str:
        """Raw file read or directory concatenation.

        Args:
            path: File or directory path
            directory: If true, concatenate all files in directory
            extensions: Filter by extensions when in directory mode (comma-separated)
        """
        return _cat_impl(path, directory=directory, extensions=extensions)

    @mcp.tool()
    def tree(path: str = ".", depth: int = 2) -> str:
        """Directory structure as JSON.

        Args:
            path: Directory path (default: current directory)
            depth: Max depth (default: 2)
        """
        return _tree_impl(path, depth)

    @mcp.tool()
    def view(file_path: str) -> str:
        """AST-based Python script overview — version, EXPOSED, signatures, CONFIG, deps.

        Args:
            file_path: Path to .py file
        """
        return _view_impl(file_path)

    @mcp.tool()
    def count(pattern: str, path: str = ".") -> str:
        """Count matches of pattern across files.

        Args:
            pattern: Text or regex to count
            path: Directory or file (default: current directory)
        """
        return _count_impl(pattern, path)

    @mcp.tool()
    def locate(pattern: str, path: str = ".") -> str:
        """Find files matching glob pattern.

        Args:
            pattern: Glob pattern (e.g. "*.py", "test_*.py")
            path: Directory to search (default: current directory)
        """
        return _locate_impl(pattern, path)

    # --- Ops: destructive, git-aware backup ---

    @mcp.tool()
    def write(path: str, content: str) -> str:
        """Write file with automatic backup and BDA verification.

        Args:
            path: File path to write
            content: Content to write
        """
        return _write_impl(path, content)

    @mcp.tool()
    def create(path: str, content: str = "") -> str:
        """Create new file. Fails if file already exists.

        Args:
            path: File path to create
            content: Initial content (default: empty)
        """
        return _create_impl(path, content)

    @mcp.tool()
    def append(path: str, content: str) -> str:
        """Append content to existing file with backup.

        Args:
            path: File path to append to
            content: Content to append
        """
        return _append_impl(path, content)

    @mcp.tool()
    def cp(src: str, dest: str) -> str:
        """Copy file or directory with backup of destination.

        Args:
            src: Source path
            dest: Destination path
        """
        return _cp_impl(src, dest)

    @mcp.tool()
    def mv(src: str, dest: str) -> str:
        """Move/rename with backup of destination.

        Args:
            src: Source path
            dest: Destination path
        """
        return _mv_impl(src, dest)

    @mcp.tool()
    def rm(path: str) -> str:
        """Delete file or directory with backup.

        Args:
            path: Path to delete
        """
        return _rm_impl(path)

    @mcp.tool()
    def recover(action: str, backup_name: str = "", target_path: str = "", pattern: str = "", confirm_delete: bool = False) -> str:
        """Backup management — list, assess, restore, clear.

        Args:
            action: Operation — list, assess, restore, clear
            backup_name: Backup file name (for assess/restore)
            target_path: Where to restore to (for restore)
            pattern: Filter pattern (for list/clear)
            confirm_delete: Must be True to actually delete (for clear)
        """
        return _recover_impl(action, backup_name=backup_name, target_path=target_path, pattern=pattern, confirm_delete=confirm_delete)

    @mcp.tool()
    def splice(
        file_path: str,
        start_line: int = 0,
        end_line: int = 0,
        start_str: str = "",
        stop_str: str = "",
        dry_run: bool = False,
        replace_all: bool = False,
    ) -> str:
        """Splice content from scratch file into target file.

        Three modes: lines only, anchor strings only, or both.
        New content comes from FO_SPLICE_FILE env var path.

        Args:
            file_path: Target file to splice into
            start_line: Start line (1-based, 0=not set)
            end_line: End line (inclusive, 0=not set)
            start_str: Start anchor string (empty=not set)
            stop_str: Stop anchor string (empty=not set)
            dry_run: Preview only
            replace_all: Allow multiple anchor matches

        Returns:
            Splice result with line ranges and verification
        """
        return _splice_impl(
            file_path,
            start_line=start_line,
            end_line=end_line,
            start_str=start_str,
            stop_str=stop_str,
            dry_run=dry_run,
            replace_all=replace_all,
        )

    print("fo MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
