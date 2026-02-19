#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp"]
# ///
"""Code reconnaissance and search — scout file structure, find files, search content."""

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
    "scout",
    "functions",
    "classes",
    "imports",
    "docstrings",
    "comments",
    "todos",
    "decorators",
    "types",
    "errors",
    "tests",
    "main_block",
    "header",
    "exports",
    "ast_structure",
    "ast_docstrings",
    "search",
    "find",
    "count",
    "python_view",
]


CONTEXT_LINES = 3

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

GREP_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yml",
    ".yaml",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".sh",
    ".toml",
}

COMMENT_SYNTAX = {
    # Line comment, block start, block end
    ".py": ("#", '"""', '"""'),
    ".pyw": ("#", '"""', '"""'),
    ".sh": ("#", None, None),
    ".bash": ("#", None, None),
    ".zsh": ("#", None, None),
    ".yml": ("#", None, None),
    ".yaml": ("#", None, None),
    ".toml": ("#", None, None),
    ".rb": ("#", "=begin", "=end"),
    ".pl": ("#", "=pod", "=cut"),
    ".r": ("#", None, None),
    ".js": ("//", "/*", "*/"),
    ".jsx": ("//", "/*", "*/"),
    ".ts": ("//", "/*", "*/"),
    ".tsx": ("//", "/*", "*/"),
    ".mjs": ("//", "/*", "*/"),
    ".cjs": ("//", "/*", "*/"),
    ".java": ("//", "/*", "*/"),
    ".c": ("//", "/*", "*/"),
    ".cpp": ("//", "/*", "*/"),
    ".cc": ("//", "/*", "*/"),
    ".h": ("//", "/*", "*/"),
    ".hpp": ("//", "/*", "*/"),
    ".cs": ("//", "/*", "*/"),
    ".go": ("//", "/*", "*/"),
    ".rs": ("//", "/*", "*/"),
    ".swift": ("//", "/*", "*/"),
    ".kt": ("//", "/*", "*/"),
    ".scala": ("//", "/*", "*/"),
    ".php": ("//", "/*", "*/"),
    ".m": ("//", "/*", "*/"),
    ".mm": ("//", "/*", "*/"),
    ".sql": ("--", "/*", "*/"),
    ".lua": ("--", "--[[", "]]"),
    ".hs": ("--", "{-", "-}"),
    ".html": (None, "<!--", "-->"),
    ".htm": (None, "<!--", "-->"),
    ".xml": (None, "<!--", "-->"),
    ".svg": (None, "<!--", "-->"),
    ".vue": ("//", "<!--", "-->"),
    ".css": (None, "/*", "*/"),
    ".scss": ("//", "/*", "*/"),
    ".sass": ("//", "/*", "*/"),
    ".less": ("//", "/*", "*/"),
    ".ini": (";", None, None),
    ".cfg": ("#", None, None),
    ".conf": ("#", None, None),
    ".vim": ('"', None, None),
    ".el": (";", None, None),
    ".lisp": (";", "#|", "|#"),
    ".clj": (";", None, None),
    ".ex": ("#", None, None),
    ".exs": ("#", None, None),
    ".erl": ("%", None, None),
    ".asm": (";", None, None),
    ".s": (";", None, None),
    ".bat": ("REM", None, None),
    ".cmd": ("REM", None, None),
    ".ps1": ("#", "<#", "#>"),
    ".psm1": ("#", "<#", "#>"),
    ".f90": ("!", None, None),
    ".f95": ("!", None, None),
    ".jl": ("#", "#=", "=#"),
    ".nim": ("#", "#[", "]#"),
    ".zig": ("//", None, None),
    ".v": ("//", "/*", "*/"),
    ".sv": ("//", "/*", "*/"),
    ".vhd": ("--", None, None),
    ".vhdl": ("--", None, None),
    ".tcl": ("#", None, None),
    ".cmake": ("#", None, None),
    ".make": ("#", None, None),
    ".mk": ("#", None, None),
    "Makefile": ("#", None, None),
    "Dockerfile": ("#", None, None),
    ".tf": ("#", "/*", "*/"),
    ".hcl": ("#", "/*", "*/"),
    ".proto": ("//", "/*", "*/"),
    ".graphql": ("#", '"""', '"""'),
    ".gql": ("#", '"""', '"""'),
    ".md": (None, "<!--", "-->"),
    ".markdown": (None, "<!--", "-->"),
    ".rst": (None, None, None),
    ".tex": ("%", None, None),
    ".latex": ("%", None, None),
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


# --- Search helpers ---


def _get_tool_path(tool_name: str, env_var: str) -> str | None:
    """Get path to external tool, checking env override first."""
    override = os.environ.get(env_var)
    if override and os.path.exists(override):
        return override
    return shutil.which(tool_name)


def _run_command(args: list[str], cwd: Path | None = None, timeout: int = 30) -> str:
    """Run external command and return stdout. Timeout in seconds (default 30)."""
    try:
        cwd_str = str(cwd) if cwd else None
        result = subprocess.run(
            args,
            cwd=cwd_str,
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


# --- Coordinate parsing ---


def parse_coordinates(coords: str) -> dict[str, Any]:
    """Parse file:line:char coordinate strings."""
    result: dict[str, Any] = {
        "path": None,
        "line_start": None,
        "line_end": None,
        "char_start": None,
        "char_end": None,
    }
    parts = coords.split(":")
    if len(parts) >= 2 and len(parts[0]) == 1 and parts[0].isalpha():
        result["path"] = f"{parts[0]}:{parts[1]}"
        parts = parts[2:]
    else:
        result["path"] = parts[0]
        parts = parts[1:]
    if parts:
        line_spec = parts[0]
        if "-" in line_spec:
            start, end = line_spec.split("-", 1)
            result["line_start"] = int(start) if start else 1
            result["line_end"] = int(end) if end else None
        else:
            result["line_start"] = int(line_spec)
            result["line_end"] = int(line_spec)
        parts = parts[1:]
    if parts:
        char_spec = parts[0]
        if "-" in char_spec:
            start, end = char_spec.split("-", 1)
            result["char_start"] = int(start) if start else 0
            result["char_end"] = int(end) if end else None
        else:
            result["char_start"] = int(char_spec)
            result["char_end"] = int(char_spec)
    return result


# --- Scout functions ---


def scout_file_structure(path: Path) -> str:
    """Show file structure with function/class definitions."""
    start_ms = time.time() * 1000
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
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
    latency_ms = round(time.time() * 1000 - start_ms, 2)
    _log(
        "INFO",
        "scout_structure",
        str(path),
        metrics=f"latency_ms={latency_ms} status=success",
    )
    return "\n".join(out)


def scout_lines(
    path: Path, line_start: int, line_end: int, char_start: int | None = None
) -> str:
    """Show specific lines with context."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines()
    total = len(lines)
    if line_start < 1 or line_start > total:
        return f"Line {line_start} out of range (1-{total})"
    line_end = min(line_end, total)
    display_start = max(1, line_start - CONTEXT_LINES)
    display_end = min(total, line_end + CONTEXT_LINES)
    range_str = (
        f":{line_start}" if line_start == line_end else f":{line_start}-{line_end}"
    )
    out = [f"{path}{range_str}", ""]
    for i in range(display_start, display_end + 1):
        line = lines[i - 1]
        marker = ">>>" if line_start <= i <= line_end else "   "
        out.append(f"{marker} {i:>4}|{line}")
    return "\n".join(out)


def scout_pattern(path: Path, pattern: str) -> str:
    """Search for regex pattern in file or directory."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"
    results: list[str] = []
    if path.is_file():
        files = [path]
    else:
        files = [
            f for f in path.rglob("*") if f.is_file() and f.suffix not in SKIP_SUFFIXES
        ]
    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    match = regex.search(line)
                    char_pos = f":{match.start()}-{match.end()}" if match else ""
                    results.append(f"  {file_path}:{i}{char_pos}")
                    results.append(f"    {line.strip()}")
        except Exception:
            continue
    if results:
        return f"Found {len(results) // 2} matches for '{pattern}':\n\n" + "\n".join(
            results
        )
    return f"No matches for '{pattern}' in {path}"


def scout_imports(path: Path) -> str:
    """Extract import statements from Python file."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines()
    imports: list[str] = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(f"{i:>4}: {stripped}")
    if imports:
        return f"{path}\n  Imports: {len(imports)}\n\n" + "\n".join(imports)
    return f"{path}\n  (No imports found)"


def scout_functions(path: Path) -> str:
    """Extract function/method definitions from Python file."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines()
    functions: list[str] = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            indent = len(line) - len(line.lstrip())
            func_sig = stripped.split("(")[0]
            func_name = func_sig.replace("def ", "").replace("async ", "").strip()
            prefix = "  " * (indent // 4)
            functions.append(f"{i:>4}: {prefix}{func_name}")
    if functions:
        return f"{path}\n  Functions: {len(functions)}\n\n" + "\n".join(functions)
    return f"{path}\n  (No functions found)"


def scout_classes(path: Path) -> str:
    """Extract class definitions with inheritance from Python file."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines()
    classes: list[tuple[int, str, list[tuple[int, str]]]] = []
    current_class: str | None = None
    current_class_indent = 0
    current_class_line = 0
    current_methods: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if stripped.startswith("class "):
            if current_class:
                classes.append((current_class_line, current_class, current_methods))
                current_methods = []
            class_part = stripped[6:]
            if "(" in class_part:
                class_name = class_part.split("(")[0]
                inheritance = class_part.split("(")[1].rstrip("):").strip()
                if inheritance:
                    current_class = f"{class_name}({inheritance})"
                else:
                    current_class = class_name
            else:
                current_class = class_part.rstrip(":")
            current_class_indent = indent
            current_class_line = i
        elif current_class and (
            stripped.startswith("def ") or stripped.startswith("async def ")
        ):
            if indent > current_class_indent:
                func_sig = stripped.split("(")[0]
                method_name = func_sig.replace("def ", "").replace("async ", "").strip()
                current_methods.append((i, method_name))
    if current_class:
        classes.append((current_class_line, current_class, current_methods))
    if classes:
        out = [f"{path}", f"  Classes: {len(classes)}", ""]
        for line_num, class_name, methods in classes:
            out.append(f"{line_num:>4}: class {class_name}")
            for method_line, method_name in methods:
                out.append(f"{method_line:>4}:   .{method_name}")
        return "\n".join(out)
    return f"{path}\n  (No classes found)"


def scout_docstrings(path: Path) -> str:
    """Extract docstrings from Python file."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines()
    in_docstring = False
    docstring_start = 0
    docstring_lines: list[str] = []
    docstrings: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines, 1):
        if '"""' in line or "'''" in line:
            if not in_docstring:
                in_docstring = True
                docstring_start = i
                docstring_lines = [line.strip()]
                quote = '"""' if '"""' in line else "'''"
                if line.count(quote) >= 2:
                    in_docstring = False
                    docstrings.append((docstring_start, i, docstring_lines[0]))
                    docstring_lines = []
            else:
                docstring_lines.append(line.strip())
                summary = ""
                for dl in docstring_lines:
                    clean = dl.strip().strip('"').strip("'").strip()
                    if clean:
                        summary = clean[:60] + ("..." if len(clean) > 60 else "")
                        break
                docstrings.append((docstring_start, i, summary))
                in_docstring = False
                docstring_lines = []
        elif in_docstring:
            docstring_lines.append(line.strip())
    if docstrings:
        out = [f"{path}", f"  Docstrings: {len(docstrings)}", ""]
        for start, end, summary in docstrings:
            if start == end:
                out.append(f"{start:>4}: {summary}")
            else:
                out.append(f"{start:>4}-{end}: {summary}")
        return "\n".join(out)
    return f"{path}\n  (No docstrings found)"


def scout_comments(path: Path) -> str:
    """Extract comments from code file, detecting syntax by extension."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    suffix = path.suffix.lower()
    if suffix not in COMMENT_SYNTAX:
        if path.name in COMMENT_SYNTAX:
            suffix = path.name
        else:
            return f"{path}\n  (Unknown file type - cannot detect comment syntax)"
    line_char, block_start, block_end = COMMENT_SYNTAX[suffix]
    lines = content.splitlines()
    comments: list[tuple[int, int, str, str]] = []
    in_block = False
    block_start_line = 0
    block_content: list[str] = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if block_start and block_end:
            if in_block:
                block_content.append(line)
                if block_end in stripped:
                    in_block = False
                    summary = block_content[0].strip()[:50]
                    if len(block_content) > 1:
                        comments.append(
                            (
                                block_start_line,
                                i,
                                "block",
                                f"{summary}... ({len(block_content)} lines)",
                            )
                        )
                    else:
                        comments.append((block_start_line, i, "block", summary))
                    block_content = []
                continue
            elif block_start in stripped:
                if block_end in stripped and stripped.index(block_end) > stripped.index(
                    block_start
                ):
                    start_idx = stripped.index(block_start)
                    end_idx = stripped.index(block_end) + len(block_end)
                    comment_text = stripped[start_idx:end_idx]
                    if start_idx > 0:
                        comments.append((i, i, "inline-block", comment_text[:60]))
                    else:
                        comments.append((i, i, "block", comment_text[:60]))
                else:
                    in_block = True
                    block_start_line = i
                    block_content = [line]
                continue
        if line_char:
            if stripped.startswith(line_char):
                comment_text = stripped[len(line_char) :].strip()
                comments.append((i, i, "line", comment_text[:60]))
            elif line_char in line:
                idx = line.find(line_char)
                before = line[:idx]
                if before.count('"') % 2 == 0 and before.count("'") % 2 == 0:
                    comment_text = line[idx + len(line_char) :].strip()
                    if comment_text:
                        comments.append((i, i, "inline", comment_text[:60]))
    if comments:
        line_count = sum(1 for c in comments if c[2] == "line")
        inline_count = sum(1 for c in comments if c[2] in ("inline", "inline-block"))
        block_count = sum(1 for c in comments if c[2] == "block")
        out = [
            f"{path}",
            f"  Comments: {len(comments)} (line: {line_count}, inline: {inline_count}, block: {block_count})",
            "",
        ]
        for start, end, ctype, text in comments:
            prefix = {
                "line": "  #",
                "inline": " +#",
                "block": "/*",
                "inline-block": "+/*",
            }[ctype]
            if start == end:
                out.append(f"{start:>4} {prefix} {text}")
            else:
                out.append(f"{start:>4}-{end:>4} {prefix} {text}")
        return "\n".join(out)
    return f"{path}\n  (No comments found)"


def scout_todos(path: Path) -> str:
    """Extract TODO, FIXME, HACK, XXX markers from code file."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines()
    marker_pattern = re.compile(
        r"\b(TODO|FIXME|HACK|XXX|NOTE|BUG)[\s:\(\-]", re.IGNORECASE
    )
    todos: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines, 1):
        match = marker_pattern.search(line)
        if match:
            marker = match.group(1).upper()
            comment_text = line[match.start() :].strip()
            todos.append((i, marker, comment_text[:80]))
    if todos:
        out = [f"{path}", f"  Markers: {len(todos)}", ""]
        for line_num, marker, text in todos:
            out.append(f"{line_num:>4}: [{marker}] {text}")
        return "\n".join(out)
    return f"{path}\n  (No TODO/FIXME/HACK/XXX markers found)"


def scout_decorators(path: Path) -> str:
    """Extract @decorator lines from Python file."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines()
    decorators: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("@") and not stripped.startswith("@@"):
            decorators.append((i, stripped))
    if decorators:
        out = [f"{path}", f"  Decorators: {len(decorators)}", ""]
        for line_num, dec_text in decorators:
            out.append(f"{line_num:>4}: {dec_text}")
        return "\n".join(out)
    return f"{path}\n  (No decorators found)"


def scout_types(path: Path) -> str:
    """Extract type hints, TypedDict, dataclass, Protocol from Python file."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines()
    type_items: list[tuple[int, str, str]] = []
    patterns = [
        ("type alias", lambda s: s.startswith("type ") or ": TypeAlias" in s),
        ("TypedDict", lambda s: "TypedDict" in s),
        ("dataclass", lambda s: "@dataclass" in s),
        ("Protocol", lambda s: "(Protocol)" in s or "(Protocol," in s),
        ("annotation", lambda s: ") ->" in s),
        ("Generic", lambda s: "(Generic[" in s or ", Generic[" in s),
    ]
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        for ptype, check in patterns:
            if check(stripped):
                type_items.append((i, ptype, stripped[:70]))
                break
    if type_items:
        out = [f"{path}", f"  Type definitions: {len(type_items)}", ""]
        for line_num, ptype, text in type_items:
            out.append(f"{line_num:>4}: [{ptype}] {text}")
        return "\n".join(out)
    return f"{path}\n  (No type definitions found)"


def scout_errors(path: Path) -> str:
    """Extract try/except/raise statements from Python file."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines()
    error_items: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("try:"):
            error_items.append((i, "try", stripped))
        elif stripped.startswith("except") and (
            ":" in stripped or stripped == "except"
        ):
            error_items.append((i, "except", stripped[:60]))
        elif stripped.startswith("raise ") or stripped == "raise":
            error_items.append((i, "raise", stripped[:60]))
        elif stripped.startswith("finally:"):
            error_items.append((i, "finally", stripped))
    if error_items:
        try_count = sum(1 for e in error_items if e[1] == "try")
        except_count = sum(1 for e in error_items if e[1] == "except")
        raise_count = sum(1 for e in error_items if e[1] == "raise")
        out = [
            f"{path}",
            f"  Error handling: {len(error_items)} (try: {try_count}, except: {except_count}, raise: {raise_count})",
            "",
        ]
        for line_num, etype, text in error_items:
            out.append(f"{line_num:>4}: [{etype}] {text}")
        return "\n".join(out)
    return f"{path}\n  (No error handling found)"


def scout_tests(path: Path) -> str:
    """Extract test functions (test_*, *_test) from file."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines()
    tests: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            func_part = stripped.replace("async def ", "def ").replace("def ", "")
            func_name = func_part.split("(")[0]
            if func_name.startswith("test_") or func_name.endswith("_test"):
                tests.append((i, func_name))
        elif stripped.startswith("class ") and "Test" in stripped:
            class_name = stripped[6:].split("(")[0].split(":")[0]
            tests.append((i, f"class {class_name}"))
    if tests:
        out = [f"{path}", f"  Tests: {len(tests)}", ""]
        for line_num, name in tests:
            out.append(f"{line_num:>4}: {name}")
        return "\n".join(out)
    return f"{path}\n  (No tests found)"


def scout_main(path: Path) -> str:
    """Find if __name__ == '__main__' block."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines()
    main_start: int | None = None
    main_lines: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        if "__name__" in line and "__main__" in line and "if" in line:
            main_start = i
            main_lines.append((i, line))
        elif main_start and i <= main_start + 20:
            if line.strip() and not line.startswith((" ", "\t")):
                break
            main_lines.append((i, line))
    if main_start:
        out = [f"{path}", f"  Main block found at line {main_start}", ""]
        for line_num, line in main_lines:
            out.append(f"{line_num:>4}|{line}")
        return "\n".join(out)
    return f"{path}\n  (No if __name__ == '__main__' block found)"


def scout_header(path: Path) -> str:
    """Extract module docstring / file header."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines()
    header_lines: list[tuple[int, str, str]] = []
    in_docstring = False
    docstring_char: str | None = None

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if i <= 2 and (
            stripped.startswith("#!")
            or stripped.startswith("# -*-")
            or stripped.startswith("# coding")
        ):
            header_lines.append((i, "meta", line))
            continue
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                in_docstring = True
                header_lines.append((i, "doc", line))
                if stripped.count(docstring_char) >= 2:
                    in_docstring = False
                    break
            elif stripped.startswith("#"):
                header_lines.append((i, "comment", line))
            elif stripped:
                break
        else:
            header_lines.append((i, "doc", line))
            if (
                docstring_char
                and docstring_char in line
                and not line.strip().startswith(docstring_char)
            ):
                break

    if header_lines:
        out = [f"{path}", f"  Header: {len(header_lines)} lines", ""]
        for line_num, htype, line in header_lines:
            out.append(f"{line_num:>4}|{line}")
        return "\n".join(out)
    return f"{path}\n  (No header/docstring found)"


def scout_exports(path: Path) -> str:
    """Extract __all__ or module.exports declarations."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines()
    exports: list[tuple[int, str, str]] = []
    in_all = False
    bracket_depth = 0

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if "__all__" in stripped and "=" in stripped:
            in_all = True
            bracket_depth = stripped.count("[") - stripped.count("]")
            exports.append((i, "__all__", stripped[:70]))
            if bracket_depth <= 0:
                in_all = False
        elif in_all:
            bracket_depth += stripped.count("[") - stripped.count("]")
            exports.append((i, "__all__", stripped[:70]))
            if bracket_depth <= 0:
                in_all = False
        elif stripped.startswith("export ") or stripped.startswith("module.exports"):
            exports.append((i, "export", stripped[:70]))
        elif stripped.startswith("exports."):
            exports.append((i, "exports", stripped[:70]))

    if exports:
        out = [f"{path}", f"  Exports: {len(exports)} lines", ""]
        for line_num, etype, text in exports:
            out.append(f"{line_num:>4}: [{etype}] {text}")
        return "\n".join(out)
    return f"{path}\n  (No exports found)"


def scout_section(path: Path, start_marker: str, end_marker: str | None = None) -> str:
    """Extract content between markers in a file."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines()
    start_idx: int | None = None
    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i
            break
    if start_idx is None:
        return f"Start marker not found: {start_marker}"
    end_idx = len(lines)
    if end_marker:
        for i, line in enumerate(lines[start_idx + 1 :], start=start_idx + 1):
            if end_marker in line:
                end_idx = i + 1
                break
        else:
            return f"End marker not found: {end_marker}"
    selected = lines[start_idx:end_idx]
    out = [f"{path}:{start_idx + 1}-{end_idx}", f"  Section: {len(selected)} lines", ""]
    for i, line in enumerate(selected, start=start_idx + 1):
        out.append(f"{i:>4}|{line}")
    return "\n".join(out)


def _scout_impl(coords: str, pattern: str | None = None) -> tuple[str, dict[str, Any]]:
    """Unified scout function — reconnaissance for code exploration.

    Returns (result_str, metrics_dict).
    """
    start_ms = time.time() * 1000
    parsed = parse_coordinates(coords)
    path = _normalize_path(parsed["path"])
    if not path.exists():
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        return f"Path not found: {path}", {"status": "error", "latency_ms": latency_ms}

    result: str
    if pattern:
        lower = pattern.lower()
        if lower in ("imports", "import"):
            result = (
                scout_imports(path)
                if path.is_file()
                else "imports mode requires a file path"
            )
        elif lower in ("functions", "funcs", "defs"):
            result = (
                scout_functions(path)
                if path.is_file()
                else "functions mode requires a file path"
            )
        elif lower in ("classes", "class"):
            result = (
                scout_classes(path)
                if path.is_file()
                else "classes mode requires a file path"
            )
        elif lower in ("docstrings", "docs"):
            result = (
                scout_docstrings(path)
                if path.is_file()
                else "docstrings mode requires a file path"
            )
        elif lower in ("comments", "comment"):
            result = (
                scout_comments(path)
                if path.is_file()
                else "comments mode requires a file path"
            )
        elif lower in ("todos", "todo", "fixme"):
            result = (
                scout_todos(path)
                if path.is_file()
                else "todos mode requires a file path"
            )
        elif lower in ("decorators", "decorator", "deco"):
            result = (
                scout_decorators(path)
                if path.is_file()
                else "decorators mode requires a file path"
            )
        elif lower in ("types", "type", "typing"):
            result = (
                scout_types(path)
                if path.is_file()
                else "types mode requires a file path"
            )
        elif lower in ("errors", "error", "exceptions"):
            result = (
                scout_errors(path)
                if path.is_file()
                else "errors mode requires a file path"
            )
        elif lower in ("tests", "test"):
            result = (
                scout_tests(path)
                if path.is_file()
                else "tests mode requires a file path"
            )
        elif lower in ("main", "entry", "entrypoint"):
            result = (
                scout_main(path) if path.is_file() else "main mode requires a file path"
            )
        elif lower in ("header", "module", "preamble"):
            result = (
                scout_header(path)
                if path.is_file()
                else "header mode requires a file path"
            )
        elif lower in ("exports", "export", "__all__"):
            result = (
                scout_exports(path)
                if path.is_file()
                else "exports mode requires a file path"
            )
        elif lower.startswith("section:"):
            if path.is_file():
                parts = pattern.split(":", 2)
                if len(parts) == 2:
                    result = scout_section(path, parts[1])
                elif len(parts) >= 3:
                    result = scout_section(path, parts[1], parts[2])
                else:
                    result = "section mode requires: section:START_MARKER or section:START:END"
            else:
                result = "section mode requires a file path"
        else:
            result = scout_pattern(path, pattern)
    elif path.is_dir():
        result = f"Directory: {path}\nUse pattern to search, or specify a file."
    elif parsed["line_start"]:
        result = scout_lines(
            path,
            parsed["line_start"],
            parsed["line_end"] or parsed["line_start"],
            parsed["char_start"],
        )
    else:
        result = scout_file_structure(path)

    latency_ms = round(time.time() * 1000 - start_ms, 2)
    metrics = {
        "status": "success",
        "latency_ms": latency_ms,
        "coords": coords,
        "pattern": pattern,
    }
    _log(
        "INFO",
        "scout",
        f"{coords} pattern={pattern}",
        metrics=f"latency_ms={latency_ms} status=success",
    )
    return result, metrics


# --- AST functions ---


def _code_structure_impl(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse Python file and return AST structure.

    Returns (structure_dict, metrics_dict).
    """
    import ast as ast_mod

    start_ms = time.time() * 1000
    file_path = _normalize_path(path)
    assert file_path.suffix == ".py", (
        f"Only .py files supported for AST analysis (got {file_path.suffix})"
    )
    assert file_path.exists(), f"File not found: {path}"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast_mod.parse(f.read())
        structure: dict[str, list[Any]] = {
            "imports": [],
            "classes": [],
            "functions": [],
        }
        for node in ast_mod.walk(tree):
            if isinstance(node, ast_mod.Import):
                for name in node.names:
                    structure["imports"].append(name.name)
            elif isinstance(node, ast_mod.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    structure["imports"].append(f"{module}.{name.name}")
            elif isinstance(node, ast_mod.FunctionDef):
                structure["functions"].append(
                    {
                        "name": node.name,
                        "lineno": node.lineno,
                        "args": [a.arg for a in node.args.args],
                    }
                )
            elif isinstance(node, ast_mod.ClassDef):
                methods = [
                    n.name for n in node.body if isinstance(n, ast_mod.FunctionDef)
                ]
                structure["classes"].append(
                    {
                        "name": node.name,
                        "lineno": node.lineno,
                        "methods": methods,
                    }
                )
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "ast_structure",
            path,
            metrics=f"latency_ms={latency_ms} status=success",
        )
        return structure, {"status": "success", "latency_ms": latency_ms, "path": path}
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        return {"error": f"Failed to parse AST: {e}"}, {
            "status": "error",
            "latency_ms": latency_ms,
            "path": path,
        }


def _code_docstrings_impl(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract docstrings from Python file using AST.

    Returns (docstrings_dict, metrics_dict).
    """
    import ast as ast_mod

    start_ms = time.time() * 1000
    file_path = _normalize_path(path)
    assert file_path.suffix == ".py", (
        f"Only .py files supported for docstring extraction (got {file_path.suffix})"
    )
    assert file_path.exists(), f"File not found: {path}"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast_mod.parse(f.read())
        docs: dict[str, Any] = {
            "module": ast_mod.get_docstring(tree),
            "classes": {},
            "functions": {},
        }
        for node in ast_mod.walk(tree):
            if isinstance(node, ast_mod.FunctionDef):
                docs["functions"][node.name] = ast_mod.get_docstring(node)
            elif isinstance(node, ast_mod.ClassDef):
                class_doc = ast_mod.get_docstring(node)
                method_docs = {
                    item.name: ast_mod.get_docstring(item)
                    for item in node.body
                    if isinstance(item, ast_mod.FunctionDef)
                }
                docs["classes"][node.name] = {"doc": class_doc, "methods": method_docs}
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "ast_docstrings",
            path,
            metrics=f"latency_ms={latency_ms} status=success",
        )
        return docs, {"status": "success", "latency_ms": latency_ms, "path": path}
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        return {"error": f"Failed to extract docstrings: {e}"}, {
            "status": "error",
            "latency_ms": latency_ms,
            "path": path,
        }


# --- Search functions ---


def _search_impl(
    pattern: str,
    path: str = ".",
    context_lines: int = 0,
    before_context: int = 0,
    after_context: int = 0,
    include: str = "",
) -> tuple[list[str], dict[str, Any]]:
    """Search for text content within files.

    Uses rg (ripgrep) if available, falls back to git grep, then Python.

    Returns (results_list, metrics_dict).
    """
    start_ms = time.time() * 1000
    root_path = _normalize_path(path)
    results: list[str] = []

    if root_path.is_file():
        cwd_path = root_path.parent
        target_arg: str | None = root_path.name
    else:
        cwd_path = root_path
        target_arg = None

    # 1. Try Ripgrep (rg)
    rg_path = _get_tool_path("rg", "SFA_RG_PATH")
    if rg_path:
        cmd = [rg_path, "--line-number", "--column", "--no-heading", pattern]
        if include:
            cmd.extend(["--glob", include])
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        if before_context > 0:
            cmd.extend(["-B", str(before_context)])
        if after_context > 0:
            cmd.extend(["-A", str(after_context)])
        if target_arg:
            cmd.append(target_arg)
        output = _run_command(cmd, cwd=cwd_path)
        if output:
            latency_ms = round(time.time() * 1000 - start_ms, 2)
            results = output.splitlines()
            _log(
                "INFO",
                "search",
                f"rg: {pattern}",
                metrics=f"latency_ms={latency_ms} status=success matches={len(results)}",
            )
            return results, {
                "status": "success",
                "latency_ms": latency_ms,
                "tool": "rg",
                "pattern": pattern,
            }

    # 2. Try Git Grep
    git_path = _get_tool_path("git", "SFA_GIT_PATH")
    if git_path and (cwd_path / ".git").exists():
        cmd = [git_path, "grep", "-n", "--column", pattern]
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        if before_context > 0:
            cmd.extend(["-B", str(before_context)])
        if after_context > 0:
            cmd.extend(["-A", str(after_context)])
        if target_arg:
            cmd.extend(["--", target_arg])
        elif include:
            cmd.extend(["--", include])
        output = _run_command(cmd, cwd=cwd_path)
        if output:
            latency_ms = round(time.time() * 1000 - start_ms, 2)
            results = output.splitlines()
            _log(
                "INFO",
                "search",
                f"git: {pattern}",
                metrics=f"latency_ms={latency_ms} status=success matches={len(results)}",
            )
            return results, {
                "status": "success",
                "latency_ms": latency_ms,
                "tool": "git",
                "pattern": pattern,
            }

    # 3. Fallback: Python
    import fnmatch

    # Directories to exclude from search (performance optimization)
    EXCLUDE_DIRS = {
        ".git", "__pycache__", ".scripts_hold",
        "node_modules", "venv", ".venv", "env",
        "build", "dist", ".tox", ".pytest_cache",
        ".eggs", "*.egg-info", "htmlcov"
    }

    if root_path.is_file():
        files_to_search = [root_path]
        cwd_path = root_path.parent
    else:
        files_to_search = []
        for root, dirs, files in os.walk(root_path):
            # Skip excluded directories (modifies dirs in-place to prevent recursion)
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
            for filename in files:
                if include and not fnmatch.fnmatch(filename, include):
                    continue
                if Path(filename).suffix in GREP_EXTENSIONS:
                    files_to_search.append(Path(root) / filename)

    for file_path in files_to_search:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if pattern in line:
                        rel_path = str(file_path.relative_to(cwd_path)).replace(
                            "\\", "/"
                        )
                        results.append(f"{rel_path}:{i}:1:{line.strip()}")
        except Exception:
            continue

    latency_ms = round(time.time() * 1000 - start_ms, 2)
    _log(
        "INFO",
        "search",
        f"python: {pattern}",
        metrics=f"latency_ms={latency_ms} status=success matches={len(results)}",
    )
    return results, {
        "status": "success",
        "latency_ms": latency_ms,
        "tool": "python",
        "pattern": pattern,
    }


def _count_impl(pattern: str, path: str = ".") -> tuple[int, dict[str, Any]]:
    """Count total occurrences of a pattern.

    Returns (count, metrics_dict).
    """
    start_ms = time.time() * 1000
    root_path = _normalize_path(path)
    rg_path = _get_tool_path("rg", "SFA_RG_PATH")
    total = 0
    if rg_path:
        if root_path.is_file():
            cmd = [rg_path, "-c", pattern, root_path.name]
            cwd = root_path.parent
        else:
            cmd = [rg_path, "-c", pattern]
            cwd = root_path
        output = _run_command(cmd, cwd=cwd)
        for line in output.splitlines():
            parts = line.rsplit(":", 1)
            if len(parts) == 2 and parts[1].isdigit():
                total += int(parts[1])
            elif len(parts) == 1 and parts[0].isdigit():
                total += int(parts[0])
    else:
        results, _ = _search_impl(pattern, path)
        total = len(results)
    latency_ms = round(time.time() * 1000 - start_ms, 2)
    _log(
        "INFO",
        "count",
        f"{pattern} -> {total}",
        metrics=f"latency_ms={latency_ms} status=success",
    )
    return total, {
        "status": "success",
        "latency_ms": latency_ms,
        "pattern": pattern,
        "count": total,
    }


def _find_impl(
    pattern: str, path: str = ".", use_gitignore: bool = True, limit: int = 1000
) -> tuple[bool, str, str]:
    """Find files matching glob pattern.

    Returns (success, result_json, error_str).
    """
    root_path = _normalize_path(path)

    # Strategy 1: ripgrep (fast, gitignore-aware)
    rg_path = shutil.which("rg")
    if rg_path:
        cmd = [rg_path, "--files", "--glob", pattern]
        if not use_gitignore:
            cmd.append("--no-ignore")
        try:
            output = subprocess.check_output(
                cmd, cwd=root_path, text=True, errors="replace", timeout=10
            ).strip()
            results = (
                [str(root_path / p).replace("\\", "/") for p in output.splitlines()]
                if output
                else []
            )
            return True, json.dumps(results[:limit], indent=2), ""
        except subprocess.TimeoutExpired:
            return False, "", "rg timed out after 10s"
        except subprocess.CalledProcessError:
            pass

    # Strategy 2: git ls-files (for git repos)
    git_path = shutil.which("git")
    if git_path and (root_path / ".git").exists():
        try:
            output = subprocess.check_output(
                [git_path, "ls-files", pattern],
                cwd=root_path,
                text=True,
                errors="replace",
                timeout=10,
            ).strip()
            results = (
                [str(root_path / p).replace("\\", "/") for p in output.splitlines()]
                if output
                else []
            )
            return True, json.dumps(results[:limit], indent=2), ""
        except subprocess.TimeoutExpired:
            return False, "", "git ls-files timed out after 10s"
        except subprocess.CalledProcessError:
            pass

    # Strategy 3: Path.glob fallback
    results: list[str] = []
    try:
        for p in root_path.glob(pattern):
            if p.is_file():
                results.append(str(p).replace("\\", "/"))
                if len(results) >= limit:
                    break
    except Exception as e:
        return False, "", f"glob error: {e}"

    return True, json.dumps(results, indent=2), ""


# --- python_view function ---


def _python_view_impl(file_path: str) -> tuple[str, dict[str, Any]]:
    """AST-based Python script overview for AI consumption.

    Extracts: version, MCP server name, EXPOSED list, function signatures
    with type hints, top-level assertions (contracts), CONFIG contents,
    and PEP 723 dependencies.

    Falls back to regex for non-parseable files.

    Returns (result_str, metrics_dict).
    """
    import ast as ast_mod

    start_ms = time.time() * 1000
    path = _normalize_path(file_path)
    assert path.suffix == ".py", f"python_view requires a .py file (got {path.suffix})"
    assert path.exists(), f"file not found: {path}"

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Try AST first
    try:
        tree = ast_mod.parse(content)
    except SyntaxError:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        result = _python_view_regex(path, content, lines)
        return result, {
            "status": "success",
            "latency_ms": latency_ms,
            "path": file_path,
            "mode": "regex",
        }

    out: list[str] = []

    # Module docstring (first line of output)
    module_doc = ast_mod.get_docstring(tree)
    first_line = ""
    if module_doc:
        first_line = module_doc.split("\n")[0].strip()
        out.append(f"{path.name} — {first_line}")
    else:
        out.append(f"{path.name}")

    # Version — look for add_argument with -V/--version flag and version= kwarg
    # or VERSION = "X.Y.Z"
    version: str | None = None
    for line in lines:
        stripped = line.strip()
        if 'version="' in stripped and (
            '"-V"' in stripped or '"--version"' in stripped or "'-V'" in stripped
        ):
            m = re.search(r'version="([^"]+)"', stripped)
            if m:
                version = m.group(1)
                break
        if stripped.startswith("VERSION") and "=" in stripped:
            m = re.search(r'"([^"]+)"', stripped)
            if m and not version:
                version = m.group(1)

    # MCP server name — look for the FastMCP constructor call
    mcp_name: str | None = None
    for line in lines:
        stripped_l = line.strip()
        # Skip comments and string-check lines
        if (
            stripped_l.startswith("#")
            or "in line" in stripped_l
            or "in stripped" in stripped_l
        ):
            continue
        if "FastMCP(" in line:
            m = re.search(r'=\s*FastMCP\(["\']([^"\']+)', line)
            if m:
                mcp_name = m.group(1)
                break

    if version:
        out[0] = f"{path.name} v{version}" + (f" — {first_line}" if module_doc else "")
    if mcp_name:
        out.append(f"MCP: {mcp_name}")

    # EXPOSED list
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("EXPOSED"):
            out.append(f"\n{stripped}")
            break

    # Functions with signatures and top-level asserts
    out.append("\nFUNCTIONS:")
    for node in ast_mod.iter_child_nodes(tree):
        if isinstance(node, (ast_mod.FunctionDef, ast_mod.AsyncFunctionDef)):
            func_name = node.name
            args_list: list[str] = []
            defaults_offset = len(node.args.args) - len(node.args.defaults)
            for i, arg in enumerate(node.args.args):
                ann = ""
                if arg.annotation:
                    ann = f": {ast_mod.unparse(arg.annotation)}"
                default_idx = i - defaults_offset
                default = ""
                if 0 <= default_idx < len(node.args.defaults):
                    default = f" = {ast_mod.unparse(node.args.defaults[default_idx])}"
                args_list.append(f"{arg.arg}{ann}{default}")

            returns = ""
            if node.returns:
                returns = f" -> {ast_mod.unparse(node.returns)}"

            sig = f"  {func_name}({', '.join(args_list)}){returns}"
            out.append(sig)

            # Extract top-level assert statements (contracts/preconditions)
            for stmt in node.body:
                if isinstance(stmt, ast_mod.Assert):
                    test_str = ast_mod.unparse(stmt.test)
                    msg_str = f" — {ast_mod.unparse(stmt.msg)}" if stmt.msg else ""
                    out.append(f"    assert {test_str}{msg_str}")
                elif isinstance(stmt, ast_mod.Expr) and isinstance(
                    stmt.value, (ast_mod.Constant, ast_mod.JoinedStr)
                ):
                    continue  # Skip docstrings
                else:
                    break  # Stop at first non-assert, non-docstring statement

    # CONFIG section
    for node in ast_mod.iter_child_nodes(tree):
        if isinstance(node, ast_mod.Assign):
            for target in node.targets:
                if isinstance(target, ast_mod.Name) and target.id == "CONFIG":
                    out.append("\nCONFIG:")
                    if isinstance(node.value, ast_mod.Dict):
                        for key, val in zip(node.value.keys, node.value.values):
                            if key:
                                k = ast_mod.unparse(key)
                                v = ast_mod.unparse(val)
                                out.append(f"  {k}: {v}")

    # PEP 723 dependencies
    in_script_block = False
    for line in lines:
        if line.strip() == "# /// script":
            in_script_block = True
        elif line.strip() == "# ///" and in_script_block:
            break
        elif in_script_block and "dependencies" in line:
            out.append(f"\nDEPENDENCIES: {line.strip().lstrip('# ')}")

    latency_ms = round(time.time() * 1000 - start_ms, 2)
    _log(
        "INFO",
        "python_view",
        file_path,
        metrics=f"latency_ms={latency_ms} status=success",
    )
    return "\n".join(out), {
        "status": "success",
        "latency_ms": latency_ms,
        "path": file_path,
        "mode": "ast",
    }


def _python_view_regex(path: Path, content: str, lines: list[str]) -> str:
    """Regex fallback for python_view when AST parsing fails."""
    out = [f"{path.name} (AST parse failed — regex fallback)"]

    # Version
    for line in lines:
        m = re.search(r'version="([^"]+)"', line)
        if m:
            out.append(f"Version: {m.group(1)}")
            break

    # EXPOSED
    for line in lines:
        if line.strip().startswith("EXPOSED"):
            out.append(f"\n{line.strip()}")
            break

    # Functions
    out.append("\nFUNCTIONS:")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            out.append(f"  {i}: {stripped.rstrip(':')}")

    return "\n".join(out)


# =============================================================================
# CLI INTERFACE
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Code reconnaissance and search — scout file structure, find files, search content"
    )
    parser.add_argument("-V", "--version", action="version", version="1.1.0")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # --- mcp-stdio ---
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")

    # --- python_view ---
    p_pv = subparsers.add_parser(
        "python-view", help="AI-readable Python script overview"
    )
    p_pv.add_argument("file_path", nargs="?", default=None)

    # --- scout ---
    p_scout = subparsers.add_parser("scout", help="Unified code reconnaissance")
    p_scout.add_argument("file_path", nargs="?", default=None)
    p_scout.add_argument("pattern", nargs="?", default=None)
    p_scout.add_argument("-p", "--pattern", dest="pattern_flag", default=None)

    # --- ast ---
    p_ast = subparsers.add_parser("ast", help="Python AST structure")
    p_ast.add_argument("file_path", nargs="?", default=None)

    # --- docstrings ---
    p_docs = subparsers.add_parser("docstrings", help="Extract docstrings")
    p_docs.add_argument("file_path", nargs="?", default=None)

    # --- ast-structure (alias matching EXPOSED name) ---
    p_ast2 = subparsers.add_parser("ast-structure", help="Python AST structure (alias)")
    p_ast2.add_argument("file_path", nargs="?", default=None)

    # --- ast-docstrings (alias matching EXPOSED name) ---
    p_docs2 = subparsers.add_parser("ast-docstrings", help="Extract docstrings (alias)")
    p_docs2.add_argument("file_path", nargs="?", default=None)

    # --- Scout shortcut commands (literal add_parser for sfa_check parity) ---
    p_fn = subparsers.add_parser("functions", help="Function/method definitions")
    p_fn.add_argument("file_path", nargs="?", default=None)
    p_cl = subparsers.add_parser("classes", help="Class definitions")
    p_cl.add_argument("file_path", nargs="?", default=None)
    p_im = subparsers.add_parser("imports", help="Import statements")
    p_im.add_argument("file_path", nargs="?", default=None)
    p_co = subparsers.add_parser("comments", help="Comments")
    p_co.add_argument("file_path", nargs="?", default=None)
    p_td = subparsers.add_parser("todos", help="TODO/FIXME/HACK/XXX markers")
    p_td.add_argument("file_path", nargs="?", default=None)
    p_dc = subparsers.add_parser("decorators", help="Decorator lines")
    p_dc.add_argument("file_path", nargs="?", default=None)
    p_ty = subparsers.add_parser("types", help="Type hints")
    p_ty.add_argument("file_path", nargs="?", default=None)
    p_er = subparsers.add_parser("errors", help="Error handling")
    p_er.add_argument("file_path", nargs="?", default=None)
    p_te = subparsers.add_parser("tests", help="Test functions")
    p_te.add_argument("file_path", nargs="?", default=None)
    p_mn = subparsers.add_parser("main", help="Main block")
    p_mn.add_argument("file_path", nargs="?", default=None)
    p_mb = subparsers.add_parser("main-block", help="Main block (alias)")
    p_mb.add_argument("file_path", nargs="?", default=None)
    p_hd = subparsers.add_parser("header", help="File header")
    p_hd.add_argument("file_path", nargs="?", default=None)
    p_ex = subparsers.add_parser("exports", help="Exports")
    p_ex.add_argument("file_path", nargs="?", default=None)
    scout_shortcuts = {
        "functions",
        "classes",
        "imports",
        "comments",
        "todos",
        "decorators",
        "types",
        "errors",
        "tests",
        "main",
        "main-block",
        "header",
        "exports",
    }

    # --- find (primary) and glob (alias) ---
    def _add_find_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("pattern")
        p.add_argument("path", nargs="?", default=".")
        p.add_argument("-n", "--no-ignore", action="store_false", dest="use_gitignore")

    p_find = subparsers.add_parser("find", help="Find files by glob pattern")
    _add_find_args(p_find)
    p_glob = subparsers.add_parser("glob", help="Find files by glob pattern (alias)")
    _add_find_args(p_glob)

    # --- search (primary) and grep (alias) ---
    def _add_search_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("pattern_or_path")
        p.add_argument("pattern2", nargs="?", default=None)
        p.add_argument("-d", "--path", dest="search_path", default=None)
        p.add_argument(
            "-i", "--include", "--glob", "--filter", dest="include", default=None
        )
        p.add_argument(
            "-A", "--after-context", type=int, default=0, help="Lines after match"
        )
        p.add_argument(
            "-B", "--before-context", type=int, default=0, help="Lines before match"
        )
        p.add_argument(
            "-C", "--context", type=int, default=0, help="Lines before and after match"
        )

    p_search = subparsers.add_parser("search", help="Search file contents")
    _add_search_args(p_search)
    p_grep = subparsers.add_parser("grep", help="Search file contents (alias)")
    _add_search_args(p_grep)

    # --- count ---
    p_count = subparsers.add_parser("count", help="Count pattern matches")
    p_count.add_argument("pattern")
    p_count.add_argument("path", nargs="?", default=".")
    p_count.add_argument(
        "-i", "--include", "--glob", "--filter", dest="include", default=None
    )

    args = parser.parse_args()

    try:
        # --- Dispatch ---
        if args.command == "mcp-stdio":
            _run_mcp()

        elif args.command == "python-view":
            if not args.file_path and not sys.stdin.isatty():
                args.file_path = sys.stdin.read().strip()
            assert args.file_path, (
                "file_path required. Usage: sfa_scout.py python-view <file>"
            )
            result, _ = _python_view_impl(args.file_path)
            print(result)

        elif args.command == "scout":
            if not args.file_path and not sys.stdin.isatty():
                args.file_path = sys.stdin.read().strip()
            assert args.file_path, (
                "file_path required. Usage: sfa_scout.py scout <file> [pattern]"
            )
            pattern = args.pattern_flag or args.pattern
            result, _ = _scout_impl(args.file_path, pattern)
            print(result)

        elif args.command in ("ast", "ast-structure"):
            if not args.file_path and not sys.stdin.isatty():
                args.file_path = sys.stdin.read().strip()
            assert args.file_path, "file_path required. Usage: sfa_scout.py ast <file>"
            structure, _ = _code_structure_impl(args.file_path)
            print(json.dumps(structure, indent=2))

        elif args.command in ("docstrings", "ast-docstrings"):
            if not args.file_path and not sys.stdin.isatty():
                args.file_path = sys.stdin.read().strip()
            assert args.file_path, (
                "file_path required. Usage: sfa_scout.py docstrings <file>"
            )
            docs, _ = _code_docstrings_impl(args.file_path)
            print(json.dumps(docs, indent=2))

        elif args.command in scout_shortcuts:
            if not args.file_path and not sys.stdin.isatty():
                args.file_path = sys.stdin.read().strip()
            assert args.file_path, (
                f"file_path required. Usage: sfa_scout.py {args.command} <file>"
            )
            # Normalize CLI command name to scout pattern (main-block -> main)
            scout_pattern = args.command.replace("-", "_").replace("main_block", "main")
            result, _ = _scout_impl(args.file_path, scout_pattern)
            print(result)

        elif args.command in ("find", "glob"):
            success, out, err = _find_impl(args.pattern, args.path, args.use_gitignore)
            if success:
                files = json.loads(out)
                if files:
                    print(out)
                else:
                    print(f"No files matching '{args.pattern}'", file=sys.stderr)
                    sys.exit(2)
            else:
                print(f"Error: {err}", file=sys.stderr)
                sys.exit(1)

        elif args.command in ("search", "grep"):
            # Smart argument detection: search FILE PATTERN vs search PATTERN
            first_arg = args.pattern_or_path
            second_arg = args.pattern2
            explicit_path = args.search_path

            if explicit_path:
                pattern = first_arg
                path = explicit_path
            elif second_arg:
                path = first_arg
                pattern = second_arg
            else:
                pattern = first_arg
                path = "."

            results, _ = _search_impl(
                pattern,
                path,
                args.context,
                args.before_context,
                args.after_context,
                include=args.include or "",
            )
            if results:
                MAX_LINES = 500
                MAX_LINE_LEN = 500
                capped = [
                    line[:MAX_LINE_LEN] + ("..." if len(line) > MAX_LINE_LEN else "")
                    for line in results[:MAX_LINES]
                ]
                suffix = (
                    f"\n... ({len(results) - MAX_LINES} more results truncated)"
                    if len(results) > MAX_LINES
                    else ""
                )
                print("\n".join(capped) + suffix)
            else:
                print(f"No matches for '{pattern}'", file=sys.stderr)
                sys.exit(2)

        elif args.command == "count":
            total, _ = _count_impl(args.pattern, args.path)
            print(json.dumps({"pattern": args.pattern, "count": total}))

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

    mcp = FastMCP("fs")

    @mcp.tool()
    def python_view(file_path: str) -> str:
        """AST-based Python script overview for AI consumption.

        Extracts version, MCP server name, EXPOSED list, function signatures
        with type hints, top-level assertions (contracts), CONFIG contents,
        and PEP 723 dependencies.

        Args:
            file_path: Path to a .py file
        """
        result, _ = _python_view_impl(file_path)
        return result

    @mcp.tool()
    def scout(coords: str, pattern: str = "") -> str:
        """Code reconnaissance — analyze file structure and content.

        Unified dispatcher for all scout operations. Without a pattern,
        returns file structure overview. With a pattern, filters by type
        (functions, classes, imports, etc.) or searches by regex.

        Args:
            coords: File path, optionally with line spec (file:line or file:start-end)
            pattern: Filter: functions, classes, imports, docstrings, comments,
                     todos, decorators, types, errors, tests, main, header, exports,
                     section:NAME, or a regex pattern
        """
        result, _ = _scout_impl(coords, pattern if pattern else None)
        return result

    @mcp.tool()
    def functions(path: str) -> str:
        """List function/method definitions in a file.

        Args:
            path: File path to analyze
        """
        result, _ = _scout_impl(path, "functions")
        return result

    @mcp.tool()
    def classes(path: str) -> str:
        """List class definitions in a file.

        Args:
            path: File path to analyze
        """
        result, _ = _scout_impl(path, "classes")
        return result

    @mcp.tool()
    def imports(path: str) -> str:
        """List import statements in a file.

        Args:
            path: File path to analyze
        """
        result, _ = _scout_impl(path, "imports")
        return result

    @mcp.tool()
    def docstrings(path: str) -> str:
        """Extract all docstrings from a Python file.

        Args:
            path: File path to analyze
        """
        result, _ = _scout_impl(path, "docstrings")
        return result

    @mcp.tool()
    def comments(path: str) -> str:
        """Extract comments from a code file.

        Args:
            path: File path to analyze
        """
        result, _ = _scout_impl(path, "comments")
        return result

    @mcp.tool()
    def todos(path: str) -> str:
        """Find TODO, FIXME, HACK, XXX markers in a file.

        Args:
            path: File path to analyze
        """
        result, _ = _scout_impl(path, "todos")
        return result

    @mcp.tool()
    def decorators(path: str) -> str:
        """Extract @decorator lines from a Python file.

        Args:
            path: File path to analyze
        """
        result, _ = _scout_impl(path, "decorators")
        return result

    @mcp.tool()
    def types(path: str) -> str:
        """Extract type hints, TypedDict, dataclass, Protocol from a Python file.

        Args:
            path: File path to analyze
        """
        result, _ = _scout_impl(path, "types")
        return result

    @mcp.tool()
    def errors(path: str) -> str:
        """Extract try/except/raise statements from a Python file.

        Args:
            path: File path to analyze
        """
        result, _ = _scout_impl(path, "errors")
        return result

    @mcp.tool()
    def tests(path: str) -> str:
        """Extract test functions (test_*, *_test) from a file.

        Args:
            path: File path to analyze
        """
        result, _ = _scout_impl(path, "tests")
        return result

    @mcp.tool()
    def main_block(path: str) -> str:
        """Find if __name__ == '__main__' block in a file.

        Args:
            path: File path to analyze
        """
        result, _ = _scout_impl(path, "main")
        return result

    @mcp.tool()
    def header(path: str) -> str:
        """Extract module docstring / file header.

        Args:
            path: File path to analyze
        """
        result, _ = _scout_impl(path, "header")
        return result

    @mcp.tool()
    def exports(path: str) -> str:
        """Extract __all__ or module.exports declarations.

        Args:
            path: File path to analyze
        """
        result, _ = _scout_impl(path, "exports")
        return result

    @mcp.tool()
    def ast_structure(path: str) -> str:
        """Python AST structure analysis — returns JSON with functions, classes, imports.

        Args:
            path: Path to a .py file
        """
        structure, _ = _code_structure_impl(path)
        return json.dumps(structure, indent=2)

    @mcp.tool()
    def ast_docstrings(path: str) -> str:
        """Extract all docstrings from a Python file using AST — returns JSON.

        Args:
            path: Path to a .py file
        """
        docs, _ = _code_docstrings_impl(path)
        return json.dumps(docs, indent=2)

    @mcp.tool()
    def search(
        pattern: str, path: str = ".", include: str = "", context: int = 0
    ) -> str:
        """Search for pattern in files. Returns matches with file:line:content.

        Args:
            pattern: Text or regex pattern to search for
            path: Directory or file to search in (default: current directory)
            include: File glob filter (e.g. "*.py")
            context: Lines of context around matches (default: 0)
        """
        results, _ = _search_impl(pattern, path, context_lines=context, include=include)
        if not results:
            return "No matches found"
        MAX_LINES = 500
        MAX_LINE_LEN = 500
        capped = [
            line[:MAX_LINE_LEN] + ("..." if len(line) > MAX_LINE_LEN else "")
            for line in results[:MAX_LINES]
        ]
        suffix = (
            f"\n... ({len(results) - MAX_LINES} more results truncated)"
            if len(results) > MAX_LINES
            else ""
        )
        return "\n".join(capped) + suffix

    @mcp.tool()
    def find(pattern: str, path: str = ".") -> str:
        """Find files matching glob pattern.

        Args:
            pattern: Glob pattern (e.g. "*.py", "test_*.py")
            path: Directory to search in (default: current directory)
        """
        success, result, error = _find_impl(pattern, path)
        return (
            (result if result else "No files found") if success else f"ERROR: {error}"
        )

    @mcp.tool()
    def count(pattern: str, path: str = ".") -> str:
        """Count matches of pattern in files.

        Args:
            pattern: Text or regex pattern to count
            path: Directory or file to search in (default: current directory)
        """
        total, _ = _count_impl(pattern, path)
        return str(total)

    print("fs MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
