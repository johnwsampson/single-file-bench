#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp"]
# ///
"""Destructive file operations: write, edit, delete with automatic backups."""

import os
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
    "write",
    "create",
    "append",
    "write_line",
    "write_string",
    "delete_range",
    "cp",
    "mv",
    "rm",
    "target",
    "confirm",
    "recover",
]

BACKUP_DIR = Path.home() / ".sfb" / "backups"

CONFIG = {
    "backup_dir": BACKUP_DIR,
    "context_lines": 3,
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
    """Check if file matches git HEAD. Returns True if clean, False if dirty or git fails."""
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


def _get_content(arg: str) -> str:
    """Return arg or stdin if arg is '-'. For pipe support."""
    if arg == "-":
        return sys.stdin.read()
    return arg


def _report(msg: str):
    """Print action report to stderr (keeps stdout clean for pipes)."""
    print(msg, file=sys.stderr)


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} B"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _get_dir_stats(path: Path) -> tuple[int, int, int]:
    """Get directory stats: (file_count, dir_count, total_bytes)."""
    file_count = 0
    dir_count = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            try:
                total_bytes += item.stat().st_size
            except OSError:
                pass
        elif item.is_dir():
            dir_count += 1
    return file_count, dir_count, total_bytes


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


def read_surgical_content(content: str) -> str:
    """Read content from @file reference or return literal."""
    if content.startswith("@@"):
        return content[1:]
    if content.startswith("@"):
        file_path = Path(content[1:])
        if not file_path.exists():
            raise FileNotFoundError(f"Content file not found: {file_path}")
        return file_path.read_text(encoding="utf-8")
    return content


# --- Core impl functions ---


def _write_impl(
    file_path: str, content: str, backup: bool = True
) -> tuple[bool, str, str]:
    """Write content to file with optional backup.

    Returns (success, message, error).
    """
    start_ms = time.time() * 1000
    path = _normalize_path(file_path)
    is_new = not path.exists()

    backup_path = None
    if backup and path.exists():
        backup_path = _create_backup(path) if not _is_git_clean(path) else None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        # BDA: Verify write succeeded
        if not path.exists():
            return False, "", f"BDA failed: {path} not created after write"
        written = path.read_text(encoding="utf-8")
        if written != content:
            return (
                False,
                "",
                f"BDA failed: content mismatch in {path} "
                f"(wrote {len(content)}, read {len(written)})",
            )

        size = len(content.encode("utf-8"))
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "write",
            str(path),
            metrics=f"latency_ms={latency_ms} status=success bytes={size}",
        )

        if is_new:
            return (
                True,
                f"Created {path} ({size} bytes) [verified]\n     (new file)",
                "",
            )
        else:
            backup_line = f"\n     Backup: {backup_path}" if backup_path else ""
            return (
                True,
                f"Wrote {path} ({size} bytes) [verified]{backup_line}",
                "",
            )
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "write",
            str(path),
            metrics=f"latency_ms={latency_ms} status=error",
            detail=str(e),
        )
        return False, "", str(e)


def _create_impl(
    file_path: str, content: str = "", overwrite: bool = False
) -> tuple[bool, str, str]:
    """Create a new file. Fails if exists unless overwrite=True.

    Returns (success, message, error).
    """
    start_ms = time.time() * 1000
    path = _normalize_path(file_path)
    is_new = not path.exists()

    if path.exists() and not overwrite:
        return (
            False,
            "",
            f"File already exists: {file_path}. Set overwrite=True to replace.",
        )

    # Backup if overwriting existing file
    backup_path = None
    if path.exists():
        backup_path = _create_backup(path) if not _is_git_clean(path) else None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        # BDA: Verify write succeeded
        if not path.exists():
            return False, "", f"BDA failed: {file_path} not created after write"
        written = path.read_text(encoding="utf-8")
        if written != content:
            return (
                False,
                "",
                f"BDA failed: content mismatch in {file_path} "
                f"(wrote {len(content)}, read {len(written)})",
            )

        size = len(content.encode("utf-8"))
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "create",
            str(path),
            metrics=f"latency_ms={latency_ms} status=success bytes={size}",
        )

        if is_new:
            return (
                True,
                f"Created {path} ({size} bytes) [verified]\n     (new file)",
                "",
            )
        else:
            backup_line = f"\n     Backup: {backup_path}" if backup_path else ""
            return (
                True,
                f"Created {path} ({size} bytes) [verified]{backup_line}",
                "",
            )
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "create",
            str(path),
            metrics=f"latency_ms={latency_ms} status=error",
            detail=str(e),
        )
        return False, "", str(e)


def _append_impl(file_path: str, content: str) -> str:
    """Append content to file with automatic backup.

    Returns action report string.
    """
    start_ms = time.time() * 1000
    path = _normalize_path(file_path)
    if not path.exists():
        return f"Error: File not found: {file_path}"
    try:
        backup_path = _create_backup(path) if not _is_git_clean(path) else None
        bytes_added = 0
        with open(path, "a", encoding="utf-8") as f:
            if not content.startswith("\n"):
                f.write("\n")
                bytes_added += 1
            f.write(content)
            bytes_added += len(content.encode("utf-8"))

        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "append",
            str(path),
            metrics=f"latency_ms={latency_ms} status=success bytes={bytes_added}",
        )
        backup_line = f"\n     Backup: {backup_path}" if backup_path else ""
        return f"[OK] Appended to {path} ({bytes_added} bytes added){backup_line}"
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "append",
            str(path),
            metrics=f"latency_ms={latency_ms} status=error",
            detail=str(e),
        )
        return f"Error appending to file: {e}"


def _write_line_impl(file_path: str, line_no: int, new_content: str) -> str:
    """Replace specific line in file with automatic backup.

    Returns action report string.
    """
    start_ms = time.time() * 1000
    path = _normalize_path(file_path)
    if not path.exists():
        return f"Error: File not found: {file_path}"
    try:
        backup_path = _create_backup(path) if not _is_git_clean(path) else None
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if line_no < 1 or line_no > len(lines):
            return f"Error: Line {line_no} out of range (1-{len(lines)})"

        original_line = lines[line_no - 1].rstrip()
        if lines[line_no - 1].endswith("\n") and not new_content.endswith("\n"):
            new_content += "\n"
        lines[line_no - 1] = new_content

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "write_line",
            f"{path}:{line_no}",
            metrics=f"latency_ms={latency_ms} status=success",
        )
        out = [
            f"[OK] Replaced line {line_no} in {path}",
            f"     Old: {original_line}",
            f"     New: {new_content.rstrip()}",
        ]
        if backup_path:
            out.append(f"Backup: {backup_path}")
        return "\n".join(out)
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "write_line",
            f"{path}:{line_no}",
            metrics=f"latency_ms={latency_ms} status=error",
            detail=str(e),
        )
        return f"Error replacing line: {e}"


def _write_string_impl(
    file_path: str, old: str, new: str, replace_all: bool = True
) -> str:
    """Replace string in file. Supports multiline matches.

    Returns action report string.
    """
    start_ms = time.time() * 1000
    path = _normalize_path(file_path)
    if not path.exists():
        return f"Error: File not found: {file_path}"
    try:
        backup_path = _create_backup(path) if not _is_git_clean(path) else None
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        total_count = content.count(old)
        if total_count == 0:
            return f"Error: String not found in {file_path}"

        first_match_pos = content.find(old)
        first_match_line = content[:first_match_pos].count("\n") + 1

        if replace_all:
            new_content = content.replace(old, new)
            replaced_count = total_count
        else:
            new_content = content.replace(old, new, 1)
            replaced_count = 1

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "write_string",
            str(path),
            metrics=(
                f"latency_ms={latency_ms} status=success replacements={replaced_count}"
            ),
        )

        old_snippet = old[:20] + "..." if len(old) > 20 else old
        new_snippet = new[:20] + "..." if len(new) > 20 else new
        is_multiline = "\n" in old

        out = [f"[OK] {replaced_count} replacement(s) in {path}"]
        if is_multiline:
            line_count = old.count("\n") + 1
            out.append(
                f"     L{first_match_line}-"
                f"{first_match_line + line_count - 1}: "
                f"{old_snippet} -> {new_snippet}"
            )
        else:
            out.append(f"     L{first_match_line}: {old_snippet} -> {new_snippet}")
        if replace_all and total_count > 1:
            out.append(f"     (replaced all {total_count} occurrences)")
        elif not replace_all and total_count > 1:
            out.append(f"     ({total_count - 1} more occurrence(s) not replaced)")
        if backup_path:
            out.append(f"Backup: {backup_path}")
        return "\n".join(out)
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "write_string",
            str(path),
            metrics=f"latency_ms={latency_ms} status=error",
            detail=str(e),
        )
        return f"Error replacing string: {e}"


def _delete_range_impl(file_path: str, start: int, end: int) -> str:
    """Delete lines from start to end inclusive with automatic backup.

    Returns action report string.
    """
    start_ms = time.time() * 1000
    path = _normalize_path(file_path)
    if not path.exists():
        return f"Error: File not found: {file_path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)
        if start < 1 or end > total_lines or start > end:
            return (
                f"Error: Invalid line range {start}-{end} "
                f"(file has {total_lines} lines)"
            )

        backup_path = _create_backup(path) if not _is_git_clean(path) else None

        deleted_lines = lines[start - 1 : end]
        deleted_count = len(deleted_lines)
        first_line = deleted_lines[0].rstrip() if deleted_lines else ""
        last_line = deleted_lines[-1].rstrip() if deleted_lines else ""

        del lines[start - 1 : end]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "delete_range",
            f"{path}:{start}-{end}",
            metrics=(f"latency_ms={latency_ms} status=success deleted={deleted_count}"),
        )

        out = [f"[OK] Deleted lines {start}-{end} ({deleted_count} lines) from {path}"]
        if deleted_count == 1:
            out.append(f"     Removed: {first_line[:60]}...")
        else:
            out.append(f"     First: {first_line[:50]}...")
            out.append(f"     Last:  {last_line[:50]}...")
        out.append(f"     File now has {total_lines - deleted_count} lines")
        if backup_path:
            out.append(f"Backup: {backup_path}")
        return "\n".join(out)
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "delete_range",
            f"{path}:{start}-{end}",
            metrics=f"latency_ms={latency_ms} status=error",
            detail=str(e),
        )
        return f"Error deleting lines: {e}"


def _cp_impl(src: str, dest: str) -> str:
    """Copy file or directory with backup.

    Returns action report string.
    """
    start_ms = time.time() * 1000
    src_path = _normalize_path(src)
    dest_path = _normalize_path(dest)

    if not src_path.exists():
        return f"[FAIL] Source not found: {src_path}"

    backup_path = None
    if dest_path.exists():
        backup_path = (
            _create_backup(dest_path) if not _is_git_clean(dest_path) else None
        )

    try:
        if src_path.is_dir():
            file_count, dir_count, total_bytes = _get_dir_stats(src_path)
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
            result = f"[OK] Copied directory: {src_path} -> {dest_path}"
            result += (
                f"\n     {file_count} file(s), {dir_count} subdir(s), "
                f"{_format_size(total_bytes)}"
            )
        else:
            size = src_path.stat().st_size
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)
            result = (
                f"[OK] Copied file: {src_path} -> {dest_path} ({_format_size(size)})"
            )

        if backup_path:
            result += f"\n     Backup: {backup_path}"

        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "cp",
            f"{src_path} -> {dest_path}",
            metrics=f"latency_ms={latency_ms} status=success",
        )
        return result
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "cp",
            f"{src_path} -> {dest_path}",
            metrics=f"latency_ms={latency_ms} status=error",
            detail=str(e),
        )
        return f"[FAIL] Copy failed: {e}"


def _mv_impl(src: str, dest: str) -> str:
    """Move/rename file or directory with backup.

    Returns action report string.
    """
    start_ms = time.time() * 1000
    src_path = _normalize_path(src)
    dest_path = _normalize_path(dest)

    if not src_path.exists():
        return f"[FAIL] Source not found: {src_path}"

    if src_path.is_dir():
        file_count, dir_count, total_bytes = _get_dir_stats(src_path)
        is_dir = True
    else:
        total_bytes = src_path.stat().st_size
        is_dir = False

    backup_path = None
    if dest_path.exists():
        backup_path = (
            _create_backup(dest_path) if not _is_git_clean(dest_path) else None
        )

    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dest_path))

        if is_dir:
            result = f"[OK] Moved directory: {src_path} -> {dest_path}"
            result += (
                f"\n     {file_count} file(s), {dir_count} subdir(s), "
                f"{_format_size(total_bytes)}"
            )
        else:
            result = (
                f"[OK] Moved file: {src_path} -> {dest_path} "
                f"({_format_size(total_bytes)})"
            )

        if backup_path:
            result += f"\n     Backup (overwritten): {backup_path}"

        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "mv",
            f"{src_path} -> {dest_path}",
            metrics=f"latency_ms={latency_ms} status=success",
        )
        return result
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "mv",
            f"{src_path} -> {dest_path}",
            metrics=f"latency_ms={latency_ms} status=error",
            detail=str(e),
        )
        return f"[FAIL] Move failed: {e}"


def _rm_impl(file_path: str, confirm: bool = False) -> str:
    """Delete file or directory with backup. Directories require confirm=True.

    Returns action report string.
    """
    start_ms = time.time() * 1000
    target_path = _normalize_path(file_path)

    if not target_path.exists():
        return f"[FAIL] Not found: {target_path}"

    is_dir = target_path.is_dir()

    # Require confirm for directories
    if is_dir and not confirm:
        try:
            file_count, dir_count, total_bytes = _get_dir_stats(target_path)
            return (
                f"[ASSESS] Directory: {target_path}\n"
                f"     {file_count} file(s), {dir_count} subdir(s), "
                f"{_format_size(total_bytes)}\n"
                f"     Use --confirm to delete"
            )
        except Exception as e:
            return f"[FAIL] Cannot assess directory: {e}"

    # Backup before delete
    backup_path = (
        (_create_backup(target_path) if not _is_git_clean(target_path) else None)
        if not is_dir
        else None
    )

    try:
        if is_dir:
            # For directories, back up as a tarball
            import tarfile

            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = BACKUP_DIR / f"{target_path.name}.{ts}.tar.gz"
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(target_path, arcname=target_path.name)
            shutil.rmtree(target_path)
            item_type = "directory"
        else:
            target_path.unlink()
            item_type = "file"

        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "rm",
            str(target_path),
            metrics=f"latency_ms={latency_ms} status=success type={item_type}",
        )

        result = f"[OK] Deleted {item_type}: {target_path}"
        if backup_path:
            result += f"\n     Backup: {backup_path}"
        return result
    except Exception as e:
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "ERROR",
            "rm",
            str(target_path),
            metrics=f"latency_ms={latency_ms} status=error",
            detail=str(e),
        )
        return f"[FAIL] Delete failed: {e}"


# --- Surgical editing: target ---


def _target_item_assess(
    path: Path,
    old: str,
    new: str,
    line_start: int | None,
    line_end: int | None,
) -> str:
    """Assess item-mode replacements (preview only)."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines(keepends=True)
    total = len(lines)
    start = (line_start or 1) - 1
    end = min(line_end or total, total)

    replacements = []
    for i in range(start, end):
        line = lines[i]
        if old in line:
            new_line = line.replace(old, new)
            replacements.append(
                {
                    "line_num": i + 1,
                    "old_line": line.rstrip("\n"),
                    "new_line": new_line.rstrip("\n"),
                    "count": line.count(old),
                }
            )

    if not replacements:
        return f"No matches for '{old}' in {path}"

    total_count = sum(r["count"] for r in replacements)
    out = [
        f"Found {total_count} replacement(s) in {len(replacements)} line(s):",
        "",
        f"--- {path}",
        f"+++ {path}",
    ]
    for r in replacements:
        out.extend(
            [
                f"@@ line {r['line_num']} @@",
                f"-{r['old_line']}",
                f"+{r['new_line']}",
                "",
            ]
        )
    out.append(f"[ASSESS] {total_count} replacement(s) ready. Run strike to apply.")
    return "\n".join(out)


def _target_item_strike(
    path: Path,
    old: str,
    new: str,
    line_start: int | None,
    line_end: int | None,
) -> str:
    """Execute item-mode replacements."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines(keepends=True)
    start = (line_start or 1) - 1
    end = min(line_end or len(lines), len(lines))

    backup_path = _create_backup(path) if not _is_git_clean(path) else None

    changes = []
    total_count = 0
    for i in range(start, end):
        count = lines[i].count(old)
        if count:
            old_line = lines[i].rstrip()
            lines[i] = lines[i].replace(old, new)
            new_line = lines[i].rstrip()
            total_count += count
            changes.append((i + 1, old_line, new_line))

    if total_count == 0:
        return f"No matches for '{old}' in {path}"

    path.write_text("".join(lines), encoding="utf-8")

    coords_str = str(path)
    if line_start:
        coords_str += f":{line_start}"
        if line_end and line_end != line_start:
            coords_str += f"-{line_end}"

    out = [f"[OK] Strike at {coords_str}"]
    out.append(f"     {total_count} replacement(s) in {len(changes)} line(s)")
    for line_num, old_l, new_l in changes[:3]:
        old_display = old_l[:50] + "..." if len(old_l) > 50 else old_l
        new_display = new_l[:50] + "..." if len(new_l) > 50 else new_l
        out.append(f"     L{line_num}: {old_display}")
        out.append(f"         -> {new_display}")
    if len(changes) > 3:
        out.append(f"     ... and {len(changes) - 3} more line(s)")
    if backup_path:
        out.append(f"Backup: {backup_path}")
    return "\n".join(out)


def _target_location_assess(
    path: Path,
    line_start: int,
    line_end: int,
    new_content: str | None,
    mode: str,
) -> str:
    """Assess location-mode changes (preview only)."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines()
    total = len(lines)
    line_end = min(line_end, total)

    out = [
        f"--- {path}",
        f"+++ {path}",
        f"@@ lines {line_start}-{line_end} ({mode}) @@",
    ]

    if mode == "delete":
        for i in range(line_start - 1, line_end):
            out.append(f"-{lines[i]}")
        out.extend(
            [
                "",
                f"[ASSESS] {line_end - line_start + 1} line(s) will be deleted.",
            ]
        )
    elif mode == "insert":
        if line_start > 1:
            out.append(f" {lines[line_start - 2]}")
        if new_content:
            for line in new_content.splitlines():
                out.append(f"+{line}")
        if line_start <= total:
            out.append(f" {lines[line_start - 1]}")
        out.extend(
            [
                "",
                f"[ASSESS] Content will be inserted before line {line_start}.",
            ]
        )
    elif mode == "append":
        out.append(f" {lines[line_start - 1]}")
        if new_content:
            for line in new_content.splitlines():
                out.append(f"+{line}")
        if line_start < total:
            out.append(f" {lines[line_start]}")
        out.extend(
            [
                "",
                f"[ASSESS] Content will be appended after line {line_start}.",
            ]
        )
    else:  # replace
        for i in range(line_start - 1, line_end):
            out.append(f"-{lines[i]}")
        if new_content:
            for line in new_content.splitlines():
                out.append(f"+{line}")
        out.extend(["", "[ASSESS] Ready to replace."])
    return "\n".join(out)


def _target_location_strike(
    path: Path,
    line_start: int,
    line_end: int,
    new_content: str | None,
    mode: str,
) -> str:
    """Execute location-mode changes."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = content.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    total = len(lines)
    line_end = min(line_end, total)

    backup_path = _create_backup(path) if not _is_git_clean(path) else None

    affected_lines = line_end - line_start + 1

    if mode == "delete":
        lines = lines[: line_start - 1] + lines[line_end:]
        change_summary = f"Deleted {affected_lines} line(s) at L{line_start}-{line_end}"
    elif mode == "insert":
        insert_content = (new_content or "") + (
            "\n" if new_content and not new_content.endswith("\n") else ""
        )
        lines = lines[: line_start - 1] + [insert_content] + lines[line_start - 1 :]
        new_line_count = len((new_content or "").splitlines())
        change_summary = f"Inserted {new_line_count} line(s) before L{line_start}"
    elif mode == "append":
        append_content = (new_content or "") + (
            "\n" if new_content and not new_content.endswith("\n") else ""
        )
        lines = lines[:line_start] + [append_content] + lines[line_start:]
        new_line_count = len((new_content or "").splitlines())
        change_summary = f"Appended {new_line_count} line(s) after L{line_start}"
    else:  # replace
        new_lines = [
            line + ("\n" if not line.endswith("\n") else "")
            for line in (new_content or "").splitlines(keepends=True)
        ]
        lines = lines[: line_start - 1] + new_lines + lines[line_end:]
        new_line_count = len(new_lines)
        change_summary = (
            f"Replaced L{line_start}-{line_end} ({affected_lines} lines) "
            f"with {new_line_count} line(s)"
        )

    path.write_text("".join(lines), encoding="utf-8")

    coords_str = f"{path}:{line_start}"
    if line_end != line_start:
        coords_str += f"-{line_end}"

    out = [f"[OK] Strike at {coords_str}"]
    out.append(f"     {change_summary}")
    if backup_path:
        out.append(f"Backup: {backup_path}")
    return "\n".join(out)


def _target_impl(
    coords: str,
    mode: str = "item",
    action: str = "strike",
    old: str = "",
    new: str = "",
    content: str = "",
    op: str = "replace",
) -> str:
    """Surgical editing: assess or strike at coordinates.

    Modes:
        item     — find/replace old->new within line range
        location — line-based operations (replace/insert/delete/append)

    Actions:
        assess — preview changes (dry run)
        strike — execute changes

    Returns action report string.
    """
    start_ms = time.time() * 1000
    parsed = parse_coordinates(coords)
    path = _normalize_path(parsed["path"])

    if not path.exists():
        return f"Error: File not found: {parsed['path']}"
    if not path.is_file():
        return f"Error: Not a file: {parsed['path']}"

    old_text = read_surgical_content(old) if old else ""
    new_text = read_surgical_content(new) if new else ""
    content_text = read_surgical_content(content) if content else ""

    if mode == "item":
        if action == "assess":
            result = _target_item_assess(
                path,
                old_text,
                new_text,
                parsed["line_start"],
                parsed["line_end"],
            )
        else:
            result = _target_item_strike(
                path,
                old_text,
                new_text,
                parsed["line_start"],
                parsed["line_end"],
            )
    else:  # location
        if action == "assess":
            result = _target_location_assess(
                path,
                parsed["line_start"] or 1,
                parsed["line_end"] or parsed["line_start"] or 1,
                content_text,
                op,
            )
        else:
            result = _target_location_strike(
                path,
                parsed["line_start"] or 1,
                parsed["line_end"] or parsed["line_start"] or 1,
                content_text,
                op,
            )

    latency_ms = round(time.time() * 1000 - start_ms, 2)
    _log(
        "INFO",
        "target",
        f"{coords} mode={mode} action={action}",
        metrics=f"latency_ms={latency_ms} status=success",
    )
    return result


# --- Confirmation: BDA ---


def _confirm_impl(coords: str, expected: str | None = None) -> str:
    """Verify file state (Battle Damage Assessment).

    Returns confirmation report string.
    """
    start_ms = time.time() * 1000
    parsed = parse_coordinates(coords)
    path = _normalize_path(parsed["path"])

    if not path.exists():
        return f"x File not found: {path}"
    if not path.is_file():
        return f"x Not a file: {path}"
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"x Cannot read: {e}"

    lines = content.splitlines()
    total = len(lines)
    expected_text = read_surgical_content(expected) if expected else None

    if parsed["line_start"] is not None:
        line_start = parsed["line_start"]
        line_end = parsed["line_end"] if parsed["line_end"] else line_start
        if line_start < 1 or line_start > total:
            return f"x Line {line_start} out of range (1-{total})"
        if line_end > total:
            line_end = total
        target_lines = lines[line_start - 1 : line_end]
        target_content = "\n".join(target_lines)

        if expected_text:
            if expected_text in target_content:
                out = [
                    f"OK Confirmed: '{expected_text}' found at {path}:{line_start}",
                    "",
                ]
                for i, line in enumerate(target_lines, line_start):
                    marker = ">>>" if expected_text in line else "   "
                    out.append(f"{marker} {i:>4}|{line}")
                latency_ms = round(time.time() * 1000 - start_ms, 2)
                _log(
                    "INFO",
                    "confirm",
                    f"{coords} expected=found",
                    metrics=f"latency_ms={latency_ms} status=success",
                )
                return "\n".join(out)
            else:
                out = [
                    f"x Not found: '{expected_text}' not in "
                    f"{path}:{line_start}-{line_end}",
                    "",
                    "Actual content:",
                ]
                for i, line in enumerate(target_lines, line_start):
                    out.append(f"    {i:>4}|{line}")
                latency_ms = round(time.time() * 1000 - start_ms, 2)
                _log(
                    "WARN",
                    "confirm",
                    f"{coords} expected=not_found",
                    metrics=f"latency_ms={latency_ms} status=mismatch",
                )
                return "\n".join(out)

        out = [
            f"OK {path}:{line_start}"
            + (f"-{line_end}" if line_end != line_start else ""),
            "",
        ]
        for i, line in enumerate(target_lines, line_start):
            out.append(f"    {i:>4}|{line}")
        latency_ms = round(time.time() * 1000 - start_ms, 2)
        _log(
            "INFO",
            "confirm",
            coords,
            metrics=f"latency_ms={latency_ms} status=success",
        )
        return "\n".join(out)

    if expected_text:
        if expected_text in content:
            for i, line in enumerate(lines, 1):
                if expected_text in line:
                    latency_ms = round(time.time() * 1000 - start_ms, 2)
                    _log(
                        "INFO",
                        "confirm",
                        f"{coords} expected=found",
                        metrics=f"latency_ms={latency_ms} status=success",
                    )
                    return (
                        f"OK Confirmed: '{expected_text}' found in {path}"
                        f"\n\n>>> {i:>4}|{line}"
                    )
        else:
            latency_ms = round(time.time() * 1000 - start_ms, 2)
            _log(
                "WARN",
                "confirm",
                f"{coords} expected=not_found",
                metrics=f"latency_ms={latency_ms} status=mismatch",
            )
            return f"x Not found: '{expected_text}' not in {path}"

    latency_ms = round(time.time() * 1000 - start_ms, 2)
    _log(
        "INFO",
        "confirm",
        coords,
        metrics=f"latency_ms={latency_ms} status=success",
    )
    return f"OK {path}\n  Lines: {total}\n  Size: {path.stat().st_size} bytes"


# --- Recovery: backup management ---


def _recover_list(pattern: str | None = None) -> str:
    """List available backups."""
    if not BACKUP_DIR.exists():
        return "No backup directory found"
    backups = sorted(
        BACKUP_DIR.glob("*.backup"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pattern:
        backups = [b for b in backups if pattern.lower() in b.name.lower()]
    if not backups:
        return "No backups found"
    out = [f"Backups: {len(backups)}", f"Location: {BACKUP_DIR}", ""]
    for b in backups:
        out.append(f"  {b.name} (Size: {b.stat().st_size} bytes)")
    return "\n".join(out)


def _recover_assess(backup_name: str) -> str:
    """Preview what would be restored from a backup."""
    backup_path = BACKUP_DIR / backup_name
    if not backup_path.exists():
        if not backup_name.endswith(".backup"):
            backup_path = BACKUP_DIR / f"{backup_name}.backup"
        if not backup_path.exists():
            return f"Backup not found: {backup_name}"
    parts = backup_path.name.rsplit(".", 2)
    original_name = parts[0] if len(parts) >= 3 else backup_path.stem
    try:
        backup_content = backup_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Cannot read backup: {e}"
    backup_lines = backup_content.splitlines()
    out = [
        f"Backup: {backup_path.name}",
        f"Original name: {original_name}",
        f"Backup size: {backup_path.stat().st_size} bytes",
        f"Backup lines: {len(backup_lines)}",
        "",
        "Content preview (first 20 lines):",
        "",
    ]
    for i, line in enumerate(backup_lines[:20], 1):
        out.append(f"{i:>4}|{line}")
    if len(backup_lines) > 20:
        out.append(f"  ... ({len(backup_lines) - 20} more lines)")
    out.extend(["", "[ASSESS] Use recover strike <backup> <target_path> to restore"])
    return "\n".join(out)


def _recover_strike(backup_name: str, target_path: str) -> str:
    """Restore a backup to target path."""
    backup_path = BACKUP_DIR / backup_name
    if not backup_path.exists():
        if not backup_name.endswith(".backup"):
            backup_path = BACKUP_DIR / f"{backup_name}.backup"
        if not backup_path.exists():
            return f"Backup not found: {backup_name}"
    target = Path(target_path).resolve()
    current_backup = None
    if target.exists():
        current_backup = _create_backup(target) if not _is_git_clean(target) else None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_path, target)

    out = [f"[OK] Restored {target} from {backup_name}"]
    if current_backup:
        out.append(f"     Current backed up to: {current_backup}")
    else:
        out.append("     (target was new file)")
    return "\n".join(out)


def _recover_clear(pattern: str | None = None, confirm_delete: bool = False) -> str:
    """Remove backups. Use pattern to filter, confirm_delete=True to execute."""
    if not BACKUP_DIR.exists():
        return "No backup directory found"
    backups = list(BACKUP_DIR.glob("*.backup"))
    if pattern:
        backups = [b for b in backups if pattern.lower() in b.name.lower()]
    if not backups:
        if pattern:
            return f"No backups matching '{pattern}'"
        return "No backups to clear"

    if not confirm_delete:
        total_size = sum(b.stat().st_size for b in backups)
        out = [
            f"[ASSESS] Would delete {len(backups)} backup(s)",
            f"Total size: {total_size:,} bytes",
            "",
            "Files to delete:",
        ]
        for b in backups[:10]:
            out.append(f"  {b.name}")
        if len(backups) > 10:
            out.append(f"  ... and {len(backups) - 10} more")
        out.extend(["", "Use confirm=True to execute deletion"])
        return "\n".join(out)

    deleted = 0
    errors = []
    for b in backups:
        try:
            b.unlink()
            deleted += 1
        except Exception as e:
            errors.append(f"{b.name}: {e}")
    out = [f"[STRIKE] Deleted {deleted} backup(s)"]
    if errors:
        out.append(f"Errors ({len(errors)}):")
        for err in errors:
            out.append(f"  {err}")
    return "\n".join(out)


def _recover_impl(
    action: str,
    backup_name: str | None = None,
    target_path: str | None = None,
    pattern: str | None = None,
    confirm_delete: bool = False,
) -> str:
    """Backup management: list, assess, restore, clear.

    Actions:
        list   — List available backups (pattern optional)
        assess — Preview backup contents (backup_name required)
        strike — Restore backup (backup_name and target_path required)
        clear  — Delete backups (pattern optional, confirm_delete to execute)

    Returns action report string.
    """
    start_ms = time.time() * 1000
    action = action.lower()

    if action == "list":
        result = _recover_list(pattern)
    elif action == "assess":
        if not backup_name:
            return "assess requires backup_name"
        result = _recover_assess(backup_name)
    elif action in ("strike", "restore"):
        if not backup_name or not target_path:
            return "strike requires backup_name and target_path"
        result = _recover_strike(backup_name, target_path)
    elif action == "clear":
        result = _recover_clear(pattern, confirm_delete)
    else:
        return f"Unknown action: {action}. Use: list, assess, strike, clear"

    latency_ms = round(time.time() * 1000 - start_ms, 2)
    _log(
        "INFO",
        "recover",
        f"action={action}",
        metrics=f"latency_ms={latency_ms} status=success",
    )
    return result


# =============================================================================
# CLI INTERFACE
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Destructive file operations: write, edit, delete with automatic backups"
        )
    )
    # -V (capital) for version: lowercase -v reserved for future --verbose flag alignment
    parser.add_argument("-V", "--version", action="version", version="1.1.0")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # --- mcp-stdio ---
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")

    # --- write ---
    p_write = subparsers.add_parser("write", help="Write content to file")
    p_write.add_argument("file_path")
    p_write.add_argument("content", nargs="?", default=None)
    p_write.add_argument(
        "-c", "--content", "--text", dest="content_named", default=None
    )
    p_write.add_argument("-B", "--no-backup", action="store_false", dest="backup")

    # --- create ---
    p_create = subparsers.add_parser("create", help="Create new file")
    p_create.add_argument("file_path")
    p_create.add_argument("content", nargs="?", default="")
    p_create.add_argument("-f", "--overwrite", action="store_true")

    # --- append ---
    p_append = subparsers.add_parser("append", help="Append content to file")
    p_append.add_argument("file_path")
    p_append.add_argument("content", nargs="?", default=None)
    p_append.add_argument(
        "-c", "--content", "--text", dest="content_named", default=None
    )

    # --- write-line ---
    p_wl = subparsers.add_parser("write-line", help="Replace specific line")
    p_wl.add_argument("file_path")
    p_wl.add_argument("line_no", type=int)
    p_wl.add_argument("content", nargs="?", default=None)
    p_wl.add_argument("-c", "--content", "--text", dest="content_named", default=None)

    # --- write-string ---
    p_ws = subparsers.add_parser("write-string", help="Replace string in file")
    p_ws.add_argument("file_path")
    p_ws.add_argument("old", help="String to find")
    p_ws.add_argument("new", help="Replacement string")
    p_ws.add_argument(
        "-a",
        "--all",
        action="store_true",
        dest="replace_all",
        default=True,
        help="Replace all occurrences (default)",
    )
    p_ws.add_argument(
        "-1",
        "--first",
        action="store_true",
        dest="first_only",
        help="Replace only first occurrence",
    )

    # --- delete-range (alias: rml) ---
    p_dr = subparsers.add_parser("delete-range", help="Delete line range")
    p_dr.add_argument("file_path")
    p_dr.add_argument("start", type=int, help="Start line (1-based)")
    p_dr.add_argument("end", type=int, help="End line (inclusive)")

    p_rml = subparsers.add_parser("rml", help="Delete line range (alias)")
    p_rml.add_argument("file_path")
    p_rml.add_argument("start", type=int, help="Start line (1-based)")
    p_rml.add_argument("end", type=int, help="End line (inclusive)")

    # --- cp ---
    p_cp = subparsers.add_parser("cp", help="Copy file or directory")
    p_cp.add_argument("src")
    p_cp.add_argument("dest")

    # --- mv ---
    p_mv = subparsers.add_parser("mv", help="Move/rename file or directory")
    p_mv.add_argument("src")
    p_mv.add_argument("dest")

    # --- rm ---
    p_rm = subparsers.add_parser("rm", help="Delete file or directory")
    p_rm.add_argument("path")
    p_rm.add_argument(
        "-y",
        "--confirm",
        action="store_true",
        help="Confirm deletion (required for directories)",
    )

    # --- target ---
    p_target = subparsers.add_parser("target", help="Surgical editing workflow")
    p_target.add_argument("coords")
    p_target.add_argument("-a", "--assess", action="store_true", help="Preview only")
    p_target.add_argument("-s", "--strike", action="store_true", help="Execute change")
    p_target.add_argument("-m", "--mode", default="item")
    p_target.add_argument("-o", "--old", "--find", dest="old")
    p_target.add_argument("-n", "--new", "--replace", "--with", dest="new")
    p_target.add_argument("-c", "--content", "--text", dest="content")
    p_target.add_argument("-O", "--op", default="replace")

    # --- confirm ---
    p_confirm = subparsers.add_parser("confirm", help="Verify file state (BDA)")
    p_confirm.add_argument("coords")
    p_confirm.add_argument("expected", nargs="?")

    # --- recover ---
    p_recover = subparsers.add_parser("recover", help="Backup management")
    p_recover.add_argument("action", help="list, assess, restore, clear")
    p_recover.add_argument(
        "args",
        nargs="*",
        help="backup_name [target_path] for restore; pattern for list/clear",
    )
    p_recover.add_argument(
        "-y", "--confirm", action="store_true", help="Confirm deletion (for clear)"
    )

    args = parser.parse_args()

    try:
        # --- Dispatch ---
        exit_code = 0

        if args.command == "mcp-stdio":
            _run_mcp()

        elif args.command == "write":
            content = args.content_named or args.content
            if content is None and not sys.stdin.isatty():
                content = sys.stdin.read()
            assert content is not None, (
                "content required. Usage: write <file> <content> or echo | write <file> -"
            )
            content = _get_content(content)
            success, msg, err = _write_impl(args.file_path, content, backup=args.backup)
            if success:
                _report(msg)
            else:
                _report(f"Error: {err}")
                exit_code = 1

        elif args.command == "create":
            content = args.content or ""
            if content == "" and not sys.stdin.isatty():
                content = sys.stdin.read()
            content = _get_content(content)
            success, msg, err = _create_impl(
                args.file_path, content, overwrite=args.overwrite
            )
            if success:
                _report(msg)
            else:
                _report(f"Error: {err}")
                exit_code = 1

        elif args.command == "append":
            content = args.content_named or args.content
            if content is None and not sys.stdin.isatty():
                content = sys.stdin.read()
            assert content is not None, (
                "content required. Usage: append <file> <content>"
            )
            content = _get_content(content)
            result = _append_impl(args.file_path, content)
            if result.startswith("[OK]"):
                _report(result)
            else:
                _report(result)
                exit_code = 1

        elif args.command == "write-line":
            content = args.content_named or args.content
            if content is None and not sys.stdin.isatty():
                content = sys.stdin.read().rstrip("\n")
            assert content is not None, (
                "content required. Usage: write-line <file> <line_no> <content>"
            )
            content = _get_content(content)
            result = _write_line_impl(args.file_path, args.line_no, content)
            if result.startswith("[OK]"):
                _report(result)
            else:
                _report(result)
                exit_code = 1

        elif args.command == "write-string":
            replace_all = not args.first_only
            result = _write_string_impl(
                args.file_path, args.old, args.new, replace_all=replace_all
            )
            if result.startswith("[OK]"):
                _report(result)
            else:
                _report(result)
                exit_code = 1

        elif args.command in ("delete-range", "rml"):
            result = _delete_range_impl(args.file_path, args.start, args.end)
            if result.startswith("[OK]"):
                _report(result)
            else:
                _report(result)
                exit_code = 1

        elif args.command == "cp":
            result = _cp_impl(args.src, args.dest)
            if result.startswith("[OK]"):
                _report(result)
            else:
                _report(result)
                exit_code = 1

        elif args.command == "mv":
            result = _mv_impl(args.src, args.dest)
            if result.startswith("[OK]"):
                _report(result)
            else:
                _report(result)
                exit_code = 1

        elif args.command == "rm":
            result = _rm_impl(args.path, confirm=args.confirm)
            if result.startswith("[OK]") or result.startswith("[ASSESS]"):
                _report(result)
            else:
                _report(result)
                exit_code = 1

        elif args.command == "target":
            action = "assess" if args.assess else "strike"
            result = _target_impl(
                args.coords,
                mode=args.mode,
                action=action,
                old=args.old or "",
                new=args.new or "",
                content=args.content or "",
                op=args.op,
            )
            if result.startswith("[OK]"):
                _report(result)
            elif "No matches" in result:
                _report(result)
                exit_code = 2
            else:
                _report(result)
                exit_code = 1

        elif args.command == "confirm":
            result = _confirm_impl(args.coords, args.expected)
            if result.startswith("OK"):
                print(result)
            else:
                print(result)
                exit_code = 1

        elif args.command == "recover":
            action = args.action.lower()
            extra_args = args.args if hasattr(args, "args") else []

            if action == "list":
                pattern = extra_args[0] if extra_args else None
                result = _recover_impl("list", pattern=pattern)
                if "No backup" in result:
                    _report(result)
                    exit_code = 2
                else:
                    print(result)

            elif action in ("restore", "strike"):
                if not extra_args:
                    _report("Error: restore requires backup_name")
                    exit_code = 1
                else:
                    backup_name = extra_args[0]
                    parts = backup_name.rsplit(".", 2)
                    default_target = (
                        parts[0]
                        if len(parts) >= 3
                        else backup_name.replace(".backup", "")
                    )
                    target_path = (
                        extra_args[1] if len(extra_args) > 1 else default_target
                    )
                    result = _recover_impl(
                        "strike",
                        backup_name=backup_name,
                        target_path=target_path,
                    )
                    if result.startswith("[OK]"):
                        _report(result)
                    else:
                        _report(result)
                        exit_code = 1

            elif action == "assess":
                if not extra_args:
                    _report("Error: assess requires backup_name")
                    exit_code = 1
                else:
                    result = _recover_impl("assess", backup_name=extra_args[0])
                    print(result)

            elif action == "clear":
                pattern = extra_args[0] if extra_args else None
                confirm_delete = getattr(args, "confirm", False)
                result = _recover_impl(
                    "clear", pattern=pattern, confirm_delete=confirm_delete
                )
                if result.startswith("[STRIKE]") or result.startswith("[ASSESS]"):
                    _report(result)
                elif "No backup" in result:
                    _report(result)
                    exit_code = 2
                else:
                    _report(result)
                    exit_code = 1

            else:
                _report(
                    f"Unknown recover action: {action}. Use: list, assess, restore, clear"
                )
                exit_code = 1

        else:
            parser.print_help()

        sys.exit(exit_code)
    except (AssertionError, Exception) as e:
        _log("ERROR", args.command or "unknown", str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# FASTMCP SERVER
# =============================================================================
def _run_mcp():
    from fastmcp import FastMCP

    mcp = FastMCP("fw")

    @mcp.tool()
    def write(path: str, content: str) -> str:
        """Write content to file with automatic backup.

        Creates parent directories if needed. Verifies write via BDA
        (read-back check). Backs up existing files unless git-clean.

        Args:
            path: File path to write to
            content: Content to write
        """
        success, msg, err = _write_impl(path, content)
        return msg if success else f"Error: {err}"

    @mcp.tool()
    def create(path: str, content: str = "") -> str:
        """Create a new file. Fails if file already exists.

        Creates parent directories if needed. Verifies write via BDA.

        Args:
            path: File path to create
            content: Initial content (default: empty)
        """
        success, msg, err = _create_impl(path, content)
        return msg if success else f"Error: {err}"

    @mcp.tool()
    def append(path: str, content: str) -> str:
        """Append content to existing file with automatic backup.

        Adds a newline before content if content doesn't start with one.

        Args:
            path: File path to append to
            content: Content to append
        """
        return _append_impl(path, content)

    @mcp.tool()
    def write_line(path: str, line_no: int, content: str) -> str:
        """Replace a specific line in a file.

        Line numbers are 1-based. Creates backup before modification.

        Args:
            path: File path
            line_no: Line number to replace (1-based)
            content: New content for the line
        """
        return _write_line_impl(path, line_no, content)

    @mcp.tool()
    def write_string(path: str, old: str, new: str, replace_all: bool = True) -> str:
        """Replace string in file. Supports multiline matches.

        Creates backup before modification. Reports match locations.

        Args:
            path: File path
            old: String to find
            new: Replacement string
            replace_all: Replace all occurrences (True) or first only (False)
        """
        return _write_string_impl(path, old, new, replace_all=replace_all)

    @mcp.tool()
    def delete_range(path: str, start: int, end: int) -> str:
        """Delete lines from start to end (inclusive).

        Creates backup before deletion. Reports removed content.

        Args:
            path: File path
            start: Start line number (1-based)
            end: End line number (inclusive)
        """
        return _delete_range_impl(path, start, end)

    @mcp.tool()
    def cp(src: str, dest: str) -> str:
        """Copy file or directory with backup of destination.

        Supports both files and directories. Backs up destination
        if it exists and is not git-clean.

        Args:
            src: Source path
            dest: Destination path
        """
        return _cp_impl(src, dest)

    @mcp.tool()
    def mv(src: str, dest: str) -> str:
        """Move/rename file or directory with backup.

        Backs up destination if it exists and is not git-clean.

        Args:
            src: Source path
            dest: Destination path
        """
        return _mv_impl(src, dest)

    @mcp.tool()
    def rm(path: str) -> str:
        """Delete file or directory with backup.

        Files are backed up before deletion. Directories are archived
        as tarballs before removal.

        Args:
            path: Path to delete
        """
        return _rm_impl(path, confirm=True)

    @mcp.tool()
    def target(
        coords: str,
        mode: str = "item",
        action: str = "strike",
        old: str = "",
        new: str = "",
        content: str = "",
        op: str = "replace",
    ) -> str:
        """Surgical editing: assess or strike at coordinates.

        Two modes:
        - item: find/replace old->new within optional line range
        - location: line-based operations (replace/insert/delete/append)

        Two actions:
        - assess: preview changes (dry run)
        - strike: execute changes

        Args:
            coords: File path with optional line spec (file:line or file:start-end)
            mode: Editing mode — 'item' or 'location'
            action: 'assess' (preview) or 'strike' (execute)
            old: Text to find (item mode)
            new: Replacement text (item mode)
            content: New content (location mode)
            op: Operation for location mode — replace, insert, delete, append
        """
        return _target_impl(
            coords,
            mode=mode,
            action=action,
            old=old,
            new=new,
            content=content,
            op=op,
        )

    @mcp.tool()
    def confirm(coords: str, expected: str = "") -> str:
        """Verify file state — Battle Damage Assessment (BDA).

        Check that a file or line range contains expected content.
        Without expected text, shows file/line info.

        Args:
            coords: File path with optional line spec (file:line or file:start-end)
            expected: Text that should be present (optional). Use @file for file content.
        """
        return _confirm_impl(coords, expected if expected else None)

    @mcp.tool()
    def recover(
        action: str,
        backup_name: str = "",
        target_path: str = "",
        pattern: str = "",
        confirm_delete: bool = False,
    ) -> str:
        """Backup management: list, assess, restore, clear.

        Actions:
        - list: show available backups (filter with pattern)
        - assess: preview backup contents
        - strike/restore: restore backup to target path
        - clear: delete backups (requires confirm_delete=True)

        Args:
            action: Operation — list, assess, strike, restore, clear
            backup_name: Name of backup file (for assess/strike)
            target_path: Where to restore to (for strike)
            pattern: Filter pattern (for list/clear)
            confirm_delete: Must be True to actually delete (for clear)
        """
        return _recover_impl(
            action,
            backup_name=backup_name if backup_name else None,
            target_path=target_path if target_path else None,
            pattern=pattern if pattern else None,
            confirm_delete=confirm_delete,
        )

    print("fw MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
