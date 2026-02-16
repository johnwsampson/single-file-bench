#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastmcp",
# ]
# ///
"""Persistent task tracking with hierarchy, dependencies, and session passdowns.

Manages tasks and passdown notes in TSV format with CLI and MCP interfaces.
Data persists across sessions in ~/.sfb/track.tsv with automatic archiving.

Usage:
    sft_track.py task "Review PR" --tags urgent,code
    sft_track.py task "Fix bug" --parent abc123 --blocked-by def456
    sft_track.py tasks --status active
    sft_track.py passdown "Session ended, need to test auth flow"
    sft_track.py blocked
    sft_track.py recent --n 10
    sft_track.py archive --before 2026-01-01
    sft_track.py mcp-stdio
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =============================================================================
# LOGGING (EXACT TEMPLATE)
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
            f.write(
                f"{ts}\t{_SCRIPT}\t{level}\t{event}\t{msg}\t{detail}\t{metrics}\t{trace}\n"
            )
    except Exception:
        pass


# =============================================================================
# CONFIGURATION
# =============================================================================
EXPOSED = [
    "task",
    "tasks",
    "passdown",
    "passdowns",
    "status",
    "search",
    "blocked",
    "recent",
    "archive",
    "dashboard",
]


def _get_project_id() -> str:
    """Generate unique project ID from current working directory."""
    cwd = Path.cwd()
    # Use directory name + partial path hash for uniqueness
    path_str = str(cwd)
    import hashlib

    path_hash = hashlib.md5(path_str.encode()).hexdigest()[:8]
    return f"{cwd.name}-{path_hash}"


# Project-specific data paths
_PROJECT_ID = _get_project_id()
_PROJECT_DATA_DIR = Path.home() / ".sfb" / "projects" / _PROJECT_ID

CONFIG = {
    "version": "1.0.0",
    "project_id": _PROJECT_ID,
    "data_dir": _PROJECT_DATA_DIR,
    "data_file": _PROJECT_DATA_DIR / "track.tsv",
    "archive_dir": _PROJECT_DATA_DIR / "archive",
    "valid_task_statuses": {"pending", "active", "done", "dropped", "waiting"},
    "valid_passdown_statuses": {"current", "superseded"},
    "valid_priorities": {"low", "medium", "high", "critical"},
}

OPS_COLS = [
    "id",
    "type",
    "status",
    "priority",
    "tags",
    "content",
    "parent",
    "blocked_by",
    "due_date",
    "created",
    "completed",
]


# Dashboard — constant path in project root for bookmarkable refresh
_DASHBOARD_HTML = Path.cwd() / "dashboard.html"
_PASSDOWN_MD = Path.cwd() / "passdown.md"
VIEWS = ["kanban", "table", "deps", "timeline", "chart", "passdown"]

# =============================================================================
# CORE UTILITIES
# =============================================================================


def _html_esc(s: str) -> str:
    """Escape HTML special characters."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _short_id() -> str:
    """Generate 8-char hex ID from uuid4."""
    return uuid.uuid4().hex[:8]


def _now() -> str:
    """ISO 8601 timestamp with milliseconds."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _esc(s: str) -> str:
    """Escape special chars for TSV."""
    if s is None:
        return ""
    return (
        str(s)
        .replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def _unesc(s: str) -> str:
    """Unescape TSV special chars."""
    return (
        s.replace("\\r", "\r")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\\", "\\")
    )


def _ensure_data_dir() -> None:
    """Create ~/.sfb/ if not exists."""
    CONFIG["data_dir"].mkdir(parents=True, exist_ok=True)


def _read_all(include_archive: bool = False) -> list[dict]:
    """Read all rows from track.tsv. If include_archive, also read archive files."""
    _ensure_data_dir()

    rows = []

    # Read main data file
    data_file = CONFIG["data_file"]
    if data_file.exists():
        with open(data_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                for col in OPS_COLS:
                    if col in row:
                        row[col] = _unesc(row[col])
                rows.append(row)

    # Read archive files if requested
    if include_archive:
        archive_dir = CONFIG["archive_dir"]
        if archive_dir.exists():
            for archive_file in sorted(archive_dir.glob("*.tsv")):
                try:
                    with open(archive_file, newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f, delimiter="\t")
                        for row in reader:
                            for col in OPS_COLS:
                                if col in row:
                                    row[col] = _unesc(row[col])
                            rows.append(row)
                except Exception:
                    continue

    return rows


def _append_row(row: dict) -> None:
    """Append single row to track.tsv."""
    _ensure_data_dir()
    data_file = CONFIG["data_file"]

    with open(data_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=OPS_COLS, delimiter="\t", quoting=csv.QUOTE_MINIMAL
        )
        # Write header if file empty
        if data_file.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({_esc(k): _esc(v) for k, v in row.items()})


def _rewrite_all(rows: list[dict]) -> None:
    """Rewrite entire track.tsv with new rows."""
    _ensure_data_dir()
    data_file = CONFIG["data_file"]

    with open(data_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=OPS_COLS, delimiter="\t", quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({_esc(k): _esc(v) for k, v in row.items()})


# =============================================================================
# CORE FUNCTIONS - TASKS
# =============================================================================


def _task_impl(
    content: str,
    status: str = "pending",
    priority: str = "medium",
    tags: str = "",
    task_id: str = "",
    parent: str = "",
    blocked_by: str = "",
    due_date: str = "",
) -> tuple[dict, dict]:
    """Create or update task. Returns (result_dict, metrics_dict).

    CLI: task, MCP: task
    """
    start_ms = time.time() * 1000

    if status not in CONFIG["valid_task_statuses"]:
        status = "pending"
    if priority not in CONFIG["valid_priorities"]:
        priority = "medium"

    if task_id:
        # Update existing
        rows = _read_all()
        found = False
        for r in rows:
            if r["id"] == task_id and r["type"] == "task":
                r["status"] = status
                r["priority"] = priority
                r["tags"] = tags
                r["content"] = content
                r["parent"] = parent
                r["blocked_by"] = blocked_by
                r["due_date"] = due_date
                if status == "done":
                    r["completed"] = _now()
                found = True
                break

        if not found:
            latency_ms = time.time() * 1000 - start_ms
            return {"status": "error", "message": f"Task {task_id} not found"}, {
                "status": "error",
                "latency_ms": round(latency_ms, 2),
            }

        _rewrite_all(rows)
        _log(
            "INFO",
            "task_updated",
            f"task_id={task_id}",
            detail=f"status={status},priority={priority}",
        )

        latency_ms = time.time() * 1000 - start_ms
        metrics = {
            "status": "success",
            "action": "updated",
            "latency_ms": round(latency_ms, 2),
        }
        result = {"task_id": task_id, "action": "updated", "status": status}

    else:
        # Create new
        tid = _short_id()
        row = {
            "id": tid,
            "type": "task",
            "status": status,
            "priority": priority,
            "tags": tags,
            "content": content,
            "parent": parent,
            "blocked_by": blocked_by,
            "due_date": due_date,
            "created": _now(),
            "completed": "",
        }
        _append_row(row)
        _log(
            "INFO",
            "task_created",
            f"task_id={tid}",
            detail=f"status={status},priority={priority}",
        )

        latency_ms = time.time() * 1000 - start_ms
        metrics = {
            "status": "success",
            "action": "created",
            "latency_ms": round(latency_ms, 2),
        }
        result = {"task_id": tid, "action": "created", "status": status}

    _regenerate_dashboards()
    return result, metrics


def _tasks_impl(
    status: str = "",
    priority: str = "",
    tags: str = "",
    parent: str = "",
    limit: int = 50,
) -> tuple[list[dict], dict]:
    """List tasks with filters. CLI: tasks, MCP: tasks."""
    start_ms = time.time() * 1000

    rows = _read_all()
    tasks = [r for r in rows if r["type"] == "task"]

    if status and status in CONFIG["valid_task_statuses"]:
        tasks = [r for r in tasks if r.get("status") == status]
    if priority and priority in CONFIG["valid_priorities"]:
        tasks = [r for r in tasks if r.get("priority") == priority]
    if tags:
        filter_tags = set(t.strip() for t in tags.split(","))
        tasks = [
            r
            for r in tasks
            if filter_tags & set(t.strip() for t in r.get("tags", "").split(","))
        ]
    if parent:
        tasks = [r for r in tasks if r.get("parent") == parent]

    # Sort by created desc
    tasks.sort(key=lambda r: r.get("created", ""), reverse=True)
    tasks = tasks[:limit]

    result = []
    for r in tasks:
        entry = {
            "task_id": r["id"],
            "status": r["status"],
            "priority": r["priority"],
            "tags": r["tags"],
            "content": r["content"],
            "parent": r.get("parent", ""),
            "blocked_by": r.get("blocked_by", ""),
            "due_date": r.get("due_date", ""),
            "created": r.get("created", ""),
            "completed": r.get("completed", ""),
        }
        result.append(entry)

    latency_ms = time.time() * 1000 - start_ms
    metrics = {
        "status": "success",
        "count": len(result),
        "latency_ms": round(latency_ms, 2),
    }

    return result, metrics


def _passdown_impl(
    content: str,
    tags: str = "",
) -> tuple[dict, dict]:
    """Create passdown note. CLI: passdown, MCP: passdown."""
    start_ms = time.time() * 1000

    rows = _read_all()

    # Supersede existing current passdowns
    changed = False
    for r in rows:
        if r["type"] == "passdown" and r.get("status") == "current":
            r["status"] = "superseded"
            changed = True

    if changed:
        _rewrite_all(rows)

    pid = _short_id()
    row = {
        "id": pid,
        "type": "passdown",
        "status": "current",
        "priority": "",
        "tags": tags,
        "content": content,
        "parent": "",
        "blocked_by": "",
        "due_date": "",
        "created": _now(),
        "completed": "",
    }
    _append_row(row)

    _log("INFO", "passdown_created", f"passdown_id={pid}")

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}
    result = {"passdown_id": pid, "action": "created", "status": "current"}

    _regenerate_dashboards()
    return result, metrics


def _passdowns_impl(
    limit: int = 10,
) -> tuple[list[dict], dict]:
    """List passdown notes. CLI: passdowns, MCP: passdowns."""
    start_ms = time.time() * 1000

    rows = _read_all()
    passdowns = [r for r in rows if r["type"] == "passdown"]

    # Sort by created desc
    passdowns.sort(key=lambda r: r.get("created", ""), reverse=True)
    passdowns = passdowns[:limit]

    result = []
    for r in passdowns:
        entry = {
            "passdown_id": r["id"],
            "status": r["status"],
            "tags": r["tags"],
            "content": r["content"],
            "created": r["created"],
        }
        result.append(entry)

    latency_ms = time.time() * 1000 - start_ms
    metrics = {
        "status": "success",
        "count": len(result),
        "latency_ms": round(latency_ms, 2),
    }

    return result, metrics


def _status_impl() -> tuple[dict, dict]:
    """Get current status. CLI: status, MCP: status."""
    start_ms = time.time() * 1000

    pending_tasks, _ = _tasks_impl(status="pending")
    active_tasks, _ = _tasks_impl(status="active")
    waiting_tasks, _ = _tasks_impl(status="waiting")
    passdowns, _ = _passdowns_impl(limit=1)

    result = {
        "pending_tasks": pending_tasks,
        "active_tasks": active_tasks,
        "waiting_tasks": waiting_tasks,
        "task_count": len(pending_tasks) + len(active_tasks) + len(waiting_tasks),
        "current_passdown": passdowns[0] if passdowns else None,
    }

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}

    return result, metrics


def _search_impl(
    query: str = "",
    type_filter: str = "",
    tags: str = "",
    status: str = "",
    limit: int = 20,
) -> tuple[list[dict], dict]:
    """Search across tasks and passdowns. CLI: search, MCP: search."""
    start_ms = time.time() * 1000

    rows = _read_all()

    if type_filter and type_filter in {"task", "passdown"}:
        rows = [r for r in rows if r.get("type") == type_filter]

    if status:
        rows = [r for r in rows if r.get("status") == status]

    if tags:
        filter_tags = set(t.strip() for t in tags.split(","))
        rows = [
            r
            for r in rows
            if filter_tags & set(t.strip() for t in r.get("tags", "").split(","))
        ]

    if query:
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
        rows = [r for r in rows if pattern.search(r.get("content", ""))]

    # Sort by created desc
    rows.sort(key=lambda r: r.get("created", ""), reverse=True)
    rows = rows[:limit]

    result = []
    for r in rows:
        entry = {
            "id": r["id"],
            "type": r["type"],
            "status": r["status"],
            "priority": r.get("priority", ""),
            "tags": r["tags"],
            "content": r["content"],
            "parent": r.get("parent", ""),
            "blocked_by": r.get("blocked_by", ""),
            "due_date": r.get("due_date", ""),
            "created": r["created"],
            "completed": r.get("completed", ""),
        }
        result.append(entry)

    latency_ms = time.time() * 1000 - start_ms
    metrics = {
        "status": "success",
        "count": len(result),
        "latency_ms": round(latency_ms, 2),
    }

    return result, metrics


def _blocked_impl() -> tuple[list[dict], dict]:
    """Show blocked tasks. CLI: blocked, MCP: blocked."""
    start_ms = time.time() * 1000

    rows = _read_all()
    tasks = [r for r in rows if r["type"] == "task"]

    # Find blocked tasks
    blocked_tasks = []
    blockers = {}

    for task in tasks:
        blocked_by = task.get("blocked_by", "")
        if blocked_by:
            # Populate blocking task info
            for blocker_id in blocked_by.split(","):
                blocker_id = blocker_id.strip()
                blocker = next((t for t in tasks if t["id"] == blocker_id), None)
                blockers[blocker_id] = {
                    "task_id": blocker_id,
                    "status": blocker["status"] if blocker else "UNKNOWN",
                    "content": blocker["content"][:100] if blocker else "UNKNOWN",
                }

            task_entry = {
                "task_id": task["id"],
                "status": task["status"],
                "content": task.get("content", "")[:100],
                "blocked_by": [
                    blockers.get(bid, {"task_id": bid, "status": "UNKNOWN"})
                    for bid in blocked_by.split(",")
                ],
            }
            blocked_tasks.append(task_entry)

    latency_ms = time.time() * 1000 - start_ms
    metrics = {
        "status": "success",
        "blocked_count": len(blocked_tasks),
        "latency_ms": round(latency_ms, 2),
    }

    return blocked_tasks, metrics


def _recent_impl(n: int = 10) -> tuple[list[dict], dict]:
    """Show recently completed tasks. CLI: recent, MCP: recent."""
    start_ms = time.time() * 1000

    rows = _read_all()
    completed = [
        r
        for r in rows
        if r["type"] == "task" and r.get("status") in ("done", "dropped")
    ]

    # Sort by completed date
    completed.sort(key=lambda r: r.get("completed", ""), reverse=True)
    completed = completed[:n]

    result = []
    for r in completed:
        entry = {
            "task_id": r["id"],
            "status": r["status"],
            "priority": r.get("priority", ""),
            "tags": r["tags"],
            "content": r["content"],
            "completed": r.get("completed", ""),
        }
        result.append(entry)

    latency_ms = time.time() * 1000 - start_ms
    metrics = {
        "status": "success",
        "count": len(result),
        "latency_ms": round(latency_ms, 2),
    }

    return result, metrics


def _archive_impl(
    before_date: str = "",
    days: int = 30,
    dry_run: bool = False,
) -> tuple[dict, dict]:
    """Archive completed tasks. CLI: archive, MCP: archive."""
    start_ms = time.time() * 1000

    cutoff_date = None
    if before_date:
        try:
            cutoff_date = datetime.fromisoformat(before_date).timestamp()
        except ValueError:
            return {
                "status": "error",
                "message": "Invalid date format. Use YYYY-MM-DD",
            }, {
                "status": "error",
                "latency_ms": round(time.time() * 1000 - start_ms, 2),
            }
    else:
        cutoff_date = time.time() - (days * 24 * 3600)

    rows = _read_all()
    active_rows = []
    archived_rows = []

    archive_file = CONFIG["archive_dir"] / datetime.now().strftime("%Y-%m.tsv")
    archive_file.parent.mkdir(parents=True, exist_ok=True)

    archived_count = 0

    for r in rows:
        if r["type"] == "task" and r.get("status") in ("done", "dropped"):
            completed_ts = r.get("completed")
            if completed_ts:
                try:
                    completed_ts_num = datetime.fromisoformat(completed_ts).timestamp()
                    if completed_ts_num < cutoff_date:
                        archived_rows.append(r)
                        archived_count += 1
                        continue
                except ValueError:
                    pass
        active_rows.append(r)

    if dry_run:
        latency_ms = time.time() * 1000 - start_ms
        return {
            "status": "dry_run",
            "archived_count": archived_count,
            "would_archive": archive_file,
            "remaining_count": len(active_rows),
        }, {"status": "success", "latency_ms": round(latency_ms, 2)}

    # Write archive
    if archived_rows:
        archive_cols = [
            col
            for col in OPS_COLS
            if col != "completed" or any(r.get("completed") for r in archived_rows)
        ]
        with open(archive_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=OPS_COLS, delimiter="\t", quoting=csv.QUOTE_MINIMAL
            )
            writer.writeheader()
            for row in archived_rows:
                writer.writerow({_esc(k): _esc(v) for k, v in row.items()})

    # Rewrite active
    _rewrite_all(active_rows)

    _log("INFO", "archive", f"archived={archived_count}", detail=f"file={archive_file}")

    latency_ms = time.time() * 1000 - start_ms
    metrics = {
        "status": "success",
        "archived_count": archived_count,
        "latency_ms": round(latency_ms, 2),
    }
    result = {
        "status": "success",
        "archived_count": archived_count,
        "archive_file": str(archive_file),
    }

    _regenerate_dashboards()
    return result, metrics


def _generate_kanban_html(tasks: list[dict]) -> str:
    """Generate kanban board HTML."""
    columns: dict[str, list[dict]] = {
        "pending": [],
        "active": [],
        "waiting": [],
        "done": [],
        "dropped": [],
    }
    for task in tasks:
        s = task.get("status", "pending")
        if s in columns:
            columns[s].append(task)

    parts = ['<div class="kanban">']
    for status, col_tasks in columns.items():
        if not col_tasks and status == "dropped":
            continue  # hide empty dropped column
        parts.append(
            f'<div class="kanban-column"><h3>{_html_esc(status.upper())} ({len(col_tasks)})</h3>'
        )
        for t in col_tasks:
            p = t.get("priority", "medium")
            tid = t.get("id", "?")
            parts.append(f'<div class="kanban-card priority-{_html_esc(p)}">')
            parts.append(f'<div class="id">{_html_esc(tid)}</div>')
            parts.append(
                f'<div class="content">{_html_esc(t.get("content", "")[:120])}</div>'
            )
            tags = t.get("tags", "")
            if tags:
                parts.append(f'<div class="meta">{_html_esc(tags)}</div>')
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</div>")
    return "\n".join(parts)


def _generate_table_html(tasks: list[dict]) -> str:
    """Generate table view HTML."""
    parts = [
        "<table><thead><tr>",
        "<th>ID</th><th>Status</th><th>Priority</th><th>Tags</th>"
        "<th>Content</th><th>Parent</th><th>Blocked</th><th>Due</th><th>Created</th>",
        "</tr></thead><tbody>",
    ]
    for t in tasks:
        s = t.get("status", "")
        tid = t.get("id", "?")
        parts.append("<tr>")
        parts.append(f"<td><code>{_html_esc(tid)}</code></td>")
        parts.append(
            f'<td><span class="badge status-{_html_esc(s)}">{_html_esc(s)}</span></td>'
        )
        parts.append(f"<td>{_html_esc(t.get('priority', ''))}</td>")
        parts.append(f"<td>{_html_esc(t.get('tags', ''))}</td>")
        parts.append(f"<td>{_html_esc(t.get('content', '')[:100])}</td>")
        parts.append(f"<td><code>{_html_esc(t.get('parent', ''))}</code></td>")
        parts.append(f"<td><code>{_html_esc(t.get('blocked_by', ''))}</code></td>")
        parts.append(f"<td>{_html_esc(t.get('due_date', ''))}</td>")
        parts.append(f"<td>{_html_esc(t.get('created', '')[:10])}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _generate_deps_mermaid(tasks: list[dict]) -> str:
    """Generate Mermaid dependency graph."""
    lines = ["graph TD"]
    task_ids = set()
    for t in tasks:
        tid = t.get("id", "?")
        task_ids.add(tid)
        content = t.get("content", "")[:30].replace('"', "'").replace("\n", " ")
        s = t.get("status", "")
        lines.append(f'    {tid}["{_html_esc(tid)}: {_html_esc(content)}"]')
        # Style by status
        if s == "done":
            lines.append(f"    style {tid} fill:#d4edda,stroke:#155724")
        elif s == "active":
            lines.append(f"    style {tid} fill:#d1ecf1,stroke:#0c5460")
        elif s == "waiting":
            lines.append(f"    style {tid} fill:#e2e3e5,stroke:#383d41")

    for t in tasks:
        tid = t.get("id", "?")
        parent = t.get("parent", "")
        blocked_by = t.get("blocked_by", "")
        if parent and parent in task_ids:
            lines.append(f"    {parent} --> {tid}")
        if blocked_by:
            for b in blocked_by.split(","):
                b = b.strip()
                if b and b in task_ids:
                    lines.append(f"    {b} -.->|blocks| {tid}")
    return "\n".join(lines)


def _generate_timeline_mermaid(tasks: list[dict]) -> str:
    """Generate Mermaid timeline grouped by date."""
    lines = ["timeline", "    title Task Timeline"]
    by_date: dict[str, list[dict]] = {}
    for t in tasks:
        created = t.get("created", "")
        if created:
            date = created[:10]
            by_date.setdefault(date, []).append(t)

    for date in sorted(by_date.keys())[-14:]:
        names = [t["content"][:25].replace(":", "-") for t in by_date[date]]
        lines.append(f"    {date} : " + " : ".join(names))
    return "\n".join(lines)


def _generate_chart_json(tasks: list[dict]) -> str:
    """Generate Chart.js-compatible JSON data."""
    status_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    for t in tasks:
        s = t.get("status", "unknown")
        p = t.get("priority", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
        priority_counts[p] = priority_counts.get(p, 0) + 1
    return json.dumps({"status": status_counts, "priority": priority_counts})


def _generate_passdown_html(passdowns: list[dict]) -> str:
    """Generate passdown view HTML for the dashboard tab."""
    parts = ['<div class="passdown-view">']

    current = [p for p in passdowns if p.get("status") == "current"]
    superseded = [p for p in passdowns if p.get("status") == "superseded"]

    if current:
        p = current[0]
        parts.append('<div class="passdown-card passdown-current">')
        parts.append(
            f'<div class="passdown-header">Current Passdown '
            f'<span class="passdown-id">{_html_esc(p.get("passdown_id", "?"))}</span>'
            f' &middot; {_html_esc(p.get("created", "")[:19])}'
        )
        tags = p.get("tags", "")
        if tags:
            parts.append(f' &middot; <span class="passdown-tags">{_html_esc(tags)}</span>')
        parts.append("</div>")
        # Render content — convert markdown-ish newlines to HTML
        content = _html_esc(p.get("content", ""))
        # Convert markdown headers
        lines = content.split("\n")
        html_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("### "):
                html_lines.append(f"<h4>{stripped[4:]}</h4>")
            elif stripped.startswith("## "):
                html_lines.append(f"<h3>{stripped[3:]}</h3>")
            elif stripped.startswith("# "):
                html_lines.append(f"<h2>{stripped[2:]}</h2>")
            elif stripped.startswith("- "):
                html_lines.append(f"<li>{stripped[2:]}</li>")
            elif stripped == "":
                html_lines.append("<br>")
            else:
                html_lines.append(f"<p>{line}</p>")
        parts.append(f'<div class="passdown-body">{"".join(html_lines)}</div>')
        parts.append("</div>")
    else:
        parts.append('<div class="passdown-empty">No current passdown.</div>')

    if superseded:
        parts.append(
            f'<details class="passdown-history"><summary>Previous passdowns '
            f"({len(superseded)})</summary>"
        )
        for p in superseded[:10]:  # limit to recent 10
            parts.append('<div class="passdown-card passdown-superseded">')
            parts.append(
                f'<div class="passdown-header">'
                f'<span class="passdown-id">{_html_esc(p.get("passdown_id", "?"))}</span>'
                f' &middot; {_html_esc(p.get("created", "")[:19])}'
            )
            tags = p.get("tags", "")
            if tags:
                parts.append(
                    f' &middot; <span class="passdown-tags">{_html_esc(tags)}</span>'
                )
            parts.append("</div>")
            content = _html_esc(p.get("content", ""))
            # Truncate superseded passdowns for display
            if len(content) > 500:
                content = content[:500] + "..."
            parts.append(f'<div class="passdown-body"><pre>{content}</pre></div>')
            parts.append("</div>")
        parts.append("</details>")

    parts.append("</div>")
    return "\n".join(parts)


def _generate_passdown_md(passdowns: list[dict]) -> str:
    """Generate plain-text markdown passdown for LLM context injection.

    Only includes the current passdown — this is what gets loaded as
    opencode instructions at session startup.
    """
    current = [p for p in passdowns if p.get("status") == "current"]
    if not current:
        return "# Passdown\n\nNo current passdown.\n"

    p = current[0]
    lines = [
        "# Passdown",
        "",
        f"*ID: {p.get('passdown_id', '?')} | Created: {p.get('created', '')}*",
        "",
        p.get("content", ""),
        "",
    ]
    return "\n".join(lines)


# The complete dashboard HTML template. Uses __PLACEHOLDER__ markers to avoid
# escaping conflicts with CSS/JS braces. All 6 views are embedded as tab panels.
_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__PROJECT__ - Task Dashboard</title>
<style>
:root { --bg:#fafafa; --fg:#222; --accent:#0066cc; --border:#ddd; --code-bg:#f4f4f4; }
* { box-sizing: border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       line-height:1.6; max-width:1400px; margin:0 auto; padding:1.5rem;
       background:var(--bg); color:var(--fg); }
header { display:flex; justify-content:space-between; align-items:baseline;
         border-bottom:2px solid var(--accent); padding-bottom:0.5rem; margin-bottom:1rem; }
header h1 { margin:0; font-size:1.4rem; }
header .meta { font-size:0.8rem; color:#666; }
.tabs { display:flex; gap:0; margin-bottom:1.5rem; border-bottom:1px solid var(--border); }
.tab { padding:0.6rem 1.2rem; cursor:pointer; color:var(--fg); text-decoration:none;
       border-bottom:2px solid transparent; font-size:0.9rem; user-select:none; }
.tab:hover { background:var(--code-bg); }
.tab.active { border-bottom-color:var(--accent); font-weight:600; color:var(--accent); }
.view { display:none; }
.view.active { display:block; }

/* Kanban */
.kanban { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:1rem; }
.kanban-column { background:white; border:1px solid var(--border); border-radius:8px; padding:1rem; min-height:200px; }
.kanban-column h3 { margin:0 0 0.75rem 0; font-size:0.8rem; text-transform:uppercase; color:#666;
                    letter-spacing:0.05em; }
.kanban-card { background:var(--code-bg); border-radius:4px; padding:0.75rem; margin-bottom:0.5rem; }
.kanban-card .id { font-size:0.7rem; color:#888; font-family:monospace; }
.kanban-card .content { font-size:0.85rem; margin:0.2rem 0; }
.kanban-card .meta { font-size:0.7rem; color:#999; }
.priority-critical { border-left:3px solid #dc3545; }
.priority-high { border-left:3px solid #fd7e14; }
.priority-medium { border-left:3px solid #ffc107; }
.priority-low { border-left:3px solid #28a745; }

/* Table */
table { width:100%; border-collapse:collapse; background:white; border-radius:8px; overflow:hidden;
        font-size:0.85rem; }
th,td { padding:0.5rem 0.75rem; text-align:left; border-bottom:1px solid var(--border); }
th { background:var(--code-bg); font-weight:600; font-size:0.8rem; white-space:nowrap; }
tr:hover { background:#f8f8f8; }
code { font-size:0.8em; background:var(--code-bg); padding:0.1em 0.3em; border-radius:2px; }
.badge { display:inline-block; padding:0.1rem 0.5rem; border-radius:3px;
         font-size:0.75rem; font-weight:500; }
.status-pending { background:#fff3cd; color:#856404; }
.status-active { background:#d1ecf1; color:#0c5460; }
.status-done { background:#d4edda; color:#155724; }
.status-dropped { background:#f8d7da; color:#721c24; }
.status-waiting { background:#e2e3e5; color:#383d41; }

/* Mermaid */
.mermaid { background:white; padding:1rem; border-radius:8px; border:1px solid var(--border); }

/* Charts */
.chart-grid { display:grid; grid-template-columns:1fr 1fr; gap:2rem; max-width:800px; }
.chart-box { background:white; padding:1rem; border-radius:8px; border:1px solid var(--border); }

/* Passdown */
.passdown-view { max-width:900px; }
.passdown-card { background:white; border:1px solid var(--border); border-radius:8px;
                 padding:1.25rem; margin-bottom:1rem; }
.passdown-current { border-left:4px solid var(--accent); }
.passdown-superseded { border-left:4px solid #ccc; opacity:0.7; }
.passdown-header { font-size:0.85rem; color:#666; margin-bottom:0.75rem;
                   padding-bottom:0.5rem; border-bottom:1px solid var(--border); }
.passdown-id { font-family:monospace; font-size:0.8rem; color:var(--accent); }
.passdown-tags { font-style:italic; }
.passdown-body { font-size:0.9rem; line-height:1.7; }
.passdown-body h2 { font-size:1.1rem; margin:1rem 0 0.5rem 0; }
.passdown-body h3 { font-size:1rem; margin:0.75rem 0 0.4rem 0; }
.passdown-body h4 { font-size:0.9rem; margin:0.5rem 0 0.3rem 0; color:#444; }
.passdown-body li { margin-left:1.5rem; margin-bottom:0.2rem; }
.passdown-body p { margin:0.3rem 0; }
.passdown-body pre { white-space:pre-wrap; font-size:0.85rem; color:#555; }
.passdown-empty { color:#999; font-style:italic; padding:2rem; text-align:center; }
.passdown-history { margin-top:1rem; }
.passdown-history summary { cursor:pointer; font-size:0.85rem; color:#666;
                            padding:0.5rem 0; }
</style>
</head>
<body>
<header>
    <h1>__PROJECT__</h1>
    <span class="meta">__COUNT__ tasks &middot; __UPDATED__</span>
</header>
<div class="tabs">
    <a class="tab active" onclick="showTab('kanban')" id="tab-kanban">Kanban</a>
    <a class="tab" onclick="showTab('table')" id="tab-table">Table</a>
    <a class="tab" onclick="showTab('deps')" id="tab-deps">Dependencies</a>
    <a class="tab" onclick="showTab('timeline')" id="tab-timeline">Timeline</a>
    <a class="tab" onclick="showTab('chart')" id="tab-chart">Chart</a>
    <a class="tab" onclick="showTab('passdown')" id="tab-passdown">Passdown</a>
</div>
<div id="view-kanban" class="view active">__KANBAN__</div>
<div id="view-table" class="view">__TABLE__</div>
<div id="view-deps" class="view"><pre class="mermaid">__DEPS__</pre></div>
<div id="view-timeline" class="view"><pre class="mermaid">__TIMELINE__</pre></div>
<div id="view-chart" class="view">
    <div class="chart-grid">
        <div class="chart-box"><canvas id="chart-status"></canvas></div>
        <div class="chart-box"><canvas id="chart-priority"></canvas></div>
    </div>
</div>
<div id="view-passdown" class="view">__PASSDOWN__</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
mermaid.initialize({startOnLoad:false, theme:'neutral'});

const chartData = __CHART_JSON__;
let chartRendered = false;
let mermaidRendered = false;

const STATUS_COLORS = {
    pending:'#fff3cd', active:'#d1ecf1', waiting:'#e2e3e5',
    done:'#d4edda', dropped:'#f8d7da', unknown:'#ccc'
};
const PRIORITY_COLORS = {
    low:'#28a745', medium:'#ffc107', high:'#fd7e14',
    critical:'#dc3545', unknown:'#ccc'
};

window.showTab = function(view) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    const el = document.getElementById('view-'+view);
    if(el) el.classList.add('active');
    const tab = document.getElementById('tab-'+view);
    if(tab) tab.classList.add('active');
    history.replaceState(null,'','#'+view);
    if((view==='deps'||view==='timeline') && !mermaidRendered) {
        mermaid.run(); mermaidRendered=true;
    }
    if(view==='chart' && !chartRendered) { renderCharts(); chartRendered=true; }
};

function renderCharts() {
    const sd = chartData.status || {};
    new Chart(document.getElementById('chart-status'), {
        type:'doughnut',
        data:{labels:Object.keys(sd),
              datasets:[{data:Object.values(sd),
                         backgroundColor:Object.keys(sd).map(k=>STATUS_COLORS[k]||'#ccc')}]},
        options:{plugins:{title:{display:true,text:'By Status'},legend:{position:'bottom'}}}
    });
    const pd = chartData.priority || {};
    new Chart(document.getElementById('chart-priority'), {
        type:'doughnut',
        data:{labels:Object.keys(pd),
              datasets:[{data:Object.values(pd),
                         backgroundColor:Object.keys(pd).map(k=>PRIORITY_COLORS[k]||'#ccc')}]},
        options:{plugins:{title:{display:true,text:'By Priority'},legend:{position:'bottom'}}}
    });
}

const hash = window.location.hash.slice(1);
if(hash && document.getElementById('view-'+hash)) showTab(hash);
</script>
</body>
</html>"""


def _regenerate_dashboards() -> str | None:
    """Regenerate all dashboard views into one HTML file. Called after mutations.

    Writes to a constant path so the user can bookmark and just refresh.
    Also writes passdown.md for LLM context injection (opencode instructions).
    Returns the HTML file path, or None on failure.
    """
    try:
        tasks, _ = _tasks_impl(limit=500)
        passdowns, _ = _passdowns_impl(limit=20)

        kanban = _generate_kanban_html(tasks)
        table = _generate_table_html(tasks)
        deps = _generate_deps_mermaid(tasks)
        timeline = _generate_timeline_mermaid(tasks)
        chart_json = _generate_chart_json(tasks)
        passdown_html = _generate_passdown_html(passdowns)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        project = Path.cwd().name

        html = _DASHBOARD_TEMPLATE
        html = html.replace("__PROJECT__", _html_esc(project))
        html = html.replace("__UPDATED__", now)
        html = html.replace("__COUNT__", str(len(tasks)))
        html = html.replace("__KANBAN__", kanban)
        html = html.replace("__TABLE__", table)
        html = html.replace("__DEPS__", deps)
        html = html.replace("__TIMELINE__", timeline)
        html = html.replace("__CHART_JSON__", chart_json)
        html = html.replace("__PASSDOWN__", passdown_html)

        _DASHBOARD_HTML.parent.mkdir(parents=True, exist_ok=True)
        _DASHBOARD_HTML.write_text(html, encoding="utf-8")

        # Also write passdown.md for LLM context injection
        passdown_md = _generate_passdown_md(passdowns)
        _PASSDOWN_MD.write_text(passdown_md, encoding="utf-8")

        _log(
            "DEBUG",
            "dashboard_regen",
            f"project={project},tasks={len(tasks)},passdowns={len(passdowns)}",
            detail=f"html={_DASHBOARD_HTML},md={_PASSDOWN_MD}",
        )
        return str(_DASHBOARD_HTML)
    except Exception as e:
        _log("WARN", "dashboard_regen_fail", str(e))
        return None


def _dashboard_impl(
    view: str = "kanban",
    status_filter: str = "",
    priority_filter: str = "",
    tags_filter: str = "",
    auto_open: bool = True,
) -> tuple[dict, dict]:
    """Generate HTML dashboard with all views. CLI: dashboard, MCP: dashboard."""
    start_ms = time.time() * 1000

    html_file = _regenerate_dashboards()

    if html_file and auto_open:
        import webbrowser

        webbrowser.open(f"file://{html_file}#{view}")

    _log("INFO", "dashboard", f"view={view}")

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}
    result = {
        "html_file": html_file or "generation failed",
        "view": view,
        "auto_open": auto_open,
    }

    return result, metrics


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Persistent task tracking with hierarchy and dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sft_track.py task "Review PR #123" --tags urgent,code --priority high
  sft_track.py task "Fix auth bug" --parent abc123 --blocked-by def456
  sft_track.py tasks --status active
  echo "Session ended, test auth flow tomorrow" | sft_track.py passdown
  sft_track.py blocked
  sft_track.py recent --n 5
  sft_track.py archive --days 30
        """,
    )

    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {CONFIG['version']}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # task
    p_task = subparsers.add_parser("task", help="Create or update task")
    p_task.add_argument("content", nargs="?", help="Task content")
    p_task.add_argument("-i", "--id", help="Task ID to update")
    p_task.add_argument(
        "-s", "--status", choices=list(CONFIG["valid_task_statuses"]), default="pending"
    )
    p_task.add_argument(
        "-p", "--priority", choices=list(CONFIG["valid_priorities"]), default="medium"
    )
    p_task.add_argument("-t", "--tags", help="Comma-separated tags")
    p_task.add_argument("-P", "--parent", help="Parent task ID")
    p_task.add_argument("-B", "--blocked-by", help="Comma-separated blocking task IDs")
    p_task.add_argument("-D", "--due", help="Due date YYYY-MM-DD")

    # tasks
    p_tasks = subparsers.add_parser("tasks", help="List tasks")
    p_tasks.add_argument("-s", "--status", help="Filter by status")
    p_tasks.add_argument("-p", "--priority", help="Filter by priority")
    p_tasks.add_argument("-t", "--tags", help="Filter by tags")
    p_tasks.add_argument("-P", "--parent", help="Filter by parent ID")
    p_tasks.add_argument("-n", "--limit", type=int, default=50)

    # passdown
    p_pass = subparsers.add_parser("passdown", help="Create passdown note")
    p_pass.add_argument("content", nargs="?", help="Passdown content")
    p_pass.add_argument("-t", "--tags", help="Comma-separated tags")

    # passdowns
    p_passes = subparsers.add_parser("passdowns", help="List passdown notes")
    p_passes.add_argument("-n", "--limit", type=int, default=10)

    # status
    subparsers.add_parser(
        "status", help="Show pending/active tasks and current passdown"
    )

    # search
    p_search = subparsers.add_parser("search", help="Search tasks and passdowns")
    p_search.add_argument("query", nargs="?", help="Search regex")
    p_search.add_argument("-t", "--type", choices=["task", "passdown"])
    p_search.add_argument("-g", "--tags", help="Filter by tags")
    p_search.add_argument("-s", "--status", help="Filter by status")
    p_search.add_argument("-n", "--limit", type=int, default=20)

    # blocked
    subparsers.add_parser("blocked", help="Show blocked tasks and blockers")

    # recent
    p_recent = subparsers.add_parser("recent", help="Show recently completed tasks")
    p_recent.add_argument("-n", "--number", type=int, default=10, dest="n")

    # archive
    p_archive = subparsers.add_parser("archive", help="Archive completed tasks")
    p_archive.add_argument("-b", "--before", help="Archive before date YYYY-MM-DD")
    p_archive.add_argument("-d", "--days", type=int, default=30)
    p_archive.add_argument("-r", "--dry-run", action="store_true")

    # dashboard
    p_dashboard = subparsers.add_parser("dashboard", help="Generate HTML dashboard")
    p_dashboard.add_argument(
        "-v",
        "--view",
        default="kanban",
        choices=VIEWS,
    )
    p_dashboard.add_argument("-s", "--status", help="Filter by status")
    p_dashboard.add_argument("-p", "--priority", help="Filter by priority")
    p_dashboard.add_argument("-t", "--tags", help="Filter by tags")
    p_dashboard.add_argument(
        "-n", "--no-open", action="store_true", help="Don't open browser"
    )
    # mcp-stdio
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")

    args = parser.parse_args()

    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "task":
            content = args.content
            if not content and not sys.stdin.isatty():
                content = sys.stdin.read().strip()
            assert content, "content required (positional argument or stdin)"

            result, metrics = _task_impl(
                content=content,
                status=args.status,
                priority=args.priority,
                tags=args.tags,
                task_id=args.id,
                parent=args.parent,
                blocked_by=args.blocked_by,
                due_date=args.due,
            )
            print(json.dumps(result, indent=2))
            _log("INFO", "task", result["action"], metrics=json.dumps(metrics))
        elif args.command == "tasks":
            result, metrics = _tasks_impl(
                status=args.status,
                priority=args.priority,
                tags=args.tags,
                parent=args.parent,
                limit=args.limit,
            )
            print(json.dumps(result, indent=2))
            _log("INFO", "tasks", f"count={len(result)}", metrics=json.dumps(metrics))
        elif args.command == "passdown":
            content = args.content
            if not content and not sys.stdin.isatty():
                content = sys.stdin.read().strip()
            assert content, "content required (positional argument or stdin)"

            result, metrics = _passdown_impl(content, args.tags)
            print(json.dumps(result, indent=2))
            _log("INFO", "passdown", result["action"], metrics=json.dumps(metrics))
        elif args.command == "passdowns":
            result, metrics = _passdowns_impl(args.limit)
            print(json.dumps(result, indent=2))
            _log(
                "INFO", "passdowns", f"count={len(result)}", metrics=json.dumps(metrics)
            )
        elif args.command == "status":
            result, metrics = _status_impl()
            print(json.dumps(result, indent=2))
            _log(
                "INFO",
                "status",
                f"tasks={result['task_count']}",
                metrics=json.dumps(metrics),
            )
        elif args.command == "search":
            query = args.query
            if not query and not sys.stdin.isatty():
                query = sys.stdin.read().strip()
            assert query, "query required (positional argument or stdin)"

            result, metrics = _search_impl(
                query=query,
                type_filter=args.type,
                tags=args.tags,
                status=args.status,
                limit=args.limit,
            )
            print(json.dumps(result, indent=2))
            _log("INFO", "search", f"count={len(result)}", metrics=json.dumps(metrics))
        elif args.command == "blocked":
            result, metrics = _blocked_impl()
            print(json.dumps(result, indent=2))
            _log("INFO", "blocked", f"count={len(result)}", metrics=json.dumps(metrics))
        elif args.command == "recent":
            result, metrics = _recent_impl(args.n)
            print(json.dumps(result, indent=2))
            _log("INFO", "recent", f"count={len(result)}", metrics=json.dumps(metrics))
        elif args.command == "archive":
            result, metrics = _archive_impl(args.before, args.days, args.dry_run)
            print(json.dumps(result, indent=2))
            _log("INFO", "archive", result["status"], metrics=json.dumps(metrics))
        elif args.command == "dashboard":
            result, metrics = _dashboard_impl(
                view=args.view,
                status_filter=args.status,
                priority_filter=args.priority,
                tags_filter=args.tags,
                auto_open=not args.no_open,
            )
            print(json.dumps(result, indent=2))
            _log("INFO", "dashboard", f"view={args.view}", metrics=json.dumps(metrics))
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
    from fastmcp import FastMCP

    mcp = FastMCP("track")

    @mcp.tool()
    def task(
        content: str,
        status: str = "pending",
        priority: str = "medium",
        tags: str = "",
        task_id: str = "",
        parent: str = "",
        blocked_by: str = "",
        due_date: str = "",
    ) -> str:
        """Create or update task.

        Args:
            content: Task description
            status: pending, active, done, dropped, waiting
            priority: low, medium, high, critical
            tags: Comma-separated tags
            task_id: Update existing task if provided
            parent: Parent task ID for hierarchy
            blocked_by: Comma-separated blocking task IDs
            due_date: Due date YYYY-MM-DD

        Returns:
            JSON with task_id, action (created/updated), status
        """
        result, metrics = _task_impl(
            content, status, priority, tags, task_id, parent, blocked_by, due_date
        )
        return json.dumps(result, indent=2)

    @mcp.tool()
    def tasks(
        status: str = "",
        priority: str = "",
        tags: str = "",
        parent: str = "",
        limit: int = 50,
    ) -> str:
        """List tasks with optional filters.

        Args:
            status: Filter by status
            priority: Filter by priority
            tags: Comma-separated tags
            parent: Filter by parent task ID
            limit: Max results

        Returns:
            JSON list of tasks
        """
        result, metrics = _tasks_impl(status, priority, tags, parent, limit)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def passdown(content: str, tags: str = "") -> str:
        """Create session passdown note. Supersedes previous.

        Args:
            content: Passdown content
            tags: Comma-separated tags

        Returns:
            JSON with passdown_id
        """
        result, metrics = _passdown_impl(content, tags)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def passdowns(limit: int = 10) -> str:
        """List passdown notes.

        Args:
            limit: Number of passdowns to return

        Returns:
            JSON list of passdowns
        """
        result, metrics = _passdowns_impl(limit)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def status() -> str:
        """Get current status: pending/active/waiting tasks + current passdown.

        Args:
            (no arguments)

        Returns:
            JSON with task summary and current passdown
        """
        result, metrics = _status_impl()
        return json.dumps(result, indent=2)

    @mcp.tool()
    def search(
        query: str = "",
        type_filter: str = "",
        tags: str = "",
        status: str = "",
        limit: int = 20,
    ) -> str:
        """Search tasks and passdowns.

        Args:
            query: Search regex pattern
            type_filter: "task" or "passdown"
            tags: Comma-separated tags
            status: Filter by status
            limit: Max results

        Returns:
            JSON list of matching items
        """
        result, metrics = _search_impl(query, type_filter, tags, status, limit)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def blocked() -> str:
        """Show blocked tasks and their blockers.

        Args:
            (no arguments)

        Returns:
            JSON list of blocked tasks with blocking info
        """
        result, metrics = _blocked_impl()
        return json.dumps(result, indent=2)

    @mcp.tool()
    def recent(n: int = 10) -> str:
        """Show recently completed tasks.

        Args:
            n: Number of tasks to return

        Returns:
            JSON list of completed tasks
        """
        result, metrics = _recent_impl(n)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def archive(before_date: str = "", days: int = 30, dry_run: bool = False) -> str:
        """Archive completed tasks older than cutoff.

        Args:
            before_date: Archive before YYYY-MM-DD
            days: Archive older than N days
            dry_run: Show what would be archived

        Returns:
            JSON with archive results
        """
        result, metrics = _archive_impl(before_date, days, dry_run)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def dashboard(
        view: str = "kanban",
        status: str = "",
        priority: str = "",
        tags: str = "",
        auto_open: bool = False,
    ) -> str:
        """Generate HTML dashboard with task visualizations.

        Args:
            view: "kanban", "table", "deps", "timeline", or "chart"
            status: Filter by status
            priority: Filter by priority
            tags: Comma-separated tags
            auto_open: Whether to open browser automatically

        Returns:
            JSON with html_file path and metadata
        """
        result, metrics = _dashboard_impl(view, status, priority, tags, auto_open)
        return json.dumps(result, indent=2)

    print("Task tracking MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
