#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["fastmcp"]
# ///
"""
Reasoning — persistent structured thinking with branches and brain promotion.

Architecture:
  brain/thoughts.tsv → reasoning sessions, branches, thoughts
  No cache — dataset is small, TSV reads at invocation time.
  Promotion writes directly to brain tables via sft_brain.py CLI.

Usage:
  sft_reason.py new "purpose"                    # Start reasoning session
  sft_reason.py think "thought" [-s stage]       # Add thought to active session
  sft_reason.py revise "thought" -r N            # Revise thought #N
  sft_reason.py conclude "conclusion"            # End active session
  sft_reason.py branch NAME "reason"             # Create/switch branch
  sft_reason.py merge BRANCH [-s strategy]       # Merge branch to main
  sft_reason.py promote N -t TABLE               # Graduate thought to brain
  sft_reason.py search [-q query] [-t tags]      # Search reasoning history
  sft_reason.py status                           # Active session info
  sft_reason.py history [--limit N]              # Past sessions
  sft_reason.py export [session_id]              # Export as markdown/JSON
  sft_reason.py mcp-stdio                        # MCP server mode
"""

# =============================================================================
# EXPOSED — tools available via MCP
# =============================================================================

EXPOSED = [
    "think",
    "merge",
    "promote",
    "search",
    "status",
    "new",
    "history",
    "export",
]

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# LOGGING — TSV structured log
# =============================================================================

_LOG_LEVEL = os.environ.get("SFB_LOG_LEVEL", "INFO").upper()
_LOG_DIR = os.environ.get("SFB_LOG_DIR", str(Path(__file__).parent))
_LOG_FILE = Path(_LOG_DIR) / (Path(__file__).stem + "_log.tsv")
_LOG_LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
_HEADER = "#timestamp\tlevel\tevent\tmessage\tdetail\n"


def _log(level: str, event: str, msg: str, **kw):
    """Append structured TSV log line."""
    if _LOG_LEVELS.get(level, 1) < _LOG_LEVELS.get(_LOG_LEVEL, 1):
        return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    extra = " ".join(f"{k}={v}" for k, v in kw.items()) if kw else ""
    line = f"{ts}\t{level}\t{event}\t{msg}"
    if extra:
        line += f"\t{extra}"
    try:
        if not _LOG_FILE.exists() or _LOG_FILE.stat().st_size == 0:
            with open(_LOG_FILE, "w") as f:
                f.write(_HEADER)
        with open(_LOG_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


# =============================================================================
# CONFIGURATION
# =============================================================================

VERA_ROOT = Path(os.environ.get("VERA_ROOT", Path(__file__).parent.parent))
BRAIN_DIR = VERA_ROOT / "brain"
THOUGHTS_FILE = BRAIN_DIR / "thoughts.tsv"
BRAIN_SCRIPT = Path(__file__).parent / "sft_brain.py"

THOUGHTS_COLS = [
    "session_id",
    "thought_id",
    "type",
    "branch_id",
    "thought_number",
    "stage",
    "status",
    "tags",
    "content",
    "parent_thought",
    "revises_thought",
    "created",
]

CONFIG = {
    "version": "1.0.0",
    "thoughts_file": str(THOUGHTS_FILE),
    "brain_dir": str(BRAIN_DIR),
    "valid_stages": {"define", "research", "analyze", "synthesize", "conclude"},
    "valid_merge_strategies": {"conclusion", "full", "summary"},
}

# =============================================================================
# CORE FUNCTIONS
# =============================================================================


def _short_id() -> str:
    """Generate a short unique ID (8 hex chars)."""
    return uuid.uuid4().hex[:8]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _esc(s) -> str:
    """Escape a value for TSV storage."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\x00", "")
    return (
        s.replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def _unesc(s: str) -> str:
    """Unescape a TSV value."""
    return (
        s.replace("\\r", "\r")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\\", "\\")
    )


# =============================================================================
# TSV I/O
# =============================================================================


def _ensure_file():
    """Create thoughts.tsv with header if it doesn't exist."""
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    if not THOUGHTS_FILE.exists() or THOUGHTS_FILE.stat().st_size == 0:
        with open(THOUGHTS_FILE, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            w.writerow(THOUGHTS_COLS)


def _append_row(row: dict):
    """Append a single thought row to TSV."""
    _ensure_file()
    # Newline guard — ensure file ends with newline before appending
    with open(THOUGHTS_FILE, "rb") as f:
        f.seek(0, 2)
        if f.tell() > 0:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                with open(THOUGHTS_FILE, "a") as af:
                    af.write("\n")
    with open(THOUGHTS_FILE, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow([_esc(row.get(c, "")) for c in THOUGHTS_COLS])


def _read_all() -> list[dict]:
    """Read all rows from thoughts.tsv."""
    if not THOUGHTS_FILE.exists():
        return []
    with open(THOUGHTS_FILE, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        row["content"] = _unesc(row.get("content", ""))
    return rows


def _rewrite_all(rows: list[dict]):
    """Full rewrite of thoughts.tsv."""
    _ensure_file()
    with open(THOUGHTS_FILE, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(THOUGHTS_COLS)
        for row in rows:
            w.writerow([_esc(row.get(c, "")) for c in THOUGHTS_COLS])


# =============================================================================
# SESSION STATE
# =============================================================================

_active_session_id: str | None = None
_active_branch: str = "main"


def _find_active_session() -> str | None:
    """Find the most recent unconcluded reasoning session."""
    rows = _read_all()
    sessions: dict[str, dict] = {}
    for r in rows:
        if r.get("type") == "reason":
            sid = r["session_id"]
            if sid not in sessions:
                sessions[sid] = {"concluded": False, "created": r["created"]}
            if r.get("stage") == "conclude":
                sessions[sid]["concluded"] = True
    unconcluded = [
        (sid, info) for sid, info in sessions.items() if not info["concluded"]
    ]
    if unconcluded:
        unconcluded.sort(key=lambda x: x[1]["created"], reverse=True)
        return unconcluded[0][0]
    return None


def _get_session_rows(session_id: str) -> list[dict]:
    """Get all rows for a specific session."""
    return [r for r in _read_all() if r["session_id"] == session_id]


def _get_branch_rows(session_id: str, branch_id: str) -> list[dict]:
    """Get rows for a specific branch in a session."""
    return [
        r
        for r in _get_session_rows(session_id)
        if r.get("branch_id", "main") == branch_id and r.get("type") == "reason"
    ]


def _next_thought_number(session_id: str, branch_id: str) -> int:
    """Get the next thought number for a branch."""
    branch_rows = _get_branch_rows(session_id, branch_id)
    if not branch_rows:
        return 1
    nums = []
    for r in branch_rows:
        try:
            nums.append(int(r["thought_number"]))
        except (ValueError, KeyError):
            pass
    return max(nums) + 1 if nums else 1


def _get_branches(session_id: str) -> list[str]:
    """Get all branch names for a session."""
    rows = _get_session_rows(session_id)
    branches = set()
    for r in rows:
        if r.get("type") == "reason":
            branches.add(r.get("branch_id", "main") or "main")
    return sorted(branches)


# =============================================================================
# IMPLEMENTATION — Reasoning
# =============================================================================


def _think_impl(
    thought: str, stage: str = "", tags: str = ""
) -> tuple[dict, dict]:
    """Add a thought to the active reasoning session. CLI: think, MCP: think."""
    global _active_session_id, _active_branch
    start_ms = time.time() * 1000

    if not _active_session_id:
        _active_session_id = _find_active_session()
    if not _active_session_id:
        _active_session_id = _short_id()
        _active_branch = "main"
        _log("INFO", "think", f"Auto-created session {_active_session_id}")

    num = _next_thought_number(_active_session_id, _active_branch)
    thought_id = _short_id()

    row = {
        "session_id": _active_session_id,
        "thought_id": thought_id,
        "type": "reason",
        "branch_id": _active_branch,
        "thought_number": str(num),
        "stage": stage,
        "status": "",
        "tags": tags,
        "content": thought,
        "parent_thought": "",
        "revises_thought": "",
        "created": _now(),
    }
    _append_row(row)
    _log("INFO", "think", f"#{num} on {_active_branch} in {_active_session_id}")

    total = len(_get_session_rows(_active_session_id))
    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}
    result = {
        "thought_number": num,
        "thought_id": thought_id,
        "branch": _active_branch,
        "session_id": _active_session_id,
        "session_total": total,
    }
    return result, metrics


def _revise_impl(thought: str, revises: int) -> tuple[dict, dict]:
    """Revise a previous thought in the active session. CLI: revise."""
    global _active_session_id, _active_branch
    start_ms = time.time() * 1000

    if not _active_session_id:
        _active_session_id = _find_active_session()
    if not _active_session_id:
        latency_ms = time.time() * 1000 - start_ms
        return (
            {"status": "error", "message": "No active reasoning session"},
            {"status": "error", "latency_ms": round(latency_ms, 2)},
        )

    branch_rows = _get_branch_rows(_active_session_id, _active_branch)
    original = None
    for r in branch_rows:
        try:
            if int(r["thought_number"]) == revises:
                original = r
                break
        except (ValueError, KeyError):
            pass
    if not original:
        latency_ms = time.time() * 1000 - start_ms
        return (
            {
                "status": "error",
                "message": f"Thought #{revises} not found on branch {_active_branch}",
            },
            {"status": "error", "latency_ms": round(latency_ms, 2)},
        )

    num = _next_thought_number(_active_session_id, _active_branch)
    thought_id = _short_id()

    row = {
        "session_id": _active_session_id,
        "thought_id": thought_id,
        "type": "reason",
        "branch_id": _active_branch,
        "thought_number": str(num),
        "stage": original.get("stage", ""),
        "status": "",
        "tags": original.get("tags", ""),
        "content": thought,
        "parent_thought": "",
        "revises_thought": original["thought_id"],
        "created": _now(),
    }
    _append_row(row)
    _log("INFO", "revise", f"#{num} revises #{revises} in {_active_session_id}")

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}
    result = {
        "thought_number": num,
        "thought_id": thought_id,
        "revises": revises,
        "original_id": original["thought_id"],
        "branch": _active_branch,
    }
    return result, metrics


def _conclude_impl(conclusion: str) -> tuple[dict, dict]:
    """Complete the active reasoning session. CLI: conclude."""
    global _active_session_id, _active_branch
    start_ms = time.time() * 1000

    if not _active_session_id:
        _active_session_id = _find_active_session()
    if not _active_session_id:
        latency_ms = time.time() * 1000 - start_ms
        return (
            {"status": "error", "message": "No active reasoning session"},
            {"status": "error", "latency_ms": round(latency_ms, 2)},
        )

    num = _next_thought_number(_active_session_id, "main")
    thought_id = _short_id()

    row = {
        "session_id": _active_session_id,
        "thought_id": thought_id,
        "type": "reason",
        "branch_id": "main",
        "thought_number": str(num),
        "stage": "conclude",
        "status": "concluded",
        "tags": "conclusion",
        "content": conclusion,
        "parent_thought": "",
        "revises_thought": "",
        "created": _now(),
    }
    _append_row(row)

    session_rows = _get_session_rows(_active_session_id)
    branches = _get_branches(_active_session_id)
    key_thoughts = [
        r for r in session_rows if "key" in r.get("tags", "").split(",")
    ]

    _log(
        "INFO",
        "conclude",
        f"Session {_active_session_id} concluded with {len(session_rows)} thoughts",
    )

    sid = _active_session_id
    _active_session_id = None
    _active_branch = "main"

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}
    result = {
        "session_id": sid,
        "total_thoughts": len(session_rows),
        "branches": branches,
        "key_thoughts": len(key_thoughts),
        "concluded": True,
    }
    return result, metrics


# =============================================================================
# IMPLEMENTATION — Branching
# =============================================================================


def _branch_impl(name: str, reason: str) -> tuple[dict, dict]:
    """Create a new reasoning branch. CLI: branch."""
    global _active_session_id, _active_branch
    start_ms = time.time() * 1000

    if not _active_session_id:
        _active_session_id = _find_active_session()
    if not _active_session_id:
        latency_ms = time.time() * 1000 - start_ms
        return (
            {"status": "error", "message": "No active reasoning session"},
            {"status": "error", "latency_ms": round(latency_ms, 2)},
        )

    existing = _get_branches(_active_session_id)
    if name in existing:
        # Switch to existing branch
        _active_branch = name
        branch_rows = _get_branch_rows(_active_session_id, name)
        latency_ms = time.time() * 1000 - start_ms
        return (
            {
                "branch": name,
                "action": "switched",
                "thought_count": len(branch_rows),
            },
            {"status": "success", "latency_ms": round(latency_ms, 2)},
        )

    # New branch
    current_rows = _get_branch_rows(_active_session_id, _active_branch)
    parent_id = current_rows[-1]["thought_id"] if current_rows else ""

    thought_id = _short_id()
    row = {
        "session_id": _active_session_id,
        "thought_id": thought_id,
        "type": "reason",
        "branch_id": name,
        "thought_number": "1",
        "stage": "define",
        "status": "",
        "tags": "branch-origin",
        "content": f"Branch reason: {reason}",
        "parent_thought": parent_id,
        "revises_thought": "",
        "created": _now(),
    }
    _append_row(row)
    prev_branch = _active_branch
    _active_branch = name
    _log("INFO", "branch", f"Created branch '{name}' from {prev_branch}")

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}
    result = {
        "branch": name,
        "action": "created",
        "branched_from": prev_branch,
        "parent_thought": parent_id,
    }
    return result, metrics


def _merge_impl(
    branch: str, strategy: str = "conclusion"
) -> tuple[dict, dict]:
    """Merge branch insights back to main. CLI: merge."""
    global _active_session_id, _active_branch
    start_ms = time.time() * 1000

    if not _active_session_id:
        _active_session_id = _find_active_session()
    if not _active_session_id:
        latency_ms = time.time() * 1000 - start_ms
        return (
            {"status": "error", "message": "No active reasoning session"},
            {"status": "error", "latency_ms": round(latency_ms, 2)},
        )

    if strategy not in CONFIG["valid_merge_strategies"]:
        latency_ms = time.time() * 1000 - start_ms
        valid = ", ".join(sorted(CONFIG["valid_merge_strategies"]))
        return (
            {"status": "error", "message": f"Invalid strategy. Valid: {valid}"},
            {"status": "error", "latency_ms": round(latency_ms, 2)},
        )

    branch_rows = _get_branch_rows(_active_session_id, branch)
    if not branch_rows:
        latency_ms = time.time() * 1000 - start_ms
        return (
            {
                "status": "error",
                "message": f"Branch '{branch}' is empty or doesn't exist",
            },
            {"status": "error", "latency_ms": round(latency_ms, 2)},
        )

    num = _next_thought_number(_active_session_id, "main")

    if strategy == "conclusion":
        last = branch_rows[-1]
        content = f"[Merged from '{branch}' — conclusion only] {last['content']}"
    elif strategy == "full":
        parts = [f"#{r['thought_number']}: {r['content']}" for r in branch_rows]
        content = f"[Merged from '{branch}' — full integration]\n" + "\n".join(
            parts
        )
    else:  # summary
        content = (
            f"[Merged from '{branch}' — summary] "
            f"{len(branch_rows)} thoughts. "
            f"Origin: {branch_rows[0]['content'][:80]}. "
            f"Final: {branch_rows[-1]['content'][:80]}"
        )

    thought_id = _short_id()
    row = {
        "session_id": _active_session_id,
        "thought_id": thought_id,
        "type": "reason",
        "branch_id": "main",
        "thought_number": str(num),
        "stage": "synthesize",
        "status": "",
        "tags": f"merged,from-{branch}",
        "content": content,
        "parent_thought": branch_rows[-1]["thought_id"],
        "revises_thought": "",
        "created": _now(),
    }
    _append_row(row)
    _active_branch = "main"
    _log("INFO", "merge", f"Merged '{branch}' to main via {strategy}")

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}
    result = {
        "merged_branch": branch,
        "strategy": strategy,
        "thought_number": num,
        "source_thoughts": len(branch_rows),
    }
    return result, metrics


# =============================================================================
# IMPLEMENTATION — Promote & Search
# =============================================================================


def _promote_impl(
    thought_number: int, table: str, key: str = "", meta: str = ""
) -> tuple[dict, dict]:
    """Graduate a thought from active session to a brain entry. CLI: promote."""
    global _active_session_id, _active_branch
    start_ms = time.time() * 1000

    if not _active_session_id:
        _active_session_id = _find_active_session()
    if not _active_session_id:
        latency_ms = time.time() * 1000 - start_ms
        return (
            {"status": "error", "message": "No active reasoning session"},
            {"status": "error", "latency_ms": round(latency_ms, 2)},
        )

    # Find the thought
    branch_rows = _get_branch_rows(_active_session_id, _active_branch)
    target = None
    for r in branch_rows:
        try:
            if int(r["thought_number"]) == thought_number:
                target = r
                break
        except (ValueError, KeyError):
            pass
    if not target:
        latency_ms = time.time() * 1000 - start_ms
        return (
            {
                "status": "error",
                "message": f"Thought #{thought_number} not found on branch {_active_branch}",
            },
            {"status": "error", "latency_ms": round(latency_ms, 2)},
        )

    if not BRAIN_SCRIPT.exists():
        latency_ms = time.time() * 1000 - start_ms
        return (
            {"status": "error", "message": f"sft_brain.py not found at {BRAIN_SCRIPT}"},
            {"status": "error", "latency_ms": round(latency_ms, 2)},
        )

    # Shell out to sft_brain.py inscribe
    cmd = [
        "uv",
        "run",
        "--script",
        str(BRAIN_SCRIPT),
        "inscribe",
        "-t",
        table,
        "-E",  # skip embedding rebuild (caller can rebuild)
    ]
    if key:
        cmd.extend(["-k", key])
    if meta:
        cmd.extend(["-m", meta])
    else:
        cmd.extend(["-m", f"reason-{_active_session_id}"])
    cmd.append(target["content"])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(VERA_ROOT),
        )
        if proc.returncode != 0:
            _log("ERROR", "promote_fail", proc.stderr[:200])
            latency_ms = time.time() * 1000 - start_ms
            return (
                {"status": "error", "message": f"Brain inscribe failed: {proc.stderr[:200]}"},
                {"status": "error", "latency_ms": round(latency_ms, 2)},
            )
        brain_result = json.loads(proc.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        _log("ERROR", "promote_fail", str(e))
        latency_ms = time.time() * 1000 - start_ms
        return (
            {"status": "error", "message": str(e)},
            {"status": "error", "latency_ms": round(latency_ms, 2)},
        )

    _log(
        "INFO",
        "promote",
        f"Thought #{thought_number} → brain {table} key={brain_result.get('key', '?')}",
    )

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}
    result = {
        "promoted": True,
        "thought_number": thought_number,
        "table": table,
        "brain_key": brain_result.get("key", ""),
        "session_id": _active_session_id,
    }
    return result, metrics


def _search_impl(
    query: str = "", tags: str = "", status: str = "", limit: int = 20
) -> tuple[list[dict], dict]:
    """Search across reasoning sessions. CLI: search, MCP: search."""
    start_ms = time.time() * 1000

    rows = _read_all()

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

    rows = rows[-limit:]
    results = [
        {
            "thought_id": r.get("thought_id", ""),
            "session_id": r.get("session_id", ""),
            "branch": r.get("branch_id", ""),
            "number": r.get("thought_number", ""),
            "stage": r.get("stage", ""),
            "status": r.get("status", ""),
            "tags": r.get("tags", ""),
            "content": r.get("content", "")[:200],
            "created": r.get("created", ""),
        }
        for r in rows
    ]

    _log(
        "INFO",
        "search",
        f"query={query[:30] if query else 'none'} found={len(results)}",
    )

    latency_ms = time.time() * 1000 - start_ms
    metrics = {
        "status": "success",
        "count": len(results),
        "latency_ms": round(latency_ms, 2),
    }
    return results, metrics


# =============================================================================
# IMPLEMENTATION — Status / History / New / Export
# =============================================================================


def _status_impl() -> tuple[dict, dict]:
    """Get current reasoning state. CLI: status, MCP: status."""
    start_ms = time.time() * 1000

    active_sid = _find_active_session()
    session_info = None
    if active_sid:
        session_rows = _get_session_rows(active_sid)
        branches = set(r.get("branch_id", "main") for r in session_rows)
        session_info = {
            "session_id": active_sid,
            "thought_count": len(session_rows),
            "branches": sorted(branches),
            "last_thought": session_rows[-1]["content"][:100]
            if session_rows
            else None,
        }

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}
    return {"active_session": session_info}, metrics


def _history_impl(limit: int = 20) -> tuple[dict, dict]:
    """List past reasoning sessions. CLI: history, MCP: history."""
    start_ms = time.time() * 1000

    rows = _read_all()
    sessions: dict[str, dict] = {}
    for r in rows:
        if r.get("type") == "reason":
            sid = r["session_id"]
            if sid not in sessions:
                sessions[sid] = {
                    "session_id": sid,
                    "thought_count": 0,
                    "concluded": False,
                    "started": r["created"],
                    "branches": set(),
                }
            sessions[sid]["thought_count"] += 1
            sessions[sid]["branches"].add(r.get("branch_id", "main"))
            if r.get("stage") == "conclude":
                sessions[sid]["concluded"] = True

    result = sorted(
        sessions.values(), key=lambda x: x["started"], reverse=True
    )[:limit]
    for s in result:
        s["branches"] = sorted(s["branches"])

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}
    return {"sessions": result, "total": len(sessions)}, metrics


def _new_impl(purpose: str) -> tuple[dict, dict]:
    """Start a new reasoning session. CLI: new, MCP: new."""
    global _active_session_id, _active_branch
    start_ms = time.time() * 1000

    existing = _find_active_session()
    if existing:
        latency_ms = time.time() * 1000 - start_ms
        return (
            {
                "status": "error",
                "message": f"Session {existing} is still active. Conclude it first.",
            },
            {"status": "error", "latency_ms": round(latency_ms, 2)},
        )

    _active_session_id = _short_id()
    _active_branch = "main"

    thought_id = _short_id()
    row = {
        "session_id": _active_session_id,
        "thought_id": thought_id,
        "type": "reason",
        "branch_id": "main",
        "thought_number": "1",
        "stage": "define",
        "status": "",
        "tags": "purpose",
        "content": purpose,
        "parent_thought": "",
        "revises_thought": "",
        "created": _now(),
    }
    _append_row(row)
    _log(
        "INFO", "new_session", f"Started session {_active_session_id}: {purpose[:50]}"
    )

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}
    result = {
        "session_id": _active_session_id,
        "purpose": purpose,
    }
    return result, metrics


def _export_impl(
    session_id: str = "", fmt: str = "md"
) -> tuple[str, dict]:
    """Export a reasoning session as markdown or JSON. CLI: export, MCP: export."""
    start_ms = time.time() * 1000

    if not session_id:
        session_id = _find_active_session()
    if not session_id:
        latency_ms = time.time() * 1000 - start_ms
        return "No session to export.", {
            "status": "error",
            "latency_ms": round(latency_ms, 2),
        }

    session_rows = _get_session_rows(session_id)
    if not session_rows:
        latency_ms = time.time() * 1000 - start_ms
        return f"Session {session_id} not found.", {
            "status": "error",
            "latency_ms": round(latency_ms, 2),
        }

    if fmt == "json":
        output = json.dumps(session_rows, indent=2, default=str)
    else:
        lines = [f"# Reasoning Session {session_id}\n"]
        branches: dict[str, list[dict]] = {}
        for r in session_rows:
            if r.get("type") != "reason":
                continue
            bid = r.get("branch_id", "main") or "main"
            if bid not in branches:
                branches[bid] = []
            branches[bid].append(r)

        for bid in sorted(branches.keys(), key=lambda x: (x != "main", x)):
            lines.append(f"\n## Branch: {bid}\n")
            for r in branches[bid]:
                num = r.get("thought_number", "?")
                stage = f" [{r['stage']}]" if r.get("stage") else ""
                tag_str = f" `{r['tags']}`" if r.get("tags") else ""
                revision = (
                    f" (revises {r['revises_thought']})"
                    if r.get("revises_thought")
                    else ""
                )
                lines.append(f"### Thought {num}{stage}{tag_str}{revision}\n")
                lines.append(r.get("content", "") + "\n")
        output = "\n".join(lines)

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}
    return output, metrics


# =============================================================================
# UNIFIED THINK — collapsed surface for MCP
# =============================================================================


def _unified_think_impl(
    thought: str,
    next_needed: bool = True,
    stage: str = "",
    tags: str = "",
    is_revision: bool = False,
    revises: int = 0,
    branch_id: str = "",
    branch_from: int = 0,
) -> tuple[dict, dict]:
    """Unified entry point: normal thoughts, revisions, branches, conclusions.

    MCP: think.
    """
    global _active_session_id, _active_branch

    # === Conclude ===
    if not next_needed:
        return _conclude_impl(thought)

    # === Ensure session ===
    if not _active_session_id:
        _active_session_id = _find_active_session()
    if not _active_session_id:
        _active_session_id = _short_id()
        _active_branch = "main"
        _log("INFO", "think", f"Auto-created session {_active_session_id}")

    # === Branch handling ===
    if branch_id and branch_id != _active_branch:
        existing = _get_branches(_active_session_id)
        if branch_id not in existing:
            # New branch — this thought is the first on it
            start_ms = time.time() * 1000
            current_rows = _get_branch_rows(_active_session_id, _active_branch)
            parent_id = ""
            if branch_from:
                for r in current_rows:
                    try:
                        if int(r["thought_number"]) == branch_from:
                            parent_id = r["thought_id"]
                            break
                    except (ValueError, KeyError):
                        pass
            elif current_rows:
                parent_id = current_rows[-1]["thought_id"]

            thought_id = _short_id()
            row = {
                "session_id": _active_session_id,
                "thought_id": thought_id,
                "type": "reason",
                "branch_id": branch_id,
                "thought_number": "1",
                "stage": stage or "define",
                "status": "",
                "tags": tags or "branch-origin",
                "content": thought,
                "parent_thought": parent_id,
                "revises_thought": "",
                "created": _now(),
            }
            _append_row(row)
            _active_branch = branch_id
            _log("INFO", "branch", f"Created branch '{branch_id}'")

            latency_ms = time.time() * 1000 - start_ms
            return (
                {
                    "thought_number": 1,
                    "thought_id": thought_id,
                    "branch": branch_id,
                    "session_id": _active_session_id,
                    "branches": _get_branches(_active_session_id),
                    "next_needed": True,
                },
                {"status": "success", "latency_ms": round(latency_ms, 2)},
            )
        else:
            _active_branch = branch_id

    # === Revision ===
    if is_revision and revises > 0:
        return _revise_impl(thought, revises)

    # === Normal thought ===
    return _think_impl(thought, stage=stage, tags=tags)


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Persistent structured thinking with branches and brain promotion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sft_reason.py new "Evaluate caching strategies for brain queries"
  sft_reason.py think "First, consider the access patterns..."
  sft_reason.py think "On reflection, LRU is simpler" -s analyze
  sft_reason.py revise "Actually, TTL-based is better for this" -r 2
  sft_reason.py branch alternatives "Explore a completely different approach"
  sft_reason.py merge alternatives -s conclusion
  sft_reason.py conclude "Decision: use TTL with 5-minute window"
  sft_reason.py promote 3 -t fact -k "cache-strategy-ttl"
  sft_reason.py status
  sft_reason.py history --limit 5
  sft_reason.py export abc12345 --format json
""",
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {CONFIG['version']}"
    )
    sub = parser.add_subparsers(dest="command")

    # new
    p_new = sub.add_parser("new", help="Start a new reasoning session")
    p_new.add_argument("purpose", help="Purpose of the reasoning session")

    # think
    p_think = sub.add_parser("think", help="Add a thought to active session")
    p_think.add_argument("thought", help="The thought content")
    p_think.add_argument("-s", "--stage", default="", help="Stage: define, research, analyze, synthesize")
    p_think.add_argument("-t", "--tags", default="", help="Comma-separated tags")

    # revise
    p_revise = sub.add_parser("revise", help="Revise a previous thought")
    p_revise.add_argument("thought", help="The revised thought content")
    p_revise.add_argument("-r", "--revises", type=int, required=True, help="Thought number to revise")

    # conclude
    p_conclude = sub.add_parser("conclude", help="Conclude the active session")
    p_conclude.add_argument("conclusion", help="The conclusion")

    # branch
    p_branch = sub.add_parser("branch", help="Create or switch to a branch")
    p_branch.add_argument("name", help="Branch name")
    p_branch.add_argument("reason", nargs="?", default="", help="Reason for branching")

    # merge
    p_merge = sub.add_parser("merge", help="Merge branch insights to main")
    p_merge.add_argument("branch", help="Branch to merge")
    p_merge.add_argument(
        "-s",
        "--strategy",
        default="conclusion",
        choices=sorted(CONFIG["valid_merge_strategies"]),
        help="Merge strategy",
    )

    # promote
    p_promote = sub.add_parser("promote", help="Graduate thought to brain")
    p_promote.add_argument("number", type=int, help="Thought number to promote")
    p_promote.add_argument(
        "-t", "--table", required=True, help="Brain table: fact, opinion, question, aspiration"
    )
    p_promote.add_argument("-k", "--key", default="", help="Semantic key for brain entry")
    p_promote.add_argument("-m", "--meta", default="", help="Metadata for brain entry")

    # search
    p_search = sub.add_parser("search", help="Search reasoning history")
    p_search.add_argument("-q", "--query", default="", help="Search pattern (regex)")
    p_search.add_argument("-t", "--tags", default="", help="Filter by tags")
    p_search.add_argument("-s", "--status", default="", help="Filter by status")
    p_search.add_argument("-l", "--limit", type=int, default=20, help="Max results")

    # status
    sub.add_parser("status", help="Show active reasoning session")

    # history
    p_history = sub.add_parser("history", help="List past reasoning sessions")
    p_history.add_argument("-l", "--limit", type=int, default=20, help="Max sessions")

    # export
    p_export = sub.add_parser("export", help="Export session as markdown/JSON")
    p_export.add_argument("session_id", nargs="?", default="", help="Session ID (default: active)")
    p_export.add_argument(
        "-f", "--format", dest="fmt", default="md", choices=["md", "json"], help="Output format"
    )

    # mcp-stdio
    sub.add_parser("mcp-stdio", help="Run as MCP stdio server")

    # Parse — support stdin for think/revise/conclude
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "mcp-stdio":
        _run_mcp()
        return

    # stdin support for content arguments
    if args.command in ("think", "revise", "conclude", "new") and not sys.stdin.isatty():
        stdin_content = sys.stdin.read().strip()
        if stdin_content:
            if args.command == "new" and not args.purpose:
                args.purpose = stdin_content
            elif args.command in ("think", "revise") and not args.thought:
                args.thought = stdin_content
            elif args.command == "conclude" and not args.conclusion:
                args.conclusion = stdin_content

    try:
        if args.command == "new":
            result, _ = _new_impl(args.purpose)
        elif args.command == "think":
            result, _ = _think_impl(args.thought, stage=args.stage, tags=args.tags)
        elif args.command == "revise":
            result, _ = _revise_impl(args.thought, args.revises)
        elif args.command == "conclude":
            result, _ = _conclude_impl(args.conclusion)
        elif args.command == "branch":
            result, _ = _branch_impl(args.name, args.reason)
        elif args.command == "merge":
            result, _ = _merge_impl(args.branch, args.strategy)
        elif args.command == "promote":
            result, _ = _promote_impl(args.number, args.table, key=args.key, meta=args.meta)
        elif args.command == "search":
            result, _ = _search_impl(
                query=args.query, tags=args.tags, status=args.status, limit=args.limit
            )
        elif args.command == "status":
            result, _ = _status_impl()
        elif args.command == "history":
            result, _ = _history_impl(limit=args.limit)
        elif args.command == "export":
            output, _ = _export_impl(session_id=args.session_id, fmt=args.fmt)
            if args.fmt == "md":
                print(output)
            else:
                print(output)
            return
        else:
            parser.print_help()
            sys.exit(1)

        print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        _log("ERROR", "cli_error", str(e))
        print(json.dumps({"status": "error", "message": str(e)}), file=sys.stderr)
        sys.exit(1)


# =============================================================================
# FASTMCP SERVER
# =============================================================================


def _run_mcp():
    """Build and run the FastMCP server."""
    from fastmcp import FastMCP

    mcp = FastMCP("reason")

    @mcp.tool()
    def think(
        thought: str,
        next_needed: bool = True,
        stage: str = "",
        tags: str = "",
        is_revision: bool = False,
        revises: int = 0,
        branch_id: str = "",
        branch_from: int = 0,
    ) -> str:
        """Persistent structured thinking — sequential thoughts that survive context loss.

        Use the bookend pattern: opening thought → deep internal reasoning → closing thought.
        Minimum two thoughts per session. Branch at real pivot points, not every step.

        Args:
            thought: Your current thinking step
            next_needed: True if more thinking needed, False to conclude session
            stage: define, research, analyze, synthesize, conclude
            tags: Comma-separated — key, hypothesis, evidence, decision
            is_revision: True if this revises previous thinking
            revises: Which thought number to reconsider
            branch_id: Think on a named branch (auto-creates if new)
            branch_from: Which thought number to branch from
        """
        result, _ = _unified_think_impl(
            thought,
            next_needed=next_needed,
            stage=stage,
            tags=tags,
            is_revision=is_revision,
            revises=revises,
            branch_id=branch_id,
            branch_from=branch_from,
        )
        return json.dumps(result, indent=2)

    @mcp.tool()
    def merge(branch: str, strategy: str = "conclusion") -> str:
        """Merge branch insights to main.

        Args:
            branch: Branch name to merge
            strategy: conclusion, full, or summary
        """
        result, _ = _merge_impl(branch, strategy)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def promote(
        thought_number: int, table: str, key: str = "", meta: str = ""
    ) -> str:
        """Graduate a thought from active session to a brain entry.

        Args:
            thought_number: Which thought to promote
            table: Brain table — fact, opinion, question, aspiration
            key: Semantic key (auto-generated if omitted)
            meta: Metadata (source/basis/context/type)
        """
        result, _ = _promote_impl(thought_number, table, key=key, meta=meta)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def search(
        query: str = "", tags: str = "", status: str = "", limit: int = 20
    ) -> str:
        """Search across reasoning sessions.

        Args:
            query: Regex search pattern
            tags: Comma-separated tag filter
            status: Filter by status
            limit: Max results
        """
        result, _ = _search_impl(query, tags=tags, status=status, limit=limit)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def status() -> str:
        """Get current reasoning state — active session info.

        Args:
            (none)
        """
        result, _ = _status_impl()
        return json.dumps(result, indent=2)

    @mcp.tool()
    def new(purpose: str) -> str:
        """Start a new reasoning session.

        Args:
            purpose: What this reasoning session is about
        """
        result, _ = _new_impl(purpose)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def history(limit: int = 20) -> str:
        """List past reasoning sessions.

        Args:
            limit: Max sessions to return
        """
        result, _ = _history_impl(limit=limit)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def export(session_id: str = "", fmt: str = "md") -> str:
        """Export a reasoning session as markdown or JSON.

        Args:
            session_id: Session to export (default: active)
            fmt: Output format — md or json
        """
        output, _ = _export_impl(session_id=session_id, fmt=fmt)
        return output

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
