#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp"]
# ///
"""Chat History — OpenCode session reader for startup context and post-compaction recovery.

Reads OpenCode's SQLite database to reconstruct conversation history.
Scoped to the current project directory (detected from cwd).
Primary use case: get the tail of the previous or current session so Vera
can pick up context after startup or compaction.

Data source:
  ~/.local/share/opencode/opencode.db (SQLite)
    project     — project registry
    session     — session metadata (title, directory, time, summary)
    message     — message envelopes (role, time, tokens in JSON data column)
    part        — message content (text, tool calls in JSON data column)

Usage:
    sft_chat_history_oc.py sessions                        # List project sessions
    sft_chat_history_oc.py sessions -n 10                  # List 10 most recent
    sft_chat_history_oc.py session-current                  # Metadata for current session
    sft_chat_history_oc.py session-last                     # Metadata for previous session
    sft_chat_history_oc.py session-messages-cache ses_XXX    # Cache session messages to .cache/
    sft_chat_history_oc.py messages-tail-current             # Message tail of current session
    sft_chat_history_oc.py messages-tail-current -n 30       # Last 30 messages
    sft_chat_history_oc.py messages-tail-last                # Message tail of previous session
    sft_chat_history_oc.py messages-tail-last -n 50 -t       # Last 50 with tool details
    sft_chat_history_oc.py mcp-stdio
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# LOGGING (TSV format)
# =============================================================================
_LEVELS = {"TRACE": 5, "DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40, "FATAL": 50}
_THRESHOLD = _LEVELS.get(os.environ.get("SFB_LOG_LEVEL", "INFO"), 20)
_LOG_DIR = os.environ.get("SFB_LOG_DIR", "")
_SCRIPT = Path(__file__).stem
_LOG = Path(_LOG_DIR) / f"{_SCRIPT}_log.tsv" if _LOG_DIR else Path(__file__).parent / f"{_SCRIPT}_log.tsv"
_HEADER = "#timestamp\tscript\tlevel\tevent\tmessage\tdetail\tmetrics\ttrace\n"


def _log(level: str, event: str, msg: str, *, detail: str = "", metrics: str = "", trace: str = ""):
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
EXPOSED = ["sessions", "session_current", "session_last", "session_messages_cache", "messages_tail_current", "messages_tail_last", "search"]

OC_DB = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
CACHE_DIR = Path.cwd() / ".cache"

CONFIG = {
    "default_tail_messages": 20,
    "default_list_count": 20,
    "max_tool_output_chars": 200,
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def _get_db() -> sqlite3.Connection:
    """Open a read-only connection to the OC database."""
    assert OC_DB.exists(), f"OpenCode database not found at {OC_DB}"
    conn = sqlite3.connect(f"file:{OC_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _epoch_ms_to_iso(ms: int) -> str:
    """Convert epoch milliseconds to ISO 8601 datetime string."""
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _epoch_ms_to_short(ms: int) -> str:
    """Convert epoch milliseconds to short time string for display."""
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")


def _project_dir() -> str:
    """Return the current working directory as the project directory for session filtering."""
    return str(Path.cwd().resolve())


def _render_message(role: str, created_ms: int, parts: list[dict], include_tools: bool = False) -> str:
    """Render a message and its parts into readable text.

    Args:
        role: Message role (user, assistant).
        created_ms: Message creation time in epoch ms.
        parts: List of part data dicts.
        include_tools: Whether to include tool call/result details.

    Returns:
        Formatted string for this message.
    """
    ts = _epoch_ms_to_short(created_ms)
    role_label = "HUMAN" if role == "user" else "ASSISTANT" if role == "assistant" else role.upper()

    lines = []
    text_parts = []
    tool_parts = []

    for part in parts:
        ptype = part.get("type", "")

        if ptype == "text":
            text = part.get("text", "").strip()
            if text:
                text_parts.append(text)

        elif ptype == "tool" and include_tools:
            tool_name = part.get("tool", part.get("callID", "unknown"))
            state = part.get("state", {})
            tool_input = state.get("input", {})
            tool_output = state.get("output", "")

            input_summary = ""
            if isinstance(tool_input, dict):
                if "command" in tool_input:
                    input_summary = tool_input["command"][:120]
                elif "file_path" in tool_input:
                    input_summary = tool_input["file_path"]
                elif "query" in tool_input:
                    input_summary = tool_input["query"][:120]
                elif "pattern" in tool_input:
                    input_summary = tool_input["pattern"][:120]
                else:
                    input_summary = str(tool_input)[:120]

            output_summary = ""
            if tool_output and isinstance(tool_output, str):
                max_chars = CONFIG["max_tool_output_chars"]
                output_summary = tool_output[:max_chars]
                if len(tool_output) > max_chars:
                    output_summary += "..."

            tool_parts.append(f"  [{tool_name}] {input_summary}")
            if output_summary:
                tool_parts.append(f"    -> {output_summary}")

        elif ptype in ("step-start", "step-finish"):
            pass

    if text_parts:
        combined_text = "\n".join(text_parts)
        lines.append(f"[{ts}] {role_label}:")
        lines.append(combined_text)
    elif tool_parts and include_tools:
        lines.append(f"[{ts}] {role_label}: (tools only)")

    if tool_parts and include_tools:
        lines.extend(tool_parts)

    return "\n".join(lines)


# =============================================================================
# IMPL FUNCTIONS (CLI + MCP call these)
# =============================================================================

def _sessions_impl(count: int = 20) -> list[dict]:
    """List sessions for the current project directory.

    CLI: sessions
    MCP: sessions

    Args:
        count: Maximum number of sessions to return.

    Returns:
        List of session metadata dicts.
    """
    t0 = time.time()
    project_dir = _project_dir()
    conn = _get_db()

    rows = conn.execute("""
        SELECT s.id, s.title, s.directory, s.parent_id,
               s.time_created, s.time_updated,
               (SELECT COUNT(*) FROM message m WHERE m.session_id = s.id) as num_messages,
               (SELECT COUNT(*) FROM part p
                WHERE p.session_id = s.id
                AND json_extract(p.data, '$.type') = 'text') as num_text
        FROM session s
        WHERE s.directory = ?
        ORDER BY s.time_updated DESC
        LIMIT ?
    """, (project_dir, count)).fetchall()
    conn.close()

    results = [{
        "session_id": r["id"],
        "title": r["title"] or "",
        "start_datetime": _epoch_ms_to_iso(r["time_created"]),
        "stop_datetime": _epoch_ms_to_iso(r["time_updated"]),
        "num_messages": r["num_messages"],
        "num_text": r["num_text"],
        "parent_id": r["parent_id"] or "",
    } for r in rows]

    elapsed = time.time() - t0
    _log("INFO", "sessions", f"Listed {len(results)} sessions",
         detail=f"directory={project_dir} count={count}",
         metrics=f"results={len(results)} latency_ms={round(elapsed * 1000, 1)}")
    return results


def _session_current_impl() -> dict:
    """Metadata for the current (most recently updated) top-level session.

    CLI: session-current
    MCP: session_current

    Returns:
        Session metadata dict.
    """
    project_dir = _project_dir()
    conn = _get_db()

    row = conn.execute("""
        SELECT s.id, s.title, s.time_created, s.time_updated,
               (SELECT COUNT(*) FROM message m WHERE m.session_id = s.id) as num_messages,
               (SELECT COUNT(*) FROM part p
                WHERE p.session_id = s.id
                AND json_extract(p.data, '$.type') = 'text') as num_text
        FROM session s
        WHERE s.directory = ? AND s.parent_id IS NULL
        ORDER BY s.time_updated DESC LIMIT 1
    """, (project_dir,)).fetchone()
    conn.close()

    assert row, f"No sessions found for {project_dir}"
    return {
        "session_id": row["id"],
        "title": row["title"] or "",
        "start_datetime": _epoch_ms_to_iso(row["time_created"]),
        "stop_datetime": _epoch_ms_to_iso(row["time_updated"]),
        "num_messages": row["num_messages"],
        "num_text": row["num_text"],
    }


def _session_last_impl() -> dict:
    """Metadata for the previous (second most recently updated) top-level session.

    CLI: session-last
    MCP: session_last

    Returns:
        Session metadata dict.
    """
    project_dir = _project_dir()
    conn = _get_db()

    rows = conn.execute("""
        SELECT s.id, s.title, s.time_created, s.time_updated,
               (SELECT COUNT(*) FROM message m WHERE m.session_id = s.id) as num_messages,
               (SELECT COUNT(*) FROM part p
                WHERE p.session_id = s.id
                AND json_extract(p.data, '$.type') = 'text') as num_text
        FROM session s
        WHERE s.directory = ? AND s.parent_id IS NULL
        ORDER BY s.time_updated DESC LIMIT 2
    """, (project_dir,)).fetchall()
    conn.close()

    assert len(rows) >= 2, f"Need at least 2 top-level sessions, found {len(rows)}"
    row = rows[1]
    return {
        "session_id": row["id"],
        "title": row["title"] or "",
        "start_datetime": _epoch_ms_to_iso(row["time_created"]),
        "stop_datetime": _epoch_ms_to_iso(row["time_updated"]),
        "num_messages": row["num_messages"],
        "num_text": row["num_text"],
    }


def _session_messages_cache_impl(session_id: str) -> dict:
    """Cache a session's messages to .cache/ as a single pre-joined JSON file.

    Reads all messages and parts from the OC database, assembles them into one
    file at .cache/oc_session_<session_id>.json for fast subsequent reads.

    CLI: session-messages-cache
    MCP: session_messages_cache

    Args:
        session_id: The session ID to cache.

    Returns:
        Dict with cache path, session metadata, and message count.
    """
    t0 = time.time()
    assert session_id, "session_id required"
    conn = _get_db()

    # Session metadata
    session = conn.execute("""
        SELECT id, title, directory, time_created, time_updated
        FROM session WHERE id = ?
    """, (session_id,)).fetchone()
    assert session, f"Session {session_id} not found"

    # All messages
    msg_rows = conn.execute("""
        SELECT id, time_created, time_updated, data
        FROM message WHERE session_id = ?
        ORDER BY time_created ASC
    """, (session_id,)).fetchall()

    # All parts for this session, grouped by message
    part_rows = conn.execute("""
        SELECT id, message_id, data
        FROM part WHERE session_id = ?
        ORDER BY time_created ASC
    """, (session_id,)).fetchall()
    conn.close()

    # Index parts by message_id
    parts_by_msg = {}
    for p in part_rows:
        mid = p["message_id"]
        if mid not in parts_by_msg:
            parts_by_msg[mid] = []
        pdata = json.loads(p["data"])
        pdata["id"] = p["id"]
        parts_by_msg[mid].append(pdata)

    # Assemble messages
    messages = []
    num_text = 0
    for m in msg_rows:
        mdata = json.loads(m["data"])
        msg_parts = parts_by_msg.get(m["id"], [])
        num_text += sum(1 for p in msg_parts if p.get("type") == "text")
        messages.append({
            "message_id": m["id"],
            "role": mdata.get("role", ""),
            "created": _epoch_ms_to_iso(m["time_created"]),
            "model": mdata.get("modelID", ""),
            "tokens": mdata.get("tokens", {}),
            "cost": mdata.get("cost", 0),
            "parent_id": mdata.get("parentID", ""),
            "parts": msg_parts,
        })

    cache_data = {
        "session_id": session_id,
        "title": session["title"] or "",
        "start_datetime": _epoch_ms_to_iso(session["time_created"]),
        "stop_datetime": _epoch_ms_to_iso(session["time_updated"]),
        "num_messages": len(messages),
        "num_text": num_text,
        "cached_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "messages": messages,
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"oc_session_{session_id}.json"
    with open(cache_path, "w") as fh:
        json.dump(cache_data, fh, indent=2)

    elapsed = time.time() - t0
    cache_size_kb = cache_path.stat().st_size / 1024

    result = {
        "session_id": session_id,
        "title": session["title"] or "",
        "cache_path": str(cache_path),
        "num_messages": len(messages),
        "num_text": num_text,
        "cache_size_kb": round(cache_size_kb, 1),
    }

    _log("INFO", "session_messages_cache", f"Cached {len(messages)} messages to {cache_path}",
         detail=f"session_id={session_id} num_text={num_text}",
         metrics=f"messages={len(messages)} size_kb={cache_size_kb:.1f} latency_ms={round(elapsed * 1000, 1)}")
    return result


def _messages_tail_impl(session_id: str, count: int = 20, include_tools: bool = False) -> dict:
    """Get the tail of a specific session rendered as readable conversation.

    Args:
        session_id: Session ID.
        count: Number of messages from the end to include.
        include_tools: Whether to include tool call details.

    Returns:
        Dict with session metadata and rendered conversation tail.
    """
    t0 = time.time()
    assert session_id, "session_id required"
    conn = _get_db()

    # Session metadata
    session = conn.execute("""
        SELECT id, title, time_created, time_updated
        FROM session WHERE id = ?
    """, (session_id,)).fetchone()
    assert session, f"Session {session_id} not found"

    # Total message count
    total = conn.execute(
        "SELECT COUNT(*) as c FROM message WHERE session_id = ?", (session_id,)
    ).fetchone()["c"]

    # Tail messages
    msg_rows = conn.execute("""
        SELECT id, time_created, data
        FROM message WHERE session_id = ?
        ORDER BY time_created DESC LIMIT ?
    """, (session_id, count)).fetchall()
    msg_rows = list(reversed(msg_rows))  # Back to chronological

    # Parts for these messages
    msg_ids = [r["id"] for r in msg_rows]
    if msg_ids:
        placeholders = ",".join("?" * len(msg_ids))
        part_rows = conn.execute(f"""
            SELECT message_id, data FROM part
            WHERE message_id IN ({placeholders})
            ORDER BY time_created ASC
        """, msg_ids).fetchall()
    else:
        part_rows = []
    conn.close()

    # Index parts by message
    parts_by_msg = {}
    for p in part_rows:
        mid = p["message_id"]
        if mid not in parts_by_msg:
            parts_by_msg[mid] = []
        parts_by_msg[mid].append(json.loads(p["data"]))

    # Render
    rendered_lines = []
    for m in msg_rows:
        mdata = json.loads(m["data"])
        role = mdata.get("role", "")
        msg_parts = parts_by_msg.get(m["id"], [])
        rendered = _render_message(role, m["time_created"], msg_parts, include_tools=include_tools)
        if rendered:
            rendered_lines.append(rendered)

    result = {
        "session_id": session_id,
        "title": session["title"] or "",
        "start_datetime": _epoch_ms_to_iso(session["time_created"]),
        "stop_datetime": _epoch_ms_to_iso(session["time_updated"]),
        "total_messages": total,
        "showing": len(msg_rows),
        "conversation": "\n\n".join(rendered_lines),
    }

    elapsed = time.time() - t0
    _log("INFO", "messages_tail", f"Rendered {len(msg_rows)}/{total} messages",
         detail=f"session_id={session_id} count={count}",
         metrics=f"total={total} rendered={len(msg_rows)} latency_ms={round(elapsed * 1000, 1)}")
    return result


def _messages_tail_current_impl(count: int = 20, include_tools: bool = False) -> dict:
    """Message tail of the current (most recently updated) session.

    CLI: messages-tail-current
    MCP: messages_tail_current

    Args:
        count: Number of messages from the end to include.
        include_tools: Whether to include tool call details.

    Returns:
        Dict with session metadata and rendered conversation tail.
    """
    project_dir = _project_dir()
    conn = _get_db()
    row = conn.execute("""
        SELECT id FROM session
        WHERE directory = ? AND parent_id IS NULL
        ORDER BY time_updated DESC LIMIT 1
    """, (project_dir,)).fetchone()
    conn.close()
    assert row, f"No sessions found for {project_dir}"
    return _messages_tail_impl(session_id=row["id"], count=count, include_tools=include_tools)


def _messages_tail_last_impl(count: int = 20, include_tools: bool = False) -> dict:
    """Message tail of the previous (second most recently updated) session.

    CLI: messages-tail-last
    MCP: messages_tail_last

    Args:
        count: Number of messages from the end to include.
        include_tools: Whether to include tool call details.

    Returns:
        Dict with session metadata and rendered conversation tail.
    """
    project_dir = _project_dir()
    conn = _get_db()
    rows = conn.execute("""
        SELECT id FROM session
        WHERE directory = ? AND parent_id IS NULL
        ORDER BY time_updated DESC LIMIT 2
    """, (project_dir,)).fetchall()
    conn.close()
    assert len(rows) >= 2, f"Need at least 2 top-level sessions, found {len(rows)}"
    return _messages_tail_impl(session_id=rows[1]["id"], count=count, include_tools=include_tools)


def _search_impl(pattern: str, count: int = 20, role: str = "", all_projects: bool = False) -> list[dict]:
    """Search message text across sessions directly from SQLite.

    CLI: search (current project), search-all (all projects)
    MCP: search

    Args:
        pattern: Regex pattern to search message text.
        count: Maximum number of matches to return.
        role: Filter by role ("user" or "assistant", empty for both).
        all_projects: Search all projects, not just current directory.

    Returns:
        List of match dicts with session, role, timestamp, and matched text.
    """
    import re
    assert pattern, "search pattern required"
    t0 = time.time()
    regex = re.compile(pattern, re.IGNORECASE)
    conn = _get_db()

    # Build query — join part -> message -> session, extract text parts
    where_clauses = ["json_extract(p.data, '$.type') = 'text'"]
    params = []

    if not all_projects:
        where_clauses.append("s.directory = ?")
        params.append(_project_dir())

    if role:
        where_clauses.append("json_extract(m.data, '$.role') = ?")
        params.append(role)

    where_sql = " AND ".join(where_clauses)

    rows = conn.execute(f"""
        SELECT p.data as part_data, m.data as msg_data, m.time_created,
               s.id as session_id, s.title as session_title, s.directory
        FROM part p
        JOIN message m ON p.message_id = m.id
        JOIN session s ON p.session_id = s.id
        WHERE {where_sql}
        ORDER BY m.time_created DESC
    """, params).fetchall()
    conn.close()

    results = []
    for row in rows:
        pdata = json.loads(row["part_data"])
        text = pdata.get("text", "")
        if not text or not regex.search(text):
            continue

        mdata = json.loads(row["msg_data"])
        results.append({
            "session_id": row["session_id"],
            "session_title": row["session_title"],
            "directory": row["directory"],
            "role": mdata.get("role", ""),
            "timestamp": _epoch_ms_to_iso(row["time_created"]),
            "text": text[:500],
        })
        if len(results) >= count:
            break

    elapsed = time.time() - t0
    _log("INFO", "search", f"Found {len(results)} matches for '{pattern}'",
         detail=f"pattern={pattern} role={role or 'all'} all_projects={all_projects}",
         metrics=f"scanned={len(rows)} matches={len(results)} latency_ms={round(elapsed * 1000, 1)}")
    return results


# =============================================================================
# CLI INTERFACE (argparse)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="sft_chat_history_oc",
        description="OpenCode chat history reader — session metadata and message tails",
    )
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 1.0.0")
    sub = parser.add_subparsers(dest="command")

    # CLI for _sessions_impl
    p_sessions = sub.add_parser("sessions", help="List project sessions")
    p_sessions.add_argument("-n", "--count", type=int, default=CONFIG["default_list_count"],
                            help="Max sessions to list")

    # CLI for _session_current_impl
    sub.add_parser("session-current", help="Metadata for the current session")

    # CLI for _session_last_impl
    sub.add_parser("session-last", help="Metadata for the previous session")

    # CLI for _session_messages_cache_impl
    p_cache = sub.add_parser("session-messages-cache", help="Cache a session's messages to .cache/")
    p_cache.add_argument("session_id", nargs="?", default="",
                         help="Session ID to cache (or pipe one per line via stdin)")

    # CLI for _messages_tail_current_impl
    p_mtc = sub.add_parser("messages-tail-current", help="Message tail of the current session")
    p_mtc.add_argument("-n", "--count", type=int, default=CONFIG["default_tail_messages"],
                       help="Number of messages from the end")
    p_mtc.add_argument("-t", "--tools", action="store_true",
                       help="Include tool call details")
    p_mtc.add_argument("session_id", nargs="?", default="",
                       help="Session ID (default: most recent)")

    # CLI for _messages_tail_last_impl
    p_mtl = sub.add_parser("messages-tail-last", help="Message tail of the previous session")
    p_mtl.add_argument("-n", "--count", type=int, default=CONFIG["default_tail_messages"],
                       help="Number of messages from the end")
    p_mtl.add_argument("-t", "--tools", action="store_true",
                       help="Include tool call details")
    p_mtl.add_argument("session_id", nargs="?", default="",
                       help="Session ID (default: second most recent)")

    # CLI for _search_impl (current project)
    p_srch = sub.add_parser("search", help="Search message text across sessions (current project)")
    p_srch.add_argument("pattern", nargs="?", help="Regex pattern to search")
    p_srch.add_argument("-n", "--count", type=int, default=20, help="Max results")
    p_srch.add_argument("-r", "--role", default="", help="Filter by role (user/assistant)")

    # CLI for _search_impl (all projects)
    p_srcha = sub.add_parser("search-all", help="Search message text across ALL projects")
    p_srcha.add_argument("pattern", nargs="?", help="Regex pattern to search")
    p_srcha.add_argument("-n", "--count", type=int, default=20, help="Max results")
    p_srcha.add_argument("-r", "--role", default="", help="Filter by role (user/assistant)")

    # MCP
    sub.add_parser("mcp-stdio", help="Run as MCP server")

    args = parser.parse_args()

    try:
        if args.command == "mcp-stdio":
            _run_mcp()
            return

        elif args.command == "sessions":
            results = _sessions_impl(count=args.count)
            print(json.dumps(results, indent=2))

        elif args.command == "session-current":
            print(json.dumps(_session_current_impl(), indent=2))

        elif args.command == "session-last":
            print(json.dumps(_session_last_impl(), indent=2))

        elif args.command == "session-messages-cache":
            sid = args.session_id
            if not sid and not sys.stdin.isatty():
                sids = [line.strip() for line in sys.stdin if line.strip()]
            else:
                sids = [sid] if sid else []
            assert sids, "session_id required (positional argument or stdin, one per line)"
            for sid in sids:
                result = _session_messages_cache_impl(session_id=sid)
                print(json.dumps(result))

        elif args.command == "messages-tail-current":
            sid = args.session_id
            if not sid and not sys.stdin.isatty():
                sid = sys.stdin.readline().strip()
            if sid:
                result = _messages_tail_impl(session_id=sid, count=args.count, include_tools=args.tools)
            else:
                result = _messages_tail_current_impl(count=args.count, include_tools=args.tools)
            print(json.dumps(result, indent=2))

        elif args.command == "messages-tail-last":
            sid = args.session_id
            if not sid and not sys.stdin.isatty():
                sid = sys.stdin.readline().strip()
            if sid:
                result = _messages_tail_impl(session_id=sid, count=args.count, include_tools=args.tools)
            else:
                result = _messages_tail_last_impl(count=args.count, include_tools=args.tools)
            print(json.dumps(result, indent=2))

        elif args.command in ("search", "search-all"):
            pattern = args.pattern
            if not pattern and not sys.stdin.isatty():
                pattern = sys.stdin.readline().strip()
            assert pattern, "search pattern required (positional argument or stdin)"
            all_projects = args.command == "search-all"
            results = _search_impl(pattern, args.count, args.role, all_projects=all_projects)
            for r in results:
                role_label = "HUMAN" if r["role"] == "user" else "ASST"
                print(f"[{r['timestamp']}] [{role_label}] ({r['session_title']})")
                print(f"  {r['text'][:300]}")
                print()
            print(f"{len(results)} matches")

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
# FASTMCP SERVER (lazy — only loaded when mcp-stdio is invoked)
# =============================================================================

def _run_mcp():
    from fastmcp import FastMCP
    mcp = FastMCP("chat_history_oc")

    # MCP for _sessions_impl
    @mcp.tool()
    def sessions(count: int = 20) -> str:
        """List OpenCode sessions for the current project.

        Args:
            count: Maximum number of sessions to return.
        """
        return json.dumps(_sessions_impl(count=count), indent=2)

    # MCP for _session_current_impl
    @mcp.tool()
    def session_current() -> str:
        """Metadata for the current (most recent) OpenCode session.

        Args:
            (no arguments — returns metadata for the most recently updated session)
        """
        return json.dumps(_session_current_impl(), indent=2)

    # MCP for _session_last_impl
    @mcp.tool()
    def session_last() -> str:
        """Metadata for the previous OpenCode session.

        Args:
            (no arguments — returns metadata for the second most recently updated session)
        """
        return json.dumps(_session_last_impl(), indent=2)

    # MCP for _session_messages_cache_impl
    @mcp.tool()
    def session_messages_cache(session_id: str) -> str:
        """Cache a session's messages to .cache/ as a single pre-joined JSON file.

        Assembles all messages and parts from OC database into one file for fast reads.

        Args:
            session_id: The session ID to cache.
        """
        return json.dumps(_session_messages_cache_impl(session_id=session_id), indent=2)

    # MCP for _messages_tail_current_impl
    @mcp.tool()
    def messages_tail_current(count: int = 20, include_tools: bool = False) -> str:
        """Get the message tail of the current (most recent) OpenCode session.

        Use on startup or after compaction to recover conversation context.

        Args:
            count: Number of messages from the end to include.
            include_tools: Whether to include tool call details.
        """
        return json.dumps(_messages_tail_current_impl(count=count, include_tools=include_tools), indent=2)

    # MCP for _messages_tail_last_impl
    @mcp.tool()
    def messages_tail_last(count: int = 20, include_tools: bool = False) -> str:
        """Get the message tail of the previous OpenCode session.

        Use on startup to recover context from the last session.

        Args:
            count: Number of messages from the end to include.
            include_tools: Whether to include tool call details.
        """
        return json.dumps(_messages_tail_last_impl(count=count, include_tools=include_tools), indent=2)

    # MCP for _search_impl
    @mcp.tool()
    def search(pattern: str, count: int = 20, role: str = "", all_projects: bool = False) -> str:
        """Search message text across sessions directly from SQLite.

        Args:
            pattern: Regex pattern to search message text.
            count: Maximum number of matches to return.
            role: Filter by role ("user" or "assistant", empty for both).
            all_projects: Search all projects, not just current directory.
        """
        return json.dumps(_search_impl(pattern, count, role, all_projects), indent=2)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
