#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp>=0.4.0", "watchdog>=3.0.0", "httpx>=0.25.0"]
# ///
"""Sanctum event bus — pub/sub nerve center for agent orchestration.

One daemon, many projects. Pub/sub messaging with project-scoped channels,
JSONL audit with correlation IDs, and extensible trigger engine.

Usage:
    sfs_events.py serve [-s SOCKET] [-b]
    sfs_events.py status [-s SOCKET]
    sfs_events.py stop [-s SOCKET]
    sfs_events.py publish -c CHANNEL -p PAYLOAD [-s SOCKET]
    sfs_events.py subscribe -c CHANNEL [-n COUNT] [-s SOCKET]
    sfs_events.py channels [-s SOCKET]
    sfs_events.py query [-c CHANNEL] [--since TIME] [--correlation-id ID] [-s SOCKET]
    sfs_events.py trigger-add -T TYPE -c CHANNEL [-C CONFIG] [-s SOCKET]
    sfs_events.py trigger-list [-s SOCKET]
    sfs_events.py trigger-remove -I TRIGGER_ID [-s SOCKET]
    sfs_events.py agent-spawn -N NAME [-T TYPE] [-c CHANNELS] [-C CONFIG] [-s SOCKET]
    sfs_events.py agent-list [-s SOCKET]
    sfs_events.py agent-send -A AGENT -p PAYLOAD [-s SOCKET]
    sfs_events.py agent-retire -A AGENT [-s SOCKET]
    sfs_events.py mcp-stdio
"""

import argparse
import asyncio
import json
import os
import signal
import struct
import sys
import time
import uuid
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path

# =============================================================================
# LOGGING (TSV format — SPEC_SFB Principle 6)
# =============================================================================
_LEVELS = {"TRACE": 5, "DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40, "FATAL": 50}
_THRESHOLD = _LEVELS.get(
    os.environ.get("SFB_LOG_LEVEL", os.environ.get("SFA_LOG_LEVEL", "INFO")), 20
)
_LOG_DIR = os.environ.get("SFB_LOG_DIR", os.environ.get("SFA_LOG_DIR", ""))
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
    "status",
    "channels",
    "publish",
    "subscribe",
    "query",
    "stop",
    "trigger-add",
    "trigger-list",
    "trigger-remove",
    "agent-spawn",
    "agent-list",
    "agent-send",
    "agent-retire",
]

SANCTUM_DIR = Path.home() / ".sanctum"
DEFAULT_SOCKET = str(SANCTUM_DIR / "bus.sock")
PID_FILE = SANCTUM_DIR / "bus.pid"
AUDIT_DIR = SANCTUM_DIR / "audit"
AUDIT_FILE = AUDIT_DIR / "events.jsonl"

TRIGGERS_FILE = SANCTUM_DIR / "triggers.jsonl"

CONFIG = {
    "version": "0.3.0",
    "max_message_bytes": 1_048_576,  # 1MB per message
    "max_recent_per_channel": 100,  # recent messages kept in memory
    "audit_flush_interval_seconds": 1.0,
    "default_subscribe_count": 20,
    "url_trigger_default_interval_seconds": 300,
    "file_trigger_debounce_seconds": 1.0,
}

# Reserved project namespace for system-wide channels
_SYSTEM_PROJECT = "_sanctum"


# =============================================================================
# CORE — Project Scoping
# =============================================================================


def _detect_project() -> str:
    """Derive project ID from working directory name."""
    return Path.cwd().name


def _resolve_channel(channel: str, project: str) -> str:
    """Resolve channel to full project-scoped form.

    Channels starting with @ are absolute (bypass auto-scoping):
        @single-file-bench/alerts:*  → single-file-bench/alerts:*
        @*/alerts:*                  → */alerts:*
        @_sanctum/system:health      → _sanctum/system:health

    All other channels get project-prefixed:
        alerts:error  → {project}/alerts:error
    """
    if channel.startswith("@"):
        return channel[1:]
    return f"{project}/{channel}"


def _channel_matches(pattern: str, channel: str) -> bool:
    """Check if a channel matches a subscription pattern (glob-style)."""
    return fnmatch(channel, pattern)


# =============================================================================
# CORE — Message Protocol
# =============================================================================


def _make_message(
    channel: str, payload: dict | str, *, source: str = "", correlation_id: str = ""
) -> dict:
    """Create a bus message with metadata."""
    return {
        "id": uuid.uuid4().hex[:12],
        "channel": channel,
        "payload": payload,
        "source": source,
        "correlation_id": correlation_id or uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    }


def _encode_frame(data: dict) -> bytes:
    """Encode a dict as a length-prefixed JSON frame for the socket protocol."""
    raw = json.dumps(data).encode("utf-8")
    return struct.pack("!I", len(raw)) + raw


async def _read_frame(reader: asyncio.StreamReader) -> dict | None:
    """Read a length-prefixed JSON frame. Returns None on EOF/error."""
    try:
        header = await reader.readexactly(4)
        length = struct.unpack("!I", header)[0]
        assert length <= CONFIG["max_message_bytes"], (
            f"Frame too large: {length} > {CONFIG['max_message_bytes']}"
        )
        raw = await reader.readexactly(length)
        return json.loads(raw.decode("utf-8"))
    except (asyncio.IncompleteReadError, ConnectionError, AssertionError):
        return None


async def _write_frame(writer: asyncio.StreamWriter, data: dict) -> bool:
    """Write a length-prefixed JSON frame. Returns False on failure."""
    try:
        writer.write(_encode_frame(data))
        await writer.drain()
        return True
    except (ConnectionError, OSError):
        return False


# =============================================================================
# CORE — Audit Logger
# =============================================================================

_audit_buffer: list[dict] = []
_audit_lock = None  # initialized in async context


async def _audit_log(message: dict) -> None:
    """Buffer an audit entry. Flushed periodically by the server."""
    _audit_buffer.append(
        {
            "id": message["id"],
            "channel": message["channel"],
            "source": message.get("source", ""),
            "correlation_id": message.get("correlation_id", ""),
            "timestamp": message.get("timestamp", ""),
            "payload_preview": str(message.get("payload", ""))[:200],
        }
    )


async def _audit_flush() -> int:
    """Flush buffered audit entries to JSONL. Returns count flushed."""
    if not _audit_buffer:
        return 0
    entries = _audit_buffer.copy()
    _audit_buffer.clear()
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_FILE, "a") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        return len(entries)
    except Exception as e:
        _log("ERROR", "audit_flush_fail", str(e))
        return 0


def _query_audit_impl(
    channel: str = "", since: str = "", correlation_id: str = "", limit: int = 50
) -> tuple[list[dict], dict]:
    """Search the audit log. CLI: query, MCP: query."""
    start_ms = time.time() * 1000
    results: list[dict] = []

    if not AUDIT_FILE.exists():
        latency_ms = time.time() * 1000 - start_ms
        return [], {"status": "success", "count": 0, "latency_ms": round(latency_ms, 2)}

    # Parse since into a comparable timestamp
    since_ts = ""
    if since:
        try:
            # Support relative times like "1h", "30m", "2d"
            amount = int(since[:-1])
            unit = since[-1]
            seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(unit, 0)
            if seconds:
                cutoff = datetime.now(timezone.utc).timestamp() - (amount * seconds)
                since_ts = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
            else:
                since_ts = since  # Assume ISO format
        except (ValueError, IndexError):
            since_ts = since

    try:
        with open(AUDIT_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Apply filters
                if correlation_id and entry.get("correlation_id") != correlation_id:
                    continue
                if since_ts and entry.get("timestamp", "") < since_ts:
                    continue
                if channel and not _channel_matches(channel, entry.get("channel", "")):
                    continue

                results.append(entry)
    except Exception as e:
        _log("ERROR", "query_read_fail", str(e))

    # Return most recent first, limited
    results = results[-limit:]
    results.reverse()

    latency_ms = time.time() * 1000 - start_ms
    return results, {
        "status": "success",
        "count": len(results),
        "latency_ms": round(latency_ms, 2),
    }


# =============================================================================
# CORE — Trigger Engine
# =============================================================================

# Trigger registry — maps type name to async start function.
# Adding a trigger type: define one async function, add one entry here.
TRIGGER_TYPES: dict[str, str] = {}  # populated after trigger functions are defined

# Active triggers — managed by the server
_triggers: dict[
    str, dict
] = {}  # trigger_id → {type, config, channel, project, status, task}
_trigger_tasks: dict[str, asyncio.Task] = {}  # trigger_id → asyncio.Task


def _save_triggers() -> None:
    """Persist trigger definitions to JSONL (survives restart)."""
    try:
        SANCTUM_DIR.mkdir(parents=True, exist_ok=True)
        with open(TRIGGERS_FILE, "w") as f:
            for tid, t in _triggers.items():
                entry = {k: v for k, v in t.items() if k != "task"}
                entry["id"] = tid
                f.write(json.dumps(entry) + "\n")
    except Exception as e:
        _log("ERROR", "trigger_save_fail", str(e))


def _load_triggers() -> list[dict]:
    """Load trigger definitions from JSONL."""
    if not TRIGGERS_FILE.exists():
        return []
    results = []
    try:
        with open(TRIGGERS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    except Exception as e:
        _log("ERROR", "trigger_load_fail", str(e))
    return results


async def _deliver_message(msg: dict) -> int:
    """Deliver a message to all matching subscribers (socket clients + agents).

    Handles channel stats, recent buffer, socket delivery, agent queue delivery,
    and audit logging. Returns number of deliveries.
    """
    global _message_count
    channel = msg["channel"]

    # Track channel stats
    if channel not in _channels:
        _channels[channel] = {"subscribers": 0, "message_count": 0, "last_activity": ""}
    _channels[channel]["message_count"] += 1
    _channels[channel]["last_activity"] = msg["timestamp"]

    # Store in recent buffer
    recent = _recent.setdefault(channel, [])
    recent.append(msg)
    if len(recent) > CONFIG["max_recent_per_channel"]:
        _recent[channel] = recent[-CONFIG["max_recent_per_channel"] :]

    # Deliver to socket subscribers
    delivered = 0
    for cid, c in _clients.items():
        for pattern in c["subscriptions"]:
            if _channel_matches(pattern, channel):
                await _write_frame(c["writer"], {"event": "message", "message": msg})
                delivered += 1
                break

    # Deliver to agent queues
    for aid, agent in _agents.items():
        if agent.get("status") != "running":
            continue
        for pattern in agent.get("channels", []):
            if _channel_matches(pattern, channel):
                queue = _agent_queues.get(aid)
                if queue:
                    await queue.put(msg)
                    delivered += 1
                break

    # Audit
    await _audit_log(msg)
    _message_count += 1
    return delivered


async def _internal_publish(channel: str, payload: dict, source: str) -> dict:
    """Publish a message from an internal source (trigger/agent). Returns the message."""
    msg = _make_message(channel=channel, payload=payload, source=source)
    await _deliver_message(msg)
    return msg


async def _file_trigger_run(trigger_id: str, config: dict, channel: str) -> None:
    """File trigger — uses watchdog to monitor a path. Publishes changes to channel."""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    path = config.get("path", ".")
    events = set(config.get("events", ["modified", "created", "deleted"]))
    debounce = config.get("debounce_seconds", CONFIG["file_trigger_debounce_seconds"])

    last_event_time: dict[str, float] = {}
    # Capture the running loop BEFORE watchdog starts its thread
    loop = asyncio.get_running_loop()

    class Handler(FileSystemEventHandler):
        def _handle(self, event):
            if event.is_directory:
                return
            etype = event.event_type
            if etype not in events:
                return
            key = f"{etype}:{event.src_path}"
            now = time.time()
            if key in last_event_time and now - last_event_time[key] < debounce:
                return
            last_event_time[key] = now
            # Schedule the publish on the event loop (from watchdog's thread)
            loop.call_soon_threadsafe(
                asyncio.ensure_future,
                _internal_publish(
                    channel,
                    {
                        "trigger_id": trigger_id,
                        "event": etype,
                        "path": str(event.src_path),
                        "timestamp": datetime.now(timezone.utc).isoformat(
                            timespec="milliseconds"
                        ),
                    },
                    source=f"trigger:{trigger_id}",
                ),
            )

        def on_modified(self, event):
            self._handle(event)

        def on_created(self, event):
            self._handle(event)

        def on_deleted(self, event):
            self._handle(event)

        def on_moved(self, event):
            self._handle(event)

    observer = Observer()
    handler = Handler()
    recursive = config.get("recursive", True)

    try:
        observer.schedule(handler, path, recursive=recursive)
        observer.start()
        _log("INFO", "trigger_start", f"file trigger {trigger_id} watching {path}")

        # Keep running until cancelled
        while True:
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        pass
    finally:
        observer.stop()
        observer.join(timeout=5)
        _log("INFO", "trigger_stop", f"file trigger {trigger_id} stopped")


async def _url_trigger_run(trigger_id: str, config: dict, channel: str) -> None:
    """URL trigger — polls a URL at intervals, publishes on content change."""
    import httpx

    url = config.get("url", "")
    interval = config.get(
        "interval_seconds", CONFIG["url_trigger_default_interval_seconds"]
    )
    assert url, "URL trigger requires 'url' in config"

    last_hash = ""

    try:
        _log(
            "INFO",
            "trigger_start",
            f"url trigger {trigger_id} polling {url} every {interval}s",
        )
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            while True:
                try:
                    resp = await client.get(url)
                    import hashlib

                    content_hash = hashlib.sha256(resp.content).hexdigest()[:16]

                    if last_hash and content_hash != last_hash:
                        await _internal_publish(
                            channel,
                            {
                                "trigger_id": trigger_id,
                                "event": "url_changed",
                                "url": url,
                                "status_code": resp.status_code,
                                "content_hash": content_hash,
                                "previous_hash": last_hash,
                                "content_length": len(resp.content),
                                "timestamp": datetime.now(timezone.utc).isoformat(
                                    timespec="milliseconds"
                                ),
                            },
                            source=f"trigger:{trigger_id}",
                        )
                    last_hash = content_hash

                except httpx.HTTPError as e:
                    _log("WARN", "url_trigger_error", f"{trigger_id}: {e}")

                await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass
    finally:
        _log("INFO", "trigger_stop", f"url trigger {trigger_id} stopped")


# Register built-in trigger types
TRIGGER_TYPES["file"] = "file"
TRIGGER_TYPES["url"] = "url"

_TRIGGER_RUNNERS = {
    "file": _file_trigger_run,
    "url": _url_trigger_run,
}


async def _start_trigger(trigger_id: str, trigger_def: dict) -> bool:
    """Start a trigger's async task. Returns True on success."""
    ttype = trigger_def.get("type", "")
    runner = _TRIGGER_RUNNERS.get(ttype)
    if not runner:
        _log("ERROR", "trigger_unknown_type", f"type={ttype}, id={trigger_id}")
        return False

    channel = trigger_def["channel"]
    config = trigger_def.get("config", {})
    task = asyncio.create_task(runner(trigger_id, config, channel))
    _trigger_tasks[trigger_id] = task
    trigger_def["status"] = "running"
    return True


async def _stop_trigger(trigger_id: str) -> bool:
    """Stop a trigger's async task. Returns True if found and stopped."""
    task = _trigger_tasks.pop(trigger_id, None)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    t = _triggers.get(trigger_id)
    if t:
        t["status"] = "stopped"
    return task is not None


async def _restore_triggers() -> int:
    """Restore triggers from persistent storage on server start. Returns count."""
    defs = _load_triggers()
    started = 0
    for d in defs:
        tid = d.get("id", "")
        if not tid or d.get("status") == "removed":
            continue
        _triggers[tid] = d
        if await _start_trigger(tid, d):
            started += 1
    if started:
        _log("INFO", "triggers_restored", f"count={started}")
    return started


# =============================================================================
# CORE — Agent Engine
# =============================================================================

AGENTS_FILE = SANCTUM_DIR / "agents.jsonl"

# Agent type registry — extensible like triggers
AGENT_TYPES: dict[str, str] = {}  # populated after runner functions are defined

# Active agents — managed by the server
_agents: dict[str, dict] = {}  # agent_id → {name, type, channels, config, status, ...}
_agent_tasks: dict[str, asyncio.Task] = {}  # agent_id → asyncio.Task
_agent_queues: dict[str, asyncio.Queue] = {}  # agent_id → message queue


def _save_agents() -> None:
    """Persist agent definitions to JSONL (survives restart)."""
    try:
        SANCTUM_DIR.mkdir(parents=True, exist_ok=True)
        with open(AGENTS_FILE, "w") as f:
            for aid, a in _agents.items():
                entry = {k: v for k, v in a.items() if k not in ("task",)}
                entry["id"] = aid
                f.write(json.dumps(entry) + "\n")
    except Exception as e:
        _log("ERROR", "agent_save_fail", str(e))


def _load_agents() -> list[dict]:
    """Load agent definitions from JSONL."""
    if not AGENTS_FILE.exists():
        return []
    results = []
    try:
        with open(AGENTS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    except Exception as e:
        _log("ERROR", "agent_load_fail", str(e))
    return results


async def _subprocess_agent_run(
    agent_id: str, agent_def: dict, queue: asyncio.Queue
) -> None:
    """Subprocess agent — runs a command for each incoming message.

    The message JSON is piped to stdin. If publish_to is set and the command
    produces stdout, the output is published to that channel.
    """
    import shlex

    command = agent_def["config"].get("command", "")
    publish_to = agent_def["config"].get("publish_to", "")
    project = agent_def.get("project", _SYSTEM_PROJECT)
    timeout_s = agent_def["config"].get("timeout_seconds", 60)

    assert command, "subprocess agent requires 'command' in config"
    _log("INFO", "agent_start", f"subprocess agent {agent_id} cmd={command}")

    try:
        while True:
            msg = await queue.get()
            try:
                proc = await asyncio.create_subprocess_exec(
                    *shlex.split(command),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(json.dumps(msg).encode("utf-8")),
                    timeout=timeout_s,
                )

                agent_def["messages_processed"] = (
                    agent_def.get("messages_processed", 0) + 1
                )

                if stderr.strip():
                    _log(
                        "WARN",
                        "agent_stderr",
                        f"agent={agent_id}: {stderr.decode()[:200]}",
                    )

                # Publish stdout as response if configured
                if publish_to and stdout.strip():
                    out_channel = _resolve_channel(publish_to, project)
                    try:
                        result_payload = json.loads(stdout)
                    except json.JSONDecodeError:
                        result_payload = {"output": stdout.decode("utf-8").strip()}
                    await _internal_publish(
                        out_channel, result_payload, source=f"agent:{agent_id}"
                    )

            except asyncio.TimeoutError:
                _log("WARN", "agent_timeout", f"agent={agent_id} timeout={timeout_s}s")
            except Exception as e:
                _log("ERROR", "agent_process_error", f"agent={agent_id}: {e}")
    except asyncio.CancelledError:
        pass
    finally:
        _log("INFO", "agent_stop", f"subprocess agent {agent_id} stopped")


# Register built-in agent types
AGENT_TYPES["subprocess"] = "subprocess"

_AGENT_RUNNERS = {
    "subprocess": _subprocess_agent_run,
}


async def _start_agent(agent_id: str, agent_def: dict) -> bool:
    """Start an agent's async task. Returns True on success."""
    atype = agent_def.get("type", "")
    runner = _AGENT_RUNNERS.get(atype)
    if not runner:
        _log("ERROR", "agent_unknown_type", f"type={atype}, id={agent_id}")
        return False

    queue: asyncio.Queue = asyncio.Queue()
    _agent_queues[agent_id] = queue

    task = asyncio.create_task(runner(agent_id, agent_def, queue))
    _agent_tasks[agent_id] = task
    agent_def["status"] = "running"
    return True


async def _stop_agent(agent_id: str) -> bool:
    """Stop an agent's async task. Returns True if found and stopped."""
    task = _agent_tasks.pop(agent_id, None)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    _agent_queues.pop(agent_id, None)
    a = _agents.get(agent_id)
    if a:
        a["status"] = "retired"
    return task is not None


async def _restore_agents() -> int:
    """Restore agents from persistent storage on server start. Returns count."""
    defs = _load_agents()
    started = 0
    for d in defs:
        aid = d.get("id", "")
        if not aid or d.get("status") == "retired":
            continue
        d["messages_processed"] = 0  # reset counter on restart
        _agents[aid] = d
        if await _start_agent(aid, d):
            started += 1
    if started:
        _log("INFO", "agents_restored", f"count={started}")
    return started


# =============================================================================
# CORE — Bus Server
# =============================================================================

# Server state — module-level, managed by the serve loop
_clients: dict[str, dict] = {}  # client_id → {writer, project, subscriptions}
_channels: dict[
    str, dict
] = {}  # full_channel → {subscribers, message_count, last_activity}
_recent: dict[str, list[dict]] = {}  # full_channel → recent messages
_server_start_time: float = 0.0
_message_count: int = 0
_running: bool = False


async def _handle_client(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Handle a single client connection."""
    client_id = uuid.uuid4().hex[:8]
    _clients[client_id] = {
        "writer": writer,
        "project": "",
        "subscriptions": set(),
        "connected_at": time.time(),
    }
    _log("DEBUG", "client_connect", f"client={client_id}")

    try:
        while _running:
            frame = await _read_frame(reader)
            if frame is None:
                break
            await _process_request(client_id, frame)
    except Exception as e:
        _log("WARN", "client_error", f"client={client_id}: {e}")
    finally:
        # Clean up subscriptions
        client = _clients.pop(client_id, None)
        if client:
            for pattern in client["subscriptions"]:
                ch = _channels.get(pattern)
                if ch:
                    ch["subscribers"] = max(0, ch["subscribers"] - 1)
        writer.close()
        _log("DEBUG", "client_disconnect", f"client={client_id}")


async def _process_request(client_id: str, frame: dict) -> None:
    """Process a client request frame and send response."""
    global _message_count
    client = _clients.get(client_id)
    if not client:
        return

    cmd = frame.get("cmd", "")
    writer = client["writer"]

    if cmd == "identify":
        client["project"] = frame.get("project", _detect_project())
        await _write_frame(writer, {"ok": True, "client_id": client_id})

    elif cmd == "publish":
        project = client["project"] or frame.get("project", _SYSTEM_PROJECT)
        channel = _resolve_channel(frame.get("channel", ""), project)
        msg = _make_message(
            channel=channel,
            payload=frame.get("payload", {}),
            source=frame.get("source", f"client:{client_id}"),
            correlation_id=frame.get("correlation_id", ""),
        )

        delivered = await _deliver_message(msg)

        await _write_frame(
            writer,
            {
                "ok": True,
                "message_id": msg["id"],
                "channel": channel,
                "delivered": delivered,
            },
        )

    elif cmd == "subscribe":
        project = client["project"] or frame.get("project", _SYSTEM_PROJECT)
        channel = _resolve_channel(frame.get("channel", ""), project)
        client["subscriptions"].add(channel)

        if channel not in _channels:
            _channels[channel] = {
                "subscribers": 0,
                "message_count": 0,
                "last_activity": "",
            }
        _channels[channel]["subscribers"] += 1

        # Return recent messages matching this pattern
        count = frame.get("count", CONFIG["default_subscribe_count"])
        recent_msgs: list[dict] = []
        for ch, msgs in _recent.items():
            if _channel_matches(channel, ch):
                recent_msgs.extend(msgs)
        recent_msgs.sort(key=lambda m: m.get("timestamp", ""))
        recent_msgs = recent_msgs[-count:]

        await _write_frame(
            writer,
            {
                "ok": True,
                "channel": channel,
                "recent": recent_msgs,
            },
        )

    elif cmd == "channels":
        await _write_frame(writer, {"ok": True, "channels": _channels})

    elif cmd == "status":
        uptime = time.time() - _server_start_time if _server_start_time else 0
        await _write_frame(
            writer,
            {
                "ok": True,
                "uptime_seconds": round(uptime, 1),
                "clients": len(_clients),
                "channels": len(_channels),
                "messages_total": _message_count,
                "audit_buffer": len(_audit_buffer),
                "socket": DEFAULT_SOCKET,
                "pid": os.getpid(),
                "version": CONFIG["version"],
            },
        )

    elif cmd == "stop":
        await _write_frame(writer, {"ok": True, "message": "shutting down"})
        _log("INFO", "stop_requested", f"by client={client_id}")
        # Signal the server to stop
        global _running
        _running = False

    elif cmd == "trigger-add":
        project = client["project"] or frame.get("project", _SYSTEM_PROJECT)
        ttype = frame.get("type", "")
        assert ttype in _TRIGGER_RUNNERS, f"unknown trigger type: {ttype}"
        channel = _resolve_channel(frame.get("channel", ""), project)
        config = frame.get("config", {})
        tid = uuid.uuid4().hex[:12]
        trigger_def = {
            "type": ttype,
            "channel": channel,
            "project": project,
            "config": config,
            "status": "pending",
        }
        _triggers[tid] = trigger_def
        ok = await _start_trigger(tid, trigger_def)
        if ok:
            _save_triggers()
            await _write_frame(
                writer,
                {"ok": True, "trigger_id": tid, "type": ttype, "channel": channel},
            )
        else:
            del _triggers[tid]
            await _write_frame(writer, {"ok": False, "error": f"failed to start trigger"})

    elif cmd == "trigger-list":
        items = []
        for tid, t in _triggers.items():
            items.append({
                "id": tid,
                "type": t.get("type", ""),
                "channel": t.get("channel", ""),
                "project": t.get("project", ""),
                "config": t.get("config", {}),
                "status": t.get("status", "unknown"),
            })
        await _write_frame(writer, {"ok": True, "triggers": items})

    elif cmd == "trigger-remove":
        tid = frame.get("trigger_id", "")
        assert tid, "trigger_id required"
        if tid not in _triggers:
            await _write_frame(writer, {"ok": False, "error": f"trigger not found: {tid}"})
        else:
            await _stop_trigger(tid)
            _triggers[tid]["status"] = "removed"
            _save_triggers()
            del _triggers[tid]
            await _write_frame(writer, {"ok": True, "trigger_id": tid, "removed": True})

    elif cmd == "agent-spawn":
        project = client["project"] or frame.get("project", _SYSTEM_PROJECT)
        atype = frame.get("type", "subprocess")
        assert atype in _AGENT_RUNNERS, f"unknown agent type: {atype}"
        name = frame.get("name", "")
        assert name, "agent name required"
        # Check name uniqueness among running agents
        for existing in _agents.values():
            if existing.get("name") == name and existing.get("status") == "running":
                await _write_frame(
                    writer, {"ok": False, "error": f"agent '{name}' already running"}
                )
                return
        channels_raw = frame.get("channels", [])
        # Resolve all channels with project scoping
        resolved_channels = [_resolve_channel(c, project) for c in channels_raw]
        # Auto-add inbox channel
        aid = uuid.uuid4().hex[:12]
        inbox = f"_sanctum/_agent/{aid}/inbox"
        resolved_channels.append(inbox)
        config = frame.get("config", {})
        agent_def = {
            "name": name,
            "type": atype,
            "channels": resolved_channels,
            "project": project,
            "config": config,
            "status": "pending",
            "messages_processed": 0,
            "spawned_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "inbox": inbox,
        }
        _agents[aid] = agent_def
        ok = await _start_agent(aid, agent_def)
        if ok:
            _save_agents()
            await _write_frame(
                writer,
                {
                    "ok": True,
                    "agent_id": aid,
                    "name": name,
                    "type": atype,
                    "channels": resolved_channels,
                    "inbox": inbox,
                },
            )
        else:
            del _agents[aid]
            await _write_frame(
                writer, {"ok": False, "error": "failed to start agent"}
            )

    elif cmd == "agent-list":
        items = []
        for aid, a in _agents.items():
            items.append({
                "id": aid,
                "name": a.get("name", ""),
                "type": a.get("type", ""),
                "channels": a.get("channels", []),
                "project": a.get("project", ""),
                "status": a.get("status", "unknown"),
                "messages_processed": a.get("messages_processed", 0),
                "spawned_at": a.get("spawned_at", ""),
                "inbox": a.get("inbox", ""),
            })
        await _write_frame(writer, {"ok": True, "agents": items})

    elif cmd == "agent-send":
        # Direct message to an agent by name or ID
        target = frame.get("agent", "")
        assert target, "agent name or ID required"
        payload = frame.get("payload", {})
        # Resolve agent — try by name first, then by ID
        target_id = ""
        target_inbox = ""
        for aid, a in _agents.items():
            if a.get("name") == target or aid == target:
                if a.get("status") == "running":
                    target_id = aid
                    target_inbox = a.get("inbox", "")
                    break
        if not target_id:
            await _write_frame(
                writer, {"ok": False, "error": f"agent not found or not running: {target}"}
            )
        else:
            source = frame.get("source", f"client:{client_id}")
            msg = _make_message(
                channel=target_inbox,
                payload=payload,
                source=source,
                correlation_id=frame.get("correlation_id", ""),
            )
            delivered = await _deliver_message(msg)
            await _write_frame(
                writer,
                {
                    "ok": True,
                    "agent_id": target_id,
                    "inbox": target_inbox,
                    "message_id": msg["id"],
                    "delivered": delivered,
                },
            )

    elif cmd == "agent-retire":
        target = frame.get("agent", "")
        assert target, "agent name or ID required"
        # Resolve agent
        target_id = ""
        for aid, a in _agents.items():
            if a.get("name") == target or aid == target:
                target_id = aid
                break
        if not target_id:
            await _write_frame(
                writer, {"ok": False, "error": f"agent not found: {target}"}
            )
        else:
            await _stop_agent(target_id)
            _save_agents()
            await _write_frame(
                writer, {"ok": True, "agent_id": target_id, "retired": True}
            )

    elif cmd == "ping":
        await _write_frame(writer, {"ok": True, "pong": True})

    else:
        await _write_frame(writer, {"ok": False, "error": f"unknown command: {cmd}"})


async def _audit_flush_loop() -> None:
    """Periodically flush audit buffer to disk."""
    while _running:
        await asyncio.sleep(CONFIG["audit_flush_interval_seconds"])
        flushed = await _audit_flush()
        if flushed:
            _log("TRACE", "audit_flushed", f"entries={flushed}")


async def _serve_impl(
    socket_path: str = DEFAULT_SOCKET, background: bool = False
) -> None:
    """Start the bus server. CLI: serve (not in EXPOSED — it's an entry point)."""
    global _server_start_time, _running

    socket_p = Path(socket_path)
    socket_p.parent.mkdir(parents=True, exist_ok=True)

    # Clean up stale socket
    if socket_p.exists():
        try:
            socket_p.unlink()
        except OSError:
            pass

    _running = True
    _server_start_time = time.time()

    # Write PID file
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    server = await asyncio.start_unix_server(_handle_client, path=socket_path)
    _log("INFO", "server_start", f"socket={socket_path},pid={os.getpid()}")
    print(f"Bus server listening on {socket_path} (pid {os.getpid()})", file=sys.stderr)

    # Set up signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: _stop_server())

    # Restore persisted triggers and agents
    t_restored = await _restore_triggers()
    if t_restored:
        print(f"Restored {t_restored} trigger(s) from previous session", file=sys.stderr)
    a_restored = await _restore_agents()
    if a_restored:
        print(f"Restored {a_restored} agent(s) from previous session", file=sys.stderr)

    # Run audit flush loop alongside server
    flush_task = asyncio.create_task(_audit_flush_loop())

    try:
        while _running:
            await asyncio.sleep(0.1)
    finally:
        # Stop all triggers and agents
        for tid in list(_trigger_tasks):
            await _stop_trigger(tid)
        for aid in list(_agent_tasks):
            await _stop_agent(aid)

        # Cleanup
        _running = False
        flush_task.cancel()
        await _audit_flush()  # Final flush
        server.close()
        await server.wait_closed()
        if socket_p.exists():
            socket_p.unlink()
        if PID_FILE.exists():
            PID_FILE.unlink()
        _log(
            "INFO",
            "server_stop",
            f"uptime={round(time.time() - _server_start_time, 1)}s",
        )
        print("Bus server stopped.", file=sys.stderr)


def _stop_server() -> None:
    """Signal the server loop to stop."""
    global _running
    _running = False


# =============================================================================
# CORE — Bus Client (used by CLI and MCP _impl functions)
# =============================================================================


def _connect_sync(socket_path: str = DEFAULT_SOCKET) -> tuple | None:
    """Connect to the bus synchronously. Returns (reader, writer) or None."""
    import socket as sock

    try:
        s = sock.socket(sock.AF_UNIX, sock.SOCK_STREAM)
        s.connect(socket_path)
        s.settimeout(10.0)
        return s
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        return None


def _send_recv_sync(s, request: dict) -> dict:
    """Send a request and receive response over a sync socket."""
    raw = json.dumps(request).encode("utf-8")
    s.sendall(struct.pack("!I", len(raw)) + raw)

    # Read response
    header = b""
    while len(header) < 4:
        chunk = s.recv(4 - len(header))
        assert chunk, "Connection closed"
        header += chunk
    length = struct.unpack("!I", header)[0]

    data = b""
    while len(data) < length:
        chunk = s.recv(min(length - len(data), 65536))
        assert chunk, "Connection closed"
        data += chunk
    return json.loads(data.decode("utf-8"))


def _autostart_daemon(socket_path: str = DEFAULT_SOCKET) -> bool:
    """Start the bus daemon if not running. One attempt, short wait."""
    import subprocess as sp
    script = str(Path(__file__).resolve())
    proc = sp.Popen(
        ["uv", "run", "--script", script, "serve", "--background", "-s", socket_path],
        stdout=sp.DEVNULL, stderr=sp.PIPE
    )
    proc.wait(timeout=10)
    if proc.returncode != 0:
        _log("ERROR", "autostart", f"Daemon failed to start (exit={proc.returncode})")
        return False
    # Wait for socket to appear — daemon forks, parent exits, child binds
    for _ in range(10):
        time.sleep(0.5)
        if _connect_sync(socket_path) is not None:
            _log("INFO", "autostart", "Bus daemon auto-started successfully")
            return True
    _log("ERROR", "autostart", "Daemon started but socket not reachable")
    return False


def _bus_request(request: dict, socket_path: str = DEFAULT_SOCKET) -> dict:
    """Send a request to the bus and return the response. Auto-starts daemon if needed."""
    s = _connect_sync(socket_path)
    if s is None:
        _autostart_daemon(socket_path)
        s = _connect_sync(socket_path)
    assert s is not None, (
        f"Cannot connect to bus at {socket_path}. Auto-start failed."
    )
    try:
        # Identify with project
        _send_recv_sync(s, {"cmd": "identify", "project": _detect_project()})
        return _send_recv_sync(s, request)
    finally:
        s.close()


# =============================================================================
# CORE FUNCTIONS (the actual logic — CLI and MCP both call these)
# =============================================================================


def _status_impl(socket_path: str = DEFAULT_SOCKET) -> tuple[dict, dict]:
    """Get bus server status. CLI: status, MCP: status."""
    start_ms = time.time() * 1000
    result = _bus_request({"cmd": "status"}, socket_path)
    latency_ms = time.time() * 1000 - start_ms
    return result, {"status": "success", "latency_ms": round(latency_ms, 2)}


def _channels_impl(socket_path: str = DEFAULT_SOCKET) -> tuple[dict, dict]:
    """List active channels with stats. CLI: channels, MCP: channels."""
    start_ms = time.time() * 1000
    result = _bus_request({"cmd": "channels"}, socket_path)
    latency_ms = time.time() * 1000 - start_ms
    return result, {"status": "success", "latency_ms": round(latency_ms, 2)}


def _publish_impl(
    channel: str,
    payload: str,
    *,
    source: str = "",
    correlation_id: str = "",
    socket_path: str = DEFAULT_SOCKET,
) -> tuple[dict, dict]:
    """Publish a message to a channel. CLI: publish, MCP: publish."""
    start_ms = time.time() * 1000

    # Parse payload as JSON if possible, otherwise wrap as string
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        parsed = payload

    result = _bus_request(
        {
            "cmd": "publish",
            "channel": channel,
            "payload": parsed,
            "source": source,
            "correlation_id": correlation_id,
        },
        socket_path,
    )
    latency_ms = time.time() * 1000 - start_ms
    _log(
        "INFO",
        "publish",
        f"channel={channel}",
        metrics=f"latency_ms={round(latency_ms, 2)}",
    )
    return result, {"status": "success", "latency_ms": round(latency_ms, 2)}


def _subscribe_impl(
    channel: str, count: int = 20, socket_path: str = DEFAULT_SOCKET
) -> tuple[dict, dict]:
    """Read recent messages from a channel pattern. CLI: subscribe, MCP: subscribe."""
    start_ms = time.time() * 1000
    result = _bus_request(
        {
            "cmd": "subscribe",
            "channel": channel,
            "count": count,
        },
        socket_path,
    )
    latency_ms = time.time() * 1000 - start_ms
    return result, {"status": "success", "latency_ms": round(latency_ms, 2)}


def _stop_impl(socket_path: str = DEFAULT_SOCKET) -> tuple[dict, dict]:
    """Gracefully stop the bus daemon. CLI: stop, MCP: stop."""
    start_ms = time.time() * 1000
    result = _bus_request({"cmd": "stop"}, socket_path)
    latency_ms = time.time() * 1000 - start_ms
    _log("INFO", "stop_sent", "stop command sent to daemon")
    return result, {"status": "success", "latency_ms": round(latency_ms, 2)}


def _trigger_add_impl(
    trigger_type: str,
    channel: str,
    config: str = "{}",
    *,
    socket_path: str = DEFAULT_SOCKET,
) -> tuple[dict, dict]:
    """Add a trigger to the bus. CLI: trigger-add, MCP: trigger_add.

    Args:
        trigger_type: Trigger type — "file" or "url"
        channel: Channel to publish events to (auto-scoped to project)
        config: JSON config string — file: {path, recursive, events, debounce_seconds}
                                    — url: {url, interval_seconds}
    """
    start_ms = time.time() * 1000
    try:
        parsed_config = json.loads(config)
    except (json.JSONDecodeError, TypeError):
        parsed_config = {}

    result = _bus_request(
        {
            "cmd": "trigger-add",
            "type": trigger_type,
            "channel": channel,
            "config": parsed_config,
        },
        socket_path,
    )
    latency_ms = time.time() * 1000 - start_ms
    _log(
        "INFO",
        "trigger_add",
        f"type={trigger_type},channel={channel}",
        metrics=f"latency_ms={round(latency_ms, 2)}",
    )
    return result, {"status": "success", "latency_ms": round(latency_ms, 2)}


def _trigger_list_impl(
    socket_path: str = DEFAULT_SOCKET,
) -> tuple[dict, dict]:
    """List active triggers. CLI: trigger-list, MCP: trigger_list."""
    start_ms = time.time() * 1000
    result = _bus_request({"cmd": "trigger-list"}, socket_path)
    latency_ms = time.time() * 1000 - start_ms
    return result, {"status": "success", "latency_ms": round(latency_ms, 2)}


def _trigger_remove_impl(
    trigger_id: str,
    *,
    socket_path: str = DEFAULT_SOCKET,
) -> tuple[dict, dict]:
    """Remove a trigger by ID. CLI: trigger-remove, MCP: trigger_remove."""
    start_ms = time.time() * 1000
    result = _bus_request(
        {"cmd": "trigger-remove", "trigger_id": trigger_id},
        socket_path,
    )
    latency_ms = time.time() * 1000 - start_ms
    _log(
        "INFO",
        "trigger_remove",
        f"trigger_id={trigger_id}",
        metrics=f"latency_ms={round(latency_ms, 2)}",
    )
    return result, {"status": "success", "latency_ms": round(latency_ms, 2)}


def _agent_spawn_impl(
    name: str,
    agent_type: str = "subprocess",
    channels: str = "",
    config: str = "{}",
    *,
    socket_path: str = DEFAULT_SOCKET,
) -> tuple[dict, dict]:
    """Spawn a new agent. CLI: agent-spawn, MCP: agent_spawn.

    Args:
        name: Agent name (unique among running agents)
        agent_type: Agent runtime — "subprocess"
        channels: Comma-separated channel patterns to subscribe to
        config: JSON config — subprocess: {command, publish_to, timeout_seconds}
    """
    start_ms = time.time() * 1000
    try:
        parsed_config = json.loads(config)
    except (json.JSONDecodeError, TypeError):
        parsed_config = {}

    channel_list = [c.strip() for c in channels.split(",") if c.strip()] if channels else []

    result = _bus_request(
        {
            "cmd": "agent-spawn",
            "name": name,
            "type": agent_type,
            "channels": channel_list,
            "config": parsed_config,
        },
        socket_path,
    )
    latency_ms = time.time() * 1000 - start_ms
    _log(
        "INFO",
        "agent_spawn",
        f"name={name},type={agent_type}",
        metrics=f"latency_ms={round(latency_ms, 2)}",
    )
    return result, {"status": "success", "latency_ms": round(latency_ms, 2)}


def _agent_list_impl(
    socket_path: str = DEFAULT_SOCKET,
) -> tuple[dict, dict]:
    """List active agents. CLI: agent-list, MCP: agent_list."""
    start_ms = time.time() * 1000
    result = _bus_request({"cmd": "agent-list"}, socket_path)
    latency_ms = time.time() * 1000 - start_ms
    return result, {"status": "success", "latency_ms": round(latency_ms, 2)}


def _agent_send_impl(
    agent: str,
    payload: str,
    *,
    source: str = "",
    correlation_id: str = "",
    socket_path: str = DEFAULT_SOCKET,
) -> tuple[dict, dict]:
    """Send a direct message to a named agent. CLI: agent-send, MCP: agent_send."""
    start_ms = time.time() * 1000
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        parsed = payload

    result = _bus_request(
        {
            "cmd": "agent-send",
            "agent": agent,
            "payload": parsed,
            "source": source,
            "correlation_id": correlation_id,
        },
        socket_path,
    )
    latency_ms = time.time() * 1000 - start_ms
    _log(
        "INFO",
        "agent_send",
        f"agent={agent}",
        metrics=f"latency_ms={round(latency_ms, 2)}",
    )
    return result, {"status": "success", "latency_ms": round(latency_ms, 2)}


def _agent_retire_impl(
    agent: str,
    *,
    socket_path: str = DEFAULT_SOCKET,
) -> tuple[dict, dict]:
    """Retire (stop) an agent by name or ID. CLI: agent-retire, MCP: agent_retire."""
    start_ms = time.time() * 1000
    result = _bus_request(
        {"cmd": "agent-retire", "agent": agent},
        socket_path,
    )
    latency_ms = time.time() * 1000 - start_ms
    _log(
        "INFO",
        "agent_retire",
        f"agent={agent}",
        metrics=f"latency_ms={round(latency_ms, 2)}",
    )
    return result, {"status": "success", "latency_ms": round(latency_ms, 2)}


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Sanctum event bus — pub/sub nerve center",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sfs_events.py serve                          Start bus daemon
  sfs_events.py serve -b                       Start in background
  sfs_events.py status                         Bus health check
  sfs_events.py publish -c alerts:error -p '{"msg":"disk full"}'
  sfs_events.py subscribe -c "alerts:*"        Recent messages on alerts
  sfs_events.py channels                       Active channels
  sfs_events.py query --since 1h               Last hour of audit log
  sfs_events.py query -i abc123                Follow correlation ID
  sfs_events.py trigger-add -T file -c watch:changes -C '{"path":"./src"}'
  sfs_events.py trigger-add -T url -c watch:releases -C '{"url":"https://example.com/releases","interval_seconds":60}'
  sfs_events.py trigger-list                   Show active triggers
  sfs_events.py trigger-remove -I abc123       Remove a trigger
  sfs_events.py agent-spawn -N triage -c "alerts:*" -C '{"command":"python3 handler.py"}'
  sfs_events.py agent-list                     Show active agents
  sfs_events.py agent-send -A triage -p '{"task":"investigate"}'
  sfs_events.py agent-retire -A triage         Stop an agent
  sfs_events.py stop                           Graceful shutdown
        """,
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {CONFIG['version']}"
    )

    sub = parser.add_subparsers(dest="command", help="Commands")

    # serve
    p_serve = sub.add_parser("serve", help="Start bus daemon")
    p_serve.add_argument(
        "-s", "--socket", default=DEFAULT_SOCKET, help="Unix socket path"
    )
    p_serve.add_argument("-b", "--background", action="store_true", help="Daemonize")

    # status
    p_status = sub.add_parser("status", help="Bus health check")
    p_status.add_argument("-s", "--socket", default=DEFAULT_SOCKET)

    # stop
    p_stop = sub.add_parser("stop", help="Graceful shutdown")
    p_stop.add_argument("-s", "--socket", default=DEFAULT_SOCKET)

    # publish
    p_pub = sub.add_parser("publish", help="Publish message to channel")
    p_pub.add_argument("-c", "--channel", required=True, help="Target channel")
    p_pub.add_argument("-p", "--payload", required=True, help="JSON payload")
    p_pub.add_argument("-r", "--source", default="", help="Source identifier")
    p_pub.add_argument("-i", "--correlation-id", default="", help="Correlation ID")
    p_pub.add_argument("-s", "--socket", default=DEFAULT_SOCKET)

    # subscribe
    p_sub = sub.add_parser("subscribe", help="Read recent messages from channel")
    p_sub.add_argument("-c", "--channel", required=True, help="Channel pattern (glob)")
    p_sub.add_argument("-n", "--count", type=int, default=20, help="Max messages")
    p_sub.add_argument("-s", "--socket", default=DEFAULT_SOCKET)

    # channels
    p_ch = sub.add_parser("channels", help="List active channels")
    p_ch.add_argument("-s", "--socket", default=DEFAULT_SOCKET)

    # query
    p_query = sub.add_parser("query", help="Search audit log")
    p_query.add_argument("-c", "--channel", default="", help="Channel pattern")
    p_query.add_argument(
        "-t", "--since", default="", help="Time range (e.g. 1h, 30m, 2d)"
    )
    p_query.add_argument("-i", "--correlation-id", default="", help="Correlation ID")
    p_query.add_argument("-n", "--limit", type=int, default=50, help="Max results")
    p_query.add_argument("-s", "--socket", default=DEFAULT_SOCKET)

    # trigger-add
    p_tadd = sub.add_parser("trigger-add", help="Add a file or URL trigger")
    p_tadd.add_argument(
        "-T", "--type", required=True, choices=list(_TRIGGER_RUNNERS.keys()),
        help="Trigger type",
    )
    p_tadd.add_argument("-c", "--channel", required=True, help="Target channel")
    p_tadd.add_argument(
        "-C", "--config", default="{}", help="JSON config (file: {path}, url: {url})",
    )
    p_tadd.add_argument("-s", "--socket", default=DEFAULT_SOCKET)

    # trigger-list
    p_tlist = sub.add_parser("trigger-list", help="List active triggers")
    p_tlist.add_argument("-s", "--socket", default=DEFAULT_SOCKET)

    # trigger-remove
    p_trem = sub.add_parser("trigger-remove", help="Remove a trigger")
    p_trem.add_argument(
        "-I", "--trigger-id", required=True, help="Trigger ID to remove",
    )
    p_trem.add_argument("-s", "--socket", default=DEFAULT_SOCKET)

    # agent-spawn
    p_aspawn = sub.add_parser("agent-spawn", help="Spawn a new agent")
    p_aspawn.add_argument("-N", "--name", required=True, help="Agent name")
    p_aspawn.add_argument(
        "-T", "--type", default="subprocess", choices=list(_AGENT_RUNNERS.keys()),
        help="Agent runtime type",
    )
    p_aspawn.add_argument(
        "-c", "--channels", default="", help="Comma-separated channel patterns",
    )
    p_aspawn.add_argument(
        "-C", "--config", default="{}", help='JSON config (e.g. \'{"command":"..."}\')',
    )
    p_aspawn.add_argument("-s", "--socket", default=DEFAULT_SOCKET)

    # agent-list
    p_alist = sub.add_parser("agent-list", help="List active agents")
    p_alist.add_argument("-s", "--socket", default=DEFAULT_SOCKET)

    # agent-send
    p_asend = sub.add_parser("agent-send", help="Send message to agent")
    p_asend.add_argument("-A", "--agent", required=True, help="Agent name or ID")
    p_asend.add_argument("-p", "--payload", required=True, help="JSON payload")
    p_asend.add_argument("-r", "--source", default="", help="Source identifier")
    p_asend.add_argument("-i", "--correlation-id", default="", help="Correlation ID")
    p_asend.add_argument("-s", "--socket", default=DEFAULT_SOCKET)

    # agent-retire
    p_aret = sub.add_parser("agent-retire", help="Retire (stop) an agent")
    p_aret.add_argument("-A", "--agent", required=True, help="Agent name or ID")
    p_aret.add_argument("-s", "--socket", default=DEFAULT_SOCKET)

    # mcp-stdio
    sub.add_parser("mcp-stdio", help="Run as MCP server")

    args = parser.parse_args()

    # stdin support — pipe JSON to publish: echo '{"channel":"x","payload":{}}' | sfs_events.py
    if not args.command and not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        if raw:
            try:
                data = json.loads(raw)
                assert "channel" in data, "stdin JSON must have 'channel' field"
                result, _ = _publish_impl(
                    channel=data["channel"],
                    payload=json.dumps(data.get("payload", {})),
                    source=data.get("source", "stdin"),
                    correlation_id=data.get("correlation_id", ""),
                )
                print(json.dumps(result, indent=2))
                sys.exit(0)
            except (json.JSONDecodeError, AssertionError) as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

    try:
        if args.command == "mcp-stdio":
            _run_mcp()

        elif args.command == "serve":
            if args.background:
                _daemonize(args.socket)
            else:
                asyncio.run(_serve_impl(args.socket))

        elif args.command == "status":
            result, _ = _status_impl(args.socket)
            print(json.dumps(result, indent=2))

        elif args.command == "stop":
            result, _ = _stop_impl(args.socket)
            print(json.dumps(result, indent=2))

        elif args.command == "publish":
            result, _ = _publish_impl(
                channel=args.channel,
                payload=args.payload,
                source=args.source,
                correlation_id=args.correlation_id,
                socket_path=args.socket,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "subscribe":
            result, _ = _subscribe_impl(
                channel=args.channel,
                count=args.count,
                socket_path=args.socket,
            )
            # Print recent messages, one per line
            for msg in result.get("recent", []):
                print(json.dumps(msg))

        elif args.command == "channels":
            result, _ = _channels_impl(args.socket)
            channels = result.get("channels", {})
            if channels:
                for ch, stats in channels.items():
                    print(
                        f"{ch}  subs={stats.get('subscribers', 0)}  "
                        f"msgs={stats.get('message_count', 0)}  "
                        f"last={stats.get('last_activity', 'never')}"
                    )
            else:
                print("No active channels.")

        elif args.command == "query":
            # Query reads the audit log directly (no bus connection needed)
            channel = args.channel
            if channel and not channel.startswith("@"):
                channel = _resolve_channel(channel, _detect_project())
            elif channel and channel.startswith("@"):
                channel = channel[1:]
            results, _ = _query_audit_impl(
                channel=channel,
                since=args.since,
                correlation_id=args.correlation_id,
                limit=args.limit,
            )
            for entry in results:
                print(json.dumps(entry))

        elif args.command == "trigger-add":
            result, _ = _trigger_add_impl(
                trigger_type=args.type,
                channel=args.channel,
                config=args.config,
                socket_path=args.socket,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "trigger-list":
            result, _ = _trigger_list_impl(socket_path=args.socket)
            triggers = result.get("triggers", [])
            if triggers:
                for t in triggers:
                    cfg_str = json.dumps(t.get("config", {}))
                    print(
                        f"{t['id']}  type={t['type']}  channel={t['channel']}  "
                        f"status={t['status']}  config={cfg_str}"
                    )
            else:
                print("No active triggers.")

        elif args.command == "trigger-remove":
            result, _ = _trigger_remove_impl(
                trigger_id=args.trigger_id,
                socket_path=args.socket,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "agent-spawn":
            result, _ = _agent_spawn_impl(
                name=args.name,
                agent_type=args.type,
                channels=args.channels,
                config=args.config,
                socket_path=args.socket,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "agent-list":
            result, _ = _agent_list_impl(socket_path=args.socket)
            agents = result.get("agents", [])
            if agents:
                for a in agents:
                    ch_str = ",".join(
                        c for c in a.get("channels", [])
                        if not c.startswith("_sanctum/_agent/")
                    )
                    print(
                        f"{a['id']}  name={a['name']}  type={a['type']}  "
                        f"status={a['status']}  msgs={a.get('messages_processed', 0)}  "
                        f"channels={ch_str or '(inbox only)'}"
                    )
            else:
                print("No active agents.")

        elif args.command == "agent-send":
            result, _ = _agent_send_impl(
                agent=args.agent,
                payload=args.payload,
                source=args.source,
                correlation_id=args.correlation_id,
                socket_path=args.socket,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "agent-retire":
            result, _ = _agent_retire_impl(
                agent=args.agent,
                socket_path=args.socket,
            )
            print(json.dumps(result, indent=2))

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


def _daemonize(socket_path: str) -> None:
    """Fork into background daemon."""
    pid = os.fork()
    if pid > 0:
        # Parent — print PID and exit
        print(f"Bus daemon started (pid {pid})", file=sys.stderr)
        sys.exit(0)
    # Child — become session leader
    os.setsid()
    # Redirect stdio
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    # Keep stderr for logging
    asyncio.run(_serve_impl(socket_path, background=True))


# =============================================================================
# FASTMCP SERVER (lazy — only loaded when mcp-stdio is invoked)
# =============================================================================


def _run_mcp():
    from fastmcp import FastMCP

    mcp = FastMCP("events")

    # MCP for _status_impl
    @mcp.tool()
    def status() -> str:
        """Get bus server health: uptime, clients, channels, message count.

        Args:
            (none)

        Returns:
            JSON with bus status fields
        """
        result, _ = _status_impl()
        return json.dumps(result, indent=2)

    # MCP for _channels_impl
    @mcp.tool()
    def channels() -> str:
        """List active channels with subscriber counts and message stats.

        Args:
            (none)

        Returns:
            JSON with channel name → stats mapping
        """
        result, _ = _channels_impl()
        return json.dumps(result, indent=2)

    # MCP for _publish_impl
    @mcp.tool()
    def publish(
        channel: str, payload: str, source: str = "", correlation_id: str = ""
    ) -> str:
        """Publish a message to a channel.

        Args:
            channel: Target channel (auto-scoped to project, use @ for absolute)
            payload: JSON payload string
            source: Source identifier
            correlation_id: Optional correlation ID for tracing

        Returns:
            JSON with message_id, channel, delivery count
        """
        result, _ = _publish_impl(
            channel, payload, source=source, correlation_id=correlation_id
        )
        return json.dumps(result, indent=2)

    # MCP for _subscribe_impl
    @mcp.tool()
    def subscribe(channel: str, count: int = 20) -> str:
        """Read recent messages from a channel pattern.

        Args:
            channel: Channel pattern with glob support (e.g. "alerts:*")
            count: Max messages to return

        Returns:
            JSON with channel and recent messages array
        """
        result, _ = _subscribe_impl(channel, count)
        return json.dumps(result, indent=2)

    # MCP for _query_audit_impl
    @mcp.tool()
    def query(
        channel: str = "", since: str = "", correlation_id: str = "", limit: int = 50
    ) -> str:
        """Search the audit log for past events.

        Args:
            channel: Channel pattern to filter (glob)
            since: Time range like "1h", "30m", "2d", or ISO timestamp
            correlation_id: Filter by correlation ID
            limit: Max results

        Returns:
            JSON list of audit entries
        """
        # Resolve channel with project scoping
        resolved = ""
        if channel:
            if channel.startswith("@"):
                resolved = channel[1:]
            else:
                resolved = _resolve_channel(channel, _detect_project())
        results, metrics = _query_audit_impl(resolved, since, correlation_id, limit)
        return json.dumps({"results": results, "metrics": metrics}, indent=2)

    # MCP for _stop_impl
    @mcp.tool()
    def stop() -> str:
        """Gracefully stop the bus daemon.

        Args:
            (none)

        Returns:
            JSON confirmation
        """
        result, _ = _stop_impl()
        return json.dumps(result, indent=2)

    # MCP for _trigger_add_impl
    @mcp.tool()
    def trigger_add(trigger_type: str, channel: str, config: str = "{}") -> str:
        """Add a file or URL trigger that publishes events to a channel.

        Args:
            trigger_type: "file" (watchdog) or "url" (HTTP polling)
            channel: Target channel (auto-scoped to project, use @ for absolute)
            config: JSON config — file: {"path": "/dir", "recursive": true, "events": ["modified","created"]}
                    url: {"url": "https://...", "interval_seconds": 300}

        Returns:
            JSON with trigger_id, type, channel
        """
        result, _ = _trigger_add_impl(trigger_type, channel, config)
        return json.dumps(result, indent=2)

    # MCP for _trigger_list_impl
    @mcp.tool()
    def trigger_list() -> str:
        """List all active triggers with their type, channel, config, and status.

        Args:
            (none)

        Returns:
            JSON with triggers array
        """
        result, _ = _trigger_list_impl()
        return json.dumps(result, indent=2)

    # MCP for _trigger_remove_impl
    @mcp.tool()
    def trigger_remove(trigger_id: str) -> str:
        """Remove a trigger by ID. Stops the trigger and removes from persistence.

        Args:
            trigger_id: The trigger ID returned by trigger_add

        Returns:
            JSON confirmation with trigger_id and removed status
        """
        result, _ = _trigger_remove_impl(trigger_id)
        return json.dumps(result, indent=2)

    # MCP for _agent_spawn_impl
    @mcp.tool()
    def agent_spawn(
        name: str,
        agent_type: str = "subprocess",
        channels: str = "",
        config: str = "{}",
    ) -> str:
        """Spawn a new agent that subscribes to channels and processes messages.

        Args:
            name: Agent name (unique among running agents)
            agent_type: Runtime type — "subprocess" (runs command per message)
            channels: Comma-separated channel patterns (e.g. "alerts:*,watch:*")
            config: JSON config — subprocess: {"command": "/path/to/handler", "publish_to": "results:*", "timeout_seconds": 60}

        Returns:
            JSON with agent_id, name, type, channels, inbox
        """
        result, _ = _agent_spawn_impl(name, agent_type, channels, config)
        return json.dumps(result, indent=2)

    # MCP for _agent_list_impl
    @mcp.tool()
    def agent_list() -> str:
        """List all agents with their status, channels, and message counts.

        Args:
            (none)

        Returns:
            JSON with agents array
        """
        result, _ = _agent_list_impl()
        return json.dumps(result, indent=2)

    # MCP for _agent_send_impl
    @mcp.tool()
    def agent_send(
        agent: str, payload: str, source: str = "", correlation_id: str = ""
    ) -> str:
        """Send a direct message to a named agent's inbox.

        Args:
            agent: Agent name or ID
            payload: JSON payload string
            source: Source identifier
            correlation_id: Optional correlation ID for tracing

        Returns:
            JSON with agent_id, inbox channel, message_id, delivery count
        """
        result, _ = _agent_send_impl(
            agent, payload, source=source, correlation_id=correlation_id
        )
        return json.dumps(result, indent=2)

    # MCP for _agent_retire_impl
    @mcp.tool()
    def agent_retire(agent: str) -> str:
        """Retire (stop) an agent by name or ID.

        Args:
            agent: Agent name or ID to retire

        Returns:
            JSON confirmation with agent_id and retired status
        """
        result, _ = _agent_retire_impl(agent)
        return json.dumps(result, indent=2)

    print("Sanctum event bus MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
