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
    sfs_events.py dashboard [static|serve] [-n] [-p PORT] [-s SOCKET]
    sfs_events.py mcp-stdio
"""

import argparse
import asyncio
import ast
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
import html as _html_mod

# =============================================================================
# LOGGING (TSV format — SPEC_SFB Principle 6)
# =============================================================================
_LEVELS = {"TRACE": 5, "DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40, "FATAL": 50}
_THRESHOLD = _LEVELS.get(
    os.environ.get("SFB_LOG_LEVEL", os.environ.get("SFB_LOG_LEVEL", "INFO")), 20
)
_LOG_DIR = os.environ.get("SFB_LOG_DIR", os.environ.get("SFB_LOG_DIR", ""))
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
    "dashboard",
    "stage",
    "staged",
    "refine",
    "release",
    "pull",
    "drop",
]

SANCTUM_DIR = Path.home() / ".sanctum"
DEFAULT_SOCKET = str(SANCTUM_DIR / "bus.sock")
PID_FILE = SANCTUM_DIR / "bus.pid"
AUDIT_DIR = SANCTUM_DIR / "audit"
AUDIT_FILE = AUDIT_DIR / "events.jsonl"

TRIGGERS_FILE = SANCTUM_DIR / "triggers.jsonl"
QUEUE_DIR = SANCTUM_DIR / "bus-queues"

STAGING_FILE = Path(__file__).resolve().parent.parent / "data" / "staging.jsonl"
_VALID_PRIORITIES = ("critical", "high", "medium", "low")
_VALID_TARGETS = ("review", "fix", "docs", "research")
_PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

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
    for cid, c in list(_clients.items()):
        for pattern in c["subscriptions"]:
            if _channel_matches(pattern, channel):
                try:
                    await _write_frame(c["writer"], {"event": "message", "message": msg})
                    delivered += 1
                except Exception as e:
                    _log(
                        "WARN", "delivery_failed",
                        f"client={cid} channel={channel}: {e}",
                    )
                break

    # Deliver to agent queues
    for aid, agent in _agents.items():
        if agent.get("status") != "running":
            continue
        for pattern in agent.get("channels", []):
            if _channel_matches(pattern, channel):
                queue = _agent_queues.get(aid)
                if queue:
                    if agent.get("config", {}).get("durable"):
                        _durable_enqueue(aid, msg)
                    try:
                        queue.put_nowait(msg)
                        delivered += 1
                    except asyncio.QueueFull:
                        aname = agent.get("name", aid)
                        _log(
                            "ERROR", "agent_queue_full",
                            f"agent={aname} qsize={queue.qsize()} — message dropped",
                        )
                break

    # Audit
    await _audit_log(msg)
    _message_count += 1
    return delivered


async def _internal_publish(
    channel: str, payload: dict, source: str, correlation_id: str = ""
) -> dict:
    """Publish a message from an internal source (trigger/agent). Returns the message."""
    msg = _make_message(
        channel=channel, payload=payload, source=source, correlation_id=correlation_id
    )
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
async def _schedule_trigger_run(
    trigger_id: str, config: dict, channel: str
) -> None:
    """Schedule trigger — publishes a tick message at fixed intervals."""
    interval = config.get("interval_seconds", 300)
    label = config.get("label", "tick")
    assert interval >= 10, f"interval_seconds >= 10 (got {interval})"

    try:
        _log(
            "INFO",
            "trigger_start",
            f"schedule trigger {trigger_id} every {interval}s label={label}",
        )
        while True:
            await asyncio.sleep(interval)
            await _internal_publish(
                channel,
                {
                    "trigger_id": trigger_id,
                    "event": "schedule",
                    "label": label,
                    "timestamp": datetime.now(timezone.utc).isoformat(
                        timespec="milliseconds"
                    ),
                },
                source=f"trigger:{trigger_id}",
            )
    except asyncio.CancelledError:
        pass
    finally:
        _log("INFO", "trigger_stop", f"schedule trigger {trigger_id} stopped")


TRIGGER_TYPES["file"] = "file"
TRIGGER_TYPES["url"] = "url"
TRIGGER_TYPES["schedule"] = "schedule"

_TRIGGER_RUNNERS = {
    "file": _file_trigger_run,
    "url": _url_trigger_run,
    "schedule": _schedule_trigger_run,
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


# =============================================================================
# CORE — Durable Agent Queues
# =============================================================================


def _durable_enqueue(agent_id: str, msg: dict) -> None:
    """Append message to durable queue file. Called before in-memory enqueue."""
    try:
        QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        with open(QUEUE_DIR / f"{agent_id}.jsonl", "a") as f:
            f.write(json.dumps({"type": "msg", "msg": msg}) + "\n")
    except Exception as e:
        _log("ERROR", "durable_enqueue_fail", f"agent={agent_id}: {e}")


def _durable_ack(agent_id: str, msg_id: str) -> None:
    """Mark message as processed in durable queue."""
    queue_file = QUEUE_DIR / f"{agent_id}.jsonl"
    if not queue_file.exists():
        return
    try:
        with open(queue_file, "a") as f:
            f.write(json.dumps({"type": "ack", "msg_id": msg_id}) + "\n")
    except Exception as e:
        _log("ERROR", "durable_ack_fail", f"agent={agent_id} msg={msg_id}: {e}")


def _durable_replay(agent_id: str) -> list[dict]:
    """Return unacked messages and compact the queue file."""
    queue_file = QUEUE_DIR / f"{agent_id}.jsonl"
    if not queue_file.exists():
        return []

    messages: dict[str, dict] = {}  # msg_id → msg (preserves order)
    acked: set[str] = set()

    try:
        with open(queue_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry["type"] == "msg":
                    msg = entry["msg"]
                    messages[msg["id"]] = msg
                elif entry["type"] == "ack":
                    acked.add(entry["msg_id"])
    except Exception as e:
        _log("ERROR", "durable_replay_read_fail", f"agent={agent_id}: {e}")
        return []

    unacked = [msg for mid, msg in messages.items() if mid not in acked]

    # Compact: rewrite with only unacked messages
    try:
        if unacked:
            with open(queue_file, "w") as f:
                for msg in unacked:
                    f.write(json.dumps({"type": "msg", "msg": msg}) + "\n")
        else:
            queue_file.unlink(missing_ok=True)
    except Exception as e:
        _log("ERROR", "durable_compact_fail", f"agent={agent_id}: {e}")

    return unacked


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
            max_retries = agent_def["config"].get("timeout_retries", 0)
            backoff = agent_def["config"].get("retry_backoff_seconds", 5.0)
            inbound_cid = msg.get("correlation_id", "")

            for attempt in range(max_retries + 1):
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

                    # Ack durable queue after successful processing
                    if agent_def.get("config", {}).get("durable"):
                        _durable_ack(agent_id, msg.get("id", ""))

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
                            out_channel, result_payload, source=f"agent:{agent_id}",
                            correlation_id=inbound_cid,
                        )
                    break  # Success — exit retry loop

                except asyncio.TimeoutError:
                    # Kill the timed-out process
                    try:
                        proc.kill()
                        await proc.wait()
                    except Exception:
                        pass

                    if attempt < max_retries:
                        delay = backoff * (2 ** attempt)
                        _log(
                            "WARN", "agent_retry",
                            f"agent={agent_id} attempt={attempt + 1}/{max_retries + 1}"
                            f" delay={delay}s",
                        )
                        await asyncio.sleep(delay)
                        continue

                    # All retries exhausted — publish failure event
                    _log(
                        "ERROR", "agent_timeout_exhausted",
                        f"agent={agent_id} attempts={max_retries + 1}",
                    )
                    await _internal_publish(
                        _resolve_channel("agents:failed", project),
                        {
                            "agent_id": agent_id,
                            "agent_name": agent_def.get("name", ""),
                            "reason": "timeout_exhausted",
                            "attempts": max_retries + 1,
                            "timeout_seconds": timeout_s,
                            "message_id": msg.get("id", ""),
                            "message_channel": msg.get("channel", ""),
                        },
                        source=f"agent:{agent_id}",
                        correlation_id=inbound_cid,
                    )

                    # Dead-letter queue — preserve the original message
                    orig_ch = msg.get("channel", "")
                    dlq_ch = (
                        orig_ch.replace("-tasks", "-dlq")
                        if "-tasks" in orig_ch
                        else f"dlq:{agent_def.get('name', 'unknown')}"
                    )
                    await _internal_publish(
                        dlq_ch,
                        {
                            "reason": "timeout_exhausted",
                            "attempts": max_retries + 1,
                            "original_message": msg,
                        },
                        source=f"agent:{agent_id}",
                        correlation_id=inbound_cid,
                    )

                except Exception as e:
                    _log("ERROR", "agent_process_error", f"agent={agent_id}: {e}")
                    break  # Non-timeout errors are not retried
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

    max_q = agent_def.get("config", {}).get("max_queue_size", 50)
    queue: asyncio.Queue = asyncio.Queue(maxsize=max_q)
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

    # Replay durable queues for agents with durable=true
    for aid, d in _agents.items():
        if d.get("config", {}).get("durable") and d.get("status") == "running":
            unacked = _durable_replay(aid)
            if unacked:
                queue = _agent_queues.get(aid)
                if queue:
                    for m in unacked:
                        await queue.put(m)
                    _log(
                        "INFO", "durable_replay",
                        f"agent={aid} name={d.get('name', '?')} replayed={len(unacked)}",
                    )

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
        trigger_type: Trigger type — "file", "url", or "schedule"
        channel: Channel to publish events to (auto-scoped to project)
        config: JSON config string — file: {path, recursive, events, debounce_seconds}
                                    — url: {url, interval_seconds}
                                    — schedule: {interval_seconds, label}
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
# STAGING — Task Queue (append-only JSONL)
# =============================================================================


def _staging_read_all() -> list[dict]:
    """Read all lines from staging.jsonl."""
    if not STAGING_FILE.exists():
        return []
    entries: list[dict] = []
    with open(STAGING_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
    return entries


def _staging_current() -> list[dict]:
    """Get current state of all tasks (last snapshot per ID)."""
    all_entries = _staging_read_all()
    latest: dict[str, dict] = {}
    for entry in all_entries:
        task_id = entry.get("id", "")
        if task_id:
            latest[task_id] = entry
    return list(latest.values())


def _staging_active() -> list[dict]:
    """Get only staged tasks, sorted by priority then age."""
    current = _staging_current()
    active = [t for t in current if t.get("status") == "staged"]
    active.sort(
        key=lambda t: (
            _PRIORITY_ORDER.get(t.get("priority", "medium"), 2),
            t.get("created", ""),
        )
    )
    return active


def _staging_append(entry: dict) -> None:
    """Append a staging entry to the JSONL file."""
    STAGING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STAGING_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# -- Staging _impl functions --------------------------------------------------


def _stage_impl(
    title: str,
    description: str = "",
    priority: str = "medium",
    target: str = "fix",
) -> tuple[dict, dict]:
    """Stage a new task for team execution. CLI: stage, MCP: stage."""
    start_ms = time.time() * 1000
    assert title.strip(), "Title is required"
    assert priority in _VALID_PRIORITIES, f"Invalid priority: {priority}. Must be one of {_VALID_PRIORITIES}"
    assert target in _VALID_TARGETS, f"Invalid target: {target}. Must be one of {_VALID_TARGETS}"

    now = datetime.now(timezone.utc).isoformat()
    task_id = uuid.uuid4().hex[:8]
    entry = {
        "id": task_id,
        "status": "staged",
        "title": title.strip(),
        "description": description.strip(),
        "priority": priority,
        "target": target,
        "created": now,
        "updated": now,
        "correlation_id": "",
    }
    _staging_append(entry)
    _log("INFO", "stage", f"id={task_id} title={title[:60]} target={target} priority={priority}")

    latency_ms = time.time() * 1000 - start_ms
    return entry, {"status": "success", "action": "staged", "latency_ms": round(latency_ms, 2)}


def _staged_impl(
    include_released: bool = False,
) -> tuple[list[dict], dict]:
    """List staged tasks sorted by priority. CLI: staged, MCP: staged."""
    start_ms = time.time() * 1000

    if include_released:
        current = _staging_current()
        tasks = [t for t in current if t.get("status") in ("staged", "released", "pulled")]
        tasks.sort(
            key=lambda t: (
                0 if t["status"] == "staged" else 1,
                _PRIORITY_ORDER.get(t.get("priority", "medium"), 2),
                t.get("created", ""),
            )
        )
    else:
        tasks = _staging_active()

    latency_ms = time.time() * 1000 - start_ms
    return tasks, {"status": "success", "count": len(tasks), "latency_ms": round(latency_ms, 2)}


def _refine_impl(
    task_id: str,
    title: str = "",
    description: str = "",
    priority: str = "",
    target: str = "",
) -> tuple[dict, dict]:
    """Refine a staged task. CLI: refine, MCP: refine."""
    start_ms = time.time() * 1000
    assert task_id.strip(), "Task ID is required"
    if priority:
        assert priority in _VALID_PRIORITIES, f"Invalid priority: {priority}"
    if target:
        assert target in _VALID_TARGETS, f"Invalid target: {target}"

    current = _staging_current()
    task = next((t for t in current if t["id"] == task_id), None)
    assert task is not None, f"Task {task_id} not found"
    assert task["status"] == "staged", f"Can only refine staged tasks (current status: {task['status']})"

    updated = dict(task)
    if title.strip():
        updated["title"] = title.strip()
    if description.strip():
        updated["description"] = description.strip()
    if priority:
        updated["priority"] = priority
    if target:
        updated["target"] = target
    updated["updated"] = datetime.now(timezone.utc).isoformat()

    _staging_append(updated)
    _log("INFO", "refine", f"id={task_id}")

    latency_ms = time.time() * 1000 - start_ms
    return updated, {"status": "success", "action": "refined", "latency_ms": round(latency_ms, 2)}


def _release_impl(
    task_id: str,
    *,
    socket_path: str = DEFAULT_SOCKET,
) -> tuple[dict, dict]:
    """Release a staged task to its target coordinator. CLI: release, MCP: release."""
    start_ms = time.time() * 1000
    assert task_id.strip(), "Task ID is required"

    current = _staging_current()
    task = next((t for t in current if t["id"] == task_id), None)
    assert task is not None, f"Task {task_id} not found"
    assert task["status"] == "staged", f"Can only release staged tasks (current status: {task['status']})"

    correlation_id = f"staging-{task_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    target = task["target"]
    channel = f"coord:{target}-tasks"

    payload = {
        "title": task["title"],
        "prompt": task["description"],
        "source": "staging",
        "correlation_id": correlation_id,
    }

    pub_result = _bus_request(
        {
            "cmd": "publish",
            "channel": channel,
            "payload": payload,
            "source": "staging",
            "correlation_id": correlation_id,
        },
        socket_path,
    )

    released = dict(task)
    released["status"] = "released"
    released["correlation_id"] = correlation_id
    released["updated"] = datetime.now(timezone.utc).isoformat()
    _staging_append(released)

    _log("INFO", "release", f"id={task_id} target={target} correlation_id={correlation_id}")

    latency_ms = time.time() * 1000 - start_ms
    result = {
        "task": released,
        "published_to": channel,
        "correlation_id": correlation_id,
        "delivery": pub_result.get("delivery", 0),
    }
    return result, {"status": "success", "action": "released", "latency_ms": round(latency_ms, 2)}


def _pull_impl(task_id: str) -> tuple[dict, dict]:
    """Pull a staged task for direct work (exception path). CLI: pull, MCP: pull."""
    start_ms = time.time() * 1000
    assert task_id.strip(), "Task ID is required"

    current = _staging_current()
    task = next((t for t in current if t["id"] == task_id), None)
    assert task is not None, f"Task {task_id} not found"
    assert task["status"] == "staged", f"Can only pull staged tasks (current status: {task['status']})"

    pulled = dict(task)
    pulled["status"] = "pulled"
    pulled["updated"] = datetime.now(timezone.utc).isoformat()
    _staging_append(pulled)

    _log("INFO", "pull", f"id={task_id} title={task['title'][:60]}")

    latency_ms = time.time() * 1000 - start_ms
    return pulled, {"status": "success", "action": "pulled", "latency_ms": round(latency_ms, 2)}


def _drop_impl(task_id: str) -> tuple[dict, dict]:
    """Drop (cancel) a staged task. CLI: drop, MCP: drop."""
    start_ms = time.time() * 1000
    assert task_id.strip(), "Task ID is required"

    current = _staging_current()
    task = next((t for t in current if t["id"] == task_id), None)
    assert task is not None, f"Task {task_id} not found"
    assert task["status"] in ("staged", "pulled"), f"Can only drop staged or pulled tasks (current status: {task['status']})"

    dropped = dict(task)
    dropped["status"] = "dropped"
    dropped["updated"] = datetime.now(timezone.utc).isoformat()
    _staging_append(dropped)

    _log("INFO", "drop", f"id={task_id}")

    latency_ms = time.time() * 1000 - start_ms
    return dropped, {"status": "success", "action": "dropped", "latency_ms": round(latency_ms, 2)}


# =============================================================================
# DASHBOARD — The War Room
# =============================================================================

_BUS_DASHBOARD_PATH = Path.cwd() / "bus-dashboard.html"
_COORDINATOR_NAMES = frozenset({
    "review-coordinator", "fix-coordinator",
    "docs-coordinator", "research-coordinator",
})
_CHANNEL_COLORS = {
    "agents:": "#58a6ff",
    "coord:": "#bc8cff",
    "research:": "#3fb950",
    "alerts:": "#f7768e",
    "watch:": "#8b949e",
    "tick:": "#8b949e",
    "results:": "#79c0ff",
}
_FEED_EXCLUDE_PREFIXES = ("tick:", "watch:")
_DASHBOARD_PORT = 8420


# -- Utility functions --------------------------------------------------------


def _html_esc(s: str) -> str:
    """HTML-escape a string."""
    return _html_mod.escape(str(s))


def _tooltip_esc(s: str) -> str:
    """Escape for data-tooltip attributes. HTML-escape + newline to &#10;."""
    return _html_mod.escape(str(s)).replace("\n", "&#10;")


def _relative_time(ts: str | int | float) -> str:
    """Convert ISO timestamp or epoch to human-relative string."""
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            return str(ts)
    elif isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    else:
        return str(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        h, m = seconds // 3600, (seconds % 3600) // 60
        return f"{h}h {m}m ago"
    return f"{seconds // 86400}d ago"


def _human_duration(seconds: float) -> str:
    """Convert seconds to readable duration: 4h 23m, 2d 5h, 45m, 12s."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    if s < 86400:
        return f"{s // 3600}h {(s % 3600) // 60}m"
    return f"{s // 86400}d {(s % 86400) // 3600}h"


def _channel_color(channel: str) -> str:
    """Hex color for a channel based on prefix matching."""
    bare = channel.split("/")[-1] if "/" in channel else channel
    for prefix, color in _CHANNEL_COLORS.items():
        if bare.startswith(prefix):
            return color
    return "#8b949e"


def _strip_project(channel: str) -> str:
    """Remove project/ prefix from channel for display."""
    return channel.split("/", 1)[1] if "/" in channel else channel


def _payload_summary(payload: dict | str | None) -> str:
    """Extract a one-line summary from a message payload."""
    if payload is None:
        return "\u2014"
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return str(payload)[:120]
    if isinstance(payload, dict):
        if not payload:
            return "\u2014"
        for key in ("summary", "title", "task", "content", "status", "msg", "message"):
            if key in payload:
                val = str(payload[key])
                return val[:120] + ("\u2026" if len(val) > 120 else "")
        raw = json.dumps(payload, default=str)
        return raw[:120] + ("\u2026" if len(raw) > 120 else "")
    return str(payload)[:120]


# -- Data collection ----------------------------------------------------------


def _collect_dashboard_data(socket_path: str = DEFAULT_SOCKET) -> dict | None:
    """Collect all data for dashboard rendering in one pass.

    Returns dict with keys: status, agents, triggers, channels,
    activity, research, coord_tasks, coord_results, failures.
    Returns None if bus daemon is not running.
    """
    try:
        status = _bus_request({"cmd": "status"}, socket_path)
        if not status.get("ok"):
            return None
    except Exception:
        return None

    project = _detect_project()
    agents = _bus_request({"cmd": "agent-list"}, socket_path)
    triggers = _bus_request({"cmd": "trigger-list"}, socket_path)
    channels = _bus_request({"cmd": "channels"}, socket_path)

    # Audit log — direct file read, no bus connection needed
    activity, _ = _query_audit_impl(since="1h", limit=100)

    # Channel subscriptions for coordinator + research data
    research = _bus_request(
        {"cmd": "subscribe", "channel": "research:*", "count": 50},
        socket_path,
    )
    coord_tasks = _bus_request(
        {"cmd": "subscribe", "channel": "coord:*-tasks", "count": 50},
        socket_path,
    )
    coord_results = _bus_request(
        {"cmd": "subscribe", "channel": "coord:*-results", "count": 50},
        socket_path,
    )

    # Failures in last 1h (clears once resolved)
    failures, _ = _query_audit_impl(
        channel=f"{project}/agents:failed", since="1h", limit=50,
    )

    return {
        "status": status,
        "agents": agents,
        "triggers": triggers,
        "channels": channels,
        "activity": activity,
        "research": research,
        "coord_tasks": coord_tasks,
        "coord_results": coord_results,
        "failures": failures,
    }


# -- Overall status computation -----------------------------------------------


def _compute_overall_status(data: dict) -> str:
    """Determine overall health: green, amber, or red."""
    agents_list = data.get("agents", {}).get("agents", [])
    coord_running = {
        a["name"]
        for a in agents_list
        if a.get("name") in _COORDINATOR_NAMES and a.get("status") == "running"
    }

    # Red: any coordinator missing or not running
    if len(coord_running) < len(_COORDINATOR_NAMES):
        return "red"

    # Amber: real failures in last 1h (ignore timeout_exhausted — expected for big tasks)
    failures = data.get("failures", [])
    real_failures = [
        f for f in failures
        if "timeout_exhausted" not in str(f.get("payload_preview", ""))
    ]
    if real_failures:
        return "amber"

    # Amber: no bus activity for >5 min despite being up
    status = data.get("status", {})
    activity = data.get("activity", [])
    if status.get("uptime_seconds", 0) > 300 and not activity:
        return "amber"

    return "green"


# -- Panel renderers ----------------------------------------------------------


def _render_health_bar(data: dict) -> str:
    """P1: System health bar -- top strip."""
    status = data.get("status", {})
    overall = _compute_overall_status(data)

    labels = {
        "green": "All Systems Operational",
        "amber": "Degraded",
        "red": "Failures Detected",
    }
    uptime = _human_duration(status.get("uptime_seconds", 0))
    msgs = status.get("messages_total", 0)
    chans = status.get("channels", 0)
    pid = status.get("pid", "?")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    tip = _tooltip_esc(
        f"{msgs} messages \u2022 {chans} channels \u2022 PID {pid} \u2022 Generated {generated}"
    )

    return (
        f'<div id="health-bar" class="health-bar health-{overall}"'
        f' data-tooltip="{tip}">'
        f'<span class="health-dot"></span>'
        f'<span class="health-label">{_html_esc(labels[overall])}</span>'
        f'<span class="health-uptime">{_html_esc(uptime)}</span>'
        f'</div>'
    )


def _render_coordinators(data: dict) -> str:
    """P2: Coordinator cards."""
    agents_list = data.get("agents", {}).get("agents", [])
    coord_map = {a["name"]: a for a in agents_list if a.get("name") in _COORDINATOR_NAMES}
    tasks = data.get("coord_tasks", {}).get("recent", [])
    results = data.get("coord_results", {}).get("recent", [])

    cards = []
    for name in sorted(_COORDINATOR_NAMES):
        short = name.replace("-coordinator", "")
        agent = coord_map.get(name)

        if agent is None:
            cards.append(
                f'<div class="coord-card coord-missing"'
                f' data-tooltip="Coordinator not spawned">'
                f'<span class="coord-dot status-red">\u25cf</span>'
                f'<span class="coord-name">{_html_esc(short)}</span>'
                f'<span class="coord-state">Missing</span>'
                f'</div>'
            )
            continue

        st = agent.get("status", "unknown")
        st_class = "green" if st == "running" else "red"
        msgs_processed = agent.get("messages_processed", 0)
        spawned = agent.get("spawned_at", "")
        up = _relative_time(spawned) if spawned else "?"

        # Find last task and result for this coordinator
        role_tasks = [t for t in tasks
                      if _strip_project(t.get("channel", "")) == f"coord:{short}-tasks"]
        role_results = [r for r in results
                        if _strip_project(r.get("channel", "")) == f"coord:{short}-results"]

        last_task = role_tasks[-1] if role_tasks else None
        last_result = role_results[-1] if role_results else None

        # Derive current state
        if last_task and last_result:
            task_ts = last_task.get("timestamp", "")
            result_ts = last_result.get("timestamp", "")
            if task_ts > result_ts:
                task_title = _payload_summary(last_task.get("payload"))
                state = f"Working: {task_title[:40]}"
            else:
                state = "Idle"
        elif last_task and not last_result:
            task_title = _payload_summary(last_task.get("payload"))
            state = f"Working: {task_title[:40]}"
        else:
            state = "No activity"

        # Tooltip
        last_r_summary = _payload_summary(last_result.get("payload")) if last_result else "none"
        tip_lines = [
            f"\U0001f4e8 {msgs_processed} processed",
            f"Last: {last_r_summary[:60]}",
            f"\u23f1 Up {up}",
        ]
        tip = _tooltip_esc("\n".join(tip_lines))

        cards.append(
            f'<div class="coord-card coord-{"working" if state.startswith("Working") else st}"'
            f' data-tooltip="{tip}">'
            f'<span class="coord-dot status-{st_class}">\u25cf</span>'
            f'<span class="coord-name">{_html_esc(short)}</span>'
            f'<span class="coord-state">{_html_esc(state)}</span>'
            f'</div>'
        )

    return (
        '<div id="panel-coordinators" class="section coord-section">'
        '<h2 class="section-title">Coordinators</h2>'
        '<div class="coord-grid">'
        + "".join(cards)
        + '</div></div>'
    )


def _render_agents(data: dict) -> str:
    """P3: Non-coordinator bus agents grid."""
    agents_list = data.get("agents", {}).get("agents", [])
    non_coord = [a for a in agents_list if a.get("name") not in _COORDINATOR_NAMES]

    if not non_coord:
        return (
            '<div id="panel-agents" class="section agent-section">'
            '<h2 class="section-title">Agents</h2>'
            '<p class="empty">No utility agents running</p>'
            '</div>'
        )

    cards = []
    for a in non_coord:
        name = a.get("name", "?")
        st = a.get("status", "unknown")
        st_class = "green" if st == "running" else "red" if st == "retired" else "grey"
        channels = [
            _strip_project(c) for c in a.get("channels", [])
            if not c.startswith("_sanctum/")
        ]
        msgs = a.get("messages_processed", 0)
        spawned = a.get("spawned_at", "")
        tip_lines = [
            f"\U0001f4e1 {', '.join(channels) or '(inbox only)'}",
            f"\U0001f4e8 {msgs} processed",
            f"Spawned {_relative_time(spawned)}" if spawned else "",
        ]
        tip = _tooltip_esc("\n".join(line for line in tip_lines if line))

        cards.append(
            f'<div class="agent-card agent-{st}"'
            f' data-tooltip="{tip}">'
            f'<span class="agent-dot status-{st_class}">\u25cf</span>'
            f'<span class="agent-name">{_html_esc(name)}</span>'
            f'</div>'
        )

    return (
        '<div id="panel-agents" class="section agent-section">'
        '<h2 class="section-title">Agents</h2>'
        '<div class="agent-grid">'
        + "".join(cards)
        + '</div></div>'
    )


def _render_active_tasks(data: dict) -> str:
    """P3.5: Split activity — current missions (left) + completed history (right)."""
    tasks = data.get("coord_tasks", {}).get("recent", [])
    results = data.get("coord_results", {}).get("recent", [])
    activity = data.get("activity", [])

    if not tasks:
        return '<div id="panel-active-tasks"></div>'

    # -- Classify tasks: top-level (from primary agent) vs sub-tasks (from coordinators)
    top_level: list[dict] = []
    sub_tasks: list[dict] = []
    coord_sources = {n.replace("-coordinator", "") for n in _COORDINATOR_NAMES}

    for t in tasks:
        src = t.get("source", "")
        # Sub-task if source is a coordinator name or agent:ID
        is_sub = any(cs in src for cs in coord_sources) or src.startswith("agent:")
        (sub_tasks if is_sub else top_level).append(t)

    # -- Build missions from top-level tasks, keyed by correlation_id
    missions: dict[str, dict] = {}
    for t in top_level:
        cid = t.get("correlation_id", "") or t.get("id", "?")
        ch = _strip_project(t.get("channel", ""))
        coord = ch.replace("coord:", "").replace("-tasks", "")
        payload = t.get("payload", {})
        if isinstance(payload, str):
            try:
                payload = ast.literal_eval(payload)
            except Exception:
                try:
                    payload = json.loads(payload)
                except Exception:
                    pass
        title = ""
        if isinstance(payload, dict):
            title = payload.get("title", "") or _payload_summary(payload)
        if not title:
            title = _payload_summary(payload)

        if cid not in missions:
            missions[cid] = {
                "title": title,
                "coordinator": coord,
                "sub_coordinators": set(),
                "sub_count": 0,
                "completed_count": 0,
                "first_ts": t.get("timestamp", ""),
                "last_result_ts": "",
            }
        else:
            missions[cid]["sub_coordinators"].add(coord)

    # -- Attach sub-tasks to their parent missions
    for t in sub_tasks:
        cid = t.get("correlation_id", "")
        if cid not in missions:
            continue
        missions[cid]["sub_count"] += 1
        ch = _strip_project(t.get("channel", ""))
        coord = ch.replace("coord:", "").replace("-tasks", "")
        missions[cid]["sub_coordinators"].add(coord)

    # -- Count completed results per mission correlation_id
    for r in results:
        cid = r.get("correlation_id", "")
        if cid not in missions:
            continue
        payload = r.get("payload", {})
        if isinstance(payload, str):
            try:
                payload = ast.literal_eval(payload)
            except Exception:
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {}
        st = ""
        if isinstance(payload, dict):
            st = str(payload.get("status", "")).lower()
        if st in ("completed", "complete", "success", "compliant", "already_compliant", "failed"):
            missions[cid]["completed_count"] += 1
            rts = r.get("timestamp", "")
            if rts > missions[cid]["last_result_ts"]:
                missions[cid]["last_result_ts"] = rts

    # -- Extract active workers from agent lifecycle events
    workers: dict[str, list[dict]] = {}
    for evt in activity:
        ch = _strip_project(evt.get("channel", ""))
        if ch not in ("agents:started", "agents:completed", "agents:failed"):
            continue
        cid = evt.get("correlation_id", "")
        if not cid:
            continue
        payload = evt.get("payload") or evt.get("payload_preview", "")
        if isinstance(payload, str):
            try:
                payload = ast.literal_eval(payload)
            except Exception:
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {}
        if isinstance(payload, dict):
            workers.setdefault(cid, []).append({
                "agent": payload.get("agent", ""),
                "action": ch.split(":")[-1],
                "detail": payload.get("summary", payload.get("file", "")),
                "ts": evt.get("timestamp", ""),
            })

    # -- Classify missions as current vs completed
    current_missions: list[tuple[str, dict]] = []
    completed_missions: list[tuple[str, dict]] = []

    for cid, m in sorted(missions.items(), key=lambda x: x[1]["first_ts"], reverse=True):
        sub_count = m["sub_count"]
        completed = m["completed_count"]
        # Staleness: flag missions with no activity for >30min
        stale = False
        if m["last_result_ts"]:
            try:
                last_dt = datetime.fromisoformat(m["last_result_ts"])
                stale = (datetime.now(timezone.utc) - last_dt).total_seconds() > 1800
            except (ValueError, TypeError):
                pass
        m["stale"] = stale
        is_done = (
            (sub_count == 0 and completed > 0)
            or (sub_count > 0 and completed >= sub_count)
        )
        if is_done:
            completed_missions.append((cid, m))
        else:
            current_missions.append((cid, m))

    # -- Render CURRENT activity cards (left column)
    current_cards: list[str] = []
    for cid, m in current_missions:
        title = m["title"]
        coord = m["coordinator"]
        sub_coords = m["sub_coordinators"]
        sub_count = m["sub_count"]
        completed = m["completed_count"]

        chain_parts = [coord]
        for sc in sorted(sub_coords):
            if sc != coord:
                chain_parts.append(sc)
        chain = " \u2192 ".join(chain_parts)

        if m.get("stale"):
            s_class, s_dot, s_label = "stale", "status-red", f"STALE {completed}/{sub_count}"
        elif sub_count == 0:
            s_class, s_dot, s_label = "active", "status-accent", "Active"
        elif completed > 0:
            s_class, s_dot, s_label = "working", "status-amber", f"{completed}/{sub_count}"
        else:
            s_class, s_dot, s_label = "active", "status-accent", f"0/{sub_count}"

        # Worker badges
        mission_workers = workers.get(cid, [])
        worker_html = ""
        if mission_workers:
            seen: set[str] = set()
            unique: list[dict] = []
            for w in sorted(mission_workers, key=lambda x: x["ts"], reverse=True):
                agent = w["agent"]
                if agent and agent not in seen:
                    seen.add(agent)
                    unique.append(w)
            w_parts: list[str] = []
            for w in unique[:6]:
                dc = "status-green" if w["action"] == "completed" else "status-amber" if w["action"] == "started" else "status-red"
                w_parts.append(
                    f'<span class="task-worker"><span class="{dc}">\u25cf</span> {_html_esc(w["agent"])}</span>'
                )
            if w_parts:
                worker_html = '<div class="task-workers">' + "".join(w_parts) + '</div>'

        tip_parts = [f"Correlation: {cid}"]
        if sub_count:
            tip_parts.append(f"Sub-tasks: {completed}/{sub_count} completed")
        tip_parts.append(f"Coordinators: {chain}")
        started = _relative_time(m["first_ts"]) if m["first_ts"] else "?"
        tip_parts.append(f"Started: {started}")
        tip = _tooltip_esc("\n".join(tip_parts))

        current_cards.append(
            f'<div class="task-card task-{s_class}" data-tooltip="{tip}">'
            f'<div class="task-header">'
            f'<span class="task-dot {s_dot}">\u25cf</span>'
            f'<span class="task-title">{_html_esc(title[:80])}</span>'
            f'<span class="task-status">{_html_esc(s_label)}</span>'
            f'</div>'
            f'<div class="task-meta">{_html_esc(chain)}{worker_html}</div>'
            f'</div>'
        )

    # -- Render COMPLETED activity rows (right column)
    completed_rows: list[str] = []
    for cid, m in completed_missions:
        title = m["title"]
        coord = m["coordinator"]
        sub_coords = m["sub_coordinators"]
        sub_count = m["sub_count"]
        completed = m["completed_count"]

        chain_parts = [coord]
        for sc in sorted(sub_coords):
            if sc != coord:
                chain_parts.append(sc)
        chain = " \u2192 ".join(chain_parts)

        # Compute runtime
        started_ts = m["first_ts"]
        finished_ts = m["last_result_ts"]
        runtime = ""
        if started_ts and finished_ts:
            try:
                t0 = datetime.fromisoformat(started_ts)
                t1 = datetime.fromisoformat(finished_ts)
                delta = (t1 - t0).total_seconds()
                runtime = _human_duration(max(0, delta))
            except (ValueError, TypeError):
                pass

        started_rel = _relative_time(started_ts) if started_ts else "?"
        finished_rel = _relative_time(finished_ts) if finished_ts else "?"

        # Success indicator
        if sub_count > 0 and completed >= sub_count:
            result_icon = '<span class="status-green">\u2713</span>'
            result_text = f"{completed}/{sub_count}"
        else:
            result_icon = '<span class="status-green">\u2713</span>'
            result_text = "Done"

        tip_parts = [
            f"Correlation: {cid}",
            f"Coordinators: {chain}",
            f"Started: {started_rel}",
            f"Finished: {finished_rel}",
        ]
        if runtime:
            tip_parts.append(f"Runtime: {runtime}")
        if sub_count:
            tip_parts.append(f"Sub-tasks: {completed}/{sub_count}")
        tip = _tooltip_esc("\n".join(tip_parts))

        completed_rows.append(
            f'<div class="completed-row" data-tooltip="{tip}">'
            f'<span class="completed-icon">{result_icon}</span>'
            f'<span class="completed-title">{_html_esc(title[:60])}</span>'
            f'<span class="completed-coord">{_html_esc(chain)}</span>'
            f'<span class="completed-runtime">{_html_esc(runtime) if runtime else "\u2014"}</span>'
            f'</div>'
        )

    # -- Assemble the split layout
    if not current_cards and not completed_rows:
        return '<div id="panel-active-tasks"></div>'

    left_html = ""
    if current_cards:
        left_html = (
            '<div class="activity-col activity-current">'
            '<h3 class="activity-subtitle">Current Activity</h3>'
            '<div class="tasks-list">'
            + "".join(current_cards)
            + '</div></div>'
        )
    else:
        left_html = (
            '<div class="activity-col activity-current">'
            '<h3 class="activity-subtitle">Current Activity</h3>'
            '<p class="empty">No active missions</p>'
            '</div>'
        )

    right_html = ""
    if completed_rows:
        right_html = (
            '<div class="activity-col activity-completed">'
            '<h3 class="activity-subtitle">Completed Activity</h3>'
            '<div class="completed-list">'
            + "".join(completed_rows)
            + '</div></div>'
        )

    return (
        '<div id="panel-active-tasks" class="section tasks-section">'
        '<h2 class="section-title">Activity</h2>'
        '<div class="activity-row">'
        + left_html + right_html
        + '</div></div>'
    )





def _render_triggers(data: dict) -> str:
    """P4: Triggers compact strip, grouped by type."""
    trigger_list = data.get("triggers", {}).get("triggers", [])

    if not trigger_list:
        return (
            '<div id="panel-triggers" class="section trigger-section">'
            '<h2 class="section-title">Triggers</h2>'
            '<p class="empty">No active triggers</p>'
            '</div>'
        )

    type_icons = {"file": "\U0001f4c1", "schedule": "\u23f0", "url": "\U0001f310"}
    type_order = {"file": 0, "schedule": 1, "url": 2}

    # Sort by type then channel for grouping
    sorted_triggers = sorted(
        trigger_list,
        key=lambda t: (type_order.get(t.get("type", ""), 9), t.get("channel", "")),
    )

    rows = []
    for t in sorted_triggers:
        ttype = t.get("type", "?")
        icon = type_icons.get(ttype, "\u2753")
        channel = _strip_project(t.get("channel", ""))
        st = t.get("status", "unknown")
        st_class = "green" if st == "running" else "red"
        config = t.get("config", {})

        # Config summary with human-readable times
        if ttype == "file":
            detail = config.get("path", "?")
            if config.get("recursive"):
                detail += " (recursive)"
            events = config.get("events", [])
            if events:
                detail += f"\nEvents: {', '.join(events)}"
        elif ttype == "schedule":
            interval = config.get("interval_seconds", 0)
            label = config.get("label", "")
            detail = f"every {_human_duration(interval)}" + (f" ({label})" if label else "")
        elif ttype == "url":
            url = config.get("url", "?")[:60]
            interval = config.get("interval_seconds", 0)
            detail = f"{url}\nPolling: every {_human_duration(interval)}" if interval else url
        else:
            detail = json.dumps(config)[:60]

        tid = t.get("id", "?")[:8]
        tip = _tooltip_esc(f"{detail}\nID: {tid}")

        rows.append(
            f'<div class="trigger-row" data-tooltip="{tip}">'
            f'<span class="trigger-type">{icon}</span>'
            f'<span class="trigger-channel">{_html_esc(channel)}</span>'
            f'<span class="trigger-dot status-{st_class}">\u25cf</span>'
            f'</div>'
        )

    return (
        '<div id="panel-triggers" class="section trigger-section">'
        '<h2 class="section-title">Triggers</h2>'
        '<div class="trigger-list">'
        + "".join(rows)
        + '</div></div>'
    )




def _render_research(data: dict) -> str:
    """P5: Research pipeline -- conditional, only if activity in last hour."""
    research_msgs = data.get("research", {}).get("recent", [])
    coord_tasks = data.get("coord_tasks", {}).get("recent", [])
    coord_results = data.get("coord_results", {}).get("recent", [])

    # Filter to research-related coordinator messages
    research_tasks = [
        t for t in coord_tasks
        if "research" in _strip_project(t.get("channel", ""))
    ]
    research_results = [
        r for r in coord_results
        if "research" in _strip_project(r.get("channel", ""))
    ]

    # No research activity -> omit panel entirely
    if not research_msgs and not research_tasks:
        return '<div id="panel-research"></div>'

    # Group by correlation_id
    groups: dict[str, list] = {}
    for msg in research_tasks + research_results + research_msgs:
        cid = msg.get("correlation_id", "")
        if cid:
            groups.setdefault(cid, []).append(msg)

    cards = []
    for cid, msgs in groups.items():
        # Sort by timestamp
        msgs.sort(key=lambda m: m.get("timestamp", ""))

        # Derive topic from first task
        topic = "Research"
        for m in msgs:
            p = m.get("payload", {})
            if isinstance(p, dict) and p.get("topic"):
                topic = p["topic"]
                break
            summary = _payload_summary(p)
            if summary and summary != "Research":
                topic = summary[:60]
                break

        # Derive phase from last message
        last = msgs[-1]
        last_ch = _strip_project(last.get("channel", ""))
        last_payload = last.get("payload", {})
        if isinstance(last_payload, str):
            try:
                last_payload = json.loads(last_payload)
            except (json.JSONDecodeError, TypeError):
                last_payload = {}

        if "research:complete" in last_ch or (isinstance(last_payload, dict) and last_payload.get("status") == "complete"):
            phase, phase_class = "Complete", "complete"
        elif "coord:docs" in last_ch and "results" in last_ch:
            phase, phase_class = "Documented", "complete"
        elif "coord:docs" in last_ch:
            phase, phase_class = "Documenting", "documenting"
        elif "research:finding" in last_ch or "research:" in last_ch:
            phase, phase_class = "Researching", "researching"
        elif isinstance(last_payload, dict) and last_payload.get("type") == "rfi":
            phase, phase_class = "RFI", "rfi"
        else:
            phase, phase_class = "Active", "researching"

        findings_count = sum(
            1 for m in msgs if "finding" in _strip_project(m.get("channel", ""))
        )

        # Build timeline for tooltip
        timeline_lines = []
        for m in msgs[-5:]:
            ts = _relative_time(m.get("timestamp", ""))
            ch = _strip_project(m.get("channel", ""))
            timeline_lines.append(f"{ts} \u2022 {ch}")

        tip = _tooltip_esc(
            "\n".join(timeline_lines) + f"\n\U0001f517 {cid[:8]}"
        )

        cards.append(
            f'<div class="research-card" data-tooltip="{tip}">'
            f'<span class="research-phase phase-{phase_class}">{_html_esc(phase)}</span>'
            f'<span class="research-topic">{_html_esc(topic)}</span>'
            f'<span class="research-findings">\U0001f4cb {findings_count}</span>'
            f'</div>'
        )

    if not cards:
        return '<div id="panel-research"></div>'

    return (
        '<div id="panel-research" class="section research-section">'
        '<h2 class="section-title">Research Pipeline</h2>'
        '<div class="research-list">'
        + "".join(cards)
        + '</div></div>'
    )


def _render_intel_feed() -> str:
    """Render intel log entries as rich cards."""
    intel_log = Path(__file__).resolve().parent.parent / "data" / "intel_log.jsonl"
    if not intel_log.exists():
        return '<div class="intel-scroll"><p class="empty">No intel entries yet</p></div>'

    entries: list[dict] = []
    with open(intel_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue

    entries.reverse()  # most recent first
    entries = entries[:30]

    if not entries:
        return '<div class="intel-scroll"><p class="empty">No intel entries yet</p></div>'

    source_icons = {
        "youtube": "\U0001f4fa",  # 📺
        "x": "\U0001d54f",  # 𝕏
        "subject": "\U0001f50d",  # 🔍
        "youtube-music": "\U0001f3b5",  # 🎵
    }
    relevance_colors = {
        "high": "#f7768e",
        "medium": "#e0af68",
        "low": "#7aa2f7",
        "noise": "#565f89",
    }

    cards: list[str] = []
    for entry in entries:
        source = entry.get("source", "unknown")
        src_type = entry.get("type", "subject")
        icon = source_icons.get(src_type, "\U0001f4cb")
        title = _html_esc(entry.get("title", "Untitled"))
        analysis = _html_esc(entry.get("analysis", ""))
        relevance = entry.get("relevance", "medium")
        rel_color = relevance_colors.get(relevance, "#565f89")
        ts = _relative_time(entry.get("timestamp", ""))
        urls = entry.get("urls", [])
        followup = _html_esc(entry.get("followup", ""))

        url_links = ""
        if urls:
            link_items = []
            for u in urls[:3]:
                domain = u.split("/")[2] if len(u.split("/")) > 2 else "link"
                link_items.append(f'<a href="{_html_esc(u)}" target="_blank" class="intel-link">{_html_esc(domain)}</a>')
            url_links = f'<span class="intel-urls">\U0001f517 {"  ".join(link_items)}</span>'

        followup_html = ""
        if followup:
            followup_html = f'<div class="intel-followup">\u2192 {followup}</div>'

        # Truncate analysis for card display, full in tooltip
        analysis_short = analysis[:250] + ("\u2026" if len(analysis) > 250 else "")
        tip = _tooltip_esc(entry.get("analysis", ""))

        cards.append(
            f'<div class="intel-entry intel-{relevance}" data-tooltip="{tip}">'
            f'<div class="intel-header">'
            f'<span class="intel-icon">{icon}</span>'
            f'<span class="intel-source">{_html_esc(source)}</span>'
            f'<span class="intel-relevance" style="background:{rel_color}25;color:{rel_color};'
            f'border:1px solid {rel_color}50">{_html_esc(relevance.upper())}</span>'
            f'<span class="intel-time">{_html_esc(ts)}</span>'
            f'</div>'
            f'<div class="intel-title">{title}</div>'
            f'<div class="intel-analysis">{analysis_short}</div>'
            f'<div class="intel-meta">'
            f'{url_links}'
            f'{followup_html}'
            f'</div>'
            f'</div>'
        )

    return '<div class="intel-scroll">' + "".join(cards) + '</div>'


def _render_activity_feed(data: dict) -> str:
    """P6: Combined intel + debug activity panel with toggle."""
    # -- Intel panel --
    intel_html = _render_intel_feed()

    # -- Debug panel (existing event log) --
    activity = data.get("activity", [])
    filtered = []
    for entry in activity:
        ch = _strip_project(entry.get("channel", ""))
        if any(ch.startswith(p) for p in _FEED_EXCLUDE_PREFIXES):
            continue
        filtered.append(entry)

    filtered.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    filtered = filtered[:50]

    if not filtered:
        debug_html = '<div class="feed-scroll"><p class="empty">No recent events</p></div>'
    else:
        rows = []
        for entry in filtered:
            ts = _relative_time(entry.get("timestamp", ""))
            ch = _strip_project(entry.get("channel", ""))
            color = _channel_color(entry.get("channel", ""))
            source = entry.get("source", "")
            payload = entry.get("payload") or entry.get("payload_preview", "")
            if isinstance(payload, str):
                try:
                    payload = ast.literal_eval(payload)
                except Exception:
                    pass
            summary = _payload_summary(payload)
            cid = entry.get("correlation_id", "")

            full_payload = json.dumps(payload, indent=2, default=str) if isinstance(payload, dict) else str(payload)
            if len(full_payload) > 500:
                full_payload = full_payload[:500] + "\u2026"
            tip_lines = [full_payload]
            if source:
                tip_lines.append(f"Source: {source}")
            if cid:
                tip_lines.append(f"\U0001f517 {cid[:12]}")
            tip_lines.append(entry.get("timestamp", ""))
            tip = _tooltip_esc("\n".join(tip_lines))

            rows.append(
                f'<div class="feed-entry" data-tooltip="{tip}">'
                f'<span class="feed-time">{_html_esc(ts)}</span>'
                f'<span class="feed-channel" style="background:{color}20;color:{color};'
                f'border:1px solid {color}40">{_html_esc(ch)}</span>'
                f'<span class="feed-summary">{_html_esc(summary)}</span>'
                f'</div>'
            )
        debug_html = '<div class="feed-scroll">' + "".join(rows) + '</div>'

    debug_count = len(filtered)

    return (
        f'<div id="panel-activity" class="section feed-section">'
        f'<h2 class="section-title">'
        f'<span class="feed-toggle">'
        f'<button class="toggle-btn active" data-view="intel" onclick="toggleFeed(\'intel\')">Intel</button>'
        f'<button class="toggle-btn" data-view="debug" onclick="toggleFeed(\'debug\')">Debug'
        f'<span class="feed-count">{debug_count}</span></button>'
        f'</span>'
        f'</h2>'
        f'<div id="feed-intel" class="feed-view active">{intel_html}</div>'
        f'<div id="feed-debug" class="feed-view" style="display:none">{debug_html}</div>'
        f'</div>'
    )


def _render_staging() -> str:
    """Render staging queue panel."""
    tasks = _staging_active()

    if not tasks:
        return (
            '<div class="section" id="panel-staging">'
            '<h2 class="section-title">\U0001f4cb Staging Queue</h2>'
            '<p class="empty">No staged tasks</p>'
            '</div>'
        )

    priority_colors = {
        "critical": "#f7768e",
        "high": "#e0af68",
        "medium": "#7aa2f7",
        "low": "#8b949e",
    }
    target_icons = {
        "review": "\U0001f50d",
        "fix": "\U0001f527",
        "docs": "\U0001f4dd",
        "research": "\U0001f52c",
    }

    cards: list[str] = []
    for task in tasks:
        tid = _html_esc(task.get("id", ""))
        title = _html_esc(task.get("title", "Untitled"))
        desc = _html_esc(task.get("description", ""))
        desc_short = desc[:200] + ("\u2026" if len(desc) > 200 else "")
        priority = task.get("priority", "medium")
        target = task.get("target", "fix")
        p_color = priority_colors.get(priority, "#8b949e")
        t_icon = target_icons.get(target, "\U0001f4cb")
        created = _relative_time(task.get("created", ""))
        tip = _tooltip_esc(task.get("description", ""))

        cards.append(
            f'<div class="staging-card" data-tooltip="{tip}">'
            f'<div class="staging-header">'
            f'<span class="staging-priority" style="background:{p_color}25;color:{p_color};'
            f'border:1px solid {p_color}50">{_html_esc(priority.upper())}</span>'
            f'<span class="staging-target">{t_icon} {_html_esc(target)}</span>'
            f'<span class="staging-id">{tid}</span>'
            f'<span class="staging-time">{_html_esc(created)}</span>'
            f'</div>'
            f'<div class="staging-title">{title}</div>'
            f'<div class="staging-desc">{desc_short}</div>'
            f'</div>'
        )

    count = len(tasks)
    header_detail = f'<span class="section-count">{count}</span>'

    return (
        f'<div class="section" id="panel-staging">'
        f'<h2 class="section-title">\U0001f4cb Staging Queue {header_detail}</h2>'
        f'<div class="staging-list">'
        + "".join(cards)
        + '</div></div>'
    )


def _render_all_panels(data: dict) -> dict:
    """Render all panels and return as dict for JSON API."""
    return {
        "health_bar": _render_health_bar(data),
        "coordinators": _render_coordinators(data),
        "agents": _render_agents(data),
        "staging": _render_staging(),
        "active_tasks": _render_active_tasks(data),
        "triggers": _render_triggers(data),
        "research": _render_research(data),
        "activity": _render_activity_feed(data),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "overall_status": _compute_overall_status(data),
    }


# -- HTML Template + Error Page -----------------------------------------------

_BUS_ERROR_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>War Room — Offline</title>
<style>
:root { --bg:#0d1117; --fg:#c9d1d9; --border:#30363d; }
body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       display:flex; align-items:center; justify-content:center; min-height:100vh;
       background:var(--bg); color:var(--fg); margin:0; }
.error { text-align:center; max-width:400px; }
.error h1 { font-size:3rem; margin:0; }
.error h2 { color:#f7768e; margin:0.5rem 0; }
.error p { color:#8b949e; line-height:1.6; }
.error code { background:#161b22; padding:0.2em 0.5em; border-radius:3px;
              border:1px solid var(--border); color:#58a6ff; }
</style>
</head>
<body>
<div class="error">
  <h1>\U0001f534</h1>
  <h2>Bus Daemon Offline</h2>
  <p>Cannot connect to the event bus. Start it with:</p>
  <p><code>sfs_events.py serve</code></p>
  <p>Then refresh this page.</p>
</div>
</body>
</html>"""


_BUS_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>War Room \u2014 __PROJECT__</title>
<style>
:root {
    --bg:      #0d1117;
    --fg:      #c9d1d9;
    --accent:  #58a6ff;
    --border:  #30363d;
    --code-bg: #161b22;
    --muted:   #8b949e;
    --bright:  #e6edf3;
}
*, *::before, *::after { box-sizing:border-box; }
body {
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    background:var(--bg); color:var(--fg); margin:0; padding:1.25rem;
    line-height:1.5;
    display:flex; flex-direction:column; height:100vh; overflow-y:auto; box-sizing:border-box;
    scrollbar-width:thin; scrollbar-color:var(--border) transparent;
}
header {
    display:flex; align-items:baseline; gap:1rem;
    margin-bottom:1.25rem; padding-bottom:0.75rem;
    border-bottom:1px solid var(--border);
}
header h1 { margin:0; font-size:1.3rem; color:var(--bright); }
header .meta { font-size:0.8rem; color:var(--muted); }
#slot-activity { flex:1; display:flex; flex-direction:column; min-height:400px; }
.footer { text-align:center; padding:0.5rem 0 0; font-size:0.75rem; color:var(--muted); flex-shrink:0; }

/* Status colors */
.status-green  { color:#3fb950; }
.status-amber  { color:#e0af68; }
.status-red    { color:#f7768e; }
.status-grey   { color:#8b949e; }

/* Tooltip system */
[data-tooltip] { position:relative; cursor:default; }
[data-tooltip]::after {
    content:attr(data-tooltip);
    position:absolute; top:100%; left:50%; transform:translateX(-50%);
    background:#1c2128; color:#c9d1d9; padding:0.5rem 0.75rem;
    border-radius:6px; border:1px solid #30363d;
    font-size:0.78rem; line-height:1.4; white-space:pre-line;
    z-index:100; pointer-events:none;
    opacity:0; transition:opacity 0.15s ease-in;
    min-width:200px; max-width:350px;
    box-shadow:0 4px 12px rgba(0,0,0,0.4);
    margin-top:4px;
}
[data-tooltip]:hover::after { opacity:1; }

/* Health bar */
.health-bar {
    display:flex; align-items:center; gap:1.25rem;
    padding:0.75rem 1.25rem; border-radius:8px;
    margin-bottom:1.25rem; border:1px solid var(--border);
}
.health-green { background:#0d2818; border-color:#1a4d2e; }
.health-amber { background:#2d2000; border-color:#4d3800; }
.health-red   { background:#2d1520; border-color:#5c1a2a; }
.health-dot {
    width:10px; height:10px; border-radius:50%;
    display:inline-block;
}
.health-green .health-dot { background:#3fb950; box-shadow:0 0 6px #3fb950; }
.health-amber .health-dot { background:#e0af68; box-shadow:0 0 6px #e0af68; }
.health-red   .health-dot { background:#f7768e; box-shadow:0 0 6px #f7768e; }
.health-label { font-weight:600; font-size:0.95rem; color:var(--bright); }
.health-uptime { font-size:0.85rem; color:var(--muted); margin-left:auto; }

/* Connection lost indicator */
.conn-lost {
    display:none; align-items:center; gap:0.5rem;
    padding:0.4rem 0.8rem; border-radius:6px;
    background:#2d1520; border:1px solid #5c1a2a;
    font-size:0.8rem; color:#f7768e; margin-left:auto;
}
.conn-lost.visible { display:flex; }

/* Section containers */
.section-row {
    display:grid; grid-template-columns:1fr 1.5fr;
    gap:1.25rem; margin-bottom:1.25rem;
}
.section {
    background:var(--code-bg); border:1px solid var(--border);
    border-radius:8px; padding:1rem;
}
.collapse-btn {
    position:absolute; right:0; top:50%; transform:translateY(-50%);
    background:none; border:1px solid var(--border); border-radius:4px;
    color:var(--muted); font-size:0.85rem; width:1.4rem; height:1.4rem;
    cursor:pointer; display:flex; align-items:center; justify-content:center;
    line-height:1; padding:0; transition:color 0.15s, border-color 0.15s;
}
.collapse-btn:hover { color:var(--bright); border-color:var(--bright); }
.section.collapsed > *:not(.section-title) { display:none !important; }
.section.collapsed { padding-bottom:0.25rem; }
.section-title {
    font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;
    color:var(--muted); margin:0 0 0.75rem 0;
}
.empty { color:var(--muted); font-size:0.85rem; font-style:italic; margin:0.5rem 0; }

/* Coordinator cards */
.coord-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; }
.coord-card {
    display:flex; align-items:center; gap:0.5rem;
    padding:0.5rem 0.75rem; border-radius:6px;
    background:var(--bg); border:1px solid var(--border);
}
.coord-card.coord-running { border-left:3px solid #3fb950; }
.coord-card.coord-working { border-left:3px solid #58a6ff; background:#101d2e; }
.coord-card.coord-missing { border-left:3px solid #f7768e; opacity:0.6; }
.coord-card.coord-retired { border-left:3px solid #8b949e; opacity:0.5; }
.coord-dot { font-size:0.6rem; line-height:1; }
.coord-name { font-weight:600; font-size:0.85rem; color:var(--bright); }
.coord-state {
    font-size:0.75rem; color:var(--muted);
    margin-left:auto; overflow:hidden;
    text-overflow:ellipsis; white-space:nowrap; max-width:220px;
}

/* Agent cards */
.agent-grid {
    display:grid; grid-template-columns:repeat(auto-fill, minmax(160px, 1fr));
    gap:0.5rem;
}
.agent-card {
    display:flex; align-items:center; gap:0.5rem;
    padding:0.4rem 0.65rem; border-radius:6px;
    background:var(--bg); border:1px solid var(--border);
}
.agent-card.agent-running { border-left:2px solid #3fb950; }
.agent-card.agent-active { border-left:2px solid #58a6ff; background:#101d2e; }
.agent-card.agent-retired { border-left:2px solid #f7768e; opacity:0.5; }
.agent-dot { font-size:0.5rem; line-height:1; }
.agent-name { font-size:0.8rem; color:var(--fg); }


/* Activity split layout */
.tasks-section { margin-bottom:1.25rem; }
.activity-row { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
.activity-subtitle { font-size:0.8rem; color:var(--muted); margin:0 0 0.5rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; }

/* Current activity cards (left) */
.tasks-list { display:flex; flex-direction:column; gap:0.5rem; }
.task-card {
    display:flex; flex-direction:column; gap:0.25rem;
    padding:0.6rem 0.85rem; border-radius:6px;
    background:var(--bg); border:1px solid var(--border);
    border-left:3px solid var(--border);
}
.task-card.task-active  { border-left-color:#58a6ff; background:#101d2e; }
.task-card.task-working { border-left-color:#e0af68; background:#1d1a0e; }
.task-card.task-stale   { border-left-color:#f7768e; background:#2d1520; animation:stale-pulse 2s ease-in-out infinite; }
@keyframes stale-pulse { 0%,100%{opacity:1} 50%{opacity:0.7} }
.task-header { display:flex; align-items:center; gap:0.5rem; }
.task-dot { font-size:0.55rem; line-height:1; flex-shrink:0; }
.task-title {
    font-weight:600; font-size:0.85rem; color:var(--bright);
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    flex:1; min-width:0;
}
.task-status {
    font-size:0.75rem; color:var(--muted);
    margin-left:auto; white-space:nowrap; flex-shrink:0;
}
.task-meta { font-size:0.75rem; color:var(--muted); padding-left:1.1rem; }
.task-workers { display:inline-flex; gap:0.5rem; margin-left:0.75rem; }
.task-worker { display:inline-flex; align-items:center; gap:0.25rem; font-size:0.7rem; }

/* Completed activity list (right) */
.completed-list { display:flex; flex-direction:column; gap:0.25rem; }
.completed-row {
    display:grid; grid-template-columns:1.25rem 1fr auto auto;
    align-items:center; gap:0.5rem;
    padding:0.35rem 0.6rem; border-radius:4px;
    background:var(--bg); border:1px solid var(--border);
    font-size:0.8rem; opacity:0.75;
}
.completed-row:hover { opacity:1; }
.completed-icon { font-size:0.75rem; text-align:center; }
.completed-title {
    color:var(--fg); overflow:hidden;
    text-overflow:ellipsis; white-space:nowrap;
}
.completed-coord { color:var(--muted); font-size:0.75rem; white-space:nowrap; }
.completed-runtime { color:var(--muted); font-size:0.75rem; font-family:monospace; white-space:nowrap; }

@media (max-width: 900px) {
    .activity-row { grid-template-columns:1fr; }
}

/* Trigger strip */
.trigger-section { margin-bottom:1.25rem; }
.trigger-list { display:flex; flex-wrap:wrap; gap:0.5rem; }
.trigger-row {
    display:inline-flex; align-items:center; gap:0.4rem;
    padding:0.35rem 0.65rem; border-radius:6px;
    background:var(--bg); border:1px solid var(--border);
    font-size:0.8rem;
}
.trigger-type { font-size:0.85rem; }
.trigger-channel { font-family:monospace; font-size:0.75rem; color:var(--fg); }
.trigger-dot { font-size:0.45rem; }

/* Research pipeline */
.research-section { margin-bottom:1.25rem; }
.research-list {
    display:flex; flex-direction:column; gap:0.5rem;
    max-height:300px; overflow-y:auto;
    scrollbar-width:thin; scrollbar-color:var(--border) transparent;
}
.research-card {
    display:flex; align-items:center; gap:0.75rem;
    padding:0.5rem 0.75rem; border-radius:6px;
    background:var(--bg); border:1px solid var(--border);
    border-left:3px solid #3fb950;
}
.research-phase {
    display:inline-block; padding:0.15rem 0.5rem; border-radius:3px;
    font-size:0.7rem; font-weight:600; text-transform:uppercase;
    letter-spacing:0.03em; min-width:80px; text-align:center;
}
.phase-researching { background:#1a2332; color:#58a6ff; }
.phase-documenting { background:#1e1533; color:#bc8cff; }
.phase-rfi         { background:#2d2000; color:#e0af68; }
.phase-complete    { background:#0d2818; color:#3fb950; }
.research-topic { font-size:0.85rem; color:var(--bright); flex:1; }
.research-findings { font-size:0.8rem; color:var(--muted); }

/* Activity feed */
.feed-section { margin-bottom:0; flex:1; display:flex; flex-direction:column; min-height:350px; }
.feed-count { font-size:0.7rem; color:var(--muted); font-weight:normal; }
.feed-scroll {
    flex:1; overflow-y:auto; min-height:0;
    scrollbar-width:thin; scrollbar-color:var(--border) transparent;
    border:1px solid var(--border); border-radius:6px;
    background:var(--bg);
}
.feed-entry {
    display:flex; align-items:center; gap:0.65rem;
    padding:0.35rem 0.65rem;
    border-bottom:1px solid var(--border);
    font-size:0.8rem;
}
.feed-entry:last-child { border-bottom:none; }
.feed-entry:hover { background:var(--code-bg); }
.feed-time { min-width:55px; color:var(--muted); font-size:0.75rem; }
.feed-channel {
    display:inline-block; padding:0.1rem 0.4rem; border-radius:3px;
    font-size:0.7rem; font-weight:600;
    min-width:70px; text-align:center; white-space:nowrap;
}
.feed-summary {
    flex:1; overflow:hidden; text-overflow:ellipsis;
    white-space:nowrap; color:var(--fg);
}


/* Feed toggle */
.feed-toggle { display:flex; gap:0.25rem; }
.toggle-btn {
    background:var(--code-bg); border:1px solid var(--border); color:var(--muted);
    padding:0.2rem 0.7rem; border-radius:4px; font-size:0.75rem; cursor:pointer;
    transition:all 0.15s;
}
.toggle-btn.active { background:#1f6feb20; color:#58a6ff; border-color:#1f6feb50; }
.toggle-btn .feed-count { margin-left:0.3rem; font-size:0.65rem; opacity:0.7; }
.feed-view { flex:1; display:flex; flex-direction:column; min-height:0; }

/* Intel cards */
.intel-scroll {
    flex:1; overflow-y:auto; min-height:0;
    scrollbar-width:thin; scrollbar-color:var(--border) transparent;
}
.intel-entry {
    padding:0.6rem 0.7rem; border-bottom:1px solid var(--border);
    transition:background 0.1s;
}
.intel-entry:hover { background:var(--code-bg); }
.intel-header { display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem; }
.intel-icon { font-size:0.85rem; }
.intel-source { font-weight:600; color:var(--bright); font-size:0.8rem; }
.intel-relevance {
    display:inline-block; padding:0.05rem 0.35rem; border-radius:3px;
    font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.03em;
}
.intel-time { margin-left:auto; color:var(--muted); font-size:0.7rem; }
.intel-title { color:var(--fg); font-size:0.85rem; font-weight:500; margin-bottom:0.2rem; }
.intel-analysis { color:var(--muted); font-size:0.78rem; line-height:1.4; margin-bottom:0.25rem; }
.intel-meta { display:flex; align-items:center; gap:0.7rem; flex-wrap:wrap; }
.intel-urls { font-size:0.7rem; color:var(--muted); }
.intel-link { color:#58a6ff; text-decoration:none; font-size:0.7rem; }
.intel-link:hover { text-decoration:underline; }
.intel-followup {
    font-size:0.73rem; color:#3fb950; font-style:italic;
}
/* Staging Queue */
.staging-list { display:flex; flex-direction:column; gap:0.4rem; padding:0.5rem; }
.staging-card {
    background:var(--card-bg); border:1px solid var(--border); border-radius:6px;
    padding:0.6rem 0.7rem; transition:border-color 0.15s;
}
.staging-card:hover { border-color:var(--accent); }
.staging-header { display:flex; align-items:center; gap:0.5rem; margin-bottom:0.3rem; }
.staging-priority {
    display:inline-block; padding:0.05rem 0.35rem; border-radius:3px;
    font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.03em;
}
.staging-target { font-size:0.8rem; color:var(--muted); }
.staging-id { font-size:0.7rem; color:var(--muted); font-family:var(--mono); }
.staging-time { margin-left:auto; color:var(--muted); font-size:0.7rem; }
.staging-title { color:var(--bright); font-size:0.85rem; font-weight:500; margin-bottom:0.15rem; }
.staging-desc { color:var(--muted); font-size:0.78rem; line-height:1.4; }
.section-count {
    display:inline-block; background:var(--accent); color:#000; border-radius:10px;
    padding:0 0.4rem; font-size:0.7rem; font-weight:700; margin-left:0.4rem;
}
/* Refresh pulse animation */
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.3; } }
.refreshing .health-dot { animation:pulse 0.6s ease-in-out; }

/* Responsive */
@media (max-width: 900px) {
    .section-row { grid-template-columns:1fr; }
    .coord-grid { grid-template-columns:1fr; }
}
</style>
</head>
<body>
    <header>
        <h1>\u2694 War Room</h1>
        <span class="meta">__PROJECT__ \u2022 <span id="generated">__GENERATED__</span></span>
        <div id="conn-lost" class="conn-lost">\u26a0 Connection lost</div>
    </header>

    <div id="slot-health">__HEALTH_BAR__</div>

    <div class="section-row">
        <div id="slot-coordinators">__COORDINATORS__</div>
        <div id="slot-agents">__AGENTS__</div>
    </div>

    <div id="slot-staging">__STAGING__</div>

    <div id="slot-active-tasks">__ACTIVE_TASKS__</div>

    <div id="slot-triggers">__TRIGGERS__</div>

    <div id="slot-research">__RESEARCH__</div>

    <div id="slot-activity">__ACTIVITY__</div>

    <footer class="footer">
        Generated by sfs_events.py \u2022 __REFRESH_LABEL__
    </footer>

    <script>
    let currentFeedView = 'intel';
    const collapsedSections = {};

    function toggleFeed(view) {
        currentFeedView = view;
        const intelEl = document.getElementById('feed-intel');
        const debugEl = document.getElementById('feed-debug');
        if (intelEl) intelEl.style.display = view === 'intel' ? '' : 'none';
        if (debugEl) debugEl.style.display = view === 'debug' ? '' : 'none';
        document.querySelectorAll('.toggle-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === view);
        });
    }

    function toggleSection(panelId) {
        collapsedSections[panelId] = !collapsedSections[panelId];
        applyCollapsed();
    }

    function applyCollapsed() {
        document.querySelectorAll('.section[id^="panel-"]').forEach(panel => {
            const id = panel.id;
            const btn = panel.querySelector('.collapse-btn');
            if (collapsedSections[id]) {
                panel.classList.add('collapsed');
                if (btn) btn.textContent = '+';
            } else {
                panel.classList.remove('collapsed');
                if (btn) btn.textContent = '\u2212';
            }
        });
    }

    function injectCollapseButtons() {
        document.querySelectorAll('.section[id^="panel-"]').forEach(panel => {
            const title = panel.querySelector('.section-title');
            if (!title || title.querySelector('.collapse-btn')) return;
            const btn = document.createElement('button');
            btn.className = 'collapse-btn';
            btn.textContent = collapsedSections[panel.id] ? '+' : '\u2212';
            btn.onclick = function(e) { e.stopPropagation(); toggleSection(panel.id); };
            title.style.position = 'relative';
            title.appendChild(btn);
        });
        applyCollapsed();
    }

    document.addEventListener('DOMContentLoaded', injectCollapseButtons);
    </script>

    __POLL_SCRIPT__
</body>
</html>"""


_POLL_SCRIPT = """<script>
(function() {
    const POLL_MS = 5000;
    const connLost = document.getElementById('conn-lost');
    const body = document.body;
    let failures = 0;

    async function refresh() {
        try {
            body.classList.add('refreshing');
            const resp = await fetch('/data');
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            const data = await resp.json();

            document.getElementById('slot-health').innerHTML = data.health_bar;
            document.getElementById('slot-coordinators').innerHTML = data.coordinators;
            document.getElementById('slot-agents').innerHTML = data.agents;
            document.getElementById('slot-staging').innerHTML = data.staging;
            document.getElementById('slot-active-tasks').innerHTML = data.active_tasks;
            document.getElementById('slot-triggers').innerHTML = data.triggers;
            document.getElementById('slot-research').innerHTML = data.research;
            document.getElementById('slot-activity').innerHTML = data.activity;
            toggleFeed(currentFeedView);
            injectCollapseButtons();
            document.getElementById('generated').textContent = data.generated;

            connLost.classList.remove('visible');
            failures = 0;
        } catch(e) {
            failures++;
            if (failures >= 2) {
                connLost.classList.add('visible');
            }
        } finally {
            body.classList.remove('refreshing');
        }
    }

    setInterval(refresh, POLL_MS);
})();
</script>"""


# -- Dashboard generation + serve ---------------------------------------------


def _generate_dashboard_html(
    data: dict | None,
    *,
    live: bool = False,
) -> str:
    """Render complete bus ops dashboard from collected data."""
    if data is None:
        return _BUS_ERROR_TEMPLATE

    project = Path.cwd().name
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html = _BUS_DASHBOARD_TEMPLATE
    html = html.replace("__PROJECT__", _html_esc(project))
    html = html.replace("__GENERATED__", generated)
    html = html.replace("__HEALTH_BAR__", _render_health_bar(data))
    html = html.replace("__COORDINATORS__", _render_coordinators(data))
    html = html.replace("__AGENTS__", _render_agents(data))
    html = html.replace("__STAGING__", _render_staging())
    html = html.replace("__ACTIVE_TASKS__", _render_active_tasks(data))
    html = html.replace("__TRIGGERS__", _render_triggers(data))
    html = html.replace("__RESEARCH__", _render_research(data))
    html = html.replace("__ACTIVITY__", _render_activity_feed(data))

    if live:
        html = html.replace("__POLL_SCRIPT__", _POLL_SCRIPT)
        html = html.replace("__REFRESH_LABEL__", "Live \u2022 updates every 5s")
    else:
        html = html.replace("__POLL_SCRIPT__", "")
        html = html.replace("__REFRESH_LABEL__", "Refresh to update")

    return html


def _dashboard_impl(
    auto_open: bool = True,
    *,
    socket_path: str = DEFAULT_SOCKET,
) -> tuple[dict, dict]:
    """Generate static bus ops dashboard. CLI: dashboard, MCP: dashboard."""
    start_ms = time.time() * 1000

    data = _collect_dashboard_data(socket_path)
    html = _generate_dashboard_html(data, live=False)

    _BUS_DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    _BUS_DASHBOARD_PATH.write_text(html, encoding="utf-8")

    if auto_open:
        import webbrowser
        webbrowser.open(f"file://{_BUS_DASHBOARD_PATH}")

    latency_ms = time.time() * 1000 - start_ms
    _log("INFO", "dashboard", f"html={_BUS_DASHBOARD_PATH}")

    result = {
        "html_file": str(_BUS_DASHBOARD_PATH),
        "data_sources": 9 if data else 0,
        "agents": len(data["agents"].get("agents", [])) if data else 0,
        "triggers": len(data["triggers"].get("triggers", [])) if data else 0,
    }
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}
    return result, metrics


def _dashboard_serve_impl(
    port: int = _DASHBOARD_PORT,
    *,
    socket_path: str = DEFAULT_SOCKET,
) -> None:
    """Start live dashboard HTTP server. CLI: dashboard serve."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class DashboardHandler(BaseHTTPRequestHandler):
        """Serves dashboard HTML and JSON data API."""

        def do_GET(self) -> None:
            if self.path == "/" or self.path == "":
                self._serve_html()
            elif self.path == "/data":
                self._serve_data()
            elif self.path == "/health":
                self._respond(200, "application/json", b'{"ok":true}')
            else:
                self._respond(404, "text/plain", b"Not found")

        def _serve_html(self) -> None:
            data = _collect_dashboard_data(socket_path)
            html = _generate_dashboard_html(data, live=True)
            self._respond(200, "text/html; charset=utf-8", html.encode("utf-8"))

        def _serve_data(self) -> None:
            data = _collect_dashboard_data(socket_path)
            if data is None:
                self._respond(503, "application/json", b'{"error":"bus offline"}')
                return
            panels = _render_all_panels(data)
            body = json.dumps(panels).encode("utf-8")
            self._respond(200, "application/json", body)

        def _respond(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            # Suppress default stderr logging; use our TSV logger
            _log("TRACE", "dashboard_http", fmt % args)

    server = HTTPServer(("127.0.0.1", port), DashboardHandler)
    _log("INFO", "dashboard_serve", f"port={port}")
    print(f"\u2694  War Room live at http://127.0.0.1:{port}", file=sys.stderr)
    print(f"   Press Ctrl+C to stop", file=sys.stderr)

    import webbrowser
    webbrowser.open(f"http://127.0.0.1:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server...", file=sys.stderr)
        server.shutdown()



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

    # stage
    p_stage = sub.add_parser("stage", help="Stage a task for team execution")
    p_stage.add_argument("-t", "--title", required=True, help="Task title")
    p_stage.add_argument("-d", "--description", default="", help="Task description")
    p_stage.add_argument(
        "-p", "--priority", default="medium",
        choices=list(_VALID_PRIORITIES), help="Priority",
    )
    p_stage.add_argument(
        "-T", "--target", default="fix",
        choices=list(_VALID_TARGETS), help="Target coordinator",
    )

    # staged
    p_staged = sub.add_parser("staged", help="List staged tasks")
    p_staged.add_argument(
        "-a", "--all", action="store_true", dest="include_released",
        help="Include released and pulled tasks",
    )

    # refine
    p_refine = sub.add_parser("refine", help="Refine a staged task")
    p_refine.add_argument("task_id", help="Task ID to refine")
    p_refine.add_argument("-t", "--title", default="", help="New title")
    p_refine.add_argument("-d", "--description", default="", help="New description")
    p_refine.add_argument("-p", "--priority", default="", help="New priority")
    p_refine.add_argument("-T", "--target", default="", help="New target coordinator")

    # release
    p_release = sub.add_parser("release", help="Release staged task to coordinator")
    p_release.add_argument("task_id", help="Task ID to release")
    p_release.add_argument("-s", "--socket", default=DEFAULT_SOCKET)

    # pull
    p_pull = sub.add_parser("pull", help="Pull staged task for direct work")
    p_pull.add_argument("task_id", help="Task ID to pull")

    # drop
    p_drop = sub.add_parser("drop", help="Drop (cancel) a staged task")
    p_drop.add_argument("task_id", help="Task ID to drop")

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Generate bus ops dashboard")
    p_dash.add_argument("mode", nargs="?", default="static", choices=["static", "serve"],
                        help="static (default) or serve (live HTTP server)")
    p_dash.add_argument("-n", "--no-open", action="store_true", help="Do not open browser")
    p_dash.add_argument("-p", "--port", type=int, default=_DASHBOARD_PORT,
                        help=f"Port for serve mode (default {_DASHBOARD_PORT})")
    p_dash.add_argument("-s", "--socket", default=DEFAULT_SOCKET)

    # mcp-stdio
    sub.add_parser("mcp-stdio", help="Run as MCP server")

    args = parser.parse_args()

    # MCP dispatch — must come before any stdin reads
    if args.command == "mcp-stdio":
        _run_mcp()
        sys.exit(0)

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
        if args.command == "serve":
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

        elif args.command == "stage":
            result, _ = _stage_impl(
                args.title, args.description, args.priority, args.target,
            )
            print(json.dumps(result, indent=2, default=str))

        elif args.command == "staged":
            result, _ = _staged_impl(include_released=args.include_released)
            print(json.dumps(result, indent=2, default=str))

        elif args.command == "refine":
            result, _ = _refine_impl(
                args.task_id, args.title, args.description, args.priority, args.target,
            )
            print(json.dumps(result, indent=2, default=str))

        elif args.command == "release":
            result, _ = _release_impl(args.task_id, socket_path=args.socket)
            print(json.dumps(result, indent=2, default=str))

        elif args.command == "pull":
            result, _ = _pull_impl(args.task_id)
            print(json.dumps(result, indent=2, default=str))

        elif args.command == "drop":
            result, _ = _drop_impl(args.task_id)
            print(json.dumps(result, indent=2, default=str))

        elif args.command == "dashboard":
            if args.mode == "serve":
                _dashboard_serve_impl(
                    port=args.port,
                    socket_path=args.socket,
                )
            else:
                result, metrics = _dashboard_impl(
                    auto_open=not args.no_open,
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
        """Add a file, URL, or schedule trigger that publishes events to a channel.

        Args:
            trigger_type: "file" (watchdog), "url" (HTTP polling), or "schedule" (fixed interval)
            channel: Target channel (auto-scoped to project, use @ for absolute)
            config: JSON config — file: {"path": "/dir", "recursive": true, "events": ["modified","created"]}
                    url: {"url": "https://...", "interval_seconds": 300}
                    schedule: {"interval_seconds": 300, "label": "health-tick"}

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

    # MCP for _stage_impl
    @mcp.tool()
    def stage(
        title: str, description: str = "", priority: str = "medium", target: str = "fix"
    ) -> str:
        """Stage a task for team execution.

        Args:
            title: Short task title
            description: Detailed description (becomes coordinator prompt on release)
            priority: Task priority (critical, high, medium, low)
            target: Target coordinator (review, fix, docs, research)

        Returns:
            JSON with staged task details
        """
        result, _ = _stage_impl(title, description, priority, target)
        return json.dumps(result, indent=2, default=str)

    # MCP for _staged_impl
    @mcp.tool()
    def staged(include_released: bool = False) -> str:
        """List staged tasks sorted by priority.

        Args:
            include_released: Also show released and pulled tasks

        Returns:
            JSON list of staged tasks
        """
        result, _ = _staged_impl(include_released=include_released)
        return json.dumps(result, indent=2, default=str)

    # MCP for _refine_impl
    @mcp.tool()
    def refine(
        task_id: str,
        title: str = "",
        description: str = "",
        priority: str = "",
        target: str = "",
    ) -> str:
        """Refine a staged task.

        Args:
            task_id: ID of the staged task
            title: New title (empty to keep current)
            description: New description (empty to keep current)
            priority: New priority (empty to keep current)
            target: New target coordinator (empty to keep current)

        Returns:
            JSON with updated task details
        """
        result, _ = _refine_impl(task_id, title, description, priority, target)
        return json.dumps(result, indent=2, default=str)

    # MCP for _release_impl
    @mcp.tool()
    def release(task_id: str) -> str:
        """Release a staged task to its target coordinator for execution.

        Args:
            task_id: ID of the staged task to release

        Returns:
            JSON with release details including correlation_id
        """
        result, _ = _release_impl(task_id)
        return json.dumps(result, indent=2, default=str)

    # MCP for _pull_impl
    @mcp.tool()
    def pull(task_id: str) -> str:
        """Pull a staged task for direct work (exception — team execution is default).

        Args:
            task_id: ID of the staged task to pull

        Returns:
            JSON with pulled task details
        """
        result, _ = _pull_impl(task_id)
        return json.dumps(result, indent=2, default=str)

    # MCP for _drop_impl
    @mcp.tool()
    def drop(task_id: str) -> str:
        """Drop (cancel) a staged task.

        Args:
            task_id: ID of the staged task to drop

        Returns:
            JSON with dropped task details
        """
        result, _ = _drop_impl(task_id)
        return json.dumps(result, indent=2, default=str)

    # MCP for _dashboard_impl
    @mcp.tool()
    def dashboard(auto_open: bool = False) -> str:
        """Generate bus ops dashboard HTML.

        Args:
            auto_open: Open in browser after generation

        Returns:
            JSON with html_file path and dashboard metadata
        """
        result, _ = _dashboard_impl(auto_open=auto_open)
        return json.dumps(result, indent=2)

    print("Sanctum event bus MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
