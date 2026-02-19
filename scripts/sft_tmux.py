#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastmcp",
# ]
# ///
"""Tmux agent orchestration - spawn, control, and message tmux sessions.

Manage background agent sessions with safe pane injection that detects
human activity to avoid interference.

Usage:
    sft_tmux.py spawn [--id ID] [--cwd PATH] [COMMAND...]
    sft_tmux.py kill --id ID
    sft_tmux.py send --id ID TEXT
    sft_tmux.py read --id ID [--lines N]
    sft_tmux.py list [--json]
    sft_tmux.py msg --to ID [--from ID] MESSAGE
    sft_tmux.py inbox --id ID [--json]
    sft_tmux.py ack --id ID [--all|--msg MSG_ID]
    sft_tmux.py mcp-stdio
"""

import argparse
import hashlib
import json
import os
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
    "spawn",
    "kill",
    "list",
    "send",
    "type",
    "read",
    "pipe",
    "attach",
    "msg",
    "inbox",
    "ack",
    "pane_send",
    "pane_send_safe",
    "deliver",
    "drain",
]

CONFIG = {
    "agents_dir": Path.home() / ".sfb" / "agents",
    "version": "1.0.0",
    "pane_settle_ms": 250,
    "pane_sig_lines": 20,
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def _run_tmux(*args: str, capture: bool = True) -> tuple[int, str, str]:
    """Run tmux command. Returns (returncode, stdout, stderr)."""
    cmd = ["tmux"] + list(args)
    try:
        result = subprocess.run(cmd, capture_output=capture, text=True)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except FileNotFoundError:
        return 1, "", "tmux not found in PATH"


def _session_name(agent_id: str) -> str:
    """Convert agent ID to tmux session name."""
    return f"agent-{agent_id}"


def _agent_dir(agent_id: str) -> Path:
    """Get agent's working directory."""
    return CONFIG["agents_dir"] / agent_id


def _ensure_agent_dirs(agent_id: str) -> Path:
    """Ensure agent directory structure exists."""
    agent_dir = _agent_dir(agent_id)
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "artifacts").mkdir(exist_ok=True)
    (agent_dir / "inbox").mkdir(exist_ok=True)
    (agent_dir / "inbox" / "archive").mkdir(exist_ok=True)
    return agent_dir


def _pane_fmt(pane: str, fmt: str) -> tuple[bool, str]:
    """Read tmux pane format string."""
    code, stdout, stderr = _run_tmux("display-message", "-p", "-t", pane, fmt)
    if code == 0:
        return True, stdout.strip()
    return False, stderr or "Failed to read pane format"


def _pane_capture(pane: str, lines: int = 20) -> tuple[bool, str]:
    """Capture last N lines of a pane."""
    args = ["capture-pane", "-t", pane, "-p"]
    if lines and lines > 0:
        args.extend(["-S", str(-lines)])
    code, stdout, stderr = _run_tmux(*args)
    if code == 0:
        return True, stdout
    return False, stderr or "Failed to capture pane"


def _screen_sig(pane: str, lines: int = 20) -> tuple[bool, str]:
    """Fingerprint last N lines of a pane screen."""
    ok, text = _pane_capture(pane, lines=lines)
    if not ok:
        return False, text
    sig = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return True, sig


def _pane_is_safe_to_inject(
    pane: str,
    settle_ms: int = 250,
    sig_lines: int = 20,
    allow_active: bool = False,
) -> tuple[bool, str]:
    """Heuristic gate: avoid injecting into a pane while a human is typing."""
    ok, active = _pane_fmt(pane, "#{pane_active}")
    if ok and active.strip() == "1" and not allow_active:
        return False, "pane_active"

    ok, in_mode = _pane_fmt(pane, "#{pane_in_mode}")
    if ok and in_mode.strip() == "1":
        return False, "pane_in_mode"

    ok, sig1 = _screen_sig(pane, lines=sig_lines)
    if not ok:
        return False, f"capture_failed:{sig1}"
    time.sleep(max(0.05, settle_ms / 1000.0))
    ok, sig2 = _screen_sig(pane, lines=sig_lines)
    if not ok:
        return False, f"capture_failed:{sig2}"
    if sig1 != sig2:
        return False, "screen_changing"

    return True, "ok"


# =============================================================================
# AGENT LIFECYCLE
# =============================================================================

def _spawn_impl(
    agent_id: str | None = None,
    command: list[str] | None = None,
    cwd: str | None = None,
) -> tuple[bool, str, dict]:
    """Spawn new agent session. CLI: spawn, MCP: spawn.
    
    Returns (success, session_id_or_error, metrics).
    """
    start_ms = time.time() * 1000
    
    if not agent_id:
        agent_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    session = _session_name(agent_id)
    agent_dir = _ensure_agent_dirs(agent_id)
    (agent_dir / "status").write_text("running\n")

    work_dir = cwd or str(agent_dir)
    if cwd and not os.path.isdir(cwd):
        return False, f"Working directory does not exist: {cwd}", {"latency_ms": 0}

    args = ["new-session", "-d", "-s", session, "-c", work_dir]
    if command:
        args.extend(command)

    code, stdout, stderr = _run_tmux(*args)
    latency_ms = time.time() * 1000 - start_ms
    
    metrics = {
        "latency_ms": round(latency_ms, 2),
        "agent_id": agent_id,
        "status": "success" if code == 0 else "error",
    }

    if code == 0:
        return True, agent_id, metrics

    if "duplicate session" in stderr:
        return True, agent_id, metrics

    return False, stderr or "Failed to spawn agent", metrics


def _kill_impl(agent_id: str) -> tuple[bool, str, dict]:
    """Kill agent session. CLI: kill, MCP: kill.
    
    Returns (success, message, metrics).
    """
    start_ms = time.time() * 1000
    session = _session_name(agent_id)
    agent_dir = _agent_dir(agent_id)

    code, stdout, stderr = _run_tmux("kill-session", "-t", session)

    if agent_dir.exists():
        (agent_dir / "status").write_text("killed\n")

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"latency_ms": round(latency_ms, 2), "agent_id": agent_id}

    if code == 0:
        return True, f"Agent {agent_id} killed", metrics
    return False, stderr or "Failed to kill agent", metrics


def _list_impl() -> tuple[list[dict], dict]:
    """List all agent sessions. CLI: list, MCP: list.
    
    Returns (agents_list, metrics).
    """
    start_ms = time.time() * 1000
    code, stdout, stderr = _run_tmux(
        "list-sessions", "-F", "#{session_name}|#{session_created}|#{session_windows}"
    )

    agents = []
    if code == 0:
        for line in stdout.split("\n"):
            if line.startswith("agent-"):
                parts = line.split("|")
                aid = parts[0].replace("agent-", "")
                agent_dir = _agent_dir(aid)
                status = "unknown"
                if agent_dir.exists() and (agent_dir / "status").exists():
                    status = (agent_dir / "status").read_text().strip()

                inbox_count = 0
                inbox_dir = agent_dir / "inbox"
                if inbox_dir.exists():
                    inbox_count = len([f for f in inbox_dir.glob("*.msg")])

                agents.append({
                    "id": aid,
                    "session": parts[0],
                    "created": parts[1] if len(parts) > 1 else "",
                    "windows": parts[2] if len(parts) > 2 else "1",
                    "status": status,
                    "inbox": inbox_count,
                    "dir": str(agent_dir),
                })

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"latency_ms": round(latency_ms, 2), "count": len(agents)}
    return agents, metrics


# =============================================================================
# TERMINAL COMMUNICATION
# =============================================================================

def _send_impl(agent_id: str, text: str) -> tuple[bool, str, dict]:
    """Send command to agent (with Enter). CLI: send, MCP: send."""
    start_ms = time.time() * 1000
    session = _session_name(agent_id)

    code1, _, stderr1 = _run_tmux("send-keys", "-t", session, text)
    if code1 != 0:
        latency_ms = time.time() * 1000 - start_ms
        return False, stderr1 or "Failed to send", {"latency_ms": round(latency_ms, 2)}

    code2, _, stderr2 = _run_tmux("send-keys", "-t", session, "Enter")
    latency_ms = time.time() * 1000 - start_ms
    metrics = {"latency_ms": round(latency_ms, 2), "agent_id": agent_id}

    if code2 != 0:
        return False, stderr2 or "Failed to send Enter", metrics
    return True, "Sent", metrics


def _type_impl(agent_id: str, text: str, literal: bool = False) -> tuple[bool, str, dict]:
    """Type raw text to agent (no Enter). CLI: type, MCP: type."""
    start_ms = time.time() * 1000
    session = _session_name(agent_id)

    args = ["send-keys", "-t", session]
    if literal:
        args.append("-l")
    args.append(text)

    code, _, stderr = _run_tmux(*args)
    latency_ms = time.time() * 1000 - start_ms
    metrics = {"latency_ms": round(latency_ms, 2), "agent_id": agent_id}

    if code == 0:
        return True, "Typed", metrics
    return False, stderr or "Failed to type", metrics


def _read_impl(agent_id: str, lines: int | None = None, save: bool = False) -> tuple[bool, str, dict]:
    """Read agent output. CLI: read, MCP: read."""
    start_ms = time.time() * 1000
    session = _session_name(agent_id)

    args = ["capture-pane", "-t", session, "-p"]
    if lines:
        args.extend(["-S", str(-lines)])

    code, stdout, stderr = _run_tmux(*args)
    latency_ms = time.time() * 1000 - start_ms
    metrics = {"latency_ms": round(latency_ms, 2), "agent_id": agent_id, "lines": lines}

    if code != 0:
        return False, stderr or "Failed to read agent", metrics

    if save:
        agent_dir = _agent_dir(agent_id)
        if agent_dir.exists():
            log_file = agent_dir / "output.log"
            with open(log_file, "a") as f:
                f.write(f"\n--- {datetime.now().isoformat()} ---\n")
                f.write(stdout)
                f.write("\n")

    return True, stdout, metrics


def _pipe_impl(agent_id: str, filepath: str) -> tuple[bool, str, dict]:
    """Pipe file contents to agent. CLI: pipe, MCP: pipe."""
    start_ms = time.time() * 1000
    session = _session_name(agent_id)

    try:
        content = Path(filepath).read_text()
    except Exception as e:
        return False, f"Failed to read file: {e}", {"latency_ms": 0}

    code1, _, stderr1 = _run_tmux("load-buffer", "-b", "pipe", filepath)
    if code1 != 0:
        latency_ms = time.time() * 1000 - start_ms
        return False, stderr1 or "Failed to load buffer", {"latency_ms": round(latency_ms, 2)}

    code2, _, stderr2 = _run_tmux("paste-buffer", "-b", "pipe", "-t", session)
    latency_ms = time.time() * 1000 - start_ms
    metrics = {"latency_ms": round(latency_ms, 2), "bytes": len(content)}

    if code2 != 0:
        return False, stderr2 or "Failed to paste buffer", metrics
    return True, f"Piped {len(content)} bytes", metrics


def _attach_impl(agent_id: str) -> tuple[bool, str, dict]:
    """Attach to agent session. CLI: attach."""
    start_ms = time.time() * 1000
    session = _session_name(agent_id)
    code = subprocess.call(["tmux", "attach-session", "-t", session])
    latency_ms = time.time() * 1000 - start_ms
    metrics = {"latency_ms": round(latency_ms, 2), "agent_id": agent_id}

    if code == 0:
        return True, "Detached", metrics
    return False, f"Failed to attach (code {code})", metrics


# =============================================================================
# MESSAGE BROKER
# =============================================================================

def _post_message_impl(
    to_agent: str, message: str, from_agent: str = "sft_tmux"
) -> tuple[bool, str, dict]:
    """Post message to agent's inbox. CLI: msg, MCP: post_message."""
    start_ms = time.time() * 1000
    agent_dir = _ensure_agent_dirs(to_agent)
    inbox_dir = agent_dir / "inbox"

    ts = datetime.now()
    msg_id = ts.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    msg_file = inbox_dir / f"{msg_id}.msg"

    content = {
        "id": msg_id,
        "from": from_agent,
        "to": to_agent,
        "timestamp": ts.isoformat(),
        "message": message,
    }

    try:
        msg_file.write_text(json.dumps(content, indent=2))
        latency_ms = time.time() * 1000 - start_ms
        metrics = {"latency_ms": round(latency_ms, 2), "msg_id": msg_id}
        return True, msg_id, metrics
    except Exception as e:
        latency_ms = time.time() * 1000 - start_ms
        return False, f"Failed to write message: {e}", {"latency_ms": round(latency_ms, 2)}


def _read_inbox_impl(agent_id: str) -> tuple[list[dict], dict]:
    """Read agent's inbox. CLI: inbox, MCP: read_inbox."""
    start_ms = time.time() * 1000
    agent_dir = _agent_dir(agent_id)
    inbox_dir = agent_dir / "inbox"

    messages = []
    if inbox_dir.exists():
        for msg_file in sorted(inbox_dir.glob("*.msg")):
            try:
                content = json.loads(msg_file.read_text())
                messages.append(content)
            except Exception:
                continue

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"latency_ms": round(latency_ms, 2), "count": len(messages)}
    return messages, metrics


def _ack_impl(
    agent_id: str, msg_id: str | None = None, ack_all: bool = False
) -> tuple[bool, str, dict]:
    """Acknowledge messages. CLI: ack, MCP: ack_messages."""
    start_ms = time.time() * 1000
    agent_dir = _agent_dir(agent_id)
    inbox_dir = agent_dir / "inbox"
    archive_dir = inbox_dir / "archive"

    if not inbox_dir.exists():
        latency_ms = time.time() * 1000 - start_ms
        return True, "No messages to ack", {"latency_ms": round(latency_ms, 2), "count": 0}

    archive_dir.mkdir(exist_ok=True)
    count = 0

    if ack_all:
        for msg_file in inbox_dir.glob("*.msg"):
            msg_file.rename(archive_dir / msg_file.name)
            count += 1
    elif msg_id:
        msg_file = inbox_dir / f"{msg_id}.msg"
        if msg_file.exists():
            msg_file.rename(archive_dir / msg_file.name)
            count = 1
        else:
            latency_ms = time.time() * 1000 - start_ms
            return False, f"Message {msg_id} not found", {"latency_ms": round(latency_ms, 2), "count": 0}
    else:
        latency_ms = time.time() * 1000 - start_ms
        return False, "Specify --all or --msg MSG_ID", {"latency_ms": round(latency_ms, 2), "count": 0}

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"latency_ms": round(latency_ms, 2), "count": count}
    return True, f"Acknowledged {count} message(s)", metrics


# =============================================================================
# SAFE PANE INJECTION
# =============================================================================

def _pane_send_impl(pane: str, text: str, enter: bool = False) -> tuple[bool, str, dict]:
    """Send keys to tmux pane (unsafe). CLI: pane-send, MCP: pane_send."""
    start_ms = time.time() * 1000
    code, _, stderr = _run_tmux("send-keys", "-t", pane, text)
    if code != 0:
        latency_ms = time.time() * 1000 - start_ms
        return False, stderr or "Failed to send", {"latency_ms": round(latency_ms, 2)}

    if enter:
        code2, _, stderr2 = _run_tmux("send-keys", "-t", pane, "Enter")
        if code2 != 0:
            latency_ms = time.time() * 1000 - start_ms
            return False, stderr2 or "Failed to send Enter", {"latency_ms": round(latency_ms, 2)}

    latency_ms = time.time() * 1000 - start_ms
    return True, "Sent", {"latency_ms": round(latency_ms, 2), "pane": pane}


def _pane_send_safe_impl(
    pane: str,
    text: str,
    timeout_sec: float = 30.0,
    settle_ms: int = 250,
    sig_lines: int = 20,
    allow_active: bool = False,
) -> tuple[bool, str, dict]:
    """Send keys to pane only when safe. CLI: pane-send-safe, MCP: pane_send_safe."""
    start_ms = time.time() * 1000
    deadline = time.time() + max(0.0, timeout_sec)
    last_reason = "unknown"

    while time.time() <= deadline:
        ok, reason = _pane_is_safe_to_inject(
            pane, settle_ms=settle_ms, sig_lines=sig_lines, allow_active=allow_active
        )
        if ok:
            success, msg, metrics = _pane_send_impl(pane, text, enter=True)
            metrics["wait_ms"] = round(time.time() * 1000 - start_ms, 2)
            return success, msg, metrics
        last_reason = reason
        time.sleep(0.2)

    latency_ms = time.time() * 1000 - start_ms
    return False, f"timeout:{last_reason}", {
        "latency_ms": round(latency_ms, 2),
        "reason": last_reason,
    }


def _deliver_impl(
    to_agent: str,
    message: str,
    from_agent: str = "sft_tmux",
    to_pane: str = "",
    timeout_sec: float = 30.0,
    allow_active: bool = False,
) -> tuple[dict, dict]:
    """Durable delivery + optional safe pane mirror. CLI: deliver, MCP: deliver."""
    start_ms = time.time() * 1000
    
    ok, msg_id_or_err, post_metrics = _post_message_impl(to_agent, message, from_agent)
    if not ok:
        latency_ms = time.time() * 1000 - start_ms
        return {"error": msg_id_or_err}, {"latency_ms": round(latency_ms, 2)}

    result = {
        "status": "ok",
        "msg_id": msg_id_or_err,
        "to": to_agent,
        "from": from_agent,
        "pane": to_pane or None,
        "pane_injected": False,
        "pane_reason": "not_requested",
    }

    if to_pane:
        sent, why, send_metrics = _pane_send_safe_impl(
            to_pane, message, timeout_sec=timeout_sec, allow_active=allow_active
        )
        result["pane_injected"] = bool(sent)
        result["pane_reason"] = "ok" if sent else why
        result["pane_metrics"] = send_metrics

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"latency_ms": round(latency_ms, 2), **post_metrics}
    return result, metrics


def _drain_impl(
    agent_id: str,
    pane: str = "",
    notify: bool = False,
    ack: bool = True,
    max_msgs: int = 50,
    timeout_sec: float = 0.5,
) -> tuple[dict, dict]:
    """Drain agent inbox. CLI: drain, MCP: drain."""
    start_ms = time.time() * 1000
    agent_dir = _agent_dir(agent_id)
    inbox_dir = agent_dir / "inbox"
    
    if not inbox_dir.exists():
        latency_ms = time.time() * 1000 - start_ms
        return {"drained": 0, "remaining": 0}, {"latency_ms": round(latency_ms, 2)}

    msg_files = sorted(inbox_dir.glob("*.msg"))
    drained = 0
    blocked_reason = ""

    for msg_file in msg_files[:max(0, max_msgs)]:
        try:
            msg = json.loads(msg_file.read_text())
        except Exception:
            continue

        text = str(msg.get("message", "")).strip()
        if not text:
            continue

        if notify:
            try:
                subprocess.Popen(
                    ["notify-send", f"Inbox: {agent_id}", text[:180]],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception:
                pass

        if pane:
            sent, why, _ = _pane_send_safe_impl(pane, text, timeout_sec=timeout_sec)
            if not sent:
                blocked_reason = why
                break

        if ack:
            archive_dir = inbox_dir / "archive"
            archive_dir.mkdir(exist_ok=True)
            try:
                msg_file.rename(archive_dir / msg_file.name)
            except Exception:
                pass

        drained += 1

    remaining = len(list(inbox_dir.glob("*.msg")))
    latency_ms = time.time() * 1000 - start_ms
    
    result = {
        "drained": drained,
        "remaining": remaining,
        "blocked": bool(blocked_reason),
        "blocked_reason": blocked_reason or None,
    }
    metrics = {"latency_ms": round(latency_ms, 2)}
    return result, metrics


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tmux agent orchestration with safe pane injection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  spawn      Create agent session
  kill       Kill agent session
  list       List agent sessions
  send       Send command to agent (with Enter)
  type       Type raw text to agent (no Enter)
  read       Read agent output
  pipe       Pipe file to agent
  attach     Attach to agent session
  msg        Post message to agent
  inbox      Read agent inbox
  ack        Acknowledge messages
  pane-send  Send keys to tmux pane (unsafe)
  pane-send-safe  Send to pane only when safe
  deliver    Durable delivery + optional safe pane mirror
  drain      Drain inbox (optional safe pane mirror + ack)

Examples:
  sft_tmux.py spawn --id worker1 python worker.py
  sft_tmux.py send --id worker1 "status"
  sft_tmux.py deliver --to worker1 --pane session:0.0 "Hello"
        """,
    )
    
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {CONFIG['version']}",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # spawn
    p_spawn = subparsers.add_parser("spawn", help="Create agent session")
    p_spawn.add_argument("-i", "--id", help="Agent ID (auto-generated if not provided)")
    p_spawn.add_argument("-d", "--cwd", help="Working directory")
    p_spawn.add_argument("command", nargs="*", help="Command to run")
    
    # kill
    p_kill = subparsers.add_parser("kill", help="Kill agent session")
    p_kill.add_argument("-i", "--id", required=True, help="Agent ID")
    
    # list
    p_list = subparsers.add_parser("list", help="List agent sessions")
    p_list.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    
    # send
    p_send = subparsers.add_parser("send", help="Send command to agent (with Enter)")
    p_send.add_argument("-i", "--id", required=True, help="Agent ID")
    p_send.add_argument("text", help="Command to send")
    
    # type
    p_type = subparsers.add_parser("type", help="Type raw text to agent (no Enter)")
    p_type.add_argument("-i", "--id", required=True, help="Agent ID")
    p_type.add_argument("-l", "--literal", action="store_true", help="Send literal text")
    p_type.add_argument("text", help="Text to type")
    
    # read
    p_read = subparsers.add_parser("read", help="Read agent output")
    p_read.add_argument("-i", "--id", required=True, help="Agent ID")
    p_read.add_argument("-n", "--lines", type=int, help="Number of lines")
    p_read.add_argument("-s", "--save", action="store_true", help="Save to output.log")
    
    # pipe
    p_pipe = subparsers.add_parser("pipe", help="Pipe file to agent")
    p_pipe.add_argument("-i", "--id", required=True, help="Agent ID")
    p_pipe.add_argument("file", help="File to pipe")
    
    # attach
    p_attach = subparsers.add_parser("attach", help="Attach to agent session")
    p_attach.add_argument("-i", "--id", required=True, help="Agent ID")
    
    # msg
    p_msg = subparsers.add_parser("msg", help="Post message to agent")
    p_msg.add_argument("-t", "--to", required=True, help="Target agent ID")
    p_msg.add_argument("-f", "--from", dest="from_agent", default="sft_tmux", help="Sender ID")
    p_msg.add_argument("message", help="Message content")
    
    # inbox
    p_inbox = subparsers.add_parser("inbox", help="Read agent inbox")
    p_inbox.add_argument("-i", "--id", required=True, help="Agent ID")
    p_inbox.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    
    # ack
    p_ack = subparsers.add_parser("ack", help="Acknowledge messages")
    p_ack.add_argument("-i", "--id", required=True, help="Agent ID")
    p_ack.add_argument("-a", "--all", action="store_true", help="Ack all messages")
    p_ack.add_argument("-m", "--msg", help="Specific message ID to ack")
    
    # pane-send
    p_pane_send = subparsers.add_parser("pane-send", help="Send keys to tmux pane (unsafe)")
    p_pane_send.add_argument("-p", "--pane", required=True, help="Target tmux pane (e.g., session:0.1)")
    p_pane_send.add_argument("-e", "--enter", action="store_true", help="Send Enter after text")
    p_pane_send.add_argument("text", help="Text to send")
    
    # pane-send-safe
    p_pane_safe = subparsers.add_parser("pane-send-safe", help="Send to tmux pane only when safe")
    p_pane_safe.add_argument("-p", "--pane", required=True, help="Target tmux pane")
    p_pane_safe.add_argument("-t", "--timeout", type=float, default=30.0, help="Max seconds to wait")
    p_pane_safe.add_argument("-a", "--allow-active", action="store_true", help="Allow injecting into active pane")
    p_pane_safe.add_argument("text", help="Text to send")
    
    # deliver
    p_deliver = subparsers.add_parser("deliver", help="Durable delivery + optional safe pane mirror")
    p_deliver.add_argument("-t", "--to", required=True, help="Target agent ID (inbox)")
    p_deliver.add_argument("-f", "--from", dest="from_agent", default="sft_tmux", help="Sender ID")
    p_deliver.add_argument("-p", "--pane", default="", help="Optional tmux pane to mirror into safely")
    p_deliver.add_argument("-T", "--timeout", type=float, default=30.0, help="Max seconds to wait for safe inject")
    p_deliver.add_argument("-a", "--allow-active", action="store_true", help="Allow injecting into active pane")
    p_deliver.add_argument("message", help="Message content")
    
    # drain
    p_drain = subparsers.add_parser("drain", help="Drain inbox (optional safe pane mirror + ack)")
    p_drain.add_argument("-i", "--id", required=True, help="Agent ID")
    p_drain.add_argument("-p", "--pane", default="", help="Optional tmux pane to mirror into")
    p_drain.add_argument("-n", "--notify", action="store_true", help="notify-send per message")
    p_drain.add_argument("-A", "--no-ack", action="store_true", help="Do not ack/archive")
    p_drain.add_argument("-m", "--max", type=int, default=50, help="Max messages to process")
    p_drain.add_argument("-T", "--timeout", type=float, default=0.5, help="Max seconds to wait per message")
    
    # MCP server
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")
    
    args = parser.parse_args()
    
    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "spawn":
            cmd_arg = args.command if args.command else None
            if not cmd_arg and not sys.stdin.isatty():
                cmd_arg = sys.stdin.read().strip().split()
            success, output, metrics = _spawn_impl(
                agent_id=args.id, command=cmd_arg, cwd=args.cwd
            )
            if success:
                print(output)
            else:
                print(f"Error: {output}", file=sys.stderr)
                sys.exit(1)
            _log("INFO", "spawn", f"agent_id={output}", metrics=json.dumps(metrics))
        elif args.command == "kill":
            success, output, metrics = _kill_impl(args.id)
            print(output if success else f"Error: {output}", file=sys.stderr if not success else sys.stdout)
            _log("INFO" if success else "ERROR", "kill", output, metrics=json.dumps(metrics))
            if not success:
                sys.exit(1)
        elif args.command == "list":
            agents, metrics = _list_impl()
            if args.json:
                print(json.dumps(agents, indent=2))
            else:
                if not agents:
                    print("No agent sessions")
                else:
                    for a in agents:
                        inbox_str = f" [{a['inbox']} msg]" if a["inbox"] > 0 else ""
                        print(f"{a['id']}: {a['status']}{inbox_str}")
            _log("INFO", "list", f"count={len(agents)}", metrics=json.dumps(metrics))
        elif args.command == "send":
            text = args.text
            if not text and not sys.stdin.isatty():
                text = sys.stdin.read().strip()
            assert text, "text required (positional argument or stdin)"
            success, output, metrics = _send_impl(args.id, text)
            print(output if success else f"Error: {output}", file=sys.stderr if not success else sys.stdout)
            _log("INFO" if success else "ERROR", "send", output, metrics=json.dumps(metrics))
            if not success:
                sys.exit(1)
        elif args.command == "type":
            text = args.text
            if not text and not sys.stdin.isatty():
                text = sys.stdin.read().strip()
            assert text, "text required (positional argument or stdin)"
            success, output, metrics = _type_impl(args.id, text, args.literal)
            print(output if success else f"Error: {output}", file=sys.stderr if not success else sys.stdout)
            _log("INFO" if success else "ERROR", "type", output, metrics=json.dumps(metrics))
            if not success:
                sys.exit(1)
        elif args.command == "read":
            success, output, metrics = _read_impl(args.id, args.lines, args.save)
            print(output if success else f"Error: {output}", file=sys.stderr if not success else sys.stdout)
            _log("INFO" if success else "ERROR", "read", f"agent_id={args.id}", metrics=json.dumps(metrics))
            if not success:
                sys.exit(1)
        elif args.command == "pipe":
            success, output, metrics = _pipe_impl(args.id, args.file)
            print(output if success else f"Error: {output}", file=sys.stderr if not success else sys.stdout)
            _log("INFO" if success else "ERROR", "pipe", output, metrics=json.dumps(metrics))
            if not success:
                sys.exit(1)
        elif args.command == "attach":
            success, output, metrics = _attach_impl(args.id)
            print(output if success else f"Error: {output}", file=sys.stderr if not success else sys.stdout)
            if not success:
                sys.exit(1)
        elif args.command == "msg":
            message = args.message
            if not message and not sys.stdin.isatty():
                message = sys.stdin.read().strip()
            assert message, "message required (positional argument or stdin)"
            success, output, metrics = _post_message_impl(args.to, message, args.from_agent)
            print(output if success else f"Error: {output}", file=sys.stderr if not success else sys.stdout)
            _log("INFO" if success else "ERROR", "msg", f"msg_id={output}", metrics=json.dumps(metrics))
            if not success:
                sys.exit(1)
        elif args.command == "inbox":
            messages, metrics = _read_inbox_impl(args.id)
            if args.json:
                print(json.dumps(messages, indent=2))
            else:
                if not messages:
                    print("No messages")
                else:
                    for m in messages:
                        print(f"[{m['id']}] from {m['from']}: {m['message'][:80]}...")
            _log("INFO", "inbox", f"count={len(messages)}", metrics=json.dumps(metrics))
        elif args.command == "ack":
            success, output, metrics = _ack_impl(args.id, args.msg, args.all)
            print(output if success else f"Error: {output}", file=sys.stderr if not success else sys.stdout)
            _log("INFO" if success else "ERROR", "ack", output, metrics=json.dumps(metrics))
            if not success:
                sys.exit(1)
        elif args.command == "pane-send":
            text = args.text
            if not text and not sys.stdin.isatty():
                text = sys.stdin.read().strip()
            assert text, "text required (positional argument or stdin)"
            success, output, metrics = _pane_send_impl(args.pane, text, args.enter)
            print(output if success else f"Error: {output}", file=sys.stderr if not success else sys.stdout)
            _log("INFO" if success else "ERROR", "pane_send", output, metrics=json.dumps(metrics))
            if not success:
                sys.exit(1)
        elif args.command == "pane-send-safe":
            text = args.text
            if not text and not sys.stdin.isatty():
                text = sys.stdin.read().strip()
            assert text, "text required (positional argument or stdin)"
            success, output, metrics = _pane_send_safe_impl(
                args.pane, text, timeout_sec=args.timeout, allow_active=args.allow_active
            )
            print(output if success else f"Error: {output}", file=sys.stderr if not success else sys.stdout)
            _log("INFO" if success else "ERROR", "pane_send_safe", output, metrics=json.dumps(metrics))
            if not success:
                sys.exit(1)
        elif args.command == "deliver":
            message = args.message
            if not message and not sys.stdin.isatty():
                message = sys.stdin.read().strip()
            assert message, "message required (positional argument or stdin)"
            result, metrics = _deliver_impl(
                args.to, message, args.from_agent, args.pane, args.timeout, args.allow_active
            )
            print(json.dumps(result, indent=2))
            _log("INFO", "deliver", f"msg_id={result.get('msg_id')}", metrics=json.dumps(metrics))
        elif args.command == "drain":
            result, metrics = _drain_impl(
                args.id, args.pane, args.notify, not args.no_ack, args.max, args.timeout
            )
            print(json.dumps(result, indent=2))
            _log("INFO", "drain", f"drained={result['drained']}", metrics=json.dumps(metrics))
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
    """Run as MCP server."""
    from fastmcp import FastMCP
    
    mcp = FastMCP("tmux")
    
    @mcp.tool()
    def spawn(agent_id: str = "", command: str = "", cwd: str = "") -> str:
        """Create a new tmux agent session.
        
        Args:
            agent_id: Optional agent ID (auto-generated if empty)
            command: Command to run in the session
            cwd: Working directory for the session
        
        Returns:
            JSON string with agent_id and status
        """
        cmd_list = [command] if command else None
        success, output, metrics = _spawn_impl(
            agent_id=agent_id or None,
            command=cmd_list,
            cwd=cwd or None,
        )
        return json.dumps({
            "success": success,
            "agent_id": output if success else None,
            "error": None if success else output,
            "metrics": metrics,
        })
    
    @mcp.tool()
    def kill(agent_id: str) -> str:
        """Kill an agent session.
        
        Args:
            agent_id: ID of the agent to kill
        
        Returns:
            JSON string with status
        """
        success, output, metrics = _kill_impl(agent_id)
        return json.dumps({
            "success": success,
            "message": output,
            "metrics": metrics,
        })
    
    @mcp.tool()
    def list() -> str:
        """List all agent sessions.
        
        Args:
            (no arguments)
        
        Returns:
            JSON string with list of agents
        """
        agents, metrics = _list_impl()
        return json.dumps({
            "agents": agents,
            "metrics": metrics,
        })
    
    @mcp.tool()
    def send(agent_id: str, text: str) -> str:
        """Send command to agent (with Enter to execute).
        
        Args:
            agent_id: Target agent ID
            text: Command text to send
        
        Returns:
            JSON string with status
        """
        success, output, metrics = _send_impl(agent_id, text)
        return json.dumps({
            "success": success,
            "message": output,
            "metrics": metrics,
        })
    
    @mcp.tool()
    def read(agent_id: str, lines: int = 0, save: bool = False) -> str:
        """Read output from agent session.
        
        Args:
            agent_id: Target agent ID
            lines: Number of lines to read (0 = all)
            save: Whether to save to output.log
        
        Returns:
            JSON string with output text
        """
        lines_val = lines if lines > 0 else None
        success, output, metrics = _read_impl(agent_id, lines_val, save)
        return json.dumps({
            "success": success,
            "output": output if success else None,
            "error": None if success else output,
            "metrics": metrics,
        })
    
    @mcp.tool()
    def msg(to_agent: str, message: str, from_agent: str = "sft_tmux") -> str:
        """Post a message to agent's inbox.
        
        Args:
            to_agent: Target agent ID
            message: Message content
            from_agent: Sender ID
        
        Returns:
            JSON string with message ID
        """
        success, output, metrics = _post_message_impl(to_agent, message, from_agent)
        return json.dumps({
            "success": success,
            "msg_id": output if success else None,
            "error": None if success else output,
            "metrics": metrics,
        })
    
    @mcp.tool()
    def inbox(agent_id: str) -> str:
        """Read messages from agent's inbox.
        
        Args:
            agent_id: Target agent ID
        
        Returns:
            JSON string with list of messages
        """
        messages, metrics = _read_inbox_impl(agent_id)
        return json.dumps({
            "messages": messages,
            "metrics": metrics,
        })
    
    @mcp.tool()
    def ack(agent_id: str, msg_id: str = "", ack_all: bool = False) -> str:
        """Acknowledge/archive messages.
        
        Args:
            agent_id: Target agent ID
            msg_id: Specific message ID to ack (ignored if ack_all)
            ack_all: Whether to ack all messages
        
        Returns:
            JSON string with status
        """
        msg_id_val = msg_id if msg_id else None
        success, output, metrics = _ack_impl(agent_id, msg_id_val, ack_all)
        return json.dumps({
            "success": success,
            "message": output,
            "metrics": metrics,
        })
    
    @mcp.tool()
    def pane_send_safe(pane: str, text: str, timeout: float = 30.0, allow_active: bool = False) -> str:
        """Send text to tmux pane only when safe (no human activity).
        
        Args:
            pane: Target tmux pane (e.g., 'session:0.1')
            text: Text to send
            timeout: Max seconds to wait for safe condition
            allow_active: Whether to allow injection into active pane
        
        Returns:
            JSON string with status
        """
        success, output, metrics = _pane_send_safe_impl(
            pane, text, timeout_sec=timeout, allow_active=allow_active
        )
        return json.dumps({
            "success": success,
            "message": output,
            "metrics": metrics,
        })
    
    @mcp.tool()
    def type(agent_id: str, text: str, literal: bool = False) -> str:
        """Type raw text to agent (no Enter to execute).
        
        Args:
            agent_id: Target agent ID
            text: Text to type
            literal: Whether to send literal text (no key name interpretation)
        
        Returns:
            JSON string with status
        """
        success, output, metrics = _type_impl(agent_id, text, literal)
        return json.dumps({
            "success": success,
            "message": output,
            "metrics": metrics,
        })
    
    @mcp.tool()
    def pipe(agent_id: str, filepath: str) -> str:
        """Pipe file contents to agent.
        
        Args:
            agent_id: Target agent ID
            filepath: Path to file to pipe
        
        Returns:
            JSON string with status
        """
        success, output, metrics = _pipe_impl(agent_id, filepath)
        return json.dumps({
            "success": success,
            "message": output,
            "metrics": metrics,
        })
    
    @mcp.tool()
    def attach(agent_id: str) -> str:
        """Attach to agent session interactively.
        
        Note: This will take over the terminal. Use with caution in MCP context.
        
        Args:
            agent_id: Target agent ID
        
        Returns:
            JSON string with status
        """
        success, output, metrics = _attach_impl(agent_id)
        return json.dumps({
            "success": success,
            "message": output,
            "metrics": metrics,
        })
    
    @mcp.tool()
    def pane_send(pane: str, text: str, enter: bool = False) -> str:
        """Send keys to tmux pane (unsafe - no activity detection).
        
        Args:
            pane: Target tmux pane (e.g., 'session:0.1')
            text: Text to send
            enter: Whether to send Enter after text
        
        Returns:
            JSON string with status
        """
        success, output, metrics = _pane_send_impl(pane, text, enter)
        return json.dumps({
            "success": success,
            "message": output,
            "metrics": metrics,
        })
    
    @mcp.tool()
    def deliver(
        to_agent: str,
        message: str,
        from_agent: str = "sft_tmux",
        pane: str = "",
        timeout: float = 30.0,
        allow_active: bool = False,
    ) -> str:
        """Durable delivery + optional safe pane mirror.
        
        Posts message to inbox and optionally mirrors to pane when safe.
        
        Args:
            to_agent: Target agent ID (inbox)
            message: Message content
            from_agent: Sender ID
            pane: Optional tmux pane to mirror into safely
            timeout: Max seconds to wait for safe inject
            allow_active: Whether to allow injection into active pane
        
        Returns:
            JSON string with delivery result
        """
        result, metrics = _deliver_impl(
            to_agent, message, from_agent, pane, timeout, allow_active
        )
        return json.dumps({
            "result": result,
            "metrics": metrics,
        })
    
    @mcp.tool()
    def drain(
        agent_id: str,
        pane: str = "",
        notify: bool = False,
        do_ack: bool = True,
        max_msgs: int = 50,
        timeout: float = 0.5,
    ) -> str:
        """Drain agent inbox (optional safe pane mirror + ack).
        
        Args:
            agent_id: Agent ID
            pane: Optional tmux pane to mirror into
            notify: Whether to notify-send per message
            do_ack: Whether to ack/archive messages (default: True)
            max_msgs: Max messages to process
            timeout: Max seconds to wait for safe pane inject per message
        
        Returns:
            JSON string with drain result
        """
        result, metrics = _drain_impl(
            agent_id, pane, notify, do_ack, max_msgs, timeout
        )
        return json.dumps({
            "result": result,
            "metrics": metrics,
        })
    
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
