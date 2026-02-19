#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "websocket-client>=1.6.0",
#     "httpx>=0.25.0",
#     "psutil",
#     "fastmcp",
# ]
# ///
"""Chrome automation via CDP — sandbox (anon), work, personal profiles with auto-launch.

Usage:
    sft_chrome.py <command> -p <profile> [options]
    sft_chrome.py mcp-stdio
"""

import argparse
import base64
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import websocket


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
    "look",
    "read",
    "goto",
    "tabs",
    "click",
    "fill",
    "js",
    "key",
    "focus",
    "wait",
    "hover",
    "scroll",
    "survey",
    "find",
    "act",
    "do",
    "status",
    "kill",
    "ensure",
    "network",
    "console",
    "history",
    "bookmarks",
    "dialog",
    "drag",
    "emulate",
    "trail_start",
    "trail_stop",
    "trail_status",
    "headless_start",
    "headless_stop",
    "headless_list",
    "headless_action",
    "replay",
]  # CLI + MCP — both interfaces

# When to use which profile:
# default - User's main browser for debugging and collaborative work
#   sandbox - Anonymous testing, CDP automation, no Google auth
PROFILES = {
    "sandbox": {"port": 9220, "desc": "Anonymous CDP testing (no Google auth)"},
    "default": {"port": 9221, "desc": "user@example.com"},
}
DEFAULT_PORT = PROFILES["default"]["port"]  # For CDPClient class default only

# Headless session port range (avoids collisions with named profiles)
_HEADLESS_PORT_START = 9230
_HEADLESS_PORT_END = 9250
_HEADLESS_SESSIONS: dict[str, dict] = {}  # name -> {port, process, profile_dir}

# Trail state — module-level singleton
_TRAIL_DIR: Path | None = None  # Set by chrome_trail_start, cleared by chrome_trail_stop
_TRAIL_STEP: int = 0
_TRAIL_RECORDING: list[dict] = []  # For save/replay — records actions when trail is active


def get_chrome_path():
    """Get Chrome/Chromium path for current platform."""
    import platform

    system = platform.system()

    if system == "Linux":
        # Dynamic discovery first
        for cmd in [
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
        ]:
            path = shutil.which(cmd)
            if path:
                return path

        # Hardcoded fallbacks if which fails
        candidates = [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/snap/bin/chromium",
        ]
        for path in candidates:
            if Path(path).exists():
                return path
        return "google-chrome"
    elif system == "Darwin":
        path = shutil.which("google-chrome")
        if path:
            return path
        return "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    else:  # Windows
        path = shutil.which("chrome.exe")
        if path:
            return path
        return r"C:\Program Files\Google\Chrome\Application\chrome.exe"


CHROME_PATH = get_chrome_path()


def get_profile_path(profile: str):
    """Get profile directory for specified profile."""
    import platform

    if platform.system() == "Linux":
        base_dir = Path.home() / ".config" / "sfb" / "chrome_profiles"
    elif platform.system() == "Darwin":
        base_dir = Path.home() / ".config" / "sfb" / "chrome_profiles"
    else:
        base_dir = Path(__file__).parent / "chrome_profiles"
    return base_dir / profile


def get_port(profile: str):
    """Get port for specified profile."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Valid: {list(PROFILES.keys())}")
    return PROFILES[profile]["port"]


def launch_chrome(profile: str, port: int = None):
    """Launch Chrome with the specified profile if not already running."""
    port = port or get_port(profile)
    profile_path = get_profile_path(profile)
    profile_info = PROFILES.get(profile, {})

    # Ensure profile directory exists
    if not profile_path.exists():
        print(f"Creating new profile directory at: {profile_path}", file=sys.stderr)
        profile_path.mkdir(parents=True, exist_ok=True)

    # Check if already running on this port
    try:
        httpx.get(f"http://localhost:{port}/json", timeout=1.0)
        # If successful, it's running
        return
    except Exception:
        pass  # Not running, proceed to launch

    desc = profile_info.get("desc", profile)
    print(f"Launching Chrome [{profile}] on port {port} ({desc})...", file=sys.stderr)

    cmd = [
        CHROME_PATH,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_path}",
        "--remote-allow-origins=*",
        "--no-first-run",
        "--no-default-browser-check",
    ]

    # Launch in background
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (FileNotFoundError, OSError) as e:
        print(f"Error: Failed to launch Chrome: {e}", file=sys.stderr)
        return

    # Wait for it to come up
    for i in range(10):
        try:
            httpx.get(f"http://localhost:{port}/json", timeout=1.0)
            print("Chrome launched successfully.", file=sys.stderr)
            return
        except Exception:
            time.sleep(1)

    print(
        "Warning: Chrome launch initiated but port not yet responsive.", file=sys.stderr
    )


# =============================================================================
# TRAIL SYSTEM — auto-screenshot + action recording
# =============================================================================

def _trail_capture(action: str, profile: str, page_idx: int = 0, detail: str = ""):
    """If a trail is active, capture a screenshot and record the action."""
    global _TRAIL_STEP
    if _TRAIL_DIR is None:
        return
    _TRAIL_STEP += 1
    fname = f"{_TRAIL_STEP:03d}_{action}.png"
    fpath = _TRAIL_DIR / fname

    try:
        port = _resolve_port(profile)
        ext = Extractor(port)
        if ext.connect(page_idx):
            result = ext.send_command("Page.captureScreenshot")
            image_bytes = base64.b64decode(result["data"])
            fpath.write_bytes(image_bytes)
            ext.close()
    except Exception:
        pass  # Trail capture never crashes the main flow

    # Record for replay
    _TRAIL_RECORDING.append({
        "step": _TRAIL_STEP,
        "action": action,
        "detail": detail,
        "screenshot": fname,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def chrome_trail_start(name: str = "") -> tuple[bool, str]:
    """Start a screenshot trail. Every action auto-captures a screenshot.

    Args:
        name: Trail name (default: timestamp-based)

    Returns:
        (success, trail_directory_path)
    """
    global _TRAIL_DIR, _TRAIL_STEP, _TRAIL_RECORDING
    if not name:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
    trail_dir = Path("data/chrome_trails") / name
    trail_dir.mkdir(parents=True, exist_ok=True)
    _TRAIL_DIR = trail_dir
    _TRAIL_STEP = 0
    _TRAIL_RECORDING = []
    _log("INFO", "trail_start", f"trail={name} dir={trail_dir}")
    return True, str(trail_dir)


def chrome_trail_stop() -> tuple[bool, dict]:
    """Stop the active trail. Saves the action log as trail.json.

    Returns:
        (success, summary_dict)
    """
    global _TRAIL_DIR, _TRAIL_STEP, _TRAIL_RECORDING
    if _TRAIL_DIR is None:
        return False, {"error": "No active trail"}

    # Save action log
    log_path = _TRAIL_DIR / "trail.json"
    with open(log_path, "w") as f:
        json.dump({"steps": _TRAIL_RECORDING, "total": _TRAIL_STEP}, f, indent=2)

    summary = {
        "trail_dir": str(_TRAIL_DIR),
        "total_steps": _TRAIL_STEP,
        "log_file": str(log_path),
    }
    _log("INFO", "trail_stop", f"steps={_TRAIL_STEP} dir={_TRAIL_DIR}")
    _TRAIL_DIR = None
    _TRAIL_STEP = 0
    _TRAIL_RECORDING = []
    return True, summary


def chrome_trail_status() -> dict:
    """Check if a trail is active and its current state."""
    if _TRAIL_DIR is None:
        return {"active": False}
    return {
        "active": True,
        "trail_dir": str(_TRAIL_DIR),
        "steps_captured": _TRAIL_STEP,
    }


# =============================================================================
# HEADLESS SESSIONS — parallel Chrome instances via --headless=new
# =============================================================================

def _find_free_headless_port() -> int:
    """Find next available port in the headless range."""
    used_ports = {s["port"] for s in _HEADLESS_SESSIONS.values()}
    for port in range(_HEADLESS_PORT_START, _HEADLESS_PORT_END):
        if port not in used_ports:
            # Check if actually free
            try:
                httpx.get(f"http://localhost:{port}/json", timeout=0.5)
                continue  # Something is already on this port
            except Exception:
                return port
    assert False, f"No free ports in range {_HEADLESS_PORT_START}-{_HEADLESS_PORT_END}"


def chrome_headless_start(name: str = "") -> tuple[bool, dict, str]:
    """Start a headless Chrome session. Returns (success, session_info, error).

    Args:
        name: Session name (default: auto-generated)
    """
    if not name:
        name = f"headless_{len(_HEADLESS_SESSIONS) + 1}"

    if name in _HEADLESS_SESSIONS:
        return False, {}, f"Session '{name}' already exists"

    port = _find_free_headless_port()
    profile_dir = Path(tempfile.mkdtemp(prefix=f"chrome_headless_{name}_"))

    cmd = [
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_dir}",
        "--remote-allow-origins=*",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-gpu",
        "--no-sandbox",
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (FileNotFoundError, OSError) as e:
        return False, {}, f"Failed to launch headless Chrome: {e}"

    # Wait for it to come up
    for _ in range(15):
        try:
            httpx.get(f"http://localhost:{port}/json", timeout=1.0)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        return False, {}, "Headless Chrome failed to start within timeout"

    _HEADLESS_SESSIONS[name] = {
        "port": port,
        "process": proc,
        "profile_dir": str(profile_dir),
        "pid": proc.pid,
    }

    info = {"name": name, "port": port, "pid": proc.pid, "profile_dir": str(profile_dir)}
    _log("INFO", "headless_start", f"name={name} port={port} pid={proc.pid}")
    return True, info, ""


def chrome_headless_stop(name: str) -> tuple[bool, str]:
    """Stop a headless Chrome session.

    Args:
        name: Session name to stop
    """
    if name not in _HEADLESS_SESSIONS:
        return False, f"Session '{name}' not found. Active: {list(_HEADLESS_SESSIONS.keys())}"

    session = _HEADLESS_SESSIONS.pop(name)
    try:
        proc = session["process"]
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    # Clean up temp profile
    try:
        shutil.rmtree(session["profile_dir"], ignore_errors=True)
    except Exception:
        pass

    _log("INFO", "headless_stop", f"name={name}")
    return True, f"Session '{name}' stopped"


def chrome_headless_list() -> list[dict]:
    """List all active headless sessions."""
    result = []
    for name, session in _HEADLESS_SESSIONS.items():
        alive = session["process"].poll() is None
        result.append({
            "name": name,
            "port": session["port"],
            "pid": session["pid"],
            "alive": alive,
        })
    return result


def chrome_headless_action(name: str, action: str, **kwargs) -> tuple[bool, Any, str]:
    """Execute a chrome action against a headless session by name.

    Maps the session name to its port, then delegates to the standard chrome_* functions.

    Args:
        name: Headless session name
        action: Action name (goto, click, fill, look, read, js, etc.)
        **kwargs: Arguments passed to the action function
    """
    if name not in _HEADLESS_SESSIONS:
        return False, None, f"Session '{name}' not found"

    session = _HEADLESS_SESSIONS[name]
    port = session["port"]

    # Map action to function, injecting port via a temporary profile entry
    temp_profile = f"_headless_{name}"
    PROFILES[temp_profile] = {"port": port, "desc": f"Headless: {name}"}

    try:
        action_map = {
            "goto": chrome_goto,
            "click": chrome_click,
            "fill": chrome_fill,
            "look": chrome_look,
            "read": chrome_read,
            "js": chrome_js,
            "key": chrome_key,
            "wait": chrome_wait,
            "hover": chrome_hover,
            "scroll": chrome_scroll,
            "survey": chrome_survey,
            "find": chrome_find,
            "do": chrome_do,
        }
        assert action in action_map, f"Unknown action '{action}'. Available: {list(action_map.keys())}"
        fn = action_map[action]
        # Inject profile into kwargs
        kwargs["profile"] = temp_profile
        return fn(**kwargs)
    finally:
        PROFILES.pop(temp_profile, None)


# =============================================================================
# SAVE/REPLAY — record and replay action scripts
# =============================================================================

def chrome_replay(script_path: str, profile: str, delay: float = 0.5) -> tuple[bool, dict, str]:
    """Replay a saved trail script (trail.json) against a browser profile.

    Args:
        script_path: Path to trail.json file
        profile: Target profile to replay against
        delay: Seconds between actions (default 0.5)

    Returns:
        (success, results_dict, error)
    """
    script_file = Path(script_path)
    assert script_file.exists(), f"Script file not found: {script_path}"

    with open(script_file) as f:
        script = json.load(f)

    steps = script.get("steps", [])
    results = []
    failed = 0

    for step in steps:
        action = step.get("action", "")
        detail = step.get("detail", "")

        if not detail:
            results.append({"step": step["step"], "action": action, "status": "skipped", "reason": "no detail"})
            continue

        try:
            action_args = json.loads(detail)
        except (json.JSONDecodeError, TypeError):
            results.append({"step": step["step"], "action": action, "status": "skipped", "reason": "unparseable detail"})
            continue

        # Map action to function
        action_map = {
            "goto": chrome_goto,
            "click": chrome_click,
            "fill": chrome_fill,
            "js": chrome_js,
            "key": chrome_key,
            "wait": chrome_wait,
            "hover": chrome_hover,
            "scroll": chrome_scroll,
            "do": chrome_do,
        }

        fn = action_map.get(action)
        if not fn:
            results.append({"step": step["step"], "action": action, "status": "skipped", "reason": "unknown action"})
            continue

        try:
            action_args["profile"] = profile
            result = fn(**action_args)
            success = result[0] if isinstance(result, tuple) else True
            results.append({"step": step["step"], "action": action, "status": "ok" if success else "fail"})
            if not success:
                failed += 1
        except Exception as e:
            results.append({"step": step["step"], "action": action, "status": "error", "error": str(e)})
            failed += 1

        time.sleep(delay)

    summary = {
        "total_steps": len(steps),
        "executed": len(results),
        "failed": failed,
        "results": results,
    }
    return failed == 0, summary, "" if failed == 0 else f"{failed} step(s) failed"


def chrome_ensure(profile: str) -> tuple[bool, dict, str]:
    """Ensure Chrome is running for profile, launch if needed, focus, return status.

    Args:
        profile: REQUIRED. One of: "default" or "sandbox"

    Returns:
        (success, status_dict, message)

    Behavior:
        1. Check if Chrome running for this profile
        2. If running: focus the window
        3. If not running: launch it, wait for ready, focus
        4. Return full status
    """
    if profile not in PROFILES:
        return False, {}, f"Unknown profile '{profile}'. Valid: {list(PROFILES.keys())}"

    port = PROFILES[profile]["port"]
    was_running = False

    # Check if already running
    try:
        httpx.get(f"http://localhost:{port}/json", timeout=1.0)
        was_running = True
    except Exception:
        pass

    if not was_running:
        # Launch it
        launch_chrome(port=port, profile=profile)

        # Verify it came up
        came_up = False
        for _ in range(10):
            try:
                httpx.get(f"http://localhost:{port}/json", timeout=1.0)
                came_up = True
                break
            except Exception:
                time.sleep(0.5)

        if not came_up:
            return (
                False,
                chrome_status(),
                f"Failed to launch Chrome for profile '{profile}'",
            )

    # Focus the window
    focus_success, focus_msg = chrome_focus(profile, page_idx=0)

    # Get final status
    status = chrome_status()

    action = "focused existing" if was_running else "launched and focused"
    return True, status, f"Chrome [{profile}]: {action}"


def ensure_chrome_cli(profile: str, port: int):
    """Auto-launch Chrome if not running (for CLI commands)."""
    try:
        httpx.get(f"http://localhost:{port}/json", timeout=1.0)
    except Exception:
        launch_chrome(profile, port)


def check_port_status(port: int) -> dict:
    """Check if Chrome is running on a specific port (no launch)."""
    try:
        response = httpx.get(f"http://localhost:{port}/json", timeout=1.0)
        tabs = [t for t in response.json() if t.get("type") == "page"]
        return {"running": True, "port": port, "tabs": len(tabs)}
    except Exception:
        return {"running": False, "port": port, "tabs": 0}


def get_all_instance_status() -> list:
    """Check status of all known profile ports (purely observational)."""
    results = []
    for profile, info in PROFILES.items():
        status = check_port_status(info["port"])
        status["profile"] = profile
        status["desc"] = info["desc"]
        results.append(status)
    return results


def get_tabs_for_port(port: int) -> list:
    """Get list of tabs for a port WITHOUT auto-launching."""
    try:
        response = httpx.get(f"http://localhost:{port}/json", timeout=2.0)
        pages = [t for t in response.json() if t.get("type") == "page"]
        return [
            {"idx": i, "title": p.get("title", ""), "url": p.get("url", "")}
            for i, p in enumerate(pages)
        ]
    except Exception:
        return []


def kill_profile_chrome(port: int) -> dict:
    """Kill Chrome instance on a specific port only."""
    import signal

    killed = []
    try:
        import psutil

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                cmdline_str = " ".join(cmdline)
                name = proc.info.get("name", "").lower()

                # Only kill if it's chrome AND matches our port
                if (
                    "chrome" in name or "chromium" in name
                ) and f"--remote-debugging-port={port}" in cmdline_str:
                    proc.send_signal(signal.SIGTERM)
                    killed.append(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        return {
            "success": False,
            "message": "psutil required for profile-specific kill",
        }

    if killed:
        return {
            "success": True,
            "port": port,
            "killed_pids": killed,
            "count": len(killed),
        }
    return {
        "success": False,
        "port": port,
        "message": f"No Chrome found on port {port}",
    }


def chrome_kill(target: str) -> str:
    """Kill Chrome instance(s). REQUIRES explicit target - no defaults.

    Args:
        target: REQUIRED. One of: "default", "sandbox", or "all"

    Returns:
        Success/failure message after verification via chrome_status()
    """
    valid_targets = ["default", "sandbox", "all"]
    if target not in valid_targets:
        return f"ERROR: target must be one of {valid_targets}, got '{target}'"

    # Get status BEFORE kill
    status_before = chrome_status()

    # Determine what to kill
    if target == "all":
        profiles_to_kill = list(PROFILES.keys())
    else:
        profiles_to_kill = [target]

    # Build list of ports and profile paths to target
    target_ports = []
    target_paths = []
    for p in profiles_to_kill:
        target_ports.append(PROFILES[p]["port"])
        target_paths.append(str(get_profile_path(p)))

    # Kill processes matching our targets
    killed_pids = []
    proc_result = subprocess.run(
        ["ps", "-eo", "pid,args"], capture_output=True, text=True
    )

    for line in proc_result.stdout.strip().split("\n"):
        if not line or "chrome" not in line.lower():
            continue

        line = line.strip()
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue

        pid = parts[0]
        cmdline = parts[1]

        should_kill = False

        # Check if this process matches any target port
        for port in target_ports:
            if f"--remote-debugging-port={port}" in cmdline:
                should_kill = True
                break

        # Check if this process matches any target profile path
        if not should_kill:
            for path in target_paths:
                if path in cmdline:
                    should_kill = True
                    break

        # For "all", also kill any chrome without specific profile (Default profile)
        if target == "all" and "chrome" in cmdline.lower():
            if "--type=" not in cmdline and "crashpad" not in cmdline:
                should_kill = True

        if should_kill:
            subprocess.run(["kill", "-9", pid], capture_output=True)
            killed_pids.append(pid)

    # If killing all, also do blanket pkill and clean locks
    if target == "all":
        subprocess.run(["pkill", "-9", "-f", "chrome"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "chromium"], capture_output=True)

        # Clean up singleton locks
        chrome_config = Path.home() / ".config" / "google-chrome"
        for lockfile in ["SingletonLock", "SingletonSocket", "SingletonCookie"]:
            lock_path = chrome_config / lockfile
            if lock_path.exists():
                try:
                    lock_path.unlink()
                except OSError:
                    pass

    # Wait for processes to die
    time.sleep(1)

    # Get status AFTER kill to verify
    status_after = chrome_status()

    # Verify expected state
    errors = []
    for p in profiles_to_kill:
        # Check CDP is not running
        for profile in status_after["profiles"]:
            if profile["name"] == p and profile["cdp_running"]:
                errors.append(f"{p} CDP still running")

        # Check no uncontrolled processes for this profile
        for unctl in status_after["uncontrolled"]:
            port = PROFILES[p]["port"]
            if unctl.get("debug_port") == port:
                errors.append(f"{p} has uncontrolled process (PID {unctl['pid']})")

    if errors:
        return f"FAILED: Killed {len(killed_pids)} PIDs but verification failed: {', '.join(errors)}"

    return f"SUCCESS: Killed {len(killed_pids)} processes. Verified {', '.join(profiles_to_kill)} stopped."


# =============================================================================
# CORE FUNCTIONS
# =============================================================================


class CDPClient:
    """Base client for Chrome DevTools Protocol interactions."""

    def __init__(self, debug_port: int = DEFAULT_PORT):
        self.debug_port = debug_port
        self.ws: websocket.WebSocket | None = None
        self.msg_id = 0
        self.target_id: str | None = None
        self.session_id: str | None = None

    def connect(self, page_idx: int = 0, profile: str = None) -> bool:
        """Connect to a specific page target. Auto-launches Chrome if profile given.

        Args:
            page_idx: Which tab to connect to (0 = first)
            profile: If provided, auto-launch Chrome for this profile on connect failure
        """
        success = self._try_connect(page_idx)
        if not success and profile:
            launch_chrome(profile, self.debug_port)
            success = self._try_connect(page_idx)
        return success

    def _try_connect(self, page_idx: int = 0) -> bool:
        """Internal: single connection attempt."""
        try:
            # Get list of pages
            response = httpx.get(f"http://localhost:{self.debug_port}/json")
            targets = response.json()

            # Filter for page targets
            pages = [t for t in targets if t["type"] == "page"]

            if not pages:
                print("No open pages found.", file=sys.stderr)
                return False

            if page_idx >= len(pages):
                print(
                    f"Page index {page_idx} out of range (0-{len(pages) - 1})",
                    file=sys.stderr,
                )
                return False

            target = pages[page_idx]
            self.target_id = target["id"]
            ws_url = target.get("webSocketDebuggerUrl")

            if not ws_url:
                print("No WebSocket URL found for target.", file=sys.stderr)
                return False

            self.ws = websocket.create_connection(ws_url)
            return True

        except Exception as e:
            print(f"Connection failed: {e}", file=sys.stderr)
            return False

    def close(self):
        if self.ws:
            self.ws.close()

    def send_command(
        self, method: str, params: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Send a CDP command and wait for the result."""
        if not self.ws:
            raise RuntimeError("Not connected to Chrome")

        self.msg_id += 1
        message = {"id": self.msg_id, "method": method, "params": params or {}}

        self.ws.send(json.dumps(message))

        while True:
            resp = json.loads(self.ws.recv())
            if resp.get("id") == self.msg_id:
                if "error" in resp:
                    raise RuntimeError(f"CDP Error: {resp['error']['message']}")
                return resp.get("result", {})

    def enable_domains(self):
        """Enable necessary CDP domains."""
        domains = ["Page", "DOM", "Runtime", "Input", "Network"]
        for domain in domains:
            try:
                self.send_command(f"{domain}.enable")
            except Exception:
                pass


class Navigator(CDPClient):
    def list_pages(self) -> list[dict[str, Any]]:
        """List open pages. Chrome must already be running."""
        try:
            response = httpx.get(f"http://localhost:{self.debug_port}/json")
            return [t for t in response.json() if t["type"] == "page"]
        except Exception as e:
            print(f"Failed to list pages: {e}", file=sys.stderr)
            return []

    def goto(self, url: str):
        self.send_command("Page.navigate", {"url": url})
        time.sleep(1)

    def reload(self, ignore_cache: bool = False):
        self.send_command("Page.reload", {"ignoreCache": ignore_cache})

    def go_back(self):
        history = self.send_command("Page.getNavigationHistory")
        idx = history.get("currentIndex", 0)
        if idx > 0:
            entry_id = history["entries"][idx - 1]["id"]
            self.send_command("Page.navigateToHistoryEntry", {"entryId": entry_id})

    def go_forward(self):
        history = self.send_command("Page.getNavigationHistory")
        idx = history.get("currentIndex", 0)
        entries = history.get("entries", [])
        if idx < len(entries) - 1:
            entry_id = entries[idx + 1]["id"]
            self.send_command("Page.navigateToHistoryEntry", {"entryId": entry_id})

    def new_tab(self, url: str = "about:blank"):
        """Create new tab. Chrome must already be running."""
        try:
            httpx.put(f"http://localhost:{self.debug_port}/json/new?{url}")
        except Exception as e:
            print(f"Failed to create tab: {e}", file=sys.stderr)

    def close_tab(self, page_idx: int):
        pages = self.list_pages()
        if 0 <= page_idx < len(pages):
            target_id = pages[page_idx]["id"]
            try:
                httpx.get(f"http://localhost:{self.debug_port}/json/close/{target_id}")
            except Exception as e:
                print(f"Failed to close tab: {e}", file=sys.stderr)


class Extractor(CDPClient):
    def evaluate_js(self, expression: str) -> Any:
        result = self.send_command(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        return result.get("result", {}).get("value")

    def take_screenshot(self, output_path: str):
        result = self.send_command("Page.captureScreenshot")
        data = base64.b64decode(result["data"])
        Path(output_path).write_bytes(data)

    def capture_screenshot_base64(self) -> str:
        """Capture screenshot and return base64 data for inline reading."""
        result = self.send_command("Page.captureScreenshot")
        return result["data"]

    def get_snapshot(self) -> dict[str, Any]:
        root = self.send_command("DOM.getDocument", {"depth": -1})
        return root


class Interactor(CDPClient):
    def get_element_center(self, selector: str) -> dict[str, int] | None:
        doc = self.send_command("DOM.getDocument")
        root_id = doc["root"]["nodeId"]

        node = self.send_command(
            "DOM.querySelector", {"nodeId": root_id, "selector": selector}
        )

        if not node.get("nodeId"):
            return None

        model = self.send_command("DOM.getBoxModel", {"nodeId": node["nodeId"]})
        content = model["model"]["content"]
        x = (content[0] + content[2]) / 2
        y = (content[1] + content[5]) / 2

        return {"x": int(x), "y": int(y), "nodeId": node["nodeId"]}

    def click(self, selector: str):
        center = self.get_element_center(selector)
        if not center:
            raise RuntimeError(f"Element not found: {selector}")

        self.send_command(
            "Input.dispatchMouseEvent",
            {
                "type": "mousePressed",
                "x": center["x"],
                "y": center["y"],
                "button": "left",
                "clickCount": 1,
            },
        )
        self.send_command(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseReleased",
                "x": center["x"],
                "y": center["y"],
                "button": "left",
                "clickCount": 1,
            },
        )

    def fill(self, selector: str, value: str):
        safe_selector = json.dumps(selector)
        self.evaluate_js(f"document.querySelector({safe_selector}).focus()")
        self.evaluate_js(f"document.querySelector({safe_selector}).select()")

        for char in value:
            self.send_command("Input.dispatchKeyEvent", {"type": "char", "text": char})

    def evaluate_js(self, expression: str):
        self.send_command("Runtime.evaluate", {"expression": expression})

    def press_key(self, key: str):
        # Special key mappings
        special_keys = {
            "Enter": {"key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13},
            "Tab": {"key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9},
            "Escape": {"key": "Escape", "code": "Escape", "windowsVirtualKeyCode": 27},
            "Backspace": {
                "key": "Backspace",
                "code": "Backspace",
                "windowsVirtualKeyCode": 8,
            },
            "Delete": {"key": "Delete", "code": "Delete", "windowsVirtualKeyCode": 46},
            "ArrowUp": {
                "key": "ArrowUp",
                "code": "ArrowUp",
                "windowsVirtualKeyCode": 38,
            },
            "ArrowDown": {
                "key": "ArrowDown",
                "code": "ArrowDown",
                "windowsVirtualKeyCode": 40,
            },
            "ArrowLeft": {
                "key": "ArrowLeft",
                "code": "ArrowLeft",
                "windowsVirtualKeyCode": 37,
            },
            "ArrowRight": {
                "key": "ArrowRight",
                "code": "ArrowRight",
                "windowsVirtualKeyCode": 39,
            },
        }

        if key in special_keys:
            params = {"type": "keyDown", **special_keys[key]}
            self.send_command("Input.dispatchKeyEvent", params)
            params["type"] = "keyUp"
            self.send_command("Input.dispatchKeyEvent", params)
        else:
            # Regular character
            self.send_command("Input.dispatchKeyEvent", {"type": "char", "text": key})


def _resolve_port(profile: str) -> int:
    """Resolve profile name to port number."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Valid: {list(PROFILES.keys())}")
    return PROFILES[profile]["port"]


def chrome_status() -> dict:
    """Get browser state for ALL profiles. Shows TRUTH - both CDP-controlled and not.

    Returns dict with:
        - profiles: list of ALL profile states (default, sandbox)
        - uncontrolled: Chrome processes running but NOT under CDP control
        - summary: quick counts

    Each profile includes:
        - name, port, desc
        - cdp_running: True if responding on CDP port
        - tabs: list with idx, title, url (only if CDP connected)
        - active_tab: the currently active tab
    """
    result = {
        "profiles": [],
        "uncontrolled": [],
        "summary": {
            "total_profiles": 3,
            "cdp_running": 0,
            "total_tabs": 0,
            "uncontrolled_count": 0,
        },
    }

    # Track which ports are responding to CDP
    cdp_ports_active = set()

    # Check each known profile's CDP port
    for p in PROFILES.keys():
        info = PROFILES[p]
        port = info["port"]

        profile_data = {
            "name": p,
            "port": port,
            "desc": info["desc"],
            "cdp_running": False,
            "tabs": [],
            "active_tab": None,
        }

        try:
            response = httpx.get(f"http://localhost:{port}/json", timeout=2.0)
            pages = [t for t in response.json() if t.get("type") == "page"]
            profile_data["cdp_running"] = True
            cdp_ports_active.add(port)
            result["summary"]["cdp_running"] += 1

            for idx, page in enumerate(pages):
                tab = {
                    "idx": idx,
                    "title": page.get("title", ""),
                    "url": page.get("url", ""),
                    "id": page.get("id", ""),
                }
                profile_data["tabs"].append(tab)
                result["summary"]["total_tabs"] += 1

            if profile_data["tabs"]:
                profile_data["active_tab"] = profile_data["tabs"][0]

        except Exception:
            pass

        result["profiles"].append(profile_data)

    # Find Chrome processes NOT under CDP control
    try:
        proc_result = subprocess.run(
            ["ps", "-eo", "pid,args"], capture_output=True, text=True
        )

        for line in proc_result.stdout.strip().split("\n"):
            if not line or "chrome" not in line.lower():
                continue

            # Skip helper processes
            if any(x in line for x in ["--type=", "crashpad", "nacl", "zygote"]):
                continue

            # Parse PID and cmdline
            line = line.strip()
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue

            pid = int(parts[0])
            cmdline = parts[1]

            # Check for debug port
            debug_port = None
            if "--remote-debugging-port=" in cmdline:
                for part in cmdline.split():
                    if "--remote-debugging-port=" in part:
                        debug_port = int(part.split("=")[1])
                        break

            # Skip if this port is already tracked via CDP
            if debug_port and debug_port in cdp_ports_active:
                continue

            # Extract profile info
            profile_dir = None
            for marker in ["--profile-directory=", "--user-data-dir="]:
                if marker in cmdline:
                    for part in cmdline.split():
                        if part.startswith(marker):
                            profile_dir = part.split("=")[1]
                            break

            # This is an uncontrolled Chrome
            result["uncontrolled"].append(
                {
                    "pid": pid,
                    "debug_port": debug_port,
                    "profile_dir": profile_dir,
                    "note": "NOT under CDP control" if debug_port else "No debug port",
                }
            )
            result["summary"]["uncontrolled_count"] += 1

    except Exception:
        pass

    return result


def chrome_read(
    profile: str, page_idx: int = 0, format: str = "text"
) -> tuple[bool, str, str]:
    """Read page content as text.

    Args:
        profile: Chrome profile
        page_idx: Tab index
        format: 'text' (clean text), 'html' (raw HTML), 'interactive' (forms/buttons/links only)

    Returns: (success, content, error)
    """
    port = _resolve_port(profile)
    ext = Extractor(port)
    if not ext.connect(page_idx, profile=profile):
        return (
            False,
            "",
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        ext.enable_domains()

        if format == "html":
            content = ext.evaluate_js("document.documentElement.outerHTML")
        elif format == "interactive":
            # Get interactive elements with their text and selectors
            js = """
            (() => {
                const elements = [];
                const selectors = 'a, button, input, select, textarea, [role="button"], [onclick]';
                document.querySelectorAll(selectors).forEach((el, idx) => {
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        elements.push({
                            tag: el.tagName.toLowerCase(),
                            type: el.type || null,
                            text: (el.innerText || el.value || el.placeholder || '').slice(0, 100).trim(),
                            href: el.href || null,
                            id: el.id || null,
                            name: el.name || null,
                            selector: el.id ? '#' + el.id : 
                                      el.name ? `[name="${el.name}"]` :
                                      el.className ? '.' + el.className.split(' ')[0] : 
                                      el.tagName.toLowerCase()
                        });
                    }
                });
                return elements;
            })()
            """
            elements = ext.evaluate_js(js)
            lines = []
            for el in elements or []:
                text = el.get("text", "")[:50]
                tag = el.get("tag", "")
                selector = el.get("selector", "")
                href = el.get("href", "")
                if tag == "a" and href:
                    lines.append(f"[link] {text} -> {href[:60]} ({selector})")
                elif tag == "button" or el.get("type") == "submit":
                    lines.append(f"[button] {text} ({selector})")
                elif tag == "input":
                    input_type = el.get("type", "text")
                    lines.append(f"[input:{input_type}] {text} ({selector})")
                elif tag == "select":
                    lines.append(f"[select] {text} ({selector})")
                elif tag == "textarea":
                    lines.append(f"[textarea] {text} ({selector})")
                else:
                    lines.append(f"[{tag}] {text} ({selector})")
            content = "\n".join(lines)
        else:  # text
            content = ext.evaluate_js("document.body.innerText")

        return True, content or "", ""
    except Exception as e:
        return False, "", str(e)
    finally:
        ext.close()


def chrome_look(
    profile: str, page_idx: int = 0, full_page: bool = False, selector: str = None
) -> tuple[bool, bytes, str]:
    """Screenshot current tab. Returns (success, image_bytes, error).

    Args:
        profile: Chrome profile
        page_idx: Tab index
        full_page: Capture full scrollable page (not just viewport)
        selector: Capture specific element only
    """
    port = _resolve_port(profile)
    ext = Extractor(port)
    if not ext.connect(page_idx, profile=profile):
        return (
            False,
            b"",
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        ext.enable_domains()

        params = {}

        if selector:
            # Get element bounds for clip
            safe_selector = json.dumps(selector)
            js = f"""
            (() => {{
                const el = document.querySelector({safe_selector});
                if (!el) return null;
                const rect = el.getBoundingClientRect();
                return {{
                    x: rect.x + window.scrollX,
                    y: rect.y + window.scrollY,
                    width: rect.width,
                    height: rect.height
                }};
            }})()
            """
            bounds = ext.evaluate_js(js)
            if not bounds:
                return False, b"", f"Element not found: {selector}"
            params["clip"] = {
                "x": bounds["x"],
                "y": bounds["y"],
                "width": bounds["width"],
                "height": bounds["height"],
                "scale": 1,
            }
        elif full_page:
            # Get full page dimensions
            dims = ext.evaluate_js("""
            ({
                width: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
                height: Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)
            })
            """)
            params["clip"] = {
                "x": 0,
                "y": 0,
                "width": dims["width"],
                "height": dims["height"],
                "scale": 1,
            }
            params["captureBeyondViewport"] = True

        result = ext.send_command("Page.captureScreenshot", params)
        image_bytes = base64.b64decode(result["data"])
        return True, image_bytes, ""
    except Exception as e:
        return False, b"", f"Screenshot failed: {e}"
    finally:
        ext.close()


def chrome_goto(url: str, profile: str, page_idx: int = 0) -> tuple[bool, str]:
    """Navigate to URL. Auto-launches Chrome if needed. Returns (success, message)."""
    port = _resolve_port(profile)
    nav = Navigator(port)
    if not nav.connect(page_idx, profile=profile):
        return False, f"Failed to connect to Chrome for profile '{profile}'"

    try:
        nav.enable_domains()
        nav.goto(url)
        _trail_capture("goto", profile, page_idx, json.dumps({"url": url}))
        return True, f"Navigated to {url}"
    except Exception as e:
        return False, str(e)
    finally:
        nav.close()


def chrome_tabs(profile: str) -> tuple[bool, list, str]:
    """List open tabs. Returns (success, tabs_list, error).

    Each tab has: idx, title, url
    """
    port = _resolve_port(profile)

    # Check if Chrome is running first
    status = check_port_status(port)
    if not status["running"]:
        return (
            False,
            [],
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        tabs = get_tabs_for_port(port)
        return True, tabs, ""
    except Exception as e:
        return False, [], str(e)


def chrome_click(selector: str, profile: str, page_idx: int = 0) -> tuple[bool, str]:
    """Click element by CSS selector. Returns (success, message)."""
    port = _resolve_port(profile)
    inter = Interactor(port)
    if not inter.connect(page_idx, profile=profile):
        return (
            False,
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        inter.enable_domains()
        inter.click(selector)
        _trail_capture("click", profile, page_idx, json.dumps({"selector": selector}))
        return True, f"Clicked {selector}"
    except Exception as e:
        return False, str(e)
    finally:
        inter.close()


def chrome_fill(
    selector: str, value: str, profile: str, page_idx: int = 0
) -> tuple[bool, str]:
    """Fill input field. Returns (success, message)."""
    port = _resolve_port(profile)
    inter = Interactor(port)
    if not inter.connect(page_idx, profile=profile):
        return (
            False,
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        inter.enable_domains()
        inter.fill(selector, value)
        _trail_capture("fill", profile, page_idx, json.dumps({"selector": selector, "value": value}))
        return True, f"Filled {selector}"
    except Exception as e:
        return False, str(e)
    finally:
        inter.close()


def chrome_js(
    expression: str, profile: str, page_idx: int = 0
) -> tuple[bool, any, str]:
    """Evaluate JavaScript. Returns (success, result, error)."""
    port = _resolve_port(profile)
    ext = Extractor(port)
    if not ext.connect(page_idx, profile=profile):
        return (
            False,
            None,
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        ext.enable_domains()
        result = ext.evaluate_js(expression)
        _trail_capture("js", profile, page_idx, json.dumps({"expression": expression[:200]}))
        return True, result, ""
    except Exception as e:
        return False, None, str(e)
    finally:
        ext.close()


def chrome_key(key: str, profile: str, page_idx: int = 0) -> tuple[bool, str]:
    """Press a key. Returns (success, message)."""
    port = _resolve_port(profile)
    inter = Interactor(port)
    if not inter.connect(page_idx, profile=profile):
        return (
            False,
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        inter.enable_domains()
        inter.press_key(key)
        _trail_capture("key", profile, page_idx, json.dumps({"key": key}))
        return True, f"Pressed {key}"
    except Exception as e:
        return False, str(e)
    finally:
        inter.close()


def chrome_focus(profile: str, page_idx: int = 0) -> tuple[bool, str]:
    """Bring browser window to front.

    Args:
        profile: Chrome profile
        page_idx: Tab index

    Returns: (success, message)
    """
    port = _resolve_port(profile)
    ext = Extractor(port)
    if not ext.connect(page_idx, profile=profile):
        return (
            False,
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        ext.enable_domains()
        ext.send_command("Page.bringToFront")
        return True, f"Browser window focused"
    except Exception as e:
        return False, str(e)
    finally:
        ext.close()


def chrome_wait(
    profile: str,
    text: str = None,
    selector: str = None,
    timeout: int = 10,
    page_idx: int = 0,
) -> tuple[bool, str]:
    """Wait for text or element to appear on page.

    Args:
        text: Text to wait for (checks document.body.innerText)
        selector: CSS selector to wait for
        timeout: Max seconds to wait (default 10)
        profile: Chrome profile
        page_idx: Tab index

    Returns: (success, message)
    """
    if not text and not selector:
        return False, "Must specify either text or selector"

    port = _resolve_port(profile)
    ext = Extractor(port)
    if not ext.connect(page_idx, profile=profile):
        return (
            False,
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        ext.enable_domains()
        start = time.time()

        while time.time() - start < timeout:
            if text:
                content = ext.evaluate_js("document.body.innerText")
                if content and text in content:
                    _trail_capture("wait", profile, page_idx, json.dumps({"text": text, "elapsed": f"{time.time() - start:.1f}s"}))
                    return True, f"Found text '{text}' after {time.time() - start:.1f}s"
            if selector:
                safe_selector = json.dumps(selector)
                found = ext.evaluate_js(f"!!document.querySelector({safe_selector})")
                if found:
                    _trail_capture("wait", profile, page_idx, json.dumps({"selector": selector, "elapsed": f"{time.time() - start:.1f}s"}))
                    return (
                        True,
                        f"Found selector '{selector}' after {time.time() - start:.1f}s",
                    )
            time.sleep(0.5)

        return (
            False,
            f"Timeout after {timeout}s waiting for {'text: ' + text if text else 'selector: ' + selector}",
        )
    except Exception as e:
        return False, str(e)
    finally:
        ext.close()


def chrome_hover(selector: str, profile: str, page_idx: int = 0) -> tuple[bool, str]:
    """Hover over an element. Returns (success, message)."""
    port = _resolve_port(profile)
    inter = Interactor(port)
    if not inter.connect(page_idx, profile=profile):
        return (
            False,
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        inter.enable_domains()
        center = inter.get_element_center(selector)
        if not center:
            return False, f"Element not found: {selector}"

        inter.send_command(
            "Input.dispatchMouseEvent",
            {"type": "mouseMoved", "x": center["x"], "y": center["y"]},
        )
        _trail_capture("hover", profile, page_idx, json.dumps({"selector": selector}))
        return True, f"Hovered over {selector}"
    except Exception as e:
        return False, str(e)
    finally:
        inter.close()


def chrome_scroll(
    profile: str,
    direction: str = "down",
    amount: int = 3,
    selector: str = None,
    page_idx: int = 0,
) -> tuple[bool, str]:
    """Scroll the page or an element.

    Args:
        direction: 'up', 'down', 'left', 'right'
        amount: Number of scroll ticks (1-10, default 3)
        selector: Optional element to scroll within
        profile: Chrome profile
        page_idx: Tab index

    Returns: (success, message)
    """
    port = _resolve_port(profile)
    inter = Interactor(port)
    if not inter.connect(page_idx, profile=profile):
        return (
            False,
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        inter.enable_domains()

        # Get scroll position
        if selector:
            center = inter.get_element_center(selector)
            if not center:
                return False, f"Element not found: {selector}"
            x, y = center["x"], center["y"]
        else:
            # Center of viewport
            x, y = 400, 300

        # Calculate delta (120 pixels per tick is standard)
        delta = amount * 120
        delta_x = 0
        delta_y = 0

        if direction == "down":
            delta_y = -delta
        elif direction == "up":
            delta_y = delta
        elif direction == "right":
            delta_x = -delta
        elif direction == "left":
            delta_x = delta
        else:
            return False, f"Invalid direction: {direction}"

        inter.send_command(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseWheel",
                "x": x,
                "y": y,
                "deltaX": delta_x,
                "deltaY": delta_y,
            },
        )
        _trail_capture("scroll", profile, page_idx, json.dumps({"direction": direction, "amount": amount, "selector": selector}))
        return True, f"Scrolled {direction} by {amount} ticks"
    except Exception as e:
        return False, str(e)
    finally:
        inter.close()


def chrome_network(
    profile: str, page_idx: int = 0, clear: bool = False
) -> tuple[bool, list, str]:
    """Get network requests for the current page.

    Args:
        profile: Chrome profile
        page_idx: Tab index
        clear: Clear the request log after fetching

    Returns: (success, requests_list, error)

    Each request has: url, method, status, type, size, time

    Note: Only captures requests made AFTER Network.enable was called.
    For best results, call this after navigation completes.
    """
    port = _resolve_port(profile)
    ext = Extractor(port)
    if not ext.connect(page_idx, profile=profile):
        return (
            False,
            [],
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        ext.enable_domains()

        # Use Performance API to get resource timing (works retroactively)
        js = """
        (() => {
            const entries = performance.getEntriesByType('resource');
            return entries.map(e => ({
                url: e.name,
                type: e.initiatorType,
                size: e.transferSize || 0,
                duration: Math.round(e.duration),
                start: Math.round(e.startTime)
            }));
        })()
        """
        requests = ext.evaluate_js(js) or []

        if clear:
            ext.evaluate_js("performance.clearResourceTimings()")

        return True, requests, ""
    except Exception as e:
        return False, [], str(e)
    finally:
        ext.close()


def chrome_console(
    profile: str, page_idx: int = 0, clear: bool = False
) -> tuple[bool, list, str]:
    """Get console messages from the page.

    Note: This captures messages logged AFTER we connect. For persistent
    capture, use the browser's DevTools or implement event listening.

    Returns: (success, messages_list, error)
    """
    port = _resolve_port(profile)
    ext = Extractor(port)
    if not ext.connect(page_idx, profile=profile):
        return (
            False,
            [],
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        ext.enable_domains()

        # Inject console capture if not already present
        js = """
        (() => {
            if (!window.__sfbConsoleLog) {
                window.__sfbConsoleLog = [];
                const original = console.log;
                console.log = function(...args) {
                    window.__sfbConsoleLog.push({
                        type: 'log',
                        time: Date.now(),
                        args: args.map(a => String(a))
                    });
                    original.apply(console, args);
                };
                // Also capture errors
                window.onerror = function(msg, url, line) {
                    window.__sfbConsoleLog.push({
                        type: 'error',
                        time: Date.now(),
                        args: [`${msg} at ${url}:${line}`]
                    });
                };
            }
            return window.__sfbConsoleLog || [];
        })()
        """
        messages = ext.evaluate_js(js) or []

        if clear:
            ext.evaluate_js("window.__sfbConsoleLog = []")

        return True, messages, ""
    except Exception as e:
        return False, [], str(e)
    finally:
        ext.close()


def chrome_history(
    profile: str, query: str = None, days: int = 7, limit: int = 50
) -> tuple[bool, list, str]:
    """Search browser history.

    Args:
        query: Search term (searches URL and title). None = all history.
        days: How far back to search (default 7)
        limit: Max results (default 50)
        profile: Chrome profile

    Returns: (success, results_list, error)

    Each result has: url, title, visit_count, last_visit (ISO timestamp)

    Note: Reads from SQLite database, works even if Chrome is running.
    """
    profile_path = get_profile_path(profile) / "Default" / "History"

    if not profile_path.exists():
        return False, [], f"History database not found for profile '{profile}'"

    try:
        # Copy to temp file to avoid lock issues with running Chrome
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name
        shutil.copy2(profile_path, tmp_path)

        conn = sqlite3.connect(tmp_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Chrome timestamps are microseconds since 1601-01-01
        # Convert to Unix timestamp: (chrome_time / 1000000) - 11644473600
        days_ago_chrome = (time.time() + 11644473600) * 1000000 - (
            days * 86400 * 1000000
        )

        if query:
            cursor.execute(
                """
                SELECT url, title, visit_count, 
                       datetime(last_visit_time/1000000-11644473600, 'unixepoch') as last_visit
                FROM urls 
                WHERE (url LIKE ? OR title LIKE ?) 
                  AND last_visit_time > ?
                ORDER BY last_visit_time DESC 
                LIMIT ?
            """,
                (f"%{query}%", f"%{query}%", days_ago_chrome, limit),
            )
        else:
            cursor.execute(
                """
                SELECT url, title, visit_count,
                       datetime(last_visit_time/1000000-11644473600, 'unixepoch') as last_visit
                FROM urls 
                WHERE last_visit_time > ?
                ORDER BY last_visit_time DESC 
                LIMIT ?
            """,
                (days_ago_chrome, limit),
            )

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        Path(tmp_path).unlink()  # Clean up temp file

        return True, results, ""
    except Exception as e:
        return False, [], str(e)


def chrome_bookmarks(
    profile: str, query: str = None, folder: str = None
) -> tuple[bool, list, str]:
    """Search bookmarks.

    Args:
        query: Search term (searches name and URL). None = all bookmarks.
        folder: Filter by folder path (e.g., "bookmark_bar/AI")
        profile: Chrome profile

    Returns: (success, bookmarks_list, error)

    Each bookmark has: name, url, folder, date_added (ISO timestamp)
    """
    profile_path = get_profile_path(profile) / "Default" / "Bookmarks"

    if not profile_path.exists():
        return False, [], f"Bookmarks file not found for profile '{profile}'"

    try:
        with open(profile_path) as f:
            data = json.load(f)

        def extract_bookmarks(node, path=""):
            """Recursively extract bookmarks from folder structure."""
            results = []
            if node.get("type") == "url":
                # Chrome timestamp: microseconds since 1601-01-01
                date_added = node.get("date_added", "0")
                try:
                    unix_ts = int(date_added) / 1000000 - 11644473600
                    iso_date = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(unix_ts))
                except (ValueError, OSError):
                    iso_date = None

                results.append(
                    {
                        "name": node.get("name", ""),
                        "url": node.get("url", ""),
                        "folder": path,
                        "date_added": iso_date,
                    }
                )
            elif node.get("type") == "folder" or "children" in node:
                folder_name = node.get("name", "")
                new_path = f"{path}/{folder_name}" if path else folder_name
                for child in node.get("children", []):
                    results.extend(extract_bookmarks(child, new_path))
            return results

        all_bookmarks = []
        roots = data.get("roots", {})
        for root_name, root_node in roots.items():
            if isinstance(root_node, dict):
                all_bookmarks.extend(extract_bookmarks(root_node, root_name))

        # Filter by folder if specified
        if folder:
            all_bookmarks = [
                b for b in all_bookmarks if folder.lower() in b["folder"].lower()
            ]

        # Filter by query if specified
        if query:
            query_lower = query.lower()
            all_bookmarks = [
                b
                for b in all_bookmarks
                if query_lower in b["name"].lower() or query_lower in b["url"].lower()
            ]

        return True, all_bookmarks, ""
    except Exception as e:
        return False, [], str(e)


def chrome_bookmark_add(
    profile: str, url: str, name: str = None, folder: str = "bookmark_bar"
) -> tuple[bool, str]:
    """Add a bookmark.

    Args:
        url: URL to bookmark
        name: Bookmark name (defaults to URL)
        folder: Folder path (default: bookmark_bar). Use "/" for nested folders.
        profile: Chrome profile

    Returns: (success, message)

    Note: Chrome will pick up changes on next restart or bookmark sync.
    """
    profile_path = get_profile_path(profile) / "Default" / "Bookmarks"

    # Ensure Default directory exists
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    # If bookmarks file doesn't exist, create skeleton
    if not profile_path.exists():
        data = {
            "checksum": "",
            "roots": {
                "bookmark_bar": {
                    "children": [],
                    "name": "Bookmarks bar",
                    "type": "folder",
                },
                "other": {"children": [], "name": "Other bookmarks", "type": "folder"},
                "synced": {
                    "children": [],
                    "name": "Mobile bookmarks",
                    "type": "folder",
                },
            },
            "version": 1,
        }
    else:
        with open(profile_path) as f:
            data = json.load(f)

    # Create bookmark entry
    # Chrome timestamp: microseconds since 1601-01-01
    chrome_ts = str(int((time.time() + 11644473600) * 1000000))

    # Find max ID
    def find_max_id(node):
        max_id = int(node.get("id", 0))
        for child in node.get("children", []):
            max_id = max(max_id, find_max_id(child))
        return max_id

    max_id = 0
    for root_node in data.get("roots", {}).values():
        if isinstance(root_node, dict):
            max_id = max(max_id, find_max_id(root_node))

    import uuid

    new_bookmark = {
        "date_added": chrome_ts,
        "date_last_used": "0",
        "guid": str(uuid.uuid4()),
        "id": str(max_id + 1),
        "name": name or url,
        "type": "url",
        "url": url,
    }

    # Navigate to folder and add bookmark
    folder_parts = folder.split("/")
    root_name = folder_parts[0] if folder_parts else "bookmark_bar"

    if root_name not in data["roots"]:
        root_name = "bookmark_bar"

    current = data["roots"][root_name]

    # Navigate through subfolders (create if needed)
    for part in folder_parts[1:]:
        found = None
        for child in current.get("children", []):
            if (
                child.get("type") == "folder"
                and child.get("name", "").lower() == part.lower()
            ):
                found = child
                break
        if found:
            current = found
        else:
            # Create folder
            max_id += 1
            new_folder = {
                "children": [],
                "date_added": chrome_ts,
                "date_modified": chrome_ts,
                "guid": str(uuid.uuid4()),
                "id": str(max_id),
                "name": part,
                "type": "folder",
            }
            current.setdefault("children", []).append(new_folder)
            current = new_folder

    current.setdefault("children", []).append(new_bookmark)

    # Write back
    with open(profile_path, "w") as f:
        json.dump(data, f, indent=3)

    return True, f"Added bookmark '{name or url}' to {folder}"


def chrome_bookmark_delete(
    profile: str, url: str = None, name: str = None
) -> tuple[bool, str]:
    """Delete a bookmark by URL or name.

    Args:
        url: URL to delete (exact match)
        name: Name to delete (exact match)
        profile: Chrome profile

    Returns: (success, message)

    Deletes first match found.
    """
    if not url and not name:
        return False, "Must specify url or name"

    profile_path = get_profile_path(profile) / "Default" / "Bookmarks"

    if not profile_path.exists():
        return False, f"Bookmarks file not found for profile '{profile}'"

    with open(profile_path) as f:
        data = json.load(f)

    deleted = [False]  # Use list to allow mutation in nested function

    def remove_bookmark(node):
        if "children" not in node:
            return
        new_children = []
        for child in node["children"]:
            if child.get("type") == "url":
                if (url and child.get("url") == url) or (
                    name and child.get("name") == name
                ):
                    deleted[0] = True
                    continue  # Skip this bookmark (delete it)
            new_children.append(child)
            remove_bookmark(child)  # Recurse into folders
        node["children"] = new_children

    for root_node in data.get("roots", {}).values():
        if isinstance(root_node, dict):
            remove_bookmark(root_node)

    if deleted[0]:
        with open(profile_path, "w") as f:
            json.dump(data, f, indent=3)
        return True, f"Deleted bookmark {'url=' + url if url else 'name=' + name}"
    else:
        return False, "Bookmark not found"


def chrome_dialog_handle(
    profile: str, accept: bool = True, prompt_text: str = None, page_idx: int = 0
) -> tuple[bool, str]:
    """Handle JavaScript dialog (alert/confirm/prompt).

    Call this AFTER a dialog appears (triggered by click or navigation).

    Args:
        accept: True to click OK/Yes, False for Cancel/No
        prompt_text: Text to enter for prompt dialogs
        profile: Chrome profile
        page_idx: Tab index

    Returns: (success, message)

    Note: Dialogs block the page. If you trigger an action that shows a dialog,
    you need to handle it before the page can continue.
    """
    port = _resolve_port(profile)
    ext = Extractor(port)
    if not ext.connect(page_idx, profile=profile):
        return (
            False,
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        ext.enable_domains()

        params = {"accept": accept}
        if prompt_text is not None:
            params["promptText"] = prompt_text

        ext.send_command("Page.handleJavaScriptDialog", params)
        action = "accepted" if accept else "dismissed"
        return True, f"Dialog {action}"
    except Exception as e:
        error_msg = str(e)
        if "No dialog is showing" in error_msg:
            return False, "No dialog currently showing"
        return False, error_msg
    finally:
        ext.close()


def chrome_dialog_auto(
    profile: str, action: str = "accept", page_idx: int = 0
) -> tuple[bool, str]:
    """Set up automatic dialog handling for the session.

    Args:
        action: 'accept' (auto-OK), 'dismiss' (auto-Cancel), or 'off' (manual)
        profile: Chrome profile
        page_idx: Tab index

    Returns: (success, message)

    This injects JS that intercepts window.alert, confirm, and prompt.
    Useful for automated browsing where dialogs would block execution.
    """
    port = _resolve_port(profile)
    ext = Extractor(port)
    if not ext.connect(page_idx, profile=profile):
        return (
            False,
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        ext.enable_domains()

        if action == "off":
            # Remove interceptors
            js = """
            if (window.__sfbOriginalAlert) {
                window.alert = window.__sfbOriginalAlert;
                window.confirm = window.__sfbOriginalConfirm;
                window.prompt = window.__sfbOriginalPrompt;
                delete window.__sfbOriginalAlert;
                delete window.__sfbOriginalConfirm;
                delete window.__sfbOriginalPrompt;
            }
            'disabled'
            """
        else:
            accept = action == "accept"
            js = f"""
            (() => {{
                if (!window.__sfbOriginalAlert) {{
                    window.__sfbOriginalAlert = window.alert;
                    window.__sfbOriginalConfirm = window.confirm;
                    window.__sfbOriginalPrompt = window.prompt;
                }}
                window.alert = function(msg) {{ 
                    console.log('[SFB] Auto-handled alert:', msg);
                    return undefined;
                }};
                window.confirm = function(msg) {{ 
                    console.log('[Agent] Auto-handled confirm:', msg, '-> {accept}');
                    return {str(accept).lower()};
                }};
                window.prompt = function(msg, def) {{ 
                    console.log('[Agent] Auto-handled prompt:', msg, '-> ', def || '');
                    return {'def || ""' if accept else "null"};
                }};
                return 'enabled';
            }})()
            """

        result = ext.evaluate_js(js)
        return True, f"Auto-dialog {result}: {action}"
    except Exception as e:
        return False, str(e)
    finally:
        ext.close()


def chrome_drag(
    from_selector: str, to_selector: str, profile: str, page_idx: int = 0
) -> tuple[bool, str]:
    """Drag element from one location to another.

    Args:
        from_selector: CSS selector for element to drag
        to_selector: CSS selector for drop target
        profile: Chrome profile
        page_idx: Tab index

    Returns: (success, message)

    Note: Uses HTML5 drag events. Works for most drag-and-drop implementations.
    """
    port = _resolve_port(profile)
    inter = Interactor(port)
    if not inter.connect(page_idx, profile=profile):
        return (
            False,
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        inter.enable_domains()

        # Get source element center
        from_center = inter.get_element_center(from_selector)
        if not from_center:
            return False, f"Source element not found: {from_selector}"

        # Get target element center
        to_center = inter.get_element_center(to_selector)
        if not to_center:
            return False, f"Target element not found: {to_selector}"

        # Synthesize drag sequence using mouse events
        # 1. Mouse down on source
        inter.send_command(
            "Input.dispatchMouseEvent",
            {
                "type": "mousePressed",
                "x": from_center["x"],
                "y": from_center["y"],
                "button": "left",
                "clickCount": 1,
            },
        )

        # 2. Mouse move to target (with intermediate steps for smooth drag)
        steps = 10
        for i in range(1, steps + 1):
            t = i / steps
            x = int(from_center["x"] + (to_center["x"] - from_center["x"]) * t)
            y = int(from_center["y"] + (to_center["y"] - from_center["y"]) * t)
            inter.send_command(
                "Input.dispatchMouseEvent",
                {"type": "mouseMoved", "x": x, "y": y, "button": "left"},
            )
            time.sleep(0.02)  # Small delay for realism

        # 3. Mouse up on target
        inter.send_command(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseReleased",
                "x": to_center["x"],
                "y": to_center["y"],
                "button": "left",
                "clickCount": 1,
            },
        )

        _trail_capture("drag", profile, page_idx, json.dumps({"from": from_selector, "to": to_selector}))
        return True, f"Dragged {from_selector} to {to_selector}"
    except Exception as e:
        return False, str(e)
    finally:
        inter.close()


def chrome_drag_coords(
    from_x: int, from_y: int, to_x: int, to_y: int, profile: str, page_idx: int = 0
) -> tuple[bool, str]:
    """Drag from one coordinate to another (for canvas/SVG elements).

    Args:
        from_x, from_y: Starting coordinates
        to_x, to_y: Ending coordinates
        profile: Chrome profile
        page_idx: Tab index

    Returns: (success, message)
    """
    port = _resolve_port(profile)
    ext = Extractor(port)
    if not ext.connect(page_idx, profile=profile):
        return (
            False,
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        ext.enable_domains()

        # Mouse down
        ext.send_command(
            "Input.dispatchMouseEvent",
            {
                "type": "mousePressed",
                "x": from_x,
                "y": from_y,
                "button": "left",
                "clickCount": 1,
            },
        )

        # Move in steps
        steps = 10
        for i in range(1, steps + 1):
            t = i / steps
            x = int(from_x + (to_x - from_x) * t)
            y = int(from_y + (to_y - from_y) * t)
            ext.send_command(
                "Input.dispatchMouseEvent",
                {"type": "mouseMoved", "x": x, "y": y, "button": "left"},
            )
            time.sleep(0.02)

        # Mouse up
        ext.send_command(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseReleased",
                "x": to_x,
                "y": to_y,
                "button": "left",
                "clickCount": 1,
            },
        )

        _trail_capture("drag_coords", profile, page_idx, json.dumps({"from_x": from_x, "from_y": from_y, "to_x": to_x, "to_y": to_y}))
        return True, f"Dragged from ({from_x},{from_y}) to ({to_x},{to_y})"
    except Exception as e:
        return False, str(e)
    finally:
        ext.close()


# Common device presets
DEVICE_PRESETS = {
    "iphone_14": {
        "width": 390,
        "height": 844,
        "deviceScaleFactor": 3,
        "mobile": True,
        "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
    },
    "iphone_14_pro_max": {
        "width": 430,
        "height": 932,
        "deviceScaleFactor": 3,
        "mobile": True,
        "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
    },
    "ipad_pro": {
        "width": 1024,
        "height": 1366,
        "deviceScaleFactor": 2,
        "mobile": True,
        "userAgent": "Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
    },
    "pixel_7": {
        "width": 412,
        "height": 915,
        "deviceScaleFactor": 2.625,
        "mobile": True,
        "userAgent": "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    },
    "galaxy_s24_ultra": {
        "width": 412,
        "height": 915,
        "deviceScaleFactor": 2.8,
        "mobile": True,
        "userAgent": "Mozilla/5.0 (Linux; Android 16; SM-S928U1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36",
    },
    "desktop_1080p": {
        "width": 1920,
        "height": 1080,
        "deviceScaleFactor": 1,
        "mobile": False,
        "userAgent": None,  # Keep browser default
    },
    "desktop_1440p": {
        "width": 2560,
        "height": 1440,
        "deviceScaleFactor": 1,
        "mobile": False,
        "userAgent": None,
    },
}


def chrome_emulate(
    profile: str,
    device: str = None,
    width: int = None,
    height: int = None,
    scale: float = None,
    mobile: bool = None,
    user_agent: str = None,
    page_idx: int = 0,
) -> tuple[bool, str]:
    """Emulate a device or set custom viewport.

    Args:
        device: Preset name (iphone_14, ipad_pro, pixel_7, galaxy_s24_ultra, desktop_1080p, etc.)
        width: Custom viewport width (overrides preset)
        height: Custom viewport height (overrides preset)
        scale: Device scale factor (overrides preset)
        mobile: Enable mobile mode (overrides preset)
        user_agent: Custom user agent (overrides preset)
        profile: Chrome profile
        page_idx: Tab index

    Returns: (success, message)

    Available presets: iphone_14, iphone_14_pro_max, ipad_pro, pixel_7, galaxy_s24_ultra,
                       desktop_1080p, desktop_1440p
    """
    port = _resolve_port(profile)
    ext = Extractor(port)
    if not ext.connect(page_idx, profile=profile):
        return (
            False,
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        ext.enable_domains()

        # Start with preset if specified
        config = {}
        if device:
            device_lower = device.lower().replace(" ", "_").replace("-", "_")
            if device_lower not in DEVICE_PRESETS:
                presets = ", ".join(DEVICE_PRESETS.keys())
                return False, f"Unknown device '{device}'. Available: {presets}"
            config = DEVICE_PRESETS[device_lower].copy()

        # Override with explicit parameters
        if width is not None:
            config["width"] = width
        if height is not None:
            config["height"] = height
        if scale is not None:
            config["deviceScaleFactor"] = scale
        if mobile is not None:
            config["mobile"] = mobile
        if user_agent is not None:
            config["userAgent"] = user_agent

        # Need at least width/height
        if "width" not in config or "height" not in config:
            return False, "Must specify device preset or width/height"

        # Set device metrics
        ext.send_command(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": config["width"],
                "height": config["height"],
                "deviceScaleFactor": config.get("deviceScaleFactor", 1),
                "mobile": config.get("mobile", False),
            },
        )

        # Set user agent if specified
        if config.get("userAgent"):
            ext.send_command(
                "Emulation.setUserAgentOverride", {"userAgent": config["userAgent"]}
            )

        desc = device or f"{config['width']}x{config['height']}"
        return True, f"Emulating {desc}"
    except Exception as e:
        return False, str(e)
    finally:
        ext.close()


def chrome_emulate_reset(profile: str, page_idx: int = 0) -> tuple[bool, str]:
    """Reset emulation to default (native browser viewport).

    Args:
        profile: Chrome profile
        page_idx: Tab index

    Returns: (success, message)
    """
    port = _resolve_port(profile)
    ext = Extractor(port)
    if not ext.connect(page_idx, profile=profile):
        return (
            False,
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    try:
        ext.enable_domains()
        ext.send_command("Emulation.clearDeviceMetricsOverride")
        ext.send_command("Emulation.setUserAgentOverride", {"userAgent": ""})
        return True, "Emulation reset to default"
    except Exception as e:
        return False, str(e)
    finally:
        ext.close()


def chrome_emulate_network(
    profile: str, condition: str = "offline", page_idx: int = 0
) -> tuple[bool, str]:
    """Emulate network conditions.

    Args:
        condition: 'offline', 'slow_3g', 'fast_3g', '4g', 'wifi', or 'online' (reset)
        profile: Chrome profile
        page_idx: Tab index

    Returns: (success, message)
    """
    port = _resolve_port(profile)
    ext = Extractor(port)
    if not ext.connect(page_idx, profile=profile):
        return (
            False,
            f"Chrome not running for profile '{profile}'. Use chrome_ensure('{profile}') to start it.",
        )

    # Network condition presets (latency in ms, throughput in bytes/sec)
    conditions = {
        "offline": {
            "offline": True,
            "latency": 0,
            "downloadThroughput": 0,
            "uploadThroughput": 0,
        },
        "slow_3g": {
            "offline": False,
            "latency": 2000,
            "downloadThroughput": 50000,
            "uploadThroughput": 50000,
        },
        "fast_3g": {
            "offline": False,
            "latency": 562,
            "downloadThroughput": 180000,
            "uploadThroughput": 84375,
        },
        "4g": {
            "offline": False,
            "latency": 170,
            "downloadThroughput": 1500000,
            "uploadThroughput": 750000,
        },
        "wifi": {
            "offline": False,
            "latency": 28,
            "downloadThroughput": 3750000,
            "uploadThroughput": 3750000,
        },
        "online": {
            "offline": False,
            "latency": -1,
            "downloadThroughput": -1,
            "uploadThroughput": -1,
        },  # -1 = no throttling
    }

    if condition not in conditions:
        return (
            False,
            f"Unknown condition '{condition}'. Available: {', '.join(conditions.keys())}",
        )

    try:
        ext.enable_domains()
        config = conditions[condition]
        ext.send_command("Network.emulateNetworkConditions", config)
        return True, f"Network: {condition}"
    except Exception as e:
        return False, str(e)
    finally:
        ext.close()


#
# Provides reliable element identification across all page types including
# Shadow DOM, iframes, and complex frameworks. Three-strategy fallback:
#   1. Accessibility tree (CDP: Accessibility.getFullAXTree) - best for compliant sites
#   2. DOM walk with shadow root piercing - fallback for modern frameworks
#   3. Basic querySelectorAll with bounding boxes - universal last resort
#

# Global cache for survey results - keyed by "profile:page_idx"
_survey_cache: dict[str, dict] = {}


def _get_accessibility_tree(client: CDPClient) -> list[dict]:
    """Get interactive elements from CDP Accessibility tree.

    Returns list of elements with: role, name, backendNodeId, bounds
    """
    try:
        # Enable Accessibility domain
        client.send_command("Accessibility.enable")

        # Get full AX tree
        result = client.send_command("Accessibility.getFullAXTree")
        nodes = result.get("nodes", [])

        elements = []
        for node in nodes:
            role = node.get("role", {}).get("value", "")
            name = node.get("name", {}).get("value", "")
            backend_node_id = node.get("backendDOMNodeId")

            # Filter for interactive roles
            interactive_roles = {
                "button",
                "link",
                "textbox",
                "checkbox",
                "radio",
                "combobox",
                "listbox",
                "menuitem",
                "tab",
                "switch",
                "searchbox",
                "spinbutton",
                "slider",
                "option",
            }

            if role in interactive_roles and backend_node_id:
                elem = {
                    "role": role,
                    "name": name,
                    "backend_node_id": backend_node_id,
                    "source": "accessibility",
                }

                # Try to get bounding box
                try:
                    box = client.send_command(
                        "DOM.getBoxModel", {"backendNodeId": backend_node_id}
                    )
                    if "model" in box:
                        content = box["model"]["content"]
                        elem["rect"] = {
                            "x": int(content[0]),
                            "y": int(content[1]),
                            "width": int(content[2] - content[0]),
                            "height": int(content[5] - content[1]),
                            "center_x": int((content[0] + content[2]) / 2),
                            "center_y": int((content[1] + content[5]) / 2),
                        }
                        elements.append(elem)
                except Exception:
                    pass  # Skip elements without valid boxes

        return elements
    except Exception as e:
        return []


def _get_dom_elements_with_shadow(client: CDPClient) -> list[dict]:
    """Walk DOM including Shadow roots to find interactive elements.

    Uses JavaScript to pierce Shadow DOM boundaries.
    """
    js_code = """
    (function() {
        const results = [];
        const interactive = 'button, a, input, select, textarea, [role="button"], [role="link"], [role="checkbox"], [role="radio"], [role="textbox"], [role="combobox"], [role="tab"], [role="switch"], [onclick], [tabindex]:not([tabindex="-1"])';
        
        function walkTree(root, depth = 0) {
            if (depth > 10) return;  // Prevent infinite recursion
            
            // Query visible interactive elements
            try {
                const elements = root.querySelectorAll(interactive);
                elements.forEach(el => {
                    const rect = el.getBoundingClientRect();
                    // Only include visible elements
                    if (rect.width > 0 && rect.height > 0 && rect.top >= 0 && rect.left >= 0) {
                        const role = el.getAttribute('role') || el.tagName.toLowerCase();
                        const name = el.getAttribute('aria-label') || 
                                    el.getAttribute('title') || 
                                    el.innerText?.slice(0, 100) || 
                                    el.getAttribute('name') || 
                                    el.getAttribute('placeholder') || '';
                        
                        results.push({
                            role: role,
                            name: name.trim(),
                            tag: el.tagName.toLowerCase(),
                            type: el.type || null,
                            href: el.href || null,
                            rect: {
                                x: Math.round(rect.x),
                                y: Math.round(rect.y),
                                width: Math.round(rect.width),
                                height: Math.round(rect.height),
                                center_x: Math.round(rect.x + rect.width / 2),
                                center_y: Math.round(rect.y + rect.height / 2)
                            },
                            source: 'dom_walk'
                        });
                    }
                });
            } catch(e) {}
            
            // Recurse into shadow roots
            try {
                const allElements = root.querySelectorAll('*');
                allElements.forEach(el => {
                    if (el.shadowRoot) {
                        walkTree(el.shadowRoot, depth + 1);
                    }
                });
            } catch(e) {}
        }
        
        walkTree(document);
        
        // Also check iframes (same-origin only)
        try {
            document.querySelectorAll('iframe').forEach(iframe => {
                try {
                    if (iframe.contentDocument) {
                        walkTree(iframe.contentDocument);
                    }
                } catch(e) {}  // Cross-origin will throw
            });
        } catch(e) {}
        
        return results;
    })()
    """

    try:
        result = client.send_command(
            "Runtime.evaluate",
            {"expression": js_code, "returnByValue": True, "awaitPromise": True},
        )
        return result.get("result", {}).get("value", []) or []
    except Exception as e:
        return []


def _get_basic_elements(client: CDPClient) -> list[dict]:
    """Fallback: simple querySelectorAll for basic interactive elements."""
    js_code = """
    (function() {
        const results = [];
        const selector = 'button, a, input, select, textarea, [onclick]';
        
        document.querySelectorAll(selector).forEach(el => {
            const rect = el.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                results.push({
                    role: el.tagName.toLowerCase(),
                    name: el.innerText?.slice(0, 100) || el.getAttribute('aria-label') || '',
                    tag: el.tagName.toLowerCase(),
                    rect: {
                        x: Math.round(rect.x),
                        y: Math.round(rect.y),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height),
                        center_x: Math.round(rect.x + rect.width / 2),
                        center_y: Math.round(rect.y + rect.height / 2)
                    },
                    source: 'basic'
                });
            }
        });
        
        return results;
    })()
    """

    try:
        result = client.send_command(
            "Runtime.evaluate",
            {"expression": js_code, "returnByValue": True, "awaitPromise": True},
        )
        return result.get("result", {}).get("value", []) or []
    except Exception as e:
        return []


def _dedupe_elements(elements: list[dict]) -> list[dict]:
    """Remove duplicate elements based on position and name."""
    seen = set()
    deduped = []

    for elem in elements:
        rect = elem.get("rect", {})
        # Key by position and name
        key = (
            rect.get("center_x", 0),
            rect.get("center_y", 0),
            elem.get("name", "")[:50],
        )

        if key not in seen:
            seen.add(key)
            deduped.append(elem)

    return deduped


def chrome_survey(profile: str, page_idx: int = 0) -> tuple[bool, dict, str]:
    """Survey page for all interactive elements with intelligent fallback.

    Args:
        profile: REQUIRED. One of: "default" or "sandbox"
        page_idx: Tab index (default 0)

    Returns:
        (success, survey_data, message)

    survey_data contains:
        url: current page URL
        timestamp: when surveyed
        elements: list of interactive elements with idx, role, name, rect
        method: which strategy succeeded ("accessibility", "dom_walk", "basic")
        element_count: total elements found

    Elements are indexed starting at 1 for human-friendly references.
    Use chrome_act(profile, idx) to interact with element by index.
    """
    if profile not in PROFILES:
        return (
            False,
            {},
            f"Unknown profile '{profile}'. Use chrome_ensure(profile) first. Valid: {list(PROFILES.keys())}",
        )

    port = _resolve_port(profile)
    cache_key = f"{profile}:{page_idx}"

    # Create client and connect
    client = CDPClient(port)
    if not client.connect(page_idx, profile=profile):
        return (
            False,
            {},
            f"Cannot connect to Chrome [{profile}] tab {page_idx}. Use chrome_ensure('{profile}') first.",
        )

    try:
        client.enable_domains()

        # Get current URL
        result = client.send_command(
            "Runtime.evaluate",
            {"expression": "window.location.href", "returnByValue": True},
        )
        current_url = result.get("result", {}).get("value", "unknown")

        elements = []
        method = "none"

        # Strategy 1: Accessibility tree (best for compliant sites)
        elements = _get_accessibility_tree(client)
        if elements:
            method = "accessibility"

        # Strategy 2: DOM walk with shadow piercing (fallback)
        if not elements:
            elements = _get_dom_elements_with_shadow(client)
            if elements:
                method = "dom_walk"

        # Strategy 3: Basic querySelectorAll (universal fallback)
        if not elements:
            elements = _get_basic_elements(client)
            if elements:
                method = "basic"

        # If accessibility tree gave results, merge with DOM walk for completeness
        if method == "accessibility":
            dom_elements = _get_dom_elements_with_shadow(client)
            if dom_elements:
                elements.extend(dom_elements)
                elements = _dedupe_elements(elements)

        # Sort by position (top-to-bottom, left-to-right)
        elements.sort(
            key=lambda e: (
                e.get("rect", {}).get("y", 9999),
                e.get("rect", {}).get("x", 9999),
            )
        )

        # Add 1-based indices
        for idx, elem in enumerate(elements, 1):
            elem["idx"] = idx

        survey_data = {
            "url": current_url,
            "timestamp": time.time(),
            "elements": elements,
            "method": method,
            "element_count": len(elements),
        }

        # Cache result
        _survey_cache[cache_key] = survey_data

        return True, survey_data, f"Surveyed {len(elements)} elements via {method}"

    except Exception as e:
        return False, {}, f"Survey failed: {e}"
    finally:
        client.close()


def chrome_act(
    profile: str,
    index: int,
    action: str = "click",
    value: str = None,
    page_idx: int = 0,
) -> tuple[bool, str]:
    """Act on element by index from last survey.

    Args:
        profile: REQUIRED. One of: "default" or "sandbox"
        index: Element index from chrome_survey() (1-based)
        action: "click", "fill", "focus", "check", "uncheck" (default: "click")
        value: Text to fill (required for "fill" action)
        page_idx: Tab index (default 0)

    Returns:
        (success, message)

    Uses cached bounding boxes for coordinate-based interaction.
    If cache is stale, automatically re-surveys.
    """
    if profile not in PROFILES:
        return (
            False,
            f"Unknown profile '{profile}'. Use chrome_ensure(profile) first. Valid: {list(PROFILES.keys())}",
        )

    cache_key = f"{profile}:{page_idx}"

    # Check cache
    survey = _survey_cache.get(cache_key)
    if not survey:
        # Auto-survey if no cache
        success, survey, msg = chrome_survey(profile, page_idx)
        if not success:
            return False, f"No survey cached and auto-survey failed: {msg}"

    # Find element by index
    elements = survey.get("elements", [])
    element = None
    for elem in elements:
        if elem.get("idx") == index:
            element = elem
            break

    if not element:
        return (
            False,
            f"Element {index} not found. Survey has {len(elements)} elements (1-{len(elements)})",
        )

    rect = element.get("rect")
    if not rect:
        return False, f"Element {index} has no position data"

    port = _resolve_port(profile)
    client = Interactor(port)

    if not client.connect(page_idx, profile=profile):
        return (
            False,
            f"Cannot connect to Chrome [{profile}] tab {page_idx}. Use chrome_ensure('{profile}') first.",
        )

    try:
        client.enable_domains()

        x = rect["center_x"]
        y = rect["center_y"]

        if action == "click":
            # Coordinate-based click
            client.send_command(
                "Input.dispatchMouseEvent",
                {
                    "type": "mousePressed",
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                },
            )
            client.send_command(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseReleased",
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                },
            )
            _trail_capture("act", profile, page_idx, json.dumps({"index": index, "action": "click"}))
            return (
                True,
                f"Clicked element {index} ({element.get('role', '?')}: {element.get('name', '')[:30]}) at ({x}, {y})",
            )

        elif action == "fill":
            if value is None:
                return False, "fill action requires value parameter"

            # Click to focus first
            client.send_command(
                "Input.dispatchMouseEvent",
                {
                    "type": "mousePressed",
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                },
            )
            client.send_command(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseReleased",
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                },
            )
            time.sleep(0.1)

            # Select all existing content
            client.send_command(
                "Input.dispatchKeyEvent",
                {
                    "type": "keyDown",
                    "key": "a",
                    "code": "KeyA",
                    "modifiers": 2,  # Ctrl
                },
            )
            client.send_command(
                "Input.dispatchKeyEvent",
                {"type": "keyUp", "key": "a", "code": "KeyA", "modifiers": 2},
            )

            # Type the new value
            for char in value:
                client.send_command(
                    "Input.dispatchKeyEvent", {"type": "char", "text": char}
                )

            _trail_capture("act", profile, page_idx, json.dumps({"index": index, "action": "fill", "value": value[:100]}))
            return (
                True,
                f"Filled element {index} with '{value[:30]}{'...' if len(value) > 30 else ''}'",
            )

        elif action == "focus":
            client.send_command(
                "Input.dispatchMouseEvent",
                {
                    "type": "mousePressed",
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                },
            )
            client.send_command(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseReleased",
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                },
            )
            _trail_capture("act", profile, page_idx, json.dumps({"index": index, "action": "focus"}))
            return True, f"Focused element {index}"

        elif action in ("check", "uncheck"):
            # For checkboxes - just click
            client.send_command(
                "Input.dispatchMouseEvent",
                {
                    "type": "mousePressed",
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                },
            )
            client.send_command(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseReleased",
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                },
            )
            _trail_capture("act", profile, page_idx, json.dumps({"index": index, "action": action}))
            return (
                True,
                f"{'Checked' if action == 'check' else 'Unchecked'} element {index}",
            )

        else:
            return (
                False,
                f"Unknown action '{action}'. Valid: click, fill, focus, check, uncheck",
            )

    except Exception as e:
        return False, f"Action failed: {e}"
    finally:
        client.close()


def chrome_find(
    profile: str, query: str, limit: int = 5, page_idx: int = 0
) -> tuple[bool, list, str]:
    """Find elements matching a query string. Token-efficient alternative to full survey.

    Args:
        profile: REQUIRED. One of: "default" or "sandbox"
        query: Text to search for in element names/roles (case-insensitive)
        limit: Maximum results to return (default 5)
        page_idx: Tab index (default 0)

    Returns:
        (success, matches, message)

    matches is a list of elements with idx, role, name (compact format).
    Uses cached survey if available, otherwise runs survey first.
    """
    if profile not in PROFILES:
        return (
            False,
            [],
            f"Unknown profile '{profile}'. Use chrome_ensure(profile) first. Valid: {list(PROFILES.keys())}",
        )

    cache_key = f"{profile}:{page_idx}"

    # Check cache or run survey
    survey = _survey_cache.get(cache_key)
    if not survey:
        success, survey, msg = chrome_survey(profile, page_idx)
        if not success:
            return False, [], f"Survey failed: {msg}"

    # Search elements
    query_lower = query.lower()
    matches = []

    for elem in survey.get("elements", []):
        name = (elem.get("name") or "").lower()
        role = (elem.get("role") or "").lower()
        tag = (elem.get("tag") or "").lower()
        href = (elem.get("href") or "").lower()

        # Match against name, role, tag, or href
        if (
            query_lower in name
            or query_lower in role
            or query_lower in tag
            or query_lower in href
        ):
            matches.append(
                {
                    "idx": elem.get("idx"),
                    "role": elem.get("role", ""),
                    "name": (elem.get("name") or "")[:60],
                }
            )

            if len(matches) >= limit:
                break

    if not matches:
        return True, [], f"No elements matching '{query}' found"

    return True, matches, f"Found {len(matches)} element(s) matching '{query}'"


def chrome_do(
    profile: str,
    query: str,
    action: str = "click",
    value: str = None,
    page_idx: int = 0,
    match_index: int = 1,
) -> tuple[bool, str]:
    """Find element by query and act on it in one call. Most token-efficient interaction.

    Args:
        profile: REQUIRED. One of: "default" or "sandbox"
        query: Text to search for in element names/roles
        action: "click", "fill", "focus", "check", "uncheck" (default: "click")
        value: Text to fill (required for "fill" action)
        page_idx: Tab index (default 0)
        match_index: Which match to use if multiple found (default 1 = first match)

    Returns:
        (success, message)

    Example:
        chrome_do("default", "submit", "click")  # Finds and clicks first "submit" element
        chrome_do("default", "search", "fill", "my query")  # Finds search box and types
    """
    # Find matching elements
    success, matches, msg = chrome_find(
        profile, query, limit=match_index + 2, page_idx=page_idx
    )

    if not success:
        return False, msg

    if not matches:
        return False, f"No element matching '{query}' found"

    if match_index > len(matches):
        return (
            False,
            f"Only {len(matches)} match(es) for '{query}', requested match #{match_index}",
        )

    # Get the target element
    target = matches[match_index - 1]
    element_idx = target["idx"]

    # Act on it
    success, result = chrome_act(profile, element_idx, action, value, page_idx)

    if success:
        _trail_capture("do", profile, page_idx, json.dumps({"query": query, "action": action, "value": value}))
        return True, f"{result} (matched '{query}')"
    else:
        return False, result


def chrome_survey_compact(profile: str, page_idx: int = 0) -> tuple[bool, str, str]:
    """Get compact text list of elements. Minimal token usage.

    Args:
        profile: REQUIRED. One of: "default" or "sandbox"
        page_idx: Tab index (default 0)

    Returns:
        (success, compact_text, message)

    Output format:
        [1] button: Submit
        [2] link: Home
        [3] textbox: Search
    """
    success, data, msg = chrome_survey(profile, page_idx)
    if not success:
        return False, "", msg

    lines = []
    for elem in data.get("elements", []):
        role = elem.get("role", "?")
        name = (elem.get("name") or "")[:50]
        if name:
            lines.append(f"[{elem.get('idx')}] {role}: {name}")
        else:
            lines.append(f"[{elem.get('idx')}] {role}")

    compact = "\n".join(lines)
    return True, compact, f"{len(lines)} elements"


def chrome_survey_clear(profile: str = None) -> str:
    """Clear survey cache.

    Args:
        profile: If specified, clear only that profile's cache. Otherwise clear all.

    Returns:
        Message about what was cleared.
    """
    global _survey_cache

    if profile:
        # Clear specific profile
        keys_to_remove = [k for k in _survey_cache if k.startswith(f"{profile}:")]
        for k in keys_to_remove:
            del _survey_cache[k]
        return f"Cleared {len(keys_to_remove)} cached surveys for profile '{profile}'"
    else:
        count = len(_survey_cache)
        _survey_cache = {}
        return f"Cleared all {count} cached surveys"


# =============================================================================
# CLI INTERFACE
# =============================================================================


def handle_navigate(args):
    nav = Navigator(args.port)

    if args.action == "list":
        pages = nav.list_pages()
        print(json.dumps(pages, indent=2))
        return

    if args.action == "new":
        nav.new_tab(args.url)
        print(json.dumps({"success": True, "action": "new_tab", "url": args.url}))
        return

    if args.action == "close":
        nav.close_tab(args.page_idx)
        print(
            json.dumps({"success": True, "action": "close_tab", "idx": args.page_idx})
        )
        return

    if not nav.connect(args.page_idx):
        sys.exit(1)

    try:
        nav.enable_domains()

        if args.action == "goto":
            nav.goto(args.url)
        elif args.action == "back":
            nav.go_back()
        elif args.action == "forward":
            nav.go_forward()
        elif args.action == "reload":
            nav.reload(args.ignore_cache)

        print(json.dumps({"success": True, "action": args.action}))
    finally:
        nav.close()


def handle_extract(args):
    ext = Extractor(args.port)
    if not ext.connect(args.page_idx):
        sys.exit(1)

    try:
        ext.enable_domains()

        if args.action == "js":
            result = ext.evaluate_js(args.expression)
            print(json.dumps({"success": True, "result": result}))
        elif args.action == "screenshot":
            ext.take_screenshot(args.output)
            print(json.dumps({"success": True, "file": args.output}))
        elif args.action == "look":
            # Capture screenshot and output base64 directly
            b64_data = ext.capture_screenshot_base64()
            # Also save to fixed path for reference
            output_path = str(Path(tempfile.gettempdir()) / "sfb_chrome_look.png")
            Path(output_path).write_bytes(base64.b64decode(b64_data))
            # Output base64 - can be decoded or used directly
            print(b64_data)
        elif args.action == "snapshot":
            result = ext.get_snapshot()
            print(
                json.dumps(
                    {"success": True, "snapshot": "DOM Tree captured (truncated)"}
                )
            )
        elif args.action == "network":
            ext.close()  # Close early, chrome_network manages its own connection
            success, results, error = chrome_network(
                args.profile, args.page_idx, args.clear
            )
            if success:
                print(json.dumps(results, indent=2))
            else:
                print(f"Error: {error}", file=sys.stderr)
                sys.exit(1)
            return
        elif args.action == "console":
            ext.close()
            success, results, error = chrome_console(
                args.profile, args.page_idx, args.clear
            )
            if success:
                print(json.dumps(results, indent=2))
            else:
                print(f"Error: {error}", file=sys.stderr)
                sys.exit(1)
            return

    finally:
        ext.close()


def handle_interact(args):
    inter = Interactor(args.port)
    if not inter.connect(args.page_idx):
        sys.exit(1)

    try:
        inter.enable_domains()

        if args.action == "click":
            inter.click(args.selector)
        elif args.action == "fill":
            inter.fill(args.selector, args.value)
        elif args.action == "key":
            inter.press_key(args.key)
            print(json.dumps({"success": True, "action": args.action}))
        elif args.action == "wait":
            inter.close()  # Close early, chrome_wait manages its own connection
            success, result = chrome_wait(
                args.profile, args.text, args.selector, args.timeout, args.page_idx
            )
            if success:
                print(result)
            else:
                print(f"Error: {result}", file=sys.stderr)
                sys.exit(1)
            return
        elif args.action == "hover":
            inter.close()
            success, result = chrome_hover(args.selector, args.profile, args.page_idx)
            if success:
                print(result)
            else:
                print(f"Error: {result}", file=sys.stderr)
                sys.exit(1)
            return
        elif args.action == "scroll":
            inter.close()
            success, result = chrome_scroll(
                args.profile, args.direction, args.amount, args.selector, args.page_idx
            )
            if success:
                print(result)
            else:
                print(f"Error: {result}", file=sys.stderr)
                sys.exit(1)
            return
        elif args.action == "focus":
            inter.close()
            success, result = chrome_focus(args.profile, args.page_idx)
            if success:
                print(result)
            else:
                print(f"Error: {result}", file=sys.stderr)
                sys.exit(1)
            return
        else:
            print(json.dumps({"success": True, "action": args.action}))
    finally:
        inter.close()


def handle_status(args):
    """Handle status command - purely observational."""
    if hasattr(args, "specific_port") and args.specific_port:
        # Check specific port
        result = check_port_status(args.specific_port)
        print(json.dumps(result, indent=2))
    else:
        # Check all known ports
        results = get_all_instance_status()
        if args.format == "table":
            print(f"{'PORT':<6} {'PROFILE':<10} {'TABS':<5} {'STATUS':<10} {'DESC'}")
            print("-" * 60)
            for r in results:
                status = "running" if r["running"] else "stopped"
                print(
                    f"{r['port']:<6} {r['profile']:<10} {r['tabs']:<5} {status:<10} {r['desc']}"
                )
        else:
            print(json.dumps(results, indent=2))


def handle_tabs(args):
    """Handle tabs command - list tabs without launching."""
    tabs = get_tabs_for_port(args.port)
    if not tabs:
        print(
            json.dumps(
                {
                    "success": False,
                    "message": f"No Chrome running on port {args.port} or no tabs open",
                }
            )
        )
        return

    if args.format == "table":
        print(f"{'IDX':<4} {'TITLE':<40} {'URL'}")
        print("-" * 80)
        for t in tabs:
            title = t["title"][:38] + ".." if len(t["title"]) > 40 else t["title"]
            print(f"{t['idx']:<4} {title:<40} {t['url']}")
    else:
        print(json.dumps(tabs, indent=2))


def handle_kill(args):
    """Handle kill command - profile-specific or all."""
    if args.all:
        result = chrome_kill("all")
    else:
        result = chrome_kill(args.profile)
    print(result)


def handle_survey(args):
    """Handle survey commands."""
    if args.action == "full":
        success, result, error = chrome_survey(args.profile, args.page_idx)
        if success:
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
    elif args.action == "compact":
        success, result, error = chrome_survey_compact(args.profile, args.page_idx)
        if success:
            print(result)
        else:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
    elif args.action == "clear":
        result = chrome_survey_clear(
            args.profile if hasattr(args, "clear_profile") else None
        )
        print(result)
    elif args.action == "find":
        success, results, error = chrome_find(
            args.profile, args.query, args.limit, args.page_idx
        )
        if success:
            print(json.dumps(results, indent=2))
        else:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
    elif args.action == "act":
        success, result = chrome_act(
            args.profile, args.index, args.act_action, args.value, args.page_idx
        )
        if success:
            print(result)
        else:
            print(f"Error: {result}", file=sys.stderr)
            sys.exit(1)
    elif args.action == "do":
        success, result = chrome_do(
            args.profile,
            args.query,
            args.do_action,
            args.value,
            args.page_idx,
            args.match_index,
        )
        if success:
            print(result)
        else:
            print(f"Error: {result}", file=sys.stderr)
            sys.exit(1)


def handle_read(args):
    """Handle read command."""
    success, content, error = chrome_read(args.profile, args.page_idx, args.format)
    if success:
        print(content)
    else:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


def handle_look(args):
    """Handle look (screenshot) command."""
    success, image_bytes, error = chrome_look(
        args.profile, args.page_idx, args.full_page, args.selector
    )
    if success:
        output = args.output
        with open(output, "wb") as f:
            f.write(image_bytes)
        print(f"{output} ({len(image_bytes)} bytes)")
    else:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


def handle_history(args):
    """Handle history command."""
    success, results, error = chrome_history(
        args.profile, args.query, args.days, args.limit
    )
    if success:
        print(json.dumps(results, indent=2))
    else:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


def handle_bookmarks(args):
    """Handle bookmarks commands."""
    if args.action == "list":
        success, results, error = chrome_bookmarks(
            args.profile, args.query, args.folder
        )
        if success:
            print(json.dumps(results, indent=2))
        else:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
    elif args.action == "add":
        success, result = chrome_bookmark_add(
            args.profile, args.url, args.name, args.folder
        )
        if success:
            print(result)
        else:
            print(f"Error: {result}", file=sys.stderr)
            sys.exit(1)
    elif args.action == "delete":
        success, result = chrome_bookmark_delete(args.profile, args.url, args.name)
        if success:
            print(result)
        else:
            print(f"Error: {result}", file=sys.stderr)
            sys.exit(1)


def handle_dialog(args):
    """Handle dialog commands."""
    if args.action == "handle":
        success, result = chrome_dialog_handle(
            args.profile, args.accept, args.prompt_text, args.page_idx
        )
        if success:
            print(result)
        else:
            print(f"Error: {result}", file=sys.stderr)
            sys.exit(1)
    elif args.action == "auto":
        success, result = chrome_dialog_auto(
            args.profile, args.dialog_action, args.page_idx
        )
        if success:
            print(result)
        else:
            print(f"Error: {result}", file=sys.stderr)
            sys.exit(1)


def handle_drag(args):
    """Handle drag commands."""
    if args.action == "element":
        success, result = chrome_drag(
            args.from_selector, args.to_selector, args.profile, args.page_idx
        )
        if success:
            print(result)
        else:
            print(f"Error: {result}", file=sys.stderr)
            sys.exit(1)
    elif args.action == "coords":
        success, result = chrome_drag_coords(
            args.from_x, args.from_y, args.to_x, args.to_y, args.profile, args.page_idx
        )
        if success:
            print(result)
        else:
            print(f"Error: {result}", file=sys.stderr)
            sys.exit(1)


def handle_emulate(args):
    """Handle emulate commands."""
    if args.action == "device":
        success, result = chrome_emulate(
            args.profile,
            args.device,
            args.width,
            args.height,
            args.scale,
            args.mobile,
            args.user_agent,
            args.page_idx,
        )
        if success:
            print(result)
        else:
            print(f"Error: {result}", file=sys.stderr)
            sys.exit(1)
    elif args.action == "reset":
        success, result = chrome_emulate_reset(args.profile, args.page_idx)
        if success:
            print(result)
        else:
            print(f"Error: {result}", file=sys.stderr)
            sys.exit(1)
    elif args.action == "network":
        success, result = chrome_emulate_network(
            args.profile, args.condition, args.page_idx
        )
        if success:
            print(result)
        else:
            print(f"Error: {result}", file=sys.stderr)
            sys.exit(1)


def handle_trail(args):
    """Handle trail commands."""
    action = getattr(args, "action", None)
    if not action:
        if args.command == "trail_start": action = "start"
        elif args.command == "trail_stop": action = "stop"
        elif args.command == "trail_status": action = "status"
    
    if action == "start":
        success, path = chrome_trail_start(args.name)
        print(json.dumps({"success": success, "trail_dir": path}, indent=2))
        if not success: sys.exit(1)
    elif action == "stop":
        success, summary = chrome_trail_stop()
        print(json.dumps({"success": success, **summary} if isinstance(summary, dict) else {"success": success, "result": summary}, indent=2))
        if not success: sys.exit(1)
    elif action == "status":
        result = chrome_trail_status()
        print(json.dumps(result, indent=2))


def handle_headless(args):
    """Handle headless commands."""
    action = getattr(args, "action", None)
    if not action:
        if args.command == "headless_start": action = "start"
        elif args.command == "headless_stop": action = "stop"
        elif args.command == "headless_list": action = "list"
        elif args.command == "headless_action": action = "action"
    
    if action == "start":
        success, info, err = chrome_headless_start(args.name)
        if success:
            print(json.dumps({"success": True, **info}, indent=2))
        else:
            print(json.dumps({"success": False, "error": err}, indent=2))
            sys.exit(1)
    elif action == "stop":
        success, msg = chrome_headless_stop(args.name)
        print(json.dumps({"success": success, "message": msg}, indent=2))
        if not success: sys.exit(1)
    elif action == "list":
        sessions = chrome_headless_list()
        print(json.dumps(sessions, indent=2))
    elif action == "action":
        kwargs = {}
        hl_args = args.hl_args or []
        if args.command == "headless_action":
            # CLI parity version
            action_name = args.hl_action_name
        else:
            # Nested version
            action_name = args.hl_action_name if hasattr(args, "hl_action_name") else None

        if action_name == "goto" and hl_args:
            kwargs["url"] = hl_args[0]
        elif action_name == "click" and hl_args:
            kwargs["selector"] = hl_args[0]
        elif action_name == "fill" and len(hl_args) >= 2:
            kwargs["selector"] = hl_args[0]
            kwargs["value"] = hl_args[1]
        elif action_name == "js" and hl_args:
            kwargs["expression"] = hl_args[0]
        elif action_name == "key" and hl_args:
            kwargs["key_name"] = hl_args[0]

        result = chrome_headless_action(args.name, action_name, **kwargs)
        print(json.dumps(result, indent=2, default=str))



def handle_trail(args):
    """Handle trail commands."""
    action = getattr(args, "action", None)
    if not action:
        if args.command == "trail_start": action = "start"
        elif args.command == "trail_stop": action = "stop"
        elif args.command == "trail_status": action = "status"
    
    if action == "start":
        success, path = chrome_trail_start(args.name)
        print(json.dumps({"success": success, "trail_dir": path}, indent=2))
        if not success: sys.exit(1)
    elif action == "stop":
        success, summary = chrome_trail_stop()
        print(json.dumps({"success": success, **summary} if isinstance(summary, dict) else {"success": success, "result": summary}, indent=2))
        if not success: sys.exit(1)
    elif action == "status":
        result = chrome_trail_status()
        print(json.dumps(result, indent=2))


def handle_headless(args):
    """Handle headless commands."""
    action = getattr(args, "action", None)
    if not action:
        if args.command == "headless_start": action = "start"
        elif args.command == "headless_stop": action = "stop"
        elif args.command == "headless_list": action = "list"
        elif args.command == "headless_action": action = "action"
    
    if action == "start":
        success, info, err = chrome_headless_start(args.name)
        if success:
            print(json.dumps({"success": True, **info}, indent=2))
        else:
            print(json.dumps({"success": False, "error": err}, indent=2))
            sys.exit(1)
    elif action == "stop":
        success, msg = chrome_headless_stop(args.name)
        print(json.dumps({"success": success, "message": msg}, indent=2))
        if not success: sys.exit(1)
    elif action == "list":
        sessions = chrome_headless_list()
        print(json.dumps(sessions, indent=2))
    elif action == "action":
        kwargs = {}
        hl_args = args.hl_args or []
        if args.command == "headless_action":
            # CLI parity version
            action_name = args.hl_action_name
        else:
            # Nested version
            action_name = args.hl_action_name if hasattr(args, "hl_action_name") else None

        if action_name == "goto" and hl_args:
            kwargs["url"] = hl_args[0]
        elif action_name == "click" and hl_args:
            kwargs["selector"] = hl_args[0]
        elif action_name == "fill" and len(hl_args) >= 2:
            kwargs["selector"] = hl_args[0]
            kwargs["value"] = hl_args[1]
        elif action_name == "js" and hl_args:
            kwargs["expression"] = hl_args[0]
        elif action_name == "key" and hl_args:
            kwargs["key_name"] = hl_args[0]

        result = chrome_headless_action(args.name, action_name, **kwargs)
        print(json.dumps(result, indent=2, default=str))

def main():
    # Logging
    _log("INFO", "start", f"Command: {sys.argv[1] if len(sys.argv) > 1 else '--help'}")
    profiles_help = ", ".join(f"{k} ({v['desc']})" for k, v in PROFILES.items())
    parser = argparse.ArgumentParser(
        description="Agent Chrome - Multi-Profile Automation",
        epilog=f"Profiles: {profiles_help}",
    )
    # -V (capital) for version: lowercase -v reserved for future --verbose flag alignment
    parser.add_argument("-V", "--version", action="version", version="1.2.0")
    parser.add_argument(
        "-M",
        "--help-markdown",
        action="store_true",
        help="Output help in markdown format",
    )
    parser.add_argument(
        "--profile",
        "-p",
        choices=PROFILES.keys(),
        required=True,
        help="Chrome profile (REQUIRED)",
    )
    parser.add_argument(
        "-P", "--port", type=int, help="Override port (default: profile-specific)"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Navigate Command
    nav_parser = subparsers.add_parser("navigate", help="Navigation commands")
    nav_subs = nav_parser.add_subparsers(dest="action", required=True)

    nav_subs.add_parser("list", help="List open tabs")

    goto_p = nav_subs.add_parser("goto", help="Navigate to URL")
    goto_p.add_argument("url")
    goto_p.add_argument("-i", "--page-idx", type=int, default=0)

    back_p = nav_subs.add_parser("back", help="Go back")
    back_p.add_argument("-i", "--page-idx", type=int, default=0)

    fwd_p = nav_subs.add_parser("forward", help="Go forward")
    fwd_p.add_argument("-i", "--page-idx", type=int, default=0)

    reload_p = nav_subs.add_parser("reload", help="Reload page")
    reload_p.add_argument("-I", "--ignore-cache", action="store_true")
    reload_p.add_argument("-i", "--page-idx", type=int, default=0)

    new_p = nav_subs.add_parser("new", help="New tab")
    new_p.add_argument("url", nargs="?", default="about:blank")

    close_p = nav_subs.add_parser("close", help="Close tab")
    close_p.add_argument("page_idx", type=int)

    # Extract Command
    ext_parser = subparsers.add_parser("extract", help="Extraction commands")
    ext_subs = ext_parser.add_subparsers(dest="action", required=True)

    js_p = ext_subs.add_parser("js", help="Evaluate JavaScript")
    js_p.add_argument("expression")
    js_p.add_argument("-i", "--page-idx", type=int, default=0)

    ss_p = ext_subs.add_parser("screenshot", help="Take screenshot")
    ss_p.add_argument("-o", "--output", required=True)
    ss_p.add_argument("-i", "--page-idx", type=int, default=0)

    look_p = ext_subs.add_parser(
        "look", help="Screenshot to fixed path for inline reading"
    )
    look_p.add_argument("-i", "--page-idx", type=int, default=0)

    snap_p = ext_subs.add_parser("snapshot", help="Get DOM snapshot")
    snap_p.add_argument("-i", "--page-idx", type=int, default=0)

    net_p = ext_subs.add_parser("network", help="Get network requests")
    net_p.add_argument("-c", "--clear", action="store_true", help="Clear after fetch")
    net_p.add_argument("-i", "--page-idx", type=int, default=0)

    console_p = ext_subs.add_parser("console", help="Get console messages")
    console_p.add_argument(
        "-c", "--clear", action="store_true", help="Clear after fetch"
    )
    console_p.add_argument("-i", "--page-idx", type=int, default=0)

    # Interact Command
    int_parser = subparsers.add_parser("interact", help="Interaction commands")
    int_subs = int_parser.add_subparsers(dest="action", required=True)

    click_p = int_subs.add_parser("click", help="Click element")
    click_p.add_argument("selector")
    click_p.add_argument("-i", "--page-idx", type=int, default=0)

    fill_p = int_subs.add_parser("fill", help="Fill input")
    fill_p.add_argument("selector")
    fill_p.add_argument("value")
    fill_p.add_argument("-i", "--page-idx", type=int, default=0)

    key_p = int_subs.add_parser("key", help="Press key")
    key_p.add_argument("key")
    key_p.add_argument("-i", "--page-idx", type=int, default=0)

    wait_p = int_subs.add_parser("wait", help="Wait for text/element")
    wait_p.add_argument("-t", "--text", help="Text to wait for")
    wait_p.add_argument("-s", "--selector", help="Selector to wait for")
    wait_p.add_argument("-T", "--timeout", type=int, default=10)
    wait_p.add_argument("-i", "--page-idx", type=int, default=0)

    hover_p = int_subs.add_parser("hover", help="Hover over element")
    hover_p.add_argument("selector")
    hover_p.add_argument("-i", "--page-idx", type=int, default=0)

    scroll_p = int_subs.add_parser("scroll", help="Scroll page")
    scroll_p.add_argument(
        "-d", "--direction", choices=["up", "down", "left", "right"], default="down"
    )
    scroll_p.add_argument(
        "-a", "--amount", type=int, default=3, help="Scroll ticks 1-10"
    )
    scroll_p.add_argument("-s", "--selector", help="Element to scroll within")
    scroll_p.add_argument("-i", "--page-idx", type=int, default=0)

    focus_p = int_subs.add_parser("focus", help="Focus browser window")
    focus_p.add_argument("-i", "--page-idx", type=int, default=0)

    # Status Command (observational)
    status_parser = subparsers.add_parser(
        "status", help="Check running Chrome instances (no launch)"
    )
    status_parser.add_argument(
        "-f",
        "--format",
        choices=["json", "table"],
        default="table",
        help="Output format",
    )
    status_parser.add_argument(
        "-C",
        "--check-port",
        type=int,
        dest="specific_port",
        help="Check specific port only",
    )

    # Tabs Command (observational)
    tabs_parser = subparsers.add_parser("tabs", help="List open tabs (no launch)")
    tabs_parser.add_argument(
        "-f",
        "--format",
        choices=["json", "table"],
        default="table",
        help="Output format",
    )

    # Kill Command (profile-specific by default)
    kill_parser = subparsers.add_parser(
        "kill", help="Kill Chrome instance (current profile only)"
    )
    kill_parser.add_argument(
        "-A", "--all", action="store_true", help="Kill ALL Chrome instances (nuclear)"
    )

    # CLI for chrome_ensure
    subparsers.add_parser("ensure", help="Ensure Chrome is running (launch if needed)")

    # Survey Command Group
    survey_parser = subparsers.add_parser("survey", help="Survey page elements")
    survey_subs = survey_parser.add_subparsers(dest="action", required=True)

    survey_full = survey_subs.add_parser("full", help="Full survey with details")
    survey_full.add_argument("-i", "--page-idx", type=int, default=0)

    survey_compact = survey_subs.add_parser("compact", help="Compact text list")
    survey_compact.add_argument("-i", "--page-idx", type=int, default=0)

    survey_subs.add_parser("clear", help="Clear survey cache")

    survey_find = survey_subs.add_parser("find", help="Find elements by query")
    survey_find.add_argument("query", help="Text to search for")
    survey_find.add_argument("-l", "--limit", type=int, default=5)
    survey_find.add_argument("-i", "--page-idx", type=int, default=0)

    survey_act = survey_subs.add_parser("act", help="Act on element by index")
    survey_act.add_argument("index", type=int, help="Element index from survey")
    survey_act.add_argument(
        "-a",
        "--action",
        dest="act_action",
        default="click",
        choices=["click", "fill", "focus", "check", "uncheck"],
    )
    survey_act.add_argument("-x", "--value", help="Value for fill action")
    survey_act.add_argument("-i", "--page-idx", type=int, default=0)

    survey_do = survey_subs.add_parser("do", help="Find and act in one call")
    survey_do.add_argument("query", help="Text to search for")
    survey_do.add_argument(
        "-a",
        "--action",
        dest="do_action",
        default="click",
        choices=["click", "fill", "focus", "check", "uncheck"],
    )
    survey_do.add_argument("-x", "--value", help="Value for fill action")
    survey_do.add_argument(
        "-m", "--match-index", type=int, default=1, help="Which match (1-based)"
    )
    survey_do.add_argument("-i", "--page-idx", type=int, default=0)

    # Read Command
    read_parser = subparsers.add_parser("read", help="Read page content")
    read_parser.add_argument(
        "-f", "--format", choices=["text", "html", "interactive"], default="text"
    )
    read_parser.add_argument("-i", "--page-idx", type=int, default=0)

    # Look Command (screenshot)
    look_parser = subparsers.add_parser("look", help="Screenshot a browser tab")
    look_parser.add_argument(
        "-i", "--page-idx", type=int, default=0, help="Tab index (default: 0)"
    )
    look_parser.add_argument(
        "-F", "--full-page", action="store_true", help="Capture full scrollable page"
    )
    look_parser.add_argument(
        "-s", "--selector", help="CSS selector to capture specific element"
    )
    look_parser.add_argument(
        "-o",
        "--output",
        default=str(Path(tempfile.gettempdir()) / "chrome_look.png"),
        help="Output path (default: <tempdir>/chrome_look.png)",
    )

    # History Command
    history_parser = subparsers.add_parser("history", help="Search browser history")
    history_parser.add_argument("-q", "--query", help="Search term")
    history_parser.add_argument("-d", "--days", type=int, default=7)
    history_parser.add_argument("-l", "--limit", type=int, default=50)

    # Bookmarks Command Group
    bm_parser = subparsers.add_parser("bookmarks", help="Bookmark operations")
    bm_subs = bm_parser.add_subparsers(dest="action", required=True)

    bm_list = bm_subs.add_parser("list", help="Search bookmarks")
    bm_list.add_argument("-q", "--query", help="Search term")
    bm_list.add_argument("-D", "--folder", help="Filter by folder")

    bm_add = bm_subs.add_parser("add", help="Add bookmark")
    bm_add.add_argument("url", help="URL to bookmark")
    bm_add.add_argument("-n", "--name", help="Bookmark name")
    bm_add.add_argument("-D", "--folder", default="bookmark_bar")

    bm_del = bm_subs.add_parser("delete", help="Delete bookmark")
    bm_del.add_argument("-u", "--url", help="URL to delete")
    bm_del.add_argument("-n", "--name", help="Name to delete")

    # Dialog Command Group
    dialog_parser = subparsers.add_parser("dialog", help="Handle JS dialogs")
    dialog_subs = dialog_parser.add_subparsers(dest="action", required=True)

    dialog_handle = dialog_subs.add_parser("handle", help="Handle dialog")
    dialog_handle.add_argument("-y", "--accept", action="store_true", default=True)
    dialog_handle.add_argument("-N", "--dismiss", action="store_false", dest="accept")
    dialog_handle.add_argument("-t", "--prompt-text", help="Text for prompt dialogs")
    dialog_handle.add_argument("-i", "--page-idx", type=int, default=0)

    dialog_auto = dialog_subs.add_parser("auto", help="Auto-handle dialogs")
    dialog_auto.add_argument("dialog_action", choices=["accept", "dismiss", "off"])
    dialog_auto.add_argument("-i", "--page-idx", type=int, default=0)

    # Drag Command Group
    drag_parser = subparsers.add_parser("drag", help="Drag operations")
    drag_subs = drag_parser.add_subparsers(dest="action", required=True)

    drag_elem = drag_subs.add_parser("element", help="Drag between elements")
    drag_elem.add_argument("from_selector", help="Source element")
    drag_elem.add_argument("to_selector", help="Target element")
    drag_elem.add_argument("-i", "--page-idx", type=int, default=0)

    drag_coords = drag_subs.add_parser("coords", help="Drag between coordinates")
    drag_coords.add_argument("from_x", type=int)
    drag_coords.add_argument("from_y", type=int)
    drag_coords.add_argument("to_x", type=int)
    drag_coords.add_argument("to_y", type=int)
    drag_coords.add_argument("-i", "--page-idx", type=int, default=0)

    # Emulate Command Group
    emu_parser = subparsers.add_parser("emulate", help="Device/network emulation")
    emu_subs = emu_parser.add_subparsers(dest="action", required=True)

    emu_device = emu_subs.add_parser("device", help="Emulate device")
    emu_device.add_argument(
        "-d", "--device", help="Preset: iphone_14, ipad_pro, pixel_7, etc."
    )
    emu_device.add_argument("-w", "--width", type=int)
    emu_device.add_argument("-H", "--height", type=int)
    emu_device.add_argument("-s", "--scale", type=float)
    emu_device.add_argument("-m", "--mobile", action="store_true")
    emu_device.add_argument("-u", "--user-agent")
    emu_device.add_argument("-i", "--page-idx", type=int, default=0)

    emu_subs.add_parser("reset", help="Reset emulation").add_argument(
        "-i", "--page-idx", type=int, default=0
    )

    emu_net = emu_subs.add_parser("network", help="Emulate network")
    emu_net.add_argument(
        "condition", choices=["offline", "slow_3g", "fast_3g", "4g", "wifi", "online"]
    )
    emu_net.add_argument("-i", "--page-idx", type=int, default=0)

    # Trail Command Group
    trail_parser = subparsers.add_parser("trail", help="Screenshot trail recording")
    trail_subs = trail_parser.add_subparsers(dest="action", required=True)

    trail_start_p = trail_subs.add_parser("start", help="Start recording trail")
    trail_start_p.add_argument("name", help="Trail name (creates directory under /tmp)")

    trail_subs.add_parser("stop", help="Stop recording trail")

    trail_subs.add_parser("status", help="Show trail recording status")

    # Headless Command Group
    headless_parser = subparsers.add_parser("headless", help="Parallel headless sessions")
    headless_subs = headless_parser.add_subparsers(dest="action", required=True)

    hl_start = headless_subs.add_parser("start", help="Start headless session")
    hl_start.add_argument("name", help="Session name")

    hl_stop = headless_subs.add_parser("stop", help="Stop headless session")
    hl_stop.add_argument("name", help="Session name")

    headless_subs.add_parser("list", help="List active headless sessions")

    hl_action = headless_subs.add_parser("action", help="Run action in headless session")
    hl_action.add_argument("name", help="Session name")
    hl_action.add_argument("hl_action_name", help="Action (goto, click, fill, etc.)")
    hl_action.add_argument("hl_args", nargs="*", help="Action arguments")

    # Replay Command
    replay_parser = subparsers.add_parser("replay", help="Replay a recorded trail")
    replay_parser.add_argument("script_path", help="Path to trail.json")
    replay_parser.add_argument("-d", "--delay", type=float, default=0.5, help="Delay between steps (seconds)")

    # Trail Command Group (Underscore versions for SFB parity)
    subparsers.add_parser("trail_start", help="Start recording trail").add_argument("name")
    subparsers.add_parser("trail_stop", help="Stop recording trail")
    subparsers.add_parser("trail_status", help="Show trail recording status")

    # Headless Command Group (Underscore versions for SFB parity)
    subparsers.add_parser("headless_start", help="Start headless session").add_argument("name")
    subparsers.add_parser("headless_stop", help="Stop headless session").add_argument("name")
    subparsers.add_parser("headless_list", help="List active headless sessions")
    hl_action_p = subparsers.add_parser("headless_action", help="Run action in headless session")
    hl_action_p.add_argument("name")
    hl_action_p.add_argument("hl_action_name")
    hl_action_p.add_argument("hl_args", nargs="*")

    # MCP server subcommand
    subparsers.add_parser("mcp-stdio", help="Run as MCP stdio server")

    # Handle --help-markdown before parse_args()
    if "--help-markdown" in sys.argv:
        print(f"## {parser.prog}\n")
        print(f"{parser.description}\n")
        print("### Arguments\n")
        for action in parser._actions:
            if action.dest == "help":
                continue
            flags = (
                ", ".join(action.option_strings)
                if action.option_strings
                else action.dest
            )
            print(f"- `{flags}`: {action.help or 'No description'}")
        if parser.epilog:
            print(f"\n{parser.epilog}")
        sys.exit(0)

    args = parser.parse_args()

    # Resolve port from profile
    if args.port is None:
        args.port = get_port(args.profile)

    try:
        if args.command == "mcp-stdio":
            _run_mcp()
            return

        # stdin support — read URL from pipe for navigate goto
        if hasattr(args, "url") and not args.url and not sys.stdin.isatty():
            args.url = sys.stdin.read().strip()

        if args.command == "status":
            handle_status(args)
        elif args.command == "tabs":
            handle_tabs(args)
        elif args.command == "kill":
            handle_kill(args)
        elif args.command == "ensure":
            success, status_dict, msg = chrome_ensure(args.profile)
            print(json.dumps({"message": msg, "status": status_dict}, indent=2))
        elif args.command in ("trail", "trail_start", "trail_stop", "trail_status"):
            handle_trail(args)
        elif args.command in (
            "headless",
            "headless_start",
            "headless_stop",
            "headless_list",
            "headless_action",
        ):
            handle_headless(args)
        else:
            # All other commands need Chrome running — auto-launch if needed
            ensure_chrome_cli(args.profile, args.port)

            if args.command == "navigate":
                handle_navigate(args)
            elif args.command == "extract":
                handle_extract(args)
            elif args.command == "interact":
                handle_interact(args)
            elif args.command == "survey":
                handle_survey(args)
            elif args.command == "read":
                handle_read(args)
            elif args.command == "look":
                handle_look(args)
            elif args.command == "history":
                handle_history(args)
            elif args.command == "bookmarks":
                handle_bookmarks(args)
            elif args.command == "dialog":
                handle_dialog(args)
            elif args.command == "drag":
                handle_drag(args)
            elif args.command == "emulate":
                handle_emulate(args)
            elif args.command == "replay":
                success, summary, err = chrome_replay(args.script_path, args.profile, args.delay)
                print(json.dumps({"success": success, **summary}, indent=2, default=str))
                if not success:
                    sys.exit(1)
    except (AssertionError, Exception) as e:
        _log("ERROR", args.command or "unknown", str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# FASTMCP SERVER
# =============================================================================


def _run_mcp():
    from fastmcp import FastMCP
    from fastmcp.utilities.types import Image

    mcp = FastMCP("chrome")

    @mcp.tool()
    def look(
        profile: str,
        page_idx: int = 0,
        full_page: bool = False,
        selector: str | None = None,
    ) -> Image:
        """Screenshot the current browser tab. Returns image inline.

        Args:
            profile: Chrome profile - "default" or "sandbox"
            page_idx: Tab index (default 0, first tab)
            full_page: Capture entire scrollable page (default: viewport only)
            selector: Capture specific element only (CSS selector)
        """
        success, image_bytes, error = chrome_look(
            profile, page_idx, full_page, selector
        )
        if not success:
            raise ValueError(error)
        return Image(data=image_bytes, format="png")

    @mcp.tool()
    def read(
        profile: str,
        format: str = "text",
        page_idx: int = 0,
    ) -> str:
        """Read current page content.

        Args:
            profile: Chrome profile
            format: Output format - "text", "html", or "interactive"
            page_idx: Tab index
        """
        success, content, error = chrome_read(profile, page_idx, format)
        if not success:
            raise ValueError(error)
        return content

    @mcp.tool()
    def goto(url: str, profile: str, page_idx: int = 0) -> str:
        """Navigate to URL.

        Args:
            url: URL to navigate to
            profile: Chrome profile
            page_idx: Tab index
        """
        success, msg = chrome_goto(url, profile, page_idx)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def tabs(profile: str) -> str:
        """List open tabs.

        Args:
            profile: Chrome profile
        """
        success, tab_list, error = chrome_tabs(profile)
        if not success:
            raise ValueError(error)
        return json.dumps(tab_list, indent=2)

    @mcp.tool()
    def click(selector: str, profile: str, page_idx: int = 0) -> str:
        """Click an element.

        Args:
            selector: CSS selector
            profile: Chrome profile
            page_idx: Tab index
        """
        success, msg = chrome_click(selector, profile, page_idx)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def fill(selector: str, value: str, profile: str, page_idx: int = 0) -> str:
        """Fill an input field.

        Args:
            selector: CSS selector
            value: Text to fill
            profile: Chrome profile
            page_idx: Tab index
        """
        success, msg = chrome_fill(selector, value, profile, page_idx)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def js(expression: str, profile: str, page_idx: int = 0) -> str:
        """Evaluate JavaScript in the page.

        Args:
            expression: JavaScript expression
            profile: Chrome profile
            page_idx: Tab index
        """
        success, result, error = chrome_js(expression, profile, page_idx)
        if not success:
            raise ValueError(error)
        return json.dumps(result) if not isinstance(result, str) else result

    @mcp.tool()
    def key(key_name: str, profile: str, page_idx: int = 0) -> str:
        """Press a keyboard key.

        Args:
            key_name: Key to press (e.g. "Enter", "Tab", "Escape")
            profile: Chrome profile
            page_idx: Tab index
        """
        success, msg = chrome_key(key_name, profile, page_idx)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def focus(profile: str, page_idx: int = 0) -> str:
        """Focus the browser window.

        Args:
            profile: Chrome profile
            page_idx: Tab index
        """
        success, msg = chrome_focus(profile, page_idx)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def wait(
        profile: str,
        text: str | None = None,
        selector: str | None = None,
        timeout: int = 10,
        page_idx: int = 0,
    ) -> str:
        """Wait for text or element to appear.

        Args:
            profile: Chrome profile
            text: Text to wait for
            selector: CSS selector to wait for
            timeout: Max wait seconds (default 10)
            page_idx: Tab index
        """
        success, msg = chrome_wait(profile, text, selector, timeout, page_idx)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def hover(selector: str, profile: str, page_idx: int = 0) -> str:
        """Hover over an element.

        Args:
            selector: CSS selector
            profile: Chrome profile
            page_idx: Tab index
        """
        success, msg = chrome_hover(selector, profile, page_idx)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def scroll(
        profile: str,
        direction: str = "down",
        amount: int = 3,
        selector: str | None = None,
        page_idx: int = 0,
    ) -> str:
        """Scroll the page or element.

        Args:
            profile: Chrome profile
            direction: up, down, left, right (default down)
            amount: Scroll ticks 1-10 (default 3)
            selector: Element to scroll within
            page_idx: Tab index
        """
        success, msg = chrome_scroll(profile, direction, amount, selector, page_idx)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def survey(profile: str, page_idx: int = 0) -> str:
        """Survey page elements (compact format).

        Args:
            profile: Chrome profile
            page_idx: Tab index
        """
        success, output, error = chrome_survey_compact(profile, page_idx)
        if not success:
            raise ValueError(error)
        return output

    @mcp.tool()
    def find(profile: str, query: str, limit: int = 5, page_idx: int = 0) -> str:
        """Find elements by text query.

        Args:
            profile: Chrome profile
            query: Text to search for
            limit: Max results (default 5)
            page_idx: Tab index
        """
        success, results, error = chrome_find(profile, query, limit, page_idx)
        if not success:
            raise ValueError(error)
        return json.dumps(results, indent=2)

    @mcp.tool()
    def act(
        profile: str,
        index: int,
        action: str = "click",
        value: str | None = None,
        page_idx: int = 0,
    ) -> str:
        """Act on element by index from last survey.

        Args:
            profile: Chrome profile
            index: Element index from survey (1-based)
            action: click, fill, focus, check, uncheck (default click)
            value: Text to fill (required for fill action)
            page_idx: Tab index
        """
        success, msg = chrome_act(profile, index, action, value, page_idx)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def do(
        query: str,
        profile: str,
        action: str = "click",
        value: str | None = None,
        match_index: int = 1,
        page_idx: int = 0,
    ) -> str:
        """Find element by text and act on it.

        Args:
            query: Text to search for
            profile: Chrome profile
            action: Action - "click", "fill", "focus", "check", "uncheck"
            value: Value for fill action
            match_index: Which match (1-based)
            page_idx: Tab index
        """
        success, msg = chrome_do(profile, query, action, value, page_idx, match_index)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def status() -> str:
        """Check status of all Chrome instances.

        Args:
            (no arguments)
        """
        return json.dumps(chrome_status(), indent=2)

    @mcp.tool()
    def kill(target: str) -> str:
        """Kill Chrome instance(s).

        Args:
            target: Profile name ("default" or "sandbox") or "all"
        """
        return chrome_kill(target)

    @mcp.tool()
    def ensure(profile: str) -> str:
        """Ensure Chrome is running for profile (launch if needed).

        Args:
            profile: Chrome profile to ensure is running
        """
        success, status_dict, msg = chrome_ensure(profile)
        if not success:
            raise ValueError(msg)
        return json.dumps({"message": msg, "status": status_dict}, indent=2)

    @mcp.tool()
    def network(profile: str, page_idx: int = 0, clear: bool = False) -> str:
        """Get network requests for the current page.

        Args:
            profile: Chrome profile
            page_idx: Tab index
            clear: Clear after fetch
        """
        success, results, error = chrome_network(profile, page_idx, clear)
        if not success:
            raise ValueError(error)
        return json.dumps(results, indent=2)

    @mcp.tool()
    def console(profile: str, page_idx: int = 0, clear: bool = False) -> str:
        """Get console messages from the page.

        Args:
            profile: Chrome profile
            page_idx: Tab index
            clear: Clear after fetch
        """
        success, results, error = chrome_console(profile, page_idx, clear)
        if not success:
            raise ValueError(error)
        return json.dumps(results, indent=2)

    @mcp.tool()
    def history(
        profile: str, query: str | None = None, days: int = 7, limit: int = 50
    ) -> str:
        """Search browser history.

        Args:
            profile: Chrome profile
            query: Search term
            days: How many days back (default 7)
            limit: Max results (default 50)
        """
        success, results, error = chrome_history(profile, query, days, limit)
        if not success:
            raise ValueError(error)
        return json.dumps(results, indent=2)

    @mcp.tool()
    def bookmarks(
        profile: str, query: str | None = None, folder: str | None = None
    ) -> str:
        """Search bookmarks.

        Args:
            profile: Chrome profile
            query: Search term
            folder: Filter by folder
        """
        success, results, error = chrome_bookmarks(profile, query, folder)
        if not success:
            raise ValueError(error)
        return json.dumps(results, indent=2)

    @mcp.tool()
    def bookmark_add(
        profile: str, url: str, name: str | None = None, folder: str = "bookmark_bar"
    ) -> str:
        """Add a bookmark.

        Args:
            profile: Chrome profile
            url: URL to bookmark
            name: Bookmark name
            folder: Target folder (default bookmark_bar)
        """
        success, msg = chrome_bookmark_add(profile, url, name, folder)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def bookmark_delete(
        profile: str, url: str | None = None, name: str | None = None
    ) -> str:
        """Delete a bookmark.

        Args:
            profile: Chrome profile
            url: URL to delete
            name: Name to delete
        """
        success, msg = chrome_bookmark_delete(profile, url, name)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def dialog(
        profile: str,
        accept: bool = True,
        prompt_text: str | None = None,
        page_idx: int = 0,
    ) -> str:
        """Handle a JavaScript dialog (accept/dismiss).

        Args:
            profile: Chrome profile
            accept: Accept the dialog (True) or dismiss (False)
            prompt_text: Text for prompt dialogs
            page_idx: Tab index
        """
        success, msg = chrome_dialog_handle(profile, accept, prompt_text, page_idx)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def dialog_handle(
        profile: str,
        accept: bool = True,
        prompt_text: str | None = None,
        page_idx: int = 0,
    ) -> str:
        """Handle a JavaScript dialog.

        Args:
            profile: Chrome profile
            accept: Accept or dismiss (default accept)
            prompt_text: Text for prompt dialogs
            page_idx: Tab index
        """
        success, msg = chrome_dialog_handle(profile, accept, prompt_text, page_idx)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def dialog_auto(profile: str, action: str = "accept", page_idx: int = 0) -> str:
        """Auto-handle JavaScript dialogs.

        Args:
            profile: Chrome profile
            action: accept, dismiss, or off
            page_idx: Tab index
        """
        success, msg = chrome_dialog_auto(profile, action, page_idx)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def drag(
        from_selector: str, to_selector: str, profile: str, page_idx: int = 0
    ) -> str:
        """Drag between elements.

        Args:
            from_selector: Source element CSS selector
            to_selector: Target element CSS selector
            profile: Chrome profile
            page_idx: Tab index
        """
        success, msg = chrome_drag(from_selector, to_selector, profile, page_idx)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def drag_coords(
        from_x: int, from_y: int, to_x: int, to_y: int, profile: str, page_idx: int = 0
    ) -> str:
        """Drag between coordinates.

        Args:
            from_x: Source X coordinate
            from_y: Source Y coordinate
            to_x: Target X coordinate
            to_y: Target Y coordinate
            profile: Chrome profile
            page_idx: Tab index
        """
        success, msg = chrome_drag_coords(from_x, from_y, to_x, to_y, profile, page_idx)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def emulate(
        profile: str,
        device: str | None = None,
        width: int | None = None,
        height: int | None = None,
        scale: float | None = None,
        mobile: bool | None = None,
        user_agent: str | None = None,
        page_idx: int = 0,
    ) -> str:
        """Emulate a device.

        Args:
            profile: Chrome profile
            device: Preset device name (iphone_14, ipad_pro, pixel_7, etc.)
            width: Custom width
            height: Custom height
            scale: Device scale factor
            mobile: Mobile mode
            user_agent: Custom user agent
            page_idx: Tab index
        """
        success, msg = chrome_emulate(
            profile, device, width, height, scale, mobile, user_agent, page_idx
        )
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def emulate_reset(profile: str, page_idx: int = 0) -> str:
        """Reset device emulation.

        Args:
            profile: Chrome profile
            page_idx: Tab index
        """
        success, msg = chrome_emulate_reset(profile, page_idx)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def emulate_network(
        profile: str, condition: str = "offline", page_idx: int = 0
    ) -> str:
        """Emulate network conditions.

        Args:
            profile: Chrome profile
            condition: offline, slow_3g, fast_3g, 4g, wifi, online
            page_idx: Tab index
        """
        success, msg = chrome_emulate_network(profile, condition, page_idx)
        if not success:
            raise ValueError(msg)
        return msg

    # --- Trail tools ---

    @mcp.tool()
    def trail_start(name: str) -> str:
        """Start recording a screenshot trail. Captures screenshots after every action.

        Args:
            name: Trail name (creates directory under data/chrome_trails/)
        """
        success, path = chrome_trail_start(name)
        if not success:
            raise ValueError(path)
        return f"Trail started: {path}"

    @mcp.tool()
    def trail_stop() -> str:
        """Stop recording the active trail. Saves trail.json for replay.

        Args:
            None
        """
        success, summary = chrome_trail_stop()
        if not success:
            raise ValueError(str(summary))
        return json.dumps(summary)

    @mcp.tool()
    def trail_status() -> str:
        """Check if a trail is currently recording.

        Args:
            None
        """
        result = chrome_trail_status()
        return json.dumps(result)

    # --- Headless session tools ---

    @mcp.tool()
    def headless_start(name: str) -> str:
        """Start a headless Chrome session for parallel automation.

        Args:
            name: Unique session name
        """
        success, info, err = chrome_headless_start(name)
        if not success:
            raise ValueError(err)
        return json.dumps(info)

    @mcp.tool()
    def headless_stop(name: str) -> str:
        """Stop a headless Chrome session.

        Args:
            name: Session name to stop
        """
        success, msg = chrome_headless_stop(name)
        if not success:
            raise ValueError(msg)
        return msg

    @mcp.tool()
    def headless_list() -> str:
        """List all active headless Chrome sessions.

        Args:
            None
        """
        sessions = chrome_headless_list()
        return json.dumps(sessions)

    @mcp.tool()
    def headless_action(
        name: str,
        action: str,
        url: str = "",
        selector: str = "",
        value: str = "",
        expression: str = "",
        key_name: str = "",
        text: str = "",
        query: str = "",
        direction: str = "down",
        amount: int = 3,
        timeout: int = 10,
    ) -> str:
        """Run an action in a headless Chrome session.

        Args:
            name: Session name
            action: Action to run (goto, click, fill, js, key, look, read, wait, hover, scroll, survey, find, do)
            url: URL for goto action
            selector: CSS selector for click/fill/hover/scroll
            value: Value for fill action
            expression: JS expression for js action
            key_name: Key name for key action
            text: Text for wait action
            query: Query for find/do action
            direction: Scroll direction (up/down/left/right)
            amount: Scroll amount (ticks)
            timeout: Wait timeout (seconds)
        """
        kwargs = {}
        if url:
            kwargs["url"] = url
        if selector:
            kwargs["selector"] = selector
        if value:
            kwargs["value"] = value
        if expression:
            kwargs["expression"] = expression
        if key_name:
            kwargs["key"] = key_name
        if text:
            kwargs["text"] = text
        if query:
            kwargs["query"] = query
        if action == "scroll":
            kwargs["direction"] = direction
            kwargs["amount"] = amount
        if action == "wait":
            kwargs["timeout"] = timeout
        result = chrome_headless_action(name, action, **kwargs)
        return json.dumps(result, default=str)

    # --- Replay tool ---

    @mcp.tool()
    def replay(script_path: str, profile: str, delay: float = 0.5) -> str:
        """Replay a recorded trail against a browser profile.

        Args:
            script_path: Path to trail.json file
            profile: Target Chrome profile
            delay: Delay between steps in seconds (default 0.5)
        """
        success, summary, err = chrome_replay(script_path, profile, delay)
        if not success:
            raise ValueError(err or "Replay had failures")
        return json.dumps(summary, default=str)

    print("chrome MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
