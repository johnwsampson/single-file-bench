#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp", "httpx"]
# ///
"""Kokoro TTS via Docker - auto-starts container, keeps it running, provides shutdown.

Usage:
    sfa_voice_kokoro_docker.py say <text> [--play] [--voice VOICE]
    sfa_voice_kokoro_docker.py voices
    sfa_voice_kokoro_docker.py status
    sfa_voice_kokoro_docker.py start
    sfa_voice_kokoro_docker.py stop
    sfa_voice_kokoro_docker.py mcp-stdio
"""

import os
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# LOGGING (TSV format — see Principle 6)
# =============================================================================
_LEVELS = {"TRACE": 5, "DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40, "FATAL": 50}
# Environment variables are external - defensive access appropriate
_THRESHOLD = _LEVELS.get(os.environ.get("SFA_LOG_LEVEL", "INFO"), 20)
_LOG_DIR = os.environ.get("SFA_LOG_DIR", "")
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
EXPOSED = ["say", "voices", "status", "start", "stop"]

CONTAINER_NAME = "sfa-kokoro-tts"
CONTAINER_IMAGE = "ghcr.io/remsky/kokoro-fastapi-cpu:latest"
API_PORT = 8880
API_URL = f"http://127.0.0.1:{API_PORT}"

DEFAULT_VOICES = ["af_heart", "af_nicole", "af_kore"]

CONFIG = {
    "container_name": CONTAINER_NAME,
    "container_image": CONTAINER_IMAGE,
    "api_port": API_PORT,
    "api_url": API_URL,
    "default_voices": DEFAULT_VOICES,
    "request_timeout_seconds": 60.0,
    "health_check_timeout": 3.0,
    "startup_timeout_seconds": 120,
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def _run_docker_command(cmd: list[str]) -> tuple[bool, str, str]:
    """Run docker command, return (success, stdout, stderr)."""
    import subprocess
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        return result.returncode == 0, result.stdout, result.stderr
    except FileNotFoundError:
        return False, "", "docker command not found"


def _container_is_running_impl() -> tuple[bool, dict]:
    """Check if Kokoro container is running.
    
    CLI: status (internal)
    MCP: status (internal)
    """
    import json
    
    success, stdout, stderr = _run_docker_command([
        "docker", "ps", "--filter", f"name={CONTAINER_NAME}",
        "--format", "{{json .}}"
    ])
    
    if not success:
        return False, {"error": stderr}
    
    lines = stdout.strip().split("\n")
    for line in lines:
        if line.strip():
            try:
                container = json.loads(line)
                if container.get("State") == "running":
                    return True, {"container_id": container.get("ID", "")[:12]}
            except json.JSONDecodeError:
                continue
    
    return False, {}


def _container_exists_impl() -> bool:
    """Check if container exists (running or stopped)."""
    import json
    
    success, stdout, _ = _run_docker_command([
        "docker", "ps", "-a", "--filter", f"name={CONTAINER_NAME}",
        "--format", "{{json .}}"
    ])
    
    if not success:
        return False
    
    return bool(stdout.strip())


def _start_container_impl() -> tuple[dict, dict]:
    """Start Kokoro Docker container.
    
    CLI: start
    MCP: start
    
    Returns:
        Tuple of (result_dict, metrics_dict)
    """
    import time
    import httpx
    
    start_ms = time.time() * 1000
    
    # Check if already running
    is_running, info = _container_is_running_impl()
    if is_running:
        latency = round(time.time() * 1000 - start_ms, 2)
        return (
            {"success": True, "message": "Already running", "container_id": info.get("container_id")},
            {"status": "success", "latency_ms": latency}
        )
    
    # Check if container exists but is stopped - remove it
    if _container_exists_impl():
        _log("INFO", "container_cleanup", f"Removing stopped container {CONTAINER_NAME}")
        _run_docker_command(["docker", "rm", "-f", CONTAINER_NAME])
    
    # Pull latest image
    _log("INFO", "docker_pull", f"Pulling {CONTAINER_IMAGE}")
    success, stdout, stderr = _run_docker_command(["docker", "pull", CONTAINER_IMAGE])
    if not success:
        latency = round(time.time() * 1000 - start_ms, 2)
        return (
            {"success": False, "error": f"Failed to pull image: {stderr}"},
            {"status": "error", "latency_ms": latency}
        )
    
    # Start container
    _log("INFO", "docker_run", f"Starting container {CONTAINER_NAME}")
    success, stdout, stderr = _run_docker_command([
        "docker", "run", "-d", "--rm",
        "--name", CONTAINER_NAME,
        "-p", f"{API_PORT}:{API_PORT}",
        CONTAINER_IMAGE
    ])
    
    if not success:
        latency = round(time.time() * 1000 - start_ms, 2)
        return (
            {"success": False, "error": f"Failed to start container: {stderr}"},
            {"status": "error", "latency_ms": latency}
        )
    
    container_id = stdout.strip()[:12]
    
    # Wait for health check
    _log("INFO", "health_check", "Waiting for container to be ready")
    for i in range(int(CONFIG["startup_timeout_seconds"])):
        time.sleep(1)
        try:
            r = httpx.get(f"{API_URL}/health", timeout=CONFIG["health_check_timeout"])
            if r.status_code == 200:
                latency = round(time.time() * 1000 - start_ms, 2)
                return (
                    {"success": True, "container_id": container_id, "startup_time": i + 1},
                    {"status": "success", "latency_ms": latency}
                )
        except Exception:
            pass
    
    latency = round(time.time() * 1000 - start_ms, 2)
    return (
        {"success": False, "error": "Timeout waiting for container health"},
        {"status": "error", "latency_ms": latency}
    )


def _stop_container_impl() -> tuple[dict, dict]:
    """Stop Kokoro Docker container.
    
    CLI: stop
    MCP: stop
    """
    import time
    
    start_ms = time.time() * 1000
    
    is_running, _ = _container_is_running_impl()
    if not is_running:
        latency = round(time.time() * 1000 - start_ms, 2)
        return (
            {"success": True, "message": "Container not running"},
            {"status": "success", "latency_ms": latency}
        )
    
    _log("INFO", "docker_stop", f"Stopping container {CONTAINER_NAME}")
    success, stdout, stderr = _run_docker_command(["docker", "stop", CONTAINER_NAME])
    
    latency = round(time.time() * 1000 - start_ms, 2)
    
    if success:
        return (
            {"success": True, "message": "Container stopped"},
            {"status": "success", "latency_ms": latency}
        )
    else:
        return (
            {"success": False, "error": stderr},
            {"status": "error", "latency_ms": latency}
        )


def _ensure_running_impl() -> tuple[bool, str]:
    """Ensure container is running, start if needed.
    
    Returns:
        Tuple of (success, error_message)
    """
    is_running, _ = _container_is_running_impl()
    if is_running:
        return True, ""
    
    result, _ = _start_container_impl()
    if result.get("success"):
        return True, ""
    else:
        return False, result.get("error", "Unknown error starting container")


def _list_voices_impl() -> tuple[list[dict], dict]:
    """List available voices from Kokoro API.
    
    CLI: voices
    MCP: voices
    """
    import httpx
    import time
    
    start_ms = time.time() * 1000
    
    success, error = _ensure_running_impl()
    if not success:
        latency = round(time.time() * 1000 - start_ms, 2)
        return [], {"status": "error", "latency_ms": latency, "error": error}
    
    try:
        r = httpx.get(f"{API_URL}/v1/audio/voices", timeout=CONFIG["request_timeout_seconds"])
        if r.status_code == 200:
            data = r.json()
            voices = data.get("voices", [])
            latency = round(time.time() * 1000 - start_ms, 2)
            return voices, {"status": "success", "latency_ms": latency, "count": len(voices)}
        else:
            latency = round(time.time() * 1000 - start_ms, 2)
            return [], {"status": "error", "latency_ms": latency, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        latency = round(time.time() * 1000 - start_ms, 2)
        return [], {"status": "error", "latency_ms": latency, "error": str(e)}


def _play_audio_file(file_path: Path) -> bool:
    """Play audio file with platform-appropriate player."""
    import platform
    import subprocess
    
    system = platform.system().lower()
    players = (
        [["afplay"]] if system == "darwin" else
        [["aplay", "-q"], ["paplay"], ["mpg123", "-q"]] if system == "linux" else
        []
    )
    
    for cmd in players:
        try:
            subprocess.run(cmd + [str(file_path)], check=True, capture_output=True, timeout=60)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    
    _log("WARN", "audio_play", "No audio player found")
    return False


def _say_impl(text: str, voice: str | None = None, play: bool = True) -> tuple[dict, dict]:
    """Generate speech via Kokoro container with voice blending.
    
    CLI: say
    MCP: say
    
    Auto-starts container if not running. When no voice is specified,
    uses a blend of DEFAULT_VOICES (af_heart+af_nicole+af_kore).
    The API blends voice tensors before generation for a single unified voice.
    """
    import httpx
    import tempfile
    import time
    
    start_ms = time.time() * 1000
    
    if not text.strip():
        latency = round(time.time() * 1000 - start_ms, 2)
        return (
            {"success": False, "error": "Empty text"},
            {"status": "error", "latency_ms": latency, "error_code": "VALIDATION_FAILED"}
        )
    
    # Ensure container is running
    success, error = _ensure_running_impl()
    if not success:
        latency = round(time.time() * 1000 - start_ms, 2)
        return (
            {"success": False, "error": error},
            {"status": "error", "latency_ms": latency}
        )
    
    # Build voice string — API supports "voice1+voice2+voice3" for tensor blending
    voice_str = voice if voice else "+".join(DEFAULT_VOICES)
    
    output_dir = Path(tempfile.gettempdir()) / "sfa_kokoro"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        _log("INFO", "tts_request", f"Voice: {voice_str}, Text: {text[:50]}...")
        
        r = httpx.post(
            f"{API_URL}/v1/audio/speech",
            json={"model": "kokoro", "input": text, "voice": voice_str},
            timeout=CONFIG["request_timeout_seconds"]
        )
        
        if r.status_code != 200:
            latency = round(time.time() * 1000 - start_ms, 2)
            return (
                {"success": False, "error": f"TTS API error: {r.text}"},
                {"status": "error", "latency_ms": latency}
            )
        
        output_path = output_dir / f"speech_{int(time.time())}.mp3"
        output_path.write_bytes(r.content)
        
        # Play if requested
        if play:
            _play_audio_file(output_path)
        
        latency = round(time.time() * 1000 - start_ms, 2)
        return (
            {"success": True, "output_path": str(output_path), "voice": voice_str, "played": play},
            {"status": "success", "latency_ms": latency}
        )
        
    except Exception as e:
        latency = round(time.time() * 1000 - start_ms, 2)
        return (
            {"success": False, "error": str(e)},
            {"status": "error", "latency_ms": latency}
        )


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    import argparse
    import json
    import sys
    
    parser = argparse.ArgumentParser(description="Kokoro TTS via Docker")
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 1.0.0")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # mcp-stdio
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")
    
    # say
    p_say = subparsers.add_parser("say", help="Generate and play speech")
    p_say.add_argument("text", nargs="?", help="Text to speak")
    p_say.add_argument("-p", "--play", action="store_true", default=True, help="Play audio")
    p_say.add_argument("-n", "--no-play", dest="play", action="store_false", help="Don't play audio")
    p_say.add_argument("-v", "--voice", default=None, help="Voice ID (default: mix of af_heart+af_nicole+af_kore)")
    
    # voices
    subparsers.add_parser("voices", help="List available voices")
    
    # status
    subparsers.add_parser("status", help="Check container status")
    
    # start
    subparsers.add_parser("start", help="Start the Docker container")
    
    # stop
    subparsers.add_parser("stop", help="Stop the Docker container")
    
    args = parser.parse_args()
    
    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "say":
            text = args.text
            if not text and not sys.stdin.isatty():
                text = sys.stdin.read().strip()
            assert text, "text required (positional argument or stdin)"
            result, metrics = _say_impl(text, voice=args.voice, play=args.play)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
            sys.exit(0 if result.get("success") else 1)
        elif args.command == "voices":
            voices, metrics = _list_voices_impl()
            print(json.dumps({"voices": voices, "metrics": metrics}, indent=2))
        elif args.command == "status":
            is_running, info = _container_is_running_impl()
            print(json.dumps({
                "running": is_running,
                "container_name": CONTAINER_NAME,
                "api_url": API_URL,
                "info": info
            }, indent=2))
            sys.exit(0 if is_running else 1)
        elif args.command == "start":
            result, metrics = _start_container_impl()
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
            sys.exit(0 if result.get("success") else 1)
        elif args.command == "stop":
            result, metrics = _stop_container_impl()
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
            sys.exit(0 if result.get("success") else 1)
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
    import json
    
    mcp = FastMCP("kokoro-docker")
    
    @mcp.tool()
    def say(text: str, voice: str = "", play: bool = True) -> str:
        """Generate and play speech via Kokoro TTS.
        
        Auto-starts Docker container if not running.
        Default uses a mix of af_heart, af_nicole, and af_kore voices.
        
        Args:
            text: Text to speak
            voice: Voice ID (empty for default 3-voice mix)
            play: Whether to play audio (default True)
        """
        result, metrics = _say_impl(text, voice=voice or None, play=play)
        return json.dumps({"result": result, "metrics": metrics})
    
    @mcp.tool()
    def voices() -> str:
        """List available voices from Kokoro TTS.
        
        Args:
            None
        """
        voices_list, metrics = _list_voices_impl()
        return json.dumps({"voices": voices_list, "metrics": metrics})
    
    @mcp.tool()
    def status() -> str:
        """Check Docker container status.
        
        Args:
            None
        """
        is_running, info = _container_is_running_impl()
        return json.dumps({
            "running": is_running,
            "container_name": CONTAINER_NAME,
            "api_url": API_URL,
            "info": info
        })
    
    @mcp.tool()
    def start() -> str:
        """Start the Kokoro Docker container.
        
        Args:
            None
        """
        result, metrics = _start_container_impl()
        return json.dumps({"result": result, "metrics": metrics})
    
    @mcp.tool()
    def stop() -> str:
        """Stop the Kokoro Docker container.
        
        Args:
            None
        """
        result, metrics = _stop_container_impl()
        return json.dumps({"result": result, "metrics": metrics})
    
    print("kokoro-docker MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
