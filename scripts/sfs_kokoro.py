#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "kokoro>=0.9.4",
#   "soundfile",
#   "torch",
#   "numpy",
#   "fastmcp",
#   "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
# ]
# ///
"""Kokoro-82M TTS HTTP server with Apple Silicon MPS acceleration.

Long-running daemon that serves text-to-speech via HTTP. Preloads model
and default voice blend on startup for fast response times. Managed by
startup.sh alongside the event bus.

Usage:
    sfs_kokoro.py serve [--host HOST] [--port PORT] [--background]
    sfs_kokoro.py stop
    sfs_kokoro.py status
    sfs_kokoro.py mcp-stdio
"""

import argparse
import io
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# =============================================================================
# LOGGING (TSV format — see Principle 6)
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
EXPOSED = ["status", "stop"]

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SANCTUM_DIR = Path.home() / ".sanctum"
PID_FILE = SANCTUM_DIR / "kokoro.pid"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart+af_nicole+af_kore"
MODEL_REPO = "hexgrad/Kokoro-82M"
LANG_CODE = "a"

VOICES = {
    "af": ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
           "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"],
    "am": ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
           "am_michael", "am_onyx", "am_puck", "am_santa"],
    "bf": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily"],
    "bm": ["bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
}

CONFIG = {
    "default_host": DEFAULT_HOST,
    "default_port": DEFAULT_PORT,
    "sample_rate_hz": SAMPLE_RATE,
    "default_voice": DEFAULT_VOICE,
    "model_repo": MODEL_REPO,
    "lang_code": LANG_CODE,
    "default_speed": 1.0,
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================
_pipeline = None
_voice_cache: dict = {}
_pipeline_lock = threading.Lock()
_server_start_time = 0.0
_running = False


def _get_pipeline():
    """Lazy-load and cache KPipeline. Downloads model on first use."""
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        _log("INFO", "pipeline_init", f"Loading KPipeline lang_code={LANG_CODE}")
        _pipeline = KPipeline(lang_code=LANG_CODE, repo_id=MODEL_REPO)
        _log("INFO", "pipeline_ready", "KPipeline loaded")
    return _pipeline


def _load_voice(voice: str):
    """Load a voice — string name or blended tensor from 'name1+name2' syntax."""
    if voice in _voice_cache:
        return _voice_cache[voice]

    if "+" not in voice:
        return voice

    import torch
    from huggingface_hub import hf_hub_download

    names = [n.strip() for n in voice.split("+")]
    assert len(names) >= 2, f"Voice blend requires 2+ voices (got {names})"

    tensors = []
    for name in names:
        path = hf_hub_download(repo_id=MODEL_REPO, filename=f"voices/{name}.pt")
        tensor = torch.load(path, weights_only=True)
        tensors.append(tensor)
        _log("DEBUG", "voice_load", f"Loaded voice tensor: {name}")

    blended = torch.stack(tensors).mean(dim=0)
    _voice_cache[voice] = blended
    _log("INFO", "voice_blend", f"Blended {len(names)} voices: {'+'.join(names)}")
    return blended


def _generate(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> bytes:
    """Generate WAV audio bytes from text. Thread-safe via pipeline lock."""
    import numpy as np
    import soundfile as sf

    assert text and text.strip(), "text must not be empty"

    with _pipeline_lock:
        pipeline = _get_pipeline()
        voice_input = _load_voice(voice)

        audio_chunks = []
        for _gs, _ps, audio in pipeline(text, voice=voice_input, speed=speed, split_pattern=r"\n+"):
            audio_chunks.append(audio)

        assert audio_chunks, "Kokoro pipeline produced no audio chunks"
        full_audio = np.concatenate(audio_chunks)

    buf = io.BytesIO()
    sf.write(buf, full_audio, SAMPLE_RATE, format="WAV")
    return buf.getvalue()


# =============================================================================
# HTTP SERVER
# =============================================================================
class _Handler(BaseHTTPRequestHandler):
    """HTTP handler for Kokoro TTS API."""

    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {
                "status": "ok",
                "model": "Kokoro-82M",
                "uptime": round(time.time() - _server_start_time, 1),
                "default_voice": DEFAULT_VOICE,
                "voices_cached": len(_voice_cache),
            })
        elif self.path == "/voices":
            self._json_response(200, {"voices": VOICES, "default": DEFAULT_VOICE})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/say":
            self._handle_say()
        else:
            self.send_error(404)

    def _handle_say(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            assert length > 0, "Request body required"
            body = json.loads(self.rfile.read(length))
            text = body.get("text", "")
            voice = body.get("voice", DEFAULT_VOICE)
            speed = body.get("speed", 1.0)

            start = time.time()
            wav_bytes = _generate(text, voice, speed)
            latency = round((time.time() - start) * 1000, 1)

            _log("INFO", "say", f"Generated {len(wav_bytes)} bytes",
                 detail=f"voice={voice} speed={speed} text_len={len(text)}",
                 metrics=f"latency_ms={latency}")

            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav_bytes)))
            self.send_header("X-Latency-Ms", str(latency))
            self.end_headers()
            self.wfile.write(wav_bytes)
        except (AssertionError, json.JSONDecodeError, KeyError) as e:
            self._json_response(400, {"error": str(e)})
        except Exception as e:
            _log("ERROR", "say_fail", str(e))
            self._json_response(500, {"error": str(e)})

    def _json_response(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        _log("DEBUG", "http", fmt % args)


# =============================================================================
# SERVER LIFECYCLE
# =============================================================================
def _serve_impl(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, background: bool = False):
    """Start the Kokoro TTS HTTP server.

    CLI: serve
    Preloads model and default voice before accepting connections.
    """
    global _server_start_time, _running

    SANCTUM_DIR.mkdir(parents=True, exist_ok=True)

    # Preload model and default voice
    print(f"Loading Kokoro-82M model (this may download ~200MB on first run)...", file=sys.stderr)
    _get_pipeline()
    _load_voice(DEFAULT_VOICE)
    print(f"Model and voice blend ready.", file=sys.stderr)

    _server_start_time = time.time()
    _running = True

    # Write PID file
    PID_FILE.write_text(str(os.getpid()))

    server = ThreadingHTTPServer((host, port), _Handler)

    def _shutdown(signum, frame):
        global _running
        _running = False
        _log("INFO", "shutdown", f"Signal {signum} received")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    _log("INFO", "server_start", f"Listening on {host}:{port}, pid={os.getpid()}")
    print(f"Kokoro TTS server listening on http://{host}:{port} (pid {os.getpid()})", file=sys.stderr)

    try:
        server.serve_forever()
    finally:
        server.server_close()
        if PID_FILE.exists():
            PID_FILE.unlink()
        _log("INFO", "server_stop", f"uptime={round(time.time() - _server_start_time, 1)}s")
        print("Kokoro TTS server stopped.", file=sys.stderr)


def _daemonize(host: str, port: int):
    """Fork into background daemon."""
    pid = os.fork()
    if pid > 0:
        print(f"Kokoro daemon started (pid {pid})", file=sys.stderr)
        sys.exit(0)
    os.setsid()
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    # Keep stderr for logging
    _serve_impl(host, port, background=True)


def _stop_impl() -> tuple[dict, dict]:
    """Stop a running Kokoro server via PID file.

    CLI: stop
    MCP: stop
    """
    start_ms = time.time() * 1000

    if not PID_FILE.exists():
        latency = round(time.time() * 1000 - start_ms, 2)
        return {"success": True, "message": "No PID file — server not running"}, {"status": "success", "latency_ms": latency}

    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        _log("INFO", "stop", f"Sent SIGTERM to pid {pid}")
        latency = round(time.time() * 1000 - start_ms, 2)
        return {"success": True, "message": f"Sent SIGTERM to pid {pid}"}, {"status": "success", "latency_ms": latency}
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        latency = round(time.time() * 1000 - start_ms, 2)
        return {"success": True, "message": f"Process {pid} not found — cleaned up stale PID file"}, {"status": "success", "latency_ms": latency}


def _status_impl(port: int = DEFAULT_PORT) -> tuple[dict, dict]:
    """Check Kokoro server health.

    CLI: status
    MCP: status
    """
    import urllib.request
    import urllib.error

    start_ms = time.time() * 1000
    url = f"http://{DEFAULT_HOST}:{port}/health"

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            latency = round(time.time() * 1000 - start_ms, 2)
            return {**data, "running": True}, {"status": "success", "latency_ms": latency}
    except (urllib.error.URLError, ConnectionError, OSError):
        latency = round(time.time() * 1000 - start_ms, 2)
        return {"running": False, "message": f"Server not responding on port {port}"}, {"status": "success", "latency_ms": latency}


# =============================================================================
# CLI INTERFACE
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        prog="sfs_kokoro.py",
        description="Kokoro-82M TTS HTTP server with MPS acceleration",
    )
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="Start TTS server")
    p_serve.add_argument("-H", "--host", default=DEFAULT_HOST, help=f"Bind address (default: {DEFAULT_HOST})")
    p_serve.add_argument("-p", "--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    p_serve.add_argument("-b", "--background", action="store_true", help="Run as background daemon")

    p_stop = sub.add_parser("stop", help="Stop running server")

    p_status = sub.add_parser("status", help="Check server health")
    p_status.add_argument("-p", "--port", type=int, default=DEFAULT_PORT)

    sub.add_parser("mcp-stdio", help="Run as MCP server")

    args = parser.parse_args()

    if args.command == "mcp-stdio":
        _run_mcp()
        return

    # Handle stdin for piped commands
    if not sys.stdin.isatty() and not args.command:
        data = sys.stdin.read().strip()
        if data:
            try:
                obj = json.loads(data)
                args.command = obj.get("command", "status")
            except json.JSONDecodeError:
                args.command = "status"

    try:
        if args.command == "serve":
            if args.background:
                _daemonize(args.host, args.port)
            else:
                _serve_impl(args.host, args.port)
        elif args.command == "stop":
            result, _ = _stop_impl()
            print(json.dumps(result, indent=2))
        elif args.command == "status":
            result, _ = _status_impl(args.port)
            print(json.dumps(result, indent=2))
        else:
            parser.print_help()
    except (AssertionError, Exception) as e:
        _log("ERROR", args.command or "unknown", str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# FASTMCP SERVER (lazy — only loaded when mcp-stdio is invoked)
# =============================================================================
def _run_mcp():
    from fastmcp import FastMCP
    mcp = FastMCP("kokoro")

    @mcp.tool()
    def status(port: int = DEFAULT_PORT) -> str:
        """Check Kokoro TTS server health.

        Args:
            port: Server port (default: 8787)

        Returns:
            JSON with server status, uptime, model info
        """
        result, _ = _status_impl(port)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def stop() -> str:
        """Stop the Kokoro TTS server.

        Args:

        Returns:
            JSON confirmation
        """
        result, _ = _stop_impl()
        return json.dumps(result, indent=2)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
