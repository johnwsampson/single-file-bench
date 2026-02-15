#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["edge-tts>=7.2.7", "fastmcp"]
# ///
"""Text-to-speech using Microsoft Edge TTS.

Usage:
    sfa_tts_edge.py speak <text> [--play]
    sfa_tts_edge.py save <text> <output_path>
    sfa_tts_edge.py mcp-stdio
"""

import os
import re
import sys
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
EXPOSED = ["speak", "save"]

DEFAULT_VOICE = "en-US-AvaMultilingualNeural"
DEFAULT_RATE = "+0%"
SILENCE_DELAY_MS = 500

CONFIG = {
    "default_voice": DEFAULT_VOICE,
    "default_rate": DEFAULT_RATE,
    "silence_delay_ms": SILENCE_DELAY_MS,
    "request_timeout_seconds": 60.0,
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

async def _generate_audio_impl(
    text: str,
    output_path: Path,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    pitch: str = "+0Hz",
    volume: str = "+0%",
) -> None:
    """Generate audio using Edge TTS.
    
    CLI: speak, save (internal)
    MCP: speak, save (internal)
    """
    import edge_tts
    
    communicate = edge_tts.Communicate(
        text, voice, rate=rate, pitch=pitch, volume=volume
    )
    await communicate.save(str(output_path))


def _play_audio_impl(file_path: Path) -> tuple[bool, str]:
    """Play audio file with platform-appropriate playback.
    
    Returns:
        Tuple of (success, error_message)
    """
    import platform
    import subprocess
    import tempfile
    
    system_name = platform.system().lower()
    
    # Try ffplay first (cross-platform)
    try:
        subprocess.run(
            [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                str(file_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
        return True, ""
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    
    # Platform-specific players
    if system_name == "darwin":  # macOS
        players = [
            (["afplay"], "afplay"),
        ]
    elif system_name == "linux":
        players = [
            (["aplay", "-q"], "aplay"),
            (["paplay"], "paplay"),
            (["mpg123", "-q"], "mpg123"),
        ]
    elif system_name == "windows":
        players = [
            (["powershell", "-Command", f'(New-Object Media.SoundPlayer "{file_path}").PlaySync()'], "powershell"),
        ]
    else:
        return False, f"Unsupported platform: {system_name}"
    
    for player_cmd, player_name in players:
        try:
            subprocess.run(
                player_cmd + [str(file_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=300,
            )
            return True, ""
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    
    return False, "No audio player found. Install ffplay, aplay, or paplay"


def _say_impl(text: str, output_path: Path | None = None, play: bool = False) -> tuple[dict, dict]:
    """Generate and optionally play TTS audio.
    
    CLI: speak, save
    MCP: speak, save
    
    Returns:
        Tuple of (result_dict, metrics_dict)
    """
    import asyncio
    import tempfile
    import time
    
    start_ms = time.time() * 1000
    
    # Clean text - remove bash escape sequences
    cleaned_text = text.replace("\\!", "!").replace("\\?", "?")
    
    if not cleaned_text or not cleaned_text.strip():
        latency = round(time.time() * 1000 - start_ms, 2)
        return (
            {"success": False, "error": "Text cannot be empty"},
            {"status": "error", "latency_ms": latency, "error_code": "VALIDATION_FAILED"}
        )
    
    try:
        # Determine output path
        if output_path is None:
            output_dir = Path(tempfile.gettempdir()) / "sfa_tts"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "response.mp3"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate audio
        asyncio.run(_generate_audio_impl(cleaned_text, output_path))
        
        # Play if requested
        if play:
            success, error = _play_audio_impl(output_path)
            if not success:
                latency = round(time.time() * 1000 - start_ms, 2)
                return (
                    {"success": False, "error": error, "output_path": str(output_path)},
                    {"status": "error", "latency_ms": latency}
                )
        
        latency = round(time.time() * 1000 - start_ms, 2)
        return (
            {"success": True, "output_path": str(output_path), "played": play},
            {"status": "success", "latency_ms": latency}
        )
        
    except Exception as e:
        latency = round(time.time() * 1000 - start_ms, 2)
        return (
            {"success": False, "error": str(e)},
            {"status": "error", "latency_ms": latency, "error_code": "TTS_FAILED"}
        )


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Text-to-speech using Microsoft Edge TTS")
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 1.0.0")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # mcp-stdio
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")
    
    # speak
    p_speak = subparsers.add_parser("speak", help="Generate and speak text")
    p_speak.add_argument("text", nargs="?", help="Text to speak")
    p_speak.add_argument("-p", "--play", action="store_true", default=True, help="Play audio (default)")
    p_speak.add_argument("-n", "--no-play", dest="play", action="store_false", help="Don't play audio")
    
    # save
    p_save = subparsers.add_parser("save", help="Generate and save to file")
    p_save.add_argument("text", nargs="?", help="Text to speak")
    p_save.add_argument("output_path", nargs="?", help="Output file path")
    
    args = parser.parse_args()
    
    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "speak":
            text = args.text
            if not text and not sys.stdin.isatty():
                text = sys.stdin.read().strip()
            assert text, "text required (positional argument or stdin)"
            result, metrics = _say_impl(text, play=args.play)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
        elif args.command == "save":
            text = args.text
            output_path = args.output_path
            if not text and not sys.stdin.isatty():
                parts = sys.stdin.read().strip().split(None, 1)
                text = parts[0] if parts else ""
                output_path = parts[1] if len(parts) > 1 else None
            assert text, "text required (positional argument or stdin: 'text output_path')"
            assert output_path, "output_path required (positional argument or stdin: 'text output_path')"
            result, metrics = _say_impl(text, output_path=Path(output_path), play=False)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
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
    
    mcp = FastMCP("tts-edge")
    
    @mcp.tool()
    def speak(text: str, play: bool = True) -> str:
        """Generate and play speech from text.
        
        Args:
            text: Text to speak
            play: Whether to play audio (default True)
        """
        result, metrics = _say_impl(text, play=play)
        return json.dumps({"result": result, "metrics": metrics})
    
    @mcp.tool()
    def save(text: str, output_path: str) -> str:
        """Generate speech and save to file.
        
        Args:
            text: Text to speak
            output_path: Path to save audio file
        """
        result, metrics = _say_impl(text, output_path=Path(output_path), play=False)
        return json.dumps({"result": result, "metrics": metrics})
    
    print("tts-edge MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
