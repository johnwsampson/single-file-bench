#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "kokoro>=0.9.4",
#   "soundfile",
#   "torch",
#   "fastmcp",
#   "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
# ]
# ///
"""Native Kokoro-82M text-to-speech with Apple Silicon MPS acceleration.

Runs Kokoro-82M locally via PyTorch. Supports voice blending by averaging
voice tensors before generation. Default voice: af_heart+af_nicole+af_kore.

Usage:
    sfa_tts_kokoro.py say <text> [--voice VOICE] [--speed SPEED] [--no-play]
    sfa_tts_kokoro.py save <text> <output_path> [--voice VOICE] [--speed SPEED]
    sfa_tts_kokoro.py voices
    sfa_tts_kokoro.py mcp-stdio
"""

import os
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
EXPOSED = ["say", "save", "voices"]

# Enable MPS fallback for Apple Silicon GPU acceleration
# Environment variables are external - defensive access appropriate
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DEFAULT_VOICE_BLEND = ["af_heart", "af_nicole", "af_kore"]
SAMPLE_RATE_HZ = 24000

CONFIG = {
    "default_voice_blend": DEFAULT_VOICE_BLEND,
    "sample_rate_hz": SAMPLE_RATE_HZ,
    "default_speed": 1.0,
    "model_repo": "hexgrad/Kokoro-82M",
    "lang_code": "a",
}

# Known voices from hexgrad/Kokoro-82M
VOICES = {
    "af": ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
           "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"],
    "am": ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
           "am_michael", "am_onyx", "am_puck", "am_santa"],
    "bf": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily"],
    "bm": ["bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

_pipeline_cache = None
_voice_cache: dict = {}


def _get_pipeline():
    """Lazy-load and cache KPipeline. Downloads model on first use."""
    global _pipeline_cache
    if _pipeline_cache is None:
        from kokoro import KPipeline

        _log("INFO", "pipeline_init", f"Loading KPipeline lang_code={CONFIG['lang_code']}")
        _pipeline_cache = KPipeline(lang_code=CONFIG["lang_code"], repo_id=CONFIG["model_repo"])
        _log("INFO", "pipeline_ready", "KPipeline loaded")
    return _pipeline_cache


def _load_voice(voice: str):
    """Load a voice — string name or blended tensor from 'name1+name2+name3' syntax.

    Single names are passed through as strings (pipeline handles loading).
    Blended voices load .pt tensors from HuggingFace and average them.
    Results are cached for reuse.
    """
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
        path = hf_hub_download(repo_id=CONFIG["model_repo"], filename=f"voices/{name}.pt")
        tensor = torch.load(path, weights_only=True)
        tensors.append(tensor)
        _log("DEBUG", "voice_load", f"Loaded voice tensor: {name}")

    blended = torch.stack(tensors).mean(dim=0)
    _voice_cache[voice] = blended
    _log("INFO", "voice_blend", f"Blended {len(names)} voices: {'+'.join(names)}")
    return blended


def _play_audio_file(file_path: Path) -> bool:
    """Play audio file with platform-appropriate player."""
    import platform
    import subprocess

    system = platform.system().lower()
    players = (
        [["afplay"]] if system == "darwin"
        else [["aplay", "-q"], ["paplay"], ["mpg123", "-q"]] if system == "linux"
        else []
    )

    for cmd in players:
        try:
            subprocess.run(cmd + [str(file_path)], check=True, capture_output=True, timeout=120)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue

    _log("WARN", "audio_play", "No audio player found")
    return False


def _say_impl(
    text: str,
    voice: str | None = None,
    speed: float = 1.0,
    play: bool = True,
    output_path: str | None = None,
) -> tuple[dict, dict]:
    """Generate speech from text using Kokoro-82M.

    CLI: say
    MCP: say

    Auto-downloads model on first use. Supports voice blending via
    'name1+name2+name3' syntax (averages tensors before generation).
    Default voice: af_heart+af_nicole+af_kore.
    """
    import tempfile
    import time

    import numpy as np
    import soundfile as sf

    start_ms = time.time() * 1000

    assert text and text.strip(), "text must not be empty"

    voice_str = voice if voice else "+".join(DEFAULT_VOICE_BLEND)
    voice_input = _load_voice(voice_str)

    pipeline = _get_pipeline()
    _log("INFO", "tts_start", "Generating speech",
         detail=f"voice={voice_str} speed={speed} text_len={len(text)}")

    audio_chunks = []
    generator = pipeline(text, voice=voice_input, speed=speed, split_pattern=r"\n+")
    for i, (gs, ps, audio) in enumerate(generator):
        audio_chunks.append(audio)
        _log("DEBUG", "tts_chunk", f"Chunk {i}: {gs[:50]}")

    assert audio_chunks, "Kokoro pipeline produced no audio chunks"

    full_audio = np.concatenate(audio_chunks)

    if output_path:
        out = Path(output_path)
    else:
        out_dir = Path(tempfile.gettempdir()) / "sfa_tts_kokoro"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"speech_{int(time.time())}.wav"

    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), full_audio, SAMPLE_RATE_HZ)

    played = False
    if play:
        played = _play_audio_file(out)

    latency = round(time.time() * 1000 - start_ms, 2)
    duration_seconds = round(len(full_audio) / SAMPLE_RATE_HZ, 2)
    _log("INFO", "tts_complete", f"Generated {duration_seconds}s audio",
         detail=f"voice={voice_str} output={out}",
         metrics=f"latency_ms={latency} samples={len(full_audio)} chunks={len(audio_chunks)}")

    return (
        {
            "success": True,
            "output_path": str(out),
            "voice": voice_str,
            "played": played,
            "duration_seconds": duration_seconds,
        },
        {"status": "success", "latency_ms": latency, "voice": voice_str, "speed": speed},
    )


def _save_impl(
    text: str,
    output_path: str,
    voice: str | None = None,
    speed: float = 1.0,
) -> tuple[dict, dict]:
    """Generate speech and save to a specific file path.

    CLI: save
    MCP: save
    """
    return _say_impl(text, voice=voice, speed=speed, play=False, output_path=output_path)


def _voices_impl() -> tuple[list[dict], dict]:
    """List available Kokoro voices grouped by category.

    CLI: voices
    MCP: voices
    """
    import time

    start_ms = time.time() * 1000

    labels = {
        "af": "American Female",
        "am": "American Male",
        "bf": "British Female",
        "bm": "British Male",
    }

    result = []
    total = 0
    for category, names in VOICES.items():
        result.append({"category": category, "label": labels[category], "voices": names})
        total += len(names)

    latency = round(time.time() * 1000 - start_ms, 2)
    return (
        result,
        {
            "status": "success",
            "latency_ms": latency,
            "count": total,
            "default_blend": "+".join(DEFAULT_VOICE_BLEND),
        },
    )


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Native Kokoro-82M text-to-speech with MPS acceleration"
    )
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 1.0.0")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # mcp-stdio
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")

    # CLI for _say_impl
    p_say = subparsers.add_parser("say", help="Generate and play speech")
    p_say.add_argument("text", nargs="?", help="Text to speak")
    p_say.add_argument("-v", "--voice", default=None,
                       help="Voice name or blend (e.g. af_heart+af_nicole+af_kore)")
    p_say.add_argument("-s", "--speed", type=float, default=1.0, help="Speech speed (default: 1.0)")
    p_say.add_argument("-p", "--play", action="store_true", default=True, help="Play audio (default)")
    p_say.add_argument("-n", "--no-play", dest="play", action="store_false", help="Don't play audio")

    # CLI for _save_impl
    p_save = subparsers.add_parser("save", help="Generate and save speech to file")
    p_save.add_argument("text", nargs="?", help="Text to speak")
    p_save.add_argument("output_path", nargs="?", help="Output file path (.wav)")
    p_save.add_argument("-v", "--voice", default=None, help="Voice name or blend")
    p_save.add_argument("-s", "--speed", type=float, default=1.0, help="Speech speed (default: 1.0)")

    # CLI for _voices_impl
    subparsers.add_parser("voices", help="List available voices")

    args = parser.parse_args()

    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "say":
            text = args.text
            if not text and not sys.stdin.isatty():
                text = sys.stdin.read().strip()
            assert text, "text required (positional argument or stdin)"
            result, metrics = _say_impl(text, voice=args.voice, speed=args.speed, play=args.play)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
        elif args.command == "save":
            text = args.text
            if not text and not sys.stdin.isatty():
                text = sys.stdin.read().strip()
            assert text, "text required (positional argument or stdin)"
            assert args.output_path, "output_path required (second positional argument)"
            result, metrics = _save_impl(
                text, output_path=args.output_path, voice=args.voice, speed=args.speed
            )
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
        elif args.command == "voices":
            voices_list, metrics = _voices_impl()
            print(json.dumps({"voices": voices_list, "metrics": metrics}, indent=2))
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

    mcp = FastMCP("tts-kokoro")

    # MCP for _say_impl
    @mcp.tool()
    def say(text: str, voice: str = "", speed: float = 1.0, play: bool = True) -> str:
        """Generate and play speech using native Kokoro-82M.

        Auto-downloads model on first use. Default voice is a blend of
        af_heart, af_nicole, and af_kore. Supports voice blending via
        '+' syntax (e.g. 'af_heart+af_nicole').

        Args:
            text: Text to speak
            voice: Voice name or blend (empty for default blend)
            speed: Speech speed multiplier (default 1.0)
            play: Whether to play audio (default True)
        """
        result, metrics = _say_impl(text, voice=voice or None, speed=speed, play=play)
        return json.dumps({"result": result, "metrics": metrics})

    # MCP for _save_impl
    @mcp.tool()
    def save(text: str, output_path: str, voice: str = "", speed: float = 1.0) -> str:
        """Generate speech and save to a file.

        Args:
            text: Text to speak
            output_path: Path to save the audio file (.wav)
            voice: Voice name or blend (empty for default blend)
            speed: Speech speed multiplier (default 1.0)
        """
        result, metrics = _save_impl(
            text, output_path=output_path, voice=voice or None, speed=speed
        )
        return json.dumps({"result": result, "metrics": metrics})

    # MCP for _voices_impl
    @mcp.tool()
    def voices() -> str:
        """List available Kokoro voices grouped by category.

        Args:
            None
        """
        voices_list, metrics = _voices_impl()
        return json.dumps({"voices": voices_list, "metrics": metrics})

    print("tts-kokoro MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
