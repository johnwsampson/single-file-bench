#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mlx-whisper>=0.4",
# ]
# ///
"""Speech-to-text transcription for audio files.

Transcribe audio files using Apple Silicon MLX-optimized Whisper models.
Supports various model sizes from tiny (fast) to large-v3 (accurate).

Usage:
    sft_speech2text.py transcribe audio.wav [--model small] [--timestamps]
    sft_speech2text.py transcribe audio.wav --model large-v3 --output transcript.txt
    sft_speech2text.py mcp-stdio

Models (MLX-optimized, downloaded on first use):
    tiny     - 39M, fastest, rough accuracy
    base     - 74M, good balance
    small    - 244M, solid accuracy (default)
    medium   - 769M, high accuracy
    large-v3 - 1.5B, best accuracy
"""

import argparse
import json
import os
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
EXPOSED = ["transcribe"]

CONFIG = {
    "version": "1.0.0",
    "model_map": {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
    },
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def _transcribe_impl(
    audio_path: str,
    model: str = "small",
    timestamps: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Transcribe audio file using mlx-whisper.
    
    CLI: transcribe
    MCP: transcribe
    
    Returns (result_dict, metrics_dict).
    """
    start_ms = time.time() * 1000
    
    import mlx_whisper
    
    model_id = CONFIG["model_map"].get(model, CONFIG["model_map"]["small"])
    
    try:
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_id,
            word_timestamps=timestamps,
        )
        
        latency_ms = time.time() * 1000 - start_ms
        
        text = result.get("text", "").strip()
        segments = result.get("segments", [])
        
        metrics = {
            "model": model,
            "model_id": model_id,
            "duration_seconds": round(latency_ms / 1000, 2),
            "latency_ms": round(latency_ms, 2),
            "segments_count": len(segments),
            "status": "success",
        }
        
        return {
            "text": text,
            "segments": segments if timestamps else [],
            "language": result.get("language", "unknown"),
        }, metrics
        
    except Exception as e:
        latency_ms = time.time() * 1000 - start_ms
        metrics = {
            "model": model,
            "latency_ms": round(latency_ms, 2),
            "status": "error",
            "error": str(e),
        }
        return {"error": str(e)}, metrics


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Speech-to-text transcription via mlx-whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sft_whisper.py transcribe audio.wav
  sft_whisper.py transcribe audio.wav --model large-v3 --timestamps
  sft_whisper.py transcribe audio.wav -o transcript.txt
        """,
    )
    
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {CONFIG['version']}",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # transcribe
    p_transcribe = subparsers.add_parser("transcribe", help="Transcribe audio file")
    p_transcribe.add_argument("audio", help="Path to audio file (wav, mp3, etc.)")
    p_transcribe.add_argument(
        "-m", "--model",
        default="small",
        choices=list(CONFIG["model_map"].keys()),
        help="Whisper model size (default: small)",
    )
    p_transcribe.add_argument(
        "-t", "--timestamps",
        action="store_true",
        help="Include word-level timestamps",
    )
    p_transcribe.add_argument(
        "-o", "--output",
        help="Save transcript to file",
    )
    p_transcribe.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output as JSON",
    )
    
    # MCP server
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")
    
    args = parser.parse_args()
    
    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "transcribe":
            # Get audio path from stdin if not provided
            audio_path = args.audio
            if not audio_path and not sys.stdin.isatty():
                audio_path = sys.stdin.read().strip()
            
            assert audio_path, "audio path required (positional argument or stdin)"
            assert Path(audio_path).exists(), f"audio file not found: {audio_path}"
            
            result, metrics = _transcribe_impl(
                audio_path=audio_path,
                model=args.model,
                timestamps=args.timestamps,
            )
            
            if metrics["status"] == "success":
                _log(
                    "INFO",
                    "transcribe",
                    f"model={args.model} segments={metrics['segments_count']}",
                    metrics=json.dumps(metrics),
                )
                
                if args.json:
                    output_text = json.dumps(result, indent=2)
                else:
                    output_text = result["text"]
                
                print(output_text)
                print(f"\n[{args.model}] {metrics['duration_seconds']:.1f}s", file=sys.stderr)
                
                if args.output:
                    Path(args.output).write_text(result["text"])
                    print(f"Saved to {args.output}", file=sys.stderr)
            else:
                _log("ERROR", "transcribe", result["error"], metrics=json.dumps(metrics))
                print(f"Error: {result['error']}", file=sys.stderr)
                sys.exit(1)
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
    
    mcp = FastMCP("whisper")
    
    @mcp.tool()
    def transcribe(audio_path: str, model: str = "small", timestamps: bool = False) -> str:
        """Transcribe audio file to text using mlx-whisper.
        
        Transcribes audio files using Apple Silicon optimized Whisper models.
        Models are downloaded on first use and cached locally.
        
        Args:
            audio_path: Path to audio file (wav, mp3, m4a, etc.)
            model: Model size - tiny, base, small (default), medium, large-v3
            timestamps: Include word-level timestamps in output
        
        Returns:
            JSON string with transcript, segments (if timestamps), and metadata
        """
        result, metrics = _transcribe_impl(
            audio_path=audio_path,
            model=model,
            timestamps=timestamps,
        )
        
        response = {
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "language": result.get("language", "unknown"),
            "metrics": metrics,
        }
        
        if "error" in result:
            response["error"] = result["error"]
        
        return json.dumps(response, indent=2)
    
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
