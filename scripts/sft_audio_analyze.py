#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "librosa>=0.11.0",
#     "numpy>=1.22.3",
#     "soundfile>=0.12.1",
#     "fastmcp",
#     "yt-dlp>=2024.12.13",
# ]
# ///
"""Audio analysis for music - key, tempo, spectral character, dynamics, sections, lyrics.

Decomposes audio into structured musical features using librosa.
Also extracts lyrics from YouTube videos using yt-dlp.
DuckDB-style architecture: analysis engine produces structured data,
CLI/MCP renders it.

Usage:
    sft_audio_analyze.py analyze track.mp3
    sft_audio_analyze.py analyze track.mp3 --format json
    sft_audio_analyze.py compare track1.mp3 track2.mp3
    sft_audio_analyze.py lyrics "https://youtube.com/watch?v=..."
    sft_audio_analyze.py lyrics "https://youtube.com/watch?v=..." --output lyrics.txt
    sft_audio_analyze.py mcp-stdio
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
_THRESHOLD = _LEVELS.get(os.environ.get("SFA_LOG_LEVEL", "INFO"), 20)
_LOG_DIR = os.environ.get("SFA_LOG_DIR", "")
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
EXPOSED = ["analyze", "compare", "lyrics"]

CONFIG = {
    "version": "1.0.0",
}

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def _load_audio(filepath: str) -> tuple:
    """Load audio file, return (samples, sample_rate)."""
    import librosa
    y, sr = librosa.load(filepath, sr=22050, mono=True)
    return y, sr


def _detect_key(y, sr) -> dict:
    """Detect musical key using Krumhansl-Schmuckler algorithm."""
    import librosa
    import numpy as np

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)
    mean_chroma = mean_chroma / (np.linalg.norm(mean_chroma) + 1e-6)

    best_key = ""
    best_mode = ""
    best_corr = -1.0

    for shift in range(12):
        major_shifted = np.roll(MAJOR_PROFILE, shift)
        minor_shifted = np.roll(MINOR_PROFILE, shift)

        major_norm = major_shifted / (np.linalg.norm(major_shifted) + 1e-6)
        minor_norm = minor_shifted / (np.linalg.norm(minor_shifted) + 1e-6)

        corr_major = np.corrcoef(mean_chroma, major_norm)[0, 1]
        corr_minor = np.corrcoef(mean_chroma, minor_norm)[0, 1]

        if corr_major > best_corr:
            best_corr = corr_major
            best_key = PITCH_NAMES[shift]
            best_mode = "major"
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = PITCH_NAMES[shift]
            best_mode = "minor"

    return {
        "key": best_key,
        "mode": best_mode,
        "confidence": round(float(best_corr), 3),
        "label": f"{best_key} {best_mode}",
    }


def _detect_tempo(y, sr) -> dict:
    """Detect tempo and beat positions."""
    import librosa
    import numpy as np

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    tempo_val = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if len(beat_frames) > 4:
        beat_strengths = onset_env[beat_frames[beat_frames < len(onset_env)]]
        if len(beat_strengths) >= 4:
            acf = np.correlate(beat_strengths - np.mean(beat_strengths),
                               beat_strengths - np.mean(beat_strengths), mode='full')
            acf = acf[len(acf)//2:]
            time_sig = "3/4" if len(acf) > 4 and acf[3] > acf[4] else "4/4"
        else:
            time_sig = "4/4"
    else:
        time_sig = "4/4"

    return {
        "bpm": round(tempo_val, 1),
        "time_signature": time_sig,
        "beat_count": len(beat_times),
        "beat_times_sample": [round(float(t), 2) for t in beat_times[:8]],
    }


def _spectral_character(y, sr) -> dict:
    """Analyze spectral character — brightness, warmth, texture."""
    import librosa
    import numpy as np

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    rolloff_mean = float(np.mean(rolloff))

    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    low_mask = freqs < 300
    low_energy = float(np.sum(S[low_mask, :]**2))
    total_energy = float(np.sum(S**2)) + 1e-10
    warmth = low_energy / total_energy

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))

    if centroid_mean > 3000:
        brightness_desc = "bright"
    elif centroid_mean > 1500:
        brightness_desc = "balanced"
    else:
        brightness_desc = "dark"

    if warmth > 0.4:
        warmth_desc = "very warm, bass-heavy"
    elif warmth > 0.2:
        warmth_desc = "warm"
    elif warmth > 0.1:
        warmth_desc = "balanced"
    else:
        warmth_desc = "thin, treble-forward"

    return {
        "brightness": brightness_desc,
        "brightness_hz": round(centroid_mean),
        "warmth": warmth_desc,
        "warmth_ratio": round(warmth, 3),
        "rolloff_hz": round(rolloff_mean),
        "texture_zcr": round(zcr_mean, 4),
    }


def _loudness_contour(y, sr, n_windows: int = 20) -> dict:
    """Analyze loudness shape over time."""
    import librosa
    import numpy as np

    rms = librosa.feature.rms(y=y)[0]
    window_size = max(1, len(rms) // n_windows)
    windows = []
    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, len(rms))
        if start >= len(rms):
            break
        windows.append(float(np.mean(rms[start:end])))

    if not windows:
        return {"shape": "unknown", "windows": [], "peak_position": 0}

    max_val = max(windows) + 1e-10
    normalized = [round(w / max_val, 2) for w in windows]

    peak_idx = normalized.index(max(normalized))
    peak_pct = round(peak_idx / len(normalized), 2)

    rms_db = 20 * np.log10(np.array(windows) + 1e-10)
    dynamic_range = float(np.max(rms_db) - np.min(rms_db))

    if peak_pct < 0.3:
        shape = "front-loaded"
    elif peak_pct > 0.7:
        shape = "builds to climax"
    elif dynamic_range < 6:
        shape = "steady, compressed"
    else:
        shape = "dynamic, varied"

    return {
        "shape": shape,
        "peak_position": peak_pct,
        "dynamic_range_db": round(dynamic_range, 1),
        "contour": normalized,
    }


def _harmonic_percussive(y, sr) -> dict:
    """Analyze harmonic vs percussive balance."""
    import librosa
    import numpy as np

    y_harm, y_perc = librosa.effects.hpss(y)

    harm_energy = float(np.sum(y_harm**2))
    perc_energy = float(np.sum(y_perc**2))
    total = harm_energy + perc_energy + 1e-10

    harm_ratio = harm_energy / total
    perc_ratio = perc_energy / total

    if harm_ratio > 0.7:
        balance = "predominantly harmonic (melodic, sustained tones)"
    elif harm_ratio > 0.55:
        balance = "harmonic-leaning (melody over rhythm)"
    elif perc_ratio > 0.55:
        balance = "percussive-leaning (rhythm-driven)"
    else:
        balance = "balanced harmonic/percussive"

    return {
        "balance": balance,
        "harmonic_ratio": round(harm_ratio, 3),
        "percussive_ratio": round(perc_ratio, 3),
    }


def _sections(y, sr) -> dict:
    """Detect structural sections via feature segmentation."""
    import librosa
    import numpy as np

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    try:
        boundaries = librosa.segment.agglomerative(mfcc, k=None)
        boundary_times = librosa.frames_to_time(boundaries, sr=sr)
    except Exception:
        duration = librosa.get_duration(y=y, sr=sr)
        boundary_times = np.arange(0, duration, 30.0)

    sections = []
    duration = librosa.get_duration(y=y, sr=sr)
    times = list(boundary_times) + [duration]

    for i in range(len(times) - 1):
        start = float(times[i])
        end = float(times[i + 1])
        length = end - start
        if length < 2.0:
            continue
        sections.append({
            "start": round(start, 1),
            "end": round(end, 1),
            "duration": round(length, 1),
        })

    return {
        "section_count": len(sections),
        "sections": sections[:20],
        "duration": round(float(duration), 1),
    }


def _onset_density(y, sr) -> dict:
    """Measure note attack density — busy vs sparse."""
    import librosa
    import numpy as np

    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    density = len(onset_times) / (duration + 1e-10)

    if density > 8:
        desc = "very busy, dense note activity"
    elif density > 4:
        desc = "moderately busy"
    elif density > 2:
        desc = "measured, deliberate"
    else:
        desc = "sparse, spacious"

    return {
        "onsets_per_second": round(density, 2),
        "total_onsets": len(onset_times),
        "description": desc,
    }


def _analyze_impl(filepath: str) -> tuple[dict, dict]:
    """Full analysis of an audio file. CLI: analyze, MCP: analyze."""
    start_ms = time.time() * 1000

    path = Path(filepath)
    assert path.exists(), f"file not found: {filepath}"

    try:
        y, sr = _load_audio(filepath)
    except Exception as e:
        return {}, {"status": "error", "error": f"failed to load audio: {e}", "latency_ms": 0}

    analysis = {
        "file": path.name,
        "key": _detect_key(y, sr),
        "tempo": _detect_tempo(y, sr),
        "spectral": _spectral_character(y, sr),
        "loudness": _loudness_contour(y, sr),
        "harmonic_percussive": _harmonic_percussive(y, sr),
        "sections": _sections(y, sr),
        "onset_density": _onset_density(y, sr),
    }

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2), "file": filepath}
    
    return analysis, metrics


def _compare_impl(file1: str, file2: str) -> tuple[dict, dict]:
    """Compare two audio files. CLI: compare, MCP: compare."""
    start_ms = time.time() * 1000
    
    a1, m1 = _analyze_impl(file1)
    a2, m2 = _analyze_impl(file2)

    if m1["status"] != "success":
        return {}, m1
    if m2["status"] != "success":
        return {}, m2

    comparison = {
        "track_1": a1,
        "track_2": a2,
        "differences": {
            "key": f"{a1['key']['label']} vs {a2['key']['label']}",
            "tempo": f"{a1['tempo']['bpm']} vs {a2['tempo']['bpm']} BPM",
            "brightness": f"{a1['spectral']['brightness']} vs {a2['spectral']['brightness']}",
            "warmth": f"{a1['spectral']['warmth']} vs {a2['spectral']['warmth']}",
            "dynamics": f"{a1['loudness']['shape']} vs {a2['loudness']['shape']}",
            "balance": f"{a1['harmonic_percussive']['balance']} vs {a2['harmonic_percussive']['balance']}",
            "density": f"{a1['onset_density']['description']} vs {a2['onset_density']['description']}",
        },
    }

    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}

    return comparison, metrics


def _lyrics_impl(url: str, output_dir: str | None = None) -> tuple[dict, dict]:
    """Extract lyrics from YouTube video. CLI: lyrics, MCP: lyrics."""
    import yt_dlp
    
    start_ms = time.time() * 1000
    
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "skip_download": True,
    }
    
    if output_dir:
        ydl_opts["outtmpl"] = str(Path(output_dir) / "%(title)s.%(ext)s")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            lyrics_text = ""
            subtitles = info.get("subtitles", {})
            auto_subs = info.get("automatic_captions", {})
            
            # Try manual subtitles first
            if "en" in subtitles:
                sub_info = subtitles["en"][0] if isinstance(subtitles["en"], list) else None
                if sub_info and "url" in sub_info:
                    import urllib.request
                    with urllib.request.urlopen(sub_info["url"]) as response:
                        lyrics_text = response.read().decode("utf-8")
            
            # Fall back to auto-generated
            if not lyrics_text and "en" in auto_subs:
                sub_info = auto_subs["en"][0] if isinstance(auto_subs["en"], list) else None
                if sub_info and "url" in sub_info:
                    import urllib.request
                    with urllib.request.urlopen(sub_info["url"]) as response:
                        lyrics_text = response.read().decode("utf-8")
            
            # Parse VTT/SRT format to plain text
            if lyrics_text:
                lines = []
                for line in lyrics_text.split("\n"):
                    # Skip timestamp lines and WEBVTT header
                    if line.strip() and not line.startswith("WEBVTT") and not "-->" in line and not line[0:1].isdigit():
                        lines.append(line.strip())
                lyrics_text = "\n".join(lines)
            
            latency_ms = time.time() * 1000 - start_ms
            
            result = {
                "title": info.get("title", "Unknown"),
                "artist": info.get("artist", info.get("uploader", "Unknown")),
                "duration": info.get("duration", 0),
                "lyrics": lyrics_text if lyrics_text else "No lyrics/subtitles found",
                "url": url,
            }
            
            metrics = {
                "status": "success",
                "latency_ms": round(latency_ms, 2),
                "has_lyrics": bool(lyrics_text),
                "duration_seconds": info.get("duration", 0),
            }
            
            return result, metrics
            
    except Exception as e:
        latency_ms = time.time() * 1000 - start_ms
        metrics = {
            "status": "error",
            "latency_ms": round(latency_ms, 2),
            "error": str(e),
        }
        return {"error": str(e), "url": url}, metrics


def _print_analysis(a: dict):
    """Human-readable analysis output."""
    print(f"  {a['file']}")
    print(f"  {'─' * 60}")
    print()

    k = a["key"]
    print(f"  Key:        {k['label']} (confidence: {k['confidence']})")

    t = a["tempo"]
    print(f"  Tempo:      {t['bpm']} BPM, {t['time_signature']}")

    s = a["spectral"]
    print(f"  Tone:       {s['brightness']}, {s['warmth']}")
    print(f"              centroid {s['brightness_hz']}Hz, rolloff {s['rolloff_hz']}Hz")

    hp = a["harmonic_percussive"]
    print(f"  Balance:    {hp['balance']}")
    print(f"              harmonic {hp['harmonic_ratio']:.0%} / percussive {hp['percussive_ratio']:.0%}")

    l = a["loudness"]
    print(f"  Dynamics:   {l['shape']} (range: {l['dynamic_range_db']}dB)")

    o = a["onset_density"]
    print(f"  Density:    {o['description']} ({o['onsets_per_second']}/sec)")

    sec = a["sections"]
    print(f"  Structure:  {sec['section_count']} sections over {sec['duration']}s")
    for i, s in enumerate(sec["sections"][:8]):
        print(f"              [{i+1}] {s['start']}s - {s['end']}s ({s['duration']}s)")
    if sec["section_count"] > 8:
        print(f"              ... and {sec['section_count'] - 8} more")

    contour = l.get("contour", [])
    if contour:
        print()
        print("  Loudness contour:")
        max_bar = 40
        for i, val in enumerate(contour):
            bar = "█" * int(val * max_bar)
            pct = i / len(contour)
            print(f"    {pct:>4.0%} {bar}")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Audio analysis for music",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sft_audio_analyze.py analyze track.mp3
  sft_audio_analyze.py analyze track.mp3 --format json
  sft_audio_analyze.py compare track1.mp3 track2.mp3
        """,
    )
    
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {CONFIG['version']}",
    )
    
    sub = parser.add_subparsers(dest="command", help="Commands")

    p_analyze = sub.add_parser("analyze", help="Analyze an audio file")
    p_analyze.add_argument("file", nargs="?", help="Path to audio file")
    p_analyze.add_argument(
        "-f", "--format",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )

    p_compare = sub.add_parser("compare", help="Compare two audio files")
    p_compare.add_argument("file1", help="First audio file")
    p_compare.add_argument("file2", help="Second audio file")
    p_compare.add_argument(
        "-f", "--format",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )

    p_lyrics = sub.add_parser("lyrics", help="Extract lyrics from YouTube video")
    p_lyrics.add_argument("url", nargs="?", help="YouTube URL")
    p_lyrics.add_argument("-o", "--output", help="Output file for lyrics")
    p_lyrics.add_argument("-d", "--dir", help="Output directory")

    sub.add_parser("mcp-stdio", help="Run as MCP server")

    args = parser.parse_args()

    try:
        if args.command == "analyze":
            audio_file = args.file
            if not audio_file and not sys.stdin.isatty():
                audio_file = sys.stdin.read().strip()
            assert audio_file, "audio file required (positional argument or stdin)"
            analysis, metrics = _analyze_impl(audio_file)
            if metrics["status"] != "success":
                print(f"Error: {metrics.get('error', 'unknown')}", file=sys.stderr)
                sys.exit(1)
            
            _log("INFO", "analyze", f"file={args.file}", metrics=json.dumps(metrics))
            
            if args.format == "json":
                print(json.dumps(analysis, indent=2))
            else:
                print()
                _print_analysis(analysis)
                print(f"\n  Analyzed in {metrics['latency_ms']:.0f}ms")
                print()

        elif args.command == "compare":
            result, metrics = _compare_impl(args.file1, args.file2)
            if metrics["status"] != "success":
                print(f"Error: {metrics.get('error', 'unknown')}", file=sys.stderr)
                sys.exit(1)
            
            _log("INFO", "compare", f"files={args.file1},{args.file2}", metrics=json.dumps(metrics))
            
            if args.format == "json":
                print(json.dumps(result, indent=2))
            else:
                print()
                print("  TRACK 1")
                _print_analysis(result["track_1"])
                print()
                print("  TRACK 2")
                _print_analysis(result["track_2"])
                print()
                print("  DIFFERENCES")
                print(f"  {'─' * 60}")
                for k, v in result["differences"].items():
                    print(f"  {k:<12} {v}")
                print(f"\n  Analyzed in {metrics['latency_ms']:.0f}ms")
                print()

        elif args.command == "lyrics":
            url = args.url
            if not url and not sys.stdin.isatty():
                url = sys.stdin.read().strip()
            assert url, "YouTube URL required (positional argument or stdin)"
            
            result, metrics = _lyrics_impl(url, args.dir)
            if metrics["status"] != "success":
                print(f"Error: {metrics.get('error', 'unknown')}", file=sys.stderr)
                sys.exit(1)
            
            _log("INFO", "lyrics", f"title={result.get('title', 'unknown')}", metrics=json.dumps(metrics))
            
            if args.output:
                Path(args.output).write_text(result["lyrics"], encoding="utf-8")
                print(f"Lyrics saved to {args.output}")
            else:
                print(f"\nTitle: {result['title']}")
                print(f"Artist: {result['artist']}")
                print(f"Duration: {result['duration']}s")
                print(f"\n{'─' * 60}")
                print(result['lyrics'])
                print(f"{'─' * 60}")

        elif args.command == "mcp-stdio":
            _run_mcp()
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
    mcp = FastMCP("audio_analyze")

    @mcp.tool()
    def analyze(filepath: str) -> str:
        """Analyze an audio file - returns key, tempo, spectral character, dynamics, sections.
        
        Args:
            filepath: Path to the audio file to analyze
        
        Returns:
            JSON with full analysis and metrics
        """
        analysis, metrics = _analyze_impl(filepath)
        return json.dumps({"analysis": analysis, "metrics": metrics}, indent=2)

    @mcp.tool()
    def compare(file1: str, file2: str) -> str:
        """Compare two audio files side by side.
        
        Args:
            file1: Path to first audio file
            file2: Path to second audio file
        
        Returns:
            JSON with comparison results and metrics
        """
        result, metrics = _compare_impl(file1, file2)
        return json.dumps({"comparison": result, "metrics": metrics}, indent=2)

    @mcp.tool()
    def lyrics(url: str) -> str:
        """Extract lyrics from a YouTube video.
        
        Args:
            url: YouTube video URL
        
        Returns:
            JSON with lyrics, title, artist, and metadata
        """
        result, metrics = _lyrics_impl(url)
        return json.dumps({"result": result, "metrics": metrics}, indent=2)

    print("Audio analysis MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
