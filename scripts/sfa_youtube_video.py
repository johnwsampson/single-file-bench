#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp", "httpx", "youtube-transcript-api>=0.6.0", "yt-dlp"]
# ///
"""YouTube video operations - info, transcripts, channel search, frame extraction.

Usage:
    sfa_youtube_video.py info <url_or_id>
    sfa_youtube_video.py transcript <url_or_id> [--language LANG] [--timestamps]
    sfa_youtube_video.py channel_search <handle> <query> [--max-results N]
    sfa_youtube_video.py channel_info <handle>
    sfa_youtube_video.py channel_videos <handle> [--sort SORT] [--order ORDER] [--max-results N]
    sfa_youtube_video.py view_frames <url_or_id> [--at M:SS,M:SS] [--threshold N] [--min-gap N]
    sfa_youtube_video.py mcp-stdio
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
EXPOSED = ["info", "transcript", "channel_search", "channel_info", "channel_videos", "view_frames"]

CONFIG = {
    "default_language": "en",
    "request_timeout_seconds": 30.0,
    "max_results_default": 10,
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def _extract_video_id(url_or_id: str) -> str:
    """Extract video ID from various YouTube URL formats."""
    import re
    import urllib.parse
    
    url_or_id = url_or_id.strip()
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id):
        return url_or_id
    
    try:
        parsed = urllib.parse.urlparse(url_or_id)
        if parsed.hostname in ("youtu.be", "www.youtu.be"):
            return parsed.path.lstrip("/")
        if parsed.hostname in ("youtube.com", "www.youtube.com", "m.youtube.com"):
            if parsed.path == "/watch":
                v_param = urllib.parse.parse_qs(parsed.query).get("v")
                if v_param:
                    return v_param[0]
            elif parsed.path.startswith(("/v/", "/shorts/", "/embed/")):
                return parsed.path.split("/")[2]
    except (ValueError, IndexError, KeyError):
        pass
    
    raise ValueError(f"Could not extract video ID from: {url_or_id}")


def _extract_channel_handle(url_or_handle: str) -> str:
    """Extract channel handle from URL or return as-is if already a handle."""
    import re
    import urllib.parse
    
    url_or_handle = url_or_handle.strip()
    
    if url_or_handle.startswith("@"):
        return url_or_handle
    if re.match(r"^[a-zA-Z0-9_-]+$", url_or_handle):
        return f"@{url_or_handle}"
    
    try:
        parsed = urllib.parse.urlparse(url_or_handle)
        if parsed.hostname in ("youtube.com", "www.youtube.com"):
            path = parsed.path
            if path.startswith("/@"):
                return path.split("/")[1]
            elif path.startswith("/c/"):
                return f"@{path.split('/')[2]}"
            elif path.startswith("/channel/"):
                return path.split("/")[2]
    except (ValueError, IndexError):
        pass
    
    raise ValueError(f"Could not extract channel handle from: {url_or_handle}")


def _info_impl(url: str) -> tuple[dict, dict]:
    """Get video metadata via yt-dlp.
    
    CLI: info
    MCP: info
    
    Returns:
        Tuple of (result_dict, metrics_dict)
    """
    import json
    import time
    
    start_ms = time.time() * 1000
    
    try:
        video_id = _extract_video_id(url)
    except ValueError as e:
        latency = round(time.time() * 1000 - start_ms, 2)
        return {"error": str(e)}, {"status": "error", "latency_ms": latency}
    
    import yt_dlp
    
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
        
        result = {
            "id": info["id"],
            "title": info["title"],
            "description": info.get("description", ""),
            "channel": info.get("channel"),
            "channel_id": info.get("channel_id"),
            "channel_url": info.get("channel_url"),
            "channel_follower_count": info.get("channel_follower_count"),
            "upload_date": info.get("upload_date"),
            "duration": info.get("duration"),
            "duration_string": info.get("duration_string"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "comment_count": info.get("comment_count"),
            "categories": info.get("categories", []),
            "tags": info.get("tags", []),
            "chapters": info.get("chapters", []),
            "webpage_url": info.get("webpage_url"),
            "thumbnail": info.get("thumbnail"),
            "is_live": info.get("is_live"),
            "was_live": info.get("was_live"),
        }
        
        latency = round(time.time() * 1000 - start_ms, 2)
        metrics = {"status": "success", "latency_ms": latency}
        return result, metrics
        
    except Exception as e:
        latency = round(time.time() * 1000 - start_ms, 2)
        return {"error": str(e)}, {"status": "error", "latency_ms": latency}


def _transcript_impl(url: str, language: str = "en", timestamps: bool = False) -> tuple[str, dict]:
    """Get video transcript.
    
    CLI: transcript
    MCP: transcript
    """
    import time
    
    start_ms = time.time() * 1000
    
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        YOUTUBE_API_AVAILABLE = True
    except ImportError:
        YOUTUBE_API_AVAILABLE = False
    
    try:
        video_id = _extract_video_id(url)
    except ValueError as e:
        latency = round(time.time() * 1000 - start_ms, 2)
        return f"ERROR: {e}", {"status": "error", "latency_ms": latency}
    
    # Try youtube-transcript-api first
    if YOUTUBE_API_AVAILABLE:
        try:
            api = YouTubeTranscriptApi()
            result = api.fetch(video_id, languages=(language, "en"))
            
            lines = []
            for snippet in result.snippets:
                text = snippet.text.replace("\n", " ")
                if timestamps:
                    mins = int(snippet.start // 60)
                    secs = int(snippet.start % 60)
                    line = f"[{mins:02d}:{secs:02d}] {text}"
                else:
                    line = text
                lines.append(line)
            
            latency = round(time.time() * 1000 - start_ms, 2)
            metrics = {"status": "success", "latency_ms": latency}
            return "\n".join(lines), metrics
        except Exception:
            pass  # Fall through to yt-dlp
    
    # Fallback to yt-dlp
    import tempfile
    import re
    import yt_dlp
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts = {
            "writeautomaticsub": True,
            "writesubtitles": True,
            "subtitleslangs": ["en", ".*"],
            "skip_download": True,
            "outtmpl": f"{temp_dir}/%(id)s",
            "quiet": True,
            "no_warnings": True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            
            files = list(Path(temp_dir).glob("*.vtt"))
            if not files:
                latency = round(time.time() * 1000 - start_ms, 2)
                return "ERROR: No transcript available", {"status": "error", "latency_ms": latency}
            
            content = files[0].read_text(encoding="utf-8", errors="replace")
            
            lines = content.splitlines()
            text_lines = []
            seen = set()
            timestamp_pattern = re.compile(r"\d{2}:\d{2}:\d{2}\.\d{3}\s-->\s")
            
            for line in lines:
                line = line.strip()
                if not line or line == "WEBVTT":
                    continue
                if line.startswith(("Kind:", "Language:")) or timestamp_pattern.match(line):
                    continue
                clean = re.sub(r"<[^>]+>", "", line).strip()
                if clean and clean not in seen:
                    text_lines.append(clean)
                    seen.add(clean)
            
            latency = round(time.time() * 1000 - start_ms, 2)
            metrics = {"status": "success", "latency_ms": latency}
            return " ".join(text_lines), metrics
        except Exception as e:
            latency = round(time.time() * 1000 - start_ms, 2)
            return f"ERROR: {e}", {"status": "error", "latency_ms": latency}


def _channel_search_impl(handle: str, query: str, max_results: int = 10) -> tuple[dict, dict]:
    """Search within a YouTube channel.
    
    CLI: channel_search
    MCP: channel_search
    """
    import json
    import time
    import urllib.parse
    import yt_dlp
    
    start_ms = time.time() * 1000
    
    try:
        channel_handle = _extract_channel_handle(handle)
    except ValueError as e:
        latency = round(time.time() * 1000 - start_ms, 2)
        return {"error": str(e)}, {"status": "error", "latency_ms": latency}
    
    search_url = f"https://www.youtube.com/{channel_handle}/search?query={urllib.parse.quote(query)}"
    
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "playlistend": max_results,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_url, download=False)
        
        results = []
        entries = info.get("entries", []) if info else []
        
        for entry in entries[:max_results]:
            if entry:
                results.append({
                    "id": entry.get("id"),
                    "title": entry.get("title"),
                    "url": entry.get("url") or f"https://www.youtube.com/watch?v={entry.get('id')}",
                    "duration": entry.get("duration"),
                    "view_count": entry.get("view_count"),
                })
        
        result = {
            "channel": channel_handle,
            "query": query,
            "results": results,
            "count": len(results),
        }
        
        latency = round(time.time() * 1000 - start_ms, 2)
        metrics = {"status": "success", "latency_ms": latency, "count": len(results)}
        return result, metrics
        
    except Exception as e:
        latency = round(time.time() * 1000 - start_ms, 2)
        return {"error": str(e)}, {"status": "error", "latency_ms": latency}


def _channel_info_impl(handle: str) -> tuple[dict, dict]:
    """Get channel metadata.
    
    CLI: channel_info
    MCP: channel_info
    """
    import json
    import time
    import yt_dlp
    
    start_ms = time.time() * 1000
    
    try:
        channel_handle = _extract_channel_handle(handle)
    except ValueError as e:
        latency = round(time.time() * 1000 - start_ms, 2)
        return {"error": str(e)}, {"status": "error", "latency_ms": latency}
    
    channel_url = f"https://www.youtube.com/{channel_handle}"
    
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "playlistend": 1,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
        
        result = {
            "id": info.get("id"),
            "channel": info.get("channel") or info.get("uploader"),
            "channel_id": info.get("channel_id") or info.get("uploader_id"),
            "channel_url": info.get("channel_url") or info.get("uploader_url"),
            "description": info.get("description", ""),
            "channel_follower_count": info.get("channel_follower_count"),
            "playlist_count": info.get("playlist_count"),
            "tags": info.get("tags", []),
            "thumbnail": info.get("thumbnails", [{}])[0].get("url") if info.get("thumbnails") else None,
        }
        
        latency = round(time.time() * 1000 - start_ms, 2)
        metrics = {"status": "success", "latency_ms": latency}
        return result, metrics
        
    except Exception as e:
        latency = round(time.time() * 1000 - start_ms, 2)
        return {"error": str(e)}, {"status": "error", "latency_ms": latency}


def _channel_videos_impl(handle: str, sort: str = "time", order: str = "descending", max_results: int = 10) -> tuple[dict, dict]:
    """List channel videos with sort and order.
    
    CLI: channel_videos
    MCP: channel_videos
    """
    import json
    import time
    import yt_dlp
    
    start_ms = time.time() * 1000
    
    try:
        channel_handle = _extract_channel_handle(handle)
    except ValueError as e:
        latency = round(time.time() * 1000 - start_ms, 2)
        return {"error": str(e)}, {"status": "error", "latency_ms": latency}
    
    if sort not in ("time", "popular"):
        latency = round(time.time() * 1000 - start_ms, 2)
        return {"error": f"Invalid sort '{sort}'. Use 'time' or 'popular'."}, {"status": "error", "latency_ms": latency}
    if order not in ("ascending", "descending"):
        latency = round(time.time() * 1000 - start_ms, 2)
        return {"error": f"Invalid order '{order}'. Use 'ascending' or 'descending'."}, {"status": "error", "latency_ms": latency}
    
    fetch_count = max(max_results * 3, 50) if sort == "popular" else max_results
    channel_url = f"https://www.youtube.com/{channel_handle}/videos"
    
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "playlistend": fetch_count,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
        
        entries = info.get("entries", []) if info else []
        
        videos = []
        for entry in entries:
            if entry:
                videos.append({
                    "id": entry.get("id"),
                    "title": entry.get("title"),
                    "url": f"https://www.youtube.com/watch?v={entry.get('id')}",
                    "duration": entry.get("duration"),
                    "view_count": entry.get("view_count") or 0,
                    "upload_date": entry.get("upload_date") or "",
                })
        
        reverse = (order == "descending")
        if sort == "popular":
            videos.sort(key=lambda v: v["view_count"], reverse=reverse)
        else:
            videos.sort(key=lambda v: v["upload_date"], reverse=reverse)
        
        videos = videos[:max_results]
        
        result = {
            "channel": channel_handle,
            "sort": sort,
            "order": order,
            "results": videos,
            "count": len(videos),
        }
        
        latency = round(time.time() * 1000 - start_ms, 2)
        metrics = {"status": "success", "latency_ms": latency, "count": len(videos)}
        return result, metrics
        
    except Exception as e:
        latency = round(time.time() * 1000 - start_ms, 2)
        return {"error": str(e)}, {"status": "error", "latency_ms": latency}


def _view_frames_impl(
    url: str,
    timestamps: list[str] | None = None,
    threshold: float = 0.3,
    min_gap: float = 5.0,
    output_dir: str = "",
) -> tuple[dict, dict]:
    """Extract key frames from a YouTube video via scene detection or specific timestamps.

    Args:
        url: YouTube URL or video ID
        timestamps: Specific timestamps to extract (e.g. ["4:32", "12:15"]). If None, uses scene detection.
        threshold: Scene change sensitivity 0.0-1.0 (default 0.3, higher = fewer frames)
        min_gap: Minimum seconds between scene-detected frames (default 5.0)
        output_dir: Output directory (default: data/youtube_frames/<video_id>)
    """
    import json
    import shutil
    import subprocess
    import tempfile
    import time

    start_ms = time.time() * 1000
    video_id = _extract_video_id(url)
    _log("INFO", "view_frames_start", f"video={video_id}")

    out_dir = Path(output_dir) if output_dir else Path("data/youtube_frames") / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean any existing frames
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()

    try:
        # Step 1: Download video
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "video.mp4"
            dl_cmd = [
                "yt-dlp", "-f", "bestvideo[height<=720]",
                "-o", str(video_path),
                f"https://www.youtube.com/watch?v={video_id}",
            ]
            dl = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=120)
            assert dl.returncode == 0, f"yt-dlp failed: {dl.stderr[-500:]}"
            assert video_path.exists(), "Downloaded video file not found"

            if timestamps:
                # Mode: extract specific timestamps
                frames = []
                for i, ts in enumerate(timestamps):
                    parts = ts.strip().split(":")
                    assert len(parts) in (2, 3), f"Invalid timestamp format: {ts} (use M:SS or H:MM:SS)"
                    if len(parts) == 2:
                        secs = int(parts[0]) * 60 + int(parts[1])
                    else:
                        secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    mins = secs // 60
                    s = secs % 60
                    fname = f"frame_{i+1:02d}_{mins:02d}m{s:02d}s.jpg"
                    fpath = out_dir / fname
                    extract_cmd = [
                        "ffmpeg", "-y", "-ss", str(secs),
                        "-i", str(video_path),
                        "-frames:v", "1", "-q:v", "2",
                        str(fpath),
                    ]
                    r = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=30)
                    if r.returncode == 0 and fpath.exists():
                        frames.append({"file": fname, "timestamp": ts, "seconds": secs})
                    else:
                        _log("WARN", "frame_extract_fail", f"ts={ts}: {r.stderr[-200:]}")

            else:
                # Mode: scene detection
                # Single pass: extract frames + capture timestamps via showinfo
                extract_cmd = [
                    "ffmpeg", "-i", str(video_path),
                    "-vf", f"select='gt(scene,{threshold})',showinfo",
                    "-vsync", "vfr", "-q:v", "2",
                    str(out_dir / "frame_%04d.jpg"),
                ]
                r = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=300)

                # Parse timestamps from showinfo in stderr
                raw_times = []
                for line in r.stderr.splitlines():
                    if "pts_time:" in line:
                        pts = line.split("pts_time:")[1].split()[0]
                        raw_times.append(float(pts))

                # Apply min_gap filter and rename
                frames = []
                kept_idx = 0
                last_kept_time = -min_gap  # ensure first frame always passes
                for i, secs in enumerate(raw_times):
                    seq_file = out_dir / f"frame_{i+1:04d}.jpg"
                    if not seq_file.exists():
                        continue
                    if secs - last_kept_time < min_gap:
                        seq_file.unlink()
                        continue
                    last_kept_time = secs
                    kept_idx += 1
                    mins = int(secs) // 60
                    s = int(secs) % 60
                    fname = f"frame_{kept_idx:02d}_{mins:02d}m{s:02d}s.jpg"
                    seq_file.rename(out_dir / fname)
                    frames.append({"file": fname, "timestamp": f"{mins}:{s:02d}", "seconds": round(secs, 1)})

        latency = round(time.time() * 1000 - start_ms, 2)
        result = {
            "video_id": video_id,
            "mode": "timestamps" if timestamps else "scene_detect",
            "output_dir": str(out_dir),
            "frame_count": len(frames),
            "frames": frames,
        }
        if not timestamps:
            result["threshold"] = threshold
            result["min_gap_seconds"] = min_gap
        metrics = {"status": "success", "latency_ms": latency}
        _log("INFO", "view_frames_done", f"video={video_id} frames={len(frames)}")
        return result, metrics

    except Exception as e:
        latency = round(time.time() * 1000 - start_ms, 2)
        _log("ERROR", "view_frames_error", str(e))
        return {"error": str(e)}, {"status": "error", "latency_ms": latency}


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    import argparse
    import json
    import sys
    
    parser = argparse.ArgumentParser(description="YouTube video operations")
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 1.0.0")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # mcp-stdio
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")
    
    # info
    p_info = subparsers.add_parser("info", help="Get video metadata")
    p_info.add_argument("url", nargs="?", help="YouTube URL or video ID")
    
    # transcript
    p_trans = subparsers.add_parser("transcript", help="Get video transcript")
    p_trans.add_argument("url", nargs="?", help="YouTube URL or video ID")
    p_trans.add_argument("-l", "--language", default="en", help="Language code")
    p_trans.add_argument("-t", "--timestamps", action="store_true", help="Include timestamps")
    
    # channel_search
    p_csearch = subparsers.add_parser("channel_search", help="Search within a channel")
    p_csearch.add_argument("handle", nargs="?", help="Channel handle (@username) or URL")
    p_csearch.add_argument("query", nargs="?", help="Search query")
    p_csearch.add_argument("-n", "--max-results", type=int, default=10, help="Max results")
    
    # channel_info
    p_cinfo = subparsers.add_parser("channel_info", help="Get channel metadata")
    p_cinfo.add_argument("handle", nargs="?", help="Channel handle (@username) or URL")
    
    # channel_videos
    p_cvids = subparsers.add_parser("channel_videos", help="List channel videos with sort")
    p_cvids.add_argument("handle", nargs="?", help="Channel handle (@username) or URL")
    p_cvids.add_argument("-s", "--sort", default="time", choices=["time", "popular"], help="Sort field")
    p_cvids.add_argument("-o", "--order", default="descending", choices=["ascending", "descending"], help="Sort direction")
    p_cvids.add_argument("-n", "--max-results", type=int, default=10, help="Max results")
    
    # view_frames
    p_vf = subparsers.add_parser("view_frames", help="Extract key frames from video")
    p_vf.add_argument("url", nargs="?", help="YouTube URL or video ID")
    p_vf.add_argument("-a", "--at", help="Specific timestamps (comma-separated, e.g. '4:32,12:15')")
    p_vf.add_argument("-t", "--threshold", type=float, default=0.3, help="Scene detection sensitivity 0-1 (default 0.3)")
    p_vf.add_argument("-g", "--min-gap", type=float, default=5.0, help="Min seconds between frames (default 5)")
    p_vf.add_argument("-d", "--output-dir", default="", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "info":
            url = args.url
            if not url and not sys.stdin.isatty():
                url = sys.stdin.read().strip()
            assert url, "url required (positional argument or stdin)"
            result, metrics = _info_impl(url)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
        elif args.command == "transcript":
            url = args.url
            if not url and not sys.stdin.isatty():
                url = sys.stdin.read().strip()
            assert url, "url required (positional argument or stdin)"
            result, metrics = _transcript_impl(url, args.language, args.timestamps)
            print(json.dumps({"transcript": result, "metrics": metrics}, indent=2))
        elif args.command == "channel_search":
            handle = args.handle
            query = args.query
            if not handle and not sys.stdin.isatty():
                parts = sys.stdin.read().strip().split(None, 1)
                handle = parts[0] if parts else ""
                query = parts[1] if len(parts) > 1 else ""
            assert handle, "handle required (positional argument or stdin: 'handle query')"
            assert query, "query required (positional argument or stdin: 'handle query')"
            result, metrics = _channel_search_impl(handle, query, args.max_results)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
        elif args.command == "channel_info":
            handle = args.handle
            if not handle and not sys.stdin.isatty():
                handle = sys.stdin.read().strip()
            assert handle, "handle required (positional argument or stdin)"
            result, metrics = _channel_info_impl(handle)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
        elif args.command == "channel_videos":
            handle = args.handle
            if not handle and not sys.stdin.isatty():
                handle = sys.stdin.read().strip()
            assert handle, "handle required (positional argument or stdin)"
            result, metrics = _channel_videos_impl(handle, args.sort, args.order, args.max_results)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
        elif args.command == "view_frames":
            url = args.url
            if not url and not sys.stdin.isatty():
                url = sys.stdin.read().strip()
            assert url, "url required (positional argument or stdin)"
            timestamps = args.at.split(",") if args.at else None
            result, metrics = _view_frames_impl(url, timestamps, args.threshold, args.min_gap, args.output_dir)
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
    
    mcp = FastMCP("ytv")
    
    @mcp.tool()
    def info(url: str) -> str:
        """Get YouTube video metadata (title, description, duration, views, etc).
        
        Args:
            url: YouTube URL or video ID
        """
        result, metrics = _info_impl(url)
        return json.dumps({"result": result, "metrics": metrics})
    
    @mcp.tool()
    def transcript(url: str, language: str = "en", timestamps: bool = False) -> str:
        """Get YouTube video transcript.
        
        Args:
            url: YouTube URL or video ID
            language: Language code (default "en")
            timestamps: Include timestamps in output
        """
        result, metrics = _transcript_impl(url, language, timestamps)
        return json.dumps({"transcript": result, "metrics": metrics})
    
    @mcp.tool()
    def channel_search(handle: str, query: str, max_results: int = 10) -> str:
        """Search within a YouTube channel for videos matching a query.
        
        Args:
            handle: Channel handle (@username) or URL
            query: Search query
            max_results: Maximum number of results (default 10)
        """
        result, metrics = _channel_search_impl(handle, query, max_results)
        return json.dumps({"result": result, "metrics": metrics})
    
    @mcp.tool()
    def channel_info(handle: str) -> str:
        """Get YouTube channel metadata.
        
        Args:
            handle: Channel handle (@username) or URL
        """
        result, metrics = _channel_info_impl(handle)
        return json.dumps({"result": result, "metrics": metrics})
    
    @mcp.tool()
    def channel_videos(handle: str, sort: str = "time", order: str = "descending", max_results: int = 10) -> str:
        """List videos from a YouTube channel with sort order.
        
        Args:
            handle: Channel handle (@username) or URL
            sort: Sort order - "time" (default), "popular"
            order: Direction - "descending" (default), "ascending"
            max_results: Maximum number of results (default 10)
        """
        result, metrics = _channel_videos_impl(handle, sort, order, max_results)
        return json.dumps({"result": result, "metrics": metrics})
    
    @mcp.tool()
    def view_frames(
        url: str,
        timestamps: str = "",
        threshold: float = 0.3,
        min_gap: float = 5.0,
        output_dir: str = "",
    ) -> str:
        """Extract key frames from a YouTube video via scene detection or specific timestamps.
        
        Default: scene detection extracts frames at visual transitions (slide changes, code switches).
        Frames saved to data/youtube_frames/<video_id>/ with timestamped filenames.
        
        Args:
            url: YouTube URL or video ID
            timestamps: Comma-separated timestamps to extract (e.g. "4:32,12:15"). If empty, uses scene detection.
            threshold: Scene detection sensitivity 0.0-1.0 (default 0.3, higher = fewer frames)
            min_gap: Minimum seconds between scene-detected frames (default 5.0)
            output_dir: Output directory (default: data/youtube_frames/<video_id>)
        """
        ts_list = [t.strip() for t in timestamps.split(",") if t.strip()] if timestamps else None
        result, metrics = _view_frames_impl(url, ts_list, threshold, min_gap, output_dir)
        return json.dumps({"result": result, "metrics": metrics})
    
    print("ytv MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
