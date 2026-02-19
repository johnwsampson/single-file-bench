#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastmcp",
#     "pyobjc-framework-Quartz>=11.0",
#     "pyobjc-framework-Vision>=11.0",
# ]
# ///
"""Screen capture for AI agents: grab, scan, read.

Seven tools for screen/app capture with object-relative coordinates:
  list         — enumerate capturable windows
  grab_app     — capture app window at full resolution
  grab_screen  — capture full screen at full resolution
  scan_app     — view app capture (budget-downsized for MCP)
  scan_screen  — view screen capture (budget-downsized for MCP)
  read_app     — OCR text from app capture (full resolution)
  read_screen  — OCR text from screen capture (full resolution)

Workflow: grab first, then scan (see layout) or read (extract text).
All coordinates are object-relative logical points (0,0 = top-left of target).

Usage:
    sft_vshot.py list
    sft_vshot.py grab-app "Terminal"
    sft_vshot.py grab-app "Terminal" --coords 681,100 1135,400
    sft_vshot.py scan-app
    sft_vshot.py scan-app --coords 681,100 1135,400
    sft_vshot.py read-app
    sft_vshot.py read-app --coords 681,100 1135,400
    sft_vshot.py mcp-stdio
"""

import argparse

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
            f.write(
                f"{ts}\t{_SCRIPT}\t{level}\t{event}\t{msg}\t{detail}\t{metrics}\t{trace}\n"
            )
    except Exception:
        pass


# =============================================================================
# CONFIGURATION
# =============================================================================
EXPOSED = ["list", "grab_app", "grab_screen", "scan_app", "scan_screen", "read_app", "read_screen"]

APP_CACHE = "/tmp/vshot_app.png"
SCREEN_CACHE = "/tmp/vshot_screen.png"
SCAN_TMP = "/tmp/vshot_scan.jpg"

CONFIG = {
    "max_pixels": 1_150_000,      # Claude vision sweet spot: 1.15 megapixels (~1600 tokens)
    "max_edge": 1568,             # Claude vision max dimension without auto-resize
    "scan_quality": 0.65,         # JPEG quality for scan output
    "scale_factor": 2.0,          # Retina 2x — logical to physical
    "min_window_height_px": 100,  # Skip title bars / chrome layers
}


# =============================================================================
# IMAGE HELPERS — CoreGraphics only, no Pillow, no subprocess
# =============================================================================

def _cg_imports():
    """Lazy import of CoreGraphics symbols. Called once per command that needs capture."""
    from Quartz import (
        CGWindowListCreateImage,
        CGWindowListCopyWindowInfo,
        CGImageCreateWithImageInRect,
        CGImageGetWidth,
        CGImageGetHeight,
        CGRectInfinite,
        CGRectMake,
        kCGWindowListOptionOnScreenOnly,
        kCGWindowListOptionIncludingWindow,
        kCGNullWindowID,
        kCGWindowImageDefault,
        kCGWindowImageBoundsIgnoreFraming,
        CGBitmapContextCreate,
        CGBitmapContextCreateImage,
        CGContextDrawImage,
        CGColorSpaceCreateDeviceRGB,
        CGImageDestinationCreateWithURL,
        CGImageDestinationAddImage,
        CGImageDestinationFinalize,
    )
    from CoreFoundation import (
        CFURLCreateWithFileSystemPath,
        kCFURLPOSIXPathStyle,
        kCFAllocatorDefault,
    )
    import Quartz as _Q
    kCGBitmapByteOrder32Host = _Q.kCGBitmapByteOrder32Host
    kCGImageAlphaPremultipliedFirst = _Q.kCGImageAlphaPremultipliedFirst
    return {
        "CGWindowListCreateImage": CGWindowListCreateImage,
        "CGWindowListCopyWindowInfo": CGWindowListCopyWindowInfo,
        "CGImageCreateWithImageInRect": CGImageCreateWithImageInRect,
        "CGImageGetWidth": CGImageGetWidth,
        "CGImageGetHeight": CGImageGetHeight,
        "CGRectInfinite": CGRectInfinite,
        "CGRectMake": CGRectMake,
        "kCGWindowListOptionOnScreenOnly": kCGWindowListOptionOnScreenOnly,
        "kCGWindowListOptionIncludingWindow": kCGWindowListOptionIncludingWindow,
        "kCGNullWindowID": kCGNullWindowID,
        "kCGWindowImageDefault": kCGWindowImageDefault,
        "kCGWindowImageBoundsIgnoreFraming": kCGWindowImageBoundsIgnoreFraming,
        "CGBitmapContextCreate": CGBitmapContextCreate,
        "CGBitmapContextCreateImage": CGBitmapContextCreateImage,
        "CGContextDrawImage": CGContextDrawImage,
        "CGColorSpaceCreateDeviceRGB": CGColorSpaceCreateDeviceRGB,
        "CGImageDestinationCreateWithURL": CGImageDestinationCreateWithURL,
        "CGImageDestinationAddImage": CGImageDestinationAddImage,
        "CGImageDestinationFinalize": CGImageDestinationFinalize,
        "CFURLCreateWithFileSystemPath": CFURLCreateWithFileSystemPath,
        "kCFURLPOSIXPathStyle": kCFURLPOSIXPathStyle,
        "kCFAllocatorDefault": kCFAllocatorDefault,
        "kCGBitmapByteOrder32Host": kCGBitmapByteOrder32Host,
        "kCGImageAlphaPremultipliedFirst": kCGImageAlphaPremultipliedFirst,
    }


def _make_cf_url(path: str, cg):
    """Create a CoreFoundation URL from a filesystem path."""
    return cg["CFURLCreateWithFileSystemPath"](
        cg["kCFAllocatorDefault"],
        path,
        cg["kCFURLPOSIXPathStyle"],
        False,
    )


def _save_jpeg(cg_image, path: str, quality: float, cg: dict):
    """Save CGImage as JPEG to path using ImageIO."""
    url = _make_cf_url(path, cg)
    assert url is not None, f"CoreFoundation URL created for {path}"
    dest = cg["CGImageDestinationCreateWithURL"](url, "public.jpeg", 1, None)
    assert dest is not None, f"JPEG destination created for {path}"
    props = {"kCGImageDestinationLossyCompressionQuality": quality}
    cg["CGImageDestinationAddImage"](dest, cg_image, props)
    ok = cg["CGImageDestinationFinalize"](dest)
    assert ok, f"JPEG write finalized to {path}"


def _save_png(cg_image, path: str, cg: dict):
    """Save CGImage as PNG to path using ImageIO."""
    url = _make_cf_url(path, cg)
    assert url is not None, f"CoreFoundation URL created for {path}"
    dest = cg["CGImageDestinationCreateWithURL"](url, "public.png", 1, None)
    assert dest is not None, f"PNG destination created for {path}"
    cg["CGImageDestinationAddImage"](dest, cg_image, None)
    ok = cg["CGImageDestinationFinalize"](dest)
    assert ok, f"PNG write finalized to {path}"


def _downscale(cg_image, target_width: int, cg: dict):
    """Downscale CGImage to target_width, preserving aspect ratio."""
    src_w = cg["CGImageGetWidth"](cg_image)
    src_h = cg["CGImageGetHeight"](cg_image)
    assert src_w > 0, f"Source image has positive width (got {src_w})"
    assert src_h > 0, f"Source image has positive height (got {src_h})"

    scale = target_width / src_w
    dst_w = target_width
    dst_h = int(src_h * scale)

    color_space = cg["CGColorSpaceCreateDeviceRGB"]()
    bitmap_info = (
        cg["kCGBitmapByteOrder32Host"] | cg["kCGImageAlphaPremultipliedFirst"]
    )
    ctx = cg["CGBitmapContextCreate"](
        None, dst_w, dst_h, 8, dst_w * 4, color_space, bitmap_info
    )
    assert ctx is not None, f"CGBitmapContext created at {dst_w}x{dst_h}"

    cg["CGContextDrawImage"](ctx, cg["CGRectMake"](0, 0, dst_w, dst_h), cg_image)
    result = cg["CGBitmapContextCreateImage"](ctx)
    assert result is not None, "Downscaled CGImage extracted from context"
    return result


def _crop_image(cg_image, x1: int, y1: int, x2: int, y2: int, scale_factor: float, cg: dict):
    """Crop CGImage to logical region, mapping logical coords to physical pixels.

    CGImageCreateWithImageInRect uses top-left origin (matching the image pixel
    layout), so logical coords map directly: y=0 is the top of the image.
    """
    img_w = cg["CGImageGetWidth"](cg_image)
    img_h = cg["CGImageGetHeight"](cg_image)

    # Convert logical to physical pixels
    px1 = int(x1 * scale_factor)
    py1 = int(y1 * scale_factor)
    px2 = int(x2 * scale_factor)
    py2 = int(y2 * scale_factor)

    # Clamp to image bounds
    px1 = max(0, min(px1, img_w))
    py1 = max(0, min(py1, img_h))
    px2 = max(0, min(px2, img_w))
    py2 = max(0, min(py2, img_h))

    crop_w = px2 - px1
    crop_h = py2 - py1
    assert crop_w > 0, f"Crop width must be positive (x1={x1} x2={x2}, physical {px1}-{px2})"
    assert crop_h > 0, f"Crop height must be positive (y1={y1} y2={y2}, physical {py1}-{py2})"

    # Top-left origin: y=0 is top of image, no flip needed
    rect = cg["CGRectMake"](px1, py1, crop_w, crop_h)
    cropped = cg["CGImageCreateWithImageInRect"](cg_image, rect)
    assert cropped is not None, f"Crop succeeded at rect ({px1},{py1},{crop_w},{crop_h})"
    return cropped


def _load_png_as_cg_image(path: str, cg: dict):
    """Load a PNG file back as a CGImage for cropping."""
    import Quartz
    url = _make_cf_url(path, cg)
    assert url is not None, f"CoreFoundation URL for {path}"
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    assert src is not None, f"CGImageSource created from {path}"
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    assert img is not None, f"CGImage loaded from {path}"
    return img


def _ocr_image(cg_image) -> str:
    """Extract text from a CGImage using macOS Vision framework.

    Takes a CGImage directly — always pass the full-resolution image,
    never a compressed/downscaled JPEG. OCR quality depends on input quality.
    """
    import Vision

    assert cg_image is not None, "CGImage required for OCR"

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cg_image, None
    )
    success = handler.performRequests_error_([request], None)
    assert success[0], f"Vision OCR request succeeded (error: {success[1]})"

    results = request.results() or []
    lines = []
    for obs in results:
        candidate = obs.topCandidates_(1)
        if candidate:
            text = candidate[0].string()
            bbox = obs.boundingBox()
            y_top = 1.0 - (bbox.origin.y + bbox.size.height)
            lines.append((y_top, bbox.origin.x, text))

    lines.sort(key=lambda t: (t[0], t[1]))
    return "\n".join(line[2] for line in lines)



def _parse_coord(s: str) -> tuple[int, int]:
    """Parse 'x,y' coordinate string into (x, y) integers."""
    parts = s.split(",")
    assert len(parts) == 2, f"Coordinate must be 'x,y' format (got '{s}')"
    return int(parts[0].strip()), int(parts[1].strip())


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _has_coords(x1: int, y1: int, x2: int, y2: int) -> bool:
    """True if coordinates specify a crop region (any non-zero)."""
    return not (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0)


def _maybe_crop(cg_image, x1: int, y1: int, x2: int, y2: int, cg: dict):
    """Crop image to coords if given. Returns original if no coords."""
    if not _has_coords(x1, y1, x2, y2):
        return cg_image
    return _crop_image(cg_image, x1, y1, x2, y2, CONFIG["scale_factor"], cg)


def _budget_fit(cg_image, cg: dict):
    """Downscale image to fit Claude vision limits. Returns (image, quality).

    Constrains to max_pixels (1.15MP) and max_edge (1568px) for optimal
    vision token cost (~1600 tokens) without triggering server-side resize.
    """
    w = cg["CGImageGetWidth"](cg_image)
    h = cg["CGImageGetHeight"](cg_image)

    quality = CONFIG["scan_quality"]
    max_pixels = CONFIG["max_pixels"]
    max_edge = CONFIG["max_edge"]

    # Check edge constraint first
    longest = max(w, h)
    if longest > max_edge:
        scale = max_edge / longest
        w = int(w * scale)
        h = int(h * scale)

    # Check pixel budget
    if w * h <= max_pixels:
        if longest <= max_edge:
            return cg_image, quality
        target_w = w
    else:
        aspect = w / h
        target_w = int(math.sqrt(max_pixels * aspect))

    target_w = max(target_w, 64)
    return _downscale(cg_image, target_w, cg), quality


def _img_dims(cg_image, cg: dict) -> dict:
    """Return physical px and logical pts dimensions for a CGImage."""
    w = cg["CGImageGetWidth"](cg_image)
    h = cg["CGImageGetHeight"](cg_image)
    s = CONFIG["scale_factor"]
    return {
        "physical_px": {"w": w, "h": h},
        "logical_pts": {"w": int(w / s), "h": int(h / s)},
    }


# =============================================================================
# CORE FUNCTIONS — _impl pattern, shared by CLI and MCP
# =============================================================================


def _list_impl() -> tuple[list, dict]:
    """Enumerate on-screen windows via CGWindowListCopyWindowInfo."""
    t0 = time.monotonic()
    cg = _cg_imports()

    raw = cg["CGWindowListCopyWindowInfo"](
        cg["kCGWindowListOptionOnScreenOnly"], cg["kCGNullWindowID"]
    )
    assert raw is not None, "CGWindowListCopyWindowInfo returns window list"

    windows = []
    seen_apps = set()
    for w in raw:
        owner = w.get("kCGWindowOwnerName", "")
        name = w.get("kCGWindowName", "")
        bounds = w.get("kCGWindowBounds", {})
        wid = w.get("kCGWindowNumber", 0)
        layer = w.get("kCGWindowLayer", 0)
        if not owner:
            continue
        seen_apps.add(owner)
        windows.append({
            "app": owner,
            "window": name or "",
            "id": wid,
            "layer": layer,
            "bounds": {
                "x": int(bounds.get("X", 0)),
                "y": int(bounds.get("Y", 0)),
                "w": int(bounds.get("Width", 0)),
                "h": int(bounds.get("Height", 0)),
            },
        })

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    _log(
        "INFO", "list", f"Found {len(windows)} windows from {len(seen_apps)} apps",
        metrics=f"elapsed_ms={elapsed_ms} windows={len(windows)} apps={len(seen_apps)}"
    )
    return windows, {"elapsed_ms": elapsed_ms, "windows": len(windows), "apps": len(seen_apps)}


def _grab_screen_impl(x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0) -> tuple[dict, dict]:
    """Capture full screen to cache at full resolution. Optional coords crop."""
    t0 = time.monotonic()
    cg = _cg_imports()

    img = cg["CGWindowListCreateImage"](
        cg["CGRectInfinite"],
        cg["kCGWindowListOptionOnScreenOnly"],
        cg["kCGNullWindowID"],
        cg["kCGWindowImageDefault"],
    )
    assert img is not None, (
        "Screen capture succeeded. Ensure Screen Recording permission is granted "
        "to Terminal in System Settings > Privacy & Security > Screen Recording."
    )

    img = _maybe_crop(img, x1, y1, x2, y2, cg)
    _save_png(img, SCREEN_CACHE, cg)

    dims = _img_dims(img, cg)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    result = {
        "target": "screen",
        **dims,
        "scale_factor": CONFIG["scale_factor"],
        "cropped": _has_coords(x1, y1, x2, y2),
        "cache": SCREEN_CACHE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    px = dims["physical_px"]
    _log("INFO", "grab_screen",
         f"Captured screen {px['w']}x{px['h']}px cropped={result['cropped']}",
         metrics=f"elapsed_ms={elapsed_ms}")
    return result, {"elapsed_ms": elapsed_ms}


def _grab_app_impl(name: str, x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0) -> tuple[dict, dict]:
    """Capture app window to cache at full resolution. Optional coords crop."""
    t0 = time.monotonic()
    cg = _cg_imports()

    raw = cg["CGWindowListCopyWindowInfo"](
        cg["kCGWindowListOptionOnScreenOnly"], cg["kCGNullWindowID"]
    )
    assert raw is not None, "Window list available"

    name_lower = name.lower()
    candidates = []
    for w in raw:
        owner = w.get("kCGWindowOwnerName", "")
        if name_lower not in owner.lower():
            continue
        bounds = w.get("kCGWindowBounds", {})
        h = int(bounds.get("Height", 0))
        if h < CONFIG["min_window_height_px"]:
            continue
        w_px = int(bounds.get("Width", 0))
        candidates.append((w_px * h, w))

    if not candidates:
        all_apps = sorted({
            w.get("kCGWindowOwnerName", "")
            for w in raw
            if w.get("kCGWindowOwnerName", "")
        })
        assert False, (
            f"No window found matching '{name}'. "
            f"Running apps: {', '.join(all_apps)}"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, win = candidates[0]

    wid = win["kCGWindowNumber"]
    bounds = win["kCGWindowBounds"]
    bx = int(bounds.get("X", 0))
    by = int(bounds.get("Y", 0))
    bw = int(bounds.get("Width", 0))
    bh = int(bounds.get("Height", 0))

    win_rect = cg["CGRectMake"](bx, by, bw, bh)
    img = cg["CGWindowListCreateImage"](
        win_rect,
        cg["kCGWindowListOptionIncludingWindow"],
        wid,
        cg["kCGWindowImageBoundsIgnoreFraming"],
    )
    assert img is not None, (
        f"Window capture succeeded for '{win.get('kCGWindowOwnerName', name)}'. "
        "Ensure Screen Recording permission is granted."
    )

    img = _maybe_crop(img, x1, y1, x2, y2, cg)
    _save_png(img, APP_CACHE, cg)

    dims = _img_dims(img, cg)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    result = {
        "target": "app",
        "app": win.get("kCGWindowOwnerName", name),
        "window": win.get("kCGWindowName", ""),
        "window_id": wid,
        **dims,
        "scale_factor": CONFIG["scale_factor"],
        "cropped": _has_coords(x1, y1, x2, y2),
        "cache": APP_CACHE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    px = dims["physical_px"]
    _log("INFO", "grab_app",
         f"Captured '{result['app']}' {px['w']}x{px['h']}px cropped={result['cropped']}",
         detail=f"name={name} wid={wid}", metrics=f"elapsed_ms={elapsed_ms}")
    return result, {"elapsed_ms": elapsed_ms}


def _scan_impl(cache_path: str, x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0) -> tuple[dict, dict]:
    """Budget-downsized view from cache. Optional coords for region."""
    t0 = time.monotonic()
    assert Path(cache_path).exists(), (
        f"No capture at {cache_path}. Run grab_app or grab_screen first."
    )
    cg = _cg_imports()

    img = _load_png_as_cg_image(cache_path, cg)
    source_dims = _img_dims(img, cg)
    img = _maybe_crop(img, x1, y1, x2, y2, cg)
    img, quality = _budget_fit(img, cg)

    _save_jpeg(img, SCAN_TMP, quality, cg)

    out_w = cg["CGImageGetWidth"](img)
    out_h = cg["CGImageGetHeight"](img)
    file_bytes = Path(SCAN_TMP).stat().st_size

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    result = {
        "source_pts": source_dims["logical_pts"],
        "output_px": {"w": out_w, "h": out_h},
        "jpeg_quality": round(quality, 2),
        "file_bytes": file_bytes,
        "cropped": _has_coords(x1, y1, x2, y2),
    }
    _log("INFO", "scan",
         f"scan {cache_path} out={out_w}x{out_h} q={quality:.2f} bytes={file_bytes} cropped={result['cropped']}",
         metrics=f"elapsed_ms={elapsed_ms} file_bytes={file_bytes}")
    return result, {"elapsed_ms": elapsed_ms}


def _read_impl(cache_path: str, x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0) -> tuple[dict, dict]:
    """OCR text from full-res cache. Optional coords for region."""
    t0 = time.monotonic()
    assert Path(cache_path).exists(), (
        f"No capture at {cache_path}. Run grab_app or grab_screen first."
    )
    cg = _cg_imports()

    img = _load_png_as_cg_image(cache_path, cg)
    source_dims = _img_dims(img, cg)
    img = _maybe_crop(img, x1, y1, x2, y2, cg)

    text = _ocr_image(img)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    result = {
        "ocr_text": text,
        "source_pts": source_dims["logical_pts"],
        "cropped": _has_coords(x1, y1, x2, y2),
    }
    _log("INFO", "read",
         f"OCR {cache_path} chars={len(text)} cropped={result['cropped']}",
         metrics=f"elapsed_ms={elapsed_ms} chars={len(text)}")
    return result, {"elapsed_ms": elapsed_ms}


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Screen capture for AI agents: grab, scan, read.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. list                    -- see what's capturable
  2. grab-app / grab-screen  -- capture at full resolution
  3. scan-app / scan-screen  -- view (budget-downsized for transfer)
  4. read-app / read-screen  -- OCR text extraction (full resolution)

Coords are object-relative logical points (0,0 = top-left of target).
Use --coords x1,y1 x2,y2 to focus on a region.
""",
    )
    parser.add_argument("-V", "--version", action="version", version="sft_vshot 3.0.0")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # list
    sub.add_parser("list", help="Enumerate capturable apps and windows")

    # grab-app
    p_ga = sub.add_parser("grab-app", help="Capture app window at full resolution")
    p_ga.add_argument("name", nargs="?", default="", help="App name (case-insensitive substring match, or stdin)")
    p_ga.add_argument("--coords", "-c", nargs=2, metavar=("x1,y1", "x2,y2"),
                       help="Optional crop region")

    # grab-screen
    p_gs = sub.add_parser("grab-screen", help="Capture full screen at full resolution")
    p_gs.add_argument("--coords", "-c", nargs=2, metavar=("x1,y1", "x2,y2"),
                       help="Optional crop region")

    # scan-app
    p_sa = sub.add_parser("scan-app", help="View app capture (budget-downsized)")
    p_sa.add_argument("--coords", "-c", nargs=2, metavar=("x1,y1", "x2,y2"),
                       help="Optional region")

    # scan-screen
    p_ss = sub.add_parser("scan-screen", help="View screen capture (budget-downsized)")
    p_ss.add_argument("--coords", "-c", nargs=2, metavar=("x1,y1", "x2,y2"),
                       help="Optional region")

    # read-app
    p_ra = sub.add_parser("read-app", help="OCR text from app capture")
    p_ra.add_argument("--coords", "-c", nargs=2, metavar=("x1,y1", "x2,y2"),
                       help="Optional region")

    # read-screen
    p_rs = sub.add_parser("read-screen", help="OCR text from screen capture")
    p_rs.add_argument("--coords", "-c", nargs=2, metavar=("x1,y1", "x2,y2"),
                       help="Optional region")

    # MCP server
    sub.add_parser("mcp-stdio", help="Run as MCP server (stdio transport)")

    args = parser.parse_args()

    try:
        if args.command == "mcp-stdio":
            _run_mcp()
            return

        elif args.command == "list":
            windows, metrics = _list_impl()
            print(json.dumps({"windows": windows, "metrics": metrics}, indent=2))

        elif args.command == "grab-app":
            name = args.name
            if not name and not sys.stdin.isatty():
                name = sys.stdin.read().strip()
            assert name, "app name required (positional argument or stdin)"
            x1, y1, x2, y2 = 0, 0, 0, 0
            if args.coords:
                x1, y1 = _parse_coord(args.coords[0])
                x2, y2 = _parse_coord(args.coords[1])
            result, metrics = _grab_app_impl(name, x1, y1, x2, y2)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))

        elif args.command == "grab-screen":
            x1, y1, x2, y2 = 0, 0, 0, 0
            if args.coords:
                x1, y1 = _parse_coord(args.coords[0])
                x2, y2 = _parse_coord(args.coords[1])
            result, metrics = _grab_screen_impl(x1, y1, x2, y2)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))

        elif args.command == "scan-app":
            x1, y1, x2, y2 = 0, 0, 0, 0
            if args.coords:
                x1, y1 = _parse_coord(args.coords[0])
                x2, y2 = _parse_coord(args.coords[1])
            result, metrics = _scan_impl(APP_CACHE, x1, y1, x2, y2)
            result["scan_file"] = SCAN_TMP
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))

        elif args.command == "scan-screen":
            x1, y1, x2, y2 = 0, 0, 0, 0
            if args.coords:
                x1, y1 = _parse_coord(args.coords[0])
                x2, y2 = _parse_coord(args.coords[1])
            result, metrics = _scan_impl(SCREEN_CACHE, x1, y1, x2, y2)
            result["scan_file"] = SCAN_TMP
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))

        elif args.command == "read-app":
            x1, y1, x2, y2 = 0, 0, 0, 0
            if args.coords:
                x1, y1 = _parse_coord(args.coords[0])
                x2, y2 = _parse_coord(args.coords[1])
            result, metrics = _read_impl(APP_CACHE, x1, y1, x2, y2)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))

        elif args.command == "read-screen":
            x1, y1, x2, y2 = 0, 0, 0, 0
            if args.coords:
                x1, y1 = _parse_coord(args.coords[0])
                x2, y2 = _parse_coord(args.coords[1])
            result, metrics = _read_impl(SCREEN_CACHE, x1, y1, x2, y2)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))

        else:
            parser.print_help()
            sys.exit(1)

    except AssertionError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        _log("ERROR", args.command or "unknown", str(e))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        _log("ERROR", args.command or "unknown", str(e))
        sys.exit(1)


# =============================================================================
# FASTMCP SERVER
# =============================================================================

def _run_mcp():
    """Start FastMCP server with all tools."""
    from fastmcp import FastMCP
    from fastmcp.utilities.types import Image

    mcp = FastMCP("vshot")

    @mcp.tool()
    def list() -> str:
        """List capturable apps and their windows.

        Args:
            (no arguments)

        Returns:
            JSON with windows array and metrics.
        """
        windows, metrics = _list_impl()
        return json.dumps({"windows": windows, "metrics": metrics})

    @mcp.tool()
    def grab_app(name: str, x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0) -> str:
        """Capture app window at full resolution. Call scan_app to view, read_app for OCR.

        Captures the full app window. If coordinates are given, crops the stored
        result to that region. Subsequent scan_app/read_app operate on this cache.

        Args:
            name: App name -- case-insensitive substring match.
            x1: Left edge in logical points (0 = no crop).
            y1: Top edge in logical points (0 = no crop).
            x2: Right edge in logical points (0 = no crop).
            y2: Bottom edge in logical points (0 = no crop).

        Returns:
            JSON with capture metadata (dimensions, path, timestamp).
        """
        result, metrics = _grab_app_impl(name, x1, y1, x2, y2)
        return json.dumps({"result": result, "metrics": metrics})

    @mcp.tool()
    def grab_screen(x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0) -> str:
        """Capture full screen at full resolution. Call scan_screen to view, read_screen for OCR.

        Captures the entire display. If coordinates are given, crops the stored
        result to that region. Subsequent scan_screen/read_screen operate on this cache.

        Args:
            x1: Left edge in logical points (0 = no crop).
            y1: Top edge in logical points (0 = no crop).
            x2: Right edge in logical points (0 = no crop).
            y2: Bottom edge in logical points (0 = no crop).

        Returns:
            JSON with capture metadata (dimensions, path, timestamp).
        """
        result, metrics = _grab_screen_impl(x1, y1, x2, y2)
        return json.dumps({"result": result, "metrics": metrics})

    @mcp.tool()
    def scan_app(x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0):
        """View app capture, budget-downsized to fit MCP output.

        Returns a JPEG image of the last grab_app capture, sized for
        optimal Claude vision token cost (~1600 tokens at 1.15MP).
        If coordinates are given, shows only that region.
        Coords are relative to whatever was last grabbed.

        Args:
            x1: Left edge in logical points (0 = full image).
            y1: Top edge in logical points (0 = full image).
            x2: Right edge in logical points (0 = full image).
            y2: Bottom edge in logical points (0 = full image).

        Returns:
            Image content block + JSON metadata.
        """
        result, metrics = _scan_impl(APP_CACHE, x1, y1, x2, y2)
        return [
            Image(path=SCAN_TMP, format="jpeg"),
            json.dumps({"result": result, "metrics": metrics}),
        ]

    @mcp.tool()
    def scan_screen(x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0):
        """View screen capture, budget-downsized to fit MCP output.

        Returns a JPEG image of the last grab_screen capture, sized for
        optimal Claude vision token cost (~1600 tokens at 1.15MP).
        If coordinates are given, shows only that region.
        Coords are relative to whatever was last grabbed.

        Args:
            x1: Left edge in logical points (0 = full image).
            y1: Top edge in logical points (0 = full image).
            x2: Right edge in logical points (0 = full image).
            y2: Bottom edge in logical points (0 = full image).

        Returns:
            Image content block + JSON metadata.
        """
        result, metrics = _scan_impl(SCREEN_CACHE, x1, y1, x2, y2)
        return [
            Image(path=SCAN_TMP, format="jpeg"),
            json.dumps({"result": result, "metrics": metrics}),
        ]

    @mcp.tool()
    def read_app(x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0) -> str:
        """OCR text from app capture at full resolution.

        Extracts text from the last grab_app capture using macOS Vision.
        If coordinates are given, reads only that region.
        Coords are relative to whatever was last grabbed.

        Args:
            x1: Left edge in logical points (0 = full image).
            y1: Top edge in logical points (0 = full image).
            x2: Right edge in logical points (0 = full image).
            y2: Bottom edge in logical points (0 = full image).

        Returns:
            JSON with ocr_text and metrics. Zero image tokens consumed.
        """
        result, metrics = _read_impl(APP_CACHE, x1, y1, x2, y2)
        return json.dumps({"result": result, "metrics": metrics})

    @mcp.tool()
    def read_screen(x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0) -> str:
        """OCR text from screen capture at full resolution.

        Extracts text from the last grab_screen capture using macOS Vision.
        If coordinates are given, reads only that region.
        Coords are relative to whatever was last grabbed.

        Args:
            x1: Left edge in logical points (0 = full image).
            y1: Top edge in logical points (0 = full image).
            x2: Right edge in logical points (0 = full image).
            y2: Bottom edge in logical points (0 = full image).

        Returns:
            JSON with ocr_text and metrics. Zero image tokens consumed.
        """
        result, metrics = _read_impl(SCREEN_CACHE, x1, y1, x2, y2)
        return json.dumps({"result": result, "metrics": metrics})

    print("vshot MCP server running (stdio transport)", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
