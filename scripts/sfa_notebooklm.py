#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "fastmcp", "websocket-client"]
# ///
"""NotebookLM API client — CLI + MCP server.

Reverse-engineered Google NotebookLM batchexecute RPC API.
Cookie-based auth with CSRF token management.

Usage:
    sfa_notebooklm.py login [--port PORT] [--check]
    sfa_notebooklm.py notebooks list
    sfa_notebooklm.py notebooks create "Title"
    sfa_notebooklm.py notebooks get <notebook_id>
    sfa_notebooklm.py notebooks delete <notebook_id>
    sfa_notebooklm.py sources list <notebook_id>
    sfa_notebooklm.py sources add-url <notebook_id> <url>
    sfa_notebooklm.py sources add-text <notebook_id> <text>
    sfa_notebooklm.py query <notebook_id> "What is this about?"
    sfa_notebooklm.py mcp-stdio
"""

from __future__ import annotations

import argparse
import csv
import html as html_module
import io
import json
import os
import random
import re
import sys
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# =============================================================================
# LOGGING (TSV format — see SPEC_SFA.md Principle 6)
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
EXPOSED = [
    "login", "notebooks_list", "notebooks_create", "notebooks_get", "notebooks_delete",
    "sources_list", "sources_add_url", "sources_add_text", "sources_delete",
    "query", "create_audio", "create_video", "create_report"
]  # CLI + MCP — both interfaces

BASE_URL = "https://notebooklm.google.com"
BATCHEXECUTE_URL = f"{BASE_URL}/_/LabsTailwindUi/data/batchexecute"
QUERY_URL = f"{BASE_URL}/_/LabsTailwindUi/data/google.internal.labs.tailwind.orchestration.v1.LabsTailwindOrchestrationService/GenerateFreeFormStreamed"
UPLOAD_URL = "https://notebooklm.google.com/upload/_/"
BL_VERSION = "boq_labs-tailwind-frontend_20260108.06_p0"

CONFIG = {
    "request_timeout_seconds": 30.0,
    "max_notebooks": 100,
    "default_language": "en",
}

# RPC IDs
RPC_LIST_NOTEBOOKS = "wXbhsf"
RPC_GET_NOTEBOOK = "rLM1Ne"
RPC_CREATE_NOTEBOOK = "CCqFvf"
RPC_RENAME_NOTEBOOK = "s0tc2d"
RPC_DELETE_NOTEBOOK = "WWINqb"
RPC_ADD_SOURCE = "izAoDd"
RPC_GET_SOURCE = "hizoJc"
RPC_DELETE_SOURCE = "tGMBJ"
RPC_GET_SUMMARY = "VfAZjd"
RPC_CREATE_STUDIO = "R7cb6c"
RPC_POLL_STUDIO = "gArtLc"

# Studio types
STUDIO_TYPE_AUDIO = 1
STUDIO_TYPE_VIDEO = 3
STUDIO_TYPE_REPORT = 2

# Status codes
STATUS_MAP = {1: "in_progress", 3: "completed", 4: "failed"}
SOURCE_STATUS_READY = 2
SOURCE_STATUS_ERROR = 3

# Audio formats
AUDIO_FORMATS = {"deep_dive": 1, "brief": 2, "critique": 3, "debate": 4}
AUDIO_LENGTHS = {"short": 1, "default": 2, "long": 3}

# Video formats
VIDEO_FORMATS = {"explainer": 1, "brief": 2}
VIDEO_STYLES = {
    "auto_select": 1, "custom": 2, "classic": 3, "whiteboard": 4,
    "kawaii": 5, "anime": 6, "watercolor": 7, "retro_print": 8,
    "heritage": 9, "paper_craft": 10,
}

# Report formats
REPORT_FORMATS = {
    "Briefing Doc": ("Briefing Doc", "A comprehensive briefing document", "Write a detailed briefing document that covers all key points from the sources."),
    "Study Guide": ("Study Guide", "A study guide for learning", "Create a comprehensive study guide with key concepts, definitions, and review questions."),
    "Blog Post": ("Blog Post", "An engaging blog post", "Write an engaging and informative blog post based on the source material."),
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def _scrape_cookies_from_chrome(port: int = 9222, navigate: bool = True) -> dict:
    """Scrape Google cookies from Chrome via CDP.
    
    Opens NotebookLM in a new tab if needed, preserves user's current tab.
    
    CLI: login (internal use)
    MCP: — (internal use)
    """
    import websocket
    
    # Get list of pages
    response = httpx.get(f"http://localhost:{port}/json")
    targets = response.json()
    pages = [t for t in targets if t["type"] == "page"]
    
    if not pages:
        raise RuntimeError(f"No pages found on Chrome CDP port {port}")
    
    # Remember the current active tab (first in list is usually active)
    original_page_id = pages[0].get("id")
    original_url = pages[0].get("url", "")
    _log("DEBUG", "cookie_scrape", f"Current tab: {original_url[:60]}...")
    
    ws_url = None
    temp_tab_id = None
    
    try:
        if navigate:
            # Open NotebookLM in a new tab
            _log("INFO", "cookie_scrape", "Opening NotebookLM in new tab")
            try:
                httpx.put(f"http://localhost:{port}/json/new?https://notebooklm.google.com")
                # Wait a moment for page to load and cookies to be set
                time.sleep(2.0)
            except Exception as e:
                _log("WARN", "cookie_scrape", f"Failed to create new tab: {e}")
            
            # Refresh page list to get the new tab
            response = httpx.get(f"http://localhost:{port}/json")
            targets = response.json()
            pages = [t for t in targets if t["type"] == "page"]
            
            # Find the NotebookLM tab (should be the one with notebooklm.google.com URL)
            nlm_page = None
            for page in pages:
                if "notebooklm.google.com" in page.get("url", ""):
                    nlm_page = page
                    temp_tab_id = page.get("id")
                    break
            
            # Use NotebookLM tab if found, otherwise use the first available
            target_page = nlm_page if nlm_page else pages[0]
            ws_url = target_page.get("webSocketDebuggerUrl")
        else:
            # Use existing page
            ws_url = pages[0].get("webSocketDebuggerUrl")
        
        if not ws_url:
            raise RuntimeError("No WebSocket URL found")
        
        ws = websocket.create_connection(ws_url)
        try:
            # Enable Network domain
            ws.send(json.dumps({"id": 1, "method": "Network.enable", "params": {}}))
            ws.recv()  # Ack
            
            # Get all cookies
            ws.send(json.dumps({"id": 2, "method": "Network.getCookies", "params": {}}))
            response = json.loads(ws.recv())
            
            if "error" in response:
                raise RuntimeError(f"CDP Error: {response['error']['message']}")
            
            cookies = response.get("result", {}).get("cookies", [])
            
            _log("DEBUG", "cookie_scrape", f"Total cookies: {len(cookies)}")
            domains = sorted(set(c.get("domain", "") for c in cookies))
            _log("INFO", "cookie_domains", f"Found {len(domains)} unique domains", detail=f"domains={domains[:30]}")
            
            # Filter for Google cookies (including notebooklm.google.com)
            google_domains = [".google.com", ".googleusercontent.com", "notebooklm.google.com", ".google.co.uk", "accounts.google.com", "myaccount.google.com"]
            google_cookies = [
                c for c in cookies 
                if any(domain in c.get("domain", "") for domain in google_domains)
            ]
            
            _log("INFO", "cookie_scrape", f"Found {len(google_cookies)} Google cookies")
            
            return {
                "cookies": google_cookies,
                "csrf_token": "",
                "session_id": "",
            }
        finally:
            ws.close()
    
    finally:
        # Cleanup: close the temporary NotebookLM tab if we created one
        if temp_tab_id and temp_tab_id != original_page_id:
            try:
                _log("DEBUG", "cookie_scrape", f"Closing temporary NotebookLM tab")
                httpx.get(f"http://localhost:{port}/json/close/{temp_tab_id}")
            except Exception as e:
                _log("WARN", "cookie_scrape", f"Failed to close temp tab: {e}")
        
        # Note: We don't switch back to original tab via CDP as it can be disruptive
        # The user's browser state should remain largely unchanged


def _load_auth_impl() -> dict:
    """Load auth tokens from disk.
    
    CLI: login (internal use)
    MCP: — (internal use)
    """
    config_dir = Path.home() / ".notebooklm-mcp-cli"
    
    # Try profile-based auth first
    profile_path = config_dir / "profiles" / "default" / "cookies.json"
    if profile_path.exists():
        data = json.loads(profile_path.read_text())
        return {
            "cookies": data.get("cookies", {}),
            "csrf_token": data.get("csrf_token", ""),
            "session_id": data.get("session_id", ""),
        }
    
    # Legacy auth.json
    legacy_path = config_dir / "auth.json"
    if legacy_path.exists():
        data = json.loads(legacy_path.read_text())
        return {
            "cookies": data.get("cookies", {}),
            "csrf_token": data.get("csrf_token", ""),
            "session_id": data.get("session_id", ""),
        }
    
    raise RuntimeError("No auth found. Run 'sfa_notebooklm.py login' to authenticate.")


def _save_csrf_impl(csrf_token: str, session_id: str) -> None:
    """Update cached CSRF token on disk."""
    config_dir = Path.home() / ".notebooklm-mcp-cli"
    profile_path = config_dir / "profiles" / "default" / "cookies.json"
    legacy_path = config_dir / "auth.json"
    
    for path in [profile_path, legacy_path]:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                data["csrf_token"] = csrf_token
                data["session_id"] = session_id
                path.write_text(json.dumps(data, indent=2))
                return
            except Exception:
                pass


class NotebookLMClient:
    """NotebookLM API client — sync, single-class."""
    
    def __init__(self, cookies: dict[str, str], csrf_token: str = "", session_id: str = ""):
        self.cookies = cookies
        self.csrf_token = csrf_token
        self._session_id = session_id
        self._client: httpx.Client | None = None
        self._reqid_counter = random.randint(100000, 999999)
        self._csrf_refreshed_at: float = 0.0
        if not self.csrf_token or self._is_csrf_stale():
            self._refresh_auth_tokens_impl()
    
    def _is_csrf_stale(self) -> bool:
        """Check if CSRF token is older than 1 hour."""
        if not self.csrf_token:
            return True
        if ":" in self.csrf_token:
            try:
                ts_ms = int(self.csrf_token.rsplit(":", 1)[1])
                age_secs = time.time() - (ts_ms / 1000)
                if age_secs > 3600:
                    return True
            except (ValueError, IndexError):
                pass
        return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, *exc):
        self.close()
    
    def close(self):
        if self._client:
            self._client.close()
            self._client = None
    
    def _get_httpx_cookies(self) -> httpx.Cookies:
        cookies = httpx.Cookies()
        if isinstance(self.cookies, list):
            for c in self.cookies:
                # Handle both CDP format (name, value, domain, path) and simple dict
                if isinstance(c, dict):
                    name = c.get("name")
                    value = c.get("value")
                    domain = c.get("domain", ".google.com")
                    path = c.get("path", "/")
                    if name and value:
                        cookies.set(name, value, domain=domain, path=path)
                        if ".google.com" in domain:
                            cookies.set(name, value, domain=".googleusercontent.com", path=path)
                else:
                    # Simple [name, value] format
                    name, value = c[0], c[1]
                    cookies.set(name, value, domain=".google.com")
                    cookies.set(name, value, domain=".googleusercontent.com")
        elif isinstance(self.cookies, dict):
            for name, value in self.cookies.items():
                cookies.set(name, value, domain=".google.com")
                cookies.set(name, value, domain=".googleusercontent.com")
        return cookies
    
    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                cookies=self._get_httpx_cookies(),
                headers={
                    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                    "Origin": BASE_URL,
                    "Referer": f"{BASE_URL}/",
                    "X-Same-Domain": "1",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                },
                timeout=30.0,
            )
            if self.csrf_token:
                self._client.headers["X-Goog-Csrf-Token"] = self.csrf_token
        return self._client
    
    def _build_request_body(self, rpc_id: str, params: Any) -> str:
        params_json = json.dumps(params, separators=(",", ":"))
        f_req = [[[rpc_id, params_json, None, "generic"]]]
        f_req_json = json.dumps(f_req, separators=(",", ":"))
        body_parts = [f"f.req={urllib.parse.quote(f_req_json, safe='')}"]
        if self.csrf_token:
            body_parts.append(f"at={urllib.parse.quote(self.csrf_token, safe='')}")
        return "&".join(body_parts) + "&"
    
    def _build_url(self, rpc_id: str, source_path: str = "/") -> str:
        params = {
            "rpcids": rpc_id,
            "source-path": source_path,
            "bl": os.environ.get("NOTEBOOKLM_BL", BL_VERSION),
            "hl": "en",
            "rt": "c",
        }
        if self._session_id:
            params["f.sid"] = self._session_id
        return f"{BATCHEXECUTE_URL}?{urllib.parse.urlencode(params)}"
    
    def _parse_response(self, text: str) -> list:
        if text.startswith(")]}'"):
            text = text[4:]
        lines = text.strip().split("\n")
        results = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            try:
                int(line)
                i += 1
                if i < len(lines):
                    try:
                        results.append(json.loads(lines[i]))
                    except json.JSONDecodeError:
                        pass
                i += 1
            except ValueError:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
                i += 1
        return results
    
    def _extract_rpc_result(self, parsed: list, rpc_id: str) -> Any:
        AUTH_ERROR_CODES = {16}
        for chunk in parsed:
            if isinstance(chunk, list):
                for item in chunk:
                    if isinstance(item, list) and len(item) >= 3:
                        if item[0] == "wrb.fr" and item[1] == rpc_id:
                            if len(item) > 6 and item[6] == "generic" and isinstance(item[5], list):
                                err_codes = set(item[5]) & AUTH_ERROR_CODES
                                if err_codes:
                                    raise RuntimeError(f"RPC auth error (codes: {err_codes}). CSRF likely expired.")
                            result_str = item[2]
                            if isinstance(result_str, str):
                                try:
                                    return json.loads(result_str)
                                except json.JSONDecodeError:
                                    return result_str
                            return result_str
        return None

    def _extract_answer_from_chunk(self, json_str: str) -> tuple[str | None, bool]:
        """Extract answer text and type from streaming response chunk.
        
        Returns:
            Tuple of (answer_text, is_final_answer)
            Type 1 = final answer, Type 2 = thinking step
        """
        try:
            data = json.loads(json_str)
            if not isinstance(data, list) or len(data) < 1:
                return None, False
            answer_text = data[0][0] if isinstance(data[0], list) and len(data[0]) > 0 else None
            is_answer = False
            if isinstance(data[0], list) and len(data[0]) > 4 and isinstance(data[0][4], list):
                type_indicator = data[0][4][-1] if data[0][4] else None
                is_answer = type_indicator == 1
            return answer_text, is_answer
        except (json.JSONDecodeError, IndexError, TypeError):
            return None, False
    
    def _parse_query_response(self, response_text: str) -> str:
        """Parse streaming query response and extract answer.
        
        Returns:
            The answer text (longest chunk where type == 1, or longest overall)
        """
        if response_text.startswith(")]}'"):
            response_text = response_text[4:]
        
        lines = response_text.strip().split("\n")
        chunks = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Check if this is a length line
            try:
                int(line)
                i += 1
                if i < len(lines):
                    # Next line should be JSON
                    chunks.append(lines[i])
                i += 1
            except ValueError:
                # Not a length line, try to parse as JSON
                chunks.append(line)
                i += 1
        
        # Extract answers from all chunks
        answer_chunks = []
        thinking_chunks = []
        
        for chunk_text in chunks:
            try:
                chunk_data = json.loads(chunk_text)
                if isinstance(chunk_data, list):
                    for item in chunk_data:
                        if isinstance(item, list) and len(item) >= 3 and item[0] == "wrb.fr":
                            inner_json = item[2]
                            if isinstance(inner_json, str):
                                answer_text, is_answer = self._extract_answer_from_chunk(inner_json)
                                if answer_text:
                                    if is_answer:
                                        answer_chunks.append(answer_text)
                                    else:
                                        thinking_chunks.append(answer_text)
            except (json.JSONDecodeError, IndexError, TypeError):
                pass
        
        # Return longest answer chunk, or longest thinking chunk if no answers
        if answer_chunks:
            return max(answer_chunks, key=len)
        elif thinking_chunks:
            return max(thinking_chunks, key=len)
        else:
            raise RuntimeError("No answer extracted from response")

    
    def _call_rpc(self, rpc_id: str, params: Any, path: str = "/",
                  timeout: float | None = None, _retry: bool = False) -> Any:
        client = self._get_client()
        body = self._build_request_body(rpc_id, params)
        url = self._build_url(rpc_id, path)
        _log("DEBUG", "rpc_call", f"{rpc_id} -> {path}")
        try:
            kw = {"content": body}
            if timeout:
                kw["timeout"] = timeout
            response = client.post(url, **kw)
            response.raise_for_status()
            parsed = self._parse_response(response.text)
            return self._extract_rpc_result(parsed, rpc_id)
        except (httpx.HTTPStatusError, RuntimeError) as e:
            if _retry:
                raise
            is_auth = isinstance(e, RuntimeError) or (
                isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (400, 401, 403)
            )
            if is_auth:
                _log("WARN", "auth_refresh", "Refreshing auth tokens")
                try:
                    self._refresh_auth_tokens_impl()
                    self._client = None
                    return self._call_rpc(rpc_id, params, path, timeout, _retry=True)
                except Exception:
                    try:
                        auth = _load_auth_impl()
                        self.cookies = auth["cookies"]
                        self.csrf_token = ""
                        self._session_id = ""
                        self._client = None
                        self._refresh_auth_tokens_impl()
                        self._client = None
                        return self._call_rpc(rpc_id, params, path, timeout, _retry=True)
                    except Exception:
                        raise RuntimeError("Authentication expired. Run 'sfa_notebooklm.py login'.") from None
            raise
    
    def _refresh_auth_tokens_impl(self) -> None:
        """Refresh CSRF token from NotebookLM page."""
        cookies = self._get_httpx_cookies()
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        with httpx.Client(cookies=cookies, headers=headers, follow_redirects=True, timeout=15.0) as client:
            response = client.get(f"{BASE_URL}/")
            if "accounts.google.com" in str(response.url):
                raise ValueError("Auth expired — redirected to login. Run 'sfa_notebooklm.py login'.")
            if response.status_code != 200:
                raise ValueError(f"Failed to fetch NotebookLM page: HTTP {response.status_code}")
            html = response.text
            csrf_match = re.search(r'"SNlM0e":"([^"]+)"', html)
            if not csrf_match:
                raise ValueError("Could not extract CSRF token from page.")
            self.csrf_token = csrf_match.group(1)
            sid_match = re.search(r'"FdrFJe":"([^"]+)"', html)
            if sid_match:
                self._session_id = sid_match.group(1)
            self._csrf_refreshed_at = time.time()
            _save_csrf_impl(self.csrf_token, self._session_id)
            _log("DEBUG", "csrf_refresh", f"Token refreshed: {self.csrf_token[:20]}...")
    
    # === Notebooks ===
    
    def list_notebooks_impl(self) -> tuple[list[dict], dict]:
        """List all notebooks.
        
        CLI: notebooks list
        MCP: notebooks_list
        
        Returns:
            Tuple of (notebooks list, metrics dict)
        """
        start_ms = time.time() * 1000
        params = [None, 1, None, [2]]
        result = self._call_rpc(RPC_LIST_NOTEBOOKS, params)
        notebooks = []
        if not result or not isinstance(result, list):
            latency = round(time.time() * 1000 - start_ms, 2)
            return [], {"status": "success", "latency_ms": latency, "count": 0}
        
        nb_list = result[0] if result and isinstance(result[0], list) else result
        for nb in nb_list:
            if not isinstance(nb, list) or len(nb) < 3:
                continue
            title = nb[0] if isinstance(nb[0], str) else "Untitled"
            sources_data = nb[1] if len(nb) > 1 and isinstance(nb[1], list) else []
            notebook_id = nb[2] if len(nb) > 2 else None
            is_owned, is_shared = True, False
            created_at = modified_at = None
            if len(nb) > 5 and isinstance(nb[5], list) and len(nb[5]) > 0:
                meta = nb[5]
                is_owned = meta[0] == 1
                is_shared = bool(meta[1]) if len(meta) > 1 else False
            
            sources = []
            for src in sources_data:
                if isinstance(src, list) and len(src) >= 2:
                    sid = src[0][0] if isinstance(src[0], list) and src[0] else src[0]
                    sources.append({"id": sid, "title": src[1] if len(src) > 1 else "Untitled"})
            
            if notebook_id:
                notebooks.append({
                    "id": notebook_id, "title": title,
                    "source_count": len(sources), "sources": sources,
                    "is_owned": is_owned, "is_shared": is_shared,
                    "created_at": created_at, "modified_at": modified_at,
                    "url": f"https://notebooklm.google.com/notebook/{notebook_id}",
                })
        
        latency = round(time.time() * 1000 - start_ms, 2)
        metrics = {"status": "success", "latency_ms": latency, "count": len(notebooks)}
        return notebooks, metrics
    
    def get_notebook_impl(self, notebook_id: str) -> tuple[dict, dict]:
        """Get notebook details.
        
        CLI: notebooks get
        MCP: notebooks_get
        """
        start_ms = time.time() * 1000
        result = self._call_rpc(RPC_GET_NOTEBOOK, [notebook_id, None, [2], None, 0], f"/notebook/{notebook_id}")
        latency = round(time.time() * 1000 - start_ms, 2)
        metrics = {"status": "success", "latency_ms": latency}
        return {"id": notebook_id, "data": result}, metrics
    
    def create_notebook_impl(self, title: str = "") -> tuple[dict, dict]:
        """Create a new notebook.
        
        CLI: notebooks create
        MCP: notebooks_create
        """
        start_ms = time.time() * 1000
        params = [title, None, None, [2], [1, None, None, None, None, None, None, None, None, None, [1]]]
        result = self._call_rpc(RPC_CREATE_NOTEBOOK, params)
        latency = round(time.time() * 1000 - start_ms, 2)
        metrics = {"status": "success", "latency_ms": latency}
        if result and isinstance(result, list) and len(result) >= 3 and result[2]:
            return {"id": result[2], "title": title or "Untitled notebook"}, metrics
        return {"id": None, "title": title}, metrics
    
    def delete_notebook_impl(self, notebook_id: str) -> tuple[bool, dict]:
        """Delete a notebook.
        
        CLI: notebooks delete
        MCP: notebooks_delete
        """
        start_ms = time.time() * 1000
        result = self._call_rpc(RPC_DELETE_NOTEBOOK, [[notebook_id], [2]])
        latency = round(time.time() * 1000 - start_ms, 2)
        metrics = {"status": "success", "latency_ms": latency}
        return result is not None, metrics
    
    # === Sources ===
    
    def list_sources_impl(self, notebook_id: str) -> tuple[list[dict], dict]:
        """List all sources in a notebook.
        
        CLI: sources list
        MCP: sources_list
        """
        start_ms = time.time() * 1000
        result = self._call_rpc(RPC_GET_NOTEBOOK, [notebook_id, None, [2], None, 0], f"/notebook/{notebook_id}")
        sources = []
        if not result or not isinstance(result, list) or len(result) < 1:
            latency = round(time.time() * 1000 - start_ms, 2)
            return [], {"status": "success", "latency_ms": latency, "count": 0}
        
        nb_data = result[0] if isinstance(result[0], list) else result
        sources_data = nb_data[1] if len(nb_data) > 1 and isinstance(nb_data[1], list) else []
        
        for src in sources_data:
            if not isinstance(src, list) or len(src) < 3:
                continue
            source_id = src[0][0] if src[0] and isinstance(src[0], list) else None
            title = src[1] if len(src) > 1 else "Untitled"
            meta = src[2] if len(src) > 2 and isinstance(src[2], list) else []
            source_type = meta[4] if isinstance(meta, list) and len(meta) > 4 else None
            url = None
            if isinstance(meta, list) and len(meta) > 7 and isinstance(meta[7], list) and len(meta[7]) > 0:
                url = meta[7][0]
            
            status = SOURCE_STATUS_READY
            if len(src) > 3 and isinstance(src[3], list) and len(src[3]) > 1 and isinstance(src[3][1], int):
                status = src[3][1]
            
            sources.append({
                "id": source_id, "title": title,
                "source_type": source_type,
                "url": url, "status": status,
            })
        
        latency = round(time.time() * 1000 - start_ms, 2)
        metrics = {"status": "success", "latency_ms": latency, "count": len(sources)}
        return sources, metrics
    
    def add_url_source_impl(self, notebook_id: str, url: str) -> tuple[dict, dict]:
        """Add a URL source to a notebook.
        
        CLI: sources add-url
        MCP: sources_add_url
        """
        start_ms = time.time() * 1000
        is_youtube = "youtube.com" in url.lower() or "youtu.be" in url.lower()
        if is_youtube:
            source_data = [None, None, None, None, None, None, None, [url], None, None, 1]
        else:
            source_data = [None, None, [url], None, None, None, None, None, None, None, 1]
        params = [[source_data], notebook_id, [2], [1, None, None, None, None, None, None, None, None, None, [1]]]
        result = self._call_rpc(RPC_ADD_SOURCE, params, f"/notebook/{notebook_id}", timeout=120.0)
        
        source_result = None
        if result and isinstance(result, list) and len(result) > 0:
            sl = result[0] if result else []
            if sl and len(sl) > 0:
                sd = sl[0]
                sid = sd[0][0] if sd[0] else None
                source_result = {"id": sid, "title": sd[1] if len(sd) > 1 else "Untitled"}
        
        latency = round(time.time() * 1000 - start_ms, 2)
        metrics = {"status": "success", "latency_ms": latency}
        return source_result or {"id": None, "title": None}, metrics
    
    def add_text_source_impl(self, notebook_id: str, text: str, title: str = "Pasted Text") -> tuple[dict, dict]:
        """Add a text source to a notebook.
        
        CLI: sources add-text
        MCP: sources_add_text
        """
        start_ms = time.time() * 1000
        source_data = [None, [title, text], None, 2, None, None, None, None, None, None, 1]
        params = [[source_data], notebook_id, [2], [1, None, None, None, None, None, None, None, None, None, [1]]]
        result = self._call_rpc(RPC_ADD_SOURCE, params, f"/notebook/{notebook_id}", timeout=120.0)
        
        source_result = None
        if result and isinstance(result, list) and len(result) > 0:
            sl = result[0] if result else []
            if sl and len(sl) > 0:
                sd = sl[0]
                sid = sd[0][0] if sd[0] else None
                source_result = {"id": sid, "title": sd[1] if len(sd) > 1 else title}
        
        latency = round(time.time() * 1000 - start_ms, 2)
        metrics = {"status": "success", "latency_ms": latency}
        return source_result or {"id": None, "title": title}, metrics
    
    def delete_source_impl(self, source_id: str) -> tuple[bool, dict]:
        """Delete a source.
        
        CLI: sources delete
        MCP: sources_delete
        """
        start_ms = time.time() * 1000
        result = self._call_rpc(RPC_DELETE_SOURCE, [[[source_id]], [2]])
        latency = round(time.time() * 1000 - start_ms, 2)
        metrics = {"status": "success", "latency_ms": latency}
        return result is not None, metrics
    
    # === Query ===
    
    def query_impl(self, notebook_id: str, question: str) -> tuple[dict, dict]:
        """Query a notebook.
        
        CLI: query
        MCP: query
        """
        start_ms = time.time() * 1000
        
        # Get source IDs
        sources, _ = self.list_sources_impl(notebook_id)
        source_ids = [s["id"] for s in sources if s.get("id")]
        
        assert source_ids, "No sources found in notebook"
        
        # Build streaming request
        sources_array = [[[sid]] for sid in source_ids]
        params = [
            sources_array,
            question,
            None,  # conversation history
            [2, None, [1]],  # fixed config
            str(uuid.uuid4()),  # conversation ID
        ]
        
        params_json = json.dumps(params, separators=(",", ":"))
        f_req = [None, params_json]
        f_req_json = json.dumps(f_req, separators=(",", ":"))
        
        body_parts = [f"f.req={urllib.parse.quote(f_req_json, safe='')}"]
        if self.csrf_token:
            body_parts.append(f"at={urllib.parse.quote(self.csrf_token, safe='')}")
        body = "&".join(body_parts) + "&"
        
        self._reqid_counter += 100000
        url_params = {
            "bl": os.environ.get("NOTEBOOKLM_BL", BL_VERSION),
            "hl": "en",
            "_reqid": str(self._reqid_counter),
            "rt": "c",
        }
        if self._session_id:
            url_params["f.sid"] = self._session_id
        url = f"{QUERY_URL}?{urllib.parse.urlencode(url_params)}"
        
        _log("DEBUG", "query_call", f"Querying notebook {notebook_id}")
        
        try:
            client = self._get_client()
            response = client.post(url, content=body, timeout=120.0)
            response.raise_for_status()
            answer_text = self._parse_query_response(response.text)
            assert answer_text, "No answer extracted from response"
            
            latency = round(time.time() * 1000 - start_ms, 2)
            metrics = {"status": "success", "latency_ms": latency, "source_count": len(source_ids)}
            return {"answer": answer_text, "question": question, "notebook_id": notebook_id}, metrics
            
        except (httpx.HTTPStatusError, RuntimeError) as e:
            is_auth = isinstance(e, RuntimeError) or (
                isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (400, 401, 403)
            )
            if is_auth:
                _log("WARN", "auth_refresh", "Refreshing auth tokens for query")
                try:
                    self._refresh_auth_tokens_impl()
                    self._client = None
                    # Retry once
                    client = self._get_client()
                    response = client.post(url, content=body, timeout=120.0)
                    response.raise_for_status()
                    answer_text = self._parse_query_response(response.text)
                    assert answer_text, "No answer extracted from response"
                    
                    latency = round(time.time() * 1000 - start_ms, 2)
                    metrics = {"status": "success", "latency_ms": latency, "source_count": len(source_ids)}
                    return {"answer": answer_text, "question": question, "notebook_id": notebook_id}, metrics
                except Exception:
                    raise RuntimeError("Authentication expired. Run 'sfa_notebooklm.py login'.") from None
            raise

    # === Audio Overview ===
    
    def create_audio_overview_impl(self, notebook_id: str, format_code: int = 1, 
                                    length_code: int = 2, language: str = "en") -> tuple[dict, dict]:
        """Create an audio overview.
        
        CLI: create-audio
        MCP: create_audio
        """
        start_ms = time.time() * 1000
        sources, _ = self.list_sources_impl(notebook_id)
        source_ids = [s["id"] for s in sources if s.get("id")]
        
        if not source_ids:
            latency = round(time.time() * 1000 - start_ms, 2)
            return {"error": "No sources in notebook"}, {"status": "error", "latency_ms": latency}
        
        sources_nested = [[[sid]] for sid in source_ids]
        sources_simple = [[sid] for sid in source_ids]
        audio_options = [None, ["", length_code, None, sources_simple, language, None, format_code]]
        params = [[2], notebook_id, [None, None, STUDIO_TYPE_AUDIO, sources_nested, None, None, audio_options]]
        body = self._build_request_body(RPC_CREATE_STUDIO, params)
        url = self._build_url(RPC_CREATE_STUDIO, f"/notebook/{notebook_id}")
        resp = self._get_client().post(url, content=body)
        resp.raise_for_status()
        parsed = self._parse_response(resp.text)
        result = self._extract_rpc_result(parsed, RPC_CREATE_STUDIO)
        
        latency = round(time.time() * 1000 - start_ms, 2)
        metrics = {"status": "success", "latency_ms": latency}
        
        if result and isinstance(result, list) and len(result) > 0:
            ad = result[0]
            aid = ad[0] if isinstance(ad, list) and len(ad) > 0 else None
            return {"artifact_id": aid, "notebook_id": notebook_id, "type": "audio"}, metrics
        return {"artifact_id": None, "notebook_id": notebook_id, "type": "audio"}, metrics
    
    # === Video Overview ===
    
    def create_video_overview_impl(self, notebook_id: str, format_code: int = 1,
                                    visual_style_code: int = 1, language: str = "en") -> tuple[dict, dict]:
        """Create a video overview.
        
        CLI: create-video
        MCP: create_video
        """
        start_ms = time.time() * 1000
        sources, _ = self.list_sources_impl(notebook_id)
        source_ids = [s["id"] for s in sources if s.get("id")]
        
        if not source_ids:
            latency = round(time.time() * 1000 - start_ms, 2)
            return {"error": "No sources in notebook"}, {"status": "error", "latency_ms": latency}
        
        sources_nested = [[[sid]] for sid in source_ids]
        sources_simple = [[sid] for sid in source_ids]
        video_options = [None, None, [sources_simple, language, "", None, format_code, visual_style_code]]
        params = [[2], notebook_id, [None, None, STUDIO_TYPE_VIDEO, sources_nested, None, None, None, None, video_options]]
        body = self._build_request_body(RPC_CREATE_STUDIO, params)
        url = self._build_url(RPC_CREATE_STUDIO, f"/notebook/{notebook_id}")
        resp = self._get_client().post(url, content=body)
        resp.raise_for_status()
        parsed = self._parse_response(resp.text)
        result = self._extract_rpc_result(parsed, RPC_CREATE_STUDIO)
        
        latency = round(time.time() * 1000 - start_ms, 2)
        metrics = {"status": "success", "latency_ms": latency}
        
        if result and isinstance(result, list) and len(result) > 0:
            ad = result[0]
            aid = ad[0] if isinstance(ad, list) and len(ad) > 0 else None
            return {"artifact_id": aid, "notebook_id": notebook_id, "type": "video"}, metrics
        return {"artifact_id": None, "notebook_id": notebook_id, "type": "video"}, metrics
    
    # === Report ===
    
    def create_report_impl(self, notebook_id: str, report_format: str = "Briefing Doc",
                           language: str = "en") -> tuple[dict, dict]:
        """Create a report.
        
        CLI: create-report
        MCP: create_report
        """
        start_ms = time.time() * 1000
        sources, _ = self.list_sources_impl(notebook_id)
        source_ids = [s["id"] for s in sources if s.get("id")]
        
        if not source_ids:
            latency = round(time.time() * 1000 - start_ms, 2)
            return {"error": "No sources in notebook"}, {"status": "error", "latency_ms": latency}
        
        # Get report format details
        title, description, prompt = REPORT_FORMATS.get(
            report_format,
            (report_format, "", f"Write a {report_format} based on the source material.")
        )
        
        # Build params
        source_ids_triple = [[[sid]] for sid in source_ids]
        source_ids_double = [[sid] for sid in source_ids]
        
        params = [
            [2],
            notebook_id,
            [
                None,  # [0]
                None,  # [1]
                STUDIO_TYPE_REPORT,  # [2]: type code = 2
                source_ids_triple,   # [3]
                None,  # [4]
                None,  # [5]
                None,  # [6]
                [
                    None,
                    [
                        title,
                        description,
                        None,
                        source_ids_double,
                        language,
                        prompt,
                        None,
                        True,
                    ],
                ],  # [7]
            ],
        ]
        
        result = self._call_rpc(RPC_CREATE_STUDIO, params, f"/notebook/{notebook_id}")
        
        latency = round(time.time() * 1000 - start_ms, 2)
        metrics = {"status": "success", "latency_ms": latency}
        
        if result and isinstance(result, list) and len(result) > 0:
            ad = result[0]
            aid = ad[0] if isinstance(ad, list) and len(ad) > 0 else None
            return {"artifact_id": aid, "notebook_id": notebook_id, "type": "report", "format": report_format}, metrics
        return {"artifact_id": None, "notebook_id": notebook_id, "type": "report", "format": report_format}, metrics



# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NotebookLM API client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 1.0.0")
    sub = parser.add_subparsers(dest="command")
    
    # CLI for login
    p_login = sub.add_parser("login", help="Authenticate with NotebookLM")
    p_login.add_argument("-p", "--port", type=int, default=9222, help="Chrome CDP port")
    p_login.add_argument("-c", "--check", action="store_true", help="Validate existing auth")
    
    # CLI for notebooks_list
    p_nb_list = sub.add_parser("notebooks_list", help="List all notebooks")
    
    # CLI for notebooks_get
    p_nb_get = sub.add_parser("notebooks_get", help="Get notebook details")
    p_nb_get.add_argument("notebook_id", help="Notebook ID")
    
    # CLI for notebooks_create
    p_nb_create = sub.add_parser("notebooks_create", help="Create a new notebook")
    p_nb_create.add_argument("title", nargs="?", default="", help="Notebook title")
    
    # CLI for notebooks_delete
    p_nb_delete = sub.add_parser("notebooks_delete", help="Delete a notebook")
    p_nb_delete.add_argument("notebook_id", help="Notebook ID")
    
    # CLI for sources_list
    p_src_list = sub.add_parser("sources_list", help="List sources in a notebook")
    p_src_list.add_argument("notebook_id", help="Notebook ID")
    
    # CLI for sources_add_url
    p_src_url = sub.add_parser("sources_add_url", help="Add URL source")
    p_src_url.add_argument("notebook_id", help="Notebook ID")
    p_src_url.add_argument("url", help="URL to add")
    
    # CLI for sources_add_text
    p_src_text = sub.add_parser("sources_add_text", help="Add text source")
    p_src_text.add_argument("notebook_id", help="Notebook ID")
    p_src_text.add_argument("text", help="Text content")
    p_src_text.add_argument("-t", "--title", default="Pasted Text", help="Source title")
    
    # CLI for sources_delete
    p_src_delete = sub.add_parser("sources_delete", help="Delete a source")
    p_src_delete.add_argument("source_id", help="Source ID")
    
    # CLI for query
    p_query = sub.add_parser("query", help="Query a notebook")
    p_query.add_argument("notebook_id", help="Notebook ID")
    p_query.add_argument("question", nargs="?", help="Question to ask")
    
    # CLI for create_audio
    p_audio = sub.add_parser("create_audio", help="Create audio overview")
    p_audio.add_argument("notebook_id", help="Notebook ID")
    p_audio.add_argument("-f", "--format", default="deep_dive", choices=list(AUDIO_FORMATS.keys()))
    p_audio.add_argument("-l", "--length", default="default", choices=list(AUDIO_LENGTHS.keys()))
    p_audio.add_argument("-L", "--language", default="en", help="Language code")
    
    # CLI for create_video
    p_video = sub.add_parser("create_video", help="Create video overview")
    p_video.add_argument("notebook_id", help="Notebook ID")
    p_video.add_argument("-f", "--format", default="explainer", choices=list(VIDEO_FORMATS.keys()))
    p_video.add_argument("-s", "--style", default="auto_select", choices=list(VIDEO_STYLES.keys()))
    p_video.add_argument("-L", "--language", default="en", help="Language code")
    
    # CLI for create_report
    p_report = sub.add_parser("create_report", help="Create a report")
    p_report.add_argument("notebook_id", help="Notebook ID")
    p_report.add_argument("-f", "--format", default="Briefing Doc", help="Report format")
    p_report.add_argument("-L", "--language", default="en", help="Language code")
    
    # CLI for mcp-stdio
    sub.add_parser("mcp-stdio", help="Run as MCP server")
    
    args = parser.parse_args()
    
    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "login":
            if args.check:
                try:
                    auth = _load_auth_impl()
                    print(json.dumps({"authenticated": True, "csrf_token_prefix": auth["csrf_token"][:20]}, indent=2))
                except Exception as e:
                    print(json.dumps({"authenticated": False, "error": str(e)}, indent=2))
            else:
                try:
                    # Scrape cookies from Chrome
                    auth_data = _scrape_cookies_from_chrome(args.port)
                    
                    # Save to profile
                    config_dir = Path.home() / ".notebooklm-mcp-cli" / "profiles" / "default"
                    config_dir.mkdir(parents=True, exist_ok=True)
                    profile_path = config_dir / "cookies.json"
                    profile_path.write_text(json.dumps(auth_data, indent=2))
                    
                    print(json.dumps({
                        "authenticated": True,
                        "port": args.port,
                        "cookies_saved": len(auth_data["cookies"]),
                        "profile_path": str(profile_path),
                    }, indent=2))
                    
                    # Now refresh CSRF token
                    client = NotebookLMClient(auth_data["cookies"])
                    client.close()
                    
                    print(json.dumps({"status": "Login successful. CSRF token refreshed."}, indent=2))
                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)
                    sys.exit(1)
        elif args.command == "notebooks_list":
            auth = _load_auth_impl()
            client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
            result, metrics = client.list_notebooks_impl()
            print(json.dumps({"notebooks": result, "metrics": metrics}, indent=2))
            client.close()
        elif args.command == "notebooks_get":
            auth = _load_auth_impl()
            client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
            result, metrics = client.get_notebook_impl(args.notebook_id)
            print(json.dumps({"notebook": result, "metrics": metrics}, indent=2))
            client.close()
        elif args.command == "notebooks_create":
            auth = _load_auth_impl()
            client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
            # Check stdin for title if not provided
            title = args.title
            if not title and not sys.stdin.isatty():
                title = sys.stdin.read().strip()
            result, metrics = client.create_notebook_impl(title)
            print(json.dumps({"notebook": result, "metrics": metrics}, indent=2))
            client.close()
        elif args.command == "notebooks_delete":
            auth = _load_auth_impl()
            client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
            result, metrics = client.delete_notebook_impl(args.notebook_id)
            print(json.dumps({"deleted": result, "metrics": metrics}, indent=2))
            client.close()
        elif args.command == "sources_list":
            auth = _load_auth_impl()
            client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
            result, metrics = client.list_sources_impl(args.notebook_id)
            print(json.dumps({"sources": result, "metrics": metrics}, indent=2))
            client.close()
        elif args.command == "sources_add_url":
            auth = _load_auth_impl()
            client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
            result, metrics = client.add_url_source_impl(args.notebook_id, args.url)
            print(json.dumps({"source": result, "metrics": metrics}, indent=2))
            client.close()
        elif args.command == "sources_add_text":
            auth = _load_auth_impl()
            client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
            result, metrics = client.add_text_source_impl(args.notebook_id, args.text, args.title)
            print(json.dumps({"source": result, "metrics": metrics}, indent=2))
            client.close()
        elif args.command == "sources_delete":
            auth = _load_auth_impl()
            client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
            result, metrics = client.delete_source_impl(args.source_id)
            print(json.dumps({"deleted": result, "metrics": metrics}, indent=2))
            client.close()
        elif args.command == "query":
            auth = _load_auth_impl()
            client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
            question = args.question
            if not question and not sys.stdin.isatty():
                question = sys.stdin.read().strip()
            assert question, "question required (positional argument or stdin)"
            result, metrics = client.query_impl(args.notebook_id, question)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
            client.close()
        elif args.command == "create_audio":
            auth = _load_auth_impl()
            client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
            format_code = AUDIO_FORMATS[args.format]
            length_code = AUDIO_LENGTHS[args.length]
            result, metrics = client.create_audio_overview_impl(args.notebook_id, format_code, length_code, args.language)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
            client.close()
        elif args.command == "create_video":
            auth = _load_auth_impl()
            client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
            format_code = VIDEO_FORMATS[args.format]
            style_code = VIDEO_STYLES[args.style]
            result, metrics = client.create_video_overview_impl(args.notebook_id, format_code, style_code, args.language)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
            client.close()
        elif args.command == "create_report":
            auth = _load_auth_impl()
            client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
            result, metrics = client.create_report_impl(args.notebook_id, args.format, args.language)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
            client.close()
        else:
            parser.print_help()
    except (AssertionError, Exception) as e:
        _log("ERROR", args.command or "unknown", str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# FASTMCP SERVER
# =============================================================================

def _run_mcp():
    from fastmcp import FastMCP
    mcp = FastMCP("notebooklm")
    
    @mcp.tool()
    def login(check: bool = False) -> str:
        """Authenticate with NotebookLM or check auth status.
        
        Args:
            check: If True, only check existing auth validity.
        
        Returns:
            JSON string with auth status.
        """
        if check:
            try:
                auth = _load_auth_impl()
                return json.dumps({"authenticated": True, "csrf_token_prefix": auth["csrf_token"][:20]})
            except Exception as e:
                return json.dumps({"authenticated": False, "error": str(e)})
        return json.dumps({"status": "Login must be performed via CLI with Chrome CDP"})
    
    @mcp.tool()
    def notebooks_list() -> str:
        """List all notebooks.
        
        Args:
            None
        
        Returns:
            JSON string with notebooks list and metrics.
        """
        auth = _load_auth_impl()
        client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
        try:
            result, metrics = client.list_notebooks_impl()
            return json.dumps({"notebooks": result, "metrics": metrics})
        finally:
            client.close()
    
    @mcp.tool()
    def notebooks_get(notebook_id: str) -> str:
        """Get notebook details.
        
        Args:
            notebook_id: The notebook ID.
        
        Returns:
            JSON string with notebook data and metrics.
        """
        auth = _load_auth_impl()
        client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
        try:
            result, metrics = client.get_notebook_impl(notebook_id)
            return json.dumps({"notebook": result, "metrics": metrics})
        finally:
            client.close()
    
    @mcp.tool()
    def notebooks_create(title: str = "") -> str:
        """Create a new notebook.
        
        Args:
            title: Notebook title.
        
        Returns:
            JSON string with created notebook info and metrics.
        """
        auth = _load_auth_impl()
        client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
        try:
            result, metrics = client.create_notebook_impl(title)
            return json.dumps({"notebook": result, "metrics": metrics})
        finally:
            client.close()
    
    @mcp.tool()
    def notebooks_delete(notebook_id: str) -> str:
        """Delete a notebook.
        
        Args:
            notebook_id: The notebook ID.
        
        Returns:
            JSON string with deletion status and metrics.
        """
        auth = _load_auth_impl()
        client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
        try:
            result, metrics = client.delete_notebook_impl(notebook_id)
            return json.dumps({"deleted": result, "metrics": metrics})
        finally:
            client.close()
    
    @mcp.tool()
    def sources_list(notebook_id: str) -> str:
        """List sources in a notebook.
        
        Args:
            notebook_id: The notebook ID.
        
        Returns:
            JSON string with sources list and metrics.
        """
        auth = _load_auth_impl()
        client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
        try:
            result, metrics = client.list_sources_impl(notebook_id)
            return json.dumps({"sources": result, "metrics": metrics})
        finally:
            client.close()
    
    @mcp.tool()
    def sources_add_url(notebook_id: str, url: str) -> str:
        """Add a URL source to a notebook.
        
        Args:
            notebook_id: The notebook ID.
            url: URL to add.
        
        Returns:
            JSON string with added source info and metrics.
        """
        auth = _load_auth_impl()
        client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
        try:
            result, metrics = client.add_url_source_impl(notebook_id, url)
            return json.dumps({"source": result, "metrics": metrics})
        finally:
            client.close()
    
    @mcp.tool()
    def sources_add_text(notebook_id: str, text: str, title: str = "Pasted Text") -> str:
        """Add a text source to a notebook.
        
        Args:
            notebook_id: The notebook ID.
            text: Text content.
            title: Source title.
        
        Returns:
            JSON string with added source info and metrics.
        """
        auth = _load_auth_impl()
        client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
        try:
            result, metrics = client.add_text_source_impl(notebook_id, text, title)
            return json.dumps({"source": result, "metrics": metrics})
        finally:
            client.close()
    
    @mcp.tool()
    def sources_delete(source_id: str) -> str:
        """Delete a source.
        
        Args:
            source_id: The source ID.
        
        Returns:
            JSON string with deletion status and metrics.
        """
        auth = _load_auth_impl()
        client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
        try:
            result, metrics = client.delete_source_impl(source_id)
            return json.dumps({"deleted": result, "metrics": metrics})
        finally:
            client.close()
    
    @mcp.tool()
    def query(notebook_id: str, question: str) -> str:
        """Query a notebook.
        
        Args:
            notebook_id: The notebook ID.
            question: Question to ask.
        
        Returns:
            JSON string with query result and metrics.
        """
        auth = _load_auth_impl()
        client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
        try:
            result, metrics = client.query_impl(notebook_id, question)
            return json.dumps({"result": result, "metrics": metrics})
        finally:
            client.close()
    
    @mcp.tool()
    def create_audio(notebook_id: str, format: str = "deep_dive", 
                     length: str = "default", language: str = "en") -> str:
        """Create an audio overview.
        
        Args:
            notebook_id: The notebook ID.
            format: Audio format (deep_dive, brief, critique, debate).
            length: Length (short, default, long).
            language: Language code.
        
        Returns:
            JSON string with audio creation result and metrics.
        """
        auth = _load_auth_impl()
        client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
        try:
            format_code = AUDIO_FORMATS.get(format, 1)
            length_code = AUDIO_LENGTHS.get(length, 2)
            result, metrics = client.create_audio_overview_impl(notebook_id, format_code, length_code, language)
            return json.dumps({"result": result, "metrics": metrics})
        finally:
            client.close()
    
    @mcp.tool()
    def create_video(notebook_id: str, format: str = "explainer",
                     style: str = "auto_select", language: str = "en") -> str:
        """Create a video overview.
        
        Args:
            notebook_id: The notebook ID.
            format: Video format (explainer, brief).
            style: Visual style.
            language: Language code.
        
        Returns:
            JSON string with video creation result and metrics.
        """
        auth = _load_auth_impl()
        client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
        try:
            format_code = VIDEO_FORMATS.get(format, 1)
            style_code = VIDEO_STYLES.get(style, 1)
            result, metrics = client.create_video_overview_impl(notebook_id, format_code, style_code, language)
            return json.dumps({"result": result, "metrics": metrics})
        finally:
            client.close()
    
    @mcp.tool()
    def create_report(notebook_id: str, report_format: str = "Briefing Doc",
                      language: str = "en") -> str:
        """Create a report.
        
        Args:
            notebook_id: The notebook ID.
            report_format: Report format.
            language: Language code.
        
        Returns:
            JSON string with report creation result and metrics.
        """
        auth = _load_auth_impl()
        client = NotebookLMClient(auth["cookies"], auth["csrf_token"], auth["session_id"])
        try:
            result, metrics = client.create_report_impl(notebook_id, report_format, language)
            return json.dumps({"result": result, "metrics": metrics})
        finally:
            client.close()
    
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
