#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "kokoro>=0.9.4",
#   "soundfile",
#   "torch",
#   "starlette",
#   "uvicorn",
#   "sse-starlette",
#   "httpx",
#   "fastmcp",
#   "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
# ]
# ///
"""TTS/STT voice server with native Kokoro TTS and browser speech recognition.

Bidirectional voice: server generates TTS audio and broadcasts to browsers
via SSE; browsers capture mic input, transcribe via Web Speech API, and
send text back. CLI/MCP commands act as HTTP clients to the running server.

Usage:
    sfa_tts_stt_server.py start [-p PORT] [-H HOST] [-S]
    sfa_tts_stt_server.py stop
    sfa_tts_stt_server.py status [-p PORT]
    sfa_tts_stt_server.py speak <text> [-p PORT]
    sfa_tts_stt_server.py listen [-p PORT]
    sfa_tts_stt_server.py mcp-stdio
"""

import os
import sys
import tempfile
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
EXPOSED = ["start", "stop", "status", "speak", "listen"]

# Enable MPS fallback for Apple Silicon GPU acceleration
# Environment variables are external - defensive access appropriate
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5115
CERT_DIR = Path.home() / ".sfb" / "certs"
PID_FILE = Path(tempfile.gettempdir()) / "sfa_tts_stt_server.pid"
TMUX_SESSION = "sfa-voice"

DEFAULT_VOICE_BLEND = ["af_heart", "af_nicole", "af_kore"]
SAMPLE_RATE_HZ = 24000

CONFIG = {
    "default_host": DEFAULT_HOST,
    "default_port": DEFAULT_PORT,
    "default_voice_blend": DEFAULT_VOICE_BLEND,
    "sample_rate_hz": SAMPLE_RATE_HZ,
    "default_speed": 1.0,
    "model_repo": "hexgrad/Kokoro-82M",
    "lang_code": "a",
    "request_timeout_seconds": 120.0,
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

# ---------------------------------------------------------------------------
# Browser client HTML — served at GET /
# ---------------------------------------------------------------------------

_BROWSER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="mobile-web-app-capable" content="yes">
    <title>Syne</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { height: 100%; overflow: hidden; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f1a; color: #eee;
            display: flex; flex-direction: column;
        }
        .header {
            background: #1a1a2e; padding: 12px 16px;
            border-bottom: 1px solid #2a2a4a; flex-shrink: 0;
        }
        .header-top {
            display: flex; justify-content: space-between;
            align-items: center; margin-bottom: 12px;
        }
        .title { font-size: 20px; font-weight: 600; color: #e94560; }
        .status {
            font-size: 12px; padding: 4px 10px;
            border-radius: 12px; background: #2a2a4a;
        }
        .status.connected { background: #166534; color: #4ade80; }
        .status.disconnected { background: #7f1d1d; color: #fca5a5; }
        .modes { display: flex; gap: 16px; }
        .mode-group { display: flex; align-items: center; gap: 8px; }
        .mode-label { font-size: 11px; color: #888; text-transform: uppercase; }
        .toggle-group {
            display: flex; background: #2a2a4a;
            border-radius: 6px; overflow: hidden;
        }
        .toggle-btn {
            padding: 6px 12px; border: none; background: transparent;
            color: #888; font-size: 14px; cursor: pointer; transition: all 0.2s;
        }
        .toggle-btn.active { background: #e94560; color: white; }
        .toggle-btn:hover:not(.active) { background: #3a3a5a; }
        .conversation {
            flex: 1; overflow-y: auto; padding: 16px;
            display: flex; flex-direction: column-reverse;
        }
        .message {
            margin-bottom: 12px; padding: 12px 16px;
            border-radius: 12px; max-width: 85%; word-wrap: break-word;
        }
        .message.user {
            background: #1e3a5f; color: #93c5fd;
            align-self: flex-end; border-bottom-right-radius: 4px;
        }
        .message.system {
            background: #3d1f3d; color: #f9a8d4;
            align-self: flex-start; border-bottom-left-radius: 4px;
        }
        .message-time { font-size: 10px; color: #666; margin-top: 4px; }
        .message.system .message-time { color: #9a6b9a; }
        .message.user .message-time { color: #5a7a9a; }
        .input-area {
            background: #1a1a2e; padding: 12px 16px;
            border-top: 1px solid #2a2a4a; flex-shrink: 0;
        }
        .voice-input {
            display: flex; flex-direction: column;
            align-items: center; gap: 8px;
        }
        .mic-btn {
            width: 64px; height: 64px; border-radius: 50%; border: none;
            background: #e94560; color: white; font-size: 28px;
            cursor: pointer; transition: all 0.2s;
            -webkit-tap-highlight-color: transparent;
        }
        .mic-btn:active { transform: scale(0.95); }
        .mic-btn.listening { background: #fbbf24; animation: pulse 1s infinite; }
        .mic-btn:disabled { background: #555; }
        @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.08); } }
        .mic-status { font-size: 13px; color: #888; min-height: 20px; }
        .mic-status.active { color: #fbbf24; }
        .text-input { display: flex; gap: 8px; }
        .text-input input {
            flex: 1; padding: 12px 16px; border: none; border-radius: 24px;
            background: #2a2a4a; color: #eee; font-size: 16px; outline: none;
        }
        .text-input input::placeholder { color: #666; }
        .text-input input:focus { background: #3a3a5a; }
        .send-btn {
            width: 48px; height: 48px; border-radius: 50%; border: none;
            background: #e94560; color: white; font-size: 20px; cursor: pointer;
            -webkit-tap-highlight-color: transparent;
        }
        .send-btn:active { transform: scale(0.95); }
        .send-btn:disabled { background: #555; }
        .hidden { display: none !important; }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-top">
            <span class="title">Syne</span>
            <span id="status" class="status disconnected">Connecting...</span>
        </div>
        <div class="modes">
            <div class="mode-group">
                <span class="mode-label">Input</span>
                <div class="toggle-group">
                    <button class="toggle-btn active" data-mode="input" data-value="voice">mic</button>
                    <button class="toggle-btn" data-mode="input" data-value="text">kbd</button>
                </div>
            </div>
            <div class="mode-group">
                <span class="mode-label">Output</span>
                <div class="toggle-group">
                    <button class="toggle-btn active" data-mode="output" data-value="voice">spk</button>
                    <button class="toggle-btn" data-mode="output" data-value="text">txt</button>
                </div>
            </div>
        </div>
    </div>
    <div class="conversation" id="conversation"></div>
    <div class="input-area">
        <div class="voice-input" id="voiceInput">
            <button class="mic-btn" id="micBtn">mic</button>
            <div class="mic-status" id="micStatus">Tap to speak</div>
        </div>
        <div class="text-input hidden" id="textInput">
            <input type="text" id="textField" placeholder="Type a message..." autocomplete="off">
            <button class="send-btn" id="sendBtn">go</button>
        </div>
    </div>
    <script>
        let inputMode='voice',outputMode='voice',eventSource=null,audioContext=null,recognition=null,isListening=false,clientId=null;
        const status=document.getElementById('status'),conversation=document.getElementById('conversation'),
              voiceInput=document.getElementById('voiceInput'),textInput=document.getElementById('textInput'),
              micBtn=document.getElementById('micBtn'),micStatus=document.getElementById('micStatus'),
              textField=document.getElementById('textField'),sendBtn=document.getElementById('sendBtn');
        document.querySelectorAll('.toggle-btn').forEach(btn=>{
            btn.addEventListener('click',()=>{
                const mode=btn.dataset.mode,value=btn.dataset.value;
                document.querySelectorAll(`.toggle-btn[data-mode="${mode}"]`).forEach(b=>b.classList.remove('active'));
                btn.classList.add('active');
                if(mode==='input'){inputMode=value;voiceInput.classList.toggle('hidden',value!=='voice');textInput.classList.toggle('hidden',value!=='text');if(value==='text')textField.focus();}
                else{outputMode=value;}
            });
        });
        function unlockAudio(){if(audioContext)return;audioContext=new(window.AudioContext||window.webkitAudioContext)();const b=audioContext.createBuffer(1,1,22050),s=audioContext.createBufferSource();s.buffer=b;s.connect(audioContext.destination);s.start(0);}
        async function playAudio(b64){if(outputMode!=='voice')return;if(!audioContext)unlockAudio();if(!audioContext)return;try{const bin=atob(b64),bytes=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)bytes[i]=bin.charCodeAt(i);const ab=await audioContext.decodeAudioData(bytes.buffer),src=audioContext.createBufferSource();src.buffer=ab;src.connect(audioContext.destination);src.start(0);}catch(e){console.error('[Audio]',e);}}
        function initSpeech(){if(!('webkitSpeechRecognition' in window)&&!('SpeechRecognition' in window)){micStatus.textContent='Speech not supported';micBtn.disabled=true;return;}const SR=window.SpeechRecognition||window.webkitSpeechRecognition;recognition=new SR();recognition.continuous=false;recognition.interimResults=true;recognition.lang='en-US';recognition.onstart=()=>{isListening=true;micBtn.classList.add('listening');micStatus.textContent='Listening...';micStatus.classList.add('active');};recognition.onend=()=>{isListening=false;micBtn.classList.remove('listening');micStatus.classList.remove('active');micStatus.textContent='Tap to speak';};recognition.onresult=(e)=>{let f='',interim='';for(let i=e.resultIndex;i<e.results.length;i++){if(e.results[i].isFinal)f+=e.results[i][0].transcript;else interim+=e.results[i][0].transcript;}micStatus.textContent=f||interim||'Listening...';if(f)sendMessage(f.trim());};recognition.onerror=(e)=>{isListening=false;micBtn.classList.remove('listening');micStatus.classList.remove('active');micStatus.textContent=e.error==='not-allowed'?'Mic denied':'Error: '+e.error;};}
        micBtn.addEventListener('click',()=>{unlockAudio();if(!recognition)initSpeech();if(!recognition)return;if(isListening)recognition.stop();else recognition.start();});
        textField.addEventListener('keypress',(e)=>{if(e.key==='Enter'&&textField.value.trim()){sendMessage(textField.value.trim());textField.value='';}});
        sendBtn.addEventListener('click',()=>{if(textField.value.trim()){sendMessage(textField.value.trim());textField.value='';}});
        async function sendMessage(text){addMessage(text,'user');try{await fetch('/api/voice-input',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,client_id:clientId,source:inputMode})});}catch(e){console.error('[Send]',e);}}
        function addMessage(text,speaker){const msg=document.createElement('div');msg.className='message '+speaker;const time=new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});msg.innerHTML=text+'<div class="message-time">'+time+'</div>';conversation.insertBefore(msg,conversation.firstChild);while(conversation.children.length>50)conversation.removeChild(conversation.lastChild);}
        function connect(){eventSource=new EventSource('/events');eventSource.onopen=()=>{status.textContent='Connected';status.classList.remove('disconnected');status.classList.add('connected');};eventSource.onerror=()=>{status.textContent='Disconnected';status.classList.remove('connected');status.classList.add('disconnected');setTimeout(connect,3000);};eventSource.addEventListener('connected',(e)=>{clientId=JSON.parse(e.data).clientId;});eventSource.addEventListener('message',(e)=>{const data=JSON.parse(e.data);if(data.type==='audio'){addMessage(data.text,'system');playAudio(data.audio);}});}
        initSpeech();connect();
    </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Tmux helpers
# ---------------------------------------------------------------------------

def _tmux_session_exists() -> bool:
    """Check if the managed tmux session is running."""
    import subprocess
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", TMUX_SESSION],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _tmux_send(text: str) -> bool:
    """Send text to the managed tmux session as keystrokes, then Enter separately."""
    import subprocess
    import time as _time
    try:
        # Send text first
        subprocess.run(
            ["tmux", "send-keys", "-t", TMUX_SESSION, text],
            check=True, capture_output=True, timeout=5,
        )
        _time.sleep(0.02)
        # Then send Enter separately
        subprocess.run(
            ["tmux", "send-keys", "-t", TMUX_SESSION, "Enter"],
            check=True, capture_output=True, timeout=5,
        )
        _log("INFO", "tmux_send", f"Sent to tmux: {text[:50]}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        _log("WARN", "tmux_send_fail", f"Failed to send to tmux: {text[:50]}")
        return False


VOICE_PRIME_PROMPT = (
    "You are receiving voice input from a human via speech-to-text. "
    "Input may contain transcription errors — interpret intent generously. "
    "You are working in the SFA (Syne Function Arsenal) repository. "
    "Respond concisely. Execute commands when asked. "
    "If the input is ambiguous, ask for clarification briefly."
)

VOICE_PRIME_DELAY_SECONDS = 3


def _tmux_create(dir_path: str) -> bool:
    """Create tmux session running opencode in dir_path, then prime it."""
    import subprocess
    import time as _time
    # Kill existing session if any
    subprocess.run(["tmux", "kill-session", "-t", TMUX_SESSION], capture_output=True)
    # Create new session
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", TMUX_SESSION, "-c", dir_path],
        check=True, capture_output=True,
    )
    # Launch opencode — text then Enter separately
    subprocess.run(
        ["tmux", "send-keys", "-t", TMUX_SESSION, "opencode"],
        check=True, capture_output=True,
    )
    _time.sleep(0.02)
    subprocess.run(
        ["tmux", "send-keys", "-t", TMUX_SESSION, "Enter"],
        check=True, capture_output=True,
    )
    _log("INFO", "tmux_create", f"Started opencode in {dir_path}")
    # Wait for opencode to load, then send priming prompt
    _time.sleep(VOICE_PRIME_DELAY_SECONDS)
    subprocess.run(
        ["tmux", "send-keys", "-t", TMUX_SESSION, VOICE_PRIME_PROMPT],
        check=True, capture_output=True,
    )
    _time.sleep(0.02)
    subprocess.run(
        ["tmux", "send-keys", "-t", TMUX_SESSION, "Enter"],
        check=True, capture_output=True,
    )
    _log("INFO", "tmux_prime", f"Sent priming prompt to opencode")
    return True


def _tmux_kill() -> bool:
    """Kill the managed tmux session."""
    import subprocess
    try:
        subprocess.run(
            ["tmux", "kill-session", "-t", TMUX_SESSION],
            check=True, capture_output=True, timeout=5,
        )
        _log("INFO", "tmux_kill", f"Killed tmux session {TMUX_SESSION}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _tmux_capture() -> str:
    """Capture current tmux pane content as plain text."""
    import subprocess
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", TMUX_SESSION, "-p"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


RESPONSE_POLL_INTERVAL_SECONDS = 0.2
RESPONSE_TIMEOUT_SECONDS = 120
RESPONSE_STABLE_SECONDS = 1.0
RESPONSE_MAX_TTS_CHARS = 2000


def _tmux_wait_response(before: str, timeout: int = RESPONSE_TIMEOUT_SECONDS) -> str:
    """Poll tmux pane until opencode finishes responding, extract response text.

    Detects completion by watching for a completion marker in content that
    has changed from the before snapshot, then waits for stabilization.
    Returns extracted response text, or empty string on timeout.
    """
    import time as _time

    deadline = _time.time() + timeout
    last_content = ""
    stable_since = 0.0
    response_detected = False

    while _time.time() < deadline:
        _time.sleep(RESPONSE_POLL_INTERVAL_SECONDS)
        current = _tmux_capture()

        # Check if there's a completion marker
        has_marker = "\u25a3" in current

        if not has_marker:
            continue  # Still processing, no marker yet

        # Has content changed from before snapshot?
        if current == before:
            continue  # No new content yet

        # We have a marker and new content - response detected
        response_detected = True

        # Content stabilized?
        if current == last_content:
            if stable_since == 0.0:
                stable_since = _time.time()
            elif _time.time() - stable_since >= RESPONSE_STABLE_SECONDS:
                response = _extract_opencode_response(current)
                if response:
                    _log("INFO", "tmux_response", f"Captured: {response[:80]}")
                return response
        else:
            stable_since = 0.0
            last_content = current

    if response_detected:
        _log("WARN", "tmux_response_timeout", f"Response detected but didn't stabilize after {timeout}s")
    else:
        _log("WARN", "tmux_response_timeout", f"No response detected after {timeout}s")
    return ""


def _extract_opencode_response(pane_text: str) -> str:
    """Extract the latest opencode response from pane capture.

    Opencode format:
      ┃  user prompt text     ← user message (┃ prefix)
      ┃
         Response line 1      ← assistant response (space-indented)
         Response line 2
         ▣  Build · model · Xs ← completion marker

    Extracts text between the last ┃ block and the last ▣ marker.
    """
    lines = pane_text.splitlines()

    # Find last ▣ completion marker
    marker_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if "\u25a3" in lines[i]:
            marker_idx = i
            break
    if marker_idx < 0:
        return ""

    # Walk backwards from marker to find the response block
    # Response lines are between the last ┃ section and the ▣ marker
    response_lines = []
    for i in range(marker_idx - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith("\u2503"):  # ┃ — hit user prompt, stop
            break
        if stripped:
            response_lines.append(stripped)

    response_lines.reverse()

    # Filter out UI chrome (structural patterns only — no model/agent names)
    clean = []
    for line in response_lines:
        # Skip keybind hints, box-drawing separators
        if "ctrl+" in line:
            continue
        if line.startswith("\u2579") or line.startswith("\u2501"):  # ╹ ━
            continue
        if "\u2580" in line:  # ▀ upper half block (separator bar)
            continue
        clean.append(line)

    result = " ".join(clean).strip()
    # Cap for TTS
    if len(result) > RESPONSE_MAX_TTS_CHARS:
        result = result[:RESPONSE_MAX_TTS_CHARS] + "..."
    return result


# ---------------------------------------------------------------------------
# SSE client and voice input management
# ---------------------------------------------------------------------------

class VoiceInput:
    """A voice input received from the browser."""

    def __init__(self, text: str, client_id: str | None = None, source: str = "voice"):
        import uuid
        self.id = str(uuid.uuid4())
        self.text = text
        self.client_id = client_id
        self.source = source
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.status = "pending"

    def to_dict(self) -> dict:
        return {
            "id": self.id, "text": self.text, "client_id": self.client_id,
            "source": self.source, "timestamp": self.timestamp, "status": self.status,
        }


class ClientManager:
    """Manage SSE client connections and voice input queue."""

    def __init__(self):
        import asyncio
        from collections import deque
        self.clients: dict = {}
        self.voice_inputs: deque = deque(maxlen=100)
        self.stats = {
            "total_connections": 0, "total_speaks": 0,
            "total_voice_inputs": 0, "started_at": datetime.now(timezone.utc).isoformat(),
        }

    def add_client(self, client_id: str):
        import asyncio
        queue = asyncio.Queue()
        self.clients[client_id] = queue
        self.stats["total_connections"] += 1
        _log("INFO", "sse_connect", f"Client {client_id} connected (total: {len(self.clients)})")
        return queue

    def remove_client(self, client_id: str):
        if client_id in self.clients:
            del self.clients[client_id]
            _log("INFO", "sse_disconnect", f"Client {client_id} disconnected (total: {len(self.clients)})")

    async def broadcast(self, message: dict):
        import json
        if not self.clients:
            return
        message_str = json.dumps(message)
        for client_id, queue in list(self.clients.items()):
            try:
                await queue.put(message_str)
            except Exception:
                self.remove_client(client_id)

    async def broadcast_audio(self, audio_b64: str, text: str):
        self.stats["total_speaks"] += 1
        await self.broadcast({
            "type": "audio", "audio": audio_b64, "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def add_voice_input(self, text: str, client_id: str | None = None, source: str = "voice") -> VoiceInput:
        vi = VoiceInput(text, client_id, source)
        self.voice_inputs.append(vi)
        self.stats["total_voice_inputs"] += 1
        _log("INFO", "voice_input", f"Received: {text[:50]}", detail=f"source={source}")
        return vi

    def get_pending_inputs(self) -> list[VoiceInput]:
        pending = [vi for vi in self.voice_inputs if vi.status == "pending"]
        for vi in pending:
            vi.status = "delivered"
        return pending

    def get_status(self) -> dict:
        pending_count = sum(1 for vi in self.voice_inputs if vi.status == "pending")
        return {
            "running": True,
            "connected_clients": len(self.clients),
            "pending_voice_inputs": pending_count,
            **self.stats,
        }


# ---------------------------------------------------------------------------
# SSL certificate generation
# ---------------------------------------------------------------------------

def _ensure_certs() -> tuple[str, str]:
    """Generate self-signed certs if needed. Returns (cert_path, key_path)."""
    import subprocess

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    cert_path = CERT_DIR / "server.crt"
    key_path = CERT_DIR / "server.key"

    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    _log("INFO", "ssl_gen", "Generating self-signed certificate")
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:4096",
        "-keyout", str(key_path), "-out", str(cert_path),
        "-days", "365", "-nodes", "-subj", "/CN=sfb-voice/O=SFB/C=US",
    ], check=True, capture_output=True)
    return str(cert_path), str(key_path)


# ---------------------------------------------------------------------------
# _impl functions
# ---------------------------------------------------------------------------

def _start_impl(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    use_ssl: bool = False,
    dir_path: str | None = None,
) -> None:
    """Start the TTS/STT voice server. Blocks until shutdown.

    CLI: start
    MCP: start

    Loads native Kokoro-82M with 3-voice blend (af_heart+af_nicole+af_kore),
    serves browser UI with SSE audio broadcast and voice input capture.
    If dir_path is provided, creates a tmux session running opencode in
    that directory — voice input from browsers is piped as keystrokes.
    """
    import asyncio
    import atexit
    import base64
    import io
    import json
    import re
    import uuid

    import numpy as np
    import soundfile as sf
    import torch
    import uvicorn
    from huggingface_hub import hf_hub_download
    from kokoro import KPipeline
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route
    from sse_starlette.sse import EventSourceResponse

    # --- Initialize TTS pipeline and voice blend ---
    _log("INFO", "tts_init", "Loading Kokoro pipeline")
    pipeline = KPipeline(lang_code=CONFIG["lang_code"], repo_id=CONFIG["model_repo"])

    tensors = []
    for name in DEFAULT_VOICE_BLEND:
        path = hf_hub_download(repo_id=CONFIG["model_repo"], filename=f"voices/{name}.pt")
        tensors.append(torch.load(path, weights_only=True))
        _log("DEBUG", "voice_load", f"Loaded {name}")
    blended_voice = torch.stack(tensors).mean(dim=0)
    _log("INFO", "tts_ready", f"Voice blend: {'+'.join(DEFAULT_VOICE_BLEND)}")

    def generate_audio(text: str) -> bytes:
        clean = re.sub(r"\([^)]*\)", "", text).strip() or text
        chunks = []
        for _, (gs, ps, audio) in enumerate(pipeline(clean, voice=blended_voice, speed=CONFIG["default_speed"])):
            chunks.append(audio)
        assert chunks, "Kokoro pipeline produced no audio"
        buf = io.BytesIO()
        sf.write(buf, np.concatenate(chunks), SAMPLE_RATE_HZ, format="WAV")
        buf.seek(0)
        return buf.getvalue()

    # --- Client manager ---
    clients = ClientManager()

    # --- Route handlers ---
    async def route_index(request):
        return HTMLResponse(_BROWSER_HTML)

    async def route_events(request):
        client_id = str(uuid.uuid4())
        queue = clients.add_client(client_id)

        async def event_generator():
            yield {"event": "connected", "data": json.dumps({"clientId": client_id})}
            try:
                while True:
                    try:
                        message = await asyncio.wait_for(queue.get(), timeout=30)
                        yield {"event": "message", "data": message}
                    except asyncio.TimeoutError:
                        yield {"event": "ping", "data": ""}
            except asyncio.CancelledError:
                pass
            finally:
                clients.remove_client(client_id)

        return EventSourceResponse(event_generator())

    async def route_speak(request):
        body = await request.json()
        text = body.get("text", "")
        if not text:
            return JSONResponse({"error": "text is required"}, status_code=400)

        _log("INFO", "tts_request", f"Generating: {text[:50]}")
        loop = asyncio.get_event_loop()
        wav_bytes = await loop.run_in_executor(None, generate_audio, text)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        await clients.broadcast_audio(audio_b64, text)

        return JSONResponse({
            "success": True, "text": text,
            "clients_notified": len(clients.clients), "audio_size": len(wav_bytes),
        })

    async def _voice_response_loop(before_snapshot: str):
        """Background task: wait for opencode response, TTS it back to browser."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _tmux_wait_response, before_snapshot)
        if not response:
            _log("WARN", "voice_loop", "No response captured from opencode")
            return
        _log("INFO", "voice_loop_tts", f"Speaking response: {response[:80]}")
        wav_bytes = await loop.run_in_executor(None, generate_audio, response)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        await clients.broadcast_audio(audio_b64, response)

    async def route_voice_input(request):
        body = await request.json()
        text = body.get("text", "").strip()
        if not text:
            return JSONResponse({"error": "text is required"}, status_code=400)
        vi = clients.add_voice_input(text, body.get("client_id"), body.get("source", "voice"))

        # Pipe voice input to tmux opencode session if active
        if _tmux_session_exists():
            # Snapshot pane BEFORE sending so we can detect new response
            loop = asyncio.get_event_loop()
            before = await loop.run_in_executor(None, _tmux_capture)
            await loop.run_in_executor(None, _tmux_send, text)
            # Spawn background task to capture response and TTS it back
            asyncio.create_task(_voice_response_loop(before))

        return JSONResponse({"success": True, "input": vi.to_dict(), "tmux": _tmux_session_exists()})

    async def route_pending_input(request):
        pending = clients.get_pending_inputs()
        return JSONResponse({"inputs": [vi.to_dict() for vi in pending], "count": len(pending)})

    async def route_status(request):
        return JSONResponse(clients.get_status())

    # --- Build and run app ---
    app = Starlette(routes=[
        Route("/", route_index),
        Route("/events", route_events),
        Route("/api/speak", route_speak, methods=["POST"]),
        Route("/api/voice-input", route_voice_input, methods=["POST"]),
        Route("/api/pending-input", route_pending_input, methods=["GET"]),
        Route("/api/status", route_status),
    ])

    # --- Tmux session (if dir_path provided) ---
    if dir_path:
        _tmux_create(dir_path)
        print(f"Tmux session '{TMUX_SESSION}' running opencode in {dir_path}", file=sys.stderr)

    def _cleanup():
        PID_FILE.unlink(missing_ok=True)
        _tmux_kill()

    PID_FILE.write_text(str(os.getpid()))
    atexit.register(_cleanup)

    protocol = "https" if use_ssl else "http"
    print(f"Voice server starting on {protocol}://{host}:{port}", file=sys.stderr)
    _log("INFO", "server_start", f"Listening on {protocol}://{host}:{port}",
         detail=f"pid={os.getpid()} tmux={bool(dir_path)}")

    kwargs: dict = {"host": host, "port": port}
    if use_ssl:
        cert, key = _ensure_certs()
        kwargs["ssl_certfile"] = cert
        kwargs["ssl_keyfile"] = key
        print("Accept the certificate warning in your browser", file=sys.stderr)

    uvicorn.run(app, **kwargs)


def _stop_impl() -> tuple[dict, dict]:
    """Stop the running voice server via PID file.

    CLI: stop
    MCP: stop
    """
    import signal as sig
    import time

    start_ms = time.time() * 1000

    if not PID_FILE.exists():
        latency = round(time.time() * 1000 - start_ms, 2)
        return (
            {"success": False, "error": "No PID file — server not running"},
            {"status": "error", "latency_ms": latency},
        )

    pid = int(PID_FILE.read_text().strip())

    # Kill tmux session if it exists
    tmux_killed = _tmux_kill()

    try:
        os.kill(pid, sig.SIGTERM)
        for _ in range(10):
            time.sleep(0.2)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        PID_FILE.unlink(missing_ok=True)
        msg = f"Stopped PID {pid}"
        if tmux_killed:
            msg += f", killed tmux session '{TMUX_SESSION}'"
        latency = round(time.time() * 1000 - start_ms, 2)
        _log("INFO", "server_stop", msg)
        return (
            {"success": True, "message": msg},
            {"status": "success", "latency_ms": latency},
        )
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        msg = f"PID {pid} already gone, cleaned up"
        if tmux_killed:
            msg += f", killed tmux session '{TMUX_SESSION}'"
        latency = round(time.time() * 1000 - start_ms, 2)
        return (
            {"success": True, "message": msg},
            {"status": "success", "latency_ms": latency},
        )


def _status_impl(port: int = DEFAULT_PORT) -> tuple[dict, dict]:
    """Check voice server status via HTTP.

    CLI: status
    MCP: status
    """
    import httpx
    import time

    start_ms = time.time() * 1000

    pid = int(PID_FILE.read_text().strip()) if PID_FILE.exists() else None

    try:
        r = httpx.get(f"http://127.0.0.1:{port}/api/status", timeout=3.0)
        if r.status_code == 200:
            data = r.json()
            data["pid"] = pid
            latency = round(time.time() * 1000 - start_ms, 2)
            return data, {"status": "success", "latency_ms": latency}
    except Exception:
        pass

    latency = round(time.time() * 1000 - start_ms, 2)
    return (
        {"running": False, "pid": pid},
        {"status": "success", "latency_ms": latency},
    )


def _speak_impl(text: str, port: int = DEFAULT_PORT) -> tuple[dict, dict]:
    """Send text to the voice server for TTS broadcast.

    CLI: speak
    MCP: speak

    The server generates audio via native Kokoro TTS and broadcasts
    to all connected browser clients via SSE.
    """
    import httpx
    import time

    start_ms = time.time() * 1000

    assert text and text.strip(), "text must not be empty"

    r = httpx.post(
        f"http://127.0.0.1:{port}/api/speak",
        json={"text": text},
        timeout=CONFIG["request_timeout_seconds"],
    )
    assert r.status_code == 200, f"Server returned {r.status_code}: {r.text}"

    result = r.json()
    latency = round(time.time() * 1000 - start_ms, 2)
    _log("INFO", "speak_complete", f"Spoke: {text[:50]}",
         detail=f"clients={result.get('clients_notified', 0)}", metrics=f"latency_ms={latency}")
    return result, {"status": "success", "latency_ms": latency}


def _listen_impl(port: int = DEFAULT_PORT) -> tuple[dict, dict]:
    """Get pending voice inputs from the server.

    CLI: listen
    MCP: listen

    Returns voice/text inputs received from browser clients
    since the last poll. Inputs are marked as delivered.
    """
    import httpx
    import time

    start_ms = time.time() * 1000

    r = httpx.get(
        f"http://127.0.0.1:{port}/api/pending-input",
        timeout=10.0,
    )
    assert r.status_code == 200, f"Server returned {r.status_code}: {r.text}"

    result = r.json()
    latency = round(time.time() * 1000 - start_ms, 2)
    _log("INFO", "listen_complete", f"Got {result.get('count', 0)} inputs",
         metrics=f"latency_ms={latency}")
    return result, {"status": "success", "latency_ms": latency}


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="TTS/STT voice server with native Kokoro TTS"
    )
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 1.0.0")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # mcp-stdio
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")

    # CLI for _start_impl
    p_start = subparsers.add_parser("start", help="Start the voice server")
    p_start.add_argument("-H", "--host", default=DEFAULT_HOST, help=f"Host to bind (default: {DEFAULT_HOST})")
    p_start.add_argument("-p", "--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    p_start.add_argument("-S", "--ssl", action="store_true", help="Enable HTTPS (needed for remote mic)")
    p_start.add_argument("-d", "--dir", default=None, help="Directory to open in tmux with opencode (enables voice ghosting)")

    # CLI for _stop_impl
    subparsers.add_parser("stop", help="Stop the running server")

    # CLI for _status_impl
    p_status = subparsers.add_parser("status", help="Check server status")
    p_status.add_argument("-p", "--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")

    # CLI for _speak_impl
    p_speak = subparsers.add_parser("speak", help="Send text to server for TTS broadcast")
    p_speak.add_argument("text", nargs="?", help="Text to speak")
    p_speak.add_argument("-p", "--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")

    # CLI for _listen_impl
    p_listen = subparsers.add_parser("listen", help="Get pending voice inputs from browsers")
    p_listen.add_argument("-p", "--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")

    args = parser.parse_args()

    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "start":
            _start_impl(host=args.host, port=args.port, use_ssl=args.ssl, dir_path=args.dir)
        elif args.command == "stop":
            result, metrics = _stop_impl()
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
        elif args.command == "status":
            result, metrics = _status_impl(port=args.port)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
        elif args.command == "speak":
            text = args.text
            if not text and not sys.stdin.isatty():
                text = sys.stdin.read().strip()
            assert text, "text required (positional argument or stdin)"
            result, metrics = _speak_impl(text, port=args.port)
            print(json.dumps({"result": result, "metrics": metrics}, indent=2))
        elif args.command == "listen":
            result, metrics = _listen_impl(port=args.port)
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
    import subprocess

    mcp = FastMCP("tts-stt-server")

    # MCP for _start_impl
    @mcp.tool()
    def start(host: str = "0.0.0.0", port: int = 5115, ssl: bool = False, dir: str = "") -> str:
        """Start the voice server in background.

        Spawns the server as a detached process. Use status to check readiness.
        If dir is provided, creates a tmux session running opencode in that directory.
        Voice input from browsers is piped as keystrokes into the tmux session.

        Args:
            host: Host to bind (default 0.0.0.0)
            port: Port number (default 5115)
            ssl: Enable HTTPS for remote mic access
            dir: Directory to open in tmux with opencode (enables voice ghosting)
        """
        cmd = [sys.executable, str(Path(__file__).resolve()), "start", "-H", host, "-p", str(port)]
        if ssl:
            cmd.append("--ssl")
        if dir:
            cmd.extend(["--dir", dir])
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        msg = {"success": True, "pid": proc.pid, "port": port}
        if dir:
            msg["tmux_session"] = TMUX_SESSION
            msg["dir"] = dir
        return json.dumps(msg)

    # MCP for _stop_impl
    @mcp.tool()
    def stop() -> str:
        """Stop the running voice server.

        Args:
            None
        """
        result, metrics = _stop_impl()
        return json.dumps({"result": result, "metrics": metrics})

    # MCP for _status_impl
    @mcp.tool()
    def status(port: int = 5115) -> str:
        """Check voice server status.

        Args:
            port: Server port (default 5115)
        """
        result, metrics = _status_impl(port=port)
        return json.dumps({"result": result, "metrics": metrics})

    # MCP for _speak_impl
    @mcp.tool()
    def speak(text: str, port: int = 5115) -> str:
        """Send text to the voice server for TTS broadcast to browsers.

        The server generates audio via native Kokoro TTS and broadcasts
        to all connected browser clients via SSE.

        Args:
            text: Text to speak
            port: Server port (default 5115)
        """
        result, metrics = _speak_impl(text, port=port)
        return json.dumps({"result": result, "metrics": metrics})

    # MCP for _listen_impl
    @mcp.tool()
    def listen(port: int = 5115) -> str:
        """Get pending voice inputs from browser clients.

        Returns voice/text inputs received from browsers since last poll.
        Inputs are marked as delivered after retrieval.

        Args:
            port: Server port (default 5115)
        """
        result, metrics = _listen_impl(port=port)
        return json.dumps({"result": result, "metrics": metrics})

    print("tts-stt-server MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
