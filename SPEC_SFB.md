# SFA Specification

**Single File Bench** — self-contained scripts for AI agents.

## Purpose

Each `sft_*.py`, `sfs_*.py`, `sfa_*.py`, or `sfd_*.py` file is a self-contained module designed for:

1. **CLI** — Direct command-line execution (`sft_web.py search "pattern"`)
2. **Import** — Usable as Python library by other scripts
3. **MCP** — FastMCP server via `mcp-stdio` subcommand

One file, three interfaces. No duplication.

## Architecture

### Per-Script FastMCP

Every `sfa_*.py` file CAN expose MCP tools directly via FastMCP. The MCP server is lazy-loaded — only instantiated when `mcp-stdio` is invoked:

```python
# Lazy FastMCP — only loaded when mcp-stdio is called
from fastmcp import FastMCP
mcp = FastMCP("toolname")

@mcp.tool()
def my_tool(arg: str) -> str:
    """Tool description."""
    return core_function(arg)
```

The `mcp-stdio` subcommand runs the server. `opencode.jsonc` registers the script.

## File Location

All SFB files live in `scripts/`. They are:
- Executable (`chmod +x`)
- Self-contained via PEP 723 inline metadata

### Naming — Four Script Types

Single File Bench organizes scripts by architectural role:

```
sft_<domain>.py  # Tools - CLI utilities, MCP tools, unix-style commands
sfs_<domain>.py  # Servers - HTTP/WebSocket services, long-running daemons
sfa_<domain>.py  # Agents - Autonomous task performers, AI workers
sfd_<domain>.py  # Daemons - Background watchers, monitors, cleanup
```

Domain groups related functions: `sft_web.py`, `sfs_tts_stt.py`, `sfa_coder.py`, etc.

**The `sft_`, `sfs_`, `sfa_`, `sfd_` prefixes are certification marks, not naming conventions.** A script carrying any prefix asserts compliance with ALL principles in this spec. If any principle is violated — missing MCP interface, no EXPOSED list, no stdin support, no logging block — the script must NOT carry an SFB prefix.

Scripts that are in development, partially compliant, or utility scripts that don't need the full SFB interface should use a different prefix or no prefix at all.

**Compliance is verifiable.** Run `sft_check.py` against any script to audit it:

```bash
# Check a single script
./scripts/sft_check.py scripts/sft_web.py

# Check all SFB scripts
./scripts/sft_check.py scripts/sft_*.py scripts/sfs_*.py scripts/sfa_*.py scripts/sfd_*.py

# Piped input
ls scripts/sft_*.py scripts/sfs_*.py scripts/sfa_*.py scripts/sfd_*.py | ./scripts/sft_check.py
```

The checker validates: PEP 723 header, section structure and ordering, EXPOSED list, CLI + MCP parity, stdin support, logging block, docstring, parameter design. Any failure means the script is non-compliant and should not carry an SFB prefix until fixed.

## File Structure

Every SFA file follows this structure:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "fastmcp"]
# ///
"""
<Module docstring — describes the domain and usage>

Usage:
    sft_example.py command [options]
    sft_example.py mcp-stdio
"""

import os
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
# CONFIGURATION (no external config files)
# =============================================================================

SOME_PATH = Path.home() / ".sfb" / "data"

# =============================================================================
# CORE FUNCTIONS (the actual logic — CLI and MCP both call these)
# =============================================================================

def do_thing(arg: str) -> str:
    """Pure function. No framework dependencies."""
    _log("INFO", "do_thing", f"Processing {arg}", detail=f"arg={arg}")
    ...

# =============================================================================
# CLI INTERFACE (argparse or click)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Example tool")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("do-thing")
    sub.add_parser("mcp-stdio", help="Run as MCP server")

    args = parser.parse_args()

    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "do-thing":
            print(do_thing(args.arg))
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
    mcp = FastMCP("example")

    @mcp.tool()
    def example_do_thing(arg: str) -> str:
        """Tool description for MCP clients."""
        return do_thing(arg)

    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
```

## Key Patterns

### PEP 723 Inline Metadata

See Principle 2 for the full rationale. The mechanical format:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "fastmcp"]
# ///
```

Never `pip install` — `uv` resolves deps from this header. Dependencies are quoted. Version constraints are supported: `"httpx>=0.25.0"`.

### Lazy MCP Loading

FastMCP and its tools are defined inside `_run_mcp()`, not at module level. This keeps CLI startup fast and avoids importing FastMCP when it's not needed.

### Core Functions Stay Pure

The actual logic lives in plain Python functions with no framework dependencies. CLI handlers and MCP tools both call into these same functions. This means:
- Functions are testable without MCP or CLI
- No import-time side effects
- Single source of truth for behavior

### Logging

See Principle 6 for the full TSV format specification, column layout, level definitions, and environment variable overrides. The `_log()` template is in the File Structure section above.

## Extending

Adding a new tool:

1. Find or create the appropriate `sfa_<domain>.py`
2. Add the core function with type hints and docstring
3. Add CLI subcommand AND MCP tool — every function is accessible from both interfaces
4. Register in `opencode.jsonc` if it runs as its own MCP server
5. Test via CLI: `sfa_example.py do-thing "test"`

## Principles

1. **One source of truth** — The Python function is the implementation. CLI and MCP are interfaces to it. Every core function MUST be exposed through both interfaces — no CLI-only or MCP-only commands.
2. **Single-file scripts** — PEP 723 inline metadata + `uv run --script`. One file is the complete deployment artifact.
3. **Script organization** — Uniform structure, configuration prominent at the top, EXPOSED list as the inventory of capabilities.
4. **Intuitive interfaces** — Tools work the way your intuition tells you to call them. If you have to think about which end of the hammer to grab, the tool has failed.
5. **Self-documenting** — Type hints + docstrings + `--help`.
6. **Structured TSV logging** — Uniform verbose logging across all tools, TSV format, queryable with standard unix tools.
7. **Lazy MCP via FastMCP** — Every script is both a CLI tool and an MCP server. FastMCP only loads when `mcp-stdio` is called.
8. **Testable in isolation** — Every function works standalone.
9. **Negative space** — Assert what you know to be true. Don't defend against what you think might go wrong.
10. **Semantic versioning** — Every script has a version. Changes are tracked in CHANGELOG.md as DRs and RFCs.
11. **Pydantic Function Tools** — Scripts that interface with LLMs MUST use Pydantic models for structured function calling. This provides machine-readable schemas, runtime validation, and self-documenting contracts.

### Single-File Scripts — PEP 723 + `uv run --script`

Every SFA tool is a single `.py` file with dependencies declared inline via PEP 723. There is no `requirements.txt`, no `pyproject.toml`, no virtual environment to manage. The runtime `uv` resolves dependencies from the script header on first invocation and caches them.

This architecture was chosen specifically for an arsenal of agent-authored and agent-consumed tools. The advantages:

1. **Zero environment setup.** No virtualenv creation, no `pip install`, no dependency manifest to keep in sync. An agent or human can run any script immediately — `uv` handles everything.

2. **Each script is its own dependency island.** `sfa_brain.py` can use `numpy 1.x` while `sfa_web.py` uses `httpx 0.25` — no conflicts. There is no shared environment to corrupt. This is critical for ~90 scripts that evolve independently.

3. **The script is the deployment artifact.** Copy one `.py` file to another machine, run it. No Dockerfile, no package manifest, no build step. The metadata travels with the code.

4. **No dependency drift.** The deps are declared *in* the file that uses them, not in a separate file that can fall out of sync. If you read the script, you know exactly what it needs — the header is the truth.

5. **Agent-friendly authoring.** An LLM can create a complete, runnable tool in a single file write. No multi-file scaffolding (`setup.py`, `pyproject.toml`, `requirements.txt`, virtualenv). PEP 723 is ideal for single-file agent scripts — the agent produces one file and it works.

6. **Fast iteration.** Change deps in the header, run again. `uv` detects the change and re-resolves. No `pip install -r requirements.txt` step between edits.

### Script Organization

Every SFA script should be instantly recognizable and immediately scannable. You open the file, you see the PEP 723 header (SFA uniform), you see the CONFIGURATION section (what can be adjusted), you see the EXPOSED list (what this script does). No hunting.

**The section order is mandatory and meaningful:**

1. **PEP 723 header + docstring** — identity and purpose
2. **Imports** — `datetime` and `pathlib` first (logging block needs them)
3. **LOGGING block** — always present, always the same
4. **CONFIGURATION** — all adjustable values, prominent and minimal
5. **CORE FUNCTIONS** — `_impl` functions, the actual logic
6. **CLI INTERFACE** — argparse handlers calling `_impl` functions
7. **FASTMCP SERVER** — MCP tool wrappers calling `_impl` functions

Each section has a reason for its position. Configuration comes before functions because you should see what values a script uses before reading the logic that uses them. CLI comes before MCP because it's the primary human interface.

**The CONFIGURATION section rules:**

The CONFIGURATION section is for deployment-fixed values that rarely change but must be easy to find when they do. It is NOT a dumping ground.

- **CONFIG dict** for related settings: `CONFIG = {"request_timeout_seconds": 30.0, "max_results": 50}`
- **Module constants** for standalone values: `SEARXNG_BASE = "http://localhost:8888"`
- **Descriptive names with units** — always: `request_timeout_seconds`, `fetch_max_content_bytes`, `cache_ttl_seconds`. Never: `timeout`, `max_content`, `ttl`.
- Infrastructure (HTTP clients, connection pools) belongs in CORE FUNCTIONS, not CONFIGURATION.

**The configuration hierarchy — prefer parameters over constants:**

| Level | Use for | Example |
|-------|---------|---------|
| Function parameter with default | Anything a caller might customize | `count: int = 10` |
| CONFIG dict / module constant | Deployment-fixed values (URLs, timeouts, limits) | `SEARXNG_BASE = "http://..."` |
| Environment variable | Secrets and cross-cutting overrides | `SFA_LOG_LEVEL`, `API_KEY` |
| Config file | Never | — |

If a value can reasonably be a function parameter with a sensible default, it should be. The CONFIGURATION section should be small.

**The EXPOSED list — inventory and contract:**

Every script declares its public capabilities in the CONFIGURATION section:

```python
# =============================================================================
# CONFIGURATION
# =============================================================================
EXPOSED = ["search", "fetch", "engines"]  # CLI + MCP — both interfaces

SEARXNG_BASE = "http://localhost:8888"
CONFIG = { ... }
```

The EXPOSED list serves three purposes:
1. **Inventory** — you see immediately what this script does, without scrolling
2. **Contract** — every entry must have both a CLI subcommand and an MCP tool
3. **Assertion target** — scripts can validate at startup that both interfaces match

Each `_impl` function's docstring references its interface names:

```python
def _search_impl(query: str, count: int = 10) -> tuple[list, dict]:
    """Search via SearXNG.
    
    CLI: search
    MCP: search
    """
```

And each CLI/MCP handler comments its `_impl` link:

```python
# CLI for _search_impl
p_search = subparsers.add_parser("search", ...)

# MCP for _search_impl
@mcp.tool()
def search(...):
```

This makes mismatches grepable: `grep -c "CLI for _search_impl"` and `grep -c "MCP for _search_impl"` should return the same count.

### Intuitive Interfaces

If you have to think about which end of the hammer to grab, the tool has already lost the battle of being useful. SFA tools should work the way your intuition tells you to call them — both from the CLI and from MCP.

**The five rules:**

1. **Mirror standard unix conventions.** If the tool wraps or resembles an existing unix tool, its parameters should support the same flags at minimum. `-c` for count, `-o` for output, `-v` for verbose. Don't invent new conventions when established ones exist.

2. **Always provide both short and long flags.** Every flag has two forms: `-c 10` for interactive use, `--count 10` for scripts and readability. argparse supports this natively: `add_argument("-c", "--count", type=int, default=10)`.

3. **Support natural positional arguments.** If the primary use case is `sfa_web.py search "query"`, then `query` is positional — don't force `--query "test"`. The most natural argument comes first, in the order a human would say it.

4. **Handle argument order intelligently.** Within the constraints of argparse, scripts should accept flags and positionals in any reasonable order. If ambiguity arises and can be resolved by type or content (e.g., a URL vs a search query), do so — but don't add complex inference machinery. Keep it simple.

5. **Fail with why and provide short help.** Don't just print "error: invalid arguments." State what was wrong, what was expected, and show the relevant usage. This is negative space applied to the CLI: assert the parameter contract, name the problem.

**CLI error boundary — `try/except` in `main()`:**

Every `main()` function wraps its command dispatch in a `try/except` block. This is the bridge between Principle 9 (fail loud with assertions) and Rule 5 above (fail with why). Assertions remain the validation mechanism inside `_impl` functions — they fire at the exact line where a contract breaks. But `main()` catches them at the CLI boundary so that:

1. **Users see clean errors**, not raw Python tracebacks
2. **Errors are logged to TSV**, providing observability
3. **Exit code 1** signals failure to calling scripts and pipelines

```python
def main():
    ...
    args = parser.parse_args()

    try:
        if args.command == "search":
            query = args.query
            assert query, "query required (positional argument or stdin)"
            ...
        else:
            parser.print_help()
    except AssertionError as e:
        # Contract violation - expected, user-friendly message from _impl functions
        _log("ERROR", "contract_violation", str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Unexpected failure - external system, network, I/O, etc.
        _log("ERROR", "runtime_error", str(e))
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
```

The `mcp-stdio` command is inside the try block — if the MCP server fails to start, the error is logged and reported cleanly. Separate exception handling distinguishes contract violations (assertion failures with clear messages) from unexpected runtime failures. The assertion message — written at authoring time by someone who understands the contract — becomes the user-facing error message. No traceback noise.

**Reference implementation — `sfa_web.py search`:**

```bash
# Natural positional argument
sfa_web.py search "climate change"

# Short flags for interactive use
sfa_web.py search "climate change" -c 5 -e google,brave -t week

# Long flags for scripts and readability
sfa_web.py search "climate change" --count 5 --engines google,brave --time-range week

# Flags in any order
sfa_web.py search -c 5 "climate change" --engines google,brave
```

All four invocations do the same thing. The positional (`query`) is obvious. The flags have both short and long forms. The short forms (`-c`, `-e`, `-t`) follow unix conventions where applicable.

**MCP tools follow the same spirit:**

- Parameter names are obvious: `query`, `count`, `url` — not `q`, `n`, `target`
- Types are precise: `count: int = 10`, not `count: str = "10"`
- Defaults are sensible: the tool works with just the required parameters
- The `Args:` docstring section tells an agent everything it needs to know without guessing

**Unix piping — stdin/stdout as first-class interfaces:**

Every SFA script supports stdin and stdout for unix chaining. A tool that can't pipe is a tool that can't compose — and composability is the unix advantage.

**stdin rules:**
- If stdin is a pipe (not a terminal), read from it
- Piped input is treated as **one item per line** — no `--batch` flag needed
- If both stdin and a positional argument are provided, the **positional wins silently** (matches `grep` behavior)
- Detection via `sys.stdin.isatty()`

**stdout rules:**
- CLI output goes to stdout: JSON for structured data, plain text for human-readable
- Status messages, progress, errors, MCP server startup — all **stderr**
- stdout is sacred — never pollute it with diagnostic output

**Standard stdin reading pattern:**

```python
def main():
    ...
    if args.command == "search":
        query = args.query
        if not query and not sys.stdin.isatty():
            query = sys.stdin.read().strip()
        assert query, "query required (positional argument or stdin)"
        result, metrics = search_impl(query, args.count)
        print(json.dumps({"results": result, "metrics": metrics}, indent=2))
```

For multi-item input (e.g., fetching multiple URLs):

```python
    if args.command == "fetch":
        urls = [args.url] if args.url else []
        if not urls and not sys.stdin.isatty():
            urls = [line.strip() for line in sys.stdin if line.strip()]
        assert urls, "url required (positional argument or stdin, one per line)"
        for url in urls:
            result, metrics = fetch_impl(url)
            print(json.dumps(result))
```

**Chaining examples:**

```bash
# SFA-to-unix piping
sfa_web.py search "climate change" | jq '.results[].url'

# Unix-to-SFA piping
echo "climate change" | sfa_web.py search

# SFA-to-SFA piping
sfa_web.py search "query" | jq -r '.results[].url' | sfa_web.py fetch

# Multi-line stdin
cat urls.txt | sfa_web.py fetch

# Chaining with standard unix tools
sfa_brain.py search "pattern" | jq '.results[]' | sort | uniq
```

### Structured TSV Logging

Every SFA script logs to a local TSV file by default. Logging is verbose (INFO level) out of the box. The format is designed for rapid querying with standard unix tools — no `jq` required.

**File convention:** `sfa_web.py` logs to `sfa_web_log.tsv`, co-located with the script.

**TSV column layout (8 columns, always present):**

```
#timestamp	script	level	event	message	detail	metrics	trace
```

| Column | Always populated | Description |
|--------|-----------------|-------------|
| timestamp | Yes | ISO 8601 UTC: `2026-02-14T14:30:05` |
| script | Yes | Script name, e.g. `sfa_web` |
| level | Yes | FATAL, ERROR, WARN, INFO, DEBUG, TRACE |
| event | Yes | Operation name: `search_start`, `fetch_complete` |
| message | Yes | Human-readable summary |
| detail | WARN+ empty | Structured k=v pairs for the operation |
| metrics | INFO+ empty | latency_ms, counts, status |
| trace | TRACE only | Raw payloads, stack traces, data dumps |

**Additive fields:** Every line has all 8 tab-separated columns, but columns populate additively from left to right as verbosity increases. Lower levels leave rightward columns empty (sparse). This means:
- A FATAL/ERROR line populates columns 1-5, columns 6-8 are empty tabs
- A WARN line populates columns 1-5 (same as ERROR — detail is optional for warnings)
- An INFO line populates columns 1-7 (adds detail and metrics)
- A DEBUG line populates columns 1-7 (richer detail and metrics than INFO)
- A TRACE line populates all 8 columns

This design ensures every line parses identically — `cut -f6` always means `detail` regardless of level. Sparse columns are empty tabs, valid TSV, no ambiguity.

**Log levels (standard hierarchy):**

| Level | Numeric | Purpose |
|-------|---------|---------|
| FATAL | 50 | Unrecoverable — process exits |
| ERROR | 40 | Operation failed, process continues |
| WARN | 30 | Unexpected but handled |
| INFO | 20 | Normal operations, milestones (default threshold) |
| DEBUG | 10 | Diagnostic: parameters, internal state |
| TRACE | 5 | Maximum verbosity: full payloads, data dumps |

**Environment variable overrides:**

- `SFA_LOG_LEVEL` — Set the threshold. Default: `INFO`. Only lines at or above this level are written. Example: `SFA_LOG_LEVEL=DEBUG ./scripts/sfa_web.py search "query"`
- `SFA_LOG_DIR` — Override log directory. Default: co-located with script. Example: `SFA_LOG_DIR=/tmp/sfa-logs ./scripts/sfa_web.py search "query"`

Environment variables (not CLI args) because logging is a cross-cutting concern — it shouldn't pollute each script's argparser. Consistent with the secrets pattern (`API_KEY`, etc.).

**Header line:** The first line of every log file is a TSV header prefixed with `#`:

```
#timestamp	script	level	event	message	detail	metrics	trace
```

The `#` prefix lets `grep -v '^#'` skip it for data processing, while tools that understand headers (e.g. `column -t`) use it directly.

**Unix query examples:**

```bash
# Count log entries by level
cut -f3 sfa_web_log.tsv | grep -v '^#' | sort | uniq -c

# Show all errors
awk -F'\t' '$3=="ERROR"' sfa_web_log.tsv

# Show search events with latency
awk -F'\t' '$4 ~ /search/' sfa_web_log.tsv | cut -f1,4,7

# Tail across all scripts
tail -f scripts/*_log.tsv
```

**Why TSV over JSON lines:**
- `cut`, `awk`, `sort`, `grep` work immediately — no `jq` dependency
- Fixed column positions mean consistent queries across all scripts
- Human-readable when printed with `column -ts$'\t'`
- Sparse columns are visually obvious (consecutive tabs)

**Why verbose by default:**
- Log storage is cheap. Missing diagnostic data when you need it is expensive.
- The threshold filter (`SFA_LOG_LEVEL`) lets you quiet things down when needed, but the default posture is to capture everything at INFO level and above.
- Consistent with negative space philosophy: if something happens, record it. Don't silently skip.

### Lazy MCP via FastMCP

Every SFA script can serve as both a CLI tool and an MCP server from the same file. FastMCP is the framework that makes this practical — a `@mcp.tool()` decorator with type hints and a docstring generates the full MCP schema automatically. No hand-written JSON schema, no protocol boilerplate. Adding MCP to a script costs ~15-20 lines.

**Why FastMCP, and why it's not unnecessary complexity:**

1. **One file, two distribution channels.** The same `sft_web.py` works for a human in a terminal (`./sft_web.py search "query"`) and for an LLM agent consuming it over MCP. Without this, you'd need a separate wrapper to expose logic to agents — either a second file (duplication) or a monolithic MCP proxy (an anti-pattern this architecture replaces).

2. **The `_impl` pattern isolates the complexity.** Business logic doesn't know MCP exists. `search_impl()` returns a tuple. The CLI handler formats it for humans. The MCP tool serializes it to JSON. If FastMCP disappeared tomorrow, you'd delete `_run_mcp()` and the CLI works unchanged. The core functions are framework-free.

3. **Lazy loading means zero CLI cost.** FastMCP is imported inside `_run_mcp()`, never at module level. CLI invocations never import it, never pay the startup cost, never break if it's not installed. The two entry points are completely independent.

4. **Agent ecosystems converge on MCP.** OpenCode, Claude Code, Cursor, and other agent platforms consume MCP servers natively. By making every script MCP-capable, the entire arsenal is immediately available to any agent that speaks the protocol. The alternative — subprocess invocation with output parsing — is fragile and loses type information.

5. **Schema generation from code.** FastMCP derives the tool's JSON schema from Python type hints and parses the docstring's `Args:` section for parameter descriptions. The same type hints that make the code self-documenting also generate the MCP interface. No schema to maintain separately, no drift between code and API contract.

### Negative Space Programming

**The rules:**

- Use assertions with positive messages for internal contracts and API guarantees: `assert "results" in data, f"SearXNG returns results key (got {list(data.keys())})"`
- Access guaranteed fields directly: `item["title"]`, not `item.get("title", "")`
- Assert at trust boundaries — assert the INPUT's contract. If a file must parse, assert it parses. If an API must return 200, assert 200.
- Every assertion message states what SHOULD be true and shows what was actually received.
- Fail loud, name the aggressor, stop. The program does not run on lies.
- The only exception is transient network retry logic where retry IS the design, and external untrusted data (e.g. scraped HTML) where graceful skipping is the correct posture.

**Trust Boundaries — where defensive coding ends and negative space begins:**

External inputs (use `.get()`, validation, graceful handling):
- Environment variables (`os.environ.get()`)
- Command-line arguments from users
- HTTP response bodies (external APIs)
- Filesystem paths provided by users
- Scraped HTML/web content

Internal contracts (use `[]` and assertions):
- Parsed configuration objects (after validation)
- Database query results (schema-enforced)
- Function return values (documented contracts)
- API responses after schema validation

**The three accessor patterns carry semantic meaning:**

| Pattern | Meaning | Comment needed? |
|---------|---------|-----------------|
| `item["title"]` | Field is guaranteed by contract | No — the access *is* the documentation |
| `item.get("content", "")` | Field is genuinely optional | Rarely — the `.get()` signals optionality |
| `assert x, "message"` | Contract that must hold or we stop | No — the assertion *is* the spec |

When you see `item["title"]` followed by `item.get("content", "")`, the code is telling you: title is guaranteed, content is optional. No comment required. The choice of accessor *is* the comment.

**Anti-patterns to avoid:**

```python
# WRONG: Defensive coding on guaranteed data
result = api_call()
if result and "data" in result and result["data"]:
    title = result["data"][0].get("title", "")
    
# RIGHT: Assert contract, then direct access
result = api_call()
assert "data" in result, f"API returns data key (got {list(result.keys())})"
title = result["data"][0]["title"]  # Guaranteed by contract
```

**Why negative space is superior — the advantages:**

1. **Zero distance between cause and symptom.** An assertion fails at the exact line where the contract breaks — not somewhere downstream where corrupted data finally causes a visible problem. Defensive code with `.get()` defaults lets bad data propagate silently, and you end up debugging symptoms three functions away from the cause.

2. **Self-commenting code.** The accessor pattern (`[]` vs `.get()` vs `assert`) communicates intent without prose. An agent or human reading `item["title"]` knows instantly that the field is guaranteed. Reading `item.get("title", "")` in defensive code, you can't tell if the field is optional by design or if the author was just being cautious. Negspace eliminates that ambiguity.

3. **Shorter code.** No `if error: return default` branches, no fallback chains, no null-check pyramids. Less code means fewer places for bugs to hide.

4. **Trustworthy outputs.** If the program didn't crash, the data is valid. Downstream consumers can trust the output without re-validating. You never have to wonder "did this silently degrade?"

5. **Errors are self-reporting.** Assertion messages are written at authoring time by someone who understands the contract: `f"Asked SearXNG for JSON, got {content_type}"`. Compare that to a `TypeError: 'NoneType' object is not subscriptable` from defensive code where a `.get()` returned `None` and it propagated three functions deep.

6. **Tests become trivial.** You test the happy path and verify assertions fire on bad input. No need to test dozens of degraded-but-running states because those states don't exist.

7. **Agent-readable contracts.** An LLM reading negspace code can infer the full API contract without external documentation — just from which fields use `[]` vs `.get()`. The code *is* the spec.

8. **LLM-optimized reasoning.** When an AI agent reads negspace code, it can reason about correctness more effectively. The absence of defensive code signals "this has been validated" — the agent doesn't need to generate null-checks, default handlers, or fallback logic. The contract tells the agent exactly what assumptions hold.

### Semantic Versioning

Every SFA script carries a version number in its `--version` / `-V` flag. Versions follow [semver](https://semver.org):

```
MAJOR.MINOR.PATCH
```

| Component | When to increment | Example |
|-----------|------------------|---------|
| MAJOR | Breaking change to CLI flags, MCP tool signatures, or output format | `1.2.3` → `2.0.0` |
| MINOR | New functionality added (new subcommand, new flag, new MCP tool) | `1.2.3` → `1.3.0` |
| PATCH | Bug fix, performance improvement, internal refactor with no interface change | `1.2.3` → `1.2.4` |

**Version location:** The version string lives in the argparse `add_argument("-V", "--version", ...)` call. There is no separate `__version__` constant — the argparse declaration is the single source of truth.

**CHANGELOG.md tracks all changes** using two entry types:

- **DR (Discrepancy Report)** — an error, bug, or deviation from spec. Produces a PATCH increment when fixed.
- **RFC (Request for Change)** — new functionality or changed behavior. Produces a MINOR or MAJOR increment when implemented.

CHANGELOG format:

```markdown
## sfa_web.py
- [ ] DR: search returns empty results when SearXNG returns 403 on JSON format
- [x] RFC: Add stdin support for search and fetch commands (v4.1.0)
- [x] DR: --category flag missing short form -C (v4.0.1)
```

Each entry is a checkbox. The sfa-coder agent marks items complete and increments the version when the fix or feature ships. Completed items include the version number where they landed.

### Pydantic Function Tools — Mandatory for LLM-Interfacing Scripts

**When required:** Any script that calls an LLM API to perform tasks MUST use Pydantic models for structured function/tool calling.

**Why mandatory:**
1. **Machine-readable contracts** — Pydantic generates JSON Schema automatically from type hints
2. **Runtime validation** — Invalid tool calls fail immediately with clear errors (negative space)
3. **Self-documenting** — The model IS the documentation; no separate schema maintenance
4. **Type safety** — Catches bugs at validation time, not during LLM execution

**Implementation rules:**

```python
from pydantic import BaseModel, Field

# CORRECT: Pydantic model with Field descriptions
class ListTablesArgs(BaseModel):
    reasoning: str = Field(..., description="Why we need to list tables")
    
class DescribeTableArgs(BaseModel):
    table_name: str = Field(..., description="Name of table to describe")

# Use with OpenAI/Anthropic
from openai import pydantic_function_tool
tools = [
    pydantic_function_tool(ListTablesArgs),
    pydantic_function_tool(DescribeTableArgs),
]
```

**When NOT required:**
- Simple CLI tools that don't call LLMs (e.g., `sft_web.py` search)
- File operations and system utilities
- Any script where LLM function calling isn't the primary interface

**Validation:** `sft_check.py` checks for:
- `from pydantic import BaseModel, Field` imports (when LLM client imported)
- At least one `BaseModel` subclass definition
- `Field(..., description=...)` usage for parameters
