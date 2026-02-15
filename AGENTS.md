# AGENTS.md — Single File Bench

Guide for agentic coding assistants operating in this repository.

## Repo Overview

8 standalone Python CLI tools in `scripts/`. Each `sft_*.py`, `sfs_*.py`, `sfa_*.py`, or `sfd_*.py` is self-contained
with PEP 723 inline metadata — no centralized build system, no requirements.txt.
Dependencies are resolved at runtime by `uv run --script`.

Every script exposes up to three interfaces from one file:
1. **CLI** — `sfa_web.py search "pattern"` (argparse or click)
2. **Import** — `from sfa_web import _search_impl`
3. **MCP** — `sfa_web.py mcp-stdio` (FastMCP over stdio)

### Active Scripts

| Script | MCP | Version | EXPOSED | Description |
|---|---|---|---|---|
| `sft_web.py` | `web` | 4.2.0 | search, fetch, engines | Web search via SearXNG (reference implementation) |
| `sft_check.py` | `check` | 1.3.0 | check | SFB spec compliance checker |
| `sft_scout.py` | `fs` | 1.1.0 | 20 functions | Code reconnaissance, search, find, count, python_view |
| `sft_file_read.py` | `fr` | 1.1.0 | 11 functions | Read-only file ops: read, head, tail, cat, tree, stats |
| `sft_file_write.py` | `fw` | 1.1.0 | 12 functions | Destructive file ops: write, create, cp, mv, rm + backups |
| `sft_x.py` | — | 2.1.0 | search | Perplexity AI search |
| `sft_clipboard.py` | — | 1.1.0 | read, write | Clipboard read/write |
| `sfs_tts_stt_server.py` | — | 1.0.0 | start, stop, status, speak, listen | TTS/STT voice server with native Kokoro |
| `sft_chrome.py` | — | 1.1.0 | 22 functions | Chrome browser automation |

### python_view

`sft_scout.py python-view <file>` produces dense, structural AST output for AI
consumption. An AI agent familiar with SFB uses `python_view` to instantly understand
a script's capabilities, contracts, and structure — no need to read the full source.

## Build / Lint / Test Commands

### Running Scripts
```bash
# Direct execution (uv resolves deps from inline metadata)
uv run --script scripts/sft_web.py search "query"

# Or since scripts have shebangs and are +x:
./scripts/sft_web.py search "query"

# Run as MCP server
./scripts/sft_web.py mcp-stdio
```

### Linting & Formatting (no project configs — use defaults)
```bash
ruff check scripts/ --fix       # Lint + auto-fix
ruff format scripts/            # Format (line-length=88)
mypy scripts/                   # Type check
```

### Testing
No unit tests. Scripts use inline `assert` for runtime invariants.
Verification is manual: run the script, check its `.log` file.

**Required before promoting any script:**
```bash
# 1. Syntax validation - must pass
python3 -m py_compile scripts/sft_*.py

# 2. SPEC compliance check - must pass
./scripts/sft_check.py check scripts/sft_*.py

# 3. MCP server startup test (for scripts with MCP) - must start without errors
./scripts/sft_*.py mcp-stdio &
sleep 2
kill %1

# 4. Version flag test
./scripts/sft_*.py -V

# 5. Manual verification
./scripts/sft_web.py search "test query"
tail -20 scripts/sft_web_log.tsv
```

## Mandatory Pre-Work

**Read SPEC_SFA.md in full before writing or reviewing any code.** No exceptions.
Do not summarize from memory or rely on the guidelines below. The spec is the
source of truth; read it every session. The guidelines in this file are a quick
reference — they are not a substitute for the spec.

## Code Style Guidelines

### File Structure (mandatory order)

Every `sft_*.py`, `sfs_*.py`, `sfa_*.py`, or `sfd_*.py` follows this skeleton:

```
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [...]
# ///
"""Module docstring — one-line purpose."""

# stdlib imports (datetime, pathlib first for logging block)

# =========== LOGGING ===========
# (TSV logging block — see SPEC_SFA.md Principle 6)

# =========== CONFIGURATION ===========
EXPOSED = ["search", "fetch"]  # CLI + MCP — both interfaces
CONFIG = { ... }

# =========== CORE FUNCTIONS ===========
# =========== CLI INTERFACE ===========
# =========== FASTMCP SERVER ===========

if __name__ == "__main__":
    main()
```

Section headers use 77-char `=` lines with centered labels.
See SPEC_SFA.md Principle 3 for the EXPOSED list convention and configuration hierarchy.

### PEP 723 Inline Metadata
- Shebang: always `#!/usr/bin/env -S uv run --script`
- Python version: `>=3.11` (some use `>=3.10,<3.14`)
- Dependencies listed inline, quoted. Never `pip install` — `uv` resolves them.

### Imports
- **Order**: stdlib, then 3rd-party, then local. Grouped loosely.
- `datetime` and `pathlib` come first (needed by the logging block).
- FastMCP is imported **lazily** inside `_run_mcp()`, not at module level.
- `argparse` may be imported inside `main()` for the same reason.

### Naming
- **Files**: `sfa_<domain>.py` — the `sfa_` prefix is a **certification mark**, not decoration. Only scripts that pass ALL spec principles earn it. Verify with `sfa_check.py check scripts/sfa_*.py`.
- **Functions/variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Core functions**: `_<name>_impl()` suffix — called by both CLI and MCP
- **Private helpers**: leading underscore `_helper()`
- **MCP server names**: short lowercase (`"web"`, `"fs"`, `"fr"`, `"fw"`, `"check"`)

### Types & Formatting
- Modern syntax: `list[str]`, `dict[str, Any]`, `str | None` (not `Optional`)
- Core functions return tuples: `(data, metrics_dict)` or `(bool, str, str)`
- MCP tools always return `str` (JSON-serialized)
- Line length 88, double quotes, PEP 8 spacing

### Error Handling — Negative Space Programming
See SPEC_SFA.md Principle 9 for the full philosophy, accessor pattern table, and advantages.

Summary: assert contracts, access guaranteed fields directly (`[]` not `.get()`),
fail loud with messages stating what SHOULD be true. Never bare `except:`.
Only exceptions: transient network retries and external untrusted data (scraped HTML).

### Latency & Metrics
- Track latency in every `_impl` function: `start_ms = time.time() * 1000`
- Metrics dict always has `"status"` (`"success"` or `"error"`) and `"latency_ms"`
- Round to 2 decimals, echo original parameters in metrics

### Logging
See SPEC_SFA.md Principle 6 for the full TSV format spec, column layout, and level definitions.

- TSV format: `sfa_<name>_log.tsv` co-located with the script
- 8 columns: `timestamp script level event message detail metrics trace`
- Levels: FATAL, ERROR, WARN, INFO (default), DEBUG, TRACE
- Additive fields: columns populate left-to-right as verbosity increases
- Override via env: `SFA_LOG_LEVEL=DEBUG`, `SFA_LOG_DIR=/tmp/logs`

### Configuration
- **CONFIG dict** for related settings: `CONFIG = {"request_timeout_seconds": 30.0, "max_results": 50}`
- **Module constants** for standalone values: `SEARXNG_BASE = "http://..."`
- **Descriptive names with units** always: `request_timeout_seconds`, not `timeout`
- Prefer function parameters over module constants — see SPEC_SFA.md Principle 3

### Secrets
- Environment variables first, `scripts/.secrets` fallback, never hardcode
- Pattern: `os.environ.get("API_KEY") or _load_secrets().get("API_KEY")`

### Path Handling
- Always `pathlib.Path` objects, never string concatenation
- Create dirs safely: `path.parent.mkdir(parents=True, exist_ok=True)`

### Subprocess Execution
Standard helper pattern used across scripts:
```python
def _run(cmd: list[str], timeout: int = 30) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            err = result.stderr.strip()
            return f"ERROR: {err}" if err else f"ERROR: exit {result.returncode}"
        return result.stdout.strip()
    except FileNotFoundError:
        return f"ERROR: {cmd[0]} not found"
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out ({timeout}s)"
```

### MCP Integration
- FastMCP is **lazy-loaded** inside `_run_mcp()` — never at module level
- MCP tools wrap core `_impl` functions — no logic duplication
- MCP tools return JSON strings, not raw dicts
- MCP tool docstrings must include `Args:` section (FastMCP parses for schema)
- Print status to **stderr** only — stdout is the MCP protocol channel
- Register MCP servers in `opencode.jsonc` under the `"mcp"` key
- Two invocation patterns in config:
  - Direct: `"command": "./scripts/sfa_foo.py", "args": ["mcp-stdio"]`
  - Via uv (for compiled deps): `"command": "uv", "args": ["run", "--script", "...", "mcp-stdio"]`

### CLI Integration
- Use `argparse` with subparsers (preferred) or `click` with `@cli.command()`
- Always include `mcp-stdio` subcommand if the script has MCP tools
- CLI handlers call the same `_impl` functions as MCP tools
- Print JSON output for structured data: `json.dumps(result, indent=2)`
- Fallback: `parser.print_help()` when no command given
- All scripts support stdin/stdout for unix piping (one item per line, positional wins over stdin)
- See SPEC_SFA.md Principle 4 for the full intuitive interfaces philosophy

### Parameter Design (Principle 4)
- Always provide both short and long flags: `-c` / `--count`
- Primary argument is positional: `search "query"`, not `search --query "query"`
- Mirror standard unix flag conventions where applicable
- On error: state what was wrong, what was expected, show short help

### Adding a New Tool
1. Find or create `sfa_<domain>.py`
2. Add core function with type hints and docstring
3. Add both CLI subcommand AND MCP tool — every function needs both interfaces
4. Include the logging block and section headers
5. Register in `opencode.jsonc` if it runs as its own MCP server
6. Run `sfa_check.py check` — must pass before the script earns the `sfa_` prefix
7. Test via CLI: `./scripts/sfa_example.py do-thing "test"`

### Versioning & Changelog
See SPEC_SFA.md Principle 10 for the full semantic versioning spec.

- Every script has a version in its `-V`/`--version` flag (semver: MAJOR.MINOR.PATCH)
- `-V` (capital V) for `--version` — lowercase `-v` reserved for future `--verbose`
- DR (bug fix) = PATCH, RFC (feature) = MINOR, RFC (breaking) = MAJOR
- All changes tracked in CHANGELOG.md as checked/unchecked DR and RFC entries

### Agents
- **sfb-coder** — primary agent for implementing DRs/RFCs from CHANGELOG.md, auto-increments versions
- **sfb-reviewer** — read-only subagent that audits scripts and reports findings in DR/RFC format. **Must use sfl_test_gen** for test generation tasks.
- **python-expert** — read-only subagent for Python expertise, code review, and pattern guidance

## Environment Notes
- macOS (darwin), user john
- Passwordless sudo enabled
- `uv` (Astral) is the package runner — not pip, not poetry
- Scripts live in `scripts/`
- OpenCode config: `opencode.jsonc` (not `.mcp.json`)
- No CI/CD pipeline — local development only
- No .gitignore configured
