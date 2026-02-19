# CHANGELOG — SFB (Single-File Bench)

Changes tracked as **DR** (Discrepancy Report — bug/error) and **RFC** (Request for Change — new/changed functionality).

Completed items are checked and include the version where they landed.

---

## sfa_web.py
- [x] RFC: Add EXPOSED list to CONFIGURATION section (v4.1.0)
- [x] RFC: Add stdin support for search and fetch commands (v4.1.0)
- [x] DR: --category flag missing short form -C (v4.1.0)
- [x] DR: --all flag missing short form -a (v4.1.0)
- [x] DR: --version flag missing short form -V (v4.1.0)
- [x] DR: engines CLI command has no matching MCP tool — parity violation (v4.1.1)
- [x] DR: Logging uses old .log format — missing _LEVELS, _HEADER, SFA_LOG_LEVEL/DIR env vars (v4.1.1)
- [x] DR: 10 imports placed between LOGGING and CONFIGURATION sections (v4.1.1)
- [x] DR: MCP tools search and fetch missing Args: in docstring (v4.1.1)
- [x] RFC: Add CLI error boundary — try/except in main() for clean errors + logging (v4.1.2)
- [x] RFC: Add script column to TSV logging format (7→8 columns) (v4.2.0)

## sfa_x.py
- [x] RFC: Promote from .scripts_hold and rewrite to full SFA compliance (v2.0.0)
    - Perplexity AI search via httpx, TSV logging, section headers, EXPOSED list
    - CLI + MCP parity, lazy FastMCP, stdin support, short flags
- [x] RFC: Add CLI error boundary — try/except in main() for clean errors + logging (v2.0.1)
- [x] RFC: Add script column to TSV logging format (7→8 columns) (v2.1.0)

## sfa_clipboard.py
- [x] RFC: Add CLI error boundary — try/except in main() for clean errors + logging (v1.0.1)
- [x] RFC: Add script column to TSV logging format (7→8 columns) (v1.1.0)

## sfa_chrome.py
- [x] RFC: Promote from .scripts_hold and bring to full SFA compliance (v1.0.0)
    - TSV logging, section headers, EXPOSED list, lazy FastMCP, stdin support
    - 47 short flags added, 22 MCP tools added for parity, type hints modernized
- [x] DR: MCP `read()` swapped page_idx/format arg order — runtime TypeError (v1.0.0)
- [x] DR: MCP `do()` swapped query/profile and match_index/page_idx — runtime ValueError (v1.0.0)
- [x] DR: CLI `drag element` passed profile as from_selector — runtime ValueError (v1.0.0)
- [x] DR: CLI `drag coords` passed profile as from_x — runtime TypeError (v1.0.0)
- [x] DR: Bare `except:` in chrome_kill lockfile cleanup — catches KeyboardInterrupt (v1.0.0)
- [x] DR: Hardcoded `/tmp` screenshot paths — use `tempfile.gettempdir()` for cross-OS (v1.0.0)
- [x] RFC: Add CLI error boundary — try/except in main() for clean errors + logging (v1.0.1)
- [x] RFC: Add script column to TSV logging format (7→8 columns) (v1.1.0)

## sfa_check.py
- [x] RFC: Initial compliance checker — validates scripts against SPEC_SFA.md (v1.0.0)
- [x] DR: Section detection failed when label and === decorator on separate lines (v1.0.1)
- [x] DR: --version excluded from short-flag check without justification (v1.0.1)
- [x] DR: MCP parity used substring matching — "drag" in "drag_coords" = True (v1.1.0)
- [x] DR: Trailing comma in EXPOSED produced empty string phantom entry (v1.1.0)
- [x] DR: stdin check matched any "stdin" substring including comments/docstrings (v1.1.0)
- [x] DR: fastmcp_lazy used one-way in_function flag — module-level import after def passed (v1.1.0)
- [x] DR: short_long_flags regex missed multi-line add_argument() calls (v1.1.0)
- [x] DR: @mcp.tool() regex required empty parens — tool(name="x") invisible (v1.1.0)
- [x] DR: Shebang check only verified #!/usr/bin/env not full -S uv run --script (v1.1.0)
- [x] DR: Redundant except (AssertionError, Exception) — AssertionError is subclass (v1.1.0)
- [x] RFC: Add bare_except check — detect except: without exception type (v1.1.0)
- [x] RFC: Add mcp_return_type check — verify MCP tools return -> str or -> Image (v1.1.0)
- [x] RFC: Add version_flag check — verify -V/--version exists (v1.1.0)
- [x] RFC: Add tsv_logging check — verify _LEVELS, _HEADER, SFA_LOG_LEVEL/DIR (v1.1.0)
- [x] RFC: Add log_usage check — verify _log() actually called, not just defined (v1.1.0)
- [x] RFC: Add imports_placement check — no imports between LOGGING and CONFIGURATION (v1.1.0)
- [x] RFC: Add mcp_docstring_args check — MCP tools must have Args: in docstring (v1.1.0)
- [x] RFC: Add cli_error_boundary check — verify main() has try/except with _log + sys.exit(1) (v1.2.0)
- [x] RFC: Add CLI error boundary — try/except in own main() for clean errors + logging (v1.2.0)
- [x] RFC: Add script column to TSV logging format (7→8 columns) (v1.3.0)

## sfa_scout.py
- [x] RFC: Split from sfa_file_commands.py — scout + search domains (v1.0.0)
- [x] RFC: Add python_view subcommand for AI-readable script overview (v1.0.0)
- [x] DR: Bring to full SFA compliance (sections, EXPOSED, logging, lazy MCP, stdin, short flags) (v1.0.0)
- [x] RFC: Add CLI error boundary — try/except in main() for clean errors + logging (v1.0.1)
- [x] RFC: Add script column to TSV logging format (7→8 columns) (v1.1.0)

## sfa_file_read.py
- [x] RFC: Split from sfa_file_commands.py — read-only file operations (v1.0.0)
- [x] RFC: Add read_line and read_string commands (extracted from dual-mode string/line) (v1.0.0)
- [x] DR: Bring to full SFA compliance (sections, EXPOSED, logging, lazy MCP, stdin, short flags) (v1.0.0)
- [x] RFC: Add CLI error boundary — try/except in main() for clean errors + logging (v1.0.1)
- [x] RFC: Add script column to TSV logging format (7→8 columns) (v1.1.0)

## sfa_file_write.py
- [x] RFC: Split from sfa_file_commands.py — destructive file operations with backup system (v1.0.0)
- [x] RFC: Add write_line and write_string commands (extracted from dual-mode string/line) (v1.0.0)
- [x] DR: Bring to full SFA compliance (sections, EXPOSED, logging, lazy MCP, stdin, short flags) (v1.0.0)
- [x] RFC: Add CLI error boundary — try/except in main() for clean errors + logging (v1.0.1)
- [x] RFC: Add script column to TSV logging format (7→8 columns) (v1.1.0)

## sfa_file_commands.py (retired)
- [x] RFC: Split into sfa_scout.py, sfa_file_read.py, sfa_file_write.py (v4.0.0)
