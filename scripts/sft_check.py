#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp"]
# ///
"""SFB compliance checker — validates scripts against SPEC_SFA.md principles.

The sft_/sfs_/sfa_/sfd_ prefixes are certification marks. This tool verifies a script earns it.

Usage:
    sft_check.py check scripts/sft_web.py
    sft_check.py check scripts/sft_*.py
    ls scripts/sft_*.py | sft_check.py check
    sft_check.py mcp-stdio
"""

import os
import re
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
EXPOSED = ["check"]  # CLI + MCP — both interfaces

# Section headers in mandatory order
REQUIRED_SECTIONS = [
    "LOGGING",
    "CONFIGURATION",
    "CORE FUNCTIONS",
    "CLI INTERFACE",
    "FASTMCP SERVER",
]


# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def _check_impl(filepath: str) -> tuple[dict, dict]:
    """Check a single script for SFA compliance.

    CLI: check
    MCP: check

    Returns (result_dict, metrics_dict).
    """
    import time

    start_ms = time.time() * 1000
    path = Path(filepath)

    assert path.exists(), f"{filepath} not found"
    assert path.suffix == ".py", f"{filepath} must be a .py file"

    content = path.read_text()
    assert content is not None, f"{filepath} content read successfully"
    lines = content.splitlines()
    filename = path.name

    checks = []
    passed = 0
    failed = 0

    def _pass(name: str, detail: str = ""):
        nonlocal passed
        passed += 1
        checks.append({"check": name, "status": "PASS", "detail": detail})

    def _fail(name: str, detail: str):
        nonlocal failed
        failed += 1
        checks.append({"check": name, "status": "FAIL", "detail": detail})

    # ---- 1. PEP 723 header ----
    # DR P2: Verify full shebang, not just #!/usr/bin/env
    if lines and lines[0].startswith("#!/usr/bin/env -S uv run"):
        _pass("shebang")
    else:
        _fail("shebang", "Missing #!/usr/bin/env -S uv run --script")

    if "# /// script" in content:
        _pass("pep723_block")
    else:
        _fail("pep723_block", "Missing # /// script metadata block")

    if re.search(r'# requires-python\s*=\s*"', content):
        _pass("requires_python")
    else:
        _fail("requires_python", "Missing requires-python in PEP 723 block")

    if re.search(r"# dependencies\s*=\s*\[", content):
        _pass("dependencies")
    else:
        _fail("dependencies", "Missing dependencies in PEP 723 block")

    # ---- 2. Module docstring ----
    # Look for triple-quoted string after the PEP 723 block
    if re.search(r'^"""', content, re.MULTILINE) or re.search(
        r"^'''", content, re.MULTILINE
    ):
        _pass("module_docstring")
    else:
        _fail("module_docstring", "Missing module-level docstring")

    # ---- 3. Section headers in order ----
    # Section labels may be on the line between two === decorator lines,
    # or on the same line as ====. Check both patterns.
    section_positions: dict[str, int] = {}
    for i, line in enumerate(lines):
        for section in REQUIRED_SECTIONS:
            if section in line:
                # Check if this line has ==== or if adjacent lines do
                has_decorator = "====" in line
                if not has_decorator and i > 0:
                    has_decorator = "====" in lines[i - 1]
                if not has_decorator and i < len(lines) - 1:
                    has_decorator = "====" in lines[i + 1]
                if has_decorator:
                    section_positions[section] = i
    
    # After collection, we can assert on REQUIRED_SECTIONS membership
    # External data (lines) has been processed, now we work with guaranteed section_positions

    missing_sections = [s for s in REQUIRED_SECTIONS if s not in section_positions]
    if not missing_sections:
        _pass("sections_present")
        # Check ordering
        positions = [section_positions[s] for s in REQUIRED_SECTIONS]
        if positions == sorted(positions):
            _pass("sections_ordered")
        else:
            _fail(
                "sections_ordered",
                f"Sections out of order: {[s for s in REQUIRED_SECTIONS if s in section_positions]}",
            )
    else:
        _fail("sections_present", f"Missing sections: {missing_sections}")

    # ---- 4. EXPOSED list ----
    if re.search(r"^EXPOSED\s*=\s*\[", content, re.MULTILINE):
        _pass("exposed_list")
        # Extract EXPOSED entries (supports multi-line lists)
        m = re.search(r"EXPOSED\s*=\s*\[([^\]]+)\]", content, re.DOTALL)
        exposed_names = (
            # DR P0: Filter empty strings from trailing commas
            [
                s
                for s in (
                    t.strip().strip('"').strip("'") for t in m.group(1).split(",")
                )
                if s
            ]
            if m
            else []
        )
    else:
        _fail("exposed_list", "Missing EXPOSED list in CONFIGURATION section")
        exposed_names = []

    # ---- 5. Logging block ----
    if "def _log(" in content:
        _pass("log_function")
    else:
        _fail("log_function", "Missing _log() function")

    # ---- 6. __main__ guard ----
    if 'if __name__ == "__main__"' in content or "if __name__ == '__main__'" in content:
        _pass("main_guard")
    else:
        _fail("main_guard", 'Missing if __name__ == "__main__"')

    # ---- 7. mcp-stdio subcommand ----
    if '"mcp-stdio"' in content or "'mcp-stdio'" in content:
        _pass("mcp_stdio")
    else:
        _fail("mcp_stdio", "Missing mcp-stdio subcommand")

    # ---- 8. FastMCP lazy loading ----
    # DR P1: Use indentation check instead of one-way in_function flag.
    # A fastmcp import at column 0 (no leading whitespace) is module-level.
    # A fastmcp import with leading whitespace is inside a function.
    fastmcp_at_module = False
    fastmcp_found = False
    for line in lines:
        stripped = line.strip()
        # Skip comments
        if stripped.startswith("#"):
            continue
        if "import fastmcp" in stripped or "from fastmcp" in stripped:
            fastmcp_found = True
            # Module-level = no leading whitespace on the actual line
            if not line[0].isspace():
                fastmcp_at_module = True
            break

    if fastmcp_found or "fastmcp" in content.lower():
        if not fastmcp_at_module:
            _pass("fastmcp_lazy")
        else:
            _fail(
                "fastmcp_lazy",
                "FastMCP imported at module level — must be inside _run_mcp()",
            )
    else:
        _fail("fastmcp_lazy", "No FastMCP import found")

    # ---- 9. CLI + MCP parity (check EXPOSED entries) ----
    if exposed_names:
        # Check for CLI subparsers
        cli_commands: set[str] = set()
        for m in re.finditer(r'add_parser\(\s*["\']([^"\']+)["\']', content):
            name = m.group(1)
            if name != "mcp-stdio":
                cli_commands.add(name)

        # Check for MCP tools
        # DR P2: Allow arguments in @mcp.tool() decorator
        mcp_tools: set[str] = set()
        for m in re.finditer(r"@mcp\.tool\([^)]*\)", content):
            # Find the next def after @mcp.tool()
            pos = m.end()
            func_match = re.search(r"def\s+(\w+)\s*\(", content[pos:])
            if func_match:
                mcp_tools.add(func_match.group(1))

        # Normalize: CLI uses hyphens, code uses underscores
        cli_normalized = {c.replace("-", "_") for c in cli_commands}
        # DR P0: Normalize MCP tool names to set for exact matching
        mcp_normalized = {t.replace("-", "_") for t in mcp_tools}

        for name in exposed_names:
            name_norm = name.replace("-", "_")
            has_cli = name_norm in cli_normalized or name in cli_commands
            # DR P0: Use set membership (exact match), not substring
            has_mcp = name_norm in mcp_normalized or name in mcp_tools

            if has_cli and has_mcp:
                _pass(f"parity:{name}")
            elif has_cli and not has_mcp:
                _fail(f"parity:{name}", f"'{name}' has CLI but no MCP tool")
            elif has_mcp and not has_cli:
                _fail(f"parity:{name}", f"'{name}' has MCP tool but no CLI subcommand")
            else:
                _fail(
                    f"parity:{name}", f"'{name}' in EXPOSED but has neither CLI nor MCP"
                )

    # ---- 10. stdin support ----
    # DR P1: Require actual isatty() call, not just "stdin" substring
    if "isatty()" in content and "sys.stdin" in content:
        _pass("stdin_support")
    else:
        _fail(
            "stdin_support",
            "No stdin support found (missing sys.stdin.isatty() pattern)",
        )

    # ---- 11. Parameter design: short + long flags ----
    # Every --long flag must have a short form. No exceptions.
    # Note: argparse auto-adds -h/--help, so --help won't appear in add_argument calls.
    # DR P1: Use re.DOTALL to match multi-line add_argument() calls
    add_arg_calls = re.findall(r"add_argument\(([^)]+)\)", content, re.DOTALL)
    flags_with_both = 0
    flags_missing_short = []
    for call in add_arg_calls:
        # Skip positional arguments (no leading -)
        args = [a.strip().strip('"').strip("'") for a in call.split(",")]
        flag_args = [a for a in args if a.startswith("-")]
        if len(flag_args) >= 2:
            flags_with_both += 1
        elif len(flag_args) == 1:
            # Single flag — check if it's only long form
            if flag_args[0].startswith("--"):
                flags_missing_short.append(flag_args[0])

    if flags_missing_short:
        _fail("short_long_flags", f"Flags missing short form: {flags_missing_short}")
    else:
        _pass("short_long_flags")

    # ---- 12. Bare except clauses (Principle 9) ----
    # RFC P0: Detect except: without exception type
    bare_excepts = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r"except\s*:", stripped) and not stripped.startswith("#"):
            bare_excepts.append(i)
    if bare_excepts:
        _fail(
            "bare_except",
            f"Bare except: at line(s) {bare_excepts} — use except Exception: or narrower",
        )
    else:
        _pass("bare_except")

    # ---- 13. MCP tool return types (Principle 7) ----
    # RFC P0: Verify @mcp.tool() functions return -> str (or -> Image for screenshots)
    bad_mcp_returns = []
    for m in re.finditer(r"@mcp\.tool\([^)]*\)", content):
        pos = m.end()
        func_match = re.search(
            r"def\s+(\w+)\s*\([^)]*\)\s*->\s*(\w+)", content[pos : pos + 500]
        )
        if func_match:
            fname = func_match.group(1)
            rtype = func_match.group(2)
            if rtype not in ("str", "Image"):
                bad_mcp_returns.append(f"{fname} -> {rtype}")
    if bad_mcp_returns:
        _fail(
            "mcp_return_type",
            f"MCP tools must return str (or Image): {bad_mcp_returns}",
        )
    else:
        _pass("mcp_return_type")

    # ---- 14. Version flag (Principle 10) ----
    # RFC P1: Verify -V/--version exists
    if re.search(r'"-V".*"--version"', content) or re.search(
        r"'-V'.*'--version'", content
    ):
        _pass("version_flag")
    else:
        _fail("version_flag", "Missing -V/--version flag (Principle 10)")

    # ---- 15. TSV logging format (Principle 6) ----
    # RFC P1: Verify logging uses TSV format with env var support
    tsv_issues = []
    if "_LEVELS" not in content:
        tsv_issues.append("missing _LEVELS dict")
    if "_HEADER" not in content:
        tsv_issues.append("missing _HEADER with TSV columns")
    if "SFB_LOG_LEVEL" not in content:
        tsv_issues.append("missing SFB_LOG_LEVEL env var support")
    if "SFB_LOG_DIR" not in content:
        tsv_issues.append("missing SFB_LOG_DIR env var support")
    if "_log.tsv" not in content and "_log.tsv" not in content:
        tsv_issues.append("log file should be _log.tsv not .log")
    if tsv_issues:
        _fail("tsv_logging", f"TSV logging issues: {tsv_issues}")
    else:
        _pass("tsv_logging")

    # ---- 16. _log() actually called in code (Principle 6) ----
    # RFC P2: Verify _log() is used, not just defined
    log_calls = len(re.findall(r"(?<!def )_log\(", content))
    if log_calls >= 1:
        _pass("log_usage", f"{log_calls} _log() call(s) found")
    else:
        _fail("log_usage", "_log() defined but never called — add operational logging")

    # ---- 17. No imports between LOGGING and CONFIGURATION (Principle 3) ----
    # RFC P2: Verify import placement
    if "LOGGING" in section_positions and "CONFIGURATION" in section_positions:
        log_pos = section_positions["LOGGING"]
        config_pos = section_positions["CONFIGURATION"]
        imports_between = []
        for i in range(log_pos + 1, config_pos):
            stripped = lines[i].strip()
            if (
                stripped.startswith("import ") or stripped.startswith("from ")
            ) and not stripped.startswith("#"):
                imports_between.append(f"L{i + 1}: {stripped}")
        if imports_between:
            _fail(
                "imports_placement",
                f"Imports between LOGGING and CONFIGURATION: {imports_between}",
            )
        else:
            _pass("imports_placement")

    # ---- 18. MCP tool docstrings have Args: section (Principle 7) ----
    # RFC P2: Verify MCP tools have Args: for schema generation
    tools_missing_args = []
    for m in re.finditer(r"@mcp\.tool\([^)]*\)", content):
        pos = m.end()
        # Find the function def and its docstring
        func_match = re.search(r"def\s+(\w+)\s*\(", content[pos : pos + 500])
        if func_match:
            fname = func_match.group(1)
            # Look for Args: in the next ~500 chars (covers the docstring)
            func_body_start = pos + func_match.end()
            snippet = content[func_body_start : func_body_start + 3000]
            # Check for docstring with Args:
            docstring_match = re.search(r'"""(.*?)"""', snippet, re.DOTALL)
            if not docstring_match or "Args:" not in docstring_match.group(1):
                tools_missing_args.append(fname)
    if tools_missing_args:
        _fail(
            "mcp_docstring_args",
            f"MCP tools missing Args: in docstring: {tools_missing_args}",
        )
    else:
        _pass("mcp_docstring_args")

    # ---- 19. CLI error boundary — try/except in main() (Principle 4) ----
    # Verify main() catches errors at the CLI boundary for clean output + logging
    cli_section = ""
    if "CLI INTERFACE" in section_positions:
        cli_start = section_positions["CLI INTERFACE"]
        # Find the next section or EOF
        cli_end = len(lines)
        for sec_name, sec_pos in section_positions.items():
            if sec_pos > cli_start and sec_pos < cli_end:
                cli_end = sec_pos
        cli_section = "\n".join(lines[cli_start:cli_end])

    boundary_issues = []
    if cli_section:
        if "try:" not in cli_section:
            boundary_issues.append("no try: block in CLI section")
        if "except" not in cli_section or (
            "AssertionError" not in cli_section and "Exception" not in cli_section
        ):
            boundary_issues.append("no except catching AssertionError/Exception")
        if '_log("ERROR"' not in cli_section and "_log('ERROR'" not in cli_section:
            boundary_issues.append('no _log("ERROR", ...) in except block')
        if "sys.exit(1)" not in cli_section and "sys.exit( 1)" not in cli_section:
            boundary_issues.append("no sys.exit(1) in except block")
    else:
        boundary_issues.append("CLI INTERFACE section not found")

    if boundary_issues:
        _fail("cli_error_boundary", f"CLI error boundary issues: {boundary_issues}")
    else:
        _pass("cli_error_boundary")

    # ---- 20. Pydantic Function Tools for LLM scripts (Principle 11) ----
    # RFC P2: Verify scripts that use LLM APIs have Pydantic models
    # Only check actual import statements, not arbitrary occurrences
    llm_patterns = [
        r"^import\s+openai",
        r"^from\s+openai",
        r"^import\s+anthropic",
        r"^from\s+anthropic",
        r"^from\s+google\.generativeai",
        r"^import\s+groq",
        r"^import\s+together",
    ]
    uses_llm = any(re.search(pattern, content, re.MULTILINE) for pattern in llm_patterns)
    
    if uses_llm:
        pydantic_issues = []
        
        # Check for BaseModel import
        if "BaseModel" not in content or "pydantic" not in content:
            pydantic_issues.append("missing Pydantic BaseModel import")
        
        # Check for Field import with descriptions
        if "Field(" not in content:
            pydantic_issues.append("missing Field() usage for parameter descriptions")
        
        # Check for at least one BaseModel subclass
        if not re.search(r"class\s+\w+\s*\(\s*BaseModel\s*\)", content):
            pydantic_issues.append("no Pydantic model class found")
        
        # Check for description= in Field calls
        field_calls = re.findall(r'Field\([^)]*description\s*=\s*["\']', content)
        if not field_calls:
            pydantic_issues.append("Field() calls missing description= parameter")
        
        if pydantic_issues:
            _fail("pydantic_function_tools", f"Pydantic issues: {pydantic_issues}")
        else:
            _pass("pydantic_function_tools", "LLM script uses Pydantic models with Field descriptions")
    else:
        _pass("pydantic_function_tools", "N/A - script does not use LLM APIs")

    # ---- 21. stdin must not block mcp-stdio (Principle 4) ----
    # Eager stdin reading before mcp-stdio dispatch starves the MCP transport.
    # Bad pattern: isatty() guard appears before `args.command == "mcp-stdio"` dispatch.
    if cli_section:
        isatty_pos = cli_section.find("isatty()")
        mcp_match = re.search(r"""args\.command\s*==\s*['"]mcp-stdio['"]""", cli_section)
        mcp_dispatch_pos = mcp_match.start() if mcp_match else -1
        if isatty_pos >= 0 and mcp_dispatch_pos >= 0 and isatty_pos < mcp_dispatch_pos:
            _fail("stdin_mcp_safe",
                  "stdin reading appears before mcp-stdio dispatch — will block MCP transport. "
                  "Dispatch mcp-stdio first or read stdin per-command.")
        else:
            _pass("stdin_mcp_safe")

    # ---- Build result ----
    compliant = failed == 0
    latency_ms = round(time.time() * 1000 - start_ms, 2)
    status_label = "COMPLIANT" if compliant else "NON-COMPLIANT"

    _log(
        "INFO",
        "check_complete",
        f"{filename}: {status_label}",
        detail=f"passed={passed} failed={failed}",
        metrics=f"latency_ms={latency_ms} status=success",
    )

    result = {
        "file": str(path),
        "filename": filename,
        "status": status_label,
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "checks": checks,
    }
    metrics = {
        "file": str(path),
        "passed": passed,
        "failed": failed,
        "compliant": compliant,
        "latency_ms": latency_ms,
        "status": "success",
    }
    return result, metrics


# =============================================================================
# CLI INTERFACE
# =============================================================================
def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="SFA compliance checker — validates scripts against SPEC_SFA.md principles"
    )
    # -V (capital) for version: lowercase -v reserved for future --verbose flag alignment
    parser.add_argument("-V", "--version", action="version", version="1.4.0")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # mcp-stdio
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")

    # CLI for _check_impl
    p_check = subparsers.add_parser("check", help="Check script(s) for SFA compliance")
    p_check.add_argument("files", nargs="*", help="Script file(s) to check")

    args = parser.parse_args()

    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "check":
            files = args.files
            if not files and not sys.stdin.isatty():
                files = [line.strip() for line in sys.stdin if line.strip()]
            assert files, "no files specified (positional argument or stdin)"

            all_results = []
            total_compliant = 0
            total_non = 0

            for filepath in files:
                try:
                    result, metrics = _check_impl(filepath)
                    all_results.append(result)
                    if metrics["compliant"]:
                        total_compliant += 1
                    else:
                        total_non += 1

                    # Print per-file summary
                    status = result["status"]
                    icon = "+" if status == "COMPLIANT" else "-"
                    print(
                        f"[{icon}] {result['filename']}: {status} ({result['passed']}/{result['total']} checks passed)"
                    )

                    # Print failures
                    for check in result["checks"]:
                        if check["status"] == "FAIL":
                            print(f"    FAIL: {check['check']} — {check['detail']}")

                except Exception as e:
                    _log("ERROR", "check", str(e), detail=f"file={filepath}")
                    print(f"[-] {filepath}: ERROR — {e}")
                    total_non += 1

            # Summary
            if len(files) > 1:
                print(
                    f"\nSummary: {total_compliant} compliant, {total_non} non-compliant, {len(files)} total"
                )

            # Exit with failure if any non-compliant
            sys.exit(0 if total_non == 0 else 1)
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


# =============================================================================
# FASTMCP SERVER
# =============================================================================
def _run_mcp():
    from fastmcp import FastMCP
    import json

    mcp = FastMCP("check")

    # MCP for _check_impl
    @mcp.tool()
    def check(filepath: str) -> str:
        """Check a script for SFA compliance against SPEC_SFA.md principles.

        Returns detailed pass/fail results for each compliance check.

        Args:
            filepath: Path to the Python script to check
        """
        result, metrics = _check_impl(filepath)
        return json.dumps({"result": result, "metrics": metrics}, indent=2)

    print("check MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
