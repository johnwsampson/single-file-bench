#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp", "pydantic>=2.0.0"]
# ///
"""JQ AI agent for natural language JSON processing. Generates and executes jq commands.

Usage:
    sft_jq.py generate "filter active users" data.json
    sft_jq.py query "extract all email addresses" data.json
    sft_jq.py mcp-stdio

Examples:
    sft_jq.py query "scores above 80" analytics.json
    sft_jq.py query "convert to CSV with name and age" users.json -o output.csv
    echo "get all unique cities" | sft_jq.py query - data.json
"""

import json
import os
import re
import subprocess
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
EXPOSED = ["generate", "query"]  # CLI + MCP

CONFIG = {
    "model": "gemini-2.0-flash",
    "temperature": 0.1,
}

AGENT_PROMPT = """You are a world-class expert at crafting precise jq commands for JSON processing.
Your goal is to generate accurate, minimal jq commands that exactly match the user's data manipulation needs.

Return ONLY the jq command - no explanations, comments, or extra text.
Always reference the input file specified in the user request.
Ensure the command follows jq best practices for efficiency and readability.

When outputting to a file, use the directory of the input file if no explicit directory is specified.
For lists of objects, default to outputting valid JSON arrays.

Examples:
- Select name and age where age > 30: jq '[.[] | select(.age > 30) | {name, age}]' data.json
- Count active users: jq '[.[] | select(.status == "active")] | length' users.json
- Convert to CSV: jq -r '.[] | [.name, .age] | @csv' data.json
- Sort by age descending: jq 'sort_by(.age) | reverse' data.json
- Save filtered results: jq '[.[] | select(.score > 80)]' data.json > high_scores.json

{{user_request}}

Your jq command:"""


# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def _jq_exec(jq_command: str, input_file: str) -> tuple[bool, str]:
    """Execute jq command on a file. Returns (success, output)."""
    import shlex
    try:
        # Parse jq_command safely - it should be just the jq filter + flags
        # Expected format: "jq '.filter'" or "jq -r '.filter'"
        cmd_parts = shlex.split(jq_command)
        
        # Validate that first part is 'jq'
        assert cmd_parts and cmd_parts[0] == "jq", f"Command must start with 'jq', got: {cmd_parts[0] if cmd_parts else 'empty'}"
        
        # Build safe argument list: jq [flags] 'filter' input_file
        args = ["jq"] + cmd_parts[1:] + [input_file]
        
        result = subprocess.run(
            args,
            shell=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr
    except FileNotFoundError:
        return False, "jq not found. Install: brew install jq"
    except subprocess.TimeoutExpired:
        return False, "jq command timed out (>30s)"


def _analyze_json_structure(input_file: str) -> tuple[dict, dict]:
    """Analyze JSON file structure to understand schema.
    
    CLI: (used by generate command)
    MCP: analyze_json
    """
    start_ms = time.time() * 1000
    
    try:
        with open(input_file) as f:
            data = json.load(f)
        
        # Analyze structure
        structure = {
            "type": type(data).__name__,
            "is_array": isinstance(data, list),
            "length": len(data) if isinstance(data, (list, dict)) else 1,
        }
        
        if isinstance(data, list) and data:
            # Analyze first item
            first = data[0]
            if isinstance(first, dict):
                structure["sample_keys"] = list(first.keys())[:10]
                structure["key_types"] = {k: type(v).__name__ for k, v in list(first.items())[:5]}
        elif isinstance(data, dict):
            structure["sample_keys"] = list(data.keys())[:10]
            structure["key_types"] = {k: type(v).__name__ for k, v in list(data.items())[:5]}
        
        latency_ms = round((time.time() * 1000 - start_ms), 2)
        metrics = {"status": "success", "latency_ms": latency_ms}
        return structure, metrics
        
    except Exception as e:
        latency_ms = round((time.time() * 1000 - start_ms), 2)
        metrics = {"status": "error", "latency_ms": latency_ms, "error": str(e)}
        return {"error": str(e)}, metrics


def _generate_impl(instruction: str, input_file: str) -> tuple[str, dict]:
    """Generate jq command from natural language instruction.

    CLI: generate
    MCP: generate
    """
    start_ms = time.time() * 1000
    
    # Analyze structure first
    structure, _ = _analyze_json_structure(input_file)
    
    # Build context for the prompt
    context = f"Working with: {input_file}\n"
    if "sample_keys" in structure:
        context += f"Available fields: {', '.join(structure['sample_keys'])}\n"
    if structure.get("is_array"):
        context += f"Array with {structure['length']} items\n"
    
    # For MVP: use pattern matching instead of LLM
    # In production, this would call Gemini API
    jq_cmd = _generate_jq_from_pattern(instruction, structure)
    
    latency_ms = round((time.time() * 1000 - start_ms), 2)
    metrics = {"status": "success", "latency_ms": latency_ms}
    
    return jq_cmd, metrics


def _query_impl(instruction: str, input_file: str, output_file: str | None = None, execute: bool = False) -> tuple[str, dict]:
    """Generate and optionally execute jq command.

    CLI: query
    MCP: query
    """
    start_ms = time.time() * 1000
    
    # Generate command
    jq_cmd, gen_metrics = _generate_impl(instruction, input_file)
    
    # Prepare full command with output redirection if specified
    if output_file:
        full_cmd = f"{jq_cmd} {input_file} > {output_file}"
    else:
        full_cmd = f"{jq_cmd} {input_file}"
    
    result = ""
    if execute:
        success, output = _jq_exec(jq_cmd, input_file)
        if output_file and success:
            # Write to file
            try:
                with open(output_file, "w") as f:
                    f.write(output)
                result = f"Output saved to {output_file}"
            except Exception as e:
                result = f"Error writing output: {e}"
        else:
            result = output
        gen_metrics["executed"] = True
        gen_metrics["execution_success"] = success
    else:
        result = full_cmd
        gen_metrics["executed"] = False
    
    latency_ms = round((time.time() * 1000 - start_ms), 2)
    gen_metrics["latency_ms"] = latency_ms
    gen_metrics["output_file"] = output_file
    
    return result, gen_metrics


def _generate_jq_from_pattern(instruction: str, structure: dict) -> str:
    """Generate jq command using pattern matching."""
    instruction_lower = instruction.lower()
    
    # Get available keys for field matching
    keys = structure.get("sample_keys", [])
    keys_lower = [k.lower() for k in keys]
    
    # Pattern: filter/select with condition
    if any(word in instruction_lower for word in ["filter", "select", "where", "with", "only"]):
        # Extract potential field and value
        # Simple heuristic: look for "field > value" or "field = value" patterns
        
        # Check for numeric comparisons
        gt_match = re.search(r'(\w+)\s*(?:>|above|greater than|over)\s*(\d+)', instruction_lower)
        lt_match = re.search(r'(\w+)\s*(?:<|below|less than|under)\s*(\d+)', instruction_lower)
        eq_match = re.search(r'(\w+)\s*(?:=|is|equals?)\s*["\']?([^"\']+)["\']?', instruction_lower)
        
        field = None
        op = None
        value = None
        
        if gt_match:
            field = gt_match.group(1)
            op = ">"
            value = gt_match.group(2)
        elif lt_match:
            field = lt_match.group(1)
            op = "<"
            value = lt_match.group(2)
        elif eq_match:
            field = eq_match.group(1)
            op = "=="
            value = eq_match.group(2)
            # Check if value looks like a number
            if value.isdigit():
                value = int(value)
            else:
                value = f'"{value}"'
        
        # Match field name against available keys
        if field and keys:
            matched_key = None
            for i, k in enumerate(keys_lower):
                if field in k or k in field:
                    matched_key = keys[i]
                    break
            if matched_key:
                if op == "==" and isinstance(value, str):
                    return f'jq \'[.[] | select(.{matched_key} == {value})]\''
                else:
                    return f'jq \'[.[] | select(.{matched_key} {op} {value})]\''
        
        # Default: identity filter
        return "jq '.'"
    
    # Pattern: extract specific fields
    if any(word in instruction_lower for word in ["extract", "get", "select field", "only", "just"]):
        # Look for field names in instruction
        requested_fields = []
        for key in keys:
            if key.lower() in instruction_lower:
                requested_fields.append(key)
        
        if requested_fields:
            fields_str = ", ".join([f".{f}" for f in requested_fields])
            if len(requested_fields) == 1:
                return f"jq '.[] | {requested_fields[0]}'"
            else:
                fields_obj = ", ".join([f"{f}: .{f}" for f in requested_fields])
                return f"jq '[.[] | {{{fields_obj}}}]'"
    
    # Pattern: count
    if "count" in instruction_lower or "how many" in instruction_lower:
        return "jq 'length'"
    
    # Pattern: convert to CSV
    if "csv" in instruction_lower:
        if keys:
            fields = "], [".join([f".{k}" for k in keys[:5]])
            return f"jq -r '.[] | [[{fields}]] | @csv'"
    
    # Pattern: unique/distinct
    if any(word in instruction_lower for word in ["unique", "distinct", "different"]):
        # Look for field to get unique values of
        for key in keys:
            if key.lower() in instruction_lower:
                return f"jq '[.[] | .{key}] | unique'"
        return "jq 'unique'"
    
    # Pattern: sort
    if "sort" in instruction_lower:
        for key in keys:
            if key.lower() in instruction_lower:
                if "desc" in instruction_lower or "reverse" in instruction_lower:
                    return f"jq 'sort_by(.{key}) | reverse'"
                return f"jq 'sort_by(.{key})'"
        return "jq 'sort'"
    
    # Default: identity
    return "jq '.'"


# =============================================================================
# CLI INTERFACE
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="JQ AI agent for natural language JSON processing")
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 1.0.0")
    sub = parser.add_subparsers(dest="command")

    # generate command
    p_gen = sub.add_parser("generate", help="Generate jq command from description")
    p_gen.add_argument("instruction", help="Natural language description of what to do")
    p_gen.add_argument("input_file", help="JSON file to process")

    # query command (generate + optionally execute)
    p_query = sub.add_parser("query", help="Generate and execute jq command")
    p_query.add_argument("instruction", help="Natural language description (or use stdin)")
    p_query.add_argument("input_file", help="JSON file to process (use - for stdin)")
    p_query.add_argument("-o", "--output", help="Output file path")
    p_query.add_argument("-x", "--execute", action="store_true", help="Execute the command")

    # mcp-stdio
    p_mcp = sub.add_parser("mcp-stdio", help="Run as MCP server")

    args = parser.parse_args()

    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "generate":
            cmd, metrics = _generate_impl(args.instruction, args.input_file)
            print(cmd)
            _log("INFO", "generate", f"Generated jq command", detail=f"file={args.input_file}", metrics=str(metrics))
            
        elif args.command == "query":
            instruction = args.instruction
            if instruction == "-" or (not instruction and not sys.stdin.isatty()):
                instruction = sys.stdin.read().strip()
            assert instruction, "instruction required (positional argument or stdin)"
            
            result, metrics = _query_impl(instruction, args.input_file, args.output, args.execute)
            print(result)
            _log("INFO", "query", f"Query completed", detail=f"file={args.input_file}", metrics=str(metrics))
        else:
            parser.print_help()
            
    except AssertionError as e:
        _log("ERROR", args.command or "unknown", str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        _log("ERROR", args.command or "unknown", str(e))
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# FASTMCP SERVER
# =============================================================================
def _run_mcp():
    from fastmcp import FastMCP
    mcp = FastMCP("jq")

    @mcp.tool()
    def analyze_json(input_file: str) -> str:
        """Analyze JSON file structure.
        
        Args:
            input_file: Path to JSON file
        """
        structure, _ = _analyze_json_structure(input_file)
        return json.dumps(structure, indent=2)

    @mcp.tool()
    def generate(instruction: str, input_file: str) -> str:
        """Generate jq command from natural language.
        
        Args:
            instruction: Natural language description
            input_file: Path to JSON file
        """
        cmd, _ = _generate_impl(instruction, input_file)
        return cmd

    @mcp.tool()
    def query(instruction: str, input_file: str, output_file: str = "", execute: bool = False) -> str:
        """Generate and optionally execute jq command.
        
        Args:
            instruction: Natural language description
            input_file: Path to JSON file
            output_file: Optional output file path
            execute: Whether to execute the command
        """
        result, _ = _query_impl(instruction, input_file, output_file or None, execute)
        return result

    @mcp.tool()
    def execute_jq(jq_command: str, input_file: str) -> str:
        """Execute a jq command.
        
        Args:
            jq_command: The jq command to run
            input_file: Path to JSON file
        """
        success, output = _jq_exec(jq_command, input_file)
        return output if success else f"ERROR: {output}"

    print("Starting MCP server...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
