#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp", "pydantic>=2.0.0"]
# ///
"""Test generator agent with iterative improvement loops. Generates tests until coverage threshold met.

Usage:
    sfl_test_gen.py generate FILE --target FUNC [options]
    sfl_test_gen.py mcp-stdio

Examples:
    sfl_test_gen.py generate mymodule.py --target calculate_total
    sfl_test_gen.py generate mymodule.py --target calculate_total --coverage 90 -l 5
    echo "mymodule.py" | sfl_test_gen.py generate --target main --output tests/
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
EXPOSED = ["generate"]  # CLI + MCP

CONFIG = {
    "max_iterations": 5,
    "coverage_threshold": 80.0,
    "model": "gemini-2.0-flash",
    "temperature": 0.2,
}

AGENT_PROMPT = """You are a world-class test engineer. Generate comprehensive tests for Python code.

Your goal: achieve maximum code coverage through iterative test generation.

Available tools:
1. analyze_code - Understand the code structure
2. check_coverage - Run tests and get coverage report
3. identify_uncovered - Find lines not covered by tests
4. generate_test - Create a new test for uncovered code
5. finalize - Complete when coverage threshold is met or max iterations reached

Analyze the code thoroughly. Generate tests for edge cases, error conditions, and happy paths.
Use coverage data to identify gaps and fill them systematically.

{{context}}
"""


# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def _analyze_code_impl(file_path: str, target_func: str | None = None) -> tuple[dict, dict]:
    """Analyze Python file structure and identify testable components.

    CLI: (used by generate command)
    MCP: analyze_code
    """
    start_ms = time.time() * 1000
    
    try:
        content = Path(file_path).read_text()
        lines = content.split("\n")
        
        # Parse functions and classes
        functions = []
        classes = []
        imports = []
        
        for i, line in enumerate(lines, 1):
            # Functions
            if match := re.match(r"^def\s+(\w+)\s*\(", line):
                func_name = match.group(1)
                if not func_name.startswith("_") or target_func == func_name:
                    functions.append({
                        "name": func_name,
                        "line": i,
                        "signature": line.strip()
                    })
            
            # Classes
            if match := re.match(r"^class\s+(\w+)", line):
                classes.append({
                    "name": match.group(1),
                    "line": i
                })
            
            # Imports
            if line.startswith("import ") or line.startswith("from "):
                imports.append(line.strip())
        
        analysis = {
            "file": file_path,
            "total_lines": len(lines),
            "functions": functions,
            "classes": classes,
            "imports": imports[:10],  # First 10 imports
        }
        
        if target_func:
            # Find specific function
            target = next((f for f in functions if f["name"] == target_func), None)
            if target:
                analysis["target"] = target
                # Extract function body (simple heuristic)
                start_line = target["line"]
                end_line = start_line
                for i in range(start_line, len(lines)):
                    if i > start_line and (lines[i].strip() and not lines[i].startswith(" ") and not lines[i].startswith("\t")):
                        break
                    if lines[i].strip():
                        end_line = i
                target["body_lines"] = end_line - start_line + 1
        
        latency_ms = round((time.time() * 1000 - start_ms), 2)
        metrics = {"status": "success", "latency_ms": latency_ms, "functions_found": len(functions)}
        return analysis, metrics
        
    except Exception as e:
        latency_ms = round((time.time() * 1000 - start_ms), 2)
        metrics = {"status": "error", "latency_ms": latency_ms, "error": str(e)}
        return {"error": str(e)}, metrics


def _check_coverage_impl(test_file: str | None, source_file: str) -> tuple[dict, dict]:
    """Run coverage check and return report.

    CLI: (used by generate command)
    MCP: check_coverage
    """
    start_ms = time.time() * 1000
    
    try:
        # Run pytest with coverage
        cmd = ["python", "-m", "pytest", "--cov", source_file, "--cov-report", "json", "-v"]
        if test_file:
            cmd.append(test_file)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        # Parse coverage.json if it exists
        coverage_data = {"percent": 0.0, "missing_lines": []}
        coverage_json = Path(".coverage/coverage.json")
        if coverage_json.exists():
            with open(coverage_json) as f:
                cov = json.load(f)
            # Find our file in coverage data
            for fname, data in cov.get("files", {}).items():
                if source_file in fname:
                    coverage_data = {
                        "percent": data.get("summary", {}).get("percent_covered", 0),
                        "missing_lines": data.get("missing_lines", [])
                    }
                    break
        
        latency_ms = round((time.time() * 1000 - start_ms), 2)
        metrics = {
            "status": "success",
            "latency_ms": latency_ms,
            "coverage_percent": coverage_data["percent"],
            "tests_passed": result.returncode == 0
        }
        return coverage_data, metrics
        
    except FileNotFoundError:
        # pytest-cov not installed
        latency_ms = round((time.time() * 1000 - start_ms), 2)
        metrics = {"status": "error", "latency_ms": latency_ms, "error": "pytest-cov not installed"}
        return {"error": "pytest-cov not installed"}, metrics
    except subprocess.TimeoutExpired:
        latency_ms = round((time.time() * 1000 - start_ms), 2)
        metrics = {"status": "error", "latency_ms": latency_ms, "error": "timeout"}
        return {"error": "coverage check timed out"}, metrics
    except Exception as e:
        latency_ms = round((time.time() * 1000 - start_ms), 2)
        metrics = {"status": "error", "latency_ms": latency_ms, "error": str(e)}
        return {"error": str(e)}, metrics


def _identify_uncovered_impl(coverage_data: dict, source_file: str) -> tuple[list, dict]:
    """Identify specific lines and branches not covered by tests.

    CLI: (used by generate command)
    MCP: identify_uncovered
    """
    start_ms = time.time() * 1000
    
    missing_lines = coverage_data.get("missing_lines", [])
    
    # Group consecutive lines into blocks
    blocks = []
    current_block = []
    
    for line in sorted(missing_lines):
        if not current_block or line == current_block[-1] + 1:
            current_block.append(line)
        else:
            blocks.append(current_block)
            current_block = [line]
    if current_block:
        blocks.append(current_block)
    
    # Create human-readable descriptions
    uncovered = []
    try:
        content = Path(source_file).read_text()
        lines = content.split("\n")
        
        for block in blocks:
            start = block[0]
            end = block[-1]
            code_snippet = "\n".join(lines[start-1:end])
            
            uncovered.append({
                "lines": f"{start}-{end}" if start != end else str(start),
                "code": code_snippet[:200],  # First 200 chars
                "needs_test": True
            })
    except Exception:
        pass
    
    latency_ms = round((time.time() * 1000 - start_ms), 2)
    metrics = {"status": "success", "latency_ms": latency_ms, "uncovered_blocks": len(blocks)}
    
    return uncovered, metrics


def _generate_test_impl(
    source_file: str,
    target_func: str,
    uncovered_block: dict,
    iteration: int
) -> tuple[str, dict]:
    """Generate a test for uncovered code.

    CLI: (used by generate command)
    MCP: generate_test
    """
    start_ms = time.time() * 1000
    
    # For MVP: generate template test
    # In production, this would call LLM to generate contextual test
    
    test_code = f"""# Iteration {iteration}: Test for lines {uncovered_block['lines']}
def test_{target_func}_scenario_{iteration}():
    \"\"\"Test {target_func} - generated for uncovered lines {uncovered_block['lines']}\"\"\"
    # TODO: Implement test based on code:
    # {uncovered_block['code'][:100].replace(chr(10), ' ')}
    
    # Arrange
    
    # Act
    result = {target_func}()
    
    # Assert
    assert result is not None
"""
    
    latency_ms = round((time.time() * 1000 - start_ms), 2)
    metrics = {"status": "success", "latency_ms": latency_ms, "iteration": iteration}
    
    return test_code, metrics


def _generate_impl(
    file_path: str,
    target_func: str | None = None,
    coverage_threshold: float = 80.0,
    max_iterations: int = 5,
    output_dir: str | None = None
) -> tuple[str, dict]:
    """Run iterative test generation loop.

    CLI: generate
    MCP: generate
    """
    start_ms = time.time() * 1000
    
    # Step 1: Analyze code
    analysis, _ = _analyze_code_impl(file_path, target_func)
    
    if "error" in analysis:
        return f"Error: {analysis['error']}", {"status": "error", "latency_ms": round((time.time() * 1000 - start_ms), 2)}
    
    # Determine target
    if not target_func:
        if analysis["functions"]:
            target_func = analysis["functions"][0]["name"]
        else:
            return "No functions found to test", {"status": "error", "latency_ms": round((time.time() * 1000 - start_ms), 2)}
    
    # Generate output filename
    source_path = Path(file_path)
    test_filename = f"test_{source_path.stem}.py"
    if output_dir:
        test_path = Path(output_dir) / test_filename
        test_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        test_path = source_path.parent / test_filename
    
    # Initialize test file with imports
    imports = f"""import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from {source_path.stem} import {target_func}

"""
    
    generated_tests = [imports]
    
    # Iterative loop
    for iteration in range(1, max_iterations + 1):
        _log("INFO", "iteration_start", f"Iteration {iteration}/{max_iterations}", detail=f"target={target_func}")
        
        # Step 2: Check current coverage (if test file exists)
        coverage_data = {"percent": 0.0, "missing_lines": []}
        if test_path.exists():
            coverage_data, _ = _check_coverage_impl(str(test_path), file_path)
        
        current_coverage = coverage_data.get("percent", 0.0)
        
        if current_coverage >= coverage_threshold:
            _log("INFO", "threshold_met", f"Coverage {current_coverage}% meets threshold {coverage_threshold}%")
            break
        
        # Step 3: Identify uncovered lines
        uncovered, _ = _identify_uncovered_impl(coverage_data, file_path)
        
        if not uncovered:
            _log("INFO", "no_uncovered", "No uncovered lines to test")
            break
        
        # Step 4: Generate test for first uncovered block
        test_code, _ = _generate_test_impl(file_path, target_func, uncovered[0], iteration)
        generated_tests.append(test_code)
        
        # Write test file
        test_path.write_text("\n\n".join(generated_tests))
        _log("INFO", "test_written", f"Wrote test to {test_path}", detail=f"iteration={iteration}")
    
    # Final coverage check
    final_coverage, _ = _check_coverage_impl(str(test_path), file_path)
    
    latency_ms = round((time.time() * 1000 - start_ms), 2)
    metrics = {
        "status": "success",
        "latency_ms": latency_ms,
        "iterations": iteration,
        "final_coverage": final_coverage.get("percent", 0.0),
        "threshold": coverage_threshold,
        "test_file": str(test_path),
        "target_function": target_func
    }
    
    output = f"""Test Generation Complete
========================
Target: {target_func} in {file_path}
Iterations: {iteration}
Final Coverage: {final_coverage.get('percent', 0.0):.1f}%
Threshold: {coverage_threshold}%
Test File: {test_path}

Generated {len(generated_tests) - 1} test(s).
"""
    
    return output, metrics


# =============================================================================
# CLI INTERFACE
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test generator agent with iterative improvement")
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 1.0.0")
    sub = parser.add_subparsers(dest="command")

    # generate command
    p_gen = sub.add_parser("generate", help="Generate tests for a Python file")
    p_gen.add_argument("file", nargs="?", help="Python file to generate tests for (or use stdin)")
    p_gen.add_argument("-t", "--target", help="Target function to test (default: first function)")
    p_gen.add_argument("-c", "--coverage", type=float, default=CONFIG["coverage_threshold"], help="Coverage threshold %%")
    p_gen.add_argument("-l", "--limit", type=int, default=CONFIG["max_iterations"], help="Max iterations")
    p_gen.add_argument("-o", "--output", help="Output directory for test files")

    # mcp-stdio
    p_mcp = sub.add_parser("mcp-stdio", help="Run as MCP server")

    args = parser.parse_args()

    try:
        if args.command == "generate":
            file_path = args.file
            if not file_path and not sys.stdin.isatty():
                file_path = sys.stdin.read().strip()
            assert file_path, "file required (positional argument or stdin)"
            
            result, metrics = _generate_impl(
                file_path,
                args.target,
                args.coverage,
                args.limit,
                args.output
            )
            print(result)
            _log("INFO", "generate", f"Test generation complete", detail=f"file={file_path}", metrics=str(metrics))
            
        elif args.command == "mcp-stdio":
            _run_mcp()
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
    from pydantic import BaseModel, Field
    mcp = FastMCP("test_gen")

    class GenerateArgs(BaseModel):
        file_path: str = Field(..., description="Path to Python file to test")
        target_func: str = Field("", description="Target function name (default: auto-detect)")
        coverage_threshold: float = Field(80.0, description="Coverage threshold percentage")
        max_iterations: int = Field(5, description="Maximum generation iterations")
        output_dir: str = Field("", description="Output directory for test files")

    @mcp.tool()
    def analyze_code(file_path: str, target_func: str = "") -> str:
        """Analyze Python file structure.
        
        Args:
            file_path: Path to Python file
            target_func: Optional specific function to analyze
        """
        analysis, _ = _analyze_code_impl(file_path, target_func or None)
        return json.dumps(analysis, indent=2)

    @mcp.tool()
    def generate(file_path: str, target_func: str = "", coverage_threshold: float = 80.0, max_iterations: int = 5, output_dir: str = "") -> str:
        """Generate tests iteratively until coverage threshold is met.
        
        Args:
            file_path: Path to Python file to test
            target_func: Target function name (auto-detect if empty)
            coverage_threshold: Coverage threshold percentage (default: 80)
            max_iterations: Maximum generation iterations (default: 5)
            output_dir: Output directory for test files
        """
        result, _ = _generate_impl(
            file_path,
            target_func or None,
            coverage_threshold,
            max_iterations,
            output_dir or None
        )
        return result

    @mcp.tool()
    def check_coverage(test_file: str, source_file: str) -> str:
        """Run coverage check on test file.
        
        Args:
            test_file: Path to test file
            source_file: Path to source file being tested
        """
        coverage, _ = _check_coverage_impl(test_file or None, source_file)
        return json.dumps(coverage, indent=2)

    print("Starting MCP server...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
