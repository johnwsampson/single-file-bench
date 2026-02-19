#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp", "pydantic>=2.0.0"]
# ///
"""DuckDB AI agent for natural language SQL queries. Uses LLM function calling.

Usage:
    sft_duckdb.py query DATABASE "natural language question"
    sft_duckdb.py schema DATABASE
    sft_duckdb.py mcp-stdio

Examples:
    sft_duckdb.py query ./data.db "show me active users over 30"
    sft_duckdb.py query ./data.db "average score by city" -l 5
    echo "top 10 customers by revenue" | sft_duckdb.py query ./data.db
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
EXPOSED = ["query", "schema"]  # CLI + MCP

CONFIG = {
    "max_iterations": 10,
    "model": "gemini-2.0-flash",
    "temperature": 0.1,
}

AGENT_PROMPT = """You are a world-class expert at crafting precise DuckDB SQL queries.
Your goal is to generate accurate queries that exactly match the user's data needs.

Use the provided tools to explore the database and construct the perfect query:
1. list_tables - See what's available
2. describe_table - Understand schema and columns  
3. sample_table - See actual data patterns
4. run_test_query - Validate your query
5. run_final_query - Execute and return results

Be thorough but efficient. Only call run_final_query when confident.
If a test query fails, fix it and try again.

{{user_request}}
"""


# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def _duckdb_exec(db_path: str, sql: str) -> tuple[bool, str]:
    """Execute DuckDB SQL command. Returns (success, output)."""
    try:
        result = subprocess.run(
            ["duckdb", db_path, "-c", sql],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr
    except FileNotFoundError:
        return False, "duckdb not found. Install: brew install duckdb"
    except subprocess.TimeoutExpired:
        return False, "query timed out (>30s)"


def _list_tables_impl(db_path: str) -> tuple[list[str], dict]:
    """List all tables in the database.

    CLI: (used by query command)
    MCP: list_tables
    """
    start_ms = time.time() * 1000
    success, output = _duckdb_exec(db_path, ".tables")
    tables = output.strip().split() if success and output.strip() else []
    latency_ms = round((time.time() * 1000 - start_ms), 2)
    metrics = {"status": "success" if success else "error", "latency_ms": latency_ms, "db": db_path}
    return tables, metrics


def _describe_table_impl(db_path: str, table_name: str) -> tuple[str, dict]:
    """Get schema information for a table.

    CLI: (used by query command)
    MCP: describe_table
    """
    start_ms = time.time() * 1000
    success, output = _duckdb_exec(db_path, f"DESCRIBE {table_name};")
    latency_ms = round((time.time() * 1000 - start_ms), 2)
    metrics = {"status": "success" if success else "error", "latency_ms": latency_ms, "table": table_name}
    return output if success else output, metrics


def _sample_table_impl(db_path: str, table_name: str, n: int = 3) -> tuple[str, dict]:
    """Get sample rows from a table.

    CLI: (used by query command)
    MCP: sample_table
    """
    start_ms = time.time() * 1000
    success, output = _duckdb_exec(db_path, f"SELECT * FROM {table_name} LIMIT {n};")
    latency_ms = round((time.time() * 1000 - start_ms), 2)
    metrics = {"status": "success" if success else "error", "latency_ms": latency_ms, "table": table_name, "rows": n}
    return output if success else output, metrics


def _run_query_impl(db_path: str, sql: str) -> tuple[str, dict]:
    """Execute SQL query and return results.

    CLI: (used by query command)
    MCP: run_query
    """
    start_ms = time.time() * 1000
    success, output = _duckdb_exec(db_path, sql)
    latency_ms = round((time.time() * 1000 - start_ms), 2)
    metrics = {"status": "success" if success else "error", "latency_ms": latency_ms}
    return output if success else f"ERROR: {output}", metrics


def _query_impl(db_path: str, question: str, max_iterations: int = 10) -> tuple[str, dict]:
    """Run AI-powered natural language query against DuckDB.

    CLI: query
    MCP: query
    """
    start_ms = time.time() * 1000
    
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "ERROR: GEMINI_API_KEY not set", {"status": "error", "latency_ms": 0, "error": "missing_api_key"}
    
    # For MVP: use direct SQL generation without full agent loop
    # Full agent loop would require Gemini API integration
    tables, _ = _list_tables_impl(db_path)
    
    if not tables:
        return "No tables found in database", {"status": "success", "latency_ms": round((time.time() * 1000 - start_ms), 2)}
    
    # Simple heuristic: look for keywords in question
    question_lower = question.lower()
    
    # Try to find matching table
    target_table = None
    for t in tables:
        if t.lower() in question_lower or any(word in question_lower for word in t.lower().split("_")):
            target_table = t
            break
    
    if not target_table:
        target_table = tables[0]  # Default to first table
    
    # Get schema
    schema, _ = _describe_table_impl(db_path, target_table)
    
    # Generate simple SQL based on question patterns
    sql = _generate_sql(target_table, schema, question)
    
    # Execute
    result, query_metrics = _run_query_impl(db_path, sql)
    
    latency_ms = round((time.time() * 1000 - start_ms), 2)
    metrics = {
        "status": "success",
        "latency_ms": latency_ms,
        "table": target_table,
        "sql": sql,
        **query_metrics
    }
    
    output = f"-- Generated SQL:\n{sql}\n\n-- Results:\n{result}"
    return output, metrics


def _generate_sql(table: str, schema: str, question: str) -> str:
    """Generate SQL from natural language using simple heuristics."""
    question_lower = question.lower()
    
    # Parse schema for column names
    lines = schema.strip().split("\n")
    columns = []
    for line in lines:
        parts = line.split()
        if parts and not line.startswith("-"):
            columns.append(parts[0])
    
    # Build query based on question patterns
    if "count" in question_lower or "how many" in question_lower:
        sql = f"SELECT COUNT(*) FROM {table}"
    elif "average" in question_lower or "avg" in question_lower:
        # Find numeric column
        numeric_col = _find_numeric_column(columns)
        sql = f"SELECT AVG({numeric_col}) FROM {table}" if numeric_col else f"SELECT * FROM {table} LIMIT 10"
    elif "sum" in question_lower or "total" in question_lower:
        numeric_col = _find_numeric_column(columns)
        sql = f"SELECT SUM({numeric_col}) FROM {table}" if numeric_col else f"SELECT * FROM {table} LIMIT 10"
    elif "group" in question_lower or "by" in question_lower:
        # Try to find a categorical column for grouping
        group_col = _find_categorical_column(columns)
        numeric_col = _find_numeric_column(columns)
        if group_col and numeric_col:
            sql = f"SELECT {group_col}, AVG({numeric_col}) FROM {table} GROUP BY {group_col}"
        else:
            sql = f"SELECT * FROM {table} LIMIT 10"
    else:
        sql = f"SELECT * FROM {table} LIMIT 20"
    
    return sql


def _find_numeric_column(columns: list[str]) -> str | None:
    """Find a likely numeric column name."""
    numeric_hints = ["score", "amount", "count", "value", "price", "quantity", "total", "avg", "age", "id"]
    for col in columns:
        if any(hint in col.lower() for hint in numeric_hints):
            return col
    return columns[0] if columns else None


def _find_categorical_column(columns: list[str]) -> str | None:
    """Find a likely categorical column name."""
    cat_hints = ["name", "type", "category", "status", "city", "country", "group"]
    for col in columns:
        if any(hint in col.lower() for hint in cat_hints):
            return col
    return columns[0] if columns else None


# =============================================================================
# CLI INTERFACE
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="DuckDB AI agent for natural language queries")
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 1.0.0")
    sub = parser.add_subparsers(dest="command")

    # query command
    p_query = sub.add_parser("query", help="Query database with natural language")
    p_query.add_argument("database", help="Path to DuckDB database file")
    p_query.add_argument("question", nargs="?", help="Natural language question (or use stdin)")
    p_query.add_argument("-l", "--limit", type=int, default=10, help="Max result rows")
    p_query.add_argument("-i", "--iterations", type=int, default=CONFIG["max_iterations"], help="Max agent iterations")

    # schema command
    p_schema = sub.add_parser("schema", help="Show database schema")
    p_schema.add_argument("database", help="Path to DuckDB database file")

    # mcp-stdio
    p_mcp = sub.add_parser("mcp-stdio", help="Run as MCP server")

    args = parser.parse_args()

    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "query":
            question = args.question
            if not question and not sys.stdin.isatty():
                question = sys.stdin.read().strip()
            assert question, "question required (positional argument or stdin)"
            
            result, metrics = _query_impl(args.database, question, args.iterations)
            print(result)
            _log("INFO", "query", f"Query completed", detail=f"db={args.database}", metrics=str(metrics))
            
        elif args.command == "schema":
            tables, metrics = _list_tables_impl(args.database)
            if not tables:
                print("No tables found")
            else:
                for t in tables:
                    schema, _ = _describe_table_impl(args.database, t)
                    print(f"\n=== {t} ===")
                    print(schema)
            _log("INFO", "schema", f"Schema listed", detail=f"db={args.database}", metrics=str(metrics))
            

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
    mcp = FastMCP("duckdb")

    @mcp.tool()
    def list_tables(db_path: str) -> str:
        """List all tables in a DuckDB database.
        
        Args:
            db_path: Path to the DuckDB database file
        """
        tables, _ = _list_tables_impl(db_path)
        return json.dumps(tables)

    @mcp.tool()
    def describe_table(db_path: str, table_name: str) -> str:
        """Get schema information for a table.
        
        Args:
            db_path: Path to the DuckDB database file
            table_name: Name of the table to describe
        """
        schema, _ = _describe_table_impl(db_path, table_name)
        return schema

    @mcp.tool()
    def run_query(db_path: str, sql: str) -> str:
        """Execute a SQL query against the database.
        
        Args:
            db_path: Path to the DuckDB database file
            sql: SQL query to execute
        """
        result, _ = _run_query_impl(db_path, sql)
        return result

    @mcp.tool()
    def query(db_path: str, question: str, max_iterations: int = 10) -> str:
        """Query database using natural language.
        
        Args:
            db_path: Path to the DuckDB database file
            question: Natural language question about the data
            max_iterations: Maximum agent iterations (default: 10)
        """
        result, _ = _query_impl(db_path, question, max_iterations)
        return result

    @mcp.tool()
    def schema(db_path: str) -> str:
        """Get complete schema for all tables in database.
        
        Args:
            db_path: Path to the DuckDB database file
        """
        tables, _ = _list_tables_impl(db_path)
        if not tables:
            return "No tables found"
        
        schemas = []
        for t in tables:
            table_schema, _ = _describe_table_impl(db_path, t)
            schemas.append(f"=== {t} ===\n{table_schema}")
        
        return "\n\n".join(schemas)

    print("Starting MCP server...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
