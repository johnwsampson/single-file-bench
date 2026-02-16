#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "httpx", "duckdb", "fastmcp"]
# ///
"""Recall — unified search across all brain tables.

Blends semantic search (embedding similarity) with text search (regex)
and DuckDB analytical queries. Returns ranked results from FOQA tables,
links, catalog, and factoids.

Usage:
    sft_recall.py recall "query"
    sft_recall.py recall "query" -n 5
    sft_recall.py query "SELECT * FROM facts WHERE key LIKE '%alignment%'"
    echo "alignment" | sft_recall.py recall
    sft_recall.py mcp-stdio
"""

import json
import os
import pickle
import re
import sys
import time
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
EXPOSED = ["recall", "query"]

VERA_ROOT = Path(os.environ.get("VERA_ROOT", Path(__file__).parent.parent))
BRAIN_DIR = VERA_ROOT / "brain"
CACHE_DIR = VERA_ROOT / ".cache"
BRAIN_PKL = CACHE_DIR / "brain.pkl"
DB_PATH = CACHE_DIR / "brain.duckdb"

# Embedding model (same as sft_brain.py)
EMBED_MODEL = "openai/text-embedding-3-large"
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"

# FOQA table content columns for text search
TABLE_CONTENT_COLS = {
    "facts": "fact",
    "opinions": "opinion",
    "questions": "question",
    "aspirations": "aspiration",
}

STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "but", "and",
    "or", "if", "while", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "it", "its", "my", "your", "his", "her",
    "our", "their", "me", "him", "us", "them", "i", "you", "he", "she",
    "we", "they", "about", "up", "down",
})


# =============================================================================
# CORE FUNCTIONS (the actual logic — CLI and MCP both call these)
# =============================================================================

def _extract_terms(query: str) -> list[str]:
    """Extract meaningful search terms from natural language query."""
    words = re.findall(r"\w+", query.lower())
    return [w for w in words if w not in STOPWORDS and len(w) >= 2]


def _get_api_key() -> str:
    """Get OpenRouter API key from env or .secrets file."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        secrets_file = Path(__file__).parent / ".secrets"
        if secrets_file.exists():
            for line in secrets_file.read_text().splitlines():
                if line.startswith("export OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    return key


def _embed_query(text: str):
    """Embed a query string via OpenRouter API. Returns numpy array."""
    import httpx
    import numpy as np

    key = _get_api_key()
    assert key, "OPENROUTER_API_KEY not found in env or scripts/.secrets"
    resp = httpx.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=30,
    )
    resp.raise_for_status()
    return np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)


# ---- Semantic search (via brain.pkl embeddings) ----

def _search_semantic(query_emb, limit: int) -> list[dict]:
    """Semantic search against brain pickle cache."""
    import numpy as np

    if not BRAIN_PKL.exists():
        return []

    with open(BRAIN_PKL, "rb") as f:
        cache = pickle.load(f)

    entries = cache.get("entries", [])
    embeddings = cache.get("embeddings")
    if not entries or embeddings is None:
        return []

    # Vectorized cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    query_norm = np.linalg.norm(query_emb) + 1e-10
    sims = (embeddings @ query_emb) / (norms.squeeze() * query_norm)

    top_idx = np.argsort(sims)[::-1][:limit]

    results = []
    for idx in top_idx:
        entry = entries[idx]
        results.append({
            "source": "brain",
            "method": "semantic",
            "score": round(float(sims[idx]), 4),
            "key": entry.get("key", ""),
            "table": entry.get("_table", ""),
            "content": entry.get("_content", "")[:300],
        })

    return results


# ---- Keyword search (via brain.pkl entries) ----

def _search_keywords(terms: list[str], limit: int) -> list[dict]:
    """Keyword search against brain entries — catches terms semantic misses."""
    if not BRAIN_PKL.exists() or not terms:
        return []

    with open(BRAIN_PKL, "rb") as f:
        cache = pickle.load(f)

    entries = cache.get("entries", [])
    if not entries:
        return []

    pattern = "|".join(re.escape(t) for t in terms)
    regex = re.compile(pattern, re.IGNORECASE)
    n_terms = len(terms)
    results = []

    for entry in entries:
        content = entry.get("_content", "")
        key = entry.get("key", "")
        searchable = f"{key} {content}"
        if not regex.search(searchable):
            continue

        terms_hit = sum(1 for t in terms if re.search(re.escape(t), searchable, re.IGNORECASE))
        coverage = terms_hit / n_terms
        score = 0.35 + coverage * 0.30 + min(terms_hit, 4) * 0.03
        cap = min(0.75, 0.50 + n_terms * 0.08)

        results.append({
            "source": "brain",
            "method": "keyword",
            "score": round(min(score, cap), 4),
            "key": key,
            "table": entry.get("_table", ""),
            "content": content[:300],
        })

    results.sort(key=lambda x: -x["score"])
    return results[:limit]


# ---- Factoid search (via factoids.tsv) ----

def _search_factoids(terms: list[str], limit: int) -> list[dict]:
    """Search factoids.tsv for infrastructure/reference data."""
    factoids_path = BRAIN_DIR / "factoids.tsv"
    if not factoids_path.exists() or not terms:
        return []

    import csv
    with open(factoids_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [{k.lstrip("#"): v for k, v in r.items()} for r in reader]

    pattern = "|".join(re.escape(t) for t in terms)
    regex = re.compile(pattern, re.IGNORECASE)
    results = []

    for row in rows:
        searchable = f"{row.get('key', '')} {row.get('value', '')}"
        if regex.search(searchable):
            results.append({
                "source": "factoids",
                "method": "keyword",
                "score": 0.60,
                "key": row.get("key", ""),
                "table": "factoids",
                "content": row.get("value", ""),
            })
            if len(results) >= limit:
                break

    return results


# ---- Unified recall ----

def _recall_impl(query: str, limit: int = 10) -> dict:
    """Unified search across brain — semantic + keyword + factoids, merged and ranked.

    CLI: recall
    MCP: recall

    Args:
        query: Natural language query or search terms
        limit: Maximum results to return
    """
    assert query, "query required"
    t0 = time.time()

    terms = _extract_terms(query)

    # Semantic search with graceful degradation
    semantic_results = []
    try:
        query_emb = _embed_query(query)
        semantic_results = _search_semantic(query_emb, limit * 2)
    except Exception as e:
        _log("WARN", "recall_semantic", f"Semantic search unavailable: {e}")

    # Keyword search
    keyword_results = _search_keywords(terms, limit * 2)

    # Factoid search
    factoid_results = _search_factoids(terms, limit)

    # Merge, deduplicate by key, rank by score
    merged = semantic_results + keyword_results + factoid_results
    merged.sort(key=lambda x: -x["score"])

    seen = set()
    deduped = []
    for r in merged:
        k = r.get("key", r["content"][:80].lower())
        if k not in seen:
            seen.add(k)
            deduped.append(r)

    results = deduped[:limit]
    elapsed = round(time.time() - t0, 3)

    output = {
        "query": query,
        "terms": terms,
        "results": results,
        "counts": {
            "semantic": len(semantic_results),
            "keyword": len(keyword_results),
            "factoids": len(factoid_results),
            "returned": len(results),
        },
        "elapsed_seconds": elapsed,
    }

    _log("INFO", "recall", f"Recall for '{query[:30]}'",
         detail=f"terms={','.join(terms[:5])} results={len(results)}",
         metrics=f"semantic={len(semantic_results)} keyword={len(keyword_results)} "
                 f"factoids={len(factoid_results)} latency_s={elapsed}")
    return output


# ---- DuckDB query ----

def _query_impl(sql: str) -> dict:
    """Execute SQL against brain DuckDB cache.

    CLI: query
    MCP: query

    Args:
        sql: SQL query to execute against brain tables (facts, opinions, questions, aspirations, links, catalog, factoids)
    """
    import duckdb

    assert sql, "SQL query required"

    # Use persistent DuckDB if available, otherwise read TSVs directly
    if DB_PATH.exists():
        db = duckdb.connect(str(DB_PATH), read_only=True)
    else:
        db = duckdb.connect()
        tsv_names = ["facts", "opinions", "questions", "aspirations", "links", "catalog", "factoids"]
        for name in tsv_names:
            tsv_path = BRAIN_DIR / f"{name}.tsv"
            if tsv_path.exists():
                db.execute(f"""
                    CREATE VIEW {name} AS
                    SELECT * FROM read_csv('{tsv_path}', delim='\t', header=true, auto_detect=true)
                """)

    t0 = time.time()
    result = db.execute(sql)
    columns = [desc[0] for desc in result.description]
    rows = [dict(zip(columns, row)) for row in result.fetchall()]
    elapsed = round(time.time() - t0, 3)
    db.close()

    output = {
        "sql": sql,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "elapsed_seconds": elapsed,
    }

    _log("INFO", "query", f"DuckDB query: {sql[:50]}",
         detail=f"rows={len(rows)}",
         metrics=f"latency_s={elapsed}")
    return output


# =============================================================================
# CLI INTERFACE (argparse)
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Recall — unified search across brain tables")
    parser.add_argument("-V", "--version", action="version", version="1.0.0")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # mcp-stdio
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")

    # CLI for _recall_impl
    p_recall = subparsers.add_parser("recall", help="Unified search across brain")
    p_recall.add_argument("query", nargs="?", help="Natural language query")
    p_recall.add_argument("-n", "--limit", type=int, default=10, help="Max results")
    p_recall.add_argument("-j", "--json", dest="as_json", action="store_true", help="JSON output")

    # CLI for _query_impl
    p_query = subparsers.add_parser("query", help="Execute SQL against brain DuckDB")
    p_query.add_argument("sql", nargs="?", help="SQL query")
    p_query.add_argument("-j", "--json", dest="as_json", action="store_true", help="JSON output")

    args = parser.parse_args()

    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "recall":
            query = args.query
            if not query and not sys.stdin.isatty():
                query = sys.stdin.read().strip()
            assert query, "query required (positional argument or stdin)"
            result = _recall_impl(query, args.limit)

            if args.as_json:
                print(json.dumps(result, indent=2))
            else:
                for r in result["results"]:
                    score = r["score"]
                    method = r["method"][:3].upper()
                    table = r["table"]
                    key = r["key"]
                    content = r["content"][:120].replace("\n", " ")
                    print(f"{score:.3f}  {method:3s}  [{table}] {key}: {content}")
                c = result["counts"]
                print(f"\n{c['returned']} results from "
                      f"{c['semantic']} semantic + {c['keyword']} keyword + {c['factoids']} factoids "
                      f"({result['elapsed_seconds']}s)")
        elif args.command == "query":
            sql = args.sql
            if not sql and not sys.stdin.isatty():
                sql = sys.stdin.read().strip()
            assert sql, "SQL query required (positional argument or stdin)"
            result = _query_impl(sql)

            if args.as_json:
                print(json.dumps(result, indent=2, default=str))
            else:
                if result["rows"]:
                    cols = result["columns"]
                    print("\t".join(cols))
                    for row in result["rows"]:
                        print("\t".join(str(row.get(c, "")) for c in cols))
                print(f"\n{result['row_count']} rows ({result['elapsed_seconds']}s)")
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
# FASTMCP SERVER (lazy — only loaded when mcp-stdio is invoked)
# =============================================================================
def _run_mcp():
    from fastmcp import FastMCP

    mcp = FastMCP("recall")

    # MCP for _recall_impl
    @mcp.tool()
    def recall(query: str, limit: int = 10) -> str:
        """Unified search across brain — semantic + keyword + factoids, merged and ranked.

        Args:
            query: Natural language query or search terms
            limit: Maximum results to return
        """
        return json.dumps(_recall_impl(query, limit), indent=2)

    # MCP for _query_impl
    @mcp.tool()
    def query(sql: str) -> str:
        """Execute SQL against brain DuckDB cache. Tables: facts, opinions, questions, aspirations, links, catalog, factoids.

        Args:
            sql: SQL query to execute against brain tables
        """
        return json.dumps(_query_impl(sql), indent=2, default=str)

    print("recall MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
