#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "httpx", "fastmcp", "duckdb", "pyarrow"]
# ///
"""Brain — FOQA knowledge store with semantic search and analytical cache.

FOQA tables: facts, opinions, questions, aspirations.
Plus: links (relationships), catalog (reference index), factoids (infrastructure data).

Architecture:
  brain/*.tsv          -> canonical store (FOQA + links/catalog/factoids)
  brain/mutations.tsv  -> audit trail
  .cache/brain.pkl     -> read cache (entries + embeddings)
  .cache/*.parquet     -> columnar cache (DuckDB-ready)
  .cache/brain.duckdb  -> analytical query engine

Usage:
    sft_brain.py inscribe "content" -t fact -k my-fact-key
    sft_brain.py update my-fact-key --content "revised content"
    sft_brain.py delete my-fact-key -t fact
    sft_brain.py search "pattern"
    sft_brain.py semantic "query"
    sft_brain.py build
    sft_brain.py stats
    echo "query" | sft_brain.py search
    sft_brain.py mcp-stdio
"""

import csv
import hashlib
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
EXPOSED = ["inscribe", "update", "delete", "factoid_set", "factoid_delete", "search", "semantic", "build", "stats"]

VERA_ROOT = Path(os.environ.get("VERA_ROOT", Path(__file__).parent.parent))
BRAIN_DIR = VERA_ROOT / "brain"
CACHE_DIR = VERA_ROOT / ".cache"
BRAIN_PKL = CACHE_DIR / "brain.pkl"
DB_PATH = CACHE_DIR / "brain.duckdb"

# FOQA table definitions — each table has its own schema
TABLES = {
    "fact": {"file": "facts.tsv", "cols": ["key", "fact", "source", "recorded"], "content_col": "fact", "meta_col": "source"},
    "opinion": {"file": "opinions.tsv", "cols": ["key", "opinion", "basis", "recorded"], "content_col": "opinion", "meta_col": "basis"},
    "question": {"file": "questions.tsv", "cols": ["key", "question", "context", "recorded"], "content_col": "question", "meta_col": "context"},
    "aspiration": {"file": "aspirations.tsv", "cols": ["key", "aspiration", "type", "recorded"], "content_col": "aspiration", "meta_col": "type"},
}
VALID_TABLES = set(TABLES.keys())

LINK_COLS = ["key", "from_table", "from_key", "to_table", "to_key"]
FACTOID_COLS = ["key", "value", "updated"]
CATALOG_COLS = ["key", "path", "description"]
MUTATION_COLS = ["timestamp", "action", "table", "key", "detail"]

LINKS_FILE = BRAIN_DIR / "links.tsv"
FACTOIDS_FILE = BRAIN_DIR / "factoids.tsv"
CATALOG_FILE = BRAIN_DIR / "catalog.tsv"
MUTATIONS_FILE = BRAIN_DIR / "mutations.tsv"

# Embedding model (OpenRouter API)
EMBED_MODEL = "openai/text-embedding-3-large"
EMBED_DIM = 3072
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
EMBED_BATCH_SIZE = 500

CONFIG = {
    "cache_stale_threshold_seconds": 3600,
}


# =============================================================================
# CORE FUNCTIONS (the actual logic — CLI and MCP both call these)
# =============================================================================

# ---- TSV I/O ----

def _clean_cols(row: dict) -> dict:
    """Strip # prefix from column names (header convention)."""
    return {k.lstrip("#"): v for k, v in row.items()}


def _read_table(table: str) -> list[dict]:
    """Read all rows from a FOQA table TSV file.

    CLI: (internal)
    MCP: (internal)
    """
    tdef = TABLES[table]
    path = BRAIN_DIR / tdef["file"]
    if not path.exists():
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [_clean_cols(r) for r in reader]


def _read_factoids() -> list[dict]:
    """Read all rows from factoids.tsv.

    CLI: (internal)
    MCP: (internal)
    """
    if not FACTOIDS_FILE.exists():
        return []
    with open(FACTOIDS_FILE, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [_clean_cols(r) for r in reader]


def _write_table(table: str, rows: list[dict]):
    """Write all rows to a FOQA table TSV file (full rewrite)."""
    tdef = TABLES[table]
    path = BRAIN_DIR / tdef["file"]
    cols = tdef["cols"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(cols)
        for row in rows:
            w.writerow([row.get(c, "") for c in cols])


def _append_row(table: str, row: dict):
    """Append a single row to a FOQA table TSV file."""
    tdef = TABLES[table]
    path = BRAIN_DIR / tdef["file"]
    cols = tdef["cols"]
    write_header = not path.exists() or path.stat().st_size == 0
    # Ensure file ends with newline before appending
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                with open(path, "a") as fa:
                    fa.write("\n")
    with open(path, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        if write_header:
            w.writerow(cols)
        w.writerow([row.get(c, "") for c in cols])


def _append_link(key: str, from_table: str, from_key: str, to_table: str, to_key: str):
    """Append a link to links.tsv."""
    write_header = not LINKS_FILE.exists() or LINKS_FILE.stat().st_size == 0
    with open(LINKS_FILE, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        if write_header:
            w.writerow(LINK_COLS)
        w.writerow([key, from_table, from_key, to_table, to_key])


def _log_mutation(action: str, table: str, key: str, detail_text: str = ""):
    """Append operation to mutations.tsv for audit trail."""
    try:
        row = [
            datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            action,
            table,
            key,
            detail_text[:500].replace("\t", " ").replace("\n", "\\n"),
        ]
        write_header = not MUTATIONS_FILE.exists() or MUTATIONS_FILE.stat().st_size == 0
        with open(MUTATIONS_FILE, "a", newline="") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            if write_header:
                w.writerow(MUTATION_COLS)
            w.writerow(row)
    except Exception:
        pass


def _load_all_entries() -> list[dict]:
    """Load all entries from all FOQA tables."""
    entries = []
    for table in TABLES:
        for row in _read_table(table):
            row["_table"] = table
            row["_content"] = row.get(TABLES[table]["content_col"], "")
            entries.append(row)
    return entries


def _generate_key(content: str) -> str:
    """Generate a semantic key from content."""
    slug = re.sub(r"[^a-z0-9]+", "-", content[:60].lower()).strip("-")
    slug = slug[:40]
    h = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{slug}-{h}" if slug else h


# ---- Embedding via OpenRouter API ----

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


def _embed_api(texts: list[str]) -> list:
    """Call OpenRouter embeddings API. Returns list of numpy arrays."""
    import httpx
    import numpy as np

    key = _get_api_key()
    assert key, "OPENROUTER_API_KEY not found in env or scripts/.secrets"
    resp = httpx.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    data.sort(key=lambda x: x["index"])
    return [np.array(d["embedding"], dtype=np.float32) for d in data]


# ---- Mutation operations ----

def _inscribe_impl(content: str, table: str, key: str = "", meta: str = "",
                   links: str = "", embed: bool = True) -> dict:
    """Add a new entry to the appropriate FOQA table.

    CLI: inscribe
    MCP: inscribe

    Args:
        content: The content to inscribe
        table: FOQA table (fact, opinion, question, aspiration)
        key: Semantic key (auto-generated if empty)
        meta: Table-specific metadata (source/basis/context/type)
        links: Comma-separated link descriptions as key:from_table:from_key:to_table:to_key
        embed: Whether to rebuild cache with embeddings after inscription
    """
    assert table in VALID_TABLES, f"table must be one of {sorted(VALID_TABLES)} (got '{table}')"

    if not key:
        key = _generate_key(content)

    existing = _read_table(table)
    assert not any(r["key"] == key for r in existing), f"Key '{key}' already exists in {table}. Use update instead."

    tdef = TABLES[table]
    clean_content = content.replace("\t", " ").replace("\n", "\\n")

    row = {
        "key": key,
        tdef["content_col"]: clean_content,
        tdef["meta_col"]: meta,
        "recorded": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }

    _append_row(table, row)
    _log_mutation("inscribe", table, key, clean_content[:200])
    _log("INFO", "inscribe", f"Inscribed {key} to {table}", detail=f"table={table} key={key}")

    if embed:
        _build_impl(embed=True)

    return {"status": "ok", "key": key, "table": table, "action": "inscribed"}


def _update_impl(key: str, content: str = "", meta: str = "",
                 table: str = "", new_key: str = "") -> dict:
    """Update an existing entry. Searches all FOQA tables.

    CLI: update
    MCP: update

    Args:
        key: The key of the entry to update
        content: New content (empty to keep existing)
        meta: New metadata value (empty to keep existing)
        table: Move to a different table (empty to keep existing)
        new_key: Rename the key (empty to keep existing)
    """
    for src_table in TABLES:
        rows = _read_table(src_table)
        for i, row in enumerate(rows):
            if row["key"] == key:
                tdef = TABLES[src_table]
                old_content = row.get(tdef["content_col"], "")[:100]

                if content:
                    row[tdef["content_col"]] = content.replace("\t", " ").replace("\n", "\\n")
                if meta:
                    row[tdef["meta_col"]] = meta
                if new_key:
                    row["key"] = new_key

                # Table change = move between files
                if table and table != src_table and table in VALID_TABLES:
                    dest_tdef = TABLES[table]
                    new_row = {
                        "key": row["key"],
                        dest_tdef["content_col"]: row.get(tdef["content_col"], ""),
                        dest_tdef["meta_col"]: row.get(tdef["meta_col"], ""),
                        "recorded": row.get("recorded", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
                    }
                    rows.pop(i)
                    _write_table(src_table, rows)
                    _append_row(table, new_row)
                    detail = f"moved {src_table}->{table} was: {old_content}"
                else:
                    _write_table(src_table, rows)
                    table = src_table
                    detail = f"was: {old_content}"

                _build_impl(embed=True)
                _log_mutation("update", table, row["key"], detail)
                _log("INFO", "update", f"Updated {row['key']} in {table}", detail=f"table={table} key={row['key']}")
                return {"status": "ok", "key": row["key"], "table": table, "action": "updated"}

    return {"status": "error", "message": f"Key '{key}' not found in any FOQA table."}


def _delete_impl(key: str, table: str = "") -> dict:
    """Remove an entry from a FOQA table.

    CLI: delete
    MCP: delete

    Args:
        key: The key of the entry to delete
        table: Specific table to delete from (searches all if empty)
    """
    search_tables = [table] if table and table in VALID_TABLES else list(TABLES.keys())

    for tbl in search_tables:
        rows = _read_table(tbl)
        new_rows = [r for r in rows if r["key"] != key]
        if len(new_rows) < len(rows):
            _write_table(tbl, new_rows)
            _build_impl(embed=False)
            _log_mutation("delete", tbl, key)
            _log("INFO", "delete", f"Deleted {key} from {tbl}", detail=f"table={tbl} key={key}")
            return {"status": "ok", "key": key, "table": tbl, "action": "deleted"}

    return {"status": "error", "message": f"Key '{key}' not found."}


def _write_factoids(rows: list[dict]):
    """Write all rows to factoids.tsv (full rewrite). Preserves #key header."""
    header = ["#key", "value", "updated"]
    with open(FACTOIDS_FILE, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(header)
        for row in rows:
            w.writerow([row.get(c, "") for c in FACTOID_COLS])


def _factoid_set_impl(key: str, value: str) -> dict:
    """Set a factoid (insert or update). Rebuilds cache after write.

    CLI: factoid-set
    MCP: factoid_set

    Args:
        key: Factoid key (e.g. "oscar:ip", "lan:subnet")
        value: Factoid value
    """
    assert key, "key required"
    assert value, "value required"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = _read_factoids()
    action = "updated"
    found = False
    for row in rows:
        if row["key"] == key:
            row["value"] = value.replace("\t", " ")
            row["updated"] = today
            found = True
            break
    if not found:
        rows.append({"key": key, "value": value.replace("\t", " "), "updated": today})
        action = "created"
    _write_factoids(rows)
    _build_impl(embed=False)
    _log_mutation(action, "factoid", key, value[:200])
    _log("INFO", "factoid_set", f"{action} factoid {key}", detail=f"key={key} value={value[:100]}")
    return {"status": "ok", "key": key, "action": action}


def _factoid_delete_impl(key: str) -> dict:
    """Delete a factoid by key. Rebuilds cache after deletion.

    CLI: factoid-delete
    MCP: factoid_delete

    Args:
        key: Factoid key to delete
    """
    assert key, "key required"
    rows = _read_factoids()
    new_rows = [r for r in rows if r["key"] != key]
    assert len(new_rows) < len(rows), f"Factoid '{key}' not found."
    _write_factoids(new_rows)
    _build_impl(embed=False)
    _log_mutation("delete", "factoid", key)
    _log("INFO", "factoid_delete", f"Deleted factoid {key}", detail=f"key={key}")
    return {"status": "ok", "key": key, "action": "deleted"}


# ---- Search operations ----

def _search_impl(pattern: str, table: str = "", limit: int = 20) -> list[dict]:
    """Regex search across FOQA entries and factoids.

    CLI: search
    MCP: search

    Args:
        pattern: Regex pattern to search for
        table: Limit to specific table (empty for all, "factoid" for factoids only)
        limit: Maximum results to return
    """
    assert pattern, "search pattern required"
    regex = re.compile(pattern, re.IGNORECASE)
    results = []

    # Search FOQA tables (unless specifically asking for factoids only)
    if table != "factoid":
        search_tables = [table] if table and table in VALID_TABLES else list(TABLES.keys())
        for tbl in search_tables:
            tdef = TABLES[tbl]
            for row in _read_table(tbl):
                searchable = "\t".join(str(v) for v in row.values())
                if regex.search(searchable):
                    results.append({
                        "key": row["key"],
                        "table": tbl,
                        "content": row.get(tdef["content_col"], "")[:200],
                        "meta": row.get(tdef["meta_col"], ""),
                    })
                    if len(results) >= limit:
                        _log("INFO", "search", f"Found {len(results)} results for '{pattern}'",
                             detail=f"pattern={pattern} table={table or 'all'}")
                        return results

    # Search factoids (when table is "factoid" or searching all)
    if table in ("factoid", ""):
        for row in _read_factoids():
            searchable = "\t".join(str(v) for v in row.values())
            if regex.search(searchable):
                results.append({
                    "key": row.get("key", ""),
                    "table": "factoid",
                    "content": row.get("value", "")[:200],
                    "meta": row.get("updated", ""),
                })
                if len(results) >= limit:
                    _log("INFO", "search", f"Found {len(results)} results for '{pattern}'",
                         detail=f"pattern={pattern} table={table or 'all'}")
                    return results

    _log("INFO", "search", f"Found {len(results)} results for '{pattern}'",
         detail=f"pattern={pattern} table={table or 'all'}")
    return results


def _semantic_impl(query: str, limit: int = 10) -> list[dict]:
    """Embedding similarity search across FOQA entries.

    CLI: semantic
    MCP: semantic

    Args:
        query: Natural language query
        limit: Maximum results to return
    """
    import numpy as np

    assert query, "query required"

    cache = _load_cache()
    if not cache.get("entries") or cache.get("embeddings") is None:
        cache = _build_impl(embed=True)

    if cache.get("embeddings") is None or not cache.get("entries"):
        return []

    query_emb = _embed_api([query])[0]
    entries = cache["entries"]
    embeddings = cache["embeddings"]

    # Vectorized cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    query_norm = np.linalg.norm(query_emb) + 1e-10
    sims = (embeddings @ query_emb) / (norms.squeeze() * query_norm)

    top_idx = np.argsort(sims)[::-1][:limit]

    results = []
    for idx in top_idx:
        entry = entries[idx]
        results.append({
            "key": entry["key"],
            "table": entry["_table"],
            "content": entry["_content"][:200],
            "similarity": round(float(sims[idx]), 4),
        })

    _log("INFO", "semantic", f"Semantic search for '{query[:30]}'",
         detail=f"query={query[:50]} results={len(results)}",
         metrics=f"limit={limit}")
    return results


# ---- Cache operations ----

def _load_cache() -> dict:
    """Load brain pickle cache."""
    if BRAIN_PKL.exists():
        with open(BRAIN_PKL, "rb") as f:
            return pickle.load(f)
    return {"entries": [], "embeddings": None, "built": None}


def _build_impl(embed: bool = True) -> dict:
    """Rebuild cache from FOQA TSV source of truth.

    CLI: build
    MCP: build

    Args:
        embed: Whether to generate embeddings (requires OPENROUTER_API_KEY)
    """
    import numpy as np

    t0 = time.time()
    entries = _load_all_entries()

    cache = {
        "entries": entries,
        "embeddings": None,
        "built": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(entries),
    }

    if embed and entries:
        old_cache = _load_cache()
        old_embeddings = {}
        if old_cache.get("entries") and old_cache.get("embeddings") is not None:
            for i, old_entry in enumerate(old_cache["entries"]):
                if i < len(old_cache["embeddings"]):
                    old_embeddings[old_entry.get("key", "")] = old_cache["embeddings"][i]

        new_texts = []
        new_indices = []
        embeddings = np.zeros((len(entries), EMBED_DIM), dtype=np.float32)

        for i, entry in enumerate(entries):
            old_emb = old_embeddings.get(entry["key"])
            if old_emb is not None and old_emb.shape[0] == EMBED_DIM:
                embeddings[i] = old_emb
            else:
                new_texts.append(entry["_content"])
                new_indices.append(i)

        if new_texts:
            reused = len(entries) - len(new_texts)
            print(f"  {reused} reused, {len(new_texts)} to embed via {EMBED_MODEL}", file=sys.stderr)
            for batch_start in range(0, len(new_texts), EMBED_BATCH_SIZE):
                batch_end = min(batch_start + EMBED_BATCH_SIZE, len(new_texts))
                chunk = new_texts[batch_start:batch_end]
                chunk_indices = new_indices[batch_start:batch_end]
                print(f"  embedding batch {batch_start + 1}-{batch_end}/{len(new_texts)}...",
                      file=sys.stderr, flush=True)
                chunk_embeds = _embed_api(chunk)
                for idx, emb in zip(chunk_indices, chunk_embeds):
                    embeddings[idx] = emb
        else:
            print(f"  all {len(entries)} embeddings reused from cache", file=sys.stderr)

        cache["embeddings"] = embeddings

    # Write pickle
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(BRAIN_PKL, "wb") as f:
        pickle.dump(cache, f)

    # Build parquet + DuckDB
    _build_analytical_cache()

    elapsed = time.time() - t0
    cache["build_time_ms"] = round(elapsed * 1000, 1)
    _log("INFO", "build", f"Cache rebuilt: {len(entries)} entries",
         metrics=f"entries={len(entries)} embed={embed} latency_ms={cache['build_time_ms']}")
    return cache


def _build_analytical_cache():
    """Rebuild parquet and DuckDB from TSV source of truth."""
    try:
        import duckdb

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if DB_PATH.exists():
            DB_PATH.unlink()

        db = duckdb.connect(str(DB_PATH))
        tsv_names = ["facts", "opinions", "questions", "aspirations", "links", "catalog", "factoids"]

        for name in tsv_names:
            tsv_path = BRAIN_DIR / f"{name}.tsv"
            if not tsv_path.exists():
                continue
            db.execute(f"""
                CREATE TABLE {name} AS
                SELECT * FROM read_csv('{tsv_path}', delim='\t', header=true, auto_detect=true)
            """)
            parquet_path = CACHE_DIR / f"{name}.parquet"
            db.execute(f"COPY {name} TO '{parquet_path}' (FORMAT PARQUET)")

        db.close()
    except Exception as e:
        _log("WARN", "build_analytical", f"Analytical cache build failed: {e}")


# ---- Stats ----

def _stats_impl() -> dict:
    """Get brain statistics.

    CLI: stats
    MCP: stats
    """
    entries = _load_all_entries()
    tables = {}
    for e in entries:
        t = e.get("_table", "?")
        tables[t] = tables.get(t, 0) + 1

    source_size_kb = 0
    for tdef in TABLES.values():
        path = BRAIN_DIR / tdef["file"]
        if path.exists():
            source_size_kb += path.stat().st_size / 1024

    links_count = 0
    if LINKS_FILE.exists():
        links_count = sum(1 for _ in open(LINKS_FILE)) - 1

    factoids_count = 0
    if FACTOIDS_FILE.exists():
        factoids_count = sum(1 for _ in open(FACTOIDS_FILE)) - 1

    cache_info = {}
    if BRAIN_PKL.exists():
        cache = _load_cache()
        cache_info = {
            "cache_exists": True,
            "cache_size_kb": round(BRAIN_PKL.stat().st_size / 1024, 1),
            "cache_built": cache.get("built", "?"),
            "has_embeddings": cache.get("embeddings") is not None,
        }

    result = {
        "total_entries": len(entries),
        "tables": dict(sorted(tables.items())),
        "links": links_count,
        "factoids": factoids_count,
        "source_size_kb": round(source_size_kb, 1),
        **cache_info,
    }
    _log("INFO", "stats", f"Brain stats: {len(entries)} entries", metrics=f"entries={len(entries)} links={links_count}")
    return result


# =============================================================================
# CLI INTERFACE (argparse)
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Brain — FOQA knowledge store with semantic search")
    parser.add_argument("-V", "--version", action="version", version="1.0.0")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # mcp-stdio
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")

    # CLI for _inscribe_impl
    p_ins = subparsers.add_parser("inscribe", help="Add a new entry to a FOQA table")
    p_ins.add_argument("content", nargs="?", help="Content to inscribe")
    p_ins.add_argument("-t", "--table", required=True, help="FOQA table: fact, opinion, question, aspiration")
    p_ins.add_argument("-k", "--key", default="", help="Semantic key (auto-generated if omitted)")
    p_ins.add_argument("-m", "--meta", default="", help="Table-specific metadata (source/basis/context/type)")
    p_ins.add_argument("-l", "--links", default="", help="Comma-separated link specs")
    p_ins.add_argument("-e", "--embed", action="store_true", default=True, help="Rebuild cache with embeddings")
    p_ins.add_argument("-E", "--no-embed", dest="embed", action="store_false", help="Skip embedding rebuild")

    # CLI for _update_impl
    p_upd = subparsers.add_parser("update", help="Update an existing entry")
    p_upd.add_argument("key", help="Key of entry to update")
    p_upd.add_argument("-c", "--content", default="", help="New content")
    p_upd.add_argument("-m", "--meta", default="", help="New metadata value")
    p_upd.add_argument("-t", "--table", default="", help="Move to a different table")
    p_upd.add_argument("-k", "--new-key", default="", help="Rename the key")

    # CLI for _delete_impl
    p_del = subparsers.add_parser("delete", help="Remove an entry")
    p_del.add_argument("key", help="Key of entry to delete")
    p_del.add_argument("-t", "--table", default="", help="Specific table (searches all if omitted)")

    # CLI for _factoid_set_impl
    p_fs = subparsers.add_parser("factoid-set", help="Set a factoid (insert or update)")
    p_fs.add_argument("key", help="Factoid key (e.g. oscar:ip)")
    p_fs.add_argument("value", help="Factoid value")

    # CLI for _factoid_delete_impl
    p_fd = subparsers.add_parser("factoid-delete", help="Delete a factoid by key")
    p_fd.add_argument("key", help="Factoid key to delete")

    # CLI for _search_impl
    p_srch = subparsers.add_parser("search", help="Text/regex search across entries")
    p_srch.add_argument("pattern", nargs="?", help="Regex search pattern")
    p_srch.add_argument("-t", "--table", default="", help="Limit to specific table")
    p_srch.add_argument("-n", "--limit", type=int, default=20, help="Max results")

    # CLI for _semantic_impl
    p_sem = subparsers.add_parser("semantic", help="Semantic similarity search")
    p_sem.add_argument("query", nargs="?", help="Natural language query")
    p_sem.add_argument("-n", "--limit", type=int, default=10, help="Max results")

    # CLI for _build_impl
    p_bld = subparsers.add_parser("build", help="Rebuild cache from TSV source of truth")
    p_bld.add_argument("-e", "--embed", action="store_true", default=True, help="Generate embeddings")
    p_bld.add_argument("-E", "--no-embed", dest="embed", action="store_false", help="Skip embeddings")

    # CLI for _stats_impl
    subparsers.add_parser("stats", help="Show brain statistics")

    args = parser.parse_args()

    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "inscribe":
            content = args.content
            if not content and not sys.stdin.isatty():
                content = sys.stdin.read().strip()
            assert content, "content required (positional argument or stdin)"
            result = _inscribe_impl(content, args.table, args.key, args.meta, args.links, args.embed)
            print(json.dumps(result, indent=2))
        elif args.command == "update":
            result = _update_impl(args.key, args.content, args.meta, args.table, args.new_key)
            print(json.dumps(result, indent=2))
        elif args.command == "delete":
            result = _delete_impl(args.key, args.table)
            print(json.dumps(result, indent=2))
        elif args.command == "factoid-set":
            result = _factoid_set_impl(args.key, args.value)
            print(json.dumps(result, indent=2))
        elif args.command == "factoid-delete":
            result = _factoid_delete_impl(args.key)
            print(json.dumps(result, indent=2))
        elif args.command == "search":
            pattern = args.pattern
            if not pattern and not sys.stdin.isatty():
                pattern = sys.stdin.read().strip()
            assert pattern, "search pattern required (positional argument or stdin)"
            results = _search_impl(pattern, args.table, args.limit)
            for r in results:
                print(f"[{r['table']}] {r['key']}: {r['content']}")
            print(f"\n{len(results)} results")
        elif args.command == "semantic":
            query = args.query
            if not query and not sys.stdin.isatty():
                query = sys.stdin.read().strip()
            assert query, "query required (positional argument or stdin)"
            results = _semantic_impl(query, args.limit)
            for r in results:
                print(f"{r['similarity']:.3f}  [{r['table']}] {r['key']}: {r['content']}")
            print(f"\n{len(results)} results")
        elif args.command == "build":
            cache = _build_impl(embed=args.embed)
            print(f"Cache rebuilt: {cache['entry_count']} entries, {cache.get('build_time_ms', '?')}ms")
        elif args.command == "stats":
            print(json.dumps(_stats_impl(), indent=2))
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

    mcp = FastMCP("brain")

    # MCP for _inscribe_impl
    @mcp.tool()
    def inscribe(content: str, table: str, key: str = "", meta: str = "",
                 links: str = "", embed: bool = True) -> str:
        """Add a new entry to a FOQA table. Rebuilds cache after write.

        Args:
            content: The content to inscribe
            table: FOQA table (fact, opinion, question, aspiration)
            key: Semantic key (auto-generated if empty)
            meta: Table-specific metadata (source/basis/context/type)
            links: Comma-separated link specs
            embed: Whether to rebuild cache with embeddings
        """
        return json.dumps(_inscribe_impl(content, table, key, meta, links, embed), indent=2)

    # MCP for _update_impl
    @mcp.tool()
    def update(key: str, content: str = "", meta: str = "",
               table: str = "", new_key: str = "") -> str:
        """Update an existing entry. Searches all FOQA tables.

        Args:
            key: The key of the entry to update
            content: New content (empty to keep existing)
            meta: New metadata value (empty to keep existing)
            table: Move to a different table (empty to keep existing)
            new_key: Rename the key (empty to keep existing)
        """
        return json.dumps(_update_impl(key, content, meta, table, new_key), indent=2)

    # MCP for _delete_impl
    @mcp.tool()
    def delete(key: str, table: str = "") -> str:
        """Remove an entry from a FOQA table.

        Args:
            key: The key of the entry to delete
            table: Specific table (searches all if empty)
        """
        return json.dumps(_delete_impl(key, table), indent=2)

    # MCP for _factoid_set_impl
    @mcp.tool()
    def factoid_set(key: str, value: str) -> str:
        """Set a factoid (insert or update). Rebuilds cache after write.

        Args:
            key: Factoid key (e.g. "oscar:ip", "lan:subnet")
            value: Factoid value
        """
        return json.dumps(_factoid_set_impl(key, value), indent=2)

    # MCP for _factoid_delete_impl
    @mcp.tool()
    def factoid_delete(key: str) -> str:
        """Delete a factoid by key. Rebuilds cache after deletion.

        Args:
            key: Factoid key to delete
        """
        return json.dumps(_factoid_delete_impl(key), indent=2)

    # MCP for _search_impl
    @mcp.tool()
    def search(pattern: str, table: str = "", limit: int = 20) -> str:
        """Text/regex search across FOQA entries.

        Args:
            pattern: Regex pattern to search for
            table: Limit to specific table (empty for all)
            limit: Maximum results to return
        """
        return json.dumps(_search_impl(pattern, table, limit), indent=2)

    # MCP for _semantic_impl
    @mcp.tool()
    def semantic(query: str, limit: int = 10) -> str:
        """Semantic similarity search using embeddings.

        Args:
            query: Natural language query
            limit: Maximum results to return
        """
        return json.dumps(_semantic_impl(query, limit), indent=2)

    # MCP for _build_impl
    @mcp.tool()
    def build(embed: bool = True) -> str:
        """Rebuild cache from TSV source of truth.

        Args:
            embed: Whether to generate embeddings (requires OPENROUTER_API_KEY)
        """
        cache = _build_impl(embed=embed)
        return json.dumps({"entry_count": cache["entry_count"], "build_time_ms": cache.get("build_time_ms")}, indent=2)

    # MCP for _stats_impl
    @mcp.tool()
    def stats() -> str:
        """Get brain statistics — entry counts, cache info, sizes.

        Args:
            (no arguments)
        """
        return json.dumps(_stats_impl(), indent=2)

    print("brain MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
