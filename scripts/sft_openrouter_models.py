#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.25.0", "fastmcp"]
# ///
"""Query OpenRouter model catalog - list, search, compare, estimate costs.

Model selection based on cost, capability, and speed.
Returns structured JSON optimized for programmatic consumption.

Usage:
    sft_models.py list [--provider X] [--cap Y]
    sft_models.py search <query>
    sft_models.py compare <model1> <model2>
    sft_models.py cost <model> <input_k> <output_k>
    sft_models.py recommend <use_case>
    sft_models.py mcp-stdio
"""

import argparse
import json
import os
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
EXPOSED = ["list", "search", "compare", "cost", "recommend", "categories"]

CONFIG = {
    "version": "1.0.0",
    "api_url": "https://openrouter.ai/api/v1/models",
    "cache_file": "/tmp/openrouter_models_cache.json",
    "cache_ttl_seconds": 3600,
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def _fetch_models(
    use_cache: bool = True,
    category: str | None = None,
    supported_parameters: str | None = None,
) -> tuple[list[dict], dict]:
    """Fetch models from OpenRouter API. CLI/MCP internal."""
    import httpx
    
    start_ms = time.time() * 1000
    
    # Check cache (only if no query params)
    if use_cache and not category and not supported_parameters and Path(CONFIG["cache_file"]).exists():
        try:
            cache_age = time.time() - Path(CONFIG["cache_file"]).stat().st_mtime
            if cache_age < CONFIG["cache_ttl_seconds"]:
                with open(CONFIG["cache_file"]) as f:
                    models = json.load(f)
                latency_ms = time.time() * 1000 - start_ms
                return models, {"status": "success", "source": "cache", "latency_ms": round(latency_ms, 2)}
        except Exception:
            pass
    
    # Build query params
    params = {}
    if category:
        params["category"] = category
    if supported_parameters:
        params["supported_parameters"] = supported_parameters
    
    # Fetch fresh
    try:
        resp = httpx.get(CONFIG["api_url"], params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        
        # Cache only full list
        if not category and not supported_parameters:
            with open(CONFIG["cache_file"], "w") as f:
                json.dump(models, f)
        
        latency_ms = time.time() * 1000 - start_ms
        return models, {"status": "success", "source": "api", "count": len(models), "latency_ms": round(latency_ms, 2)}
    except Exception as e:
        latency_ms = time.time() * 1000 - start_ms
        return [], {"status": "error", "error": str(e), "latency_ms": round(latency_ms, 2)}


def _parse_price(price_str: str) -> float:
    """Parse price string to float."""
    try:
        return float(price_str)
    except (ValueError, TypeError):
        return 0.0


def _list_impl(
    provider: str | None = None,
    capability: str | None = None,
    category: str | None = None,
    supported_parameters: str | None = None,
    sort_by: str = "cost",
    limit: int = 20,
    show_free: bool = False,
) -> tuple[list[dict], dict]:
    """List models with filters. CLI: list, MCP: list."""
    start_ms = time.time() * 1000
    
    # Use API filtering for category and supported_parameters
    models, fetch_metrics = _fetch_models(
        category=category,
        supported_parameters=supported_parameters,
    )
    
    if fetch_metrics["status"] != "success":
        return [], fetch_metrics
    
    # Client-side filtering for provider, capability, free
    filtered = []
    for m in models:
        model_id = m.get("id", "")
        
        if provider and not model_id.lower().startswith(provider.lower()):
            continue
        
        if capability:
            name = m.get("name", "").lower()
            desc = m.get("description", "").lower()
            if capability.lower() not in name and capability.lower() not in desc:
                continue
        
        pricing = m.get("pricing", {})
        prompt_price = _parse_price(pricing.get("prompt", "0"))
        completion_price = _parse_price(pricing.get("completion", "0"))
        
        if not show_free and prompt_price == 0 and completion_price == 0:
            continue
        
        filtered.append({
            "id": model_id,
            "name": m.get("name", ""),
            "context": m.get("context_length", 0),
            "prompt_price": prompt_price,
            "completion_price": completion_price,
            "modality": m.get("architecture", {}).get("modality", "text->text"),
            "category": m.get("category"),
        })
    
    # Sort
    if sort_by == "cost":
        filtered.sort(key=lambda x: x["prompt_price"] + x["completion_price"])
    elif sort_by == "context":
        filtered.sort(key=lambda x: x["context"], reverse=True)
    elif sort_by == "name":
        filtered.sort(key=lambda x: x["name"])
    
    filtered = filtered[:limit]
    
    latency_ms = time.time() * 1000 - start_ms
    metrics = {
        "status": "success",
        "count": len(filtered),
        "total_available": len(models),
        "latency_ms": round(latency_ms, 2),
    }
    
    return filtered, metrics


def _search_impl(query: str, limit: int = 15) -> tuple[list[dict], dict]:
    """Search models. CLI: search, MCP: search."""
    start_ms = time.time() * 1000
    
    models, fetch_metrics = _fetch_models()
    
    if fetch_metrics["status"] != "success":
        return [], fetch_metrics
    
    query_lower = query.lower()
    matches = []
    
    for m in models:
        name = m.get("name", "").lower()
        desc = m.get("description", "").lower()
        model_id = m.get("id", "").lower()
        
        score = 0
        if query_lower in model_id:
            score += 10
        if query_lower in name:
            score += 5
        if query_lower in desc:
            score += 1
        
        if score > 0:
            pricing = m.get("pricing", {})
            matches.append({
                "id": m.get("id"),
                "name": m.get("name"),
                "score": score,
                "context": m.get("context_length", 0),
                "prompt_price": _parse_price(pricing.get("prompt", "0")),
                "completion_price": _parse_price(pricing.get("completion", "0")),
                "description": m.get("description", "")[:200],
            })
    
    matches.sort(key=lambda x: (-x["score"], x["prompt_price"]))
    matches = matches[:limit]
    
    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "count": len(matches), "latency_ms": round(latency_ms, 2)}
    
    return matches, metrics


def _compare_impl(*model_ids: str) -> tuple[dict, dict]:
    """Compare models. CLI: compare, MCP: compare."""
    start_ms = time.time() * 1000
    
    models, fetch_metrics = _fetch_models()
    
    if fetch_metrics["status"] != "success":
        return {}, fetch_metrics
    
    model_map = {m["id"]: m for m in models}
    
    comparison = {"models": [], "costs": {}}
    
    for mid in model_ids:
        if mid in model_map:
            m = model_map[mid]
            pricing = m.get("pricing", {})
            prompt = _parse_price(pricing.get("prompt", "0"))
            completion = _parse_price(pricing.get("completion", "0"))
            
            comparison["models"].append({
                "id": mid,
                "name": m.get("name", ""),
                "context": m.get("context_length", 0),
                "prompt_price": prompt,
                "completion_price": completion,
                "modality": m.get("architecture", {}).get("modality", "?"),
            })
            
            # Cost for 10k in / 2k out
            cost_10k = prompt * 10000 + completion * 2000
            comparison["costs"][mid] = round(cost_10k, 6)
        else:
            comparison["models"].append({"id": mid, "error": "NOT FOUND"})
    
    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}
    
    return comparison, metrics


def _cost_impl(model_id: str, input_k: float, output_k: float) -> tuple[dict, dict]:
    """Estimate cost. CLI: cost, MCP: cost."""
    start_ms = time.time() * 1000
    
    models, fetch_metrics = _fetch_models()
    
    if fetch_metrics["status"] != "success":
        return {}, fetch_metrics
    
    model_map = {m["id"]: m for m in models}
    
    # Try partial match if exact not found
    if model_id not in model_map:
        matches = [m for m in models if model_id.lower() in m["id"].lower()]
        if matches:
            model_id = matches[0]["id"]
        else:
            return {"error": f"Model '{model_id}' not found"}, {"status": "error", "latency_ms": round(time.time() * 1000 - start_ms, 2)}
    
    m = model_map[model_id]
    pricing = m.get("pricing", {})
    
    inp_price = _parse_price(pricing.get("prompt", "0"))
    out_price = _parse_price(pricing.get("completion", "0"))
    
    inp_cost = inp_price * input_k * 1000
    out_cost = out_price * output_k * 1000
    total = inp_cost + out_cost
    
    result = {
        "model": model_id,
        "input_tokens": input_k * 1000,
        "output_tokens": output_k * 1000,
        "input_cost": round(inp_cost, 6),
        "output_cost": round(out_cost, 6),
        "total_cost": round(total, 6),
        "input_price_per_million": inp_price * 1_000_000,
        "output_price_per_million": out_price * 1_000_000,
    }
    
    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "latency_ms": round(latency_ms, 2)}
    
    return result, metrics


def _categories_impl() -> tuple[list[str], dict]:
    """List available category enums. CLI: categories, MCP: categories."""
    # Categories are filter parameters in OpenRouter API, not returned in model data
    # Current category list: https://openrouter.ai/docs/api/api-reference/models/get-models
    # Look for "category" query parameter with enum values in the API documentation
    known_categories = [
        "api",
        "chat", 
        "code",
        "multimodal",
        "reasoning",
        "vision",
        "embedding",
        "moderation",
        "audio",
        "image",
        "video",
        "tool-use"
    ]
    
    metrics = {"status": "success", "count": len(known_categories), "source": "documentation"}
    
    return known_categories, metrics


def _recommend_impl(use_case: str) -> tuple[list[dict], dict]:
    """Get recommendations. CLI: recommend, MCP: recommend."""
    start_ms = time.time() * 1000
    
    models, fetch_metrics = _fetch_models()
    
    if fetch_metrics["status"] != "success":
        return [], fetch_metrics
    
    cases = {
        "code": ["code", "coder", "programming", "developer"],
        "fast": ["flash", "fast", "turbo", "instant"],
        "reasoning": ["thinking", "reason", "o1", "o3", "r1"],
        "cheap": [],
        "vision": ["vision", "image", "multimodal"],
        "long_context": [],
    }
    
    use_case_lower = use_case.lower().replace("_", " ")
    keywords = cases.get(use_case_lower, [use_case_lower])
    
    matches = []
    for m in models:
        name = m.get("name", "").lower()
        desc = m.get("description", "").lower()
        model_id = m.get("id", "").lower()
        
        pricing = m.get("pricing", {})
        prompt_price = _parse_price(pricing.get("prompt", "0"))
        completion_price = _parse_price(pricing.get("completion", "0"))
        
        if prompt_price == 0 and completion_price == 0:
            continue
        
        score = 0
        for kw in keywords:
            if kw in model_id or kw in name or kw in desc:
                score += 1
        
        matches.append({
            "id": m.get("id"),
            "name": m.get("name"),
            "score": score,
            "context": m.get("context_length", 0),
            "prompt_price": prompt_price,
            "completion_price": completion_price,
        })
    
    if use_case_lower == "cheap":
        matches.sort(key=lambda x: x["prompt_price"] + x["completion_price"])
    elif use_case_lower == "long_context":
        matches.sort(key=lambda x: -x["context"])
    else:
        matches.sort(key=lambda x: (-x["score"], x["prompt_price"]))
    
    matches = matches[:10]
    
    latency_ms = time.time() * 1000 - start_ms
    metrics = {"status": "success", "count": len(matches), "latency_ms": round(latency_ms, 2)}
    
    return matches, metrics


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Query OpenRouter model catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sft_models.py list --provider google --limit 10
  sft_models.py search "fast coding"
  sft_models.py compare google/gemini-2.5-pro xai/grok-4-1-fast
  sft_models.py cost google/gemini-2.5-pro 10 2
  sft_models.py recommend code
        """,
    )
    
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {CONFIG['version']}",
    )
    
    sub = parser.add_subparsers(dest="command", help="Commands")
    
    # list
    p_list = sub.add_parser("list", help="List models")
    p_list.add_argument("-p", "--provider", help="Filter by provider")
    p_list.add_argument("-c", "--cap", help="Filter by capability")
    p_list.add_argument("-g", "--category", help="Filter by category (api, multimodal, reasoning, etc)")
    p_list.add_argument("-s", "--sort", default="cost", choices=["cost", "context", "name"])
    p_list.add_argument("-n", "--limit", type=int, default=20)
    p_list.add_argument("-f", "--free", action="store_true", help="Include free models")
    p_list.add_argument("-P", "--params", help="Filter by supported parameter (e.g., tools, json_schema)")
    
    # search
    p_search = sub.add_parser("search", help="Search models")
    p_search.add_argument("query", nargs="?", help="Search query")
    p_search.add_argument("-n", "--limit", type=int, default=15)
    
    # compare
    p_compare = sub.add_parser("compare", help="Compare models")
    p_compare.add_argument("models", nargs="+", help="Model IDs")
    
    # cost
    p_cost = sub.add_parser("cost", help="Estimate cost")
    p_cost.add_argument("model", help="Model ID")
    p_cost.add_argument("input_k", type=float, help="Input tokens (thousands)")
    p_cost.add_argument("output_k", type=float, help="Output tokens (thousands)")
    
    # recommend
    p_rec = sub.add_parser("recommend", help="Get recommendations")
    p_rec.add_argument("use_case", help="code, fast, reasoning, cheap, vision, long_context")
    
    # categories
    sub.add_parser("categories", help="List available category enums")
    
    # mcp-stdio
    sub.add_parser("mcp-stdio", help="Run as MCP server")
    
    args = parser.parse_args()
    
    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "list":
            result, metrics = _list_impl(
                provider=args.provider,
                capability=args.cap,
                category=args.category,
                supported_parameters=args.params,
                sort_by=args.sort,
                limit=args.limit,
                show_free=args.free,
            )
            print(json.dumps({"models": result, "metrics": metrics}, indent=2))
            _log("INFO", "list", f"count={len(result)}", metrics=json.dumps(metrics))
        elif args.command == "search":
            query = args.query
            if not query and not sys.stdin.isatty():
                query = sys.stdin.read().strip()
            assert query, "query required (positional argument or stdin)"
            result, metrics = _search_impl(query, args.limit)
            print(json.dumps({"models": result, "metrics": metrics}, indent=2))
            _log("INFO", "search", f"query={query}", metrics=json.dumps(metrics))
        elif args.command == "compare":
            result, metrics = _compare_impl(*args.models)
            print(json.dumps({"comparison": result, "metrics": metrics}, indent=2))
            _log("INFO", "compare", f"models={','.join(args.models)}", metrics=json.dumps(metrics))
        elif args.command == "cost":
            result, metrics = _cost_impl(args.model, args.input_k, args.output_k)
            print(json.dumps({"estimate": result, "metrics": metrics}, indent=2))
            _log("INFO", "cost", f"model={args.model}", metrics=json.dumps(metrics))
        elif args.command == "recommend":
            result, metrics = _recommend_impl(args.use_case)
            print(json.dumps({"recommendations": result, "metrics": metrics}, indent=2))
            _log("INFO", "recommend", f"use_case={args.use_case}", metrics=json.dumps(metrics))
        elif args.command == "categories":
            result, metrics = _categories_impl()
            print(json.dumps({"categories": result, "metrics": metrics}, indent=2))
            _log("INFO", "categories", f"count={len(result)}", metrics=json.dumps(metrics))
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
    mcp = FastMCP("models")
    
    @mcp.tool()
    def list(
        provider: str = "",
        capability: str = "",
        category: str = "",
        supported_params: str = "",
        sort_by: str = "cost",
        limit: int = 20,
    ) -> str:
        """List OpenRouter models with optional filters.
        
        Args:
            provider: Filter by provider (e.g., "google", "openai")
            capability: Filter by capability keyword
            category: Filter by category (api, multimodal, reasoning, etc)
            supported_params: Filter by supported parameter (e.g., "tools", "json_schema")
            sort_by: Sort by "cost", "context", or "name"
            limit: Maximum results to return
        
        Returns:
            JSON with model list and metadata
        """
        result, metrics = _list_impl(
            provider=provider if provider else None,
            capability=capability if capability else None,
            category=category if category else None,
            supported_parameters=supported_params if supported_params else None,
            sort_by=sort_by,
            limit=limit,
        )
        return json.dumps({"models": result, "metrics": metrics}, indent=2)
    
    @mcp.tool()
    def search(query: str, limit: int = 15) -> str:
        """Search models by name/description.
        
        Args:
            query: Search query
            limit: Maximum results
        
        Returns:
            JSON with matching models
        """
        result, metrics = _search_impl(query, limit)
        return json.dumps({"models": result, "metrics": metrics}, indent=2)
    
    @mcp.tool()
    def compare(model_ids: str) -> str:
        """Compare specific models.
        
        Args:
            model_ids: Comma-separated model IDs
        
        Returns:
            JSON with comparison data
        """
        ids = [m.strip() for m in model_ids.split(",")]
        result, metrics = _compare_impl(*ids)
        return json.dumps({"comparison": result, "metrics": metrics}, indent=2)
    
    @mcp.tool()
    def cost(model_id: str, input_tokens: int, output_tokens: int) -> str:
        """Estimate cost for token usage.
        
        Args:
            model_id: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            JSON with cost breakdown
        """
        result, metrics = _cost_impl(model_id, input_tokens / 1000, output_tokens / 1000)
        return json.dumps({"estimate": result, "metrics": metrics}, indent=2)
    
    @mcp.tool()
    def recommend(use_case: str) -> str:
        """Get model recommendations for a use case.
        
        Args:
            use_case: "code", "fast", "reasoning", "cheap", "vision", "long_context"
        
        Returns:
            JSON with recommended models
        """
        result, metrics = _recommend_impl(use_case)
        return json.dumps({"recommendations": result, "metrics": metrics}, indent=2)
    
    @mcp.tool()
    def categories() -> str:
        """List available category enums from OpenRouter.
        
        Args:
            (no arguments required)
        
        Returns:
            JSON with list of available category values
        """
        result, metrics = _categories_impl()
        return json.dumps({"categories": result, "metrics": metrics}, indent=2)
    
    print("Models MCP server starting...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
