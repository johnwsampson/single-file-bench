# Single File Bench (SFB)

**Standing on the shoulders of giants.** This project extends the groundbreaking work of [disler](https://github.com/disler) (IndyDevDan on [YouTube](https://www.youtube.com/@indydevdan)), who pioneered the Single File Agents concept, and [Simon Willison](https://simonwillison.net), who demonstrated that [PEP 723 + `uv` enables single-file Python scripts with embedded dependencies](https://simonwillison.net/2024/Aug/20/uv-unified-python-packaging/).

> *"What if we could pack single purpose, powerful AI Agents into a single python file?"* — disler

## How SFB Extends the Original Vision

**disler's SFA established the foundation:**
- PEP 723 inline dependencies (`# /// script` blocks)
- `uv run --script` for zero-setup execution
- Single-purpose agents that "do one thing and one thing only"
- LLM function calling with Pydantic models

**SFB builds upon this foundation:**
- **Four architectural roles** (Tools, Servers, Agents, Daemons) organized by function
- **Triple interface** — Every script exposes CLI + Import + MCP (Model Context Protocol)
- **Structured TSV logging** — Queryable observability with standard unix tools
- **Negative space programming** — Assert contracts, fail loud, zero defensive coding
- **Automated compliance** — `sft_check.py` verifies spec adherence
- **Unix-native** — First-class stdin/stdout piping support

Self-contained Python CLI tools. Each script exposes three interfaces from one file:
- **CLI** — `./scripts/sft_web.py search "query"`
- **Import** — `from sft_web import _search_impl`
- **MCP** — `./scripts/sft_web.py mcp-stdio` (FastMCP over stdio)

## Quick Start

```bash
# Run any script directly (uv resolves PEP 723 inline dependencies)
./scripts/sft_web.py search "climate change"

# Pipe between tools
echo "climate change" | ./scripts/sft_web.py search | jq '.results[].url'

# Run as MCP server for agent consumption
./scripts/sft_web.py mcp-stdio
```

**Requirements:** [uv](https://github.com/astral-sh/uv) (Astral). No pip, no virtualenv, no requirements.txt.

## Architecture

Each script is fully self-contained via [PEP 723](https://peps.python.org/pep-0723/) inline metadata. Dependencies are declared in the script header and resolved by `uv` at runtime. No shared environment, no dependency conflicts between scripts.

Scripts follow a mandatory structure: PEP 723 header, logging block, configuration, core functions, CLI interface, FastMCP server. See [SPEC_SFA.md](SPEC_SFA.md) for the full specification.

## Key Design Principles

1. **One source of truth** — Core functions are the implementation. CLI and MCP are interfaces.
2. **Single-file scripts** — PEP 723 + `uv run --script`. One file = complete deployment artifact.
3. **Script organization** — Uniform structure, configuration at the top, EXPOSED list as inventory.
4. **Intuitive interfaces** — Unix piping, short+long flags, natural positional arguments.
5. **Negative space programming** — Assert contracts, fail loud. No defensive coding.
6. **Semantic versioning** — Every script versioned. Changes tracked in [CHANGELOG.md](CHANGELOG.md).

See [SPEC_SFA.md](SPEC_SFA.md) for the full 10-principle specification.

## Compliance

The `sfa_` prefix is a certification mark. Only scripts that pass all spec principles earn it.

```bash
# Check a single script
./scripts/sfa_check.py check scripts/sfa_web.py

# Check all SFA scripts
./scripts/sfa_check.py check scripts/sfa_*.py
```

## Scripts

### Tools (sft_)
| Script | MCP | Description |
|--------|-----|-------------|
| `sft_web.py` | `web` | Search (SearXNG), fetch URLs to markdown |
| `sft_check.py` | `check` | SFB compliance checker |
| `sft_scout.py` | `fs` | Code reconnaissance, search, python_view |
| `sft_file_read.py` | `fr` | Read-only file operations and inspection |
| `sft_file_write.py` | `fw` | Destructive file operations with backup system |
| `sft_duckdb.py` | `duckdb` | Natural language SQL queries with AI assistance |
| `sft_jq.py` | `jq` | Natural language JSON processing with jq |

### Loop Agents (sfl_)
| Script | MCP | Description |
|--------|-----|-------------|
| `sfl_test_gen.py` | `test_gen` | Iterative test generation until coverage threshold met |

## Development

```bash
# Lint and format
ruff check scripts/ --fix
ruff format scripts/

# Manual verification (no unit tests — scripts use runtime assertions)
./scripts/sfa_web.py search "test query"
tail -20 scripts/sfa_web_log.tsv
```

## Documentation

- [SPEC_SFA.md](SPEC_SFA.md) — Full specification (11 principles)
- [AGENTS.md](AGENTS.md) — Quick reference for agentic coding assistants
- [CHANGELOG.md](CHANGELOG.md) — DR/RFC change tracking
- [scripts/README.md](scripts/README.md) — Complete script index by bench type

## License

MIT License — This project extends [disler's Single File Agents](https://github.com/disler/single-file-agents) (MIT) and builds upon patterns demonstrated by [Simon Willison](https://simonwillison.net) (open source advocate).

See: [disler/single-file-agents LICENSE](https://github.com/disler/single-file-agents/blob/main/README.md#license)
