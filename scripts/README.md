# Single File Bench — Scripts Index

Self-contained Python CLI tools. Each script exposes three interfaces from one file: **CLI** + **Import** + **MCP**.

**Quick Links:** [Repository README](../README.md) | [Specification](../SPEC_SFA.md) | [Agent Guide](../AGENTS.md)

---

## Tools (sft_*)

General-purpose CLI utilities, MCP tools, unix-style commands. The workhorses of the bench.

| Script | MCP | Version | Description | Key Capabilities |
|--------|-----|---------|-------------|------------------|
| `sft_web.py` | `web` | 4.2.0 | Web operations via SearXNG | search, fetch, engines |
| `sft_check.py` | `check` | 1.4.0 | SFB compliance checker | validates 25 spec principles |
| `sft_scout.py` | `fs` | 1.1.0 | Code reconnaissance | search, find, python_view, count |
| `sft_file_read.py` | `fr` | 1.1.0 | Read-only file operations | read, head, tail, cat, tree, stats |
| `sft_file_write.py` | `fw` | 1.1.0 | Destructive file operations | write, create, cp, mv, rm + backups |
| `sft_duckdb.py` | `duckdb` | 1.0.0 | Natural language SQL queries | query, schema, AI-assisted |
| `sft_jq.py` | `jq` | 1.0.0 | Natural language JSON processing | generate, query, jq command builder |
| `sft_chrome.py` | — | 1.1.0 | Chrome browser automation | 22 functions, 2 profiles (john, sandbox) |
| `sft_clipboard.py` | — | 1.1.0 | Clipboard operations | read, write |
| `sft_x.py` | — | 2.1.0 | Perplexity AI search | search |

**Usage Examples:**
```bash
# Web search
./sft_web.py search "climate change" | jq '.results[].url'

# Compliance check
./sft_check.py check sft_*.py

# Natural language database query
./sft_duckdb.py query ./data.db "active users over 30"

# JSON processing
./sft_jq.py query "scores above 80" analytics.json -o high_scores.json -x
```

---

## Servers (sfs_*)

HTTP/WebSocket services, long-running daemons, voice servers.

| Script | MCP | Version | Description | Status |
|--------|-----|---------|-------------|--------|
| `sfs_tts_stt_server.py` | — | 1.0.0 | TTS/STT voice server with native Kokoro | start, stop, status, speak, listen |

---

## Loop Agents (sfl_*)

Multi-step agentic workflows with iterative improvement loops.

| Script | MCP | Version | Description | Loop Pattern |
|--------|-----|---------|-------------|--------------|
| `sfl_test_gen.py` | `test_gen` | 1.0.0 | Iterative test generation | Generate → coverage check → identify gaps → repeat until threshold |

**Usage Examples:**
```bash
# Generate tests until 90% coverage
./sfl_test_gen.py generate mymodule.py --target calculate_total --coverage 90

# Analyze code structure
./sfl_test_gen.py analyze mymodule.py
```

**Required By:**
- `sfb-reviewer` agent (must use for test generation tasks)

---

## Agents (sfa_*)

Autonomous task performers, AI workers with LLM integration. *Pending promotion from .scripts_hold.*

| Script | MCP | Planned | Description |
|--------|-----|---------|-------------|
| `sfa_coder.py` | TBD | v1.0.0 | Unified LLM coder (multi-provider) |
| `sfa_reviewer.py` | TBD | v1.0.0 | Code reviewer with DR/RFC output |

**Note:** Agent scripts in `.scripts_hold/` are being consolidated and will be promoted following full SFB compliance. See [Consolidation Plan](../docs/AGENT_CONSOLIDATION.md) (planned).

---

## Daemons (sfd_*)

Background watchers, monitors, cleanup tasks. *Reserved for future use.*

| Script | MCP | Version | Description |
|--------|-----|---------|-------------|
| *TBD* | — | — | Reserved category |

---

## Script Status Legend

| Badge | Meaning |
|-------|---------|
| ✅ | Production-ready, fully compliant |
| 🧪 | Experimental/preview |
| 📝 | Planned/proposed |
| ⏸️ | In .scripts_hold, awaiting promotion |

---

## Quick Reference

### Running Scripts
```bash
# Direct execution (uv resolves PEP 723 deps)
./sft_web.py search "query"

# Pipe between tools
echo "climate change" | ./sft_web.py search | jq '.results[].url'

# Run as MCP server
./sft_web.py mcp-stdio
```

### Compliance Check
```bash
# Check all SFB scripts
./sft_check.py check sft_*.py sfs_*.py sfl_*.py

# Check specific script
./sft_check.py check sft_web.py
```

### Adding a New Script

1. Choose bench type based on function:
   - **sft_** — CLI tools, utilities, commands
   - **sfs_** — Long-running servers
   - **sfa_** — Autonomous AI agents
   - **sfl_** — Multi-step iterative agents
   - **sfd_** — Background daemons

2. Follow [SPEC_SFA.md](../SPEC_SFA.md) — all 11 principles

3. Run compliance check before committing:
   ```bash
   ./sft_check.py check your_script.py
   ```

4. Update this index

---

## Architecture Overview

All scripts share:
- **PEP 723** inline metadata (`# /// script`)
- **`uv run --script`** dependency resolution
- **Triple interface:** CLI + Import + MCP
- **TSV logging** with 8-column format
- **Negative space** programming philosophy
- **Semantic versioning** with DR/RFC tracking

See [SPEC_SFA.md](../SPEC_SFA.md) for the full specification.

---

*Last updated: 2025-02-15*

*Total Scripts: 11 compliant, 2+ planned*
