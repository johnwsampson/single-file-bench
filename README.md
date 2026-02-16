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
- **Four architectural roles** (Tools, Servers, Agents, Loops) organized by function
- **Triple interface** — Every script exposes CLI + Import + MCP (Model Context Protocol)
- **Structured TSV logging** — Queryable observability with standard unix tools
- **Negative space programming** — Assert contracts, fail loud, zero defensive coding
- **Automated compliance** — `sft_check.py` verifies spec adherence
- **Unix-native** — First-class stdin/stdout piping support

Self-contained Python CLI tools. Each script exposes three interfaces from one file:
- **CLI** — `uv run --script scripts/sft_web.py search "query"`
- **Import** — `from sft_web import _search_impl`
- **MCP** — `uv run --script scripts/sft_web.py mcp-stdio` (FastMCP over stdio)

## Quick Start

```bash
# Run any script directly (uv resolves PEP 723 inline dependencies)
uv run --script scripts/sft_web.py search "opencode"

# Pipe between tools
echo "opencode" | uv run --script scripts/sft_web.py search | jq '.results[].url'

# Run as MCP server for agent consumption
uv run --script scripts/sft_web.py mcp-stdio
```

**Requirements:** [uv](https://github.com/astral-sh/uv) (Astral). No pip, no virtualenv, no requirements.txt.

## Architecture

Each script is fully self-contained via [PEP 723](https://peps.python.org/pep-0723/) inline metadata. Dependencies are declared in the script header and resolved by `uv` at runtime. No shared environment, no dependency conflicts between scripts.

Scripts follow a mandatory structure: PEP 723 header, logging block, configuration, core functions, CLI interface, FastMCP server. See [SPEC_SFB.md](SPEC_SFB.md) for the full specification.

## Key Design Principles

1. **One source of truth** — Core functions are the implementation. CLI and MCP are interfaces.
2. **Single-file scripts** — PEP 723 + `uv run --script`. One file = complete deployment artifact.
3. **Script organization** — Uniform structure, configuration at the top, EXPOSED list as inventory.
4. **Intuitive interfaces** — Unix piping, short+long flags, natural positional arguments.
5. **Negative space programming** — Assert contracts, fail loud. No defensive coding.
6. **Semantic versioning** — Every script versioned. Changes tracked in [CHANGELOG.md](CHANGELOG.md).

See [SPEC_SFB.md](SPEC_SFB.md) for the full 11-principle specification.

## Compliance

The `sft_`, `sfs_`, `sfa_`, `sfl_` prefixes are certification marks. Only scripts that pass all spec principles earn them.

```bash
# Check a single script
uv run --script scripts/sft_check.py check scripts/sft_web.py

# Check all SFB scripts
uv run --script scripts/sft_check.py check scripts/sft_*.py scripts/sfs_*.py scripts/sfa_*.py scripts/sfl_*.py
```

## Scripts

### Tools (`sft_`)
| Script | MCP | Description |
|--------|-----|-------------|
| `sft_web.py` | `web` | Web search (SearXNG) and URL fetch to markdown |
| `sft_check.py` | `check` | SFB spec compliance checker |
| `sft_scout.py` | `fs` | Code reconnaissance — AST analysis, file search, content search |
| `sft_file_read.py` | `fr` | Read-only file operations and inspection |
| `sft_file_write.py` | `fw` | Destructive file ops with automatic backup system |
| `sft_brain.py` | `brain` | FOQA knowledge store with semantic search |
| `sft_recall.py` | `recall` | Unified search across all brain tables |
| `sft_reason.py` | `reason` | Persistent structured thinking with branches |
| `sft_track.py` | `track` | Task tracking with hierarchy, deps, and passdowns |
| `sft_chat_history_oc.py` | `chat_history_oc` | OpenCode session reader for context recovery |
| `sft_duckdb.py` | `duckdb` | Natural language SQL queries via LLM |
| `sft_jq.py` | `jq` | Natural language JSON processing via jq |
| `sft_context7.py` | `context7` | Library documentation via Context7 API |
| `sft_chrome.py` | `chrome` | Chrome automation via CDP — sandbox/work/personal profiles |
| `sft_clipboard.py` | `clip` | System clipboard read/write |
| `sft_chart.py` | `viz` | Data visualization and chart generation as HTML |
| `sft_tmux.py` | `tmux` | Tmux session orchestration — spawn, control, message |
| `sft_openrouter_models.py` | `models` | OpenRouter model catalog — list, search, compare |
| `sft_audio_analyze.py` | `audio_analyze` | Audio analysis — key, tempo, spectral, dynamics |
| `sft_speech2text.py` | `whisper` | Speech-to-text transcription |
| `sft_grepcode.py` | `repo_grep` | Code search across GitHub repos via Grep.app |
| `sft_docx2md.py` | `docx2md` | Word (.docx) to markdown conversion |
| `sft_md2html.py` | `md2html` | Markdown to styled HTML with mermaid diagrams |
| `sft_x.py` | `x` | X/Twitter and web search via xAI Grok API |

### Servers (`sfs_`)
| Script | MCP | Description |
|--------|-----|-------------|
| `sfs_events.py` | `events` | Event bus — pub/sub for agent orchestration |
| `sfs_tts_stt_server.py` | `tts-stt-server` | Voice server — Kokoro TTS + browser speech recognition |

### Agents (`sfa_`)
| Script | MCP | Description |
|--------|-----|-------------|
| `sfa_notebooklm.py` | `notebooklm` | NotebookLM API client |
| `sfa_tts_edge.py` | `tts-edge` | Text-to-speech via Microsoft Edge TTS |
| `sfa_tts_kokoro.py` | `tts-kokoro` | Native Kokoro-82M TTS with Apple Silicon MPS |
| `sfa_voice_kokoro_docker.py` | `kokoro-docker` | Kokoro TTS via Docker with auto-start |
| `sfa_youtube_video.py` | `ytv` | YouTube video info, transcripts, channel search |

### Loop Agents (`sfl_`)
| Script | MCP | Description |
|--------|-----|-------------|
| `sfl_test_gen.py` | `test_gen` | Iterative test generation until coverage threshold |

## Development

```bash
# Lint and format
ruff check scripts/ --fix
ruff format scripts/

# Manual verification (no unit tests — scripts use runtime assertions)
uv run --script scripts/sft_web.py search "test query"
tail -20 scripts/sft_web_log.tsv
```

## Documentation

- [SPEC_SFB.md](SPEC_SFB.md) — Full specification (11 principles)
- [AGENTS.md](AGENTS.md) — Quick reference for agentic coding assistants
- [CHANGELOG.md](CHANGELOG.md) — DR/RFC change tracking
- [scripts/README.md](scripts/README.md) — Complete script index by bench type

## License

MIT License — This project extends [disler's Single File Agents](https://github.com/disler/single-file-agents) (MIT) and builds upon patterns demonstrated by [Simon Willison](https://simonwillison.net) (open source advocate).

See: [disler/single-file-agents LICENSE](https://github.com/disler/single-file-agents/blob/main/README.md#license)
