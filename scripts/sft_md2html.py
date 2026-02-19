#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "markdown",
#     "pyyaml",
#     "fastmcp",
# ]
# ///
"""Convert markdown files to standalone HTML with styling and mermaid diagrams.

Renders markdown to beautiful HTML with CSS styling, table formatting,
mermaid diagram support, and YAML frontmatter handling.

Usage:
    sft_md2html.py input.md
    sft_md2html.py input.md --output report.html
    sft_md2html.py input.md --show
    cat doc.md | sft_md2html.py --output doc.html
    sft_md2html.py mcp-stdio

Chaining:
    sft_docx2md.py read document.docx | sft_md2html.py --output doc.html --show
"""

import argparse
import json
import os
import re
import sys
import time
import webbrowser
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
EXPOSED = ["convert"]

CONFIG = {
    "version": "1.0.0",
    "extensions": ["tables", "fenced_code", "codehilite"],
}


# =============================================================================
# HTML TEMPLATE
# =============================================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg: #fafafa;
            --fg: #222;
            --accent: #0066cc;
            --border: #ddd;
            --code-bg: #f4f4f4;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            background: var(--bg);
            color: var(--fg);
        }}

        h1 {{ font-size: 2rem; border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem; }}
        h2 {{ font-size: 1.5rem; margin-top: 2rem; color: var(--accent); }}
        h3 {{ font-size: 1.2rem; margin-top: 1.5rem; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
        }}

        th, td {{
            border: 1px solid var(--border);
            padding: 0.5rem 0.75rem;
            text-align: left;
        }}

        th {{
            background: var(--code-bg);
            font-weight: 600;
        }}

        tr:nth-child(even) {{
            background: #fafafa;
        }}

        code {{
            background: var(--code-bg);
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-size: 0.9em;
        }}

        pre {{
            background: var(--code-bg);
            padding: 1rem;
            border-radius: 5px;
            overflow-x: auto;
        }}

        pre code {{
            background: none;
            padding: 0;
        }}

        .mermaid {{
            background: white;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
            text-align: center;
        }}

        blockquote {{
            border-left: 3px solid var(--accent);
            margin: 1rem 0;
            padding-left: 1rem;
            color: #555;
        }}

        hr {{
            border: none;
            border-top: 1px solid var(--border);
            margin: 2rem 0;
        }}

        .metadata {{
            background: var(--code-bg);
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 2rem;
            font-size: 0.85rem;
        }}

        .metadata dt {{ font-weight: 600; }}
        .metadata dd {{ margin: 0 0 0.5rem 0; }}
    </style>
</head>
<body>
{metadata}
{content}
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
    </script>
</body>
</html>"""


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def _extract_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from markdown."""
    import yaml
    
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                metadata = yaml.safe_load(parts[1])
                return metadata or {}, parts[2].strip()
            except Exception:
                pass
    return {}, text


def _convert_mermaid_blocks(html: str) -> str:
    """Convert ```mermaid code blocks to mermaid divs."""
    pattern = r'<pre><code class="language-mermaid">(.*?)</code></pre>'
    
    def replace_mermaid(match):
        code = match.group(1)
        code = code.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        return f'<div class="mermaid">\n{code}\n</div>'
    
    return re.sub(pattern, replace_mermaid, html, flags=re.DOTALL)


def _format_metadata(metadata: dict) -> str:
    """Format frontmatter metadata as HTML."""
    if not metadata:
        return ""
    
    skip = {"title"}
    items = [(k, v) for k, v in metadata.items() if k not in skip]
    
    if not items:
        return ""
    
    html = '<div class="metadata"><dl>'
    for key, value in items:
        html += f"<dt>{key}</dt><dd>{value}</dd>"
    html += "</dl></div>"
    return html


def _convert_impl(
    input_text: str, output_path: str | None = None, show: bool = False
) -> tuple[str, dict]:
    """Convert markdown to HTML. CLI: convert, MCP: convert."""
    import markdown
    
    start_ms = time.time() * 1000
    
    metadata, content = _extract_frontmatter(input_text)
    
    md = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite"])
    html_content = md.convert(content)
    html_content = _convert_mermaid_blocks(html_content)
    
    title = metadata.get("title", "Document")
    metadata_html = _format_metadata(metadata)
    
    final_html = HTML_TEMPLATE.format(
        title=title, metadata=metadata_html, content=html_content
    )
    
    if output_path is None:
        output_path = "output.html"
    
    out_file = Path(output_path)
    out_file.write_text(final_html, encoding="utf-8")
    
    if show:
        webbrowser.open(out_file.as_uri())
    
    latency_ms = time.time() * 1000 - start_ms
    metrics = {"latency_ms": round(latency_ms, 2), "output": str(out_file)}
    return str(out_file.absolute()), metrics


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert markdown to styled HTML with mermaid support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sft_md2html.py input.md
  sft_md2html.py input.md --output report.html
  sft_md2html.py input.md --show
  echo "# Hello" | sft_md2html.py --output hello.html
  sft_docx2md.py read document.docx | sft_md2html.py --show
        """,
    )
    
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {CONFIG['version']}",
    )
    
    parser.add_argument("input", nargs="?", help="Input markdown file (or stdin)")
    parser.add_argument("-o", "--output", help="Output HTML file")
    parser.add_argument("-s", "--show", action="store_true", help="Open in browser")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # convert (explicit subcommand)
    p_convert = subparsers.add_parser("convert", help="Convert markdown to HTML")
    p_convert.add_argument("input", nargs="?", help="Input markdown file")
    p_convert.add_argument("-o", "--output", help="Output HTML file")
    p_convert.add_argument("-s", "--show", action="store_true", help="Open in browser")
    
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")
    
    args = parser.parse_args()
    
    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "convert" or args.input:
            # Get input from file or stdin
            input_file = args.input if args.command == "convert" else args.input
            if input_file and input_file != "-":
                input_text = Path(input_file).read_text(encoding="utf-8")
            elif not sys.stdin.isatty():
                input_text = sys.stdin.read()
            else:
                assert input_file, "input required (file path or stdin)"
                input_text = Path(input_file).read_text(encoding="utf-8")
            
            assert input_text.strip(), "input is empty"
            
            show = args.show if hasattr(args, 'show') else False
            output = args.output if hasattr(args, 'output') else None
            result, metrics = _convert_impl(input_text, output, show)
            print(json.dumps({"output": result, "metrics": metrics}))
            _log("INFO", "convert", f"output={result}", metrics=json.dumps(metrics))
        else:
            parser.print_help()
    except AssertionError as e:
        _log("ERROR", "contract_violation", str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        _log("ERROR", "runtime_error", str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# FASTMCP SERVER
# =============================================================================

def _run_mcp():
    """Run as MCP server."""
    from fastmcp import FastMCP
    
    mcp = FastMCP("md2html")
    
    @mcp.tool()
    def convert(markdown_text: str, output_path: str = "output.html") -> str:
        """Convert markdown text to styled HTML.
        
        Args:
            markdown_text: Markdown content to convert
            output_path: Output HTML file path
        
        Returns:
            JSON with output path and metrics
        """
        try:
            result, metrics = _convert_impl(markdown_text, output_path, show=False)
            return json.dumps({"output": result, "metrics": metrics})
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
