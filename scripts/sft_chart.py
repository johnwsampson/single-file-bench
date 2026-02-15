#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bokeh>=3.3.0",
#     "pandas",
#     "fastmcp",
#     "numpy",
# ]
# ///
"""Chart and diagram generator - data visualization as HTML.

Generate interactive charts and diagrams from data files.
Supports histograms, line/bar/scatter/pie charts, sequence diagrams, and mermaid.
Opens results in the default browser automatically.

Usage:
    sft_chart.py histogram --file data.csv --column value
    sft_chart.py line --file data.csv --x date --y value
    sft_chart.py mermaid --code "graph TD; A-->B;"
    sft_chart.py mermaid --file diagram.mmd
    sft_chart.py mcp-stdio
"""

import argparse
import json
import math
import os
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
EXPOSED = [
    "histogram",
    "sequence",
    "line",
    "bar",
    "scatter",
    "pie",
    "mermaid",
]

CONFIG = {
    "version": "1.0.0",
    "display_char_limit": 25000,
}


# =============================================================================
# CORE FUNCTIONS - Data Loading
# =============================================================================

def _load_data(file_path: Path) -> Any:
    """Load data from CSV, JSON, or TSV."""
    import pandas as pd
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    elif ext == ".json":
        return json.loads(file_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# =============================================================================
# CORE FUNCTIONS - Mermaid
# =============================================================================

MERMAID_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SFA Mermaid Render</title>
    <style>
        body {{
            font-family: sans-serif;
            margin: 20px;
            background-color: #fafafa;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            max-width: 100%;
            overflow: auto;
        }}
        h1 {{
            color: #333;
            font-size: 1.2rem;
            margin-bottom: 1rem;
        }}
    </style>
</head>
<body>
    <h1>Mermaid Diagram Preview</h1>
    <div class="container">
        <pre class="mermaid">
{code}
        </pre>
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true }});
    </script>
</body>
</html>"""


def _mermaid_impl(code: str, output_path: str, show_plot: bool = True) -> tuple[str, dict]:
    """Generate HTML file with embedded Mermaid code. CLI: mermaid, MCP: mermaid."""
    start_ms = time.time() * 1000
    
    html_content = MERMAID_HTML_TEMPLATE.format(code=code)
    out_file = Path(output_path).resolve()
    out_file.write_text(html_content, encoding="utf-8")
    
    if show_plot:
        webbrowser.open(out_file.as_uri())
    
    latency_ms = time.time() * 1000 - start_ms
    metrics = {"latency_ms": round(latency_ms, 2), "output": str(out_file)}
    return str(out_file), metrics


# =============================================================================
# CORE FUNCTIONS - Bokeh Charts
# =============================================================================

def _histogram_impl(
    data: Any, col_name: str, output_path: str, show_plot: bool = True
) -> tuple[str, dict]:
    """Generate histogram from DataFrame column. CLI: histogram, MCP: histogram."""
    import numpy as np
    import pandas as pd
    from bokeh.plotting import figure, output_file, save, show
    from bokeh.models import ColumnDataSource, HoverTool
    
    start_ms = time.time() * 1000
    
    if col_name not in data.columns:
        raise ValueError(f"Column '{col_name}' not found. Available: {list(data.columns)}")
    
    clean_data = pd.to_numeric(data[col_name], errors="coerce").dropna()
    
    p = figure(
        title=f"Histogram of {col_name}",
        x_axis_label=col_name,
        y_axis_label="Count",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        background_fill_color="#fafafa",
    )
    
    hist, edges = np.histogram(clean_data, bins=20)
    hist_df = pd.DataFrame({
        "top": hist,
        "left": edges[:-1],
        "right": edges[1:],
        "count": hist,
        "interval": [f"{l:.2f} - {r:.2f}" for l, r in zip(edges[:-1], edges[1:])],
    })
    
    source = ColumnDataSource(hist_df)
    p.quad(
        top="top", bottom=0, left="left", right="right",
        fill_color="navy", line_color="white", alpha=0.5,
        source=source,
    )
    
    hover = HoverTool(tooltips=[("Interval", "@interval"), ("Count", "@count")])
    p.add_tools(hover)
    
    output_file(output_path, title=f"Histogram of {col_name}")
    if show_plot:
        show(p)
    else:
        save(p)
    
    latency_ms = time.time() * 1000 - start_ms
    return str(Path(output_path).absolute()), {"latency_ms": round(latency_ms, 2)}


def _sequence_impl(
    events: list[dict], output_path: str, show_plot: bool = True
) -> tuple[str, dict]:
    """Generate sequence diagram from events. CLI: sequence, MCP: sequence."""
    from bokeh.plotting import figure, output_file, save, show
    from bokeh.models import Arrow, NormalHead
    
    start_ms = time.time() * 1000
    
    # Identify actors
    actors = []
    for e in events:
        if e.get("from") and e["from"] not in actors:
            actors.append(e["from"])
        if e.get("to") and e["to"] not in actors:
            actors.append(e["to"])
    
    if not actors:
        raise ValueError("No actors found in event data.")
    
    actor_map = {actor: i for i, actor in enumerate(actors)}
    num_events = len(events)
    
    p = figure(
        title="Sequence Diagram",
        x_range=actors,
        y_range=(-num_events - 1, 1),
        height=max(400, num_events * 50 + 100),
        width=max(600, len(actors) * 150),
        tools="pan,wheel_zoom,save,reset",
        x_axis_location="above",
    )
    
    p.yaxis.visible = False
    p.grid.grid_line_color = None
    
    # Draw lifelines
    p.segment(
        x0=[i for i in range(len(actors))], y0=0,
        x1=[i for i in range(len(actors))], y1=-num_events,
        line_color="gray", line_dash="dashed", line_width=2,
    )
    
    # Draw messages
    for i, event in enumerate(events):
        y_pos = -i
        start_actor = event.get("from")
        end_actor = event.get("to")
        label = event.get("label", "")
        
        if start_actor and end_actor and start_actor in actor_map and end_actor in actor_map:
            x_start = actor_map[start_actor]
            x_end = actor_map[end_actor]
            
            p.add_layout(Arrow(
                end=NormalHead(fill_color="orange", size=10),
                x_start=x_start, y_start=y_pos,
                x_end=x_end, y_end=y_pos,
                line_color="orange", line_width=2,
            ))
            
            x_mid = (x_start + x_end) / 2
            p.text(
                x=[x_mid], y=[y_pos], text=[label],
                text_align="center", text_baseline="bottom",
                text_font_size="10pt", y_offset=-2,
            )
    
    output_file(output_path, title="Sequence Diagram")
    if show_plot:
        show(p)
    else:
        save(p)
    
    latency_ms = time.time() * 1000 - start_ms
    return str(Path(output_path).absolute()), {"latency_ms": round(latency_ms, 2)}


def _line_impl(
    data: Any, x_col: str, y_cols: str, output_path: str, show_plot: bool = True
) -> tuple[str, dict]:
    """Generate line chart. CLI: line, MCP: line."""
    import pandas as pd
    from bokeh.plotting import figure, output_file, save, show
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.palettes import Category10
    
    start_ms = time.time() * 1000
    
    y_column_list = [c.strip() for c in y_cols.split(",")]
    
    if x_col not in data.columns:
        raise ValueError(f"Column '{x_col}' not found.")
    for col in y_column_list:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found.")
    
    # Try to parse dates
    try:
        data[x_col] = pd.to_datetime(data[x_col])
        x_axis_type = "datetime"
    except Exception:
        x_axis_type = "auto"
    
    source = ColumnDataSource(data)
    
    p = figure(
        title="Line Chart",
        x_axis_label=x_col,
        y_axis_label="Value",
        x_axis_type=x_axis_type,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        background_fill_color="#fafafa",
    )
    
    colors = Category10[10]
    for i, col_name in enumerate(y_column_list):
        color = colors[i % 10]
        p.line(x=x_col, y=col_name, source=source, line_width=2, color=color, legend_label=col_name)
        p.scatter(x=x_col, y=col_name, source=source, size=8, color=color, fill_color="white", legend_label=col_name)
    
    hover_tooltips = [(x_col, f"@{x_col}{{%F}}" if x_axis_type == "datetime" else f"@{x_col}")]
    for col_name in y_column_list:
        hover_tooltips.append((col_name, f"@{col_name}"))
    
    hover = HoverTool(tooltips=hover_tooltips)
    if x_axis_type == "datetime":
        hover.formatters = {f"@{x_col}": "datetime"}
    p.add_tools(hover)
    p.legend.click_policy = "hide"
    
    output_file(output_path, title="Line Chart")
    if show_plot:
        show(p)
    else:
        save(p)
    
    latency_ms = time.time() * 1000 - start_ms
    return str(Path(output_path).absolute()), {"latency_ms": round(latency_ms, 2)}


def _bar_impl(
    data: Any, x_col: str, y_cols: str, output_path: str, show_plot: bool = True, stacked: bool = False
) -> tuple[str, dict]:
    """Generate bar chart. CLI: bar, MCP: bar."""
    import pandas as pd
    from bokeh.plotting import figure, output_file, save, show
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.palettes import Category10
    from bokeh.transform import dodge
    
    start_ms = time.time() * 1000
    
    y_column_list = [c.strip() for c in y_cols.split(",")]
    
    if x_col not in data.columns:
        raise ValueError(f"Column '{x_col}' not found.")
    for col in y_column_list:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found.")
    
    data[x_col] = data[x_col].astype(str)
    categories = data[x_col].unique().tolist()
    
    source = ColumnDataSource(data)
    
    p = figure(
        title="Bar Chart",
        x_range=categories,
        x_axis_label=x_col,
        y_axis_label="Value",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        background_fill_color="#fafafa",
    )
    
    colors = Category10[10]
    
    if stacked and len(y_column_list) > 1:
        p.vbar_stack(
            y_column_list, x=x_col, width=0.9, source=source,
            color=colors[:len(y_column_list)], legend_label=y_column_list,
        )
    elif len(y_column_list) > 1:
        width = 0.8 / len(y_column_list)
        start_offset = -0.4 + (width / 2)
        for i, col_name in enumerate(y_column_list):
            offset = start_offset + (i * width)
            p.vbar(
                x=dodge(x_col, offset, range=p.x_range),
                top=col_name, width=width, source=source,
                color=colors[i % 10], legend_label=col_name,
            )
    else:
        p.vbar(
            x=x_col, top=y_column_list[0], width=0.9, source=source,
            line_color="white", fill_color="navy",
        )
    
    hover_tooltips = [(x_col, f"@{x_col}")]
    for col_name in y_column_list:
        hover_tooltips.append((col_name, f"@{col_name}"))
    
    hover = HoverTool(tooltips=hover_tooltips)
    p.add_tools(hover)
    if len(y_column_list) > 1:
        p.legend.click_policy = "hide"
    
    output_file(output_path, title="Bar Chart")
    if show_plot:
        show(p)
    else:
        save(p)
    
    latency_ms = time.time() * 1000 - start_ms
    return str(Path(output_path).absolute()), {"latency_ms": round(latency_ms, 2)}


def _scatter_impl(
    data: Any, x_col: str, y_cols: str, output_path: str, show_plot: bool = True
) -> tuple[str, dict]:
    """Generate scatter plot. CLI: scatter, MCP: scatter."""
    import pandas as pd
    from bokeh.plotting import figure, output_file, save, show
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.palettes import Category10
    
    start_ms = time.time() * 1000
    
    y_column_list = [c.strip() for c in y_cols.split(",")]
    
    if x_col not in data.columns:
        raise ValueError(f"Column '{x_col}' not found.")
    for col in y_column_list:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found.")
    
    source = ColumnDataSource(data)
    
    p = figure(
        title="Scatter Plot",
        x_axis_label=x_col,
        y_axis_label="Value",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        background_fill_color="#fafafa",
    )
    
    colors = Category10[10]
    for i, col_name in enumerate(y_column_list):
        p.scatter(
            x=x_col, y=col_name, source=source, size=10,
            color=colors[i % 10], alpha=0.5, legend_label=col_name,
        )
    
    hover_tooltips = [(x_col, f"@{x_col}")]
    for col_name in y_column_list:
        hover_tooltips.append((col_name, f"@{col_name}"))
    
    hover = HoverTool(tooltips=hover_tooltips)
    p.add_tools(hover)
    p.legend.click_policy = "hide"
    
    output_file(output_path, title="Scatter Plot")
    if show_plot:
        show(p)
    else:
        save(p)
    
    latency_ms = time.time() * 1000 - start_ms
    return str(Path(output_path).absolute()), {"latency_ms": round(latency_ms, 2)}


def _pie_impl(
    data: Any, x_col: str, y_col: str, output_path: str, show_plot: bool = True
) -> tuple[str, dict]:
    """Generate pie chart. CLI: pie, MCP: pie."""
    import pandas as pd
    from bokeh.plotting import figure, output_file, save, show
    from bokeh.models import ColumnDataSource
    from bokeh.palettes import Category20c
    from bokeh.transform import cumsum
    
    start_ms = time.time() * 1000
    
    if x_col not in data.columns or y_col not in data.columns:
        raise ValueError(f"Columns '{x_col}' or '{y_col}' not found.")
    
    data = data.groupby(x_col)[y_col].sum().reset_index()
    data["angle"] = data[y_col] / data[y_col].sum() * 2 * math.pi
    
    if len(data) > 20:
        palette = Category20c[20] * (len(data) // 20 + 1)
        data["color"] = palette[:len(data)]
    elif len(data) < 3:
        data["color"] = Category20c[3][:len(data)]
    else:
        data["color"] = Category20c[len(data)]
    
    source = ColumnDataSource(data)
    
    p = figure(
        title="Pie Chart",
        toolbar_location=None,
        tools="hover",
        tooltips=f"@{x_col}: @{y_col}",
        x_range=(-0.5, 1.0),
    )
    
    p.wedge(
        x=0, y=1, radius=0.4,
        start_angle=cumsum("angle", include_zero=True),
        end_angle=cumsum("angle"),
        line_color="white", fill_color="color",
        legend_field=x_col, source=source,
    )
    
    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None
    
    output_file(output_path, title="Pie Chart")
    if show_plot:
        show(p)
    else:
        save(p)
    
    latency_ms = time.time() * 1000 - start_ms
    return str(Path(output_path).absolute()), {"latency_ms": round(latency_ms, 2)}


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualization generator - charts and diagrams as HTML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sft_viz.py histogram --file data.csv --column value
  sft_viz.py line --file data.csv --x date --y value
  sft_viz.py bar --file data.csv --x category --y value --stacked
  sft_viz.py scatter --file data.csv --x height --y weight
  sft_viz.py pie --file data.csv --x category --y value
  sft_viz.py sequence --file events.json
  sft_viz.py mermaid --code "graph TD; A-->B;"
  sft_viz.py mermaid --file diagram.mmd
        """,
    )
    
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {CONFIG['version']}",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Visualization type")
    
    # histogram
    p_hist = subparsers.add_parser("histogram", help="Histogram of numerical data")
    p_hist.add_argument("-f", "--file", required=True, help="Input data file")
    p_hist.add_argument("-c", "--column", required=True, help="Column name")
    p_hist.add_argument("-o", "--output", help="Output HTML file")
    p_hist.add_argument("-n", "--no-show", action="store_true", help="Don't open browser")
    
    # sequence
    p_seq = subparsers.add_parser("sequence", help="Sequence diagram from events")
    p_seq.add_argument("-f", "--file", required=True, help="Input data file (JSON/CSV)")
    p_seq.add_argument("-o", "--output", help="Output HTML file")
    p_seq.add_argument("-n", "--no-show", action="store_true", help="Don't open browser")
    
    # line
    p_line = subparsers.add_parser("line", help="Line chart")
    p_line.add_argument("-f", "--file", required=True, help="Input data file")
    p_line.add_argument("-x", required=True, help="X-axis column")
    p_line.add_argument("-y", required=True, help="Y-axis column(s), comma-separated")
    p_line.add_argument("-o", "--output", help="Output HTML file")
    p_line.add_argument("-n", "--no-show", action="store_true", help="Don't open browser")
    
    # bar
    p_bar = subparsers.add_parser("bar", help="Bar chart")
    p_bar.add_argument("-f", "--file", required=True, help="Input data file")
    p_bar.add_argument("-x", required=True, help="X-axis column (categories)")
    p_bar.add_argument("-y", required=True, help="Y-axis column(s), comma-separated")
    p_bar.add_argument("-s", "--stacked", action="store_true", help="Stack bars")
    p_bar.add_argument("-o", "--output", help="Output HTML file")
    p_bar.add_argument("-n", "--no-show", action="store_true", help="Don't open browser")
    
    # scatter
    p_scatter = subparsers.add_parser("scatter", help="Scatter plot")
    p_scatter.add_argument("-f", "--file", required=True, help="Input data file")
    p_scatter.add_argument("-x", required=True, help="X-axis column")
    p_scatter.add_argument("-y", required=True, help="Y-axis column(s), comma-separated")
    p_scatter.add_argument("-o", "--output", help="Output HTML file")
    p_scatter.add_argument("-n", "--no-show", action="store_true", help="Don't open browser")
    
    # pie
    p_pie = subparsers.add_parser("pie", help="Pie chart")
    p_pie.add_argument("-f", "--file", required=True, help="Input data file")
    p_pie.add_argument("-x", required=True, help="Category column")
    p_pie.add_argument("-y", required=True, help="Value column")
    p_pie.add_argument("-o", "--output", help="Output HTML file")
    p_pie.add_argument("-n", "--no-show", action="store_true", help="Don't open browser")
    
    # mermaid
    p_mermaid = subparsers.add_parser("mermaid", help="Mermaid diagram")
    p_mermaid.add_argument("-f", "--file", help="Input .mmd file")
    p_mermaid.add_argument("-c", "--code", help="Mermaid code string")
    p_mermaid.add_argument("-o", "--output", default="mermaid_output.html", help="Output HTML file")
    p_mermaid.add_argument("-n", "--no-show", action="store_true", help="Don't open browser")
    
    # MCP server
    subparsers.add_parser("mcp-stdio", help="Run as MCP server")
    
    args = parser.parse_args()
    
    try:
        if args.command == "mcp-stdio":
            _run_mcp()
        elif args.command == "histogram":
            data = _load_data(Path(args.file))
            output = args.output or f"{args.command}_output.html"
            result, metrics = _histogram_impl(data, args.column, output, not args.no_show)
            print(json.dumps({"output": result, "metrics": metrics}))
            _log("INFO", args.command, f"column={args.column}", metrics=json.dumps(metrics))
        elif args.command == "sequence":
            data = _load_data(Path(args.file))
            events = data if isinstance(data, list) else data.to_dict(orient="records")
            output = args.output or f"{args.command}_output.html"
            result, metrics = _sequence_impl(events, output, not args.no_show)
            print(json.dumps({"output": result, "metrics": metrics}))
            _log("INFO", args.command, f"events={len(events)}", metrics=json.dumps(metrics))
        elif args.command == "line":
            data = _load_data(Path(args.file))
            output = args.output or f"{args.command}_output.html"
            result, metrics = _line_impl(data, args.x, args.y, output, not args.no_show)
            print(json.dumps({"output": result, "metrics": metrics}))
            _log("INFO", args.command, f"x={args.x} y={args.y}", metrics=json.dumps(metrics))
        elif args.command == "bar":
            data = _load_data(Path(args.file))
            output = args.output or f"{args.command}_output.html"
            result, metrics = _bar_impl(data, args.x, args.y, output, not args.no_show, args.stacked)
            print(json.dumps({"output": result, "metrics": metrics}))
            _log("INFO", args.command, f"x={args.x} y={args.y} stacked={args.stacked}", metrics=json.dumps(metrics))
        elif args.command == "scatter":
            data = _load_data(Path(args.file))
            output = args.output or f"{args.command}_output.html"
            result, metrics = _scatter_impl(data, args.x, args.y, output, not args.no_show)
            print(json.dumps({"output": result, "metrics": metrics}))
            _log("INFO", args.command, f"x={args.x} y={args.y}", metrics=json.dumps(metrics))
        elif args.command == "pie":
            data = _load_data(Path(args.file))
            output = args.output or f"{args.command}_output.html"
            result, metrics = _pie_impl(data, args.x, args.y, output, not args.no_show)
            print(json.dumps({"output": result, "metrics": metrics}))
            _log("INFO", args.command, f"x={args.x} y={args.y}", metrics=json.dumps(metrics))
        elif args.command == "mermaid":
            if args.file:
                code = Path(args.file).read_text(encoding="utf-8")
            elif args.code:
                code = args.code
            else:
                code = sys.stdin.read().strip() if not sys.stdin.isatty() else ""
            assert code, "mermaid code required (--file, --code, or stdin)"
            result, metrics = _mermaid_impl(code, args.output, not args.no_show)
            print(json.dumps({"output": result, "metrics": metrics}))
            _log("INFO", args.command, f"chars={len(code)}", metrics=json.dumps(metrics))
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
    
    mcp = FastMCP("viz")
    
    @mcp.tool()
    def histogram(file_path: str, column: str, output_path: str = "histogram.html") -> str:
        """Create histogram from data file.
        
        Args:
            file_path: Path to CSV/TSV/JSON file
            column: Column name to histogram
            output_path: Output HTML file path
        
        Returns:
            JSON with output path and metrics
        """
        try:
            data = _load_data(Path(file_path))
            result, metrics = _histogram_impl(data, column, output_path, show_plot=False)
            return json.dumps({"output": result, "metrics": metrics})
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @mcp.tool()
    def sequence(file_path: str, output_path: str = "sequence.html") -> str:
        """Create sequence diagram from events file.
        
        Args:
            file_path: Path to JSON/CSV with events (from/to/label fields)
            output_path: Output HTML file path
        
        Returns:
            JSON with output path and metrics
        """
        try:
            data = _load_data(Path(file_path))
            events = data if isinstance(data, list) else data.to_dict(orient="records")
            result, metrics = _sequence_impl(events, output_path, show_plot=False)
            return json.dumps({"output": result, "metrics": metrics})
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @mcp.tool()
    def line(file_path: str, x: str, y: str, output_path: str = "line.html") -> str:
        """Create line chart from data file.
        
        Args:
            file_path: Path to CSV/TSV/JSON file
            x: X-axis column name
            y: Y-axis column(s), comma-separated for multiple lines
            output_path: Output HTML file path
        
        Returns:
            JSON with output path and metrics
        """
        try:
            data = _load_data(Path(file_path))
            result, metrics = _line_impl(data, x, y, output_path, show_plot=False)
            return json.dumps({"output": result, "metrics": metrics})
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @mcp.tool()
    def bar(file_path: str, x: str, y: str, output_path: str = "bar.html", stacked: bool = False) -> str:
        """Create bar chart from data file.
        
        Args:
            file_path: Path to CSV/TSV/JSON file
            x: X-axis column (categories)
            y: Y-axis column(s), comma-separated
            output_path: Output HTML file path
            stacked: Whether to stack bars (default: False)
        
        Returns:
            JSON with output path and metrics
        """
        try:
            data = _load_data(Path(file_path))
            result, metrics = _bar_impl(data, x, y, output_path, show_plot=False, stacked=stacked)
            return json.dumps({"output": result, "metrics": metrics})
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @mcp.tool()
    def scatter(file_path: str, x: str, y: str, output_path: str = "scatter.html") -> str:
        """Create scatter plot from data file.
        
        Args:
            file_path: Path to CSV/TSV/JSON file
            x: X-axis column name
            y: Y-axis column(s), comma-separated
            output_path: Output HTML file path
        
        Returns:
            JSON with output path and metrics
        """
        try:
            data = _load_data(Path(file_path))
            result, metrics = _scatter_impl(data, x, y, output_path, show_plot=False)
            return json.dumps({"output": result, "metrics": metrics})
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @mcp.tool()
    def pie(file_path: str, x: str, y: str, output_path: str = "pie.html") -> str:
        """Create pie chart from data file.
        
        Args:
            file_path: Path to CSV/TSV/JSON file
            x: Category column name
            y: Value column name
            output_path: Output HTML file path
        
        Returns:
            JSON with output path and metrics
        """
        try:
            data = _load_data(Path(file_path))
            result, metrics = _pie_impl(data, x, y, output_path, show_plot=False)
            return json.dumps({"output": result, "metrics": metrics})
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @mcp.tool()
    def mermaid(code: str, output_path: str = "mermaid_output.html") -> str:
        """Render mermaid diagram to HTML.
        
        Args:
            code: Mermaid diagram code
            output_path: Output HTML file path
        
        Returns:
            JSON with output path and metrics
        """
        try:
            result, metrics = _mermaid_impl(code, output_path, show_plot=False)
            return json.dumps({"output": result, "metrics": metrics})
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
