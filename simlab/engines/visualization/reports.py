"""Report generation engine.

Converts ExperimentResult objects into Markdown, HTML, or JSON reports.
"""

from __future__ import annotations

import json
from datetime import datetime

from simlab.core.schemas.experiment import ExperimentResult


class ReportEngine:
    """Generate human-readable experiment reports."""

    def generate_markdown_report(self, result: ExperimentResult) -> str:
        """Generate a Markdown-formatted report from an ExperimentResult."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"# SIMLAB Experiment Report",
            f"",
            f"**Experiment ID:** `{result.experiment_id}`  ",
            f"**Generated:** {now}  ",
            f"**Domain:** {result.domain}  ",
            f"**Type:** {result.type}  ",
            f"**Status:** {result.status}  ",
            f"",
        ]

        if result.warnings:
            lines.append("## Warnings")
            for w in result.warnings:
                lines.append(f"- {w}")
            lines.append("")

        if result.errors:
            lines.append("## Errors")
            for e in result.errors:
                lines.append(f"- {e}")
            lines.append("")

        lines.append("## Results")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(self._sanitize_results(result.results), indent=2))
        lines.append("```")
        lines.append("")

        if result.metadata:
            lines.append("## Metadata")
            lines.append("")
            for k, v in result.metadata.items():
                lines.append(f"- **{k}:** {v}")
            lines.append("")

        if result.plots:
            lines.append("## Plots")
            lines.append("")
            lines.append(f"*{len(result.plots)} plot(s) generated (base64 PNG).*")
            lines.append("")

        return "\n".join(lines)

    def generate_html_report(self, result: ExperimentResult) -> str:
        """Generate an HTML-formatted report from an ExperimentResult."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        status_color = {
            "success": "#4CAF50",
            "error": "#F44336",
            "partial": "#FF9800",
        }.get(result.status, "#9E9E9E")

        warnings_html = ""
        if result.warnings:
            items = "".join(f"<li>{w}</li>" for w in result.warnings)
            warnings_html = f"<h2>Warnings</h2><ul>{items}</ul>"

        errors_html = ""
        if result.errors:
            items = "".join(f"<li style='color:#F44336'>{e}</li>" for e in result.errors)
            errors_html = f"<h2>Errors</h2><ul>{items}</ul>"

        results_json = json.dumps(self._sanitize_results(result.results), indent=2)

        plots_html = ""
        if result.plots:
            imgs = "".join(
                f'<img src="data:image/png;base64,{p}" style="max-width:700px;margin:10px 0;border:1px solid #ddd;" /><br/>'
                for p in result.plots
            )
            plots_html = f"<h2>Plots</h2>{imgs}"

        metadata_html = ""
        if result.metadata:
            rows = "".join(
                f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
                for k, v in result.metadata.items()
            )
            metadata_html = f"<h2>Metadata</h2><table border='1' cellpadding='6'>{rows}</table>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>SIMLAB Report — {result.experiment_id}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
  h1 {{ color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 8px; }}
  h2 {{ color: #1976D2; }}
  .meta {{ background: #f5f5f5; padding: 12px 16px; border-radius: 4px; margin-bottom: 20px; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; color: white; background: {status_color}; font-weight: bold; }}
  pre {{ background: #1E1E1E; color: #D4D4D4; padding: 16px; border-radius: 6px; overflow-x: auto; font-size: 13px; }}
  li {{ margin: 4px 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td {{ padding: 6px 10px; }}
</style>
</head>
<body>
<h1>SIMLAB Experiment Report</h1>
<div class="meta">
  <b>Experiment ID:</b> <code>{result.experiment_id}</code><br/>
  <b>Generated:</b> {now}<br/>
  <b>Domain:</b> {result.domain}<br/>
  <b>Type:</b> {result.type}<br/>
  <b>Status:</b> <span class="badge">{result.status}</span>
</div>
{warnings_html}
{errors_html}
<h2>Results</h2>
<pre>{results_json}</pre>
{metadata_html}
{plots_html}
</body>
</html>"""
        return html

    def export_json(self, result: ExperimentResult) -> str:
        """Export the experiment result as a JSON string."""
        data = result.model_dump()
        # Truncate plot data in JSON output for readability
        if data.get("plots"):
            data["plots"] = [
                f"<base64 PNG, {len(p)} chars>" for p in data["plots"]
            ]
        return json.dumps(data, indent=2, default=str)

    @staticmethod
    def _sanitize_results(results: dict) -> dict:
        """Truncate very long array values for display in reports."""
        sanitized = {}
        for k, v in results.items():
            if isinstance(v, list) and len(v) > 20:
                sanitized[k] = v[:5] + ["..."] + v[-3:]
            else:
                sanitized[k] = v
        return sanitized
