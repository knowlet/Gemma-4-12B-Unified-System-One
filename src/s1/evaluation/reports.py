"""Standalone escaped HTML/Markdown reports built only from saved observations."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

from .artifacts import write_json
from .reporting import summarize_run


def _number(value):
    return "—" if value is None else f"{value:.4g}"


def _svg(points, title, x_label, y_label, *, x_max=1, y_max=1):
    dots = []
    for label, x, y in points:
        dots.append(
            f'<circle cx="{55 + x / max(x_max, 1e-12) * 430:.2f}" cy="{240 - y / max(y_max, 1e-12) * 200:.2f}" r="4"><title>{html.escape(label)}: {x:.4g}, {y:.4g}</title></circle>'
        )
    return f'''<figure><figcaption>{html.escape(title)}</figcaption>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 300" role="img" aria-label="{html.escape(title, quote=True)}">
<path d="M55 30V240H500" fill="none" stroke="currentColor"/>
<text x="55" y="260">0</text><text x="445" y="260">{x_max:.3g}</text>
<text x="15" y="40">{y_max:.3g}</text><text x="220" y="290">{html.escape(x_label)}</text>
<text x="60" y="20">{html.escape(y_label)}</text><g fill="#2563eb">{"".join(dots)}</g></svg></figure>'''


def build_report(directories, output):
    groups, workflows = defaultdict(list), []
    for directory in map(Path, directories):
        if (directory / "workflow_manifest.json").exists():
            workflows.append(
                {
                    "manifest": json.loads((directory / "workflow_manifest.json").read_text()),
                    "summary": json.loads((directory / "workflow_summary.json").read_text()),
                }
            )
            continue
        summary = summarize_run(directory)
        manifest = json.loads((directory / "run_manifest.json").read_text())
        for cell in manifest["plan"]["models"]:
            identity = cell["identity"]
            model = identity["model"]
            key = (
                identity["suite"]["track"],
                identity["suite"]["information_view"],
                identity["dataset_sha256"],
                model.get("hardware_label") or "hardware_unspecified",
                model.get("service_region") or "region_unspecified",
            )
            groups[key].append(
                {
                    "model_id": cell["model_id"],
                    "experiment_id": cell["experiment_id"],
                    "model": model,
                    "profile": identity["profile"],
                    "summary": summary["models"][cell["model_id"]],
                }
            )
    sections, markdown = (
        [],
        [
            "# Decision Benchmark v2 report",
            "",
            "Saved observations only; fixtures are not model quality evidence.",
            "",
        ],
    )
    for key, models in sorted(groups.items()):
        title = " / ".join((key[0], key[1], key[2][:12], key[3], key[4]))
        table, points = [], []
        markdown += [
            f"## {title}",
            "",
            "| Model | Status | Valid/eligible | Accuracy | Operational | p95 ms | Goodput |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        charts = []
        for item in models:
            s = item["summary"]
            profile = item["profile"]
            label = (
                f"{item['model_id']} · {profile.get('load_mode', 'closed_loop')} "
                f"B{profile['batch_size']}/C{profile['concurrency']} · {item['experiment_id'][:8]}"
            )
            latency = (s["successful_request_latency_ms"] or {}).get("p95")
            quality = s["quality"]
            operational = quality["operational_correctness"]
            coverage = (
                f"{s['denominators']['valid_decisions']}/{s['denominators']['eligible_decisions']}"
            )
            values = [
                label,
                s["status"],
                coverage,
                _number(quality["actual_label_accuracy"]),
                _number(operational),
                _number(latency),
                _number((s.get("systems") or {}).get("correct_within_slo_per_second")),
            ]
            table.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")
            markdown.append(
                "| " + " | ".join(v.replace("|", "\\|").replace("\n", " ") for v in values) + " |"
            )
            if latency is not None and operational is not None and s["records_complete"]:
                points.append((label, latency, operational))
            reliability = [
                (f"{label} n={b['n']}", b["confidence"], b["accuracy"])
                for b in s["metrics"]["reliability"]
                if b["n"]
            ]
            risk = [(label, p["coverage"], p["risk"]) for p in s["metrics"]["selective"]["points"]]
            if reliability:
                charts.append(_svg(reliability, f"{label}: reliability", "confidence", "accuracy"))
            if risk:
                charts.append(_svg(risk, f"{label}: risk–coverage", "coverage", "risk"))
        frontier = [
            p[0]
            for p in points
            if not any(
                q[1] <= p[1] and q[2] >= p[2] and (q[1] < p[1] or q[2] > p[2]) for q in points
            )
        ]
        frontier_note = (
            "Descriptive frontier: "
            + (", ".join(frontier) or "not available")
            + ". No significance claim; verify common eligibility and hardware before comparison."
        )
        sections.append(
            f"<section><h2>{html.escape(title)}</h2><table><thead><tr><th>Model</th><th>Status</th><th>Valid/eligible</th><th>Accuracy</th><th>Operational</th><th>p95 ms</th><th>Goodput</th></tr></thead><tbody>{''.join(table)}</tbody></table><p>{html.escape(frontier_note)}</p>"
            + (
                _svg(
                    points,
                    "Latency / operational correctness",
                    "p95 ms",
                    "operational correctness",
                    x_max=max(p[1] for p in points),
                )
                if points
                else ""
            )
            + "".join(charts)
            + "</section>"
        )
        markdown += ["", frontier_note, ""]
    for workflow in workflows:
        content = json.dumps(workflow, indent=2, ensure_ascii=False)
        sections.append(
            f"<section><h2>Closed-loop replay</h2><pre>{html.escape(content)}</pre></section>"
        )
        markdown += ["## Closed-loop replay", "", "```json", content, "```", ""]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Decision Benchmark v2</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#172033;background:#fafbfe}section{background:white;padding:24px;margin:24px 0;border:1px solid #dbe2ef;border-radius:12px}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #dbe2ef;padding:10px}figure{display:inline-block;width:min(100%,480px);vertical-align:top;margin:16px 0}svg{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}h2{overflow-wrap:anywhere}</style>
<h1>Decision Benchmark v2</h1><p>Saved observations. Separate tracks and information views; missing runs stay visible. Fixture results do not establish pretrained model quality. Costs labeled as ceilings are not invoices.</p>"""
    (output / "report.html").write_text(page + "".join(sections) + "</html>", encoding="utf-8")
    result = {
        "groups": len(groups),
        "workflow_runs": len(workflows),
        "html": str(output / "report.html"),
        "markdown": str(output / "report.md"),
    }
    write_json(output / "report.json", result)
    return result
