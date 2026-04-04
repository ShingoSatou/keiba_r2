from __future__ import annotations

import html
import json
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote

from keiba_research.common.assets import asset_relative, cache_root, read_json, run_paths, write_json
from keiba_research.evaluation.execution_report import write_execution_report

DEFAULT_VIEW_HOST = "127.0.0.1"
DEFAULT_VIEW_PORT = 8765


def build_report_view(
    run_ids: list[str],
    *,
    output_html: str | None = None,
    refresh: bool = False,
) -> Path:
    normalized = [str(run_id).strip() for run_id in run_ids if str(run_id).strip()]
    if not normalized:
        raise SystemExit("at least one --run-id is required")
    if len(normalized) > 2:
        raise SystemExit("report-view supports at most two --run-id values")

    reports = [_load_report(run_id, refresh=refresh) for run_id in normalized]
    html_path = _resolve_view_output(run_ids=normalized, output_html=output_html)
    page = _render_single_report_page(reports[0]) if len(reports) == 1 else _render_compare_page(
        reports[0], reports[1]
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")

    manifest_path = html_path.parent / "index.json"
    manifest = read_json(manifest_path, default={"pages": {}})
    pages = dict(manifest.get("pages") or {})
    pages[html_path.name] = {"run_ids": normalized, "path": asset_relative(html_path)}
    write_json(manifest_path, {"pages": pages})
    return html_path


def serve_report_view(
    *,
    html_path: Path,
    host: str = DEFAULT_VIEW_HOST,
    port: int = DEFAULT_VIEW_PORT,
    open_browser: bool = False,
) -> int:
    directory = html_path.parent
    relative = quote(html_path.name)
    url = f"http://{host}:{int(port)}/{relative}"
    if open_browser:
        webbrowser.open(url)

    handler = partial(_QuietSimpleHTTPRequestHandler, directory=str(directory))
    try:
        with _ReusableHTTPServer((host, int(port)), handler) as httpd:
            print(f"Execution report viewer: {url}")
            httpd.serve_forever()
    except OSError as exc:
        raise SystemExit(f"failed to start viewer on {host}:{port}: {exc}") from exc
    return 0


def _load_report(run_id: str, *, refresh: bool) -> dict[str, Any]:
    run = run_paths(run_id)
    if refresh or not run["execution_report_summary"].exists() or not run["execution_report_detail"].exists():
        write_execution_report(run_id)
    summary = read_json(run["execution_report_summary"])
    detail = read_json(run["execution_report_detail"])
    if not summary or not detail:
        raise SystemExit(f"execution report is missing or empty for run_id={run_id}")
    return {"run_id": run_id, "summary": summary, "detail": detail}


def _resolve_view_output(*, run_ids: list[str], output_html: str | None) -> Path:
    if str(output_html or "").strip():
        path = Path(str(output_html).strip()).resolve()
        asset_relative(path)
        return path
    slug = run_ids[0] if len(run_ids) == 1 else f"{run_ids[0]}__vs__{run_ids[1]}"
    return cache_root() / "report_view" / f"{slug}.html"


def _render_single_report_page(report: dict[str, Any]) -> str:
    summary = report["summary"]
    detail = report["detail"]
    conditions = dict(summary.get("conditions") or {})
    pipeline = dict(summary.get("pipeline") or {})
    coverage = dict(summary.get("coverage_summary") or {})
    quality = dict(summary.get("quality_summary") or {})
    issues = list(detail.get("issues") or [])
    backtest = dict(summary.get("backtest_summary") or {})

    binary_rows = [
        (
            name,
            metrics.get("logloss"),
            metrics.get("brier"),
            metrics.get("auc"),
            metrics.get("ece"),
            metrics.get("benter_r2_valid"),
        )
        for name, metrics in (quality.get("binary") or {}).items()
    ]
    stack_rows = [
        (
            name,
            metrics.get("logloss"),
            metrics.get("brier"),
            metrics.get("auc"),
            metrics.get("ece"),
        )
        for name, metrics in (quality.get("stack") or {}).items()
    ]
    pl_rows = [
        (
            name,
            metrics.get("pl_nll_valid"),
            metrics.get("top3_logloss"),
            metrics.get("top3_brier"),
            metrics.get("top3_auc"),
            metrics.get("top3_ece"),
        )
        for name, metrics in (quality.get("pl") or {}).items()
    ]
    backtest_rows = [
        (
            name,
            metrics.get("period_from"),
            metrics.get("period_to"),
            metrics.get("n_races"),
            metrics.get("n_bets"),
            metrics.get("n_hits"),
            metrics.get("hit_rate"),
            metrics.get("total_bet"),
            metrics.get("total_return"),
            metrics.get("roi"),
            metrics.get("max_drawdown"),
        )
        for name, metrics in backtest.items()
    ]

    hero_cards = [
        ("Status", summary.get("status")),
        ("Feature Build", f"{conditions.get('feature_profile')} / {conditions.get('feature_build_id')}"),
        ("Holdout Year", conditions.get("holdout_year")),
        ("Backtest Inputs", ", ".join(sorted(backtest.keys())) if backtest else "none"),
        ("ROI", ((backtest.get("pl_holdout") or {}).get("roi"))),
        ("Bets", ((backtest.get("pl_holdout") or {}).get("n_bets"))),
    ]

    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(summary.get("title"))}</title>
  <link rel="icon" href="{_inline_favicon()}">
  <style>{_shared_css()}</style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="pill status-{_status_slug(summary.get('status'))}">status: {_escape(summary.get("status"))}</div>
      <h1>{_escape(summary.get("title"))}</h1>
      <p class="subtitle">{_escape(summary.get("description"))}</p>
      <div class="cards">
        {_metric_cards(hero_cards)}
      </div>
    </section>

    <section class="layout section-gap">
      <div class="panel">
        <h2>Pipeline</h2>
        <pre>{_escape_json(pipeline)}</pre>
        <h2>Conditions</h2>
        {_key_value_table(conditions)}
      </div>
      <div class="panel">
        <h2>Backtest Summary</h2>
        {_table(
            ["input", "from", "to", "races", "bets", "hits", "hit_rate", "total_bet", "total_return", "roi", "max_drawdown"],
            backtest_rows,
        )}
        <h2>Issues</h2>
        <pre>{_escape_json(issues)}</pre>
      </div>
    </section>

    <section class="layout section-gap">
      <div class="panel">
        <h2>Binary Quality</h2>
        {_table(["model", "logloss", "brier", "auc", "ece", "benter_r2_valid"], binary_rows)}
        <h2>Stack Quality</h2>
        {_table(["task", "logloss", "brier", "auc", "ece"], stack_rows)}
      </div>
      <div class="panel">
        <h2>PL Quality</h2>
        {_table(["profile", "pl_nll_valid", "top3_logloss", "top3_brier", "top3_auc", "top3_ece"], pl_rows)}
        <h2>Coverage</h2>
        <pre>{_escape_json(coverage)}</pre>
      </div>
    </section>

    <section class="panel section-gap">
      <h2>Paths</h2>
      <pre>{_escape_json(summary.get("paths") or {})}</pre>
    </section>
  </div>
</body>
</html>
"""


def _render_compare_page(left: dict[str, Any], right: dict[str, Any]) -> str:
    left_summary = left["summary"]
    right_summary = right["summary"]
    left_detail = left["detail"]
    right_detail = right["detail"]

    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(left_summary.get("run_id"))} vs {_escape(right_summary.get("run_id"))}</title>
  <link rel="icon" href="{_inline_favicon()}">
  <style>{_shared_css()}</style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="pill">compare mode</div>
      <h1>{_escape(left_summary.get("run_id"))} vs {_escape(right_summary.get("run_id"))}</h1>
      <p class="subtitle">Execution report comparison built from curated summary/detail JSON. Delta sign is raw arithmetic only and must be interpreted by metric semantics.</p>
    </section>

    <section class="compare-grid section-gap">
      <div class="panel">
        <h2>{_escape(left_summary.get("run_id"))}</h2>
        {_metric_cards([
            ("Status", left_summary.get("status")),
            ("Feature Build", f"{(left_summary.get('conditions') or {}).get('feature_profile')} / {(left_summary.get('conditions') or {}).get('feature_build_id')}"),
            ("ROI", ((left_summary.get('backtest_summary') or {}).get('pl_holdout') or {}).get('roi')),
            ("Bets", ((left_summary.get('backtest_summary') or {}).get('pl_holdout') or {}).get('n_bets')),
        ])}
        <h2>Pipeline</h2>
        <pre>{_escape_json(left_summary.get("pipeline") or {})}</pre>
        <h2>Issues</h2>
        <pre>{_escape_json(left_detail.get("issues") or [])}</pre>
      </div>
      <div class="panel">
        <h2>{_escape(right_summary.get("run_id"))}</h2>
        {_metric_cards([
            ("Status", right_summary.get("status")),
            ("Feature Build", f"{(right_summary.get('conditions') or {}).get('feature_profile')} / {(right_summary.get('conditions') or {}).get('feature_build_id')}"),
            ("ROI", ((right_summary.get('backtest_summary') or {}).get('pl_holdout') or {}).get('roi')),
            ("Bets", ((right_summary.get('backtest_summary') or {}).get('pl_holdout') or {}).get('n_bets')),
        ])}
        <h2>Pipeline</h2>
        <pre>{_escape_json(right_summary.get("pipeline") or {})}</pre>
        <h2>Issues</h2>
        <pre>{_escape_json(right_detail.get("issues") or [])}</pre>
      </div>
    </section>

    <section class="panel section-gap">
      <h2>Conditions</h2>
      {_compare_table(left_summary.get("conditions") or {}, right_summary.get("conditions") or {})}
    </section>

    <section class="panel section-gap">
      <h2>Quality Summary</h2>
      {_compare_table(left_summary.get("quality_summary") or {}, right_summary.get("quality_summary") or {})}
    </section>

    <section class="panel section-gap">
      <h2>Coverage Summary</h2>
      {_compare_table(left_summary.get("coverage_summary") or {}, right_summary.get("coverage_summary") or {})}
    </section>

    <section class="panel section-gap">
      <h2>Backtest Summary</h2>
      {_compare_table(left_summary.get("backtest_summary") or {}, right_summary.get("backtest_summary") or {})}
    </section>
  </div>
</body>
</html>
"""


def _compare_table(left_payload: dict[str, Any], right_payload: dict[str, Any]) -> str:
    left_flat = _flatten_scalars(left_payload)
    right_flat = _flatten_scalars(right_payload)
    keys = sorted(set(left_flat) | set(right_flat))
    rows = []
    for key in keys:
        left_value = left_flat.get(key)
        right_value = right_flat.get(key)
        delta = (
            float(right_value) - float(left_value)
            if _is_number(left_value) and _is_number(right_value)
            else None
        )
        rows.append((key, left_value, right_value, delta))
    return _table(["field", "left", "right", "delta"], rows)


def _flatten_scalars(payload: Any, *, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_scalars(value, prefix=child))
        return out
    if isinstance(payload, list):
        out[prefix] = payload
        return out
    out[prefix] = payload
    return out


def _key_value_table(payload: dict[str, Any]) -> str:
    return _table(["field", "value"], [(key, value) for key, value in sorted(payload.items())])


def _table(headers: list[str], rows: list[tuple[Any, ...]]) -> str:
    if not rows:
        return '<p class="muted">No data</p>'
    head = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{_fmt_cell(cell)}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _metric_cards(cards: list[tuple[str, Any]]) -> str:
    chunks = []
    for label, value in cards:
        chunks.append(
            "<div class=\"card\">"
            f"<h3>{_escape(label)}</h3>"
            f"<div class=\"big\">{_fmt_cell(value)}</div>"
            "</div>"
        )
    return "".join(chunks)


def _shared_css() -> str:
    return """
:root {
  --bg: #f4f1ea;
  --panel: #fffdf8;
  --ink: #1e1a16;
  --muted: #6d6258;
  --line: #d8cec2;
  --accent: #0d5c63;
  --ok-bg: #edf8f0;
  --ok-fg: #1f7a3f;
  --partial-bg: #fff4df;
  --partial-fg: #8a5a00;
  --invalid-bg: #fdecea;
  --invalid-fg: #a32525;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: linear-gradient(180deg, #ece5da 0%, var(--bg) 55%);
  color: var(--ink);
  font-family: "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
}
.shell { max-width: 1380px; margin: 0 auto; padding: 28px 22px 56px; }
.hero, .panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 24px;
  box-shadow: 0 16px 32px rgba(36, 24, 12, 0.06);
}
.hero { padding: 28px; }
.panel { padding: 22px; }
.section-gap { margin-top: 18px; }
.layout, .compare-grid { display: grid; gap: 18px; }
.layout { grid-template-columns: 1.05fr 0.95fr; }
.compare-grid { grid-template-columns: 1fr 1fr; }
h1 {
  margin: 10px 0 8px;
  font-size: 32px;
  line-height: 1.12;
  letter-spacing: -0.03em;
  font-family: "Iowan Old Style", "Yu Mincho", "Times New Roman", serif;
}
h2 { margin: 0 0 12px; font-size: 20px; }
h3 {
  margin: 0 0 8px;
  font-size: 13px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}
.subtitle { margin: 0; max-width: 1000px; color: var(--muted); line-height: 1.55; }
.cards {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin-top: 18px;
}
.card {
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 16px;
  background: #fff9f1;
}
.big { font-size: 28px; font-weight: 700; }
.pill {
  display: inline-flex;
  border-radius: 999px;
  padding: 8px 12px;
  border: 1px solid var(--line);
  background: #f8f3e8;
  font-weight: 700;
}
.status-complete { background: var(--ok-bg); color: var(--ok-fg); border-color: #b8d8bf; }
.status-partial { background: var(--partial-bg); color: var(--partial-fg); border-color: #eed4a4; }
.status-invalid { background: var(--invalid-bg); color: var(--invalid-fg); border-color: #f0b8b2; }
.muted { color: var(--muted); }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  line-height: 1.55;
  background: #f8f5ef;
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 16px;
}
@media (max-width: 980px) {
  .layout, .compare-grid { grid-template-columns: 1fr; }
  h1 { font-size: 26px; }
  .big { font-size: 24px; }
}
"""


def _fmt_cell(value: Any) -> str:
    if value is None:
        return '<span class="muted">-</span>'
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return _escape(", ".join(str(item) for item in value))
    return _escape(value)


def _escape_json(payload: Any) -> str:
    return _escape(json.dumps(payload, ensure_ascii=False, indent=2))


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _inline_favicon() -> str:
    return "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='16' fill='%230d5c63'/%3E%3Cpath d='M18 46V18h8v12h12V18h8v28h-8V37H26v9z' fill='%23fffdf8'/%3E%3C/svg%3E"


def _status_slug(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text if text in {"complete", "partial", "invalid"} else "unknown"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class _ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class _QuietSimpleHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return
