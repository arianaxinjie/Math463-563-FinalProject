"""Zero-dependency dashboard server for grid-search monitoring."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse

from core_code.data import generate_blurred_noisy_cfg, load_image
from core_code.optsolve import OptParams, optsolve

from .grid_search_status import load_status

DEFAULT_OUTPUT_DIR = "output/grid_search/full_run_l1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_RECENT_ROWS = 24
DEFAULT_LOG_LINES = 40
IMAGE_CACHE_DIRNAME = "dashboard_cache"
RESULTS_FILENAME = "results.csv"
_CACHE_LOCK = threading.Lock()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_json_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    deduped: dict[str, dict[str, str]] = {}
    order: list[str] = []
    passthrough: list[dict[str, str]] = []
    for row in rows:
        key = row.get("trial_key")
        if not key:
            passthrough.append(row)
            continue
        if key not in deduped:
            order.append(key)
        deduped[key] = row
    return passthrough + [deduped[key] for key in order]


def _tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-limit:]


def _pid_state(pid_file: Path | None) -> dict[str, Any]:
    if pid_file is None or not pid_file.exists():
        return {"pid": None, "running": False, "command": None}

    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return {"pid": None, "running": False, "command": None}

    running = False
    command = None
    try:
        os.kill(pid, 0)
        running = True
    except OSError:
        running = False

    if running:
        try:
            command = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "command="],
                text=True,
            ).strip() or None
        except Exception:
            command = None

    return {"pid": pid, "running": running, "command": command}


def _save_png(array: Any, path: Path) -> None:
    from PIL import Image
    import numpy as np

    x = np.asarray(array, dtype=np.float64)
    x = np.clip(x, 0.0, 1.0)
    u8 = (x * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u8, mode="L").save(path)


def _load_observation_from_status(
    status: dict[str, Any],
    row: dict[str, str] | None = None,
) -> tuple[Any, Any, Any]:
    observation = dict((status.get("config") or {}).get("observation") or {})
    if row is not None:
        search_values = _parse_json_dict(row.get("search_values_json"))
        for key in (
            "image_path",
            "kernel_kind",
            "kernel_size",
            "kernel_sigma",
            "noise_type",
            "noise_density",
            "noise_mean",
            "noise_sigma",
            "mode",
        ):
            if key in search_values:
                observation[key] = search_values[key]
    image_path = observation["image_path"]
    x_true = load_image(image_path)
    kernel, x_blur, b = generate_blurred_noisy_cfg(
        x_true=x_true,
        kernel_kind=observation.get("kernel_kind", "gaussian"),
        kernel_size=observation.get("kernel_size", 15),
        kernel_sigma=observation.get("kernel_sigma", 3.0),
        noise_type=observation.get("noise_type", "salt_pepper"),
        noise_density=observation.get("noise_density", 0.05),
        noise_mean=observation.get("noise_mean", 0.0),
        noise_sigma=observation.get("noise_sigma", 0.01),
        mode=observation.get("mode", "periodic"),
    )
    return x_true, kernel, b, x_blur


def _best_overall_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("psnr")
    ]
    if not ok_rows:
        return None
    return max(ok_rows, key=lambda row: float(row["psnr"]))


def _latest_ok_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in reversed(rows):
        if row.get("status") == "ok":
            return row
    return None


def _render_row_image(row: dict[str, str], kernel: Any, b: Any) -> Any:
    params = json.loads(row["params_json"])
    x_rec, _hist, _info = optsolve(
        row["problem"],
        row["algorithm"],
        None,
        kernel,
        b,
        OptParams(**params),
        return_all=True,
    )
    return x_rec


def _image_cache_paths(output_dir: Path) -> dict[str, Path]:
    root = output_dir / IMAGE_CACHE_DIRNAME
    return {
        "root": root,
        "original": root / "original.png",
        "blurred": root / "blurred.png",
        "best": root / "best.png",
        "latest": root / "latest.png",
        "meta": root / "meta.json",
    }


def _image_cache_state(output_dir: Path) -> dict[str, Any]:
    status = load_status(output_dir)
    rows = _read_results(output_dir / RESULTS_FILENAME)
    cache = _image_cache_paths(output_dir)
    meta = _read_json(cache["meta"])
    best_row = _best_overall_row(rows)
    latest_row = _latest_ok_row(rows)
    desired = {
        "original_key": json.dumps(
            (status.get("config") or {}).get("observation") or {},
            sort_keys=True,
        ),
        "blurred_key": json.dumps(
            (status.get("config") or {}).get("observation") or {},
            sort_keys=True,
        ),
        "best_key": best_row.get("trial_key") if best_row else None,
        "latest_key": latest_row.get("trial_key") if latest_row else None,
    }
    return {
        "status": status,
        "rows": rows,
        "cache": cache,
        "meta": meta,
        "best_row": best_row,
        "latest_row": latest_row,
        "desired": desired,
    }


def _ensure_base_images(state: dict[str, Any]) -> None:
    cache = state["cache"]
    meta = state["meta"]
    desired = state["desired"]
    x_true, _kernel, _b, x_blur = _load_observation_from_status(state["status"])

    with _CACHE_LOCK:
        cache["root"].mkdir(parents=True, exist_ok=True)

        if meta.get("original_key") != desired["original_key"] or not cache["original"].exists():
            _save_png(x_true, cache["original"])
            meta["original_key"] = desired["original_key"]

        if meta.get("blurred_key") != desired["blurred_key"] or not cache["blurred"].exists():
            _save_png(x_blur, cache["blurred"])
            meta["blurred_key"] = desired["blurred_key"]

        cache["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _ensure_named_image(output_dir: Path, name: str) -> Path | None:
    state = _image_cache_state(output_dir)
    cache = state["cache"]
    meta = state["meta"]
    desired = state["desired"]

    if name in {"original", "blurred"}:
        _ensure_base_images(state)
        return cache[name]

    if name not in {"best", "latest"}:
        return None

    cached_path = cache[name]
    row = state["best_row"] if name == "best" else state["latest_row"]
    key_name = f"{name}_key"
    if row is None or not desired.get(key_name):
        return None

    _ensure_base_images(state)
    _x_true, kernel, b, _x_blur = _load_observation_from_status(state["status"], row=row)

    with _CACHE_LOCK:
        if meta.get(key_name) == desired[key_name] and cached_path.exists():
            return cached_path
        _save_png(_render_row_image(row, kernel, b), cached_path)
        meta[key_name] = desired[key_name]
        cache["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return cached_path


def _ensure_image_cache(output_dir: Path) -> dict[str, Any]:
    state = _image_cache_state(output_dir)
    if not ((state["status"].get("config") or {}).get("observation")):
        return {
            "paths": {key: str(value) for key, value in state["cache"].items()},
            "keys": state["desired"],
            "best_row": state["best_row"],
            "latest_row": state["latest_row"],
        }
    _ensure_base_images(state)

    return {
        "paths": {key: str(value) for key, value in state["cache"].items()},
        "keys": state["desired"],
        "best_row": state["best_row"],
        "latest_row": state["latest_row"],
    }


def collect_dashboard_data(
    *,
    output_dir: str | Path,
    pid_file: str | Path | None = None,
    log_file: str | Path | None = None,
    recent_rows: int = DEFAULT_RECENT_ROWS,
    log_lines: int = DEFAULT_LOG_LINES,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    status = load_status(out_dir)
    summary = _read_json(out_dir / "summary.json")
    rows = _read_results(out_dir / "results.csv")

    recent = rows[-recent_rows:]
    recent.reverse()
    errors = [row for row in reversed(rows) if row.get("status") == "error"][:12]
    image_cache = _ensure_image_cache(out_dir)

    pid_path = Path(pid_file) if pid_file else None
    log_path = Path(log_file) if log_file else None

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "summary": summary,
        "process": _pid_state(pid_path),
        "log_tail": _tail_lines(log_path, log_lines) if log_path else [],
        "recent_rows": recent,
        "recent_errors": errors,
        "image_cache": image_cache,
        "log_file": str(log_path) if log_path else None,
        "pid_file": str(pid_path) if pid_path else None,
    }


def _dashboard_html(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #efe8dc;
      --panel: #fffaf3;
      --panel-strong: #f8efe2;
      --ink: #1e262e;
      --muted: #62707d;
      --accent: #7b3518;
      --accent-soft: #f4d8c7;
      --good: #27543a;
      --warn: #8a5e16;
      --bad: #8b1e2d;
      --rule: #d7c8b1;
      --shadow: rgba(42, 27, 13, 0.10);
      --code-bg: #f0e4d6;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      background:
        radial-gradient(circle at top left, rgba(123, 53, 24, 0.11), transparent 26%),
        radial-gradient(circle at top right, rgba(39, 84, 58, 0.08), transparent 22%),
        linear-gradient(180deg, #f8f3ec 0%, var(--bg) 100%);
    }}

    .wrap {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 32px 18px 56px;
    }}

    .hero {{
      padding: 26px;
      border: 1px solid var(--rule);
      border-radius: 22px;
      background: linear-gradient(135deg, rgba(255,250,243,0.98), rgba(246,235,221,0.96));
      box-shadow: 0 18px 40px var(--shadow);
    }}

    h1, h2, h3 {{
      margin: 0 0 12px;
      line-height: 1.15;
    }}

    h1 {{
      font-size: clamp(2rem, 4vw, 3.2rem);
      letter-spacing: -0.03em;
    }}

    h2 {{
      font-size: 1.35rem;
      margin-top: 28px;
    }}

    .lead {{
      color: var(--muted);
      font-size: 1.06rem;
      max-width: 880px;
    }}

    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}

    .pill {{
      background: var(--accent-soft);
      border: 1px solid rgba(123, 53, 24, 0.18);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 0.94rem;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid var(--rule);
      border-radius: 18px;
      box-shadow: 0 10px 26px rgba(42, 27, 13, 0.06);
      overflow: hidden;
    }}

    .panel-body {{
      padding: 18px;
    }}

    .metric-label {{
      color: var(--muted);
      font-size: 0.92rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}

    .metric-value {{
      font-size: 1.8rem;
      font-weight: 700;
      margin-top: 6px;
    }}

    .metric-sub {{
      color: var(--muted);
      margin-top: 6px;
      font-size: 0.94rem;
    }}

    .progress-shell {{
      margin-top: 18px;
      width: 100%;
      height: 16px;
      border-radius: 999px;
      background: rgba(123, 53, 24, 0.10);
      overflow: hidden;
      border: 1px solid rgba(123, 53, 24, 0.08);
    }}

    .progress-bar {{
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #9f4622, #d27331 55%, #ebaa68);
    }}

    .split {{
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 16px;
      margin-top: 18px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}

    th, td {{
      padding: 11px 12px;
      text-align: left;
      border-bottom: 1px solid var(--rule);
      vertical-align: top;
    }}

    th {{
      background: rgba(123, 53, 24, 0.08);
      font-size: 0.88rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }}

    tr:last-child td {{ border-bottom: none; }}

    .status-good {{ color: var(--good); font-weight: 700; }}
    .status-warn {{ color: var(--warn); font-weight: 700; }}
    .status-bad {{ color: var(--bad); font-weight: 700; }}

    .alg-list {{
      display: grid;
      gap: 10px;
    }}

    .alg-item {{
      padding: 12px 14px;
      background: var(--panel-strong);
      border: 1px solid var(--rule);
      border-radius: 14px;
    }}

    .alg-top {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-weight: 700;
    }}

    .alg-meta {{
      color: var(--muted);
      font-size: 0.92rem;
      margin-top: 6px;
    }}

    pre {{
      margin: 0;
      padding: 16px;
      background: var(--code-bg);
      border-top: 1px solid var(--rule);
      overflow-x: auto;
      font-family: "SFMono-Regular", "Menlo", "Monaco", monospace;
      font-size: 0.9rem;
      line-height: 1.45;
      color: #2f3b48;
    }}

    .muted {{
      color: var(--muted);
    }}

    .wide-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 18px;
    }}

    .image-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}

    .image-card {{
      background: var(--panel);
      border: 1px solid var(--rule);
      border-radius: 18px;
      box-shadow: 0 10px 26px rgba(42, 27, 13, 0.06);
      overflow: hidden;
    }}

    .image-card img {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #e9ddce;
    }}

    .image-caption {{
      padding: 14px 16px 16px;
      border-top: 1px solid var(--rule);
    }}

    @media (max-width: 980px) {{
      .split, .wide-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <h1>{title}</h1>
      <p class="lead">
        Live view of grid-search progress, ETA, best current results, recent trial records,
        and the latest log tail. This page refreshes itself every 10 seconds.
      </p>
      <div class="meta" id="meta"></div>
      <div class="progress-shell"><div class="progress-bar" id="progress-bar"></div></div>
    </section>

    <section class="grid" id="metrics"></section>

    <section class="split">
      <article class="panel">
        <div class="panel-body">
          <h2>Best By Algorithm</h2>
          <div class="alg-list" id="best-list"></div>
        </div>
      </article>
      <article class="panel">
        <div class="panel-body">
          <h2>Run State</h2>
          <table>
            <tbody id="run-state"></tbody>
          </table>
        </div>
      </article>
    </section>

    <section class="panel" style="margin-top: 18px;">
      <div class="panel-body">
        <h2>Best By Noise</h2>
        <table>
          <thead>
            <tr>
              <th>Noise</th>
              <th>Best Gamma</th>
              <th>PSNR</th>
              <th>Iter</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody id="best-by-noise"></tbody>
        </table>
      </div>
    </section>

    <section class="wide-grid">
      <article class="panel">
        <div class="panel-body">
          <h2>Recent Trials</h2>
          <table>
            <thead>
              <tr>
                <th>Algorithm</th>
                <th>Problem</th>
                <th>Noise</th>
                <th>Gamma</th>
                <th>Status</th>
                <th>PSNR</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody id="recent-rows"></tbody>
          </table>
        </div>
      </article>
      <article class="panel">
        <div class="panel-body">
          <h2>Recent Errors</h2>
          <table>
            <thead>
              <tr>
                <th>Algorithm</th>
                <th>Problem</th>
                <th>Error</th>
              </tr>
            </thead>
            <tbody id="recent-errors"></tbody>
          </table>
        </div>
      </article>
    </section>

    <section class="wide-grid">
      <article class="panel">
        <div class="panel-body">
          <h2>Per-Algorithm Progress</h2>
          <table>
            <thead>
              <tr>
                <th>Algorithm</th>
                <th>Counts</th>
              </tr>
            </thead>
            <tbody id="algorithm-progress"></tbody>
          </table>
        </div>
      </article>
      <article class="panel">
        <div class="panel-body">
          <h2>Current Config</h2>
          <table>
            <tbody id="config-table"></tbody>
          </table>
        </div>
      </article>
    </section>

    <section class="image-grid" id="image-grid"></section>

    <section class="panel" style="margin-top: 18px;">
      <div class="panel-body">
        <h2>Log Tail</h2>
        <div class="muted" id="log-path"></div>
      </div>
      <pre id="log-tail"></pre>
    </section>
  </main>

  <script>
    const fmtNumber = (value, digits = 2) => {{
      if (value === null || value === undefined || value === "") return "—";
      const n = Number(value);
      if (!Number.isFinite(n)) return "—";
      return n.toFixed(digits);
    }};

    const formatParamValue = (value) => {{
      if (typeof value === "number") {{
        if (!Number.isFinite(value)) return "—";
        if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 1e-3)) {{
          return value.toExponential(3);
        }}
        return Number(value).toFixed(6).replace(/\\.0+$/, "").replace(/(\\.\\d*?)0+$/, "$1");
      }}
      return String(value);
    }};

    const formatParamMap = (obj, skipKeys = []) => {{
      if (!obj) return "—";
      const skip = new Set(skipKeys);
      const parts = Object.entries(obj)
        .filter(([key, value]) => !skip.has(key) && value !== null && value !== undefined)
        .map(([key, value]) => `${{key}}=${{formatParamValue(value)}}`);
      return parts.length ? parts.join(", ") : "—";
    }};

    const parseJsonObject = (value) => {{
      if (!value) return {{}};
      try {{
        const parsed = JSON.parse(value);
        return parsed && typeof parsed === "object" ? parsed : {{}};
      }} catch (_err) {{
        return {{}};
      }}
    }};

    const statusClass = (name) => {{
      if (name === "ok" || name === true) return "status-good";
      if (name === "error" || name === false) return "status-bad";
      return "status-warn";
    }};

    const setRows = (id, rows) => {{
      const root = document.getElementById(id);
      root.innerHTML = rows.join("");
    }};

    let imageState = {{}};
    const rowSearchValues = (row) => parseJsonObject(row.search_values_json);
    const rowParams = (row) => parseJsonObject(row.params_json);
    const rowNoise = (row) => {{
      const search = rowSearchValues(row);
      const value = row.noise_density ?? search.noise_density;
      return value === undefined || value === null || value === "" ? "—" : formatParamValue(Number(value));
    }};
    const rowGamma = (row) => {{
      const search = rowSearchValues(row);
      const params = rowParams(row);
      const value = row.gamma_value ?? search.gamma ?? params.gamma;
      return value === undefined || value === null || value === "" ? "—" : formatParamValue(Number(value));
    }};
    const rowNoiseSuffix = (row) => {{
      const rendered = rowNoise(row);
      return rendered === "—" ? "" : ` | noise=${{rendered}}`;
    }};

    const render = (payload) => {{
      const status = payload.status || {{}};
      const summary = payload.summary || {{}};
      const process = payload.process || {{}};
      const bestByAlgorithm = summary.best_by_algorithm || {{}};
      const completionPct = ((status.completion_ratio || 0) * 100).toFixed(2);

      document.getElementById("progress-bar").style.width = `${{completionPct}}%`;

      document.getElementById("meta").innerHTML = `
        <span class="pill">Updated: ${{payload.generated_at || "—"}}</span>
        <span class="pill">Output: ${{status.output_dir || "—"}}</span>
        <span class="pill">Workers: ${{status.workers ?? "—"}}</span>
        <span class="pill">PID: ${{process.pid ?? "—"}}</span>
      `;

      const statusCounts = Object.entries(status.status_counts || {{}})
        .map(([k, v]) => `${{k}}=${{v}}`)
        .join(", ") || "—";
      const problemCounts = Object.entries(status.problem_counts || {{}})
        .map(([k, v]) => `${{k}}=${{v}}`)
        .join(", ") || "—";

      document.getElementById("metrics").innerHTML = `
        <article class="panel"><div class="panel-body"><div class="metric-label">Progress</div><div class="metric-value">${{status.completed_trials ?? 0}} / ${{status.total_trials ?? 0}}</div><div class="metric-sub">${{completionPct}}% complete</div></div></article>
        <article class="panel"><div class="panel-body"><div class="metric-label">Remaining</div><div class="metric-value">${{status.remaining_trials ?? 0}}</div><div class="metric-sub">wall ETA ${{status.eta_timestamp || "unknown"}}</div></div></article>
        <article class="panel"><div class="panel-body"><div class="metric-label">Average Trial</div><div class="metric-value">${{fmtNumber(status.avg_trial_sec)}}s</div><div class="metric-sub">per completed trial</div></div></article>
        <article class="panel"><div class="panel-body"><div class="metric-label">Status Counts</div><div class="metric-value">${{statusCounts}}</div><div class="metric-sub">problem split: ${{problemCounts}}</div></div></article>
      `;

      const bestHtml = Object.entries(bestByAlgorithm).sort().map(([algorithm, best]) => `
        <div class="alg-item">
          <div class="alg-top">
            <span>${{algorithm}}</span>
            <span>${{fmtNumber(best.psnr, 4)}} dB</span>
          </div>
          <div class="alg-meta">problem=${{best.problem}} | iter=${{best.iterations}} | time=${{fmtNumber(best.time_sec)}}s</div>
          <div class="alg-meta">search=${{formatParamMap(best.search_values, ["problem", "operator_norm_sq"])}}</div>
          <div class="alg-meta">solver=${{formatParamMap(best.params, ["maxiter", "tol", "verbose", "compute_obj_every"])}}</div>
        </div>
      `);
      document.getElementById("best-list").innerHTML = bestHtml.join("") || '<div class="muted">No completed best rows yet.</div>';

      const bestByNoise = summary.best_by_noise || {{}};
      const bestByNoiseRows = Object.entries(bestByNoise)
        .sort((a, b) => Number(a[0]) - Number(b[0]))
        .map(([noise, best]) => `
          <tr>
            <td>${{formatParamValue(Number(noise))}}</td>
            <td>${{formatParamValue(best.search_values?.gamma ?? best.params?.gamma)}}</td>
            <td>${{fmtNumber(best.psnr, 4)}}</td>
            <td>${{best.iterations ?? "—"}}</td>
            <td>${{fmtNumber(best.time_sec)}}</td>
          </tr>
        `);
      setRows(
        "best-by-noise",
        bestByNoiseRows.length
          ? bestByNoiseRows
          : ['<tr><td colspan="5" class="muted">No per-noise results yet.</td></tr>'],
      );

      setRows("run-state", [
        `<tr><th>Process</th><td class="${{statusClass(process.running)}}">${{process.running ? "running" : "stopped"}}</td></tr>`,
        `<tr><th>PID File</th><td>${{payload.pid_file || "—"}}</td></tr>`,
        `<tr><th>Log File</th><td>${{payload.log_file || "—"}}</td></tr>`,
        `<tr><th>Estimated Finish</th><td>${{status.eta_timestamp || "unknown"}}</td></tr>`,
        `<tr><th>Status Counts</th><td>${{statusCounts}}</td></tr>`,
        `<tr><th>Command</th><td>${{process.command || "—"}}</td></tr>`,
      ]);

      setRows("recent-rows", (payload.recent_rows || []).map((row) => `
        <tr>
          <td>${{row.algorithm || "—"}}</td>
          <td>${{row.problem || "—"}}</td>
          <td>${{rowNoise(row)}}</td>
          <td>${{rowGamma(row)}}</td>
          <td class="${{statusClass(row.status)}}">${{row.status || "—"}}</td>
          <td>${{row.psnr || "—"}}</td>
          <td>${{row.time_sec || "—"}}</td>
        </tr>
      `) || ['<tr><td colspan="7" class="muted">No rows yet.</td></tr>']);

      setRows("recent-errors", (payload.recent_errors || []).map((row) => `
        <tr>
          <td>${{row.algorithm || "—"}}</td>
          <td>${{row.problem || "—"}}</td>
          <td>${{row.error || "—"}}</td>
        </tr>
      `) || ['<tr><td colspan="3" class="muted">No errors yet.</td></tr>']);

      setRows("algorithm-progress", Object.entries(status.algorithm_problem_counts || {{}}).sort().map(([algorithm, counts]) => {{
        const rendered = Object.entries(counts).sort().map(([problem, count]) => `${{problem}}=${{count}}`).join(", ");
        return `<tr><td>${{algorithm}}</td><td>${{rendered || "—"}}</td></tr>`;
      }}) || ['<tr><td colspan="2" class="muted">No progress yet.</td></tr>']);

      const config = status.config || {{}};
      setRows("config-table", [
        `<tr><th>Search Type</th><td>${{config.search_type || "grid_search"}}</td></tr>`,
        `<tr><th>Problems</th><td>${{config.problem || (config.problems || []).join(", ") || "—"}}</td></tr>`,
        `<tr><th>Algorithm</th><td>${{config.algorithm || (config.algorithms || []).join(", ") || "—"}}</td></tr>`,
        `<tr><th>Max Iter</th><td>${{config.maxiter ?? "—"}}</td></tr>`,
        `<tr><th>Tolerance</th><td>${{config.tol ?? "—"}}</td></tr>`,
        `<tr><th>Workers</th><td>${{config.workers ?? "—"}}</td></tr>`,
        `<tr><th>Compute Obj Every</th><td>${{config.compute_obj_every ?? "—"}}</td></tr>`,
        `<tr><th>Image</th><td>${{config.observation?.image_path || "—"}}</td></tr>`,
        `<tr><th>Noise Sweep</th><td>${{(config.noise_densities || []).length ? (config.noise_densities || []).map((value) => formatParamValue(Number(value))).join(", ") : "—"}}</td></tr>`,
        `<tr><th>Gamma Sweep</th><td>${{(config.gamma_values || []).length ? (config.gamma_values || []).map((value) => formatParamValue(Number(value))).join(", ") : "—"}}</td></tr>`,
        `<tr><th>Noise</th><td>${{config.observation ? `${{config.observation.noise_type}} density=${{config.observation.noise_density}}` : "—"}}</td></tr>`,
        `<tr><th>Blur</th><td>${{config.observation ? `${{config.observation.kernel_kind}} size=${{config.observation.kernel_size}} sigma=${{config.observation.kernel_sigma}}` : "—"}}</td></tr>`,
      ]);

      document.getElementById("log-path").textContent = payload.log_file || "No log file configured.";
      document.getElementById("log-tail").textContent = (payload.log_tail || []).join("\\n") || "No log lines.";

      const imageCache = payload.image_cache || {{}};
      const keys = imageCache.keys || {{}};
      const bestRow = imageCache.best_row || null;
      const latestRow = imageCache.latest_row || null;
      const imageSpecs = [
        {{
          name: "original",
          title: "Original",
          subtitle: config.observation?.image_path || "source image",
          key: keys.original_key || "original",
        }},
        {{
          name: "blurred",
          title: "Blurred + Noisy",
          subtitle: config.observation ? `${{config.observation.kernel_kind}} blur + ${{config.observation.noise_type}}` : "synthetic observation",
          key: keys.blurred_key || "blurred",
        }},
        {{
          name: "best",
          title: "Best Overall Completed Trial",
          subtitle: bestRow ? `${{bestRow.algorithm}} | ${{bestRow.problem}}${{rowNoiseSuffix(bestRow)}} | PSNR=${{bestRow.psnr}}` : "not available yet",
          key: keys.best_key || "best-none",
        }},
        {{
          name: "latest",
          title: "Latest Completed Trial",
          subtitle: latestRow ? `${{latestRow.algorithm}} | ${{latestRow.problem}}${{rowNoiseSuffix(latestRow)}} | PSNR=${{latestRow.psnr || "—"}}` : "not available yet",
          key: keys.latest_key || "latest-none",
        }},
      ];

      document.getElementById("image-grid").innerHTML = imageSpecs.map((spec) => {{
        const src = `/image/${{spec.name}}.png?key=${{encodeURIComponent(spec.key)}}`;
        imageState[spec.name] = src;
        return `
          <article class="image-card">
            <img src="${{src}}" alt="${{spec.title}}" />
            <div class="image-caption">
              <strong>${{spec.title}}</strong>
              <div class="alg-meta">${{spec.subtitle}}</div>
            </div>
          </article>
        `;
      }}).join("");
    }};

    const refresh = async () => {{
      try {{
        const res = await fetch("/api/status");
        const payload = await res.json();
        render(payload);
      }} catch (err) {{
        document.getElementById("log-tail").textContent = `Failed to load dashboard data: ${{err}}`;
      }}
    }};

    refresh();
    setInterval(refresh, 10000);
  </script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    output_dir: Path
    pid_file: Path | None
    log_file: Path | None
    recent_rows: int
    log_lines: int
    title: str

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_headers_only(
        self,
        *,
        content_type: str,
        content_length: int = 0,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            qs = parse_qs(parsed.query)
            recent_rows = int(qs.get("recent_rows", [self.recent_rows])[0])
            log_lines = int(qs.get("log_lines", [self.log_lines])[0])
            payload = collect_dashboard_data(
                output_dir=self.output_dir,
                pid_file=self.pid_file,
                log_file=self.log_file,
                recent_rows=recent_rows,
                log_lines=log_lines,
            )
            self._send_json(payload)
            return

        if parsed.path.startswith("/image/") and parsed.path.endswith(".png"):
            name = parsed.path.split("/")[-1].replace(".png", "")
            image_path = _ensure_named_image(Path(self.output_dir), name)
            if image_path is None or not image_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
                return
            data = image_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=300")
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path in ("/", "/index.html"):
            self._send_html(_dashboard_html(self.title))
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._send_headers_only(content_type="application/json; charset=utf-8")
            return
        if parsed.path.startswith("/image/") and parsed.path.endswith(".png"):
            self._send_headers_only(content_type="image/png")
            return
        if parsed.path in ("/", "/index.html"):
            self._send_headers_only(content_type="text/html; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _build_handler(
    *,
    output_dir: Path,
    pid_file: Path | None,
    log_file: Path | None,
    recent_rows: int,
    log_lines: int,
    title: str,
) -> type[DashboardHandler]:
    class _Handler(DashboardHandler):
        pass

    _Handler.output_dir = output_dir
    _Handler.pid_file = pid_file
    _Handler.log_file = log_file
    _Handler.recent_rows = recent_rows
    _Handler.log_lines = log_lines
    _Handler.title = title
    return _Handler


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a live dashboard for grid-search outputs.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--pid-file", default="logs/full_run_l1.pid")
    parser.add_argument("--log-file", default="logs/full_run_l1.log")
    parser.add_argument("--recent-rows", type=int, default=DEFAULT_RECENT_ROWS)
    parser.add_argument("--log-lines", type=int, default=DEFAULT_LOG_LINES)
    parser.add_argument("--title", default="Math 463 Grid Search Dashboard")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    pid_file = Path(args.pid_file) if args.pid_file else None
    log_file = Path(args.log_file) if args.log_file else None

    handler = _build_handler(
        output_dir=output_dir,
        pid_file=pid_file,
        log_file=log_file,
        recent_rows=args.recent_rows,
        log_lines=args.log_lines,
        title=args.title,
    )
    httpd = ReusableThreadingHTTPServer((args.host, args.port), handler)

    def _shutdown(_signum: int, _frame: Any) -> None:
        httpd.shutdown()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print(f"[dashboard] serving on http://{args.host}:{args.port}")
    print(f"[dashboard] output dir: {output_dir}")
    sys.stdout.flush()
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
