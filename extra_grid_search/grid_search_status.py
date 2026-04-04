"""Status and ETA reporting for grid-search runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

RESULTS_FILENAME = "results.csv"
CONFIG_FILENAME = "search_config.json"
SUMMARY_FILENAME = "summary.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _safe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return x


def _parse_json_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(int(round(seconds)), 0)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def load_status(output_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(output_dir)
    config = _read_json(out_dir / CONFIG_FILENAME)
    summary = _read_json(out_dir / SUMMARY_FILENAME)
    rows = _read_results(out_dir / RESULTS_FILENAME)

    total_trials = int(config.get("trial_count", len(rows)))
    completed_trials = len(rows)
    remaining_trials = max(total_trials - completed_trials, 0)
    workers = int(config.get("workers", 1) or 1)

    status_counter = Counter(row.get("status", "unknown") for row in rows)
    algorithm_counter = Counter(row.get("algorithm", "unknown") for row in rows)
    problem_counter = Counter(row.get("problem", "unknown") for row in rows)
    noise_counter = Counter()
    alg_problem_counter: dict[str, Counter[str]] = defaultdict(Counter)
    time_values: list[float] = []

    for row in rows:
        alg_problem_counter[row.get("algorithm", "unknown")][
            row.get("problem", "unknown")
        ] += 1
        search_values = _parse_json_dict(row.get("search_values_json"))
        noise_density = row.get("noise_density") or search_values.get("noise_density")
        if noise_density not in (None, ""):
            noise_counter[str(noise_density)] += 1
        value = _safe_float(row.get("time_sec"))
        if value is not None:
            time_values.append(value)

    avg_trial_sec = sum(time_values) / len(time_values) if time_values else None
    remaining_wall_sec = None
    if avg_trial_sec is not None and workers > 0:
        remaining_wall_sec = remaining_trials * avg_trial_sec / workers

    eta_timestamp = None
    if remaining_wall_sec is not None:
        eta_timestamp = datetime.fromtimestamp(
            time.time() + remaining_wall_sec,
        ).astimezone().isoformat(timespec="seconds")

    return {
        "output_dir": str(out_dir),
        "config": config,
        "summary": summary,
        "total_trials": total_trials,
        "completed_trials": completed_trials,
        "remaining_trials": remaining_trials,
        "completion_ratio": (completed_trials / total_trials) if total_trials else 0.0,
        "workers": workers,
        "status_counts": dict(status_counter),
        "algorithm_counts": dict(algorithm_counter),
        "problem_counts": dict(problem_counter),
        "noise_density_counts": dict(noise_counter),
        "algorithm_problem_counts": {
            alg: dict(counter) for alg, counter in alg_problem_counter.items()
        },
        "avg_trial_sec": avg_trial_sec,
        "remaining_wall_sec": remaining_wall_sec,
        "eta_timestamp": eta_timestamp,
    }


def _render_text(status: dict[str, Any]) -> str:
    lines = [
        f"Output Dir: {status['output_dir']}",
        (
            "Progress: "
            f"{status['completed_trials']}/{status['total_trials']} "
            f"({status['completion_ratio'] * 100:.2f}%)"
        ),
        f"Remaining: {status['remaining_trials']}",
        f"Workers: {status['workers']}",
        (
            "Status Counts: "
            + ", ".join(
                f"{key}={value}"
                for key, value in sorted(status["status_counts"].items())
            )
        ),
        (
            "Problem Counts: "
            + ", ".join(
                f"{key}={value}"
                for key, value in sorted(status["problem_counts"].items())
            )
        ),
        (
            "Noise Counts: "
            + ", ".join(
                f"{key}={value}"
                for key, value in sorted(status["noise_density_counts"].items())
            )
            if status["noise_density_counts"]
            else "Noise Counts: —"
        ),
        (
            "Algorithm Counts: "
            + ", ".join(
                f"{key}={value}"
                for key, value in sorted(status["algorithm_counts"].items())
            )
        ),
        (
            "Average Trial Time: "
            f"{status['avg_trial_sec']:.2f}s"
            if status["avg_trial_sec"] is not None
            else "Average Trial Time: unknown"
        ),
        f"Estimated Remaining Wall Time: {_format_duration(status['remaining_wall_sec'])}",
        f"Estimated Finish: {status['eta_timestamp'] or 'unknown'}",
    ]

    if status["algorithm_problem_counts"]:
        lines.append("Per-Algorithm / Problem:")
        for algorithm in sorted(status["algorithm_problem_counts"]):
            counts = status["algorithm_problem_counts"][algorithm]
            rendered = ", ".join(
                f"{problem}={count}" for problem, count in sorted(counts.items())
            )
            lines.append(f"  {algorithm}: {rendered}")

    summary = status.get("summary") or {}
    best_by_algorithm = summary.get("best_by_algorithm") or {}
    if best_by_algorithm:
        lines.append("Current Best PSNR:")
        for algorithm in sorted(best_by_algorithm):
            best = best_by_algorithm[algorithm]
            lines.append(
                f"  {algorithm}: {best['psnr']:.4f} dB ({best['problem']})",
            )

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show progress and ETA for a grid-search output directory.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="output/grid_search/cameraman_gaussian_sp",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--watch", type=float, default=None, help="Refresh every N seconds.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    while True:
        status = load_status(args.output_dir)
        if args.json:
            print(json.dumps(status, indent=2, sort_keys=True))
        else:
            print(_render_text(status))

        if args.watch is None:
            break
        time.sleep(args.watch)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
