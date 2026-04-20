

"""Parallel one-dimensional search for Algorithm 1's ``t`` parameter.

This is a practical parameter sweep for Primal Douglas-Rachford, not an
in-iteration backtracking line search. It evaluates many candidate ``t`` values
with a fixed ``rho`` and reports the best result by PSNR.

From the repository root::

    python -m extra_grid_search.line_search
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core_code.algorithms.run_common import (
    DEFAULT_PROBLEM,
    get_final_project,
    resolve_input_path,
)
from core_code.optsolve import OptParams

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "algorithm1_line_search"
RESULTS_FILENAME = "results.csv"
SUMMARY_FILENAME = "summary.json"
CONFIG_FILENAME = "search_config.json"


def _build_search_values(
    t_min: float,
    t_max: float,
    num_values: int,
    scale: str,
) -> np.ndarray:
    if num_values <= 0:
        raise ValueError("num_values must be positive")
    if t_min <= 0 or t_max <= 0:
        raise ValueError("t_min and t_max must be positive")
    if t_min >= t_max:
        raise ValueError("t_min must be smaller than t_max")

    if scale == "linear":
        return np.linspace(t_min, t_max, num_values, dtype=np.float64)
    if scale == "log":
        return np.geomspace(t_min, t_max, num_values, dtype=np.float64)
    raise ValueError(f"Unsupported scale: {scale!r}")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _append_rows(results_path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fieldnames = [
        "trial_index",
        "tprimaldr",
        "rhoprimaldr",
        "problem",
        "psnr",
        "final_obj",
        "iterations",
        "time_sec",
        "converged",
        "status",
        "error",
    ]
    write_header = not results_path.exists()
    with results_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_success_rows(results_path: Path) -> list[dict[str, Any]]:
    if not results_path.exists():
        return []
    with results_path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [row for row in rows if row.get("status") == "ok"]


def _summarize_rows(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: float(row["psnr"]), reverse=True)
    best = ranked[0] if ranked else None
    top_rows = ranked[:top_k]
    return {
        "best": {
            "tprimaldr": float(best["tprimaldr"]),
            "rhoprimaldr": float(best["rhoprimaldr"]),
            "problem": best["problem"],
            "psnr": float(best["psnr"]),
            "final_obj": float(best["final_obj"]),
            "iterations": int(best["iterations"]),
            "time_sec": float(best["time_sec"]),
            "converged": str(best["converged"]).lower() == "true",
        }
        if best
        else None,
        "top_results": [
            {
                "trial_index": int(row["trial_index"]),
                "tprimaldr": float(row["tprimaldr"]),
                "rhoprimaldr": float(row["rhoprimaldr"]),
                "problem": row["problem"],
                "psnr": float(row["psnr"]),
                "final_obj": float(row["final_obj"]),
                "iterations": int(row["iterations"]),
                "time_sec": float(row["time_sec"]),
                "converged": str(row["converged"]).lower() == "true",
            }
            for row in top_rows
        ],
    }


def run_line_search(
    *,
    image_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    problem_cfg: dict[str, Any] | None = None,
    rho: float = 0.1,
    maxiter: int = 40,
    tol: float = 5e-5,
    compute_obj_every: int = 10,
    num_values: int = 500,
    t_min: float = 0.1,
    t_max: float = 5.0,
    scale: str = "linear",
    max_workers: int | None = None,
    top_k: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    fp = get_final_project()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_PROBLEM if problem_cfg is None else problem_cfg

    np.random.seed(seed)
    x_true = fp.load_image(str(image_path))
    kernel, _x_blur, b = fp.generate_blurred_noisy_cfg(
        x_true=x_true,
        **cfg["blur"],
    )

    t_values = _build_search_values(t_min, t_max, num_values, scale)
    worker_count = max_workers or min(8, os.cpu_count() or 1)

    config = {
        "image_path": str(image_path),
        "problem": cfg["fidelity"],
        "gamma": cfg["gamma"],
        "rho": rho,
        "maxiter": maxiter,
        "tol": tol,
        "compute_obj_every": compute_obj_every,
        "num_values": num_values,
        "t_min": t_min,
        "t_max": t_max,
        "scale": scale,
        "max_workers": worker_count,
        "blur": cfg["blur"],
        "seed": seed,
    }
    (out_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )

    print(f"[algorithm1-line-search] output directory: {out_dir}")
    print(f"[algorithm1-line-search] candidates: {num_values}")
    print(f"[algorithm1-line-search] workers: {worker_count}")
    print(
        "[algorithm1-line-search] sweep: "
        f"t in [{t_min}, {t_max}] ({scale}), rho={rho}, maxiter={maxiter}",
    )

    def evaluate(trial_index: int, t_value: float) -> dict[str, Any]:
        start = time.time()
        params = OptParams(
            maxiter=maxiter,
            tol=tol,
            verbose=False,
            compute_obj_every=compute_obj_every,
            gamma=cfg["gamma"],
            tprimaldr=float(t_value),
            rhoprimaldr=float(rho),
        )
        try:
            x_rec, _obj_hist, info = fp.optsolve(
                cfg["fidelity"],
                "primaldr",
                None,
                kernel,
                b,
                params,
                return_all=True,
            )
            return {
                "trial_index": trial_index,
                "tprimaldr": f"{float(t_value):.10f}",
                "rhoprimaldr": f"{float(rho):.10f}",
                "problem": cfg["fidelity"],
                "psnr": f"{float(fp.compute_psnr(x_true, x_rec)):.8f}",
                "final_obj": f"{float(info['final_obj']):.8f}",
                "iterations": int(info["iterations"]),
                "time_sec": f"{float(time.time() - start):.6f}",
                "converged": bool(info["converged"]),
                "status": "ok",
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "trial_index": trial_index,
                "tprimaldr": f"{float(t_value):.10f}",
                "rhoprimaldr": f"{float(rho):.10f}",
                "problem": cfg["fidelity"],
                "psnr": "",
                "final_obj": "",
                "iterations": 0,
                "time_sec": f"{float(time.time() - start):.6f}",
                "converged": False,
                "status": "error",
                "error": repr(exc),
            }

    results_path = out_dir / RESULTS_FILENAME
    if results_path.exists():
        results_path.unlink()

    batch: list[dict[str, Any]] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(evaluate, idx, float(t_value)): idx
            for idx, t_value in enumerate(t_values, start=1)
        }
        for future in as_completed(futures):
            batch.append(future.result())
            completed += 1
            if len(batch) >= 20 or completed == num_values:
                _append_rows(results_path, batch)
                batch = []
            if completed % 25 == 0 or completed == num_values:
                print(f"[algorithm1-line-search] progress: {completed}/{num_values}")

    success_rows = _read_success_rows(results_path)
    summary = {
        "config": config,
        "completed_trials": num_values,
        "successful_trials": len(success_rows),
        **_summarize_rows(success_rows, top_k=top_k),
    }
    (out_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel one-dimensional search over Algorithm 1 t values.",
    )
    parser.add_argument("--image", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--tol", type=float, default=5e-5)
    parser.add_argument("--compute-obj-every", type=int, default=10)
    parser.add_argument("--num-values", type=int, default=500)
    parser.add_argument("--t-min", type=float, default=0.1)
    parser.add_argument("--t-max", type=float, default=5.0)
    parser.add_argument("--scale", choices=("linear", "log"), default="linear")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    image_path = resolve_input_path(args.image)
    summary = run_line_search(
        image_path=image_path,
        output_dir=args.output_dir,
        rho=args.rho,
        maxiter=args.maxiter,
        tol=args.tol,
        compute_obj_every=args.compute_obj_every,
        num_values=args.num_values,
        t_min=args.t_min,
        t_max=args.t_max,
        scale=args.scale,
        max_workers=args.max_workers,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
