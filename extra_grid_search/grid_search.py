"""Grid-search runner for the deblurring algorithms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from core_code.data import build_problem, generate_blurred_noisy_cfg, load_image
from core_code.helpers import operator_a_spectral_norm_sq
from core_code.optsolve import OptParams, optsolve
from core_code.viz import compute_psnr

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = REPO_ROOT / "testimages" / "cameraman.jpg"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "grid_search" / "cameraman_gaussian_sp"
RESULTS_FILENAME = "results.csv"
SUMMARY_FILENAME = "summary.json"
CONFIG_FILENAME = "search_config.json"

_WORKER_X_TRUE: np.ndarray | None = None
_WORKER_KERNEL: np.ndarray | None = None
_WORKER_B: np.ndarray | None = None

ALGORITHM_LABELS = {
    "primal_dr": "Algorithm 1 / Primal DR",
    "primal_dual_dr": "Algorithm 2 / Primal-Dual DR",
    "admm": "Algorithm 3 / ADMM",
    "chambolle_pock": "Algorithm 4 / Chambolle-Pock",
}

DEFAULT_GAMMAS = [
    0.002,
    0.003,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
]


@dataclass(frozen=True)
class ObservationConfig:
    image_path: str = str(DEFAULT_IMAGE)
    kernel_kind: str = "gaussian"
    kernel_size: int = 15
    kernel_sigma: float = 3.0
    noise_type: str = "salt_pepper"
    noise_density: float = 0.05
    noise_mean: float = 0.0
    noise_sigma: float = 0.01
    mode: str = "periodic"
    seed: int = 42


def default_search_space(
    problems: Sequence[str] = ("l1",),
) -> dict[str, dict[str, list[float | str]]]:
    """Detailed default sweep for the fixed Cameraman + blur + S&P setup."""
    return {
        "primal_dr": {
            "problem": list(problems),
            "gamma": list(DEFAULT_GAMMAS),
            "tprimaldr": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
            "rhoprimaldr": [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5],
        },
        "primal_dual_dr": {
            "problem": list(problems),
            "gamma": list(DEFAULT_GAMMAS),
            "tprimaldualdr": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
            "rhoprimaldualdr": [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5],
        },
        "admm": {
            "problem": list(problems),
            "gamma": list(DEFAULT_GAMMAS),
            "tadmm": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0],
            "rhoadmm": [0.5, 0.8, 1.0, 1.2, 1.5, 1.8],
        },
        "chambolle_pock": {
            "problem": list(problems),
            "gamma": list(DEFAULT_GAMMAS),
            "cp_theta": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "cp_ratio": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        },
    }


def _load_search_space_json(path: str | Path) -> dict[str, dict[str, list[float | str]]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("search space JSON must be an object keyed by algorithm")

    search_space: dict[str, dict[str, list[float | str]]] = {}
    for algorithm, grid in raw.items():
        if algorithm not in ALGORITHM_LABELS:
            raise ValueError(f"Unknown algorithm in search-space JSON: {algorithm!r}")
        if not isinstance(grid, dict):
            raise ValueError(f"Search-space entry for {algorithm!r} must be an object")

        normalized: dict[str, list[float | str]] = {}
        for key, values in grid.items():
            if not isinstance(values, list):
                raise ValueError(
                    f"Search-space entry {algorithm}.{key} must be a JSON array",
                )
            normalized[key] = list(values)

        if "problem" not in normalized:
            raise ValueError(
                f"Search-space entry for {algorithm!r} must include a 'problem' list",
            )
        search_space[algorithm] = normalized

    return search_space


def _load_observation(
    cfg: ObservationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(cfg.seed)
    x_true = load_image(cfg.image_path)
    kernel, x_blur, b = generate_blurred_noisy_cfg(
        x_true=x_true,
        kernel_kind=cfg.kernel_kind,
        kernel_size=cfg.kernel_size,
        kernel_sigma=cfg.kernel_sigma,
        noise_type=cfg.noise_type,
        noise_density=cfg.noise_density,
        noise_mean=cfg.noise_mean,
        noise_sigma=cfg.noise_sigma,
        mode=cfg.mode,
    )
    return x_true, kernel, x_blur, b


def _cartesian_dict(grid: dict[str, Sequence[Any]]) -> Iterable[dict[str, Any]]:
    keys = list(grid)
    values = [grid[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _trial_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=_jsonable).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _build_cp_steps(
    cp_theta: float,
    cp_ratio: float,
    kernel: np.ndarray,
    b: np.ndarray,
) -> tuple[float, float, float]:
    reference_model = build_problem(kernel, b, gamma=DEFAULT_GAMMAS[0], fidelity="l1")
    norm_sq = operator_a_spectral_norm_sq(reference_model.ops)
    norm_a = float(np.sqrt(norm_sq))
    t = float(cp_theta / (norm_a * cp_ratio))
    s = float(cp_theta * cp_ratio / norm_a)
    return t, s, norm_sq


def _iter_trials(
    search_space: dict[str, dict[str, list[float | str]]],
    kernel: np.ndarray,
    b: np.ndarray,
    maxiter: int,
    tol: float,
    compute_obj_every: int,
) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for algorithm, grid in search_space.items():
        for combo in _cartesian_dict(grid):
            params = {
                "maxiter": maxiter,
                "tol": tol,
                "verbose": False,
                "compute_obj_every": compute_obj_every,
                "gamma": combo["gamma"],
            }

            search_values = dict(combo)
            if algorithm == "primal_dr":
                params["tprimaldr"] = combo["tprimaldr"]
                params["rhoprimaldr"] = combo["rhoprimaldr"]
            elif algorithm == "primal_dual_dr":
                params["tprimaldualdr"] = combo["tprimaldualdr"]
                params["rhoprimaldualdr"] = combo["rhoprimaldualdr"]
            elif algorithm == "admm":
                params["tadmm"] = combo["tadmm"]
                params["rhoadmm"] = combo["rhoadmm"]
            elif algorithm == "chambolle_pock":
                tcp, scp, norm_sq = _build_cp_steps(
                    float(combo["cp_theta"]),
                    float(combo["cp_ratio"]),
                    kernel,
                    b,
                )
                params["tcp"] = tcp
                params["scp"] = scp
                params["cp_step_theta"] = combo["cp_theta"]
                search_values["tcp"] = tcp
                search_values["scp"] = scp
                search_values["operator_norm_sq"] = norm_sq
            else:
                raise ValueError(f"Unsupported algorithm {algorithm!r}")

            payload = {
                "algorithm": algorithm,
                "problem": combo["problem"],
                "params": params,
                "search_values": search_values,
            }
            trials.append(
                {
                    "trial_key": _trial_key(payload),
                    "algorithm": algorithm,
                    "problem": combo["problem"],
                    "params": params,
                    "search_values": search_values,
                }
            )
    return trials


def _load_completed_keys(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    with results_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row["trial_key"] for row in reader if row.get("trial_key")}


def _read_results(results_path: Path) -> list[dict[str, Any]]:
    if not results_path.exists():
        return []
    with results_path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    deduped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        key = row.get("trial_key")
        if not key:
            passthrough.append(row)
            continue
        if key not in deduped:
            order.append(key)
        deduped[key] = row
    return passthrough + [deduped[key] for key in order]


def _append_result(results_path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "trial_key",
        "algorithm",
        "algorithm_label",
        "problem",
        "psnr",
        "final_obj",
        "iterations",
        "time_sec",
        "converged",
        "status",
        "error",
        "params_json",
        "search_values_json",
    ]
    write_header = not results_path.exists()
    with results_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _leaderboard_rows(
    rows: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault(row["algorithm"], []).append(row)

    leaderboard: dict[str, list[dict[str, Any]]] = {}
    for algorithm, entries in grouped.items():
        entries.sort(key=lambda item: float(item["psnr"]), reverse=True)
        leaderboard[algorithm] = entries[:top_k]
    return leaderboard


def _write_summary(
    output_dir: Path,
    observation_cfg: ObservationConfig,
    total_trials: int,
    completed_trials: int,
    pending_trials: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    leaderboard = _leaderboard_rows(rows)
    summary = {
        "observation": asdict(observation_cfg),
        "total_trials": total_trials,
        "completed_trials": completed_trials,
        "pending_trials": pending_trials,
        "best_by_algorithm": {},
    }

    for algorithm, best_rows in leaderboard.items():
        if not best_rows:
            continue
        best = best_rows[0]
        summary["best_by_algorithm"][algorithm] = {
            "algorithm_label": best["algorithm_label"],
            "problem": best["problem"],
            "psnr": float(best["psnr"]),
            "final_obj": float(best["final_obj"]),
            "iterations": int(best["iterations"]),
            "time_sec": float(best["time_sec"]),
            "converged": str(best["converged"]).lower() == "true",
            "params": json.loads(best["params_json"]),
            "search_values": json.loads(best["search_values_json"]),
        }

    summary_path = output_dir / SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _init_worker(x_true: np.ndarray, kernel: np.ndarray, b: np.ndarray) -> None:
    global _WORKER_X_TRUE, _WORKER_KERNEL, _WORKER_B
    _WORKER_X_TRUE = x_true
    _WORKER_KERNEL = kernel
    _WORKER_B = b


def _run_trial_worker(trial: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_X_TRUE is None or _WORKER_KERNEL is None or _WORKER_B is None:
        raise RuntimeError("Worker observation is not initialized.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            x_rec, _obj_history, info = optsolve(
                trial["problem"],
                trial["algorithm"],
                None,
                _WORKER_KERNEL,
                _WORKER_B,
                OptParams(**trial["params"]),
                return_all=True,
            )
        if not np.all(np.isfinite(x_rec)):
            raise FloatingPointError("Recovered image contains non-finite values.")
        for key in ("final_obj", "time"):
            value = float(info[key])
            if not np.isfinite(value):
                raise FloatingPointError(f"Non-finite solver info field: {key}")
        return {
            "trial_key": trial["trial_key"],
            "algorithm": trial["algorithm"],
            "algorithm_label": ALGORITHM_LABELS[trial["algorithm"]],
            "problem": trial["problem"],
            "psnr": f"{compute_psnr(_WORKER_X_TRUE, x_rec):.8f}",
            "final_obj": f"{float(info['final_obj']):.8f}",
            "iterations": int(info["iterations"]),
            "time_sec": f"{float(info['time']):.6f}",
            "converged": bool(info["converged"]),
            "status": "ok",
            "error": "",
            "params_json": json.dumps(trial["params"], sort_keys=True),
            "search_values_json": json.dumps(trial["search_values"], sort_keys=True),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "trial_key": trial["trial_key"],
            "algorithm": trial["algorithm"],
            "algorithm_label": ALGORITHM_LABELS[trial["algorithm"]],
            "problem": trial["problem"],
            "psnr": "",
            "final_obj": "",
            "iterations": 0,
            "time_sec": "",
            "converged": False,
            "status": "error",
            "error": repr(exc),
            "params_json": json.dumps(trial["params"], sort_keys=True),
            "search_values_json": json.dumps(trial["search_values"], sort_keys=True),
        }


def run_grid_search(
    *,
    observation_cfg: ObservationConfig | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    problems: Sequence[str] = ("l1",),
    algorithms: Sequence[str] | None = None,
    search_space_override: dict[str, dict[str, list[float | str]]] | None = None,
    maxiter: int = 500,
    tol: float = 1e-6,
    compute_obj_every: int = 10,
    limit_per_algorithm: int | None = None,
    print_every: int = 25,
    workers: int = 1,
) -> dict[str, Any]:
    """Run the parameter sweep and write incremental artifacts under ``output_dir``."""
    obs_cfg = observation_cfg or ObservationConfig()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_true, kernel, _x_blur, b = _load_observation(obs_cfg)

    search_space = (
        search_space_override
        if search_space_override is not None
        else default_search_space(problems=problems)
    )
    if algorithms is not None:
        allowed = set(algorithms)
        search_space = {
            name: grid for name, grid in search_space.items() if name in allowed
        }
        unknown = set(algorithms) - set(search_space)
        if unknown:
            raise ValueError(f"Unknown algorithms requested: {sorted(unknown)}")

    trials = _iter_trials(
        search_space,
        kernel=kernel,
        b=b,
        maxiter=maxiter,
        tol=tol,
        compute_obj_every=compute_obj_every,
    )
    if limit_per_algorithm is not None:
        limited: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for trial in trials:
            count = counts.get(trial["algorithm"], 0)
            if count >= limit_per_algorithm:
                continue
            limited.append(trial)
            counts[trial["algorithm"]] = count + 1
        trials = limited

    config_path = out_dir / CONFIG_FILENAME
    config_path.write_text(
        json.dumps(
            {
                "observation": asdict(obs_cfg),
                "algorithms": list(search_space),
                "problems": list(problems),
                "maxiter": maxiter,
                "tol": tol,
                "compute_obj_every": compute_obj_every,
                "workers": workers,
                "trial_count": len(trials),
                "search_space": search_space,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    results_path = out_dir / RESULTS_FILENAME
    completed_keys = _load_completed_keys(results_path)
    pending_trials = [
        trial for trial in trials if trial["trial_key"] not in completed_keys
    ]

    print(f"[grid-search] output directory: {out_dir}")
    print(f"[grid-search] total trials: {len(trials)}")
    print(f"[grid-search] already completed: {len(completed_keys)}")
    print(f"[grid-search] remaining trials: {len(pending_trials)}")
    workers = max(int(workers), 1)
    print(f"[grid-search] workers: {workers}")

    existing_rows = _read_results(results_path)
    _write_summary(
        out_dir,
        obs_cfg,
        total_trials=len(trials),
        completed_trials=len(existing_rows),
        pending_trials=max(len(trials) - len(existing_rows), 0),
        rows=existing_rows,
    )

    processed = 0
    if workers == 1:
        _init_worker(x_true, kernel, b)
        for trial in pending_trials:
            row = _run_trial_worker(trial)
            processed += 1
            _append_result(results_path, row)

            if processed % print_every == 0 or processed == len(pending_trials):
                rows = _read_results(results_path)
                completed = len(rows)
                print(f"[grid-search] progress: {completed}/{len(trials)}")
                _write_summary(
                    out_dir,
                    obs_cfg,
                    total_trials=len(trials),
                    completed_trials=completed,
                    pending_trials=max(len(trials) - completed, 0),
                    rows=rows,
                )
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(x_true, kernel, b),
        ) as executor:
            futures = [
                executor.submit(_run_trial_worker, trial) for trial in pending_trials
            ]
            for future in as_completed(futures):
                row = future.result()
                processed += 1
                _append_result(results_path, row)

                if processed % print_every == 0 or processed == len(pending_trials):
                    rows = _read_results(results_path)
                    completed = len(rows)
                    print(f"[grid-search] progress: {completed}/{len(trials)}")
                    _write_summary(
                        out_dir,
                        obs_cfg,
                        total_trials=len(trials),
                        completed_trials=completed,
                        pending_trials=max(len(trials) - completed, 0),
                        rows=rows,
                    )

    rows = _read_results(results_path)
    completed = len(rows)
    return _write_summary(
        out_dir,
        obs_cfg,
        total_trials=len(trials),
        completed_trials=completed,
        pending_trials=max(len(trials) - completed, 0),
        rows=rows,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Grid search for all deblurring algorithms on "
            "Cameraman + Gaussian blur + salt-and-pepper noise."
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--kernel-size", type=int, default=15)
    parser.add_argument("--kernel-sigma", type=float, default=3.0)
    parser.add_argument("--noise-density", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--compute-obj-every", type=int, default=10)
    parser.add_argument("--limit-per-algorithm", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes. Use >1 on a multi-core server.",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=["l1"],
        choices=["l1", "l2"],
        help="Default is l1 because salt-and-pepper noise is impulse-like.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        choices=list(ALGORITHM_LABELS),
        help="Subset of algorithms to run. Omit to run all four.",
    )
    parser.add_argument(
        "--search-space-json",
        default=None,
        help="Path to a JSON file that overrides the default search grid.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    obs_cfg = ObservationConfig(
        image_path=args.image,
        kernel_size=args.kernel_size,
        kernel_sigma=args.kernel_sigma,
        noise_density=args.noise_density,
        seed=args.seed,
    )
    search_space_override = (
        _load_search_space_json(args.search_space_json)
        if args.search_space_json
        else None
    )
    summary = run_grid_search(
        observation_cfg=obs_cfg,
        output_dir=args.output_dir,
        problems=args.problems,
        algorithms=args.algorithms,
        search_space_override=search_space_override,
        maxiter=args.maxiter,
        tol=args.tol,
        compute_obj_every=args.compute_obj_every,
        limit_per_algorithm=args.limit_per_algorithm,
        print_every=args.print_every,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
