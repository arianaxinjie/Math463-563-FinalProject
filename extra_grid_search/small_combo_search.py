"""Small combination search over blur/noise setups using fixed Great-Search best params.

Runs 4 observation combinations:
- Gaussian kernel + salt-pepper noise
- Gaussian kernel + Gaussian noise
- Motion kernel + salt-pepper noise
- Motion kernel + Gaussian noise

For each combination, evaluate all 4 algorithms with both l1 and l2 fidelity,
using the previously selected best hyperparameters from Great Search.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core_code.data import build_problem, generate_blurred_noisy_cfg, load_image
from core_code.helpers import operator_a_spectral_norm_sq
from core_code.optsolve import OptParams, optsolve
from core_code.viz import compute_psnr

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = REPO_ROOT / "testimages" / "cameraman.jpg"
DEFAULT_REPORT_JSON = REPO_ROOT / "extra_grid_search" / "docs" / "grid_search_report_data.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "small_combo_search"

ALGORITHMS = ["primal_dr", "primal_dual_dr", "admm", "chambolle_pock"]
PROBLEMS = ["l1", "l2"]


@dataclass(frozen=True)
class ComboConfig:
    name: str
    kernel_kind: str
    noise_type: str
    kernel_size: int = 15
    kernel_sigma: float = 3.0
    motion_length: int = 15
    motion_angle: float = 20.0
    noise_density: float = 0.05
    noise_mean: float = 0.0
    noise_sigma: float = 0.01
    mode: str = "periodic"


COMBOS = [
    ComboConfig(name="gaussian_kernel__salt_pepper_noise", kernel_kind="gaussian", noise_type="salt_pepper"),
    ComboConfig(name="gaussian_kernel__gaussian_noise", kernel_kind="gaussian", noise_type="gaussian"),
    ComboConfig(name="motion_kernel__salt_pepper_noise", kernel_kind="motion", noise_type="salt_pepper"),
    ComboConfig(name="motion_kernel__gaussian_noise", kernel_kind="motion", noise_type="gaussian"),
]


def _load_best_from_great_search(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    final_best = payload["final_best"]

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, entry in final_best.items():
        algo, problem = key.split(":", 1)
        out[(algo, problem)] = {
            "params": dict(entry["params"]),
            "search_values": dict(entry["search_values"]),
        }
    return out


def _cp_steps_from_theta_ratio(
    kernel: np.ndarray,
    b: np.ndarray,
    gamma: float,
    problem: str,
    cp_theta: float,
    cp_ratio: float,
) -> tuple[float, float, float]:
    model = build_problem(kernel, b, gamma=gamma, fidelity=problem)
    norm_sq = float(operator_a_spectral_norm_sq(model.ops))
    norm_a = float(np.sqrt(norm_sq))
    tcp = float(cp_theta / (norm_a * cp_ratio))
    scp = float(cp_theta * cp_ratio / norm_a)
    return tcp, scp, norm_sq


def _trial_params(
    algo: str,
    problem: str,
    best_map: dict[tuple[str, str], dict[str, Any]],
    kernel: np.ndarray,
    b: np.ndarray,
    maxiter: int,
    tol: float,
    compute_obj_every: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    src = best_map[(algo, problem)]
    params = dict(src["params"])
    search_values = dict(src["search_values"])

    params["maxiter"] = maxiter
    params["tol"] = tol
    params["compute_obj_every"] = compute_obj_every
    params["verbose"] = False

    if algo == "chambolle_pock":
        cp_theta = float(search_values["cp_theta"])
        cp_ratio = float(search_values["cp_ratio"])
        tcp, scp, norm_sq = _cp_steps_from_theta_ratio(
            kernel=kernel,
            b=b,
            gamma=float(params["gamma"]),
            problem=problem,
            cp_theta=cp_theta,
            cp_ratio=cp_ratio,
        )
        params["cp_step_theta"] = cp_theta
        params["tcp"] = tcp
        params["scp"] = scp
        search_values["tcp_recomputed"] = tcp
        search_values["scp_recomputed"] = scp
        search_values["operator_norm_sq_recomputed"] = norm_sq

    return params, search_values


def run_small_combo_search(
    *,
    image_path: Path,
    report_json: Path,
    output_dir: Path,
    seed: int,
    maxiter: int,
    tol: float,
    compute_obj_every: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"

    best_map = _load_best_from_great_search(report_json)

    np.random.seed(seed)
    x_true = load_image(str(image_path))

    rows: list[dict[str, Any]] = []
    for combo in COMBOS:
        np.random.seed(seed)
        kernel, _x_blur, b = generate_blurred_noisy_cfg(
            x_true=x_true,
            kernel_kind=combo.kernel_kind,
            kernel_size=combo.kernel_size,
            kernel_sigma=combo.kernel_sigma,
            motion_angle=combo.motion_angle,
            motion_length=combo.motion_length,
            noise_type=combo.noise_type,
            noise_density=combo.noise_density,
            noise_mean=combo.noise_mean,
            noise_sigma=combo.noise_sigma,
            mode=combo.mode,
        )

        for algo in ALGORITHMS:
            for problem in PROBLEMS:
                params, search_values = _trial_params(
                    algo=algo,
                    problem=problem,
                    best_map=best_map,
                    kernel=kernel,
                    b=b,
                    maxiter=maxiter,
                    tol=tol,
                    compute_obj_every=compute_obj_every,
                )

                x_rec, _obj, info = optsolve(
                    problem,
                    algo,
                    None,
                    kernel,
                    b,
                    OptParams(**params),
                    return_all=True,
                )
                psnr = float(compute_psnr(x_true, x_rec))

                rows.append(
                    {
                        "combo": combo.name,
                        "kernel_kind": combo.kernel_kind,
                        "noise_type": combo.noise_type,
                        "algorithm": algo,
                        "problem": problem,
                        "psnr": psnr,
                        "final_obj": float(info["final_obj"]),
                        "iterations": int(info["iterations"]),
                        "time_sec": float(info["time"]),
                        "converged": bool(info["converged"]),
                        "params_json": json.dumps(params, sort_keys=True),
                        "search_values_json": json.dumps(search_values, sort_keys=True),
                    }
                )

    rows.sort(key=lambda r: (r["combo"], -r["psnr"]))

    with rows_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "combo",
                "kernel_kind",
                "noise_type",
                "algorithm",
                "problem",
                "psnr",
                "final_obj",
                "iterations",
                "time_sec",
                "converged",
                "params_json",
                "search_values_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    best_by_combo: dict[str, dict[str, Any]] = {}
    for combo in {r["combo"] for r in rows}:
        subset = [r for r in rows if r["combo"] == combo]
        best = max(subset, key=lambda r: r["psnr"])
        best_by_combo[combo] = {
            "kernel_kind": best["kernel_kind"],
            "noise_type": best["noise_type"],
            "algorithm": best["algorithm"],
            "problem": best["problem"],
            "psnr": best["psnr"],
        }

    overall_best = max(rows, key=lambda r: r["psnr"])
    summary = {
        "image": str(image_path),
        "report_json": str(report_json),
        "seed": seed,
        "maxiter": maxiter,
        "tol": tol,
        "compute_obj_every": compute_obj_every,
        "combo_count": len(COMBOS),
        "trial_count": len(rows),
        "best_by_combo": best_by_combo,
        "overall_best": {
            "combo": overall_best["combo"],
            "kernel_kind": overall_best["kernel_kind"],
            "noise_type": overall_best["noise_type"],
            "algorithm": overall_best["algorithm"],
            "problem": overall_best["problem"],
            "psnr": overall_best["psnr"],
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small 4-combination blur/noise comparison with fixed Great-Search best params."
        )
    )
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--compute-obj-every", type=int, default=10)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = run_small_combo_search(
        image_path=Path(args.image),
        report_json=Path(args.report_json),
        output_dir=Path(args.output_dir),
        seed=int(args.seed),
        maxiter=int(args.maxiter),
        tol=float(args.tol),
        compute_obj_every=int(args.compute_obj_every),
    )

    overall = summary["overall_best"]
    print("[small-combo-search] completed")
    print(f"[small-combo-search] output: {args.output_dir}")
    print(
        "[small-combo-search] overall best: "
        f"combo={overall['combo']} algo={overall['algorithm']} problem={overall['problem']} "
        f"psnr={overall['psnr']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
