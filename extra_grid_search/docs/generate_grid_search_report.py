from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
EXTRA_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = EXTRA_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from core_code.data import generate_blurred_noisy_cfg, load_image
from core_code.optsolve import OptParams, optsolve
from core_code.viz import compute_psnr

REMOTE_OUTPUT = REPO_ROOT / "output" / "grid_search_remote"
STAGE1_RESULTS = REMOTE_OUTPUT / "full_run_l1" / "results.csv"
STAGE2_L1_RESULTS = REMOTE_OUTPUT / "stage2_l1" / "results.csv"
STAGE2_L2_RESULTS = REMOTE_OUTPUT / "stage2_l2" / "results.csv"
REPORT_TEX = EXTRA_ROOT / "docs" / "grid_search_report.tex"
REPORT_JSON = EXTRA_ROOT / "docs" / "grid_search_report_data.json"
ASSETS_DIR = EXTRA_ROOT / "docs" / "grid_search_report_assets"
OBS_IMAGE = REPO_ROOT / "testimages" / "cameraman.jpg"

ALGORITHM_LABELS = {
    "primal_dr": "Primal DR",
    "primal_dual_dr": "Primal-Dual DR",
    "admm": "ADMM",
    "chambolle_pock": "Chambolle-Pock",
}

PROBLEM_ORDER = ["l1", "l2"]
ALGORITHM_ORDER = ["primal_dr", "primal_dual_dr", "admm", "chambolle_pock"]


def _read_results(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _best_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    best: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("status") != "ok" or not row.get("psnr"):
            continue
        key = (row["algorithm"], row["problem"])
        if key not in best or float(row["psnr"]) > float(best[key]["psnr"]):
            best[key] = row
    return best


def _final_best_rows() -> dict[tuple[str, str], dict[str, str]]:
    stage1 = _best_rows(_read_results(STAGE1_RESULTS))
    stage2_l1 = _best_rows(_read_results(STAGE2_L1_RESULTS))
    stage2_l2 = _best_rows(_read_results(STAGE2_L2_RESULTS))

    final: dict[tuple[str, str], dict[str, str]] = {}
    for algorithm in ALGORITHM_ORDER:
        for problem in PROBLEM_ORDER:
            candidates: list[dict[str, str]] = []
            key = (algorithm, problem)
            if key in stage1:
                candidates.append(stage1[key])
            if problem == "l1" and key in stage2_l1:
                candidates.append(stage2_l1[key])
            if problem == "l2" and key in stage2_l2:
                candidates.append(stage2_l2[key])
            final[key] = max(candidates, key=lambda row: float(row["psnr"]))
    return final


def _build_observation() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_true = load_image(str(OBS_IMAGE))
    kernel, x_blur, b = generate_blurred_noisy_cfg(
        x_true=x_true,
        kernel_kind="gaussian",
        kernel_size=15,
        kernel_sigma=3.0,
        noise_type="salt_pepper",
        noise_density=0.05,
        noise_mean=0.0,
        noise_sigma=0.01,
        mode="periodic",
    )
    return x_true, kernel, x_blur, b


def _run_case(
    *,
    problem: str,
    algorithm: str,
    params: dict[str, Any],
    kernel: np.ndarray,
    b: np.ndarray,
    x_true: np.ndarray,
) -> dict[str, Any]:
    runtime_params = dict(params)
    runtime_params["verbose"] = False
    x_rec, history, info = optsolve(
        problem,
        algorithm,
        None,
        kernel,
        b,
        OptParams(**runtime_params),
        return_all=True,
    )
    return {
        "image": x_rec,
        "history": history,
        "info": info,
        "psnr": float(compute_psnr(x_true, x_rec)),
    }


def _default_params() -> dict[str, Any]:
    return asdict(OptParams())


def _render_problem_grid(
    *,
    problem: str,
    observed: np.ndarray,
    x_true: np.ndarray,
    default_runs: dict[str, dict[str, Any]],
    tuned_runs: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for col, algorithm in enumerate(ALGORITHM_ORDER):
        axes[0, col].imshow(observed, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{ALGORITHM_LABELS[algorithm]}\nObserved input")
        axes[0, col].axis("off")

        axes[1, col].imshow(default_runs[algorithm]["image"], cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(
            f"Default\nPSNR={default_runs[algorithm]['psnr']:.3f} dB"
        )
        axes[1, col].axis("off")

        axes[2, col].imshow(tuned_runs[algorithm]["image"], cmap="gray", vmin=0, vmax=1)
        axes[2, col].set_title(
            f"Tuned\nPSNR={tuned_runs[algorithm]['psnr']:.3f} dB"
        )
        axes[2, col].axis("off")

    fig.suptitle(f"{problem.upper()} Results: Observed vs Default vs Tuned", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_observation_figure(
    *,
    x_true: np.ndarray,
    x_blur: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(x_true, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original Cameraman")
    axes[0].axis("off")
    axes[1].imshow(x_blur, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Gaussian blur + S&P noise")
    axes[1].axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_psnr_comparison(
    *,
    default_runs: dict[tuple[str, str], dict[str, Any]],
    tuned_runs: dict[tuple[str, str], dict[str, Any]],
    output_path: Path,
) -> None:
    labels = []
    default_vals = []
    tuned_vals = []
    deltas = []
    for problem in PROBLEM_ORDER:
        for algorithm in ALGORITHM_ORDER:
            key = (algorithm, problem)
            labels.append(f"{ALGORITHM_LABELS[algorithm]}\n{problem}")
            default_psnr = default_runs[key]["psnr"]
            tuned_psnr = tuned_runs[key]["psnr"]
            default_vals.append(default_psnr)
            tuned_vals.append(tuned_psnr)
            deltas.append(tuned_psnr - default_psnr)

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(x - width / 2, default_vals, width, label="Default", color="#b7c4cf")
    ax.bar(x + width / 2, tuned_vals, width, label="Tuned", color="#8b3f1f")
    for idx, delta in enumerate(deltas):
        y = max(default_vals[idx], tuned_vals[idx]) + 0.15
        ax.text(idx, y, f"{delta:+.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Default Parameters vs Tuned Parameters")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _make_table_rows(final_rows: dict[tuple[str, str], dict[str, str]]) -> str:
    rows = []
    for problem in PROBLEM_ORDER:
        for algorithm in ALGORITHM_ORDER:
            row = final_rows[(algorithm, problem)]
            params = json.loads(row["params_json"])
            if algorithm == "primal_dr":
                param_str = (
                    f"$\\gamma={params['gamma']}$, "
                    f"$t={params['tprimaldr']}$, "
                    f"$\\rho={params['rhoprimaldr']}$"
                )
            elif algorithm == "primal_dual_dr":
                param_str = (
                    f"$\\gamma={params['gamma']}$, "
                    f"$t={params['tprimaldualdr']}$, "
                    f"$\\rho={params['rhoprimaldualdr']}$"
                )
            elif algorithm == "admm":
                param_str = (
                    f"$\\gamma={params['gamma']}$, "
                    f"$t={params['tadmm']}$, "
                    f"$\\rho={params['rhoadmm']}$"
                )
            else:
                search_values = json.loads(row["search_values_json"])
                param_str = (
                    f"$\\gamma={params['gamma']}$, "
                    f"$\\theta={search_values['cp_theta']}$, "
                    f"ratio$={search_values['cp_ratio']}$"
                )

            rows.append(
                f"{problem.upper()} & {_latex_escape(ALGORITHM_LABELS[algorithm])} & "
                f"{_fmt(float(row['psnr']))} & {param_str} \\\\"
            )
    return "\n".join(rows)


def _make_improvement_rows(
    default_runs: dict[tuple[str, str], dict[str, Any]],
    tuned_runs: dict[tuple[str, str], dict[str, Any]],
) -> str:
    rows = []
    for problem in PROBLEM_ORDER:
        for algorithm in ALGORITHM_ORDER:
            key = (algorithm, problem)
            default_psnr = default_runs[key]["psnr"]
            tuned_psnr = tuned_runs[key]["psnr"]
            rows.append(
                f"{problem.upper()} & {_latex_escape(ALGORITHM_LABELS[algorithm])} & "
                f"{_fmt(default_psnr)} & {_fmt(tuned_psnr)} & {_fmt(tuned_psnr - default_psnr)} \\\\"
            )
    return "\n".join(rows)


def _write_tex(
    *,
    final_rows: dict[tuple[str, str], dict[str, str]],
    default_runs: dict[tuple[str, str], dict[str, Any]],
    tuned_runs: dict[tuple[str, str], dict[str, Any]],
) -> None:
    default_params = _default_params()
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{array}}
\usepackage{{longtable}}
\usepackage{{amsmath}}
\title{{Grid Search Report for Deblurring on Cameraman with Gaussian Blur and Salt-and-Pepper Noise}}
\author{{Automated Experimental Report}}
\date{{\today}}
\begin{{document}}
\maketitle

\section{{Objective and Experimental Setting}}
The goal of this study is to tune the four optimization algorithms in the project on a fixed restoration benchmark:
\begin{{itemize}}
\item image: \texttt{{cameraman}}
\item blur: Gaussian blur with kernel size $15$ and standard deviation $3.0$
\item noise: salt-and-pepper noise with density $0.05$
\item boundary handling: periodic
\item iteration budget per trial: $500$
\item stopping tolerance: $10^{{-6}}$
\end{{itemize}}

The observation used in all experiments is shown in Figure~\ref{{fig:observation}}.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\textwidth]{{grid_search_report_assets/observation.png}}
\caption{{Original image and the blurred/noisy observation used as the search target.}}
\label{{fig:observation}}
\end{{figure}}

\section{{Default Parameters}}
The project default parameter values are taken directly from \texttt{{OptParams}} in \texttt{{core\_code/optsolve.py}}:
\begin{{itemize}}
\item common defaults: $\gamma={default_params['gamma']}$, \texttt{{maxiter}}$={default_params['maxiter']}$, \texttt{{tol}}$={default_params['tol']}$, \texttt{{compute\_obj\_every}}$={default_params['compute_obj_every']}$
\item Primal DR: $t={default_params['tprimaldr']}$, $\rho={default_params['rhoprimaldr']}$
\item Primal-Dual DR: $t={default_params['tprimaldualdr']}$, $\rho={default_params['rhoprimaldualdr']}$
\item ADMM: $t={default_params['tadmm']}$, $\rho={default_params['rhoadmm']}$
\item Chambolle-Pock: \texttt{{cp\_step\_theta}}$={default_params['cp_step_theta']}$ with internally derived $(t,s)$
\end{{itemize}}

\section{{Search Procedure}}
We used a two-stage search strategy.

\subsection{{Stage 1: Coarse Global Grid}}
Stage 1 searched all four algorithms on both $l_1$ and $l_2$ fidelity models, for a total of $5412$ unique trials.
This stage identified the main promising parameter regions.

\subsection{{Stage 2: Local Refinement}}
Stage 2 refined the neighborhoods around the stage-1 winners:
\begin{{itemize}}
\item $l_1$ refinement: $910$ trials
\item $l_2$ refinement: $1136$ trials
\end{{itemize}}

The refinement improved the best $l_1$ solutions substantially, but did not beat the stage-1 $l_2$ optima. Therefore, the final $l_2$ recommendations remain the stage-1 best settings.

\section{{Final Best Parameters}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{llll}}
\toprule
Problem & Algorithm & PSNR (dB) & Final Parameters \\
\midrule
{_make_table_rows(final_rows)}
\bottomrule
\end{{tabular}}
\caption{{Final recommended parameters after combining the coarse and refinement stages.}}
\end{{table}}

\section{{Rendered Reconstructions}}
Figures~\ref{{fig:l1-grid}} and \ref{{fig:l2-grid}} render the recovered images for every algorithm under both the default settings and the final tuned settings.

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{grid_search_report_assets/l1_results_grid.png}}
\caption{{$l_1$ reconstructions: observed input, default reconstruction, and tuned reconstruction for each algorithm.}}
\label{{fig:l1-grid}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{grid_search_report_assets/l2_results_grid.png}}
\caption{{$l_2$ reconstructions: observed input, default reconstruction, and tuned reconstruction for each algorithm.}}
\label{{fig:l2-grid}}
\end{{figure}}

\section{{Improvement Over Default Values}}
Figure~\ref{{fig:psnr-compare}} compares the final tuned PSNR against the project default PSNR for every algorithm/problem pair.

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{grid_search_report_assets/default_vs_tuned_psnr.png}}
\caption{{PSNR under project default values versus the final tuned values. The labels above each pair show the PSNR gain relative to the defaults.}}
\label{{fig:psnr-compare}}
\end{{figure}}

\begin{{table}}[H]
\centering
\begin{{tabular}}{{lllll}}
\toprule
Problem & Algorithm & Default & Tuned & Gain \\
\midrule
{_make_improvement_rows(default_runs, tuned_runs)}
\bottomrule
\end{{tabular}}
\caption{{Numerical PSNR gains relative to the default project parameters.}}
\end{{table}}

\section{{Main Findings}}
\begin{{itemize}}
\item The best overall setting is \textbf{{Primal-Dual DR with $l_1$ fidelity}}, reaching PSNR $27.7613$ dB.
\item The best final $l_1$ parameters are: $\gamma=0.0015$, $t=5.0$, $\rho=2.0$.
\item For this salt-and-pepper noise setting, $l_1$ fidelity is clearly stronger than $l_2$.
\item Stage 2 meaningfully improved the $l_1$ solutions, especially for Primal DR and Primal-Dual DR.
\item Stage 2 did not improve the best $l_2$ results, so the coarse-grid $l_2$ optima were already sufficiently accurate.
\end{{itemize}}

\end{{document}}
"""
    REPORT_TEX.write_text(tex, encoding="utf-8")


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    x_true, kernel, x_blur, b = _build_observation()
    final_rows = _final_best_rows()

    default_runs: dict[tuple[str, str], dict[str, Any]] = {}
    tuned_runs: dict[tuple[str, str], dict[str, Any]] = {}

    for problem in PROBLEM_ORDER:
        for algorithm in ALGORITHM_ORDER:
            key = (algorithm, problem)
            default_runs[key] = _run_case(
                problem=problem,
                algorithm=algorithm,
                params=_default_params(),
                kernel=kernel,
                b=b,
                x_true=x_true,
            )
            tuned_runs[key] = _run_case(
                problem=problem,
                algorithm=algorithm,
                params=json.loads(final_rows[key]["params_json"]),
                kernel=kernel,
                b=b,
                x_true=x_true,
            )

    _render_observation_figure(
        x_true=x_true,
        x_blur=x_blur,
        output_path=ASSETS_DIR / "observation.png",
    )
    _render_problem_grid(
        problem="l1",
        observed=x_blur,
        x_true=x_true,
        default_runs={alg: default_runs[(alg, "l1")] for alg in ALGORITHM_ORDER},
        tuned_runs={alg: tuned_runs[(alg, "l1")] for alg in ALGORITHM_ORDER},
        output_path=ASSETS_DIR / "l1_results_grid.png",
    )
    _render_problem_grid(
        problem="l2",
        observed=x_blur,
        x_true=x_true,
        default_runs={alg: default_runs[(alg, "l2")] for alg in ALGORITHM_ORDER},
        tuned_runs={alg: tuned_runs[(alg, "l2")] for alg in ALGORITHM_ORDER},
        output_path=ASSETS_DIR / "l2_results_grid.png",
    )
    _render_psnr_comparison(
        default_runs=default_runs,
        tuned_runs=tuned_runs,
        output_path=ASSETS_DIR / "default_vs_tuned_psnr.png",
    )

    report_data = {
        "default_params": _default_params(),
        "final_best": {
            f"{algorithm}:{problem}": {
                "psnr": float(final_rows[(algorithm, problem)]["psnr"]),
                "params": json.loads(final_rows[(algorithm, problem)]["params_json"]),
                "search_values": json.loads(
                    final_rows[(algorithm, problem)]["search_values_json"]
                ),
                "default_psnr": default_runs[(algorithm, problem)]["psnr"],
                "tuned_psnr": tuned_runs[(algorithm, problem)]["psnr"],
            }
            for problem in PROBLEM_ORDER
            for algorithm in ALGORITHM_ORDER
        },
    }
    REPORT_JSON.write_text(
        json.dumps(report_data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_tex(final_rows=final_rows, default_runs=default_runs, tuned_runs=tuned_runs)


if __name__ == "__main__":
    main()
