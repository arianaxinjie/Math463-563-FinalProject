#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the Section 4.6.6 recommended-parameter experiment from the report.

This script uses the fixed baseline setup from Section 4.4 of
``math463FinalProject (2).pdf``:

- image: cameraman
- problem: l1
- Gaussian blur, kernel size 15, sigma 3.0
- salt-and-pepper noise, density 0.05
- periodic boundary conditions
- gamma = 0.01
- maxiter = 500
- tol = 1e-4

It then runs the four algorithms using the recommended parameters reported in
Section 4.6.6, saves one figure per algorithm, and prints a summary table.

From the repository root:

    python run_recommended_parameters.py
    python run_recommended_parameters.py testimages/cameraman.jpg
"""

from __future__ import annotations

from pathlib import Path
import sys

import core_code as cc


REPO_ROOT = Path(__file__).resolve().parent
TESTIMAGES_DIR = REPO_ROOT / "testimages"
DEFAULT_IMAGE = TESTIMAGES_DIR / "cameraman.jpg"
OUTPUT_DIR = REPO_ROOT / "output" / "recommended_parameters"


def resolve_image_path(argv: list[str]) -> Path:
    """Use CLI image path if provided, otherwise default to testimages/cameraman.jpg."""
    if len(argv) <= 1:
        return DEFAULT_IMAGE

    candidate = Path(argv[1]).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    testimage_candidate = TESTIMAGES_DIR / argv[1]
    if testimage_candidate.is_file():
        return testimage_candidate.resolve()

    raise FileNotFoundError(
        f"Image not found: {argv[1]!r}. Use a valid path or a filename in {TESTIMAGES_DIR}."
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    image_path = resolve_image_path(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fixed experiment setup from report Section 4.4.
    x_true = cc.load_image(str(image_path))
    kernel, _x_blur, b = cc.generate_blurred_noisy_cfg(
        x_true=x_true,
        kernel_kind="gaussian",
        kernel_size=15,
        kernel_sigma=3.0,
        noise_type="salt_pepper",
        noise_density=0.05,
        mode="periodic",
    )

    common_params = {
        "maxiter": 500,
        "tol": 1e-4,
        "gamma": 0.01,
        "verbose": True,
        "compute_obj_every": 1,
    }

    # Recommended parameters from report Section 4.6.6.
    recommended_cases = [
        (
            "Primal DR",
            "douglasrachfordprimal",
            {
                "tprimaldr": 2.0,
                "rhoprimaldr": 0.2,
            },
            OUTPUT_DIR / "recommended_primal_dr.png",
        ),
        (
            "Primal-Dual DR",
            "douglasrachfordprimaldual",
            {
                "tprimaldualdr": 2.0,
                "rhoprimaldualdr": 0.2,
            },
            OUTPUT_DIR / "recommended_primal_dual_dr.png",
        ),
        (
            "ADMM",
            "admm",
            {
                "tadmm": 2.0,
                "rhoadmm": 1.0,
            },
            OUTPUT_DIR / "recommended_admm.png",
        ),
        (
            "Chambolle-Pock",
            "chambollepock",
            {
                "cp_step_theta": 0.7,
            },
            OUTPUT_DIR / "recommended_chambolle_pock.png",
        ),
    ]

    print("=" * 72)
    print("Recommended-Parameter Experiment")
    print(f"image:   {image_path}")
    print("problem: l1")
    print("kernel:  gaussian, size=15, sigma=3.0")
    print("noise:   salt-and-pepper, density=0.05")
    print("gamma:   0.01")
    print("maxiter: 500")
    print("tol:     1e-4")
    print(f"output:  {OUTPUT_DIR}")
    print("=" * 72)

    results: list[tuple[str, dict]] = []

    for title, method, method_params, figure_path in recommended_cases:
        params = {**common_params, **method_params}

        print(f"\nRunning {title} ...")
        x_rec, obj_hist, info = cc.optsolve(
            "l1",
            method,
            None,
            kernel,
            b,
            params,
            return_all=True,
        )
        psnr = cc.compute_psnr(x_true, x_rec)

        cc.show_results(
            x_true,
            b,
            x_rec,
            title=f"{title} (PSNR={psnr:.3f} dB)",
            obj_history=obj_hist,
            save_figure=figure_path,
        )

        print(f"Saved figure: {figure_path}")
        print(
            "Result: "
            f"status={info['status']}, "
            f"iterations={info['iterations']}, "
            f"final_obj={info['final_obj']:.6f}, "
            f"time={info['time']:.2f}s, "
            f"psnr={psnr:.3f} dB"
        )

        results.append((title, {**info, "psnr": psnr}))

    print("\n" + "=" * 72)
    print("Final summary")
    print("=" * 72)
    for title, info in results:
        print(
            f"{title:18s} | "
            f"iter={info['iterations']:4d} | "
            f"obj={info['final_obj']:12.6f} | "
            f"time={info['time']:7.2f}s | "
            f"psnr={info['psnr']:7.3f} dB | "
            f"status={info['status']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
