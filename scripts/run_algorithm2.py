#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Algorithm 2 (Primal–Dual Douglas–Rachford).

From the repository root::

    python scripts/run_algorithm2.py

Hyperparameters for this algorithm live only in ``HYPERPARAMETERS`` below.
"""

from __future__ import annotations

import sys

import numpy as np

from run_common import (
    build_demo_model,
    ensure_matplotlib,
    figure_path,
    get_final_project,
    load_demo_image,
)

# -----------------------------------------------------------------------------
# HYPERPARAMETERS — Algorithm 2 owner edits here only
# -----------------------------------------------------------------------------
RNG_SEED = 42

PROBLEM = {
    "gamma": 0.01,
    "fidelity": "l2",
    "blur": {
        "kernel_kind": "gaussian",
        "kernel_size": 15,
        "kernel_sigma": 3.0,
        "noise_type": "gaussian",
        "noise_sigma": 0.001,
        "mode": "periodic",
    },
}

SOLVER = {
    "t": 2.0,
    "rho": 0.1,
    "maxiter": 500,
    "tol": 5e-5,
    "verbose": True,
    "compute_obj_every": 1,
}


def main():
    ensure_matplotlib()
    fp = get_final_project()

    np.random.seed(RNG_SEED)
    img, source_note = load_demo_image(fp, sys.argv)
    print(f"[input image] {source_note}")

    b, model = build_demo_model(
        fp,
        img,
        gamma=PROBLEM["gamma"],
        fidelity=PROBLEM["fidelity"],
        blur=PROBLEM["blur"],
    )
    solver = fp.PrimalDualDouglasRachford(model, t=SOLVER["t"], rho=SOLVER["rho"])
    x_sol, obj_hist, info = solver.solve(
        maxiter=SOLVER["maxiter"],
        tol=SOLVER["tol"],
        verbose=SOLVER["verbose"],
        compute_obj_every=SOLVER["compute_obj_every"],
    )

    out = figure_path(2)
    print("PSNR:", fp.compute_psnr(img, x_sol), "dB")
    fp.show_results(
        img,
        b,
        x_sol,
        "Primal–Dual DR (Algorithm 2)",
        obj_hist,
        save_figure=out,
    )
    print(f"[saved figure] {out.resolve()}")


if __name__ == "__main__":
    main()
