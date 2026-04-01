#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Algorithm 4 (Chambolle–Pock).

From the repository root::

    python scripts/run_algorithm4.py

Hyperparameters for this algorithm live only in ``HYPERPARAMETERS`` below.

For Chambolle–Pock steps: either set ``step_theta`` (passed to
``chambolle_pock_step_sizes``) or set both ``t`` and ``s`` to floats to use fixed
steps (must satisfy stability; see ``core_code.helpers``).
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
# HYPERPARAMETERS — Algorithm 4 owner edits here only
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
    "maxiter": 500,
    "tol": 5e-5,
    "verbose": True,
}

# Automatic steps: t = s = step_theta / ||A||_2. Set t and s to numbers to override.
CP_STEPS = {
    "step_theta": 0.5,
    "t": None,  # e.g. 0.25
    "s": None,  # e.g. 0.25
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

    if CP_STEPS["t"] is not None and CP_STEPS["s"] is not None:
        t, s = float(CP_STEPS["t"]), float(CP_STEPS["s"])
    else:
        t, s, _ = fp.chambolle_pock_step_sizes(
            model.ops, theta=CP_STEPS["step_theta"]
        )

    solver = fp.ChambollePock(model, t=t, s=s)
    x_sol, obj_hist, info = solver.solve(
        maxiter=SOLVER["maxiter"],
        tol=SOLVER["tol"],
        verbose=SOLVER["verbose"],
    )

    out = figure_path(4)
    print("PSNR:", fp.compute_psnr(img, x_sol), "dB")
    fp.show_results(
        img,
        b,
        x_sol,
        "Chambolle–Pock (Algorithm 4)",
        obj_hist,
        save_figure=out,
    )
    print(f"[saved figure] {out.resolve()}")


if __name__ == "__main__":
    main()
