#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Algorithm 1 (Primal Douglas–Rachford).

From the repository root::

    python core_code/algorithms/algorithm1_primal_dr/run.py

Optional argument: path to an image file, or a filename inside ``testimages/``.

Hyperparameters for this algorithm live only in ``HYPERPARAMETERS`` below (edit that
block; do not rely on teammates changing ``run_common.py``).
"""

from __future__ import annotations

from core_code.algorithms.run_common import (
    DEFAULT_PROBLEM,
    run_algorithm_demo,
)

# -----------------------------------------------------------------------------
# HYPERPARAMETERS — Algorithm 1 owner edits here only
# -----------------------------------------------------------------------------
RNG_SEED = 42

PROBLEM = {
    **DEFAULT_PROBLEM,
    "gamma": 0.01,
    "fidelity": "l1",
    "blur": {**DEFAULT_PROBLEM["blur"]},
}

SOLVER = {
    "t": 2.0,
    "rho": 0.1,
    "maxiter": 500,
    "tol": 1e-4,
    "verbose": True,
    "compute_obj_every": 1,
}


def main():
    run_algorithm_demo(
        algorithm_index=1,
        title="Primal DR (Algorithm 1)",
        rng_seed=RNG_SEED,
        problem=PROBLEM,
        build_solver=lambda fp, model: fp.PrimalDouglasRachford(
            model,
            t=SOLVER["t"],
            rho=SOLVER["rho"],
        ),
        solve_kwargs={
            "maxiter": SOLVER["maxiter"],
            "tol": SOLVER["tol"],
            "verbose": SOLVER["verbose"],
            "compute_obj_every": SOLVER["compute_obj_every"],
        },
    )


if __name__ == "__main__":
    main()
