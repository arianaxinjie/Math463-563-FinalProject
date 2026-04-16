#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Algorithm 2 (Primal–Dual Douglas–Rachford).

From the repository root::

    python core_code/algorithms/algorithm2_primal_dual_dr/run.py

Hyperparameters for this algorithm live only in ``HYPERPARAMETERS`` below.
"""

from __future__ import annotations

from core_code.algorithms.run_common import (
    DEFAULT_PROBLEM,
    run_algorithm_demo,
)

# -----------------------------------------------------------------------------
# HYPERPARAMETERS — Algorithm 2 owner edits here only
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
        algorithm_index=2,
        title="Primal–Dual DR (Algorithm 2)",
        rng_seed=RNG_SEED,
        problem=PROBLEM,
        build_solver=lambda fp, model: fp.PrimalDualDouglasRachford(
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
