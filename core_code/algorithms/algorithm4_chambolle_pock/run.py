#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Algorithm 4 (Chambolle–Pock).

From the repository root::

    python core_code/algorithms/algorithm4_chambolle_pock/run.py

Hyperparameters for this algorithm live only in ``HYPERPARAMETERS`` below.

For Chambolle–Pock steps: either set ``step_theta`` (passed to
``chambolle_pock_step_sizes``) or set both ``t`` and ``s`` to floats to use fixed
steps (must satisfy stability; see ``core_code.helpers``).
"""

from __future__ import annotations

from core_code.algorithms.run_common import (
    DEFAULT_PROBLEM,
    run_algorithm_demo,
)

# -----------------------------------------------------------------------------
# HYPERPARAMETERS — Algorithm 4 owner edits here only
# -----------------------------------------------------------------------------
RNG_SEED = 42

PROBLEM = {**DEFAULT_PROBLEM, "blur": {**DEFAULT_PROBLEM["blur"]}}

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
    def build_solver(fp, model):
        if CP_STEPS["t"] is not None and CP_STEPS["s"] is not None:
            t, s = float(CP_STEPS["t"]), float(CP_STEPS["s"])
        else:
            t, s, _ = fp.chambolle_pock_step_sizes(
                model.ops, theta=CP_STEPS["step_theta"]
            )
        return fp.ChambollePock(model, t=t, s=s)

    run_algorithm_demo(
        algorithm_index=4,
        title="Chambolle–Pock (Algorithm 4)",
        rng_seed=RNG_SEED,
        problem=PROBLEM,
        build_solver=build_solver,
        solve_kwargs={
            "maxiter": SOLVER["maxiter"],
            "tol": SOLVER["tol"],
            "verbose": SOLVER["verbose"],
        },
    )


if __name__ == "__main__":
    main()
