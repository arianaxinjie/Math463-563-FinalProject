# -*- coding: utf-8 -*-
"""
Unified iterative loop shared by all algorithms.

This preserves the original notebook behavior: objective values are evaluated on
``box_prox(x_current)``, convergence is checked on the raw iterate returned by
the algorithm-specific step, and the final output is projected back to ``[0, 1]``.
"""

import time

import numpy as np

from .objective import objective_value
from .proximal import box_prox


def run_solver(
    step_fn,
    init_state,
    ops,
    b,
    gamma,
    problem,
    maxiter=500,
    tol=1e-6,
    verbose=True,
    compute_obj_every=1,
):
    start_time = time.time()
    state = init_state
    obj_history = []
    x_prev = None
    converged = False

    for k in range(1, maxiter + 1):
        state, x_current = step_fn(state, k)

        if k % compute_obj_every == 0 or k == 1:
            obj = objective_value(box_prox(x_current), b, gamma, ops, problem)
            obj_history.append(obj)
            if verbose:
                print(f"  iter {k:4d} | obj = {obj:.6f}")

        if x_prev is not None:
            rel_change = np.linalg.norm(x_current - x_prev) / (
                np.linalg.norm(x_current) + 1e-15
            )
            if rel_change < tol:
                converged = True
                if verbose:
                    print(
                        f"  Converged at iteration {k} "
                        f"(rel_change = {rel_change:.2e})"
                    )
                break
        x_prev = x_current.copy()

    elapsed = time.time() - start_time
    x_sol = box_prox(x_current)
    final_obj = objective_value(x_sol, b, gamma, ops, problem)

    if verbose:
        print(f"\n=== Summary ===")
        print(f"  Iterations: {k}")
        print(f"  Final objective: {final_obj:.6f}")
        print(f"  CPU time: {elapsed:.2f}s")
        print(f"  Converged: {converged}")

    info = {
        "iterations": k,
        "time": elapsed,
        "converged": converged,
        "final_obj": final_obj,
    }
    return x_sol, obj_history, info
