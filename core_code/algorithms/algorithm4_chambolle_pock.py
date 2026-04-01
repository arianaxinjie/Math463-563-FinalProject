# -*- coding: utf-8 -*-
"""
Algorithm 4: Chambolle-Pock (Primal-Dual Hybrid Gradient)

    y^k = prox_{s g*}(y^{k-1} + s A z^{k-1})
    x^k = prox_{t f}(x^{k-1} - t A^T y^k)
    z^k = 2 x^k - x^{k-1}
"""

from __future__ import annotations

import numpy as np

from ..problem import DeblurProblem
from ..proximal import prox_f, prox_g_conjugate
from ..solver import run_solver


def chambolle_pock_step(ops, b, gamma, t, s, problem):
    """
    Returns the single step function for Chambolle-Pock (Algorithm 4).

    Algorithm 4:
        y^k = prox_{s g*}(y^{k-1} + s A z^{k-1})
        x^k = prox_{t f}(x^{k-1} - t A^T y^k)
        z^k = 2x^k - x^{k-1}

    Here:
        - x is the primal image variable
        - y = (y1, y2, y3) is the dual variable associated with A = [K; D]
        - z is the extrapolated primal variable
    """

    def step(state, k):
        # Previous iterates
        x_prev = state["x"]
        z_prev = state["z"]
        y1_prev = state["y1"]
        y2_prev = state["y2"]
        y3_prev = state["y3"]

        # ------------------------------------------------------------
        # Step 1: Dual update
        # y^k = prox_{s g*}( y^{k-1} + s A z^{k-1} )
        #
        # A z = [Kz; D1 z; D2 z]
        # This step updates the dual variables:
        #   y1 -> data fidelity part
        #   y2,y3 -> TV regularization part
        # ------------------------------------------------------------
        Az1, Az2, Az3 = ops.applyA(z_prev)

        y1_tilde = y1_prev + s * Az1
        y2_tilde = y2_prev + s * Az2
        y3_tilde = y3_prev + s * Az3

        y1k, y2k, y3k = prox_g_conjugate(
            y1_tilde,
            y2_tilde,
            y3_tilde,
            s=s,
            b=b,
            gamma=gamma,
            problem=problem,
        )

        # ------------------------------------------------------------
        # Step 2: Primal update
        # x^k = prox_{t f}( x^{k-1} - t A^T y^k )
        #
        # Since f = δ_[0,1], prox_f is just projection onto [0,1].
        # A^T y = K^T y1 + D1^T y2 + D2^T y3
        # ------------------------------------------------------------
        ATyk = ops.applyAT(y1k, y2k, y3k)
        xk = prox_f(x_prev - t * ATyk, t, problem)

        # ------------------------------------------------------------
        # Step 3: Extrapolation
        # z^k = 2x^k - x^{k-1}
        #
        # This gives a "look-ahead" variable used in the next dual step.
        # ------------------------------------------------------------
        zk = 2.0 * xk - x_prev

        # Save updated state
        state["x"] = xk
        state["z"] = zk
        state["y1"] = y1k
        state["y2"] = y2k
        state["y3"] = y3k

        return state, xk

    return step


def run_chambolle_pock(
    model: DeblurProblem,
    t=0.25,
    s=0.25,
    maxiter=500,
    tol=1e-6,
    verbose=True,
    init_state=None,
):
    """
    Run Chambolle-Pock (Algorithm 4) for image deblurring/denoising.

    Parameters
    ----------
    model :
        ``DeblurProblem``.
    t, s :
        Primal and dual step sizes; should satisfy s * t * ||A||^2 < 1.
    maxiter, tol, verbose :
        Same as :func:`final_project.solver.run_solver`.

    Returns
    -------
    x_sol, obj_history, info
    """
    ops, b = model.ops, model.b
    m, n = b.shape

    # A safe rough check for step sizes:
    # We need s * t * ||A||^2 < 1.
    # Exact ||A|| is not computed here, so this is only a warning.
    if verbose:
        print(f"\n{'='*60}")
        print(
            f"  Chambolle-Pock (Algorithm 4) — problem={model.problem}, "
            f"gamma={model.gamma}, t={t}, s={s}"
        )
        print(f"{'='*60}")

    if init_state is None:
        init_state = {
            "x": b.copy(),  # x^0
            "z": b.copy(),  # z^0, often initialized as x^0
            "y1": np.zeros((m, n)),  # dual for Kx
            "y2": np.zeros((m, n)),  # dual for D1x
            "y3": np.zeros((m, n)),  # dual for D2x
        }

    step_fn = chambolle_pock_step(ops, b, model.gamma, t, s, model.problem)

    return run_solver(
        step_fn,
        init_state,
        ops,
        b,
        model.gamma,
        model.problem,
        maxiter=maxiter,
        tol=tol,
        verbose=verbose,
    )


class ChambollePock:
    """Algorithm 4 with fixed primal/dual steps ``t`` and ``s``."""

    def __init__(self, model: DeblurProblem, t: float, s: float):
        self.model = model
        self.t = float(t)
        self.s = float(s)

    def solve(self, maxiter=500, tol=1e-6, verbose=True, init_state=None):
        return run_chambolle_pock(
            self.model,
            t=self.t,
            s=self.s,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
            init_state=init_state,
        )
