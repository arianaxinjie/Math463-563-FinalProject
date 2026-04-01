# -*- coding: utf-8 -*-
"""
Algorithm 2: Primal-Dual Douglas-Rachford Splitting

    x^k = prox_{t f}(p^{k-1})
    z^k = prox_{t g*}(q^{k-1})
    [w^k; v^k] = [I  tA^T; -tA  I]^{-1} [2x^k - p^{k-1}; 2z^k - q^{k-1}]
    p^k = p^{k-1} + ρ(w^k - x^k)
    q^k = q^{k-1} + ρ(v^k - z^k)
"""

from __future__ import annotations

import numpy as np

from ..problem import DeblurProblem
from ..proximal import prox_f, prox_g_conjugate
from ..solver import run_solver


def primal_dual_dr_step(ops, b, gamma, t, rho, problem, eig_inv):
    """    Returns the single-step function for Primal-Dual Douglas-Rachford (Algorithm 2).

    Problem form:
        min_x  f(x) + g(Ax)
    where in this project:
        f(x) = δ_[0,1](x)
        A = [K; D1; D2]
        g(Kx, D1x, D2x) = data_fidelity(Kx; b) + γ * TV(Dx)

    Algorithm 2 (project PDF):
        x^k = prox_{t f}(p^{k-1})
        z^k = prox_{t g*}(q^{k-1})
        [w^k; v^k] = [I  tA^T; -tA  I]^{-1} [2x^k - p^{k-1}; 2z^k - q^{k-1}]
        p^k = p^{k-1} + ρ(w^k - x^k)
        q^k = q^{k-1} + ρ(v^k - z^k)

    Notes:
        - Use `eig_inv = ops.precompute_inverse(t)` (same t) so that
          (I + t^2 A^T A)^{-1} can be applied via FFT in `ops.solve_system`.
        - State expects keys: 'p', 'q1', 'q2', 'q3' where (q1,q2,q3) correspond to (Kx, D1x, D2x).
    """

    def step(state, k):
        # Current iterates
        p = state["p"]
        q1 = state["q1"]
        q2 = state["q2"]
        q3 = state["q3"]

        # Resolvent of A: prox of f and g*
        xk = prox_f(p, t, problem)
        zk1, zk2, zk3 = prox_g_conjugate(q1, q2, q3, t, b, gamma, problem)

        # Form the RHS blocks
        r1 = 2.0 * xk - p
        r2_1 = 2.0 * zk1 - q1
        r2_2 = 2.0 * zk2 - q2
        r2_3 = 2.0 * zk3 - q3

        # Solve for w via Schur complement:
        #   (I + t^2 A^T A) w = r1 - t A^T r2
        rhs = r1 - t * ops.applyAT(r2_1, r2_2, r2_3)
        wk = ops.solve_system(rhs, eig_inv)

        # Recover v from v = r2 + t A w
        Aw1, Aw2, Aw3 = ops.applyA(wk)
        vk1 = r2_1 + t * Aw1
        vk2 = r2_2 + t * Aw2
        vk3 = r2_3 + t * Aw3

        # Relaxed updates
        state["p"] = p + rho * (wk - xk)
        state["q1"] = q1 + rho * (vk1 - zk1)
        state["q2"] = q2 + rho * (vk2 - zk2)
        state["q3"] = q3 + rho * (vk3 - zk3)

        # Return updated state and current image estimate w^k
        return state, wk

    return step


class PrimalDualDouglasRachford:
    """Algorithm 2; same parameters as Primal DR (t, rho) with Schur FFT solve."""

    def __init__(self, model: DeblurProblem, t: float, rho: float):
        self.model = model
        self.t = float(t)
        self.rho = float(rho)
        self._eig_inv = model.ops.precompute_inverse(self.t)

    def initial_state(self) -> dict:
        m, n = self.model.shape
        b = self.model.b
        return {
            "p": b.copy(),
            "q1": np.zeros((m, n)),
            "q2": np.zeros((m, n)),
            "q3": np.zeros((m, n)),
        }

    def step_fn(self):
        m = self.model
        return primal_dual_dr_step(
            m.ops, m.b, m.gamma, self.t, self.rho, m.problem, self._eig_inv
        )

    def solve(self, maxiter=500, tol=1e-6, verbose=True, compute_obj_every=1):
        m = self.model
        return run_solver(
            self.step_fn(),
            self.initial_state(),
            m.ops,
            m.b,
            m.gamma,
            m.problem,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
            compute_obj_every=compute_obj_every,
        )
