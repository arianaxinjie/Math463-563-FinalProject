# -*- coding: utf-8 -*-
"""
Algorithm 1: Primal Douglas-Rachford Splitting

    x^k = prox_{tf}(z1^{k-1})
    y^k = prox_{tg}(z2^{k-1})
    u^k = (I + t^2 A^T A)^{-1}(2x^k - z1^{k-1} + t A^T(2y^k - z2^{k-1}))
    v^k = A(u^k)
    z1^k = z1^{k-1} + ρ(u^k - x^k)
    z2^k = z2^{k-1} + ρ(v^k - y^k)
"""

from __future__ import annotations

import numpy as np

from ..operators import apply_periodic_conv
from ..problem import DeblurProblem
from ..proximal import prox_f, prox_g
from ..solver import run_solver


def primal_dr_step(ops, b, gamma, t, rho, problem, eig_inv):
    """
    Returns the single step function for Primal Douglas-Rachford (Algorithm 1).

    Algorithm 1:
        x^k = prox_{tf}(z1^{k-1})
        y^k = prox_{tg}(z2^{k-1})
        u^k = (I+A^TA)^{-1}(2x^k - z1^{k-1} + A^T(2y^k - z2^{k-1}))
        v^k = A(u^k)
        z1^k = z1^{k-1} + ρ(u^k - x^k)
        z2^k = z2^{k-1} + ρ(v^k - y^k)
    """

    def step(state, k):
        # Step1: Extract current auxiliary variables from state
        # z1 tracks the primal variable, which is image-sized
        z1 = state["z1"]
        # z2_1/2/3 track the dual variables corresponding to K, D1, D2 respectively
        z2_1 = state["z2_1"]  # Kx part
        z2_2 = state["z2_2"]  # Horizontal gradient
        z2_3 = state["z2_3"]  # Vertical gradient

        # Step 2: resolvent of A (f part), which is projection of z1 onto [0,1]
        xk = prox_f(z1, t, problem)

        # Step 3: resolvent of A (g part)
        # yk_1: prox of l1-norm ||y1 - b||_1 (if l1) or l2-norm ||y1 - b||_2^2 (if l2)
        # yk_2, yk_3: joint prox of isotropic TV term gamma*||(y2, y3)||_iso
        yk_1, yk_2, yk_3 = prox_g(z2_1, z2_2, z2_3, t, b, gamma, problem)

        # Step 4: resolvent of B
        # rhs = 2x^k - z1 + A^T(2y^k - z2)
        # A^T is split into three parts: K^T, D1^T, D2^T
        rhs = (
            2 * xk
            - z1
            + ops.applyKT(2 * yk_1 - z2_1)  # transpose of blur operator
            + apply_periodic_conv(2 * yk_2 - z2_2, ops.eig_D1T)  # transpose of horizontal gradient
            + apply_periodic_conv(2 * yk_3 - z2_3, ops.eig_D2T)  # transpose of vertical gradient
        )
        # eig_inv is precomputed in the frequency domain to avoid expensive direct matrix inversion
        uk = ops.solve_system(rhs, eig_inv)

        # Step 5: v^k = A(u^k), and apply each component of A separately to u^k
        vk_1 = ops.applyK(uk)  # apply blur operator
        vk_2 = ops.applyD1(uk)  # apply horizontal gradient operator
        vk_3 = ops.applyD2(uk)  # apply vertical gradient operator

        # Steps 6-7: Update z
        state["z1"] = z1 + rho * (uk - xk)
        state["z2_1"] = z2_1 + rho * (vk_1 - yk_1)
        state["z2_2"] = z2_2 + rho * (vk_2 - yk_2)
        state["z2_3"] = z2_3 + rho * (vk_3 - yk_3)

        # Return updated state and current image estimate u^k
        return state, uk

    # Return the step function to be used by run_solver
    return step


class PrimalDouglasRachford:
    """
    Algorithm 1 with fixed (t, rho); call :meth:`solve` to run iterations.

    Parameters
    ----------
    model :
        ``DeblurProblem`` (ops, observation ``b``, ``gamma``, ``fidelity``).
    t, rho :
        DR step and relaxation parameters (same meaning as in the project PDF).
    """

    def __init__(self, model: DeblurProblem, t: float, rho: float):
        self.model = model
        self.t = float(t)
        self.rho = float(rho)
        self._eig_inv = model.ops.precompute_inverse(self.t)

    def initial_state(self) -> dict:
        m, n = self.model.shape
        b = self.model.b
        return {
            "z1": b.copy(),
            "z2_1": np.zeros((m, n)),
            "z2_2": np.zeros((m, n)),
            "z2_3": np.zeros((m, n)),
        }

    def step_fn(self):
        m = self.model
        return primal_dr_step(
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
