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

from ...problem import DeblurProblem
from ...proximal import prox_f, prox_g_conjugate
from ...solver import run_solver


def primal_dual_dr_step(ops, b, gamma, t, rho, problem, eig_inv):
    """
    Build one iteration of the Primal-Dual Douglas-Rachford (PD-DR) method.

    The optimization model is

        minimize_x  f(x) + g(Ax),

    with:
        f(x) = δ_[0,1](x),
        A x  = (Kx, D1 x, D2 x),
        g(y1, y2, y3) = data_fidelity(y1; b) + gamma * ||(y2, y3)||_iso.

    In this implementation:
    - the primal variable is `p`,
    - the dual/block variable is `q = (q1, q2, q3)`,
    - `xk = prox_{t f}(p)`,
    - `zk = prox_{t g*}(q)`.

    The code then solves the block linear system

        [ I   t A^T ] [w] = [ 2 xk - p ]
        [ -tA   I   ] [v]   [ 2 zk - q ],

    using the Schur complement. Eliminating v gives

        (I + t^2 A^T A) w = (2 xk - p) - t A^T (2 zk - q).

    After solving for w, the variable v is recovered by

        v = (2 zk - q) + t A w.

    Finally, the relaxed updates are

        p <- p + rho * (w - xk),
        q <- q + rho * (v - zk).

    Parameters
    ----------
    ops : LinearOperators
        Object providing FFT-based applications of A, A^T, and the linear solver.
    b : ndarray
        Observed image.
    gamma : float
        TV regularization weight.
    t : float
        PD-DR step size used in both prox calls and in the block linear system.
    rho : float
        Relaxation parameter in the outer update.
    problem : {'l1', 'l2'}
        Choice of data-fidelity model.
    eig_inv : ndarray
        Precomputed FFT-domain inverse corresponding to

            (I + t^2 A^T A)^{-1}.

        This should be produced by `ops.precompute_inverse(t)` using the same `t`.

    Returns
    -------
    callable
        A closure `step(state, k)` that performs one PD-DR iteration and returns

            (new_state, wk),

        where `wk` is the current image iterate passed back to `run_solver`
        for objective evaluation and convergence monitoring.

    Expected state format
    ---------------------
    state must contain:
        'p'  : primal variable in image space
        'q1' : dual block associated with Kx
        'q2' : dual block associated with D1 x
        'q3' : dual block associated with D2 x
    """

    def step(state, k):
        # Unpack current primal / dual variables
        p = state['p']
        q1 = state['q1']
        q2 = state['q2']
        q3 = state['q3']

        # Proximal evaluations for the two resolvent blocks
        xk = prox_f(p, t, problem)
        zk1, zk2, zk3 = prox_g_conjugate(q1, q2, q3, t, b, gamma, problem)

        # Reflected points: (2 xk - p) and (2 zk - q)
        r1 = 2.0 * xk - p
        r2_1 = 2.0 * zk1 - q1
        r2_2 = 2.0 * zk2 - q2
        r2_3 = 2.0 * zk3 - q3

        # Schur-complement solve for w:
        #   (I + t^2 A^T A) w = r1 - t A^T r2
        rhs = r1 - t * ops.applyAT(r2_1, r2_2, r2_3)
        wk = ops.solve_system(rhs, eig_inv)

        # Recover v from:
        #   v = r2 + t A w
        Aw1, Aw2, Aw3 = ops.applyA(wk)
        vk1 = r2_1 + t * Aw1
        vk2 = r2_2 + t * Aw2
        vk3 = r2_3 + t * Aw3

        # Relaxed PD-DR updates
        state['p'] = p + rho * (wk - xk)
        state['q1'] = q1 + rho * (vk1 - zk1)
        state['q2'] = q2 + rho * (vk2 - zk2)
        state['q3'] = q3 + rho * (vk3 - zk3)

        # Return updated state and the current image iterate used by run_solver
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
