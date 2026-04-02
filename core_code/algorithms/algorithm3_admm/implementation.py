# -*- coding: utf-8 -*-
"""
Algorithm 3: ADMM (Dual Douglas-Rachford)

As described in Section 1.2.3 of the project description, ADMM is exactly
the Douglas-Rachford algorithm applied to the dual problem:

    min_{z}  f*(-A^T z) + g*(z)  :=  h(z) + k(z)    (11)

When phi = 1, this simplifies to the standard ADMM form:
    x_hat^k = argmin_x  L(x, y_hat^{k-1})
    y_hat^k = argmin_y  L(x_hat^k, y)
    z^k     = z^{k-1} + t(A x_hat^k - y_hat^k)

where the augmented Lagrangian is given by:
    L(x,y,z) = f(x) + g(y) + z^T(Ax - y) + (t/2)||Ax - y||^2

For our specific problem, we introduce a new variable u with x = u.
The augmented Lagrangian becomes:
    L(x,u,y,w,z) = f(u) + g(y) + w^T(x - u) + z^T(Ax - y)
                    + (t/2)( ||x - u||^2 + ||Ax - y||^2 )

Alternating minimization over x and then (u,y) gives Algorithm 3.

For the overrelaxed version (0 < phi < 2), we replace A x^k with:
    phi A x^k + (1 - phi) y^{k-1}
in the u and y updates.

Variable dimensions (with A = [K; D]):
    x^k, u^k, w^k  in  R^{n^2}
    y^k, z^k       in  R^{3n^2}
"""

from __future__ import annotations

import numpy as np

from ...problem import DeblurProblem
from ...proximal import prox_f, prox_g
from ...solver import run_solver


def default_admm_state(model: DeblurProblem) -> dict:
    """Default ADMM initialization shared by wrappers and warm starts."""
    ops, b = model.ops, model.b
    m, n = b.shape
    return {
        "u": b.copy(),
        "y_1": ops.applyK(b),
        "y_2": ops.applyD1(b),
        "y_3": ops.applyD2(b),
        "w": np.zeros((m, n)),
        "z_1": np.zeros((m, n)),
        "z_2": np.zeros((m, n)),
        "z_3": np.zeros((m, n)),
    }


def admm_step(ops, b, gamma, t, rho, problem, eig_inv):
    """
    Returns the single step function for ADMM (Algorithm 3).

    ADMM:
        x^k = (I + A^T A)^{-1} (u^{k-1} + A^T y^{k-1} - 1/t (w^{k-1} + A^T z^{k-1}))
        u^k = prox_{t^{-1} f}(phi*x^k + (1-phi)*u^{k-1} + w^{k-1}/t)
        y^k = prox_{t^{-1} g}(phi*A*x^k + (1-phi)*y^{k-1} + z^{k-1}/t)
        w^k = w^{k-1} + t*(x^k - u^k)
        z^k = z^{k-1} + t*(A*x^k - y^k)

    Unlike Primal DR which uses (I + t^2 A^T A)^{-1}, here the linear system
    is just (I + A^T A)^{-1}, so eig_inv should be precomputed with t=1.
    Also, prox operators here use step size 1/t instead of t.
    """
    # precompute 1/t since we use it a lot in this algorithm
    t_inv = 1.0 / t

    def step(state, k):
        # grab all variables from the previous iteration
        u = state["u"]  # u^{k-1}, image-sized
        y_1 = state["y_1"]  # y^{k-1} component for Kx
        y_2 = state["y_2"]  # y^{k-1} component for D1*x
        y_3 = state["y_3"]  # y^{k-1} component for D2*x
        w = state["w"]  # w^{k-1}, dual multiplier for x = u
        z_1 = state["z_1"]  # z^{k-1} component for Kx = y_1
        z_2 = state["z_2"]  # z^{k-1} component for D1*x = y_2
        z_3 = state["z_3"]  # z^{k-1} component for D2*x = y_3

        # Line 2: x-update — minimize the augmented Lagrangian L over x
        # This comes from setting grad_x L = 0, which gives the linear system
        # (I + A^T A) x = u + A^T y - (1/t)(w + A^T z)
        # We solve it in the frequency domain using precomputed eigenvalues
        ATy = ops.applyAT(y_1, y_2, y_3)  # A^T y^{k-1}
        ATz = ops.applyAT(z_1, z_2, z_3)  # A^T z^{k-1}
        rhs = u + ATy - t_inv * (w + ATz)
        xk = ops.solve_system(rhs, eig_inv)

        # compute A*x^k = [K*x^k ; D1*x^k ; D2*x^k], needed for lines 4 and 6
        Axk_1 = ops.applyK(xk)
        Axk_2 = ops.applyD1(xk)
        Axk_3 = ops.applyD2(xk)

        # Line 3: u-update — prox of f with step 1/t
        # overrelaxation: blend phi*x^k with (1-phi)*u^{k-1} (see note after Eq. 17)
        # f(x) = delta_S(x) so prox is just the box projection onto [0,1]
        u_input = rho * xk + (1 - rho) * u + w * t_inv
        uk = prox_f(u_input, t_inv, problem)

        # Line 4: y-update — prox of g with step 1/t
        # same overrelaxation applied to A*x^k
        # g splits into data fidelity (||y1-b||_1 or ||y1-b||_2^2) and TV (gamma ||(y2,y3)||_iso)
        y_input_1 = rho * Axk_1 + (1 - rho) * y_1 + z_1 * t_inv
        y_input_2 = rho * Axk_2 + (1 - rho) * y_2 + z_2 * t_inv
        y_input_3 = rho * Axk_3 + (1 - rho) * y_3 + z_3 * t_inv
        yk_1, yk_2, yk_3 = prox_g(
            y_input_1, y_input_2, y_input_3, t_inv, b, gamma, problem
        )

        # Line 5: dual update for the constraint x = u
        wk = w + t * (xk - uk)

        # Line 6: dual update for the constraint Ax = y
        zk_1 = z_1 + t * (Axk_1 - yk_1)
        zk_2 = z_2 + t * (Axk_2 - yk_2)
        zk_3 = z_3 + t * (Axk_3 - yk_3)

        # store everything for the next iteration
        state["u"] = uk
        state["y_1"] = yk_1
        state["y_2"] = yk_2
        state["y_3"] = yk_3
        state["w"] = wk
        state["z_1"] = zk_1
        state["z_2"] = zk_2
        state["z_3"] = zk_3

        return state, xk

    # Return the step function to be used by run_solver
    return step


# convenience wrapper to set up and run ADMM with sensible defaults

def run_admm(
    model: DeblurProblem,
    t=1.0,
    rho=1.0,
    maxiter=500,
    tol=1e-6,
    verbose=True,
    init_state=None,
):
    """
    Run ADMM (Algorithm 3) for image deblurring/denoising.
    Handles initialization of all 8 state variables and precomputation
    of the linear system, then delegates to run_solver.

    Parameters
    ----------
    model :
        ``DeblurProblem`` with ``ops``, ``b``, ``gamma``, ``fidelity``.
    t :
        Penalty parameter in the augmented Lagrangian.
    rho :
        Relaxation parameter phi in (0,2); phi=1 is standard ADMM (Eqs. 15-17).
    maxiter, tol, verbose :
        Same as :func:`core_code.solver.run_solver`.
    """
    ops, b = model.ops, model.b

    # ADMM's x-update solves (I + A^T A)x = rhs, NOT (I + t^2 A^T A)x = rhs
    # (the t cancels when you divide the augmented Lagrangian grad by t)
    # so we precompute with t=1 here regardless of the actual penalty t
    eig_inv = ops.precompute_inverse(t=1.0)

    # initialize: u^0 = b, y^0 = Ab, dual variables w^0 = z^0 = 0
    if init_state is None:
        init_state = default_admm_state(model)

    step_fn = admm_step(ops, b, model.gamma, t, rho, model.problem, eig_inv)

    if verbose:
        print(f"\n{'='*60}")
        print(
            f"  ADMM (Algorithm 3) — problem={model.problem}, "
            f"gamma={model.gamma}, t={t}, rho={rho}"
        )
        print(f"{'='*60}")

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


class ADMM:
    """Algorithm 3; delegates to :func:`run_admm`."""

    def __init__(self, model: DeblurProblem, t: float = 1.0, rho: float = 1.0):
        self.model = model
        self.t = float(t)
        self.rho = float(rho)

    def initial_state(self) -> dict:
        return default_admm_state(self.model)

    def solve(self, maxiter=500, tol=1e-6, verbose=True, init_state=None):
        return run_admm(
            self.model,
            t=self.t,
            rho=self.rho,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
            init_state=init_state,
        )
