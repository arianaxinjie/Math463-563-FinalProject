
"""Objective value for monitoring convergence."""

import numpy as np

from .operators import LinearOperators


def iso_norm(w1, w2):
    return np.sum(np.sqrt(w1**2 + w2**2))


def objective_value(x, b, gamma, ops: LinearOperators, problem="l1"):
    if np.any(x < -1e-10) or np.any(x > 1 + 1e-10):
        return np.inf

    residual = ops.applyK(x) - b
    Dx = ops.applyD(x)
    tv_term = gamma * iso_norm(Dx[:, :, 0], Dx[:, :, 1])

    if problem == "l1":
        data_term = np.sum(np.abs(residual))
    else:
        data_term = np.sum(residual**2)

    return data_term + tv_term
