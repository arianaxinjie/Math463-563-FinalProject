# -*- coding: utf-8 -*-
"""Small numerical helpers (e.g. Chambolle–Pock step sizes)."""

import numpy as np

from .operators import LinearOperators


def operator_a_spectral_norm_sq(ops: LinearOperators) -> float:
    """
    Squared spectral norm ||A||_2^2 for A = [K; D1; D2] under periodic BC,
    from max squared singular value = max |lambda|² in the FFT diagonalisation.
    """
    return float(
        np.max(
            np.abs(ops.eig_K) ** 2
            + np.abs(ops.eig_D1) ** 2
            + np.abs(ops.eig_D2) ** 2
        )
    )


def chambolle_pock_step_sizes(ops: LinearOperators, theta: float = 0.5):
    """
    Primal/dual steps t, s with t = s = theta / ||A||_2 so that s*t*||A||^2 < 1.
    """
    lam_max = operator_a_spectral_norm_sq(ops)
    L = np.sqrt(lam_max)
    step = theta / L
    return step, step, lam_max
