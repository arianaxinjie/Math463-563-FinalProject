# -*- coding: utf-8 -*-
"""
Proximal operators used by all four algorithms.

These follow the final effective definitions in ``finalproject.py``.
That notebook contains an earlier duplicate ``iso_prox`` cell, but the later
vector soft-thresholding version is the one that actually remains in force.
"""

import numpy as np


def box_prox(x):
    """Projection onto [0, 1]."""
    return np.clip(x, 0.0, 1.0)


def l1_prox(x, b, t):
    """prox_{t||·-b||₁}(x) = b + sign(x-b) max(|x-b|-t, 0)."""
    u = x - b
    return b + np.sign(u) * np.maximum(np.abs(u) - t, 0)


def iso_prox(w1, w2, lam):
    """
    Proximal of ``lam * ||·||_iso`` via vector soft-thresholding.

    This matches the final notebook definition used by ``prox_g``.
    """
    norm = np.sqrt(w1**2 + w2**2)
    scale = np.maximum(1.0 - lam / (norm + 1e-15), 0.0)
    return scale * w1, scale * w2


def prox_f(x, t, problem="l1"):
    """prox_{t f}(x) where ``f = δ_[0,1]``; ``t`` is unused by design."""
    return box_prox(x)


def prox_g(y1, y2, y3, t, b, gamma, problem="l1"):
    """
    prox_{t g}(y1, y2, y3) for the split model

        g(y1, y2, y3) = data_fidelity(y1; b) + gamma * ||(y2, y3)||_iso.

    As in the original notebook, the data term is L1 or squared L2 depending on
    ``problem``, while the TV part always uses ``iso_prox``.
    """
    py2, py3 = iso_prox(y2, y3, t * gamma)

    if problem == "l1":
        py1 = l1_prox(y1, b, t)
    elif problem == "l2":
        py1 = (y1 + 2 * t * b) / (1 + 2 * t)
    else:
        raise ValueError(f"Unsupported problem type: {problem}")

    return py1, py2, py3


def prox_g_conjugate(y1, y2, y3, s, b, gamma, problem="l1"):
    """
    prox_{s g*} via Moreau decomposition:
        prox_{s g*}(v) = v - s * prox_{g/s}(v/s)
    """
    py1, py2, py3 = prox_g(y1 / s, y2 / s, y3 / s, 1.0 / s, b, gamma, problem)
    return (
        y1 - s * py1,
        y2 - s * py2,
        y3 - s * py3,
    )
