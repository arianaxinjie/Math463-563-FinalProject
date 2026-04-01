# -*- coding: utf-8 -*-
"""Proximal operators for box, L1, isotropic TV, and composite g / g*."""

import numpy as np


def box_prox(x):
    """Projection onto [0, 1]."""
    return np.clip(x, 0.0, 1.0)


def l1_prox(x, b, t):
    """prox_{t||·-b||₁}(x) = b + sign(x-b) max(|x-b|-t, 0)"""
    u = x - b
    return b + np.sign(u) * np.maximum(np.abs(u) - t, 0)


def iso_prox(w1, w2, lam):
    """Proximal of isotropic TV norm (Moreau-style shrinkage)."""
    norm = np.sqrt(w1**2 + w2**2)
    scale = np.maximum(1.0 - lam / (norm + 1e-15), 0.0)
    return scale * w1, scale * w2


def prox_f(x, t, problem="l1"):
    """prox_{t f}(x)  where f = δ_[0,1]."""
    return box_prox(x)


def prox_g(y1, y2, y3, t, b, gamma, problem="l1"):
    """
    prox_{t g}(y1,y2,y3).

    g is separable in (y1) and (y2,y3).
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
    prox_{s g*}  via Moreau decomposition:
        prox_{s g*}(v) = v - s * prox_{g/s}(v/s)
    """
    py1, py2, py3 = prox_g(y1 / s, y2 / s, y3 / s, 1.0 / s, b, gamma, problem)
    return (
        y1 - s * py1,
        y2 - s * py2,
        y3 - s * py3,
    )
