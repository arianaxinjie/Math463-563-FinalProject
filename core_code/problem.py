# -*- coding: utf-8 -*-
"""Single deblurring instance: linear operators, observation, regularization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operators import LinearOperators


@dataclass(frozen=True)
class DeblurProblem:
    """
    Immutable description of one TV deblurring problem.

    Attributes
    ----------
    ops :
        FFT-periodic operators for blur K and gradient D (same shape as ``b``).
    b :
        Observed blurred + noisy image, shape (m, n), values in [0, 1].
    gamma :
        Isotropic TV weight.
    fidelity :
        ``'l1'`` for L1 data term, ``'l2'`` for squared L2 data term.
    """

    ops: LinearOperators
    b: np.ndarray
    gamma: float
    fidelity: str = "l2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "b", np.asarray(self.b, dtype=np.float64))
        if self.fidelity not in ("l1", "l2"):
            raise ValueError("fidelity must be 'l1' or 'l2'")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.b.shape[0]), int(self.b.shape[1])

    @property
    def problem(self) -> str:
        """Alias matching legacy ``problem='l1'|'l2'`` parameter."""
        return self.fidelity
