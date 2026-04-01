# -*- coding: utf-8 -*-
"""
Legacy facade around the ``core_code`` package.

Prefer::

    import core_code
    from core_code import PrimalDouglasRachford

For backward compatibility, ``run_admm`` / ``run_chambolle_pock`` here keep the
old signatures ``(img, kernel, b, ops, ...)``; ``img`` and ``kernel`` are ignored.
"""

from core_code import *  # noqa: F401,F403
from core_code import DeblurProblem


def run_admm(
    img=None,
    kernel=None,
    b=None,
    ops=None,
    problem="l1",
    gamma=0.049,
    t=1.0,
    rho=1.0,
    maxiter=500,
    tol=1e-6,
    verbose=True,
):
    """ADMM with legacy signature; ``img`` and ``kernel`` are unused."""
    if ops is None or b is None:
        raise TypeError("run_admm requires ``ops`` and ``b``")
    from core_code.algorithms.algorithm3_admm import run_admm as _run_admm

    model = DeblurProblem(ops=ops, b=b, gamma=gamma, fidelity=problem)
    return _run_admm(
        model, t=t, rho=rho, maxiter=maxiter, tol=tol, verbose=verbose
    )


def run_chambolle_pock(
    img=None,
    kernel=None,
    b=None,
    ops=None,
    problem="l1",
    gamma=0.049,
    t=0.25,
    s=0.25,
    maxiter=500,
    tol=1e-6,
    verbose=True,
):
    """Chambolle–Pock with legacy signature; ``img`` and ``kernel`` are unused."""
    if ops is None or b is None:
        raise TypeError("run_chambolle_pock requires ``ops`` and ``b``")
    from core_code.algorithms.algorithm4_chambolle_pock import (
        run_chambolle_pock as _run_cp,
    )

    model = DeblurProblem(ops=ops, b=b, gamma=gamma, fidelity=problem)
    return _run_cp(
        model, t=t, s=s, maxiter=maxiter, tol=tol, verbose=verbose
    )
