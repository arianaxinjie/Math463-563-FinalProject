# -*- coding: utf-8 -*-
"""
Unified driver matching the course ``optsolve`` pattern (MATLAB grading rubric).

Example (analogous to ``x = optsolve('l1', 'douglasrachfordprimal', x, kernel, b, i);``)::

    from core_code import optsolve, OptParams
    import numpy as np

    x0 = np.zeros_like(b)  # or any warm start; use None for library defaults
    i = OptParams(maxiter=500, gamma=0.049, tprimaldr=2.0, rhoprimaldr=0.1)
    x = optsolve("l1", "douglasrachfordprimal", x0, kernel, b, i)

Pass ``x0=None`` to use each algorithm's default initialization.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping

import numpy as np

from .algorithms.algorithm3_admm.implementation import default_admm_state, run_admm
from .algorithms.algorithm4_chambolle_pock.implementation import (
    default_chambolle_pock_state,
    run_chambolle_pock,
)
from .data import build_problem
from .proximal import box_prox
from .solver import run_solver


@dataclass
class OptParams:
    """
    Hyperparameters (MATLAB struct ``i`` style).

    ``gamma`` matches the project PDF's ``gammal1`` (TV weight in ``build_problem``).
    """

    maxiter: int = 500
    tol: float = 1e-6
    verbose: bool = True
    compute_obj_every: int = 1
    gamma: float = 0.049
    # Primal Douglas–Rachford (Algorithm 1)
    tprimaldr: float = 2.0
    rhoprimaldr: float = 0.1
    # Primal–dual DR (Algorithm 2)
    tprimaldualdr: float = 2.0
    rhoprimaldualdr: float = 0.1
    # ADMM (Algorithm 3)
    tadmm: float = 1.0
    rhoadmm: float = 1.0
    # Chambolle–Pock (Algorithm 4): if tcp and scp are both set, use them;
    # otherwise ``t, s = cp_step_theta / ||A||_2``.
    tcp: float | None = None
    scp: float | None = None
    cp_step_theta: float = 0.5


def _norm_method(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[\s_\-]+", "", s)
    return s


_METHOD_CANONICAL = {
    "douglasrachfordprimal": "primal_dr",
    "drprimal": "primal_dr",
    "primaldr": "primal_dr",
    "algorithm1": "primal_dr",
    "douglasrachfordprimaldual": "primal_dual_dr",
    "primaldualdr": "primal_dual_dr",
    "primaldual": "primal_dual_dr",
    "algorithm2": "primal_dual_dr",
    "admm": "admm",
    "dualdouglasrachford": "admm",
    "algorithm3": "admm",
    "chambollepock": "chambolle_pock",
    "cp": "chambolle_pock",
    "primaldualhybridgradient": "chambolle_pock",
    "algorithm4": "chambolle_pock",
}


def _canonical_method(method: str) -> str:
    key = _norm_method(method)
    if key not in _METHOD_CANONICAL:
        known = ", ".join(sorted(set(_METHOD_CANONICAL.keys())))
        raise ValueError(f"Unknown method {method!r}. Recognized aliases include: {known}")
    return _METHOD_CANONICAL[key]


def merge_params(params: OptParams | Mapping[str, Any] | None) -> OptParams:
    """Build :class:`OptParams` from ``None``, an instance, or a dict (PDF keys allowed)."""
    if params is None:
        return OptParams()
    if isinstance(params, OptParams):
        return params
    if not isinstance(params, Mapping):
        raise TypeError("params must be OptParams, dict, or None")

    d = dict(params)
    # PDF / MATLAB naming
    if "gammal1" in d and "gamma" not in d:
        d["gamma"] = d["gammal1"]
    # Pythonic aliases
    alias_map = {
        "t_primal_dr": "tprimaldr",
        "rho_primal_dr": "rhoprimaldr",
        "t_primal_dual_dr": "tprimaldualdr",
        "rho_primal_dual_dr": "rhoprimaldualdr",
        "t_admm": "tadmm",
        "rho_admm": "rhoadmm",
        "t_cp": "tcp",
        "s_cp": "scp",
    }
    for old, new in alias_map.items():
        if old in d and new not in d:
            d[new] = d[old]

    valid = {f.name for f in fields(OptParams)}
    unknown = set(d) - valid
    if unknown:
        raise ValueError(f"Unknown OptParams keys: {sorted(unknown)}")
    kwargs = {k: d[k] for k in valid if k in d}
    return replace(OptParams(), **kwargs)


def _state_copy(state: dict) -> dict:
    return {k: np.array(v, copy=True) for k, v in state.items()}


def _warm_primal_dr(model, base: dict, x0: np.ndarray) -> dict:
    st = _state_copy(base)
    st["z1"] = box_prox(np.asarray(x0, dtype=np.float64))
    return st


def _warm_primal_dual_dr(model, base: dict, x0: np.ndarray) -> dict:
    st = _state_copy(base)
    st["p"] = box_prox(np.asarray(x0, dtype=np.float64))
    return st


def _warm_admm(model, x0: np.ndarray) -> dict:
    st = default_admm_state(model)
    u = box_prox(np.asarray(x0, dtype=np.float64))
    st["u"] = u
    st["y_1"] = model.ops.applyK(u)
    st["y_2"] = model.ops.applyD1(u)
    st["y_3"] = model.ops.applyD2(u)
    return st


def _warm_chambolle_pock(model, x0: np.ndarray) -> dict:
    st = default_chambolle_pock_state(model)
    x = box_prox(np.asarray(x0, dtype=np.float64))
    st["x"] = x
    st["z"] = x.copy()
    return st


def optsolve(
    problem: str,
    method: str,
    x0: np.ndarray | None,
    kernel: np.ndarray,
    b: np.ndarray,
    params: OptParams | Mapping[str, Any] | None = None,
    *,
    return_all: bool = False,
):
    """
    Run one deblurring algorithm with a MATLAB-like entry point.

    Parameters
    ----------
    problem :
        Data fidelity passed to :func:`build_problem` — ``'l1'`` or ``'l2'``.
    method :
        Algorithm name (case/spacing-insensitive), e.g. ``'douglasrachfordprimal'``.
    x0 :
        Warm-start image (same shape as ``b``), or ``None`` for library defaults.
    kernel, b :
        Blur kernel and observation (same as elsewhere in the project).
    params :
        :class:`OptParams` or dict (supports ``gammal1`` like the PDF).
    return_all :
        If True, return ``(x, obj_history, info)``; else only ``x``.

    Returns
    -------
    np.ndarray or tuple
        Restored image ``x`` in ``[0, 1]``, optionally with diagnostics.
    """
    p = merge_params(params)
    fidelity = problem.lower().strip()
    if fidelity not in ("l1", "l2"):
        raise ValueError("problem must be 'l1' or 'l2'")

    model = build_problem(kernel, b, gamma=p.gamma, fidelity=fidelity)
    m = model
    key = _canonical_method(method)

    if key == "primal_dr":
        from .algorithms.algorithm1_primal_dr.implementation import (
            PrimalDouglasRachford,
        )

        sol = PrimalDouglasRachford(m, t=p.tprimaldr, rho=p.rhoprimaldr)
        st = sol.initial_state()
        if x0 is not None:
            st = _warm_primal_dr(m, st, x0)
        x, obj_hist, info = run_solver(
            sol.step_fn(),
            st,
            m.ops,
            m.b,
            m.gamma,
            m.problem,
            maxiter=p.maxiter,
            tol=p.tol,
            verbose=p.verbose,
            compute_obj_every=p.compute_obj_every,
        )

    elif key == "primal_dual_dr":
        from .algorithms.algorithm2_primal_dual_dr.implementation import (
            PrimalDualDouglasRachford,
        )

        sol = PrimalDualDouglasRachford(
            m, t=p.tprimaldualdr, rho=p.rhoprimaldualdr
        )
        st = sol.initial_state()
        if x0 is not None:
            st = _warm_primal_dual_dr(m, st, x0)
        x, obj_hist, info = run_solver(
            sol.step_fn(),
            st,
            m.ops,
            m.b,
            m.gamma,
            m.problem,
            maxiter=p.maxiter,
            tol=p.tol,
            verbose=p.verbose,
            compute_obj_every=p.compute_obj_every,
        )

    elif key == "admm":
        init = None
        if x0 is not None:
            init = _warm_admm(m, x0)
        x, obj_hist, info = run_admm(
            m,
            t=p.tadmm,
            rho=p.rhoadmm,
            maxiter=p.maxiter,
            tol=p.tol,
            verbose=p.verbose,
            init_state=init,
        )

    elif key == "chambolle_pock":
        from .helpers import chambolle_pock_step_sizes

        if p.tcp is not None and p.scp is not None:
            t, s = float(p.tcp), float(p.scp)
        else:
            t, s, _ = chambolle_pock_step_sizes(m.ops, theta=p.cp_step_theta)

        init = None
        if x0 is not None:
            init = _warm_chambolle_pock(m, x0)

        x, obj_hist, info = run_chambolle_pock(
            m,
            t=t,
            s=s,
            maxiter=p.maxiter,
            tol=p.tol,
            verbose=p.verbose,
            init_state=init,
        )

    else:
        raise RuntimeError(f"Unhandled method key {key!r}")

    if return_all:
        return x, obj_hist, info
    return x


def optparams_as_dict(p: OptParams | None = None) -> dict[str, Any]:
    """Serialize defaults or a given :class:`OptParams` (e.g. for JSON / reports)."""
    return asdict(p if p is not None else OptParams())
