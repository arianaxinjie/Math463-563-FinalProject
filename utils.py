# -*- coding: utf-8 -*-
"""
Legacy facade around the ``core_code`` package.

Prefer::

    import core_code
    from core_code import PrimalDouglasRachford

For backward compatibility, ``run_admm`` / ``run_chambolle_pock`` here keep the
old signatures ``(img, kernel, b, ops, ...)``; ``img`` and ``kernel`` are ignored.
"""

import core_code as _cc

ADMM = _cc.ADMM
ChambollePock = _cc.ChambollePock
DeblurProblem = _cc.DeblurProblem
LinearOperators = _cc.LinearOperators
OptParams = _cc.OptParams
PrimalDouglasRachford = _cc.PrimalDouglasRachford
PrimalDualDouglasRachford = _cc.PrimalDualDouglasRachford
add_noise = _cc.add_noise
admm_step = _cc.admm_step
apply_periodic_conv = _cc.apply_periodic_conv
box_prox = _cc.box_prox
build_problem = _cc.build_problem
chambolle_pock_step = _cc.chambolle_pock_step
chambolle_pock_step_sizes = _cc.chambolle_pock_step_sizes
compare_algorithms = _cc.compare_algorithms
compute_psnr = _cc.compute_psnr
eig_vals_periodic_conv = _cc.eig_vals_periodic_conv
generate_blurred_noisy_cfg = _cc.generate_blurred_noisy_cfg
iso_norm = _cc.iso_norm
iso_prox = _cc.iso_prox
l1_prox = _cc.l1_prox
load_image = _cc.load_image
make_kernel = _cc.make_kernel
merge_params = _cc.merge_params
objective_value = _cc.objective_value
operator_a_spectral_norm_sq = _cc.operator_a_spectral_norm_sq
optsolve = _cc.optsolve
optparams_as_dict = _cc.optparams_as_dict
primal_dr_step = _cc.primal_dr_step
primal_dual_dr_step = _cc.primal_dual_dr_step
prox_f = _cc.prox_f
prox_g = _cc.prox_g
prox_g_conjugate = _cc.prox_g_conjugate
run_solver = _cc.run_solver
show_results = _cc.show_results

__all__ = list(_cc.__all__) + ["run_admm", "run_chambolle_pock"]


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
    from core_code.algorithms.algorithm3_admm.implementation import (
        run_admm as _run_admm,
    )

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
    from core_code.algorithms.algorithm4_chambolle_pock.implementation import (
        run_chambolle_pock as _run_cp,
    )

    model = DeblurProblem(ops=ops, b=b, gamma=gamma, fidelity=problem)
    return _run_cp(
        model, t=t, s=s, maxiter=maxiter, tol=tol, verbose=verbose
    )
