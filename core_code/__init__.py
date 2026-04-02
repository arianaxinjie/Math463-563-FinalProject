# -*- coding: utf-8 -*-
"""
MATH 463 / COMP 463 image deblurring with TV regularisation.

**Typical usage**

::

    from core_code import build_problem, load_image, make_kernel, generate_blurred_noisy_cfg
    from core_code import PrimalDouglasRachford, ADMM

    img = load_image("cameraman.jpg")
    kernel, x_blur, b = generate_blurred_noisy_cfg(img, ...)
    model = build_problem(kernel, b, gamma=0.01, fidelity="l2")
    solver = PrimalDouglasRachford(model, t=2.0, rho=0.1)
    x_sol, obj_hist, info = solver.solve(maxiter=500, tol=5e-5)

Low-level step functions (``primal_dr_step``, etc.) remain available under
``core_code.algorithms`` for custom loops. Implementations now live under
``core_code.algorithms.algorithm1_primal_dr`` …
``algorithm4_chambolle_pock``.
"""

from .algorithms import (
    ADMM,
    ChambollePock,
    PrimalDouglasRachford,
    PrimalDualDouglasRachford,
    admm_step,
    chambolle_pock_step,
    primal_dr_step,
    primal_dual_dr_step,
    run_admm,
    run_chambolle_pock,
)
from .data import (
    add_noise,
    build_problem,
    generate_blurred_noisy_cfg,
    load_image,
    make_kernel,
)
from .helpers import chambolle_pock_step_sizes, operator_a_spectral_norm_sq
from .objective import iso_norm, objective_value
from .operators import LinearOperators, apply_periodic_conv, eig_vals_periodic_conv
from .optsolve import OptParams, merge_params, optsolve, optparams_as_dict
from .problem import DeblurProblem
from .proximal import box_prox, iso_prox, l1_prox, prox_f, prox_g, prox_g_conjugate
from .solver import run_solver
from .viz import compare_algorithms, compute_psnr, show_results

__all__ = [
    "ADMM",
    "ChambollePock",
    "DeblurProblem",
    "PrimalDouglasRachford",
    "PrimalDualDouglasRachford",
    "LinearOperators",
    "add_noise",
    "admm_step",
    "apply_periodic_conv",
    "box_prox",
    "build_problem",
    "chambolle_pock_step",
    "chambolle_pock_step_sizes",
    "compare_algorithms",
    "compute_psnr",
    "eig_vals_periodic_conv",
    "generate_blurred_noisy_cfg",
    "iso_norm",
    "iso_prox",
    "l1_prox",
    "load_image",
    "make_kernel",
    "objective_value",
    "OptParams",
    "operator_a_spectral_norm_sq",
    "optsolve",
    "optparams_as_dict",
    "merge_params",
    "primal_dr_step",
    "primal_dual_dr_step",
    "prox_f",
    "prox_g",
    "prox_g_conjugate",
    "run_admm",
    "run_chambolle_pock",
    "run_solver",
    "show_results",
]
