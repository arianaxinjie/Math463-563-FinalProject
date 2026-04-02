# MATH 463/563 Final Project

This repository contains a modularized version of the original Colab work in
`finalproject.py`. The main package is `core_code`, and the course-style entry
point is `optsolve(...)`.

## What To Use

If you are grading, testing, or reusing the project, use:

- `core_code.optsolve`
- `core_code.OptParams`
- helper functions in `core_code` such as `load_image`, `make_kernel`,
  `generate_blurred_noisy_cfg`, and `show_results`

`utils.py` exists only as a compatibility facade around `core_code`.

## Requirements

Recommended Python version: `3.10+`

Install dependencies:

```bash
pip install -r requirements.txt
```

Or minimally:

```bash
pip install numpy scipy matplotlib pillow scikit-image
```

## Repository Layout

- `core_code/`: public package
- `core_code/algorithms/algorithm1_primal_dr/`: Algorithm 1 implementation and local runner
- `core_code/algorithms/algorithm2_primal_dual_dr/`: Algorithm 2 implementation and local runner
- `core_code/algorithms/algorithm3_admm/`: Algorithm 3 implementation and local runner
- `core_code/algorithms/algorithm4_chambolle_pock/`: Algorithm 4 implementation and local runner
- `core_code/algorithms/run_common.py`: shared demo/runner utilities
- `testimages/`: sample images
- `output/`: saved figures from runner scripts
- `project_description_2026.pdf`: assignment description
- `finalproject.py`: original Colab export for reference

## Importing The Package

Run Python from the repository root:

```bash
cd <repo-root>
python
```

Then:

```python
import core_code as cc
```

## Main Entry Point: `optsolve`

Signature:

```python
x = cc.optsolve(problem, method, x0, kernel, b, params)
```

Arguments:

- `problem`: `"l1"` or `"l2"`
- `method`: algorithm name or alias
- `x0`: warm start image, or `None`
- `kernel`: 2D blur kernel
- `b`: observed blurred/noisy image
- `params`: `cc.OptParams(...)`, a `dict`, or `None`

Return value:

- default: restored image `x`
- with `return_all=True`: `(x, obj_history, info)`

Example:

```python
import core_code as cc

img = cc.load_image("testimages/cameraman.jpg")
kernel, x_blur, b = cc.generate_blurred_noisy_cfg(
    x_true=img,
    kernel_kind="gaussian",
    kernel_size=15,
    kernel_sigma=3.0,
    noise_type="gaussian",
    noise_sigma=0.001,
    mode="periodic",
)

params = cc.OptParams(
    maxiter=500,
    gamma=0.01,
    tprimaldr=2.0,
    rhoprimaldr=0.1,
)

x, hist, info = cc.optsolve(
    "l2",
    "douglasrachfordprimal",
    None,
    kernel,
    b,
    params,
    return_all=True,
)
```

## Supported Methods

Canonical methods accepted by `optsolve`:

- Algorithm 1: `douglasrachfordprimal`, `drprimal`, `primaldr`, `algorithm1`
- Algorithm 2: `douglasrachfordprimaldual`, `primaldualdr`, `primaldual`, `algorithm2`
- Algorithm 3: `admm`, `dualdouglasrachford`, `algorithm3`
- Algorithm 4: `chambollepock`, `cp`, `primaldualhybridgradient`, `algorithm4`

## `OptParams`

Defined in `core_code/optsolve.py`.

Main fields:

- `maxiter`
- `tol`
- `verbose`
- `compute_obj_every`
- `gamma`
- `tprimaldr`, `rhoprimaldr`
- `tprimaldualdr`, `rhoprimaldualdr`
- `tadmm`, `rhoadmm`
- `tcp`, `scp`, `cp_step_theta`

`dict` input also supports:

- `gammal1` as an alias for `gamma`
- `t_primal_dr`, `rho_primal_dr`
- `t_primal_dual_dr`, `rho_primal_dual_dr`
- `t_admm`, `rho_admm`
- `t_cp`, `s_cp`

## Useful Public Helpers

Available directly from `core_code`:

- `load_image`
- `make_kernel`
- `add_noise`
- `generate_blurred_noisy_cfg`
- `build_problem`
- `show_results`
- `compare_algorithms`
- `compute_psnr`
- `chambolle_pock_step_sizes`

You can also instantiate solver classes directly:

- `PrimalDouglasRachford`
- `PrimalDualDouglasRachford`
- `ADMM`
- `ChambollePock`

## Per-Algorithm Local Runners

These are mainly for project authors to tune hyperparameters and save demo figures.
Run them from the repository root.

Examples:

```bash
cd <repo-root>
python -m core_code.algorithms.algorithm1_primal_dr.run
python -m core_code.algorithms.algorithm4_chambolle_pock.run testimages/mcgill.jpg
```

Available runner modules:

- `core_code.algorithms.algorithm1_primal_dr.run`
- `core_code.algorithms.algorithm2_primal_dual_dr.run`
- `core_code.algorithms.algorithm3_admm.run`
- `core_code.algorithms.algorithm4_chambolle_pock.run`

## Notes

- The public package is `core_code`; the runner files are not required to use the algorithms.
- The code is organized to preserve the final effective logic from the original Colab notebook while making the structure cleaner.
- `utils.py` re-exports the public API and preserves the old `run_admm` / `run_chambolle_pock` calling style.
