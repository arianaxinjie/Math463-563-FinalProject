# MATH 463/563 Final Project

This repository contains a Python implementation of the final project for
MATH 463/563 Convex Optimization. The code solves TV-regularized image
deblurring and denoising problems using four proximal splitting methods:

- Algorithm 1: Primal Douglas-Rachford
- Algorithm 2: Primal-Dual Douglas-Rachford
- Algorithm 3: ADMM
- Algorithm 4: Chambolle-Pock

The public package is `core_code`. The main user-facing entry point is:

```python
core_code.optsolve(problem, method, x0, kernel, b, params)
```

## 1. Project-Description Alignment

The course project description requires, in substance, that the codebase:

- expose one unified entry point that can run different algorithms
- support both `l1` and `l2` formulations
- accept algorithm names such as `douglasrachfordprimal`, `admm`, and
  `chambollepock`
- allow the user to pass a single parameter structure containing defaults and
  algorithm-specific values
- provide defaults so the code runs even if the user does not specify every
  parameter
- avoid repeating large blocks of algorithm logic in the main driver
- print iterative progress and print a final summary containing the result,
  final objective value, and CPU time
- provide one file that runs all algorithms using default parameters on one
  fixed problem instance and one fixed starting point

This repository satisfies those items as follows:

- unified entry point: `core_code.optsolve`
- shared parameter container: `core_code.OptParams`
- shared iterative loop: `core_code.run_solver`
- reused proximal operators: `core_code/proximal.py`
- reused FFT-based operator code: `core_code/operators.py`
- single-file default driver: `run_all_defaults.py`

## 2. What To Use

If you are grading, testing, or reusing this project, the files you should care
about first are:

- `core_code/optsolve.py`
- `core_code/solver.py`
- `core_code/proximal.py`
- `core_code/operators.py`
- `run_all_defaults.py`

If you want only the high-level API, use:

- `core_code.optsolve`
- `core_code.OptParams`
- helper functions in `core_code` such as `load_image`, `make_kernel`,
  `generate_blurred_noisy_cfg`, `build_problem`, and `show_results`

## 3. Requirements

Recommended Python version:

```bash
python3 --version
```

The project was organized for Python `3.10+`.

Install dependencies:

```bash
pip install -r requirements.txt
```

Minimal dependency set:

```bash
pip install numpy scipy matplotlib pillow scikit-image pypdf
```

## 4. Repository Layout

- `core_code/`
  Main package containing the public solver API and all core numerical code.
- `core_code/optsolve.py`
  Unified project-style driver and parameter definitions.
- `core_code/solver.py`
  Shared iteration loop, logging, stopping rule, and final summary handling.
- `core_code/proximal.py`
  Box, `l1`, isotropic TV, and conjugate proximal operators.
- `core_code/operators.py`
  FFT-based periodic convolution and finite-difference operators.
- `core_code/objective.py`
  Objective-value evaluation for `l1` and `l2` formulations.
- `core_code/data.py`
  Image loading, normalization, blur generation, and noise generation.
- `core_code/algorithms/algorithm1_primal_dr/`
  Algorithm 1 implementation and local runner.
- `core_code/algorithms/algorithm2_primal_dual_dr/`
  Algorithm 2 implementation and local runner.
- `core_code/algorithms/algorithm3_admm/`
  Algorithm 3 implementation and local runner.
- `core_code/algorithms/algorithm4_chambolle_pock/`
  Algorithm 4 implementation and local runner.
- `core_code/algorithms/run_common.py`
  Shared benchmark configuration and helper logic for the per-algorithm runner
  scripts.
- `testimages/`
  Test images used for course experiments.
- `run_all_defaults.py`
  Single-file default driver that runs all four algorithms on one fixed problem
  instance, as requested by the project description.
- `run_recommended_parameters.py`
  Script that reproduces the report's Section 4.6.6 recommended-parameter
  experiment and saves one figure per algorithm.
- `project_description_2026.pdf`
  The assignment specification.
- `math463FinalProject (2).pdf`
  The current project report used to align default and baseline parameter values.
- `extra_grid_search/`
  Additional scripts and artifacts for broader parameter-search experiments.

## 5. Quick Start

### 5.1 Import The Package

From the repository root:

```bash
cd <repo-root>
python
```

Then:

```python
import core_code as cc
```

### 5.2 Run One Algorithm Through The Main Interface

```python
import core_code as cc

img = cc.load_image("testimages/cameraman.jpg")
kernel, x_blur, b = cc.generate_blurred_noisy_cfg(
    x_true=img,
    kernel_kind="gaussian",
    kernel_size=15,
    kernel_sigma=3.0,
    noise_type="salt_pepper",
    noise_density=0.05,
    mode="periodic",
)

params = cc.OptParams(
    maxiter=500,
    tol=1e-4,
    gamma=0.01,
    tprimaldr=2.0,
    rhoprimaldr=0.1,
)

x_rec, obj_hist, info = cc.optsolve(
    "l1",
    "douglasrachfordprimal",
    None,
    kernel,
    b,
    params,
    return_all=True,
)
```

### 5.3 Run The Required Default Driver

This is the file corresponding to the project-description requirement that one
file should run all algorithms with default parameters for a fixed problem and a
fixed starting point.

```bash
python run_all_defaults.py
```

You may also provide a specific image:

```bash
python run_all_defaults.py testimages/cameraman.jpg
```

### 5.4 Run The Recommended-Parameter Experiment

This script reproduces the report experiment built from:

- Section 4.4 baseline setup
- Section 4.6.6 recommended parameters

It runs all four algorithms, shows the reconstruction and convergence plot for
each one, and saves the figures under `output/recommended_parameters/`.

```bash
python run_recommended_parameters.py
```

You may also provide the image explicitly:

```bash
python run_recommended_parameters.py testimages/cameraman.jpg
```

## 6. Main Entry Point: `optsolve`

Signature:

```python
x = cc.optsolve(problem, method, x0, kernel, b, params)
```

Arguments:

- `problem`
  Either `"l1"` or `"l2"`, corresponding to the two formulations in the course
  project.
- `method`
  The algorithm name. The interface accepts both the required canonical names
  and a small set of aliases.
- `x0`
  Initial image iterate. Pass a 2D array with the same shape as `b`, or pass
  `None` to use the package default initialization for that algorithm.
- `kernel`
  The blur kernel.
- `b`
  The observed blurred and noisy image.
- `params`
  Either `None`, an `OptParams` object, or a Python `dict`.

Return value:

- default behavior: restored image `x`
- if `return_all=True`: `(x, obj_history, info)`

The `info` dictionary contains:

- `status`
- `iterations`
- `time`
- `converged`
- `final_obj`

The final printed summary includes:

- a statement of the result
- final objective value
- CPU time

This matches the reporting requirement stated in `project_description_2026.pdf`.

## 7. Supported Methods

Canonical methods accepted by `optsolve`:

- Algorithm 1: `douglasrachfordprimal`
- Algorithm 2: `douglasrachfordprimaldual`
- Algorithm 3: `admm`
- Algorithm 4: `chambollepock`

Also accepted:

- Algorithm 1 aliases: `drprimal`, `primaldr`, `algorithm1`
- Algorithm 2 aliases: `primaldualdr`, `primaldual`, `algorithm2`
- Algorithm 3 aliases: `dualdouglasrachford`, `algorithm3`
- Algorithm 4 aliases: `cp`, `primaldualhybridgradient`, `algorithm4`

## 8. Parameter Interface

The project description requires that the user be able to pass a single
parameter structure containing both common and algorithm-specific values. In
this Python version, that structure is represented by either:

- `core_code.OptParams(...)`
- a Python `dict`

### 8.1 `OptParams` Fields

Defined in `core_code/optsolve.py`.

Common fields:

- `maxiter`
- `tol`
- `verbose`
- `compute_obj_every`
- `gamma`

Algorithm-specific fields:

- Primal DR: `tprimaldr`, `rhoprimaldr`
- Primal-Dual DR: `tprimaldualdr`, `rhoprimaldualdr`
- ADMM: `tadmm`, `rhoadmm`
- Chambolle-Pock: `tcp`, `scp`, `cp_step_theta`

### 8.2 Accepted Course-Style Dictionary Keys

When passing a Python `dict`, the following course-style keys are accepted:

- `gammal1`
- `gammal2`
- `tprimaldr`
- `rhoprimaldr`
- `tprimaldualdr`
- `rhoprimaldualdr`
- `tadmm`
- `rhoadmm`
- `tcp`
- `scp`
- `maxiter`

The following alias spellings are also accepted:

- `t_primal_dr`, `rho_primal_dr`
- `t_primal_dual_dr`, `rho_primal_dual_dr`
- `t_admm`, `rho_admm`
- `t_cp`, `s_cp`

### 8.3 Current Default Values

The current defaults are aligned with the report,
specifically Section 4.4 and Section 4.5.

Common defaults:

- `maxiter = 500`
- `tol = 1e-4`
- `gamma = 0.01`
- `verbose = True`
- `compute_obj_every = 1`

Algorithm-specific defaults:

- Primal DR: `tprimaldr = 2.0`, `rhoprimaldr = 0.1`
- Primal-Dual DR: `tprimaldualdr = 2.0`, `rhoprimaldualdr = 0.1`
- ADMM: `tadmm = 1.0`, `rhoadmm = 1.0`
- Chambolle-Pock: `cp_step_theta = 0.5`

For Chambolle-Pock, if `tcp` and `scp` are both `None`, the code computes stable
step sizes automatically from `cp_step_theta` and the operator norm.

## 9. Fixed Baseline Experiment Used In The Report

The baseline sensitivity-study setup in report uses:

- `problem = l1`
- image: `testimages/cameraman.jpg`
- `kernel_kind = gaussian`
- `kernel_size = 15`
- `kernel_sigma = 3.0`
- `noise_type = salt_pepper`
- `noise_density = 0.05`
- `mode = periodic`
- `gamma = 0.01`
- `maxiter = 500`
- `tol = 1e-4`

The shared runner configuration in `core_code/algorithms/run_common.py` is set to
match that baseline benchmark.

## 10. Per-Algorithm Local Runners

These scripts are mainly for local testing, demonstration, and figure generation.
Run them from the repository root.

Examples:

```bash
python -m core_code.algorithms.algorithm1_primal_dr.run
python -m core_code.algorithms.algorithm2_primal_dual_dr.run
python -m core_code.algorithms.algorithm3_admm.run
python -m core_code.algorithms.algorithm4_chambolle_pock.run testimages/cameraman.jpg
```

Available runner modules:

- `core_code.algorithms.algorithm1_primal_dr.run`
- `core_code.algorithms.algorithm2_primal_dual_dr.run`
- `core_code.algorithms.algorithm3_admm.run`
- `core_code.algorithms.algorithm4_chambolle_pock.run`

These local runners currently use the baseline parameter values from report
Section 4.4 and Section 4.5, not the broader grid-search tuned values from
Section 5.2.

## 11. Recommended-Parameter Script

`run_recommended_parameters.py` is intentionally separate from
`run_all_defaults.py`.

Use `run_all_defaults.py` when you want:

- the assignment-style default driver
- baseline/default parameters
- one file that runs all algorithms with the defaults

Use `run_recommended_parameters.py` when you want:

- the Section 4.6.6 experiment from the report
- the recommended sensitivity-study parameter values
- saved figures for the final reconstructions and convergence plots

The saved figures are:

- `output/recommended_parameters/recommended_primal_dr.png`
- `output/recommended_parameters/recommended_primal_dual_dr.png`
- `output/recommended_parameters/recommended_admm.png`
- `output/recommended_parameters/recommended_chambolle_pock.png`

## 12. Generating Blur And Noise

The project description explicitly asks that the code support different kernels
and noise models. This repository provides helper functions for that:

- `core_code.make_kernel(...)`
- `core_code.add_noise(...)`
- `core_code.generate_blurred_noisy_cfg(...)`

Supported kernel types:

- `gaussian`
- `motion`

Supported noise types:

- `salt_pepper`
- `gaussian`

Example:

```python
import core_code as cc

img = cc.load_image("testimages/cameraman.jpg", resize_factor=0.5)
kernel, x_blur, b = cc.generate_blurred_noisy_cfg(
    x_true=img,
    kernel_kind="motion",
    motion_length=15,
    motion_angle=30,
    noise_type="gaussian",
    noise_sigma=0.01,
    mode="periodic",
)
```

## 13. Public Helper Functions

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

You may also instantiate solver classes directly:

- `PrimalDouglasRachford`
- `PrimalDualDouglasRachford`
- `ADMM`
- `ChambollePock`

## 14. Logging And Output

During iterations, the shared solver loop can print:

- iteration number
- objective value

At the end, it prints a summary including:

- result status
- iteration count
- final objective value
- CPU time
- convergence flag

This behavior is controlled mainly by:

- `verbose`
- `compute_obj_every`
- `maxiter`
- `tol`

## 15. Notes On The Report Versus Grid Search

There are two different notions of parameters in this repository:

- baseline/default parameters
- tuned/grid-search parameters

Baseline/default parameters:

- are the ones documented in report Section 4.4 and 4.5
- are the defaults used by `OptParams`
- are the settings used by `run_all_defaults.py`
- are the settings used by the per-algorithm local runner scripts

Grid-search tuned parameters:

- live in `extra_grid_search/`
- correspond to broader experimental tuning
- are useful for analysis, but they are not the same thing as the project’s
  baseline defaults

This distinction is important so that the code, README, and report do not
contradict one another.
