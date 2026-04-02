# -*- coding: utf-8 -*-
"""
Shared helpers for the per-algorithm ``run.py`` entrypoints.

Keeps image paths, ``testimages`` handling, and package import logic in one place.
Each teammate tunes **hyperparameters only inside their own** ``algorithmN/run.py``
(``PROBLEM`` / ``SOLVER`` / ``CP_STEPS`` blocks), passing values into
``build_demo_model`` and the solver constructors there.

The code package is expected at the repo root as ``core_code`` (or legacy name
``final_project``). Default blur kwargs used when building problems are
``_DEFAULT_BLUR_KWARGS`` below; per-script ``PROBLEM["blur"]`` overrides them.

This file lives under ``core_code/algorithms/``; the repository root is its
grandparent (where ``testimages``, ``output``, and the Python package directory live).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Repository root (contains testimages, core_code or final_project, output)
_ROOT = Path(__file__).resolve().parent.parent.parent
_TESTIMAGES = _ROOT / "testimages"
_DEFAULT_NAME = "cameraman.jpg"
OUTPUT_DIR = _ROOT / "output"
DEFAULT_PROBLEM = {
    "gamma": 0.01,
    "fidelity": "l2",
    "blur": {
        "kernel_kind": "gaussian",
        "kernel_size": 15,
        "kernel_sigma": 3.0,
        "noise_type": "gaussian",
        "noise_sigma": 0.001,
        "mode": "periodic",
    },
}


def ensure_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("Install matplotlib to show figures:")
        print("  pip install matplotlib")
        sys.exit(1)


def get_final_project():
    """
    Prepend the repo root to ``sys.path`` and import the main project package.

    Tries ``core_code`` first, then ``final_project`` (legacy folder name).
    """
    r = str(_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)
    import importlib

    for name in ("core_code", "final_project"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(
        f"No package 'core_code' or 'final_project' under {_ROOT}. "
        "Expected a directory core_code/ (or final_project/) with __init__.py at the repo root."
    )


def resolve_input_path(arg: str | None) -> Path:
    """Resolve CLI image argument: default file, existing path, or ``testimages/<name>``."""
    if not arg:
        p = (_TESTIMAGES / _DEFAULT_NAME).resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(
            f"Default image not found: {p}\n"
            f"  Place the course ``testimages`` folder at: {_TESTIMAGES}"
        )

    p = Path(arg).expanduser()
    if p.is_file():
        return p.resolve()

    cand = (_TESTIMAGES / arg).resolve()
    if cand.is_file():
        return cand

    raise FileNotFoundError(
        f"Image not found: {arg!r}\n"
        f"  Use an absolute/relative path, or a filename inside ``testimages``.\n"
        f"  testimages directory: {_TESTIMAGES}"
    )


def load_demo_image(fp, argv: list[str]) -> tuple[np.ndarray, str]:
    """Load demo image from CLI or defaults; returns ``(array, description_for_logging)``."""
    try:
        arg = argv[1] if len(argv) > 1 else None
        path = resolve_input_path(arg)
        return fp.load_image(str(path)), str(path)
    except FileNotFoundError as e:
        if _TESTIMAGES.is_dir():
            names = sorted(p.name for p in _TESTIMAGES.iterdir() if p.is_file())
            extra = f"\n  Files currently in testimages: {names}" if names else ""
            print(str(e) + extra)
            sys.exit(1)

    if not _TESTIMAGES.is_dir():
        try:
            from skimage import data

            return (
                fp.load_image(data.camera()),
                "scikit-image data.camera() (no testimages directory)",
            )
        except Exception:
            x = np.linspace(0, 1, 256)
            y = np.linspace(0, 1, 256)
            xx, yy = np.meshgrid(x, y)
            synth = np.sin(8 * np.pi * xx) * np.sin(8 * np.pi * yy) * 0.5 + 0.5
            return (
                synth,
                "Synthetic sine pattern (no testimages and no scikit-image)",
            )

    print(
        f"Could not load an image. Check: {_TESTIMAGES / _DEFAULT_NAME}",
        file=sys.stderr,
    )
    sys.exit(1)


# Defaults for ``generate_blurred_noisy_cfg``; override per-algorithm via ``blur=`` in
# ``build_demo_model`` (each ``algorithm*/run.py`` passes its own dict).
_DEFAULT_BLUR_KWARGS: dict = {
    "kernel_kind": "gaussian",
    "kernel_size": 15,
    "kernel_sigma": 3.0,
    "noise_type": "gaussian",
    "noise_sigma": 0.001,
    "mode": "periodic",
}


def build_demo_model(
    fp,
    img: np.ndarray,
    *,
    gamma: float = 0.01,
    fidelity: str = "l2",
    blur: dict | None = None,
):
    """
    Build blurred noisy observation and ``DeblurProblem``.

    Parameters
    ----------
    gamma :
        TV regularization weight in ``build_problem`` (larger → smoother solution).
    fidelity :
        Data fidelity in ``build_problem`` (e.g. ``"l2"``, ``"l1"`` per ``core_code``).
    blur :
        Optional overrides forwarded to ``generate_blurred_noisy_cfg`` (kernel, noise,
        ``mode``, etc.). Merged on top of ``_DEFAULT_BLUR_KWARGS``.
    """
    blur_kw = {**_DEFAULT_BLUR_KWARGS}
    if blur:
        blur_kw.update(blur)
    kernel, _x_blur, b = fp.generate_blurred_noisy_cfg(x_true=img, **blur_kw)
    model = fp.build_problem(kernel, b, gamma=gamma, fidelity=fidelity)
    return b, model


def generate_blurred_observation(fp, img: np.ndarray, blur: dict | None = None):
    """
    Same blur defaults as :func:`build_demo_model`, but return ``(kernel, x_blur, b)``
    for drivers that need ``kernel`` and ``b`` separately (e.g. :func:`optsolve`).
    """
    blur_kw = {**_DEFAULT_BLUR_KWARGS}
    if blur:
        blur_kw.update(blur)
    return fp.generate_blurred_noisy_cfg(x_true=img, **blur_kw)


def figure_path(algorithm_index: int) -> Path:
    """Path for saved figure ``output/algorithm{N}_result.png``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"algorithm{algorithm_index}_result.png"


def run_algorithm_demo(
    *,
    algorithm_index: int,
    title: str,
    build_solver,
    solve_kwargs: dict,
    argv: list[str] | None = None,
    rng_seed: int = 42,
    problem: dict | None = None,
) -> tuple[np.ndarray, list[float], dict]:
    """
    Shared runner for the per-algorithm ``run.py`` entrypoints.

    ``build_solver(fp, model)`` must return a solver instance, or a
    ``(solver, extra_solve_kwargs)`` pair when a specific script needs to add
    parameters dynamically.
    """
    ensure_matplotlib()
    fp = get_final_project()

    np.random.seed(rng_seed)
    cli_args = sys.argv if argv is None else argv
    problem_cfg = DEFAULT_PROBLEM if problem is None else problem

    img, source_note = load_demo_image(fp, cli_args)
    print(f"[input image] {source_note}")

    b, model = build_demo_model(
        fp,
        img,
        gamma=problem_cfg["gamma"],
        fidelity=problem_cfg["fidelity"],
        blur=problem_cfg["blur"],
    )

    solver_info = build_solver(fp, model)
    extra_solve_kwargs = {}
    if isinstance(solver_info, tuple):
        solver, extra_solve_kwargs = solver_info
    else:
        solver = solver_info

    x_sol, obj_hist, info = solver.solve(**{**solve_kwargs, **extra_solve_kwargs})

    out = figure_path(algorithm_index)
    print("PSNR:", fp.compute_psnr(img, x_sol), "dB")
    fp.show_results(
        img,
        b,
        x_sol,
        title,
        obj_hist,
        save_figure=out,
    )
    print(f"[saved figure] {out.resolve()}")
    return x_sol, obj_hist, info
