

"""
Run all algorithms once using the project's default parameters.

This is the Python analogue of the single grading driver requested in the
course PDF: set common inputs, set per-algorithm defaults, then call
``optsolve(...)`` for each algorithm on the same problem instance and warm
start.

From the repository root:

    python run_all_defaults.py
    python run_all_defaults.py testimages/cameraman.jpg
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

import core_code as cc


REPO_ROOT = Path(__file__).resolve().parent
TESTIMAGES_DIR = REPO_ROOT / "testimages"
DEFAULT_IMAGE = TESTIMAGES_DIR / "cameraman.jpg"


def resolve_image_path(argv: list[str]) -> Path:
    """Use CLI image path if provided, otherwise fall back to testimages/cameraman.jpg."""
    if len(argv) <= 1:
        return DEFAULT_IMAGE

    candidate = Path(argv[1]).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    testimage_candidate = TESTIMAGES_DIR / argv[1]
    if testimage_candidate.is_file():
        return testimage_candidate.resolve()

    raise FileNotFoundError(
        f"Image not found: {argv[1]!r}. Use a valid path or a filename in {TESTIMAGES_DIR}."
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    image_path = resolve_image_path(argv)

    # Fixed problem instance shared by all algorithms.
    x_true = cc.load_image(str(image_path))
    kernel, _x_blur, b = cc.generate_blurred_noisy_cfg(
        x_true=x_true,
        kernel_kind="gaussian",
        kernel_size=15,
        kernel_sigma=3.0,
        noise_type="salt_pepper",
        noise_density=0.05,
        mode="periodic",
    )

    # Same starting point for every algorithm, as requested in the project PDF.
    x0 = np.zeros_like(b)

    # Common default parameters for all algorithms.
    common_params = {
        "maxiter": 500,
        "gammal1": 0.01,
        "tol": 1e-4,
        "verbose": True,
        "compute_obj_every": 1,
    }

    # Default input parameters for each algorithm.
    algorithm_defaults = [
        (
            "douglasrachfordprimal",
            {
                "tprimaldr": 2.0,
                "rhoprimaldr": 0.1,
            },
        ),
        (
            "douglasrachfordprimaldual",
            {
                "tprimaldualdr": 2.0,
                "rhoprimaldualdr": 0.1,
            },
        ),
        (
            "admm",
            {
                "tadmm": 1.0,
                "rhoadmm": 1.0,
            },
        ),
        (
            "chambollepock",
            {
                # Keep CP on its automatic stable default via cp_step_theta.
                "cp_step_theta": 0.5,
            },
        ),
    ]

    print("=" * 72)
    print("Project default driver")
    print(f"image:   {image_path}")
    print("problem: l1")
    print("kernel:  gaussian, size=15, sigma=3.0")
    print("noise:   salt-and-pepper, density=0.05")
    print(f"shape:   {b.shape}")
    print("=" * 72)

    results: list[tuple[str, dict]] = []

    for method, method_params in algorithm_defaults:
        params = {**common_params, **method_params}

        print(f"\nRunning {method} ...")
        x_rec, _obj_hist, info = cc.optsolve(
            "l1",
            method,
            x0.copy(),
            kernel,
            b,
            params,
            return_all=True,
        )
        psnr = cc.compute_psnr(x_true, x_rec)
        results.append((method, {**info, "psnr": psnr}))

        print(
            "Result: "
            f"iterations={info['iterations']}, "
            f"final_obj={info['final_obj']:.6f}, "
            f"time={info['time']:.2f}s, "
            f"psnr={psnr:.3f} dB"
        )

    print("\n" + "=" * 72)
    print("Final summary")
    print("=" * 72)
    for method, info in results:
        print(
            f"{method:28s} | "
            f"iter={info['iterations']:4d} | "
            f"obj={info['final_obj']:12.6f} | "
            f"time={info['time']:7.2f}s | "
            f"psnr={info['psnr']:7.3f} dB | "
            f"converged={info['converged']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
