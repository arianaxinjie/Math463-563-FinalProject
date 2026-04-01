# -*- coding: utf-8 -*-
"""Plotting and PSNR."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

import numpy as np


def _open_in_default_viewer(path: str) -> None:
    """Open image file with the OS default app (e.g. Preview on macOS)."""
    path = os.path.abspath(path)
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["open", path], check=False)
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception:
        pass


def _pyplot():
    """
    Load pyplot with an interactive GUI backend (Mac: native window; else Tk).
    Must set ``matplotlib.use`` before the first ``import matplotlib.pyplot``.
    """
    import matplotlib

    if not getattr(_pyplot, "_backend_set", False):
        import sys

        try:
            if sys.platform == "darwin":
                matplotlib.use("MacOSX")
            else:
                matplotlib.use("TkAgg")
        except Exception:
            try:
                matplotlib.use("TkAgg")
            except Exception:
                pass
        _pyplot._backend_set = True

    import matplotlib.pyplot as plt

    plt.ioff()
    return plt


def _to_u8(gray: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(gray, dtype=np.float64), 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _montage_u8(panels: list[np.ndarray]) -> np.ndarray:
    u8s = [_to_u8(p) for p in panels]
    h = max(a.shape[0] for a in u8s)
    padded = []
    for a in u8s:
        if a.shape[0] < h:
            pad = np.zeros((h - a.shape[0], a.shape[1]), dtype=np.uint8)
            a = np.vstack([a, pad])
        padded.append(a)
    return np.hstack(padded)


def _save_pgm(path: str, u8: np.ndarray) -> None:
    """Binary PGM (P5); only needs numpy + stdlib."""
    h, w = u8.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(np.ascontiguousarray(u8).tobytes())


def _save_gray_montage(
    panels: list[np.ndarray],
    path_png: str = "show_results_preview.png",
    path_pgm: str = "show_results_preview.pgm",
) -> str | None:
    """Save horizontally stacked grayscale without matplotlib (PNG if Pillow, else PGM)."""
    montage = _montage_u8(panels)
    try:
        from PIL import Image

        Image.fromarray(montage, mode="L").save(path_png)
        return path_png
    except ImportError:
        _save_pgm(path_pgm, montage)
        return path_pgm


def compute_psnr(original, recovered):
    mse = np.mean((original - recovered) ** 2)
    if mse < 1e-15:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def show_results(
    original,
    blurred,
    recovered,
    title="Result",
    obj_history=None,
    *,
    open_preview=True,
    save_figure=None,
):
    """
    Display or save a 3-panel figure. Without matplotlib, saves PNG/PGM and
    on macOS/Windows/Linux calls the system viewer when ``open_preview`` is True.

    Parameters
    ----------
    save_figure :
        If set (path str or Path), save the matplotlib figure to this PNG file
        (only when matplotlib is available).
    """
    try:
        plt = _pyplot()
    except ImportError:
        if obj_history is not None and len(obj_history) > 0:
            print(
                f"  (objective: first={obj_history[0]:.4f}, "
                f"last={obj_history[-1]:.4f}, len={len(obj_history)})"
            )
        out_path = _save_gray_montage([original, blurred, recovered])
        print(
            f"matplotlib not installed; saved preview to {out_path} "
            f"(pip install matplotlib for in-window plots)"
        )
        if open_preview:
            _open_in_default_viewer(out_path)
            print("  → opened in default viewer (e.g. Preview on Mac)")
        return

    ncols = 4 if obj_history is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(blurred, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Blurred + Noisy")
    axes[1].axis("off")

    axes[2].imshow(recovered, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(title)
    axes[2].axis("off")

    if obj_history is not None:
        axes[3].semilogy(obj_history)
        axes[3].set_xlabel("Iteration")
        axes[3].set_ylabel("Objective Value")
        axes[3].set_title("Convergence")
        axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_figure is not None:
        out = Path(save_figure)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show(block=True)


def compare_algorithms(original, blurred, results_dict):
    try:
        plt = _pyplot()
    except ImportError:
        print(
            "compare_algorithms needs matplotlib. "
            "pip install matplotlib"
        )
        return

    n_algs = len(results_dict)
    fig, axes = plt.subplots(2, n_algs + 1, figsize=(4 * (n_algs + 1), 8))

    axes[0, 0].imshow(blurred, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Blurred + Noisy")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Original")
    axes[1, 0].axis("off")

    for idx, (name, (rec, hist)) in enumerate(results_dict.items(), 1):
        psnr = compute_psnr(original, rec)
        axes[0, idx].imshow(rec, cmap="gray", vmin=0, vmax=1)
        axes[0, idx].set_title(f"{name}\nPSNR={psnr:.2f}dB")
        axes[0, idx].axis("off")

        if hist is not None:
            axes[1, idx].semilogy(hist)
            axes[1, idx].set_title(f"{name} Convergence")
            axes[1, idx].set_xlabel("Iteration")
            axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=True)
