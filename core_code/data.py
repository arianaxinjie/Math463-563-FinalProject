# -*- coding: utf-8 -*-
"""Image I/O, blur kernels, synthetic observations."""

import numpy as np
from scipy import ndimage

try:
    from skimage.util import random_noise as _sk_random_noise
except ImportError:
    _sk_random_noise = None


def _add_noise_numpy(img, noise_type, density, mean, sigma):
    """Fallback when scikit-image is not installed."""
    if noise_type == "salt_pepper":
        out = img.copy()
        mask = np.random.random(img.shape) < density
        salt = np.random.random(img.shape) < 0.5
        out = np.where(mask & salt, 1.0, out)
        out = np.where(mask & (~salt), 0.0, out)
        return out
    if noise_type == "gaussian":
        return img + np.random.normal(mean, sigma, img.shape)
    raise ValueError("Unsupported noise type. Use 'salt_pepper' or 'gaussian'.")


def load_image(path, resize_factor=1.0):
    if isinstance(path, np.ndarray):
        img_array = path
    else:
        try:
            from PIL import Image
        except ImportError as err:
            raise ImportError(
                "load_image() needs Pillow to open files. "
                "Install: pip install pillow"
            ) from err
        img = Image.open(path)
        if resize_factor != 1.0:
            new_size = (
                int(img.width * resize_factor),
                int(img.height * resize_factor),
            )
            img = img.resize(new_size, Image.LANCZOS)
        img_array = np.array(img)

    if img_array.ndim == 3:
        img_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])

    img_array = np.asarray(img_array, dtype=np.float64)
    img_array = img_array - img_array.min()
    if img_array.max() > 0:
        img_array = img_array / img_array.max()

    return img_array


def make_kernel(kind="gaussian", hsize=15, sigma=3.0, length=15, theta=0):
    """
    Generate a blur kernel.

    Supported kinds:  "gaussian", "motion"
    """
    kind = kind.lower()

    if kind == "gaussian":
        if isinstance(hsize, int):
            hsize = (hsize, hsize)
        h, w = hsize
        cy, cx = (h - 1) / 2, (w - 1) / 2
        y, x = np.mgrid[0:h, 0:w]
        kernel = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    if kind == "motion":
        half = (length - 1) / 2
        theta_rad = np.deg2rad(theta)
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)

        size = int(np.ceil(half * max(abs(cos_t), abs(sin_t))) * 2 + 1)
        size = max(size, length)
        kernel = np.zeros((size, size))
        center = (size - 1) / 2

        for i in range(length):
            t = i - half
            x = center + t * cos_t
            y = center - t * sin_t

            xi, yi = int(np.floor(x)), int(np.floor(y))
            fx, fy = x - xi, y - yi

            if 0 <= xi < size and 0 <= yi < size:
                kernel[yi, xi] += (1 - fx) * (1 - fy)
            if 0 <= xi + 1 < size and 0 <= yi < size:
                kernel[yi, xi + 1] += fx * (1 - fy)
            if 0 <= xi < size and 0 <= yi + 1 < size:
                kernel[yi + 1, xi] += (1 - fx) * fy
            if 0 <= xi + 1 < size and 0 <= yi + 1 < size:
                kernel[yi + 1, xi + 1] += fx * fy

        kernel = kernel / kernel.sum()
        return kernel

    raise ValueError("Unsupported kernel kind. Use 'gaussian' or 'motion'.")


def add_noise(
    img,
    noise_type="salt_pepper",
    density=0.05,
    mean=0.0,
    sigma=0.01,
):
    img = np.asarray(img, dtype=np.float64)
    noise_type = noise_type.lower()

    if _sk_random_noise is not None:
        if noise_type == "salt_pepper":
            noisy_img = _sk_random_noise(img, mode="s&p", amount=density)
        elif noise_type == "gaussian":
            noisy_img = _sk_random_noise(
                img, mode="gaussian", mean=mean, var=sigma**2
            )
        else:
            raise ValueError(
                "Unsupported noise type. Use 'salt_pepper' or 'gaussian'."
            )
    else:
        noisy_img = _add_noise_numpy(img, noise_type, density, mean, sigma)

    return np.clip(noisy_img, 0.0, 1.0)


def generate_blurred_noisy_cfg(
    x_true,
    kernel_kind="gaussian",
    kernel_size=15,
    kernel_sigma=3.0,
    motion_angle=0.0,
    motion_length=None,
    noise_type="salt_pepper",
    noise_density=0.05,
    noise_mean=0.0,
    noise_sigma=0.01,
    mode="periodic",
):
    x_true = np.asarray(x_true, dtype=np.float64)
    if x_true.ndim != 2:
        raise ValueError("x_true must be a 2D grayscale image.")

    kernel = make_kernel(
        kind=kernel_kind,
        hsize=kernel_size,
        sigma=kernel_sigma,
        theta=motion_angle,
        length=motion_length,
    )

    mode_map = {
        "periodic": "wrap",
        "wrap": "wrap",
        "reflect": "reflect",
        "constant": "constant",
        "nearest": "nearest",
        "mirror": "mirror",
    }
    if mode not in mode_map:
        raise ValueError(
            "Unsupported mode. Use 'periodic', 'reflect', 'constant', "
            "'nearest', or 'mirror'."
        )

    x_blur = ndimage.convolve(x_true, kernel, mode=mode_map[mode])
    b = add_noise(
        x_blur,
        noise_type=noise_type,
        density=noise_density,
        mean=noise_mean,
        sigma=noise_sigma,
    )

    return kernel, np.clip(x_blur, 0.0, 1.0), b


def build_problem(
    kernel: np.ndarray,
    b: np.ndarray,
    gamma: float,
    fidelity: str = "l2",
):
    """
    Convenience: construct ``LinearOperators`` and ``DeblurProblem`` for ``b``'s shape.
    """
    from .operators import LinearOperators
    from .problem import DeblurProblem

    m, n = b.shape
    ops = LinearOperators(kernel, m, n)
    return DeblurProblem(ops=ops, b=b, gamma=gamma, fidelity=fidelity)
