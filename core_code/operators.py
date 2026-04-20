
"""
FFT-diagonal periodic convolution utilities and the stacked operator
``A = [K; D1; D2]`` from the original Colab implementation.
"""

import numpy as np
from numpy.fft import fft2, ifft2


def eig_vals_periodic_conv(kernel, num_rows, num_cols):
    """
    Compute eigenvalues of the periodic convolution operator via 2-D DFT.
    """
    padded = np.zeros((num_rows, num_cols))
    kh, kw = kernel.shape
    ch, cw = kh // 2, kw // 2
    for i in range(kh):
        for j in range(kw):
            ri = (i - ch) % num_rows
            ci = (j - cw) % num_cols
            padded[ri, ci] = kernel[i, j]
    return fft2(padded)


def apply_periodic_conv(x, eig_vals):
    """Kx = ifft2(eig_vals .* fft2(x))"""
    return np.real(ifft2(eig_vals * fft2(x)))


class LinearOperators:
    """
    FFT-diagonalised linear operators shared by all four algorithms.

    Provides: applyK, applyKT, applyD, applyDT, applyA, applyAT,
              precompute_inverse, solve_system
    """

    def __init__(self, kernel, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.eig_K = eig_vals_periodic_conv(kernel, num_rows, num_cols)
        self.eig_KT = np.conj(self.eig_K)

        d1_kernel = np.array([[-1], [1]])
        self.eig_D1 = eig_vals_periodic_conv(d1_kernel, num_rows, num_cols)
        self.eig_D1T = np.conj(self.eig_D1)

        d2_kernel = np.array([[-1, 1]])
        self.eig_D2 = eig_vals_periodic_conv(d2_kernel, num_rows, num_cols)
        self.eig_D2T = np.conj(self.eig_D2)

    def applyK(self, x):
        return apply_periodic_conv(x, self.eig_K)

    def applyKT(self, x):
        return apply_periodic_conv(x, self.eig_KT)

    def applyD1(self, x):
        return apply_periodic_conv(x, self.eig_D1)

    def applyD2(self, x):
        return apply_periodic_conv(x, self.eig_D2)

    def applyD(self, x):
        """Dx = [D1 x; D2 x], output shape (m, n, 2)"""
        return np.stack([self.applyD1(x), self.applyD2(x)], axis=-1)

    def applyDT(self, y):
        """D^T y  (divergence), y is (m, n, 2)"""
        return (
            apply_periodic_conv(y[:, :, 0], self.eig_D1T)
            + apply_periodic_conv(y[:, :, 1], self.eig_D2T)
        )

    def applyA(self, x):
        """Ax = (Kx, D1x, D2x)"""
        return (self.applyK(x), self.applyD1(x), self.applyD2(x))

    def applyAT(self, y1, y2, y3):
        """A^T [y1; y2; y3] = K^T y1 + D1^T y2 + D2^T y3"""
        return (
            self.applyKT(y1)
            + apply_periodic_conv(y2, self.eig_D1T)
            + apply_periodic_conv(y3, self.eig_D2T)
        )

    def precompute_inverse(self, t):
        """
        Eigenvalues of ``I + t^2 A^T A``.

        Primal DR and Primal-Dual DR use their algorithmic ``t`` here; ADMM
        intentionally calls this with ``t=1`` because its linear system is
        ``I + A^T A`` in the final notebook derivation.
        """
        return (
            np.ones((self.num_rows, self.num_cols))
            + t**2 * self.eig_KT * self.eig_K
            + t**2 * self.eig_D1T * self.eig_D1
            + t**2 * self.eig_D2T * self.eig_D2
        )

    def solve_system(self, rhs, eig_inv):
        """Apply the FFT-diagonal inverse defined by ``eig_inv`` to ``rhs``."""
        return np.real(ifft2(fft2(rhs) / eig_inv))
