"""
Fourier Operator returning a single line of the k-space,
Using the JIT compiler of numba.
"""
import numpy as np
import scipy as sp

from mri.operators.base import OperatorBase
from numba import njit, objmode, prange, int64, complex128


class ColumnFFT(OperatorBase):
    """
    Fourier operator optimized to compute the 2D FFT + selection of various line of the kspace.
    The FFT will be normalized in a symmetric way
    Attributes
    ----------

    shape: tuple of int
        shape of the image (not necessarly a square matrix).
    n_coils: int, default 1
        Number of coils used to acquire the signal in case of multiarray
        receiver coils acquisition. If n_coils > 1, data shape must be
        [n_coils, Nx, Ny, NZ]
    n_jobs: int, default 1
        Number of parallel workers to use for fourier computation
    """

    def __init__(self, shape, n_coils=1, line_index=0, n_jobs=1):
        """Initilize the 'FFT' class.

        Parameters
        ----------
        shape: tuple of int
            shape of the image (not necessarly a square matrix).
        n_coils: int, default 1
            Number of coils used to acquire the signal in case of
            multiarray receiver coils acquisition. If n_coils > 1,
            data shape must be equal to [n_coils, Nx, Ny, NZ]
        line_index: int
            The index of the column onto the line_axis of the kspace
        n_jobs: int, default 1
            Number of parallel workers to use for fourier computation
            All cores are used if -1
        """
        self.shape = shape
        if n_coils <= 0:
            n_coils = 1
        self.n_coils = n_coils
        self.n_jobs = n_jobs
        self._line_index = line_index
        self._exp_f = complex(0, 0)
        self._exp_b = complex(0, 0)

    @property
    def line_index(self):
        return self._line_index

    @line_index.setter
    def line_index(self, val:int , shift=True):
        if shift:
            val = (self.shape[1] // 2 + val) % self.shape[1]
        if val >= self.shape[1]:
            raise IndexError("Index out of range")
        self._line_index = val
        cos = np.cos(2 * np.pi * val / self.shape[1])
        sin = np.sin(2 * np.pi * val / self.shape[1])
        self._exp_f = cos - 1j * sin
        self._exp_b = cos + 1j * sin

    @staticmethod
    @njit(complex128[:](complex128[:, :], complex128), parallel=True)
    def __fft15(img: np.array, exp_k: complex):
        shape = np.shape(img)
        column = np.zeros(shape[0], dtype=np.complex128)
        for j in prange(shape[0]):
            factor = 1 / np.sqrt(shape[1])
            for i in prange(shape[1]):
                column[j] += img[j, i] * factor
                factor *= exp_k
        with objmode(out="complex128[:]"):
            out = sp.fft.fft(column, norm="ortho")
        return out

    def _op(self, img):
        return sp.fft.ifftshift(self.__fft15(sp.fft.fftshift(img), self._exp_f))

    def op(self, img):
        """This method calculates the masked 2D Fourier transform of a 2d or 3D image.

        Parameters
        ----------
        img: np.ndarray
            input ND array with the same shape as the mask. For multichannel
            images the coils dimension is put first

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image. For multichannel
            images the coils dimension is put first
        """
        if self.n_coils == 1:
            return self._op(img)
        return np.apply_along_axis(self._op, 0, img)

    @staticmethod
    @njit(complex128[:,:](complex128[:], int64, complex128), parallel=True)
    def __ifft15(y: np.array, dim: int, exp_b: complex) -> np.array:
        img = np.zeros((dim, y.size), dtype=np.complex128)
        for j in prange(y.size):
            u = 1 / np.sqrt(y.size)
            for i in prange(dim):
                img[j, i] = y[j] * u
                u *= exp_b
        return img

    def _adj_op(self, x):
        return sp.fft.fftshift(self.__ifft15(
            sp.fft.ifft(sp.fft.ifftshift(x), norm="ortho"), self.shape[1], self._exp_b
        ))

    def adj_op(self, x):
        """This method calculates inverse masked Fourier transform of a ND
        image.

        Parameters
        ----------
        x: np.ndarray
            masked Fourier transform data. For multichannel
            images the coils dimension is put first

        Returns
        -------
        img: np.ndarray
            inverse ND discrete Fourier transform of the input coefficients.
            For multichannel images the coils dimension is put first
        """
        if self.n_coils == 1:
            return self._adj_op(x)
        return np.apply_along_axis(self._adj_op, 0, x)

    @staticmethod
    @njit(complex128[:,:](complex128[:,:], complex128, complex128), parallel=True)
    def __ifftfft15(img, exp_f, exp_b):
        shape = np.shape(img)
        img_out = np.zeros_like(img, dtype=np.complex128)
        y = 0.0j
        for j in prange(shape[0]):
            y = 0.0j
            u = 1 / np.sqrt(shape[1])
            for i in prange(shape[1]):
                y += img[j, i] * u
                u *= exp_f
            u = 1 / np.sqrt(shape[1])
            for i in prange(shape[1]):
                img_out[j, i] = y * u
                u *= exp_b
        return img_out

    def _auto_adj_op(self, img):
        return sp.fft.fftshift(
            self.__ifftfft15(sp.fft.fftshift(img), self._exp_f, self._exp_b)
        )

    def auto_adj_op(self, img):
        """ Equivalent to self.adj_op(self.op(img)), but faster """
        if self.n_coils == 1:
            return self._auto_adj_op(img)
        return np.apply_along_axis(self._auto_adj_op, 0, img)
