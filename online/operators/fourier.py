"""
Fourier Operator returning a single line of the k-space,
Using the JIT compiler of numba.
"""
import numpy as np
import scipy as sp

from mri.operators.base import OperatorBase

from numba import njit, guvectorize, prange, complex128


def numba_dft(row, exp_k, out):
    out[:] = 0.0j
    for i in range(len(row)):
        out += row[i] * exp_k[i]


def numba_idft(y, exp_b, out):
    for i in range(len(exp_b)):
        out[i] = y * exp_b[i]


def numba_njit_dft_single(img, exp_k):
    shape = np.shape(img)
    column = np.zeros(shape[0], dtype=np.complex128)
    for j in prange(shape[0]):
        for i in prange(shape[1]):
            column[j] += img[j, i] * exp_k[i]
    return column


def numba_njit_dft_multi(img, exp_k):
    shape = np.shape(img)
    columns = np.zeros((shape[0], shape[1]), dtype=np.complex128)
    for l in prange(shape[0]):
        for j in prange(shape[1]):
            for i in prange(shape[2]):
                columns[l, j] += img[l, j, i] * exp_k[i]
    return columns


def numba_njit_idft_single(y: np.array, exp_b) -> np.array:
    img = np.zeros((y.size, exp_b.size), dtype=np.complex128)
    for j in prange(y.size):
        for i in prange(exp_b.size):
            img[j, i] = y[j] * exp_b[i]
    return img


def numba_njit_idft_multi(y: np.array, exp_b) -> np.array:
    img = np.zeros((*y.shape, exp_b.size), dtype=np.complex128)
    for l in prange(y.shape[0]):
        for j in prange(y.shape[1]):
            for i in prange(exp_b.size):
                img[l, j, i] = y[l, j] * exp_b[i]
    return img


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

    def __init__(self, shape, line_index=0, n_coils=1, platform="numpy"):
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
        package: str
            The plateform on which to run the computation. can be either 'numpy', 'numba', 'cupy'
        """
        self.shape = shape
        if n_coils <= 0:
            n_coils = 1
        self.n_coils = n_coils
        self._exp_f = np.zeros(shape[1], dtype=complex)
        self._exp_b = np.zeros(shape[1], dtype=complex)
        self._mask = line_index
        if platform == "numba" and n_coils > 1:
            self._dft = njit(
                complex128[:, :](complex128[:, :, :], complex128[:]), parallel=True
            )(numba_njit_dft_multi)
            self._idft = njit(
                complex128[:, :, :](complex128[:, :], complex128[:]), parallel=True
            )(numba_njit_idft_multi)
        elif platform == "numba":
            self._dft = njit(
                complex128[:](complex128[:, :], complex128[:]), parallel=True
            )(numba_njit_dft_single)
            self._idft = njit(
                complex128[:, :](complex128[:], complex128[:]), parallel=True
            )(numba_njit_idft_single)
        elif platform == "gufunc":
            self._dft = guvectorize(
                ["complex128[:], complex128[:], complex128[:]"],
                "(n),(n)->()",
                nopython=True,
                target="cpu",
            )(numba_dft)
            self._idft = guvectorize(
                ["complex128, complex128[:], complex128[:]"],
                "(),(n)->(n)",
                target="parallel",
                nopython=True,
            )(numba_idft)
        elif platform == "numpy":
            self._dft = np.dot
            self._idft = np.multiply.outer
        else:
            raise NotImplementedError(f"platform '{platform}' is not supported")

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, val: int, shift=True):
        if shift:
            val = (self.shape[1] // 2 + val) % self.shape[1]
        if val >= self.shape[1]:
            raise IndexError("Index out of range")
        self._mask = val
        cos = np.cos(2 * np.pi * val / self.shape[1])
        sin = np.sin(2 * np.pi * val / self.shape[1])
        exp_f = cos - 1j * sin
        exp_b = cos + 1j * sin
        self._exp_f = (1 / np.sqrt(self.shape[1])) * exp_f ** np.arange(self.shape[1])
        self._exp_b = (1 / np.sqrt(self.shape[1])) * exp_b ** np.arange(self.shape[1])

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
        # if self.n_coils == 1:
        #     return self._op(img)
        # return np.array(self._op(img_slice) for img_slice in img)
        return sp.fft.ifftshift(
            sp.fft.fft(
                self._dft(sp.fft.fftshift(img, axes=[-1, -2]), self._exp_f),
                axis=-1,
                norm="ortho",
            ),
            axes=[-1],
        )

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
        # if self.n_coils == 1:
        #     return self._adj_op(x)
        # return np.array(self._op(x_slice) for x_slice in x)

        return sp.fft.fftshift(
            self._idft(
                sp.fft.ifft(sp.fft.ifftshift(x, axes=[-1]), axis=-1, norm="ortho"),
                self._exp_b,
            ),
            axes=[-1, -2],
        )


def nc_dft(freqs_f, image):
    return np.sum(np.multiply.outer(freqs_f, image), axis=(-1, -2))


def nc_idft(freqs_b, points, shape):
    return np.sum(np.multiply.outer(freqs_b, points), axis=0)


class NonCartesianDFT(OperatorBase):
    """Non Cartesian DFT transform."""

    def __init__(self, points, shape, n_coils=1, platform="numpy"):
        self.shape = shape
        if n_coils <= 0:
            n_coils = 1
        self.n_coils = n_coils
        self._exp_f = np.zeros(shape[1], dtype=complex)
        self._exp_b = np.zeros(shape[1], dtype=complex)
        self._points = points
        if platform == "numpy":
            self._dft = nc_dft
            self._idft = nc_idft

    @property
    def sampled_points(self):
        return self._sampled_points

    @sampled_points.setter
    def sampled_points(self, points):
        self._points = points
        self.freqs_f = None
        self.freqs_b = None # memory expensive to store. -> one image per point in a shot.
            # other solution, compute on the fly from the atoms of freqs_f.
    def op(self, img):
        return self._dft(self.freqs_f, img)

    def adj_op(self, points):
        return self._idft(self.freqs_b, img)
