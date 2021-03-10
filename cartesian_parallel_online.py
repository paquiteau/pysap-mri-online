"""
online reconstruction of paralell cartesian imaging
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as pfft

# Third party import
from modopt.opt.proximity import GroupLASSO, IdentityProx
from modopt.math.metrics import ssim, psnr, nrmse

from mri.operators import FFT, WaveletN, OWL
from utils import KspaceOnlineColumnGenerator, OnlineCalibrationlessReconstructor

DATA_DIR = "data/"
RESULT_DIR = "data/results/"


def load_data(data_idx):
    # data is a list of 5-tuple:

    data = np.load(os.path.join(DATA_DIR, "all_data_full.npy"), allow_pickle=True)[data_idx]
    for d in data:
        if isinstance(d, np.ndarray) and len(d.shape) > 1:
            print(d.shape, d.dtype)
        else:
            print(d)
    kspace_real, real_img, header_file, slice = data
    # the first element of data is in fact the image.
    kspace = pfft.fftshift(pfft.fft2(kspace_real,
                              axes=[1, 2])).astype("complex128")
    mask_loc = np.load(os.path.join(DATA_DIR, "cartesian_right_mask.npy"))
    mask = np.zeros(kspace.shape[1:], dtype="int")
    mask[:, mask_loc] = 1

    return kspace, real_img, mask_loc, mask


def ref_reconstruct(full_kspace):
    I_coils = pfft.ifft2(pfft.ifftshift(full_k),
                                        axes=[1, 2]).astype("complex128")
    # ssos of the coil separate reconstruction
    I_ssos = np.sqrt(np.sum(np.abs(I_coils) ** 2, axis=0))

    return I_coils, I_ssos


def online_reconstruct(kspace_data, mask_loc):
    """
    Perform the online reconstruction, simulating an iterative feeding
    of the optimisation algorithm.
    Solve:
    || FX - Y || + OWL(WX)
    """

    kspace_gen = KspaceOnlineColumnGenerator(kspace_data, mask_loc)
    K_DIM = kspace_gen.shape[1:]
    N_COILS = kspace_gen.shape[0]

    # Get the locations of the kspace samples
    fourier_op = FFT(
        mask=mask,
        shape=K_DIM,
        n_coils=N_COILS,
        n_jobs=1
    )

    linear_op = WaveletN("sym8", nb_scale=4, n_coils=16)
    # Wavelets coefficients for each coils
    coeff_shape = linear_op.op(np.zeros(kspace_gen.shape)).shape

    alpha = 1e-05
    beta = 1e-12
    prox_op = OWL(alpha=alpha,
                  beta=beta,
                  bands_shape=linear_op.coeffs_shape,
                  mode='band_based',
                  n_coils=N_COILS,
                  n_jobs=1)

    solver = OnlineCalibrationlessReconstructor(
        fourier_op,
        linear_op=linear_op,
        regularizer_op=prox_op,
        gradient_formulation="analysis",
        n_jobs=1,
        verbose=0,
    )
    x_final, cost, metric = solver.reconstruct(kspace_gen)
    return x_final, cost, metric


if __name__ == "__main__":
    full_k, real_img, mask_loc, mask = load_data(1)
    N_COILS = full_k.shape[0]
    K_DIM = full_k.shape[1:]
    R_DIM = real_img.shape
    # Reference reconstruction
    I_coils, I_ssos = ref_reconstruct(full_k)
    plt.figure()
    plt.imshow(np.log(np.abs(full_k[0])));
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.imshow(I_ssos);
    plt.colorbar()


    x_final, cost, _ = online_reconstruct(full_k, mask_loc)

    plt.show()
    plt.figure()
    a = np.sqrt(np.sum(np.abs(pfft.ifftshift(x_final)) ** 2, axis=0))
    plt.imshow(a);plt.colorbar()
    plt.show()
    plt.figure()
    plt.plot(cost)
