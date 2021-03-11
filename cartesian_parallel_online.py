"""
online reconstruction of paralell cartesian imaging
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as pfft

# Third party import
from modopt.opt.proximity import GroupLASSO
from modopt.math.metrics import ssim, psnr

from mri.operators import FFT, WaveletN, OWL
from utils import ssos, KspaceGenerator, KspaceColumnGenerator, KspaceOnlineColumnGenerator, \
    OnlineCalibrationlessReconstructor

DATA_DIR = "data/"
RESULT_DIR = "data/results/"
N_JOBS = 1

def load_data(data_idx):
    # data is a list of 5-tuple:

    data = np.load(os.path.join(DATA_DIR, "all_data_full.npy"), allow_pickle=True)[data_idx]
    for d in data:
        if isinstance(d, np.ndarray) and len(d.shape) > 1:
            print(d.shape, d.dtype)
        else:
            print(d)
    kspace_real, real_img, header_file, _ = data
    # the first element of data is in fact the image.
    kspace = pfft.fftshift(pfft.fft2(kspace_real,
                                     axes=[1, 2])).astype("complex128")
    mask_loc = np.load(os.path.join(DATA_DIR, "cartesian_right_mask.npy"))
    mask = np.zeros(kspace.shape[1:], dtype="int")
    mask[:, mask_loc] = 1

    return kspace, real_img, mask_loc, mask


def ref_reconstruct(full_k):
    I_coils = pfft.ifft2(pfft.ifftshift(full_k),
                         axes=[1, 2]).astype("complex128")
    # ssos of the coil separate reconstruction
    I_ssos = ssos(I_coils)

    return I_coils, I_ssos


def online_reconstruct(kspace_gen,
                       prox="OWL",
                       **rec_kwargs):
    """
    Perform the online reconstruction, simulating an iterative feeding
    of the optimisation algorithm.
    Solve:
    || FX - Y || + OWL(WX)
    """
    K_DIM = kspace_gen.shape[1:]
    N_COILS = kspace_gen.shape[0]

    # Get the locations of the kspace samples
    fourier_op = FFT(
        mask=mask,
        shape=K_DIM,
        n_coils=N_COILS,
        n_jobs=N_JOBS
    )

    linear_op = WaveletN("sym8", nb_scale=4, n_coils=16, n_jobs=1)
    # Wavelets coefficients for each coils
    # initialisation of wavelet transform
    linear_op.op(np.zeros(kspace_gen.shape))
    alpha = 1e-05
    beta = 1e-12
    if prox == "OWL":
        prox_op = OWL(alpha=alpha,
                      beta=beta,
                      bands_shape=linear_op.coeffs_shape,
                      mode='band_based',
                      n_coils=N_COILS,
                      n_jobs=N_JOBS)
    elif prox == "GroupLASSO":
        prox_op = GroupLASSO(weights=1e-5)
    else:
        raise Exception(f"prox:{prox} Not implemented")

    solver = OnlineCalibrationlessReconstructor(
        fourier_op,
        linear_op=linear_op,
        regularizer_op=prox_op,
        gradient_formulation="analysis" if rec_kwargs["optimization_alg"] == "condatvu" else "synthesis",
        n_jobs=N_JOBS,
        verbose=0,
    )
    return solver.reconstruct(kspace_gen, **rec_kwargs)


def implot(array, title=None, colorbar=None):
    if np.iscomplexobj(array):
        array = np.log(np.abs(array))
    plt.figure()
    plt.imshow(array)
    if array.ndim == 3:
        for i in range(array.shape[0]):
            implot(array[i], title[i])

    if title:
        plt.title(title)
    if colorbar:
        plt.colorbar()
    plt.show()


if __name__ == "__main__":
    full_k, real_img, mask_loc, mask = load_data(1)
    # Reference reconstruction
    I_coils, I_ssos = ref_reconstruct(full_k)

    implot(I_ssos, title="Reference image")
    # simulate an online reconstruction
    algo_dict = {
        "condatvu": [{"prox": "OWL"},
                     {"prox": "GroupLASSO"},
                     {"prox": "GroupLASSO"},
                     ],
        "pogm": [{"prox": "OWL"},
                 {"prox": "GroupLASSO"}],
        # "fista": [{"prox": "OWL"},
        #           {"prox": "GroupLASSO"}],
    }
    output = dict()
    # algo_dict.pop("condatvu")
    for algo_name, params in algo_dict.items():
        output[algo_name] = dict()
        for p in params:
            kspace_gen = KspaceOnlineColumnGenerator(full_k, mask_loc)
            x_final, metrics = online_reconstruct(kspace_gen,
                                                  optimization_alg=algo_name,
                                                  ref_image=I_coils,
                                                  metric_ref=I_ssos,
                                                  metric_fun=[ssim, psnr],
                                                  **p)
            output[algo_name]["_".join(p.values())] = [x_final, metrics]
    np.save("data/results-online2.npy", output)
    print("OFFLINE RECONSTRUCTION")
    for algo_name, params in algo_dict.items():
        output[algo_name] = dict()
        for p in params:
            kspace_gen = KspaceColumnGenerator(full_k, mask_loc, max_iteration=80)
            x_final, metrics = online_reconstruct(kspace_gen,
                                                  optimization_alg=algo_name,
                                                  ref_image=I_coils,
                                                  metric_ref=I_ssos,
                                                  metric_fun=[ssim, psnr],
                                                  **p)
            output[algo_name]["_".join(p.values())] = [x_final, metrics]
    np.save("data/results-offline-mask2.npy", output)
