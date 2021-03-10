"""
online reconstruction of paralell cartesian imaging
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

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
    kspace, real_img, header_file, slice = data
    mask_loc = np.load(os.path.join(DATA_DIR, "cartesian_right_mask.npy"))
    mask = np.zeros(kspace.shape[1:],dtype="int")
    mask[:,mask_loc] = 1

    return kspace, real_img, mask_loc, mask



if __name__ == "__main__":
    full_k, real_img, mask_loc, mask = load_data(1)

    N_COILS = full_k.shape[0]
    K_DIM = full_k.shape[1:]
    R_DIM = real_img.shape

    prox_op = IdentityProx()
    # Get the locations of the kspace samples
    linear_op = WaveletN("sym8", nb_scale=4, n_coils=16)
    fourier = FFT(
        mask=mask,
        shape=K_DIM,
        n_coils=N_COILS,
    )

    kspace_gen = KspaceOnlineColumnGenerator(full_k, mask_loc)

    solver = OnlineCalibrationlessReconstructor(
        fourier,
        linear_op=linear_op,
        regularizer_op=prox_op,
        gradient_formulation="synthesis",
        n_jobs=1,
        verbose=0,
    )
    x_final, cost, metric = solver.reconstruct(kspace_gen)
