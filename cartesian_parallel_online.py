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
from utils import KspaceGenerator, OnlineCalibrationlessReconstructor


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

class IterativeColumnMask():
    """ A generator that yield partial mask"""
    def __init__(self, final_mask, steps=0, from_center = True):
        self.mask = np.zeros_like(final_mask)
        cols = np.argwhere(final_mask[0,:] == 1);
        if from_center:
            center_pos =  np.argmin(cols - self.mask.shape[1]//2)
            left = cols[:center_pos]
            right = cols[center_pos:]
            new_idx = np.nan_like(cols)
            new_idx[-1:0:-2]= cols[:center_pos]
            new_idx[::2] = cols[center_pos:]
            self.cols = cols[new_idx]
        else:
            self.cols = cols
        self.final_mask = final_mask
        self.steps = steps if steps else len(cols)
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.n > self.steps:
            raise StopIteration
        else:
            return self.final_mask[:self.cols[:self.n]]

    def __getitem__(self,idx):
        return self.final_mask[:self.cols[:idx]]



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

    kspace_gen = KspaceGenerator(full_k, mask_loc)

    solver = OnlineCalibrationlessReconstructor(
        fourier,
        linear_op=linear_op,
        regularizer_op=prox_op,
        gradient_formulation="synthesis",
        n_jobs=1,
        verbose=0,
    )
    x_final, cost, metric = solver.reconstruct(kspace_gen)
