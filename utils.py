#!/usr/bin/env python3
import numpy as np
import os
import scipy.fftpack as pfft
import matplotlib.pyplot as plt


def load_data(data_dir, data_idx):
    # data is a list of 5-tuple:

    data = np.load(os.path.join(data_dir, "all_data_full.npy"), allow_pickle=True)[data_idx]
    for d in data:
        if isinstance(d, np.ndarray) and len(d.shape) > 1:
            print(d.shape, d.dtype)
        else:
            print(d)
    kspace_real, real_img, header_file, _ = data
    # the first element of data is in fact the image.
    kspace = pfft.fftshift(pfft.fft2(kspace_real, axes=[1, 2])).astype("complex128")
    mask_loc = np.load(os.path.join(data_dir, "cartesian_right_mask.npy"))
    mask = np.zeros(kspace.shape[1:], dtype="int")
    mask[:, mask_loc] = 1

    return kspace, real_img, mask_loc, mask


def implot(array, title=None, colorbar=None, axis=False):
    if np.iscomplexobj(array):
        array = np.abs(array)
    plt.figure()
    plt.imshow(array)
    if array.ndim == 3:
        for i in range(array.shape[0]):
            implot(array[i], title[i])

    if title:
        plt.title(title)
    if colorbar:
        plt.colorbar()
    if not axis:
        plt.axis("off")
    plt.show()
