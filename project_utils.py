#!/usr/bin/env python3
import numpy as np
import os
import scipy.fftpack as pfft
import matplotlib.pyplot as plt

from online.metrics import psnr_ssos, ssim_ssos
from online.operators.gradient import OnlineGradAnalysis, OnlineGradSynthesis
from mri.operators import FFT


def load_data(data_dir, data_idx, monocoil=False):
    # data is a list of 5-tuple:

    data = np.load(os.path.join(data_dir, "all_data_full.npy"), allow_pickle=True)[data_idx]

    kspace_real, base_real_img, header_file, _ = data
    if monocoil:
        kspace_real = np.sum(kspace_real**2, axis=0)
    kspace = pfft.ifftshift(pfft.fft2(pfft.fftshift(kspace_real, axes=[-1, -2]), axes=[-1, -2]), axes=[-1, -2]).astype("complex128")
    img_size = base_real_img.shape
    real_img_size = kspace.shape[-2:]
    mask_loc = np.load(os.path.join(data_dir, "mask_quarter.npy"))
    mask = np.zeros(real_img_size, dtype="int")
    mask[:, mask_loc] = 1
    s = np.std(base_real_img[0:20,0:20],axis=None)
    m =  np.mean(base_real_img[0:20,0:20])
    real_img = s * np.random.randn(*real_img_size)+m
    real_img[real_img_size[0] // 2 - img_size[0] // 2:real_img_size[0] // 2 + img_size[0] // 2,
             real_img_size[1] // 2 - img_size[1] // 2:real_img_size[1] // 2 + img_size[1] // 2] = base_real_img
    return kspace, real_img, mask_loc, mask


def implot(array, title=None, colorbar=None, mask=None, axis=False):
    if mask is not None:
        array = array[np.ix_(mask.any(1), mask.any(0))]
    if np.iscomplexobj(array):
        array = np.abs(array)
    fig = plt.figure()
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
    return fig
def imsave(array, filename):
    if np.iscomplexobj(array):
        array = np.abs(array)
    plt.imsave(filename,array)

def create_cartesian_metrics(online_pb, real_img, final_mask, final_k, estimates=None):

    metrics_fourier_op = FFT(shape=final_k.shape[-2:],
                             n_coils=final_k.shape[0] if final_k.ndim == 3 else 1,
                             mask=final_mask)
    metrics_gradient_op = OnlineGradAnalysis(fourier_op=metrics_fourier_op)
    metrics_gradient_op.obs_data = final_k
    square_mask= np.zeros(real_img.shape)
    real_img_size = real_img.shape
    img_size = [min(real_img.shape)]*2
    square_mask[real_img_size[0] // 2 - img_size[0] // 2:real_img_size[0] // 2 + img_size[0] // 2,
                real_img_size[1] // 2 - img_size[1] // 2:real_img_size[1] // 2 + img_size[1] // 2] = 1

    def data_res_on(x):
        if isinstance(online_pb.gradient_op, OnlineGradSynthesis):
            return online_pb.gradient_op.cost(online_pb.linear_op.op(x))
        return online_pb.gradient_op.cost(x)

    metrics = {'psnr': {'metric': psnr_ssos,
                        'mapping': {'x_new': 'test'},
                        'early_stopping': False,
                        'cst_kwargs': {'ref': real_img,
                                       'mask': square_mask},
                        },
               'ssim': {'metric': ssim_ssos,
                        'mapping': {'x_new': 'test'},
                        'cst_kwargs': {'ref': real_img,
                                       'mask': square_mask},
                        'early_stopping': False,
                        },
               'data_res_off': {'metric': lambda x: metrics_gradient_op.cost(x),
                                'mapping': {'x_new': 'x'},
                                'early_stopping': False,
                                'cst_kwargs': dict(),
                                },
               'data_res_on': {'metric': data_res_on,
                               'mapping': {'x_new': 'x'},
                               'early_stopping': False,
                               'cst_kwargs': dict(),
                               },
               'reg_res': {'metric': lambda x: online_pb.prox_op.cost(online_pb.linear_op.op(x)),
                           'mapping': {'x_new': 'x'},
                           'early_stopping': False,
                           'cst_kwargs': dict(),
                           }
               }
    metrics_config = {'metrics': metrics,
                      'cost_op_kwargs': {"cost_interval": 1},
                      'metric_call_period': 1,
                      'estimate_call_period': estimates,
                      }
    return metrics_config
