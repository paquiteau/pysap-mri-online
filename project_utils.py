#!/usr/bin/env python3
import numpy as np
import os
import scipy as sp
import matplotlib.pyplot as plt

from online.metrics import psnr_ssos, ssim_ssos, ssos
from online.operators.gradient import OnlineGradAnalysis, OnlineGradSynthesis
from mri.operators import FFT


def load_data(data_dir, data_idx, monocoil=False, use_ref_kspace=True, squared=False,fourier=True):
    # data is a list of 5-tuple:

    data = np.load(os.path.join(data_dir, "all_data_full.npy"), allow_pickle=True)[data_idx]

    kspace_real, base_real_img, header_file, _ = data

    img_size = base_real_img.shape
    real_img_size = kspace_real.shape[-2:]
    mask_loc = np.load(os.path.join(data_dir, "mask_quarter.npy"))
    if squared:
        kspace_real = kspace_real[:, real_img_size[0] // 2 - img_size[0] // 2: real_img_size[0] // 2 + img_size[0] // 2,
                                     real_img_size[1] // 2 - img_size[1] // 2: real_img_size[1] // 2 + img_size[1] // 2]
        real_img = base_real_img
    else:
        s = np.std(base_real_img[0:20, 0:20], axis=None)
        m = np.mean(base_real_img[0:20, 0:20])
        real_img = s * np.random.randn(*real_img_size)+m
        real_img[real_img_size[0] // 2 - img_size[0] // 2:real_img_size[0] // 2 + img_size[0] // 2,
                 real_img_size[1] // 2 - img_size[1] // 2:real_img_size[1] // 2 + img_size[1] // 2] = base_real_img
    mask = np.zeros(real_img.shape, dtype="int")
    mask[:, mask_loc] = 1
    if fourier:
        if monocoil:
            kspace_real = np.sum(np.square(kspace_real), axis=0)
        if use_ref_kspace and monocoil:
            kspace = sp.fft.ifftshift(sp.fft.fft2(sp.fft.fftshift(real_img), norm="ortho"))
        else:
            kspace = sp.fft.ifftshift(sp.fft.fft2(sp.fft.fftshift(kspace_real, axes=[-1, -2]), norm="ortho", axes=[-1, -2]), axes=[-1, -2]).astype("complex128")
    else:
        if monocoil:
            kspace = np.sum(np.square(kspace_real), axis=0)
        else:
            kspace = kspace_real
    return kspace, real_img, mask_loc, mask


def implot(array, title=None, colorbar=None, mask=None, axis=False):
    if mask is not None:
        array = array[np.ix_(mask.any(1), mask.any(0))]
    if np.iscomplexobj(array):
        array = np.abs(array)
    fig = plt.figure()
    plt.imshow(array)
    if title:
        plt.title(title)
    if colorbar:
        plt.colorbar()
    if not axis:
        plt.axis("off")
    return fig


def imsave(array, filename,mask=None):
    if mask is not None:
        array = array[np.ix_(mask.any(1), mask.any(0))]
    if np.iscomplexobj(array):
        array = np.abs(array)
    plt.imsave(filename,array)


def create_cartesian_metrics(online_pb, real_img, final_mask, final_k, estimates=None):

    metrics_fourier_op = FFT(shape=final_k.shape[-2:],
                             n_coils=final_k.shape[0] if final_k.ndim == 3 else 1,
                             mask=final_mask)
    metrics_gradient_op = OnlineGradAnalysis(fourier_op=metrics_fourier_op)
    metrics_gradient_op.obs_data = final_k
    square_mask = np.zeros(real_img.shape)
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
               'x_new': {'metric': lambda x: ssos(x),
                         'mapping': {'x_new': 'x'},
                         'early_stopping': False,
                         'cst_kwargs': dict(),
                         },
               }
    if online_pb.opt == 'condatvu':
        metrics['reg_res'] = {'metric': lambda y: online_pb.prox_op.cost(y),
                              'mapping': {'y_new': 'y'},
                              'early_stopping': False,
                              'cst_kwargs': dict(),
                              }
    else:
        metrics['reg_res'] = {'metric': lambda x: online_pb.prox_op.cost(online_pb.linear_op.op(x)),
                              'mapping': {'x_new': 'x'},
                              'early_stopping': False,
                              'cst_kwargs': dict(),
                              }
    metrics_config = {'metrics': metrics,
                      'cost_op_kwargs': {"cost_interval": 1},
                      'metric_call_period': 1,
                      'estimate_call_period': estimates,
                      }
    return metrics_config


def plot_metric(results, name, *args, log=False, ax=None,**kwargs):
    if ax is None:
        ax = plt.gca()
    if log:
        ax.semilogy(results['metrics'][name]['index'], results['metrics'][name]['values'],*args,**kwargs,label=name)
    else:
        ax.plot(results['metrics'][name]['index'], results['metrics'][name]['values'],*args, **kwargs,label=name)
