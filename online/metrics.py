import numpy as np
from modopt.math.metrics import psnr, ssim
import scipy.fftpack as pfft


def ssos(img):
    if len(img.shape) == 3:
        return np.sqrt(np.sum(img ** 2, axis=0))
    else:
        return np.sqrt(img ** 2)

def psnr_ssos(test, ref, mask=None):
    test = ssos(test)
    return psnr(test, ref,mask=None)


def ssim_ssos(test, ref, mask=None):
    test = ssos(test)
    return ssim(test, ref, mask=None)
