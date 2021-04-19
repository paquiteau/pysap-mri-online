import numpy as np
from modopt.math.metrics import psnr, ssim
import scipy.fftpack as pfft


def crop_center_square(img: np.ndarray):
    """ crop a rectangular 2D image into a square, centered an with minimal dimension"""
    short_side = min(img.shape) // 2
    center = np.array(img.shape) // 2
    return img[center[0] - short_side:center[0] + short_side, center[1] - short_side:center[1] + short_side]

def ssos(img):
    return crop_center_square(pfft.fftshift(np.sqrt(np.sum(img ** 2, axis=0))))


def psnr_ssos(test, ref):
    test = ssos(test)
    return psnr(test, ref)


def ssim_ssos(test, ref):
    test = ssos(test)
    return ssim(test, ref)
