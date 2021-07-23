import numpy as np
from modopt.math.metrics import psnr, ssim, mse
import scipy.fftpack as pfft


def ssos(img):
    if len(img.shape) == 3:
        return np.sqrt(np.sum(abs(img) ** 2, axis=0))
    else:
        return abs(img)


def psnr_ssos(test, ref, mask=None):
    test = ssos(test)
    if mask is not None:
        mse = np.mean(np.square(mask*(test-ref)))
        return 10*np.log10(np.max(ref*mask)**2/mse)
    else:
        return 10*np.log10(np.max(ref)**2/np.mean(np.square(test-ref)))


def ssim_ssos(test, ref, mask=None):
    test = ssos(test)
    return ssim(test, ref, mask=mask)


def mse_ssos(test, ref, mask=None):
    test = ssos(test)
    if mask is not None:
        return np.mean(np.square(mask*(test-ref)))
    else:
        return np.mean(np.square(test-ref))
