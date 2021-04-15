"""
Some useful abstraction for the online reconstruction framework
"""
import copy
from typing import Any, Tuple

import numpy as np
import scipy as sp

def ssos(img, axis=0):
    """
    Return the square root of the sum of square along axis
    Parameters
    ----------
    img: ndarray
    axis: int
    """
    return np.sqrt(np.sum(np.abs(img) ** 2, axis=axis))

class MetricsTracker:
    """A static tracker, to log metrics computed during algorithms
    """
    def __init__(self, max_iter, metrics, rel_metrics):
        self.max_iter = max_iter
        self.metrics = metrics
        self.rel_metrics = rel_metrics
        self.results = {metric_name: np.zeros(self.max_iter) for metric_name in metrics.keys()}
        self.results |= {cost: np.zeros(self.max_iter) for cost in rel_metrics.keys()}

    def update(self, it, x, **kwargs):
        if it >= self.max_iter:
            print("warning: expanding array")
            for array in self.results.values():
                array.resize(it)
        for m in self.metrics:
            self.results[m][it] = self.metrics[m][0](x, self.metrics[m][1:])
        for rm in self.rel_metrics:
            args = [kwargs[arg] for arg in self.rel_metrics[rm][1:]]
            self.results[rm][it] = self.rel_metrics[rm][0](*args)


def ana_res(primal, kspace, mask):
    return 0.5 * np.linalg.norm(mask[np.newaxis, ...] *
                                (kspace -
                                 sp.fft.ifftshift(sp.fft.fft2(sp.fft.fftshift(primal),
                                                              norm="ortho",
                                                              axes=[1, 2])))
                                ) ** 2


def ana_res2(primal, kspace, mask):
    return 0.5 * np.linalg.norm((kspace - mask[np.newaxis, ...] *
                                 sp.fft.ifftshift(sp.fft.fft2(sp.fft.fftshift(primal),
                                                              norm="ortho",
                                                              axes=[1, 2])))
                                ) ** 2

