
import numpy as np
from tqdm import tqdm


class KspaceGenerator:
    """
    Basic K-space Generator emulate the acquisition of an MRI.

    Parameters
    ----------
    full_kspace: np.ndarray
        The fully sampled kspace, which will be returned incrementally, and a mask, use for the Fourier transform
    mask: np.ndarray
        A binary mask, giving the sampled location for the kspace
    """

    def __init__(self, full_kspace: np.ndarray, mask: np.ndarray, max_iter: int = 1):
        self._full_kspace = full_kspace
        self.kspace = full_kspace.copy()
        self.mask = mask
        self._len = max_iter
        self.iter = 0

    @property
    def shape(self):
        return self._full_kspace.shape

    @property
    def dtype(self):
        return self._full_kspace.dtype

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        if idx >= self._len:
            raise IndexError
        return self._full_kspace, self.mask

    def __next__(self):
        if self.iter < self._len:
            self.iter += 1
            return self._full_kspace, self.mask
        raise StopIteration

    def reset(self):
        self.iter = 0

    def opt_iterate(self, opt, reset=True):
        if reset:
            self.reset()
        for (kspace, mask) in tqdm(self):
            opt.idx += 1
            opt._grad.obs_data = kspace
            opt._grad.fourier_op.mask = mask
            opt._update()
            if opt.metrics and opt.metric_call_period is not None:
                if opt.idx % opt.metric_call_period == 0 or opt.idx == (self._len - 1):
                    opt._compute_metrics()
        opt.retrieve_outputs()

