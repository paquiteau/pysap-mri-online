#!/usr/bin/env python3

from .base import KspaceGeneratorBase

import numpy as np
from tqdm import tqdm


class CumulativeNonCartesianGenerator(KspaceGeneratorBase):
    """A non cartesian generator, the shots are provided iteratively/"""

    def __init__(
        self, image_data, trajectories, shot_size, nufft_operator, max_iter=-1
    ):
        self._image_data = image_data
        self.trajectories = trajectories
        self.shot_size = shot_size
        self._nufft_operator = nufft_operator
        if max_iter == -1:
            self._len = np.ceil(len(trajectories) / self.shot_size)
        else:
            self._len == max_iter

    def __getitem__(self, it):
        if it >= self._len:
            raise IndexError
        sampled_points = self.trajectories[: it * self.shot_size + 1, :]
        self._nufft_operator.op()

    def __next__(self, idx):
        return

    def opt_iterate(self, opt, reset, estimate_call_period):
        x_new_list = []
        if reset:
            self.reset()
        opt._grad.fourier
        for (kspace, shot_sample) in tqdm(self):
            opt.idx += 1
            opt._grad.obs_data = kspace
            opt._grad.fourier_op.samples = shot_sample
            opt._update()
            if opt.metrics and opt.metric_call_period is not None:
                if opt.idx % opt.metric_call_period == 0 or opt.idx == (self._len - 1):
                    opt._compute_metrics()
            if estimate_call_period is not None:
                if opt.idx % estimate_call_period == 0 or opt.idx == (self._len - 1):
                    x_new_list.append(opt.get_notify_observers_kwargs()["x_new"])
        opt.retrieve_outputs()
        return x_new_list
