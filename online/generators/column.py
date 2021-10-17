"""
K-space Generator to emulate the acquisition of MRI.
Providing a fully sampled k-space and the procedure
"""
import numpy as np
from tqdm import tqdm
from .base import KspaceGeneratorBase


class Column2DKspaceGenerator(KspaceGeneratorBase):
    """k-space Generator, at each step a new column fills the existing k-space"""

    def __init__(self, full_kspace, mask_cols, max_iter=0):
        mask = np.zeros(full_kspace.shape[-2:])

        def flip2center(mask_cols, center_pos):
            """reorder a list by starting by a center_position and alternating left/right"""
            mask_cols = list(mask_cols)
            left = mask_cols[center_pos::-1]
            right = mask_cols[center_pos + 1 :]
            new_cols = []
            while left or right:
                if left:
                    new_cols.append(left.pop(0))
                if right:
                    new_cols.append(right.pop(0))
            return np.array(new_cols)

        self.cols = flip2center(
            mask_cols, np.argmin(np.abs(mask_cols - full_kspace.shape[-1] // 2))
        )
        if max_iter == 0:
            max_iter = len(self.cols)
        super().__init__(full_kspace, mask, max_iter=max_iter)

    def kspace_mask(self, idx):
        mask = np.zeros(self.shape[-2:])
        mask[:, self.cols[:idx]] = 1
        kspace = np.squeeze(self._full_kspace * mask[np.newaxis, ...])
        return kspace, mask

    def __getitem__(self, it):
        if it > self._len:
            raise IndexError
        idx = min(it, len(self.cols))
        return self.kspace_mask(idx)

    def __next__(self):
        if self.iter > self._len:
            raise StopIteration
        idx = min(self.iter + 1, len(self.cols))
        self.iter += 1
        return self.kspace_mask(idx)


class PartialColumn2DKspaceGenerator(Column2DKspaceGenerator):
    """k-space Generator yielding only the newly acquired line, to be used we classical FFT operator"""

    def __getitem__(self, it: int):
        if it >= self._len:
            raise IndexError
        idx = min(it, len(self.cols) - 1)
        return self.kspace_mask(idx)

    def __next__(self):
        if self.iter >= self._len:
            raise StopIteration
        idx = min(self.iter, len(self.cols) - 1)
        self.iter += 1
        return self.kspace_mask(idx)

    def kspace_mask(self, idx: int):
        mask = np.zeros(self.shape[-2:])
        mask[:, self.cols[idx]] = 1
        kspace = np.squeeze(self._full_kspace * mask[np.newaxis, ...])
        return kspace, mask


class DataOnlyKspaceGenerator(PartialColumn2DKspaceGenerator):
    """Kspace Generator to be used with a ColumnFFT Operator"""

    def kspace_mask(self, idx: int):
        col = self.cols[idx]
        kspace = self.kspace[..., col]
        return kspace, col
