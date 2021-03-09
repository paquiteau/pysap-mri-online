"""
Some useful abstraction for the online reconstruction framework
"""
from mri.reconstructors.calibrationless import CalibrationlessReconstructor


class OnlineMaskBase():
    """ A Base generator that yield a portion of a mask"""

    def __init__(self, mask, n_steps):
        self.mask = mask
        self.n_steps = n_steps
        self.n = 0
    def __iter__(self):
        return self

    def __next__(self):
        if self.n > self.n_steps:
            raise StopIteration
        else:
            next_mask = self.get_next_mask()
            self.n += 1
            return next_mask

    def __getitem__(self, idx):
        return self.get_mask_by_idx(idx)

    def get_next_mask(self):
        """ Implement a update of the mask
        by default return the initial mask"""
        return self.mask

    def get_mask_by_idx(self, idx):
        return self.mask


class ColumnOnlineMask(OnlineMaskBase):
    """ A Mask generator, adding new sampling column at each iteration
    Parameters
    ----------
    mask_cols: the final mask to be sample, composed of columns
    k_space_dim: the 2D dimension of the k-space to be sample
    n_steps: the number of steps to be use if n_step = -1,
    from_center: if True, the column are added into the mask starting
    from the center and alternating left/right.
    """
    def __init__(self, mask_cols, kspace_dim, n_steps=-1, from_center=True):
        mask = np.zeros(kspace_dim,dtype='int')
        mask[:, mask_cols] = 1
        n_steps = n_steps if n_steps >= 0 else len(mask_cols)
        super().__init__(mask, n_steps)
        if from_center:
            center_pos = np.argmin(mask_cols-kspace_dim[1]//2)
            new_idx = np.nan(mask_cols.shape)
            new_idx[-1:0:-2] = mask_cols[:center_pos]
            new_idx[::2] = mask_cols[center_pos:]
            self.cols = mask_cols[new_idx]
        else:
            self.cols = mask_cols

    def get_next_mask(self):
        return self.get_mask_by_idx(self.n)

    def get_mask_by_idx(self,idx):
        return self.mask[:,self.cols[:idx]]


class OnlineCalibrationlessReconstructor(CalibrationlessReconstructor):
    """
    A reconstructor with an online paradigm, the data of each step is
    """
    pass
