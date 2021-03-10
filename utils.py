"""
Some useful abstraction for the online reconstruction framework
"""
import numpy as np

import warnings

# Package import
from mri.operators.linear.wavelet import WaveletUD2, WaveletN
from mri.optimizers import pogm, condatvu, fista
from mri.optimizers.utils.cost import GenericCost

# Third party import
from modopt.opt.linear import Identity
from modopt.opt.algorithms import Condat, POGM, FISTA


from mri.reconstructors.calibrationless import CalibrationlessReconstructor


class KspaceOnlineColumnGenerator:
    """ A Mask generator, adding new sampling column at each iteration
    Parameters
    ----------
    mask_cols: the final mask to be sample, composed of columns
    k_space_dim: the 2D dimension of the k-space to be sample
    n_steps: the number of steps to be use if n_step = -1,
    from_center: if True, the column are added into the mask starting
    from the center and alternating left/right.
    """

    def __init__(self, full_kspace, mask_cols, max_iteration=-1, from_center=True):
        self.full_kspace = full_kspace
        kspace_dim = full_kspace.shape[1:]
        self.full_mask = np.zeros(kspace_dim, dtype='int')
        self.full_mask[:, mask_cols] = 1
        self.mask = np.zeros(kspace_dim, dtype='int')
        self.kspace = np.zeros_like(full_kspace)
        self.max_iteration = max_iteration if max_iteration >= 0 else len(mask_cols)
        self.iteration = -1
        # reorder the column sampling by starting in the center
        # and alternating left/right expansion
        if from_center:
            center_pos = np.argmin(np.abs(mask_cols - kspace_dim[1] // 2))
            new_idx = np.zeros(mask_cols.shape)
            mask_cols = list(mask_cols)
            left = mask_cols[center_pos::-1]
            right = mask_cols[center_pos+1:]
            new_cols=[]
            while left or right:
                if left:
                    new_cols.append(left.pop(0))
                if right:
                    new_cols.append(right.pop(0))
            self.cols = np.array(new_cols)
        else:
            self.cols = mask_cols

    @property
    def shape(self):
        return self.kspace.shape

    @property
    def dtype(self):
        return self.kspace.dtype

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        self.mask[:, self.cols] = 1
        self.kspace = self.kspace * self.mask[np.newaxis, ...]
        return self.kspace, self.mask

    def __next__(self):
        if self.iteration < self.max_iteration:
            self.iteration += 1
            return self.__getitem__(self.iteration)
        else:
            raise StopIteration


class OnlineCalibrationlessReconstructor(CalibrationlessReconstructor):
    """
    A reconstructor with an online paradigm, the data of each step is
    """

    def reconstruct(self, kspace_generator, optimization_alg='pogm',
                    x_init=None, num_iterations=100, cost_op_kwargs=None,
                    **kwargs):
        """ This method calculates operator transform.

        Parameters
        ----------
        kspace_data: np.ndarray
            the acquired value in the Fourier domain.
            this is y in above equation.
        optimization_alg: str (optional, default 'pogm')
            Type of optimization algorithm to use, 'pogm' | 'fista' |
            'condatvu'
        x_init: np.ndarray (optional, default None)
            input initial guess image for reconstruction. If None, the
            initialization will be zero
        num_iterations: int (optional, default 100)
            number of iterations of algorithm
        cost_op_kwargs: dict (optional, default None)
            specifies the extra keyword arguments for cost operations.
            please refer to modopt.opt.cost.costObj for details.
        kwargs: extra keyword arguments for modopt algorithm
            Please refer to corresponding ModOpt algorithm class for details.
            https://github.com/CEA-COSMIC/ModOpt/blob/master/\
            modopt/opt/algorithms.py
        """
        kspace_data = next(kspace_generator)
        self.gradient_op.obs_data = kspace_data
        available_algorithms = ["condatvu", "fista", "pogm"]
        if optimization_alg not in available_algorithms:
            raise ValueError("The optimization_alg must be one of " +
                             str(available_algorithms))
        optimizer = eval(optimization_alg)
        if optimization_alg == "condatvu":
            kwargs["dual_regularizer"] = self.prox_op
            optimizer_type = 'primal_dual'
        else:
            kwargs["prox_op"] = self.prox_op
            optimizer_type = 'forward_backward'
        if cost_op_kwargs is None:
            cost_op_kwargs = {}
        self.cost_op = GenericCost(
            gradient_op=self.gradient_op,
            prox_op=self.prox_op,
            verbose=self.verbose >= 20,
            optimizer_type=optimizer_type,
            **cost_op_kwargs,
        )
        opt = optimizer(
                gradient_op=self.gradient_op,
                linear_op=self.linear_op,
                cost_op=self.cost_op,
                max_nb_of_iter=num_iterations,
                x_init=x_init,
                verbose=self.verbose,
                auto_iterate=False) # custom iteration handling:
        for sampled_kspace in kspace_generator:
            opt._grad




        if optimization_alg == 'condatvu':
            self.metrics, self.y_final = metrics
        else:
            self.metrics = metrics[0]
        return self.x_final, self.costs, self.metrics
