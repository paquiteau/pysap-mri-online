"""
Some useful abstraction for the online reconstruction framework
"""
import numpy as np


# Package import
from mri.optimizers.utils.cost import GenericCost

# Third party import
from modopt.opt.linear import Identity
from modopt.opt.algorithms import Condat, POGM, FISTA
from modopt.opt.proximity import IdentityProx

from mri.reconstructors.calibrationless import CalibrationlessReconstructor


class KspaceOnlineColumnGenerator:
    """A Mask generator, adding new sampling column at each iteration
    Parameters
    ----------
    full_kspace: the fully sampled kspace for every coils
    mask_cols: the final mask to be sample, composed of columns
     the 2D dimension of the k-space to be sample
    max_iteration: the number of steps to be use if n_step = -1,
    from_center: if True, the column are added into the mask starting
    from the center and alternating left/right.
    """

    def __init__(self, full_kspace, mask_cols, max_iteration=-1, from_center=True):
        self.full_kspace = full_kspace
        kspace_dim = full_kspace.shape[1:]
        self.full_mask = np.zeros(kspace_dim, dtype="int")
        self.full_mask[:, mask_cols] = 1
        self.mask = np.zeros(kspace_dim, dtype="int")
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
            right = mask_cols[center_pos + 1 :]
            new_cols = []
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

    def reconstruct(
        self,
        kspace_generator,
        optimization_alg="condatvu",
        x_init=None,
        num_iterations=100,
        cost_op_kwargs=None,
        **kwargs
    ):
        """ This method calculates operator transform.

        Parameters
        ----------
        kspace_generator: class instance
            Provides the kspace for each iteration of the algorithm
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
        # kspace_data,mask = next(kspace_generator)
        #        self.gradient_op.obs_data = kspace_data
        if x_init is None:
            x_init = np.zeros(kspace_generator.shape,dtype=kspace_generator.dtype)

        if optimization_alg == "condatvu":
            kwargs["dual_regularizer"] = IdentityProx()
            optimizer_type = "primal_dual"
        else:
            kwargs["prox_op"] = self.prox_op
            optimizer_type = "forward_backward"
        if cost_op_kwargs is None:
            cost_op_kwargs = {}

        self.cost_op = GenericCost(
            gradient_op=self.gradient_op,
            prox_op=self.prox_op,
            verbose=True,
            optimizer_type=optimizer_type,
            **cost_op_kwargs,
        )
        opt_kwargs = {
            "cost": self.cost_op,
            "grad": self.gradient_op,
            "prox_dual": self.prox_op,
            "linear": self.linear_op
        }
        coeffs_temp = self.linear_op.op(np.zeros(kspace_generator.shape))
        if optimization_alg == "condatvu":
            #python 3.9+
            opt_kwargs |= dict(
                               prox=IdentityProx(),
                               rho=1,
                               sigma=0.5,
                               eps=5e-8,
                               tau=1.0 / (self.gradient_op.spec_rad / 2 + 0.5 + 5e-8),
                               rho_update=None,
                               sigma_update=None,
                               tau_update=None)

            opt = Condat(x=x_init,
                         y=np.zeros_like(self.linear_op.op(x_init)),
                         **opt_kwargs)
        elif optimization_alg == "pogm":
            opt = POGM(x=x_init, **opt_kwargs)
        elif optimization_alg == "fista":
            opt = FISTA(x=x_init, **opt_kwargs)

        else:
            raise Exception(f"{optimization_alg}: Not Implemented yet")


        cost_func.batch
        for obs_kspace, mask in kspace_generator:
            opt._grad.obs_data = obs_kspace
            opt._grad.fourier_op.kspace_mask = mask
            opt._update()
        cost_finals= opt._cost_func._cost_list
        x_final = opt._x_new
        y_final = opt._y_new
        return x_final, y_final, cost_finals
