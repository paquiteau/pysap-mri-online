"""
Some useful abstraction for the online reconstruction framework
"""
import numpy as np
from time import perf_counter
# Package import
from mri.optimizers.utils.cost import GenericCost

# Third party import
from modopt.opt.algorithms import Condat, POGM
from modopt.opt.proximity import IdentityProx

from mri.reconstructors.calibrationless import CalibrationlessReconstructor
from tqdm import tqdm

def ssos(I, axis=0):
    """
    Return the square root of the sum of square along axis
    Parameters
    ----------
    I: ndarray
    axis: int
    """
    return np.sqrt(np.sum(np.abs(I)**2,axis=axis))

class KspaceGenerator:
    def __init__(self,full_kspace, max_iteration=200):
        self.kspace = full_kspace
        self.max_iteration = max_iteration
        self.mask = np.ones(full_kspace.shape[1:])
        self.iteration = -1

    def __len__(self):
        return self.max_iteration

    def __iter__(self):
        return self

    def __getitem__(self,idx):
        return self.kspace, self.mask

    def __next__(self):
        if self.iteration < self.max_iteration:
            self.iteration += 1
            return self.__getitem__(self.iteration)
        else:
            raise StopIteration
    @property
    def shape(self):
        return self.kspace.shape

    @property
    def dtype(self):
        return self.kspace.dtype

class KspaceColumnGenerator(KspaceGenerator):
    def __init__(self,full_kspace, mask_cols=None, max_iteration=200):
        super().__init__(full_kspace,max_iteration=max_iteration)
        if mask_cols is not None:
            self.mask = np.zeros(full_kspace.shape[1:])
            self.mask[:,mask_cols]=1

class KspaceOnlineColumnGenerator(KspaceGenerator):
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
        super().__init__(full_kspace)
        self.full_kspace = full_kspace.copy()
        kspace_dim = full_kspace.shape[1:]
        self.full_mask = np.zeros(kspace_dim, dtype="int")
        self.full_mask[:, mask_cols] = 1
        self.mask = np.zeros(kspace_dim, dtype="int")
        self.kspace = np.zeros_like(full_kspace)
        self.max_iteration = max_iteration if max_iteration >= 0 else len(mask_cols)
        # reorder the column sampling by starting in the center
        # and alternating left/right expansion
        if from_center:
            center_pos = np.argmin(np.abs(mask_cols - kspace_dim[1] // 2))
            new_idx = np.zeros(mask_cols.shape)
            mask_cols = list(mask_cols)
            left = mask_cols[center_pos::-1]
            right = mask_cols[center_pos + 1:]
            new_cols = []
            while left or right:
                if left:
                    new_cols.append(left.pop(0))
                if right:
                    new_cols.append(right.pop(0))
            self.cols = np.array(new_cols)
        else:
            self.cols = mask_cols

    def __getitem__(self, idx):
        self.mask[:, self.cols[:idx]] = 1
        self.kspace.flags.writeable = True
        self.kspace = self.full_kspace * self.mask[np.newaxis, ...]
        self.kspace.flags.writeable = False
        return self.kspace, self.mask


class OnlineCalibrationlessReconstructor(CalibrationlessReconstructor):
    """
    A reconstructor with an online paradigm, the data of each step is
    """

    def reconstruct(
            self,
            kspace_generator,
            optimization_alg="condatvu",
            x_init=None,
            ref_image=None,
            metric_fun=None,
            metric_ref=None,
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
        ref_image: np.ndarray
            A ideal reconstruction case, use to compute the relative cost function
        metric_fun: list
            A list of function handle to compare the ssos of iteration with metric_ref
        metric_ref: np.ndarray
            An image , of which the ssos of the iteration is compared.
        """

        # arguments handling
        if x_init is None:
            x_init = np.zeros(kspace_generator.shape, dtype=kspace_generator.dtype)

        optimizer_type = "primal-dual" if optimization_alg == "condatvu" else "forward_backward"
        metrics = {"time":[]}
        if metric_fun:
            metrics |= {mf.__name__: [] for mf in metric_fun}
        cost_op = GenericCost(
            gradient_op=self.gradient_op,
            prox_op=self.prox_op,
            verbose=False,
            cost_interval=1,
            optimizer_type=optimizer_type,
        )
        opt_kwargs = dict(cost=cost_op,
                          grad=self.gradient_op,
                          linear=self.linear_op,
                          auto_iterate=False)
        if optimization_alg == "condatvu":
            # python 3.9+
            opt_kwargs |= dict(prox=IdentityProx(),
                               prox_dual=self.prox_op)
            opt = Condat(x=x_init,
                         y=np.zeros_like(self.linear_op.op(x_init)),
                         **opt_kwargs)
        elif optimization_alg == "pogm":
            opt_kwargs |= dict(prox=self.prox_op,
                               beta=self.gradient_op.inv_spec_rad)
            alpha_init = self.linear_op.op(x_init)
            opt = POGM(x=alpha_init,
                       y=alpha_init,
                       z=alpha_init,
                       u=alpha_init, **opt_kwargs)
        else:
            raise Exception(f"{optimization_alg}: Not Implemented yet")
        x_final = x_init.copy()
        pbar = tqdm(kspace_generator,desc=f"{opt.__class__.__name__}:{self.prox_op.__class__.__name__}:")
        for obs_kspace, mask in pbar:
            opt._grad.obs_data = obs_kspace
            opt._grad.fourier_op.kspace_mask = mask
            ts = perf_counter()
            opt._update()
            pbar.set_postfix({"cost": opt._cost_func.cost})
            tf = perf_counter()
            x_final = self.linear_op.adj_op(opt._x_new) if optimizer_type == "forward_backward" else opt._x_new
            for mf in metric_fun:
                metrics[mf.__name__].append(mf(ssos(x_final), metric_ref))
            metrics["time"].append(tf-ts)
        cost_finals = np.array(opt._cost_func._cost_list)
        for k, v in metrics.items():
            metrics[k] = np.array(v)

        metrics["cost"] = cost_finals
        if ref_image is not None:
            if optimizer_type == "forward_backward":
                cost_ref = opt._cost_func._calc_cost(self.linear_op.op(ref_image))
            else:
                cost_ref = opt._cost_func._calc_cost(ref_image, self.linear_op.op(ref_image))
            cost_rel = cost_finals - cost_ref
            metrics["cost_rel"] = cost_rel
        return x_final, metrics
