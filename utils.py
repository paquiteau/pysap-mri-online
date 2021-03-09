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

class KspaceGenerator():
    def __init__(self,full_kspace, mask_loc):
        self.mask_gen = ColumnOnlineMask(mask_loc,full_kspace.shape[1:])
    def __iter__(self):
        return self
    def __next__(self):
        next_mask = self.mask_gen()
        next_kspace = full_kspace*next_mask[np.new_axis, ...]
        return next_kspace


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
