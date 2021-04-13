import scipy.fftpack as pfft

from time import perf_counter
# Package import

from mri.operators import GradAnalysis, GradSynthesis
# Third party import
from modopt.opt.algorithms import Condat, POGM
from modopt.opt.proximity import IdentityProx

from mri.reconstructors.calibrationless import CalibrationlessReconstructor
from tqdm import tqdm

class OnlineCalibrationlessReconstructor(CalibrationlessReconstructor):
    """
    A reconstructor with an online paradigm, the data of each step is added/loaded incrementaly

    at each step, aim to solve:
    x_k = \\frac{S}{2k}|| F_k x - y_k ||_2^2 + \\lambda || \\Psi x_k ||
    """

    def reconstruct(
            self,
            kspace_generator,
            optimization_alg="condatvu",
            x_init=None,
            ref_image=None,
            metric_fun=None,
            metric_ref=None,
            **kwargs):
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
        opt_kwargs = dict(cost=None,
                          grad=self.gradient_op,
                          linear=self.linear_op,
                          auto_iterate=False)
        # Create new operator object to track cost residuals
        # DO NOT MESS WITH REFERENCES!
        fourier_op = copy.deepcopy(self.fourier_op)
        linear_op = copy.deepcopy(self.linear_op)
        prox_op = copy.deepcopy(self.prox_op)
        grad_op = GradAnalysis(fourier_op) if optimization_alg == "condatvu" else GradSynthesis(linear_op, fourier_op)
        grad_op._obs_data = kspace_generator.full_kspace.copy()
        metrics = {m.__name__: [lambda x, y: m(ssos(x), y), metric_ref] for m in metric_fun}
        if optimizer_type == "primal-dual":
            prox_on = lambda x: prox_op.cost(linear_op.op(x))
        else:
            prox_on = prox_op.cost
        rel_metrics = dict(
            grad_on=[ana_res2, 'primal', 'full_kspace', 'mask'],
            grad_off=[ana_res2, 'primal', 'full_kspace', 'full_mask'],
            prox_on=[prox_on, 'primal'],
            time=[lambda t: t, "time"],
        )

        metricsTracker = MetricsTracker(kspace_generator.max_iteration, metrics, rel_metrics)

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
            opt = POGM(x=alpha_init.copy(),
                       y=alpha_init.copy(),
                       z=alpha_init.copy(),
                       u=alpha_init.copy(),
                       **opt_kwargs)
        else:
            raise Exception(f"{optimization_alg}: Not Implemented yet")
        pbar = tqdm(kspace_generator, desc=f"{opt.__class__.__name__}:{self.prox_op.__class__.__name__}:")
        x_final = x_init.copy()
        for it, (obs_kspace, mask) in enumerate(pbar):
            #  pbar.set_postfix({"mask":np.linalg.norm(mask)})
            opt._grad._obs_data = obs_kspace
            opt._grad.fourier_op.mask = mask
            ts = perf_counter()
            opt._update()
            tf = perf_counter()
            if optimizer_type == "forward_backward":
                x_final = self.linear_op.adj_op(opt._x_new)
            else:
                x_final = opt._x_new.copy()
            metricsTracker.update(it, x_final,
                                  primal=x_final,
                                  kspace=obs_kspace,
                                  mask=mask,
                                  full_kspace=kspace_generator.full_kspace,
                                  full_mask=kspace_generator.full_mask,
                                  time=tf - ts)
        metrics = metricsTracker.results
        metrics["sum_on"] = metrics["grad_on"] + metrics["prox_on"]
        metrics["sum_off"] = metrics["grad_off"] + metrics["prox_on"]
        return x_final, metrics
