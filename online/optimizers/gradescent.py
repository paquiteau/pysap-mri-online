import time
import numpy as np

# Third party import
from modopt.opt.algorithms import SetUp
from modopt.opt.cost import costObj
from sklearn.metrics._plot.tests.test_plot_curve_common import test_plot_curve_estimator_name_multiple_calls

from .base import online_algorithm

class GenericGradOpt(SetUp):
    """ Generic Gradient descent operator
    x_{k+1} = x_k - \frac{\eta}{\sqrt{s_k + \epsilon}} m_k
    """

    def __init__(self, x, grad, prox, linear, cost, eta=1, eta_update=None, epsilon=1e-6, metric_call_period=5,
                 metrics=None, reg_factor=1.0, **kwargs):
        # Set the initial variable values
        if metrics is None:
            metrics = dict()
        # Set default algorithm properties
        super().__init__(
            metric_call_period=metric_call_period,
            metrics=metrics, **kwargs)
        self.iter = 0
        self._check_input_data(x)
        self._x_old = np.copy(x)
        self._x_new = np.copy(x)
        self._speed_grad = np.zeros(x.shape, dtype=float)
        self._corr_grad = np.zeros_like(x)
        self.reg_factor = reg_factor
        # Set the algorithm operators
        (self._check_operator(operator) for operator in (grad, prox, cost))
        self._grad = grad
        self._prox = prox
        self._linear = linear
        if cost == 'auto':
            self._cost_func = costObj([self._grad, self._prox])
        else:
            self._cost_func = cost
        # Set the algorithm parameters
        (self._check_param(param) for param in (eta, epsilon))
        self._eta = eta
        self._eps = epsilon

        # Set the algorithm parameter update methods
        self._check_param_update(eta_update)
        self._eta_update = eta_update
        self.idx = 0

    def _update(self):
        self._grad.get_grad(self._x_old)
        self.update_grad_dir(self._grad.grad)
        self.update_grad_speed(self._grad.grad)
        Gamma = (self._eta / np.sqrt(self._speed_grad + self._eps))*self.reg_factor
        self._x_new = self._x_old - Gamma * self._corr_grad

        self.update_reg(Gamma)
        self._x_old = self._x_new.copy()

        # Test cost function for convergence.
        if self._cost_func:
            self.converge = self.any_convergence_flag() or \
                            self._cost_func.get_cost(self._x_new)

    def update_grad_dir(self, grad):
        self._corr_grad = grad

    def update_grad_speed(self, grad):
        pass

    def update_reg(self, factor):
        self._x_new = self._prox.op(self._x_new, extra_factor=factor)

    def get_notify_observers_kwargs(self):
        """Notify observers

        Return the mapping between the metrics call and the iterated
        variables.

        Returns
        -------
        notify_observers_kwargs : dict,
           The mapping between the iterated variables

        """
        return {'x_new': self._linear.adj_op(self._x_new), 'idx': self.idx}

    def retrieve_outputs(self):
        """Retrieve outputs

        Declare the outputs of the algorithms as attributes: x_final,
        y_final, metrics.

        """

        metrics = {}
        for obs in self._observers['cv_metrics']:
            metrics[obs.name] = obs.retrieve_metrics()
        self.metrics = metrics


class VanillaGenericGradOPt(GenericGradOpt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # no scale factor
        self._speed_grad = 1.0
        self._eps = 0


class AdaGenericGradOpt(GenericGradOpt):

    def update_grad_speed(self, grad):
        self._speed_grad += abs(grad) ** 2


class RMSpropGradOpt(GenericGradOpt):
    def __init__(self, *args, **kwargs):
        gamma = kwargs.pop('gamma')
        super().__init__(*args, **kwargs)
        if gamma < 0 or gamma > 1:
            raise RuntimeError("gamma is outside of range [0,1]")
        self._check_param(gamma)
        self._gamma = gamma

    def update_grad_speed(self, grad):
        self._speed_grad = self._gamma * self._speed_grad + (1 - self._gamma) * abs(grad) ** 2


class MomemtumGradOpt(GenericGradOpt):
    def __init__(self, *args, **kwargs):
        beta = kwargs.pop('beta')
        super().__init__(*args, **kwargs)
        self._check_param(beta)
        self._beta = beta
        # no scale factor
        self._speed_grad = 1.0
        self._eps = 0.0

    def update_grad_dir(self, grad):
        self._corr_grad = self._beta * self._corr_grad + grad


class ADAMOptGradOpt(GenericGradOpt):
    def __init__(self, *args, **kwargs):
        gamma = kwargs.pop('gamma')
        beta = kwargs.pop('beta')
        super().__init__(*args, **kwargs)
        self._check_param(gamma)
        self._check_param(beta)
        if gamma < 0 or gamma >= 1:
            raise RuntimeError("gamma is outside of range [0,1]")
        if beta < 0 or beta >= 1:
            raise RuntimeError("beta is outside of range [0,1]")
        self._gamma = gamma
        self._beta = beta
        self._beta_pow = 1
        self._gamma_pow = 1

    def update_grad_dir(self, grad):
        self._beta_pow *= self._beta

        self._corr_grad = (1.0 / (1.0 - self._beta_pow)) * (self._beta * self._corr_grad + (1 - self._beta) * grad)

    def update_grad_speed(self, grad):
        self._gamma_pow *= self._gamma
        self._speed_grad = (1.0 / (1.0 - self._gamma_pow)) * (
                    self._gamma * self._speed_grad + (1 - self._gamma) * abs(grad) ** 2)


class SAGAOptGradOpt(GenericGradOpt):
    def __init__(self, *args, **kwargs):
        self.epoch_size = kwargs.pop('epoch_size')
        super().__init__(*args, **kwargs)
        self._grad_memory = np.zeros((self.epoch_size, *self._x_old.size), dtype=self._x_old.dtype)

    def update_grad_dir(self, grad):
        cycle = self.iter % self.epoch_size
        self._corr_grad = self._corr_grad - self._grad_memory[cycle] + grad
        self._grad_memory[cycle] = grad


GRAD_OPT = {
    "vanilla": VanillaGenericGradOPt,
    "adagrad": AdaGenericGradOpt,
    "rmsprop": RMSpropGradOpt,
    "momentum": MomemtumGradOpt,
    "adam": ADAMOptGradOpt,
    "saga": SAGAOptGradOpt,
}


def gradient_online(opt_cls, kspace_generator, gradient_op, linear_op, prox_op, cost_op,
                    x_init=None,
                    nb_run=1,
                    metric_call_period=5,
                    metrics=None,
                    estimate_call_period=None,
                    verbose=0, **kwargs):
    if metrics is None:
        metrics = dict()
    start = time.perf_counter()

    # Define the initial primal and dual solutions
    if x_init is None:
        x_init = linear_op.op(np.squeeze(np.zeros((gradient_op.fourier_op.n_coils,
                                                   *gradient_op.fourier_op.shape),
                                                  dtype=np.complex)))
    # Welcome message
    if verbose > 0:
        print(" - mu: ", prox_op.weights)
        print(" - lipschitz constant: ", gradient_op.spec_rad)
        print(" - data: ", gradient_op.fourier_op.shape)
        if hasattr(linear_op, "nb_scale"):
            print(" - wavelet: ", linear_op, "-", linear_op.nb_scale)
        print("-" * 40)

    opt = opt_cls(x=x_init,
                  grad=gradient_op,
                  prox=prox_op,
                  linear=linear_op,
                  cost=cost_op,
                  metric_call_period=metric_call_period,
                  metrics=metrics,
                  **kwargs)
    return online_algorithm(opt,kspace_generator, estimate_call_period=estimate_call_period,nb_run=nb_run )
