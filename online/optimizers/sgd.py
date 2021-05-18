import time
import numpy as np

# Third party import
from mri.optimizers.utils.reweight import mReweight
from modopt.opt.algorithms import SetUp
from modopt.opt.cost import costObj


class GenericGradOpt(SetUp):
    """ Generic Gradient descent operator
    x_{k+1} = x_k - \frac{\eta}{\sqrt{s_k + \epsilon}} m_k
    """

    def __init__(self, x, grad, prox, linear, cost, eta, eta_update=None, epsilon=1e-6, metric_call_period=5,
                 metrics=None, **kwargs):
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
        self._z_old = np.copy(x)
        self._scale = np.zeros(x.shape, dtype=float)
        self._corr_grad = np.zeros_like(x)

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

    def _update(self):
        self._grad.get_grad(self._x_old)
        self.update_grad(self._grad.grad)
        self.update_scale(self._grad.grad)

        self._x_new = self._x_old - (self._eta / np.sqrt(self._scale + self._eps)) * self._corr_grad

        self.update_reg()
        self._x_old = self._x_new.copy()

    def update_grad(self, grad):
        self._corr_grad = grad

    def update_scale(self, grad):
        pass

    def update_reg(self):
        pass


class VanillaGenericGradOPt(GenericGradOpt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # no scale factor
        self._scale = 1.0
        self._eps = 0.0

class AdaGenericGradOpt(GenericGradOpt):
    def update_scale(self, grad):
        self._scale += grad ** 2


class RMSpropGradOpt(GenericGradOpt):
    def __init__(self, *args, **kwargs):
        gamma = kwargs.pop('gamma')
        super().__init__(*args, **kwargs)
        if gamma < 0 or gamma > 1:
            raise RuntimeError("gamma is outside of range [0,1]")
        self._check_param(gamma)
        self._gamma = gamma

    def update_scale(self, grad):
        self._scale = self._gamma * self._scale + (1-self._gamma) * grad ** 2

class MomemtumGradOpt(GenericGradOpt):
    def __init__(self, *args, **kwargs):
        beta = kwargs.pop('beta')
        super().__init__(*args, **kwargs)
        self._check_param(beta)
        self._beta = beta
        # no scale factor
        self._scale = 1.0
        self._eps = 0.0

    def update_grad(self, grad):
        self._corr_grad = self._beta * self._corr_grad + grad


class ADAMOptGradOpt(GenericGradOpt):
    def __init__(self, *args, **kwargs):
        gamma = kwargs.pop('gamma')
        beta = kwargs.pop('beta')
        super().__init__(*args, **kwargs)
        self._check_param(gamma)
        self._check_param(beta)
        if gamma < 0 or gamma > 1:
            raise RuntimeError("gamma is outside of range [0,1]")
        if beta < 0 or beta > 1:
            raise RuntimeError("beta is outside of range [0,1]")
        self._gamma = gamma
        self._beta = beta
        self._beta_pow = 1
        self._gamma_pow = 1

    def update_grad(self, grad):
        self._beta_pow *= self._beta

        self._corr_grad = (1.0/(1.0 - self._beta_pow)) * (self._beta * self._corr_grad + (1-self._beta) * grad)

    def update_scale(self, grad):
        self._gamma_pow *= self._gamma
        self._scale = (1.0/(1.0 - self._gamma_pow)) * (self._gamma * self._corr_grad + (1-self._gamma) * grad ** 2)

class SAGAOptGradOpt(GenericGradOpt):
    def __init__(self, *args, **kwargs):
        self.epoch_size = kwargs.pop('epoch_size')
        super().__init__(*args, **kwargs)
        self._grad_memory = np.zeros((self.epoch_size, *self._x_old.size),dtype=self._x_old.dtype)

    def update_grad(self, grad):
        cycle = self.iter % self.epoch_size
        self._corr_grad = self._corr_grad - self._grad_memory[cycle] + grad
        self._grad_memory[cycle] = grad
