import numpy as np

# Third party import
from modopt.opt.algorithms import SetUp
from modopt.opt.cost import costObj


class GenericGradOpt(SetUp):
    """Generic Gradient descent operator
    x_{k+1} = x_k - \\frac{\\eta}{\\sqrt{s_k + \\epsilon}} m_k
    """

    def __init__(
        self,
        x,
        grad,
        prox,
        linear,
        cost,
        eta=1,
        eta_update=None,
        epsilon=1e-6,
        metric_call_period=5,
        metrics=None,
        **kwargs
    ):
        # Set the initial variable values
        if metrics is None:
            metrics = dict()
        # Set default algorithm properties
        super().__init__(
            metric_call_period=metric_call_period, metrics=metrics, **kwargs
        )
        self.iter = 0
        self._check_input_data(x)
        self._x_old = np.copy(x)
        self._x_new = np.copy(x)
        self._speed_grad = np.zeros(x.shape, dtype=float)
        self._dir_grad = np.zeros_like(x)
        # Set the algorithm operators
        for operator in (grad, prox, cost):
            self._check_operator(operator)
        self._grad = grad
        self._prox = prox
        self._linear = linear
        if cost == "auto":
            self._cost_func = costObj([self._grad, self._prox])
        else:
            self._cost_func = cost
        # Set the algorithm parameters
        for param in (eta, epsilon):
            self._check_param(param)
        self._eta = eta
        self._eps = epsilon

        # Set the algorithm parameter update methods
        self._check_param_update(eta_update)
        self._eta_update = eta_update
        self.idx = 0
        self.epoch_size = 1

    def _update(self):
        self._grad.get_grad(self._x_old)
        self.update_grad_dir(self._grad.grad)
        self.update_grad_speed(self._grad.grad)
        step = self._eta / (np.sqrt(self._speed_grad) + self._eps)
        self._x_new = self._x_old - step * self._dir_grad
        if self.idx % self.epoch_size == 0:
            self.update_reg(step)
        self._x_old = self._x_new.copy()
        if self._eta_update is not None:
            self._eta = self._eta_update(self._eta, self.idx)
        # Test cost function for convergence.
        if self._cost_func:
            self.converge = self.any_convergence_flag() or self._cost_func.get_cost(
                self._x_new
            )

    def update_grad_dir(self, grad):
        self._dir_grad = grad

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
        return {
            "x_new": self._linear.adj_op(self._x_new),
            "dir_grad": self._dir_grad,
            "speed_grad": self._speed_grad,
            "idx": self.idx,
        }

    def retrieve_outputs(self):
        """Retrieve outputs

        Declare the outputs of the algorithms as attributes: x_final,
        y_final, metrics.

        """

        metrics = {}
        for obs in self._observers["cv_metrics"]:
            metrics[obs.name] = obs.retrieve_metrics()
        self.metrics = metrics

    def reset(self):
        pass


class VanillaGenericGradOpt(GenericGradOpt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # no scale factor
        self._speed_grad = 1.0
        self._eps = 0


class VanillaEpochGenericGradOpt(VanillaGenericGradOpt):
    def __init__(self, *args, epoch_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_size = epoch_size

    def get_notify_observers_kwargs(self):
        return {
            "x_new": self._x_new,
            "dir_grad": self._dir_grad,
            "speed_grad": self._speed_grad,
            "idx": self.idx,
        }

    def update_reg(self, factor):
        self._x_new = self._linear.adj_op(
            self._prox.op(self._linear.op(self._x_new), extra_factor=factor)
        )


class AdaGenericGradOpt(GenericGradOpt):
    def update_grad_speed(self, grad):
        self._speed_grad += abs(grad) ** 2


class RMSpropGradOpt(GenericGradOpt):
    def __init__(self, *args, **kwargs):
        gamma = kwargs.pop("gamma")
        super().__init__(*args, **kwargs)
        if gamma < 0 or gamma > 1:
            raise RuntimeError("gamma is outside of range [0,1]")
        self._check_param(gamma)
        self._gamma = gamma

    def update_grad_speed(self, grad):
        self._speed_grad = (
            self._gamma * self._speed_grad + (1 - self._gamma) * abs(grad) ** 2
        )


class MomentumGradOpt(GenericGradOpt):
    def __init__(self, *args, **kwargs):
        beta = kwargs.pop("beta")
        super().__init__(*args, **kwargs)
        self._check_param(beta)
        self._beta = beta
        # no scale factor
        self._speed_grad = 1.0
        self._eps = 0.0

    def update_grad_dir(self, grad):
        self._dir_grad = self._beta * self._dir_grad + grad

    def reset(self):
        self._dir_grad = np.zeros_like(self._x_new)


class MomentumEpochGradOpt(GenericGradOpt):
    def __init__(self, *args, epoch_size=1, **kwargs):
        beta = kwargs.pop("beta")
        super().__init__(*args, **kwargs)
        self.epoch_size = epoch_size
        self._check_param(beta)
        self._beta = beta
        # no scale factor
        self._speed_grad = 1.0
        self._eps = 0.0

    def update_grad_dir(self, grad):
        self._dir_grad = self._beta * self._dir_grad + grad

    def reset(self):
        self._dir_grad = np.zeros_like(self._x_new)

    def get_notify_observers_kwargs(self):
        return {
            "x_new": self._x_new,
            "dir_grad": self._dir_grad,
            "speed_grad": self._speed_grad,
            "idx": self.idx,
        }

    def update_reg(self, factor):
        self._x_new = self._linear.adj_op(
            self._prox.op(self._linear.op(self._x_new), extra_factor=factor)
        )

    pass


class ADAMOptGradOpt(GenericGradOpt):
    def __init__(self, *args, **kwargs):
        gamma = kwargs.pop("gamma")
        beta = kwargs.pop("beta")
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

        self._dir_grad = (1.0 / (1.0 - self._beta_pow)) * (
            self._beta * self._dir_grad + (1 - self._beta) * grad
        )

    def update_grad_speed(self, grad):
        self._gamma_pow *= self._gamma
        self._speed_grad = (1.0 / (1.0 - self._gamma_pow)) * (
            self._gamma * self._speed_grad + (1 - self._gamma) * abs(grad) ** 2
        )


class SAGAOptGradOpt(GenericGradOpt):
    def __init__(self, *args, epoch_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_size = epoch_size
        self._grad_memory = np.zeros(
            (self.epoch_size, *self._x_old.size), dtype=self._x_old.dtype
        )

    def update_grad_dir(self, grad):
        cycle = self.iter % self.epoch_size
        self._dir_grad = self._dir_grad - self._grad_memory[cycle] + grad
        self._grad_memory[cycle] = grad
