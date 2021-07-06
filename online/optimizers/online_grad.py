from .base import online_algorithm
from .gradescent import *
import time

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
    return online_algorithm(opt, kspace_generator, estimate_call_period=estimate_call_period, nb_run=nb_run)
