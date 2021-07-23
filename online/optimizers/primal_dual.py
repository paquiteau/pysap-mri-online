import numpy as np

# Third party import
from modopt.opt.algorithms import Condat
from modopt.opt.linear import Identity

from .base import online_algorithm


def condatvu_online(kspace_generator, gradient_op, linear_op, prox_op, cost_op,
                    max_nb_of_iter=150, tau=None, sigma=None, relaxation_factor=1.0,
                    x_init=None, std_est=None,
                    nb_run=1,
                    metric_call_period=5,
                    metrics=None,
                    estimate_call_period=None,
                    verbose=0,):
    """ The Condat-Vu sparse reconstruction with reweightings.

    Parameters
    ----------
    kspace_generator: instance of class KspaceGenerator
        the observed data (ie kspace) generated for each iteration of the algorithm
    gradient_op: instance of class GradBase
        the gradient operator.
    linear_op: instance of LinearBase
        the linear operator: seek the sparsity, ie. a wavelet transform.
    prox_op: instance of ProximityParent
        the  dual regularization operator
    cost_op: instance of costObj
        the cost function used to check for convergence during the
        optimization.
    max_nb_of_iter: int, default 150
        the maximum number of iterations in the Condat-Vu proximal-dual
        splitting algorithm.
    tau, sigma: float, default None
        parameters of the Condat-Vu proximal-dual splitting algorithm.
        If None estimates these parameters.
    relaxation_factor: float, default 0.5
        parameter of the Condat-Vu proximal-dual splitting algorithm.
        If 1, no relaxation.
    x_init: np.ndarray (optional, default None)
        the initial guess of image
    std_est: float, default None
        the noise std estimate.
        If None use the MAD as a consistent estimator for the std.
    relaxation_factor: float, default 0.5
        parameter of the Condat-Vu proximal-dual splitting algorithm.
        If 1, no relaxation.
    nb_of_reweights: int, default 1
        the number of reweightings.
    metric_call_period: int (default 5)
        the period on which the metrics are compute.
    metrics: dict (optional, default None)
        the list of desired convergence metrics: {'metric_name':
        [@metric, metric_parameter]}. See modopt for the metrics API.
    verbose: int, default 0
        the verbosity level.

    Returns
    -------
    x_final: ndarray
        the estimated CONDAT-VU solution.
    costs: list of float
        the cost function values.
    metrics: dict
        the requested metrics values during the optimization.
    y_final: ndarrat
        the estimated dual CONDAT-VU solution
    """

    # Check inputs
    if metrics is None:
        metrics = dict()

    # Define the initial primal and dual solutions
    if x_init is None:
        x_init = np.squeeze(np.zeros((gradient_op.fourier_op.n_coils,
                                      *gradient_op.fourier_op.shape),
                                     dtype=np.complex128))
    primal = x_init
    dual = linear_op.op(primal)

    # Define the Condat Vu optimizer: define the tau and sigma in the
    # Condat-Vu proximal-dual splitting algorithm if not already provided.
    # Check also that the combination of values will lead to convergence.

    norm = linear_op.l2norm(x_init.shape)
    lipschitz_cst = gradient_op.spec_rad
    if sigma is None:
        sigma = 0.5
    if tau is None:
        # to avoid numerics troubles with the convergence bound
        eps = 1.0e-8
        # due to the convergence bound
        tau = 1.0 / (lipschitz_cst / 2 + sigma * norm ** 2 + eps)
    convergence_test = (
            1.0 / tau - sigma * norm ** 2 >= lipschitz_cst / 2.0)

    # Welcome message
    if verbose > 0:
        print(" - mu: ", prox_op.weights)
        print(" - lipschitz constant: ", gradient_op.spec_rad)
        print(" - tau: ", tau)
        print(" - sigma: ", sigma)
        print(" - rho: ", relaxation_factor)
        print(" - std: ", std_est)
        print(" - 1/tau - sigma||L||^2 >= beta/2: ", convergence_test)
        print(" - data: ", gradient_op.fourier_op.shape)
        if hasattr(linear_op, "nb_scale"):
            print(" - wavelet: ", linear_op, "-", linear_op.nb_scale)
        print(" - max iterations: ", max_nb_of_iter)
        print(" - primal variable shape: ", primal.shape)
        print(" - dual variable shape: ", dual.shape)
        print("-" * 40)

    prox_primal = Identity()

    # Define the optimizer
    opt = Condat(
        x=primal,
        y=dual,
        grad=gradient_op,
        prox=prox_primal,
        prox_dual=prox_op,
        linear=linear_op,
        cost=cost_op,
        rho=relaxation_factor,
        # sigma=sigma,
        # tau=tau,
        rho_update=None,
        sigma_update=None,
        tau_update=None,
        auto_iterate=False,
        metric_call_period=metric_call_period,
        metrics=metrics)

    return online_algorithm(opt,kspace_generator, estimate_call_period=estimate_call_period, nb_run=nb_run)