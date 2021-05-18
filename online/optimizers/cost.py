"""
Various utility classes and function for the optimizers
"""
from modopt.opt.cost import costObj
import numpy as np

class ArrayGenericCost(costObj):
    """ Define the Generic cost function, based on the cost function of the
    gradient operator and the cost function of the proximity operator.
    """
    def __init__(self, gradient_op_list, prox_op, initial_cost=1e6,
                 tolerance=1e-4, cost_interval=None, test_range=4,
                 optimizer_type='forward_backward',
                 verbose=False, plot_output=None):
        """ Initialize the 'Cost' class.
        Parameters
        ----------
        online_gradient_op: instance of the gradient operator
            gradient operator used in the reconstruction process. It must
            implements the get_cost_function.
        offline_gradient_op: instance of the gradient operator
            gradient operator used in the reconstruction process. It must
            implements the get_cost_function.
        prox_op: instance of the proximity operator
            proximity operator used in the reconstruction process. It must
            implements the get_cost function.
        initial_cost: float, optional
            Initial value of the cost (default is "1e6")
        tolerance: float, optional
            Tolerance threshold for convergence (default is "1e-4")
        cost_interval: int, optional
            Iteration interval to calculate cost (default is "None")
            if None, cost is never calculated.
        test_range: int, optional
            Number of cost values to be used in test (default is "4")
        optimizer_type: str, default 'forward_backward'
            Specifies the type of optimizer being used. This could be
            'primal_dual' or 'forward_backward'. The cost function being
            calculated would be different in case of 'primal_dual' as
            we receive both primal and dual intermediate solutions.
        verbose: bool, optional
            Option for verbose output (default is "False")
        plot_output: str, optional
            Output file name for cost function plot
        """
        for grad in gradient_op_list:
            cost_func = getattr(grad,'cost',None)
            if not callable(cost_func):
                raise RuntimeError(f"The gradient {grad} must implements a `cost`",
                                   "function")
        prox_cost = getattr(prox_op, 'cost', None)
        if not callable(prox_cost):
            raise RuntimeError("The proximity operator must implements a",
                               " `cost` function")
        self.gradient_op_list = gradient_op_list
        self.prox_op = prox_op
        self.optimizer_type = optimizer_type

        super().__init__(
            operators=None, initial_cost=initial_cost,
            tolerance=tolerance,
            cost_interval=cost_interval, test_range=test_range,
            verbose=verbose, plot_output=plot_output)
        self._iteration = 0

    def _calc_cost(self, x_new, *args, **kwargs):
        """ Return the cost.
        Parameters
        ----------
        x_new: np.ndarray
            intermediate solution in the optimization problem.
        Returns
        -------
        cost: float
            the cost function defined by the operators (gradient + prox_op).
        """

        cost = np.zeros(len(self.gradient_op_list))
        for k, grad in enumerate(self.gradient_op_list):
            if self.optimizer_type == 'forward_backward':
                cost[k] = grad.cost(x_new) + self.prox_op.cost(x_new)
            else:
                # In primal dual algorithm, the value of args[0] is the data in
                # Wavelet Space, while x_new is data in Image space.
                # TODO, we need to generalize this
                cost[k] = grad.cost(x_new) + self.prox_op.cost(args[0])
        return cost
