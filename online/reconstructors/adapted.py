# Package import
from mri.reconstructors.calibrationless import CalibrationlessReconstructor
from mri.optimizers.utils.cost import GenericCost

from online.optimizers.forward_backward import fista_online, pogm_online
from online.optimizers.primal_dual import condatvu_online


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
        cost_op_kwargs=None,
        **kwargs
    ):
        """This method calculates operator transform.
        Parameters
        ----------
        kspace_generator: class instance
            Provides the kspace for each iteration of the algorithm,
            the optimisation algorithm will run until the last iteration of the generator.
        optimization_alg: str (optional, default 'pogm')
            Type of optimization algorithm to use, 'pogm' | 'fista' |
            'condatvu'
        cost_op_kwargs: dict (optional, default None)
            specifies the extra keyword arguments for cost operations.
            please refer to modopt.opt.cost.costObj for details.
        x_init: np.ndarray (optional, default None)
            input initial guess image for reconstruction. If None, the
            initialization will be zero

        """
        available_algorithms = ["condatvu", "fista", "pogm"]
        if optimization_alg not in available_algorithms:
            raise ValueError(
                "The optimization_alg must be one of " + str(available_algorithms)
            )
        optimizer = eval(optimization_alg + "_online")

        if optimization_alg == "condatvu":
            kwargs["dual_regularizer"] = self.prox_op
            kwargs["std_est_method"] = None
            optimizer_type = "primal_dual"
        else:
            kwargs["prox_op"] = self.prox_op
            optimizer_type = "forward_backward"

        if cost_op_kwargs is None:
            cost_op_kwargs = dict()

        cost_op = GenericCost(
            gradient_op=self.gradient_op,
            prox_op=self.prox_op,
            verbose=self.verbose >= 20,
            optimizer_type=optimizer_type,
            **cost_op_kwargs,
        )
        x_final, costs, *metrics = optimizer(
            kspace_generator=kspace_generator,
            gradient_op=self.gradient_op,
            linear_op=self.linear_op,
            cost_op=cost_op,
            x_init=x_init,
            verbose=self.verbose,
            **kwargs,
        )
        if optimization_alg == "condatvu":
            metrics, y_final = metrics
        else:
            metrics = metrics[0]
        return x_final, costs, metrics
