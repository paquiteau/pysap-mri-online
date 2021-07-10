import numpy as np

from modopt.opt.proximity import ProximityParent


class LASSO(ProximityParent):
    """ LASSO norm proximity.

    This class implements the proximity operator of the group-lasso
    regularization as defined in :cite:`yuan2006`, with groups dimension
    being the first dimension.

    Parameters
    ----------
    weights : numpy.ndarray
        Input array of weights

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.opt.proximity import GroupLASSO
    >>> A = np.arange(15).reshape(3, 5)
    >>> A
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> prox_op = GroupLASSO(weights=3)
    >>> prox_op.op(A)
    array([[ 0.        ,  0.76133281,  1.5725177 ,  2.42145809,  3.29895251],
           [ 3.65835921,  4.56799689,  5.50381195,  6.45722157,  7.42264316],
           [ 7.31671843,  8.37466096,  9.4351062 , 10.49298505, 11.5463338 ]])
    >>> prox_op.cost(A, verbose=True)
    211.37821733946427

    See Also
    --------
    ProximityParent : parent class

    """

    def __init__(self, weights):
        self.weights = weights
        self.op = self._op_method
        self.cost = self._cost_method

    def _op_method(self, input_data, extra_factor=1.0):
        """Operator.

        This method returns the input data thresholded by the weights.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor (default is ``1.0``)

        Returns
        -------
        numpy.ndarray
            With proximal of GroupLASSO regularization

        """
        norm2 = np.abs(input_data)
        denominator = np.maximum(norm2, np.finfo(np.float32).eps)

        return input_data * np.maximum(
            0,
            (1.0 - self.weights * extra_factor / denominator),
        )

    def _cost_method(self, input_data):
        """Cost function.

        This method calculate the cost function of the proximable part.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input array of the sparse code

        Returns
        -------
        float
            The cost of GroupLASSO regularizer

        """
        return np.sum(self.weights * np.linalg.norm(input_data, axis=0))
