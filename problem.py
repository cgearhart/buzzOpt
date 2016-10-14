
import numpy as np


class OptimizationProblem(object):
    """
    Base class for discrete, finite-domain optimization problems.

    The problem is defined by a description of the discrete domain
    and a cost function. The domain consists of a list of
    discrete-valued vectors, each containing every allowable value
    of the corresponding feature. The cost function is the function
    to be optimized, which should take a vector from the domain and
    return a floating point value.

    Parameters
    ----------
    domain : iterable
        The domain must be an iterable collection containing a
        sequence of element-wise unique sequences of values defining
        the domain of each element of the problem.

    Methods
    -------
    cost : (abstract)
        The cost function should take an instance of the problem
        domain as an input and return a floating point value.

    neighborhood : (abstract)
        The neighborhood function should take an instance of the
        problem domain as an input and return an iterable sequence
        of neighboring vectors.

    Examples
    --------
    TODO: Add examples
    """

    def __init__(self, domain):

        super(OptimizationProblem, self).__init__()

        self.domain = np.array(domain)

    def cost(self, sample):
        """
        Find the indices of array elements that are non-zero, grouped by element.

        Parameters
        ----------
        a : array_like
            Input data.

        Returns
        -------
        index_array : ndarray
            Indices of elements that are non-zero. Indices are grouped by element.
        """
        raise NotImplementedError("Cost function must be overridden.")

    def neighborhood(self, sample):
        """
        Find the indices of array elements that are non-zero, grouped by element.

        Parameters
        ----------
        a : array_like
            Input data.

        Returns
        -------
        index_array : ndarray
            Indices of elements that are non-zero. Indices are grouped by element.
        """
        raise NotImplementedError("Neighborhood function must be overridden.")


class BinaryOptimizationProblem(OptimizationProblem):

    def __init__(self, n):
        super(BinaryOptimizationProblem, self).__init__([[0, 1]] * n)
        self.n = n
