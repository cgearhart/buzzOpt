
import numpy as np

from problem import BinaryOptimizationProblem
import optimize


class FourPeaksProblem(BinaryOptimizationProblem):
    """
    Four peaks problem - boolean function with a maximum at T+1
    1's followed by all 0's, or all 1's preceding T+1 0's.

    Parameters
    ----------
    T : int
        Controls the number of leading/trailing 1's/0's for the
        global optimum

    n : int
        Number of entries in vector

    Methods
    -------
    fitness :
        The fitness method takes a population of instances of the
        problem domain as input and invokes the cost function on
        each element to return a sorted collection of the input
        population and corresponding fitness scores.

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

    def __init__(self, T, n):
        super(FourPeaksProblem, self).__init__(n)
        self.T = T

    def cost(self, sample):
        """
        Calculate the cost of the input sample

        Parameters
        ----------
        sample : ndarray
            Input data

        Returns
        -------
         : float
            N/A
        """
        assert len(sample) == len(self.domain)

        tail = self.n - len(np.trim_zeros(sample, 'b'))
        head = self.n - len(np.trim_zeros(np.logical_not(sample), 'f'))
        Rxt = self.n if tail > self.T and head > self.T else 0
        return -float(max(head, tail) + Rxt)

    def neighborhood(self, sample):
        """
        Enumerate the neighbors of the current sample

        Parameters
        ----------
        sample : ndarray
            Input data

        Yields
        -------
        new_sample : ndarray
            N/A
        """
        assert len(sample) == len(self.domain)
        for idx in range(len(sample)):
            new_sample = np.array(sample)
            new_sample[idx] = int(np.logical_not(sample[idx]))
            yield new_sample

    def fitness(self):
        raise NotImplementedError


if __name__ == "__main__":
    problem = FourPeaksProblem(3, 30)
    solution = optimize.mimic(problem, niter=10)
    print solution, problem.cost(solution)
