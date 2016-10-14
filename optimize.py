
import numpy as np

from collections import Counter


class DomainError(Exception):
    """
    Exception raised for values in the output of a pmf outside the domain of
    the distribution.

    Parameters
    ----------
    msg : str
        Description of the source of the error
    """

    def __init__(self, msg, *args, **kwargs):
        super(DomainError, self).__init__(*args, **kwargs)
        self.msg = msg


def entropy(population):
    """
    Calculate the entropy of the sample population treating each
    row of the population as a tuple. Finds the joint entropy if
    the population is 2-dimensional.

    Parameters
    ----------
    population : ndarray
        Observations in each row with features in each column

    Returns
    -------
    h : float64
        the empirical (joint) entropy of the population

    Examples
    --------
    TODO: add examples
    >>> x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    >>> entropy(x)
    0.721928094887
    >>> x = np.array([[0, 0], [0, 1], [1, 1]])
    >>> entropy(x)
    1.58496250072
    """
    if population.ndim > 1:
        population = map(tuple, population)
    counts = np.array(Counter(population).values(), dtype=np.float64)
    probs = counts / counts.sum()
    h = np.sum(-p * np.log2(p) for p in probs if p > 0)
    return h


def sample_from_pmf(X_data, domain, size=1, eps=0.):
    """
    Sample a discrete empirical probability mass function given by observed
    data and the discrete levels allowed for each feature by a problem domain

    Parameters
    ----------
    X_data : ndarray
        Observations in each row with features in each column

    domain : OptimizationProblem
        Problem domain that establishes the number of values for each feature

    size : int (optional)
        The number of values to sample from the pmf

    eps : float (optional)
        The value to add to features with zero observed instances

    Returns
    -------
    samples : ndarray
        Elements sampled from the domain of each feature according to the
        observed frequencies in the data

    Examples
    --------
    TODO: Add examples
    """
    if X_data.size == 0:
        return np.array([])
    counts = np.array(map(Counter(X_data).__getitem__, domain))
    zeros = np.argwhere(counts == 0)
    scale = (1 - eps * zeros.shape[0]) / counts.sum()
    probs = counts * scale
    probs[zeros] = eps
    samples = np.random.choice(domain, p=probs, size=size)
    return samples


def sample_conditional_pmf(X, XY_data, domains, eps=0.):
    """
    Generate samples from the conditional probability mass function according
    to the observed frequencies in the input data.

    Parameters
    ----------
    X : ndarray
        1-D array of values to condition on

    XY_data : ndarray
        array of observation pairs

    domains : list-like
        A pair of OptimizationProblem domains corresponding to the feature
        columns in XY_data

    eps : float (optional)
        The value to add to features with zero observed instances

    Returns
    -------
    samples : ndarray
        Elements sampled from the domain of each feature according to the
        observed frequencies in the data

    Examples
    --------
    TODO: Add examples
    """
    X_data = XY_data[:, 0]
    Y_data = XY_data[:, 1]
    x_domain, y_domain = domains

    samples = np.empty(X.shape, dtype=X.dtype)
    samples.fill(np.NAN)
    for xv in x_domain:
        dlocs = np.where(X_data == xv)
        xlocs = np.argwhere(X == xv)
        samples[xlocs] = sample_from_pmf(Y_data[dlocs], y_domain,
                                         size=len(xlocs), eps=eps)

    if np.in1d(samples, y_domain, invert=True).any():
        raise DomainError("")

    return samples


def mimic(problem, size=1000, cutoff=0.1, niter=50, decay=0):
    """
    Mutual Information Maximization Input Clustering

    Parameters
    ----------
    problem : OptimizationProblem
        Input data.

    size : int (optional)
        N/A

    cutoff : float (optional)
        N/A

    niter : int (optional)
        N/A

    decay : float (optional)
        N/A

    Returns
    -------
    sample : ndarray
        The optimal element from the last population cohort

    Examples
    --------
    TODO: Add examples
    """

    # initialize the population with uniform sampling
    samples = np.array([np.random.choice(d, size=size) for d in problem.domain]).T

    for _ in range(niter):

        new_samples = np.ndarray(samples.shape)

        # evaluate the population and keep only the most fit members
        scores = np.apply_along_axis(problem.cost, 1, samples)
        order = np.argsort(scores)
        cut_idx = int(np.ceil(cutoff * size))
        cutoff *= (1. - decay)
        samples = samples[order[:cut_idx]]

        # resample the population for the next round based on the conditional
        # probability distributions observed in the current population
        mst_candidates = np.ones(len(problem.domain), dtype=bool)
        hx = np.array([entropy(samples[:, idx]) for idx in range(len(problem.domain))])

        # start from the column with the smallest entropy
        idx = np.argmin(hx)
        mst_candidates[idx] = False
        new_samples[:, idx] = sample_from_pmf(samples[:, idx],
                                              problem.domain[idx],
                                              size=size)

        # initialize an array list of the node indices, the current
        # best parent index, and the distance (negative mutual
        # information)
        mst_idx = np.arange(len(problem.domain), dtype=np.int)
        mst = np.ndarray((len(problem.domain), 3))
        mst[:, 0] = mst_idx
        mst[:, 1] = idx  # the starting index is the root of the tree
        mst[:, 2] = ([entropy(samples[:, [idx, jdx]]) for jdx in mst_idx] -
                     hx[idx] - hx[mst_idx])

        while mst_candidates.any():

            # Select the next feature index according to the selection criteria
            # from prim's algorithm for the MST
            min_idx = np.argmin(mst[mst_candidates, 2])
            prev, idx = idx, mst[mst_candidates, 0][min_idx]
            mst_candidates[idx] = False

            # Generate samples from the empirical conditional probability
            # distribution
            cols = [prev, idx]
            new_samples[:, idx] = sample_conditional_pmf(new_samples[:, prev],
                                                         samples[:, cols],
                                                         problem.domain[cols])

            # Update the minimum spanning tree table
            info = ([entropy(samples[:, [idx, jdx]]) for jdx in mst_idx] -
                    hx[idx] - hx[mst_idx])
            updates = np.where(info < mst[:, 2])
            mst[updates, 1] = idx
            mst[updates, 2] = info[updates]

        samples = new_samples

    scores = np.apply_along_axis(problem.cost, 1, samples)
    min_idx = np.argmin(scores)

    return samples[min_idx]

