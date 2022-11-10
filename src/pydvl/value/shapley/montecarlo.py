r"""
Monte Carlo approximations to Shapley Data values.

**Note:** You probably want to use the common interface provided by
:func:`~pydvl.value.shapley.compute_shapley_values` instead of using the functions in
this module.

Exact computation of Shapley value requires $\mathcal{O}(2^n)$ retrainings of
the model. Recall the definition of the value of sample $i$:

$$v_i = \frac{1}{N}  \sum_{S \subseteq D_{\backslash \{ i \}}}
\frac{1}{\binom{N - 1}{ | S | }} [U (S_{\cup \{ i \}}) - U (S)] ,$$

where $D$ is the set of indices in the training set, which we identify with the
data itself.

To overcome this limitation, it is possible to only sample some subsets of the
training set (or permutations thereof) to obtain a Monte Carlo approximation to
the true value. This is done in
:func:`~pydvl.value.shapley.montecarlo.combinatorial_montecarlo_shapley`. Alternatively,
employing the reformulation of the expression above as a sum over permutations,
one has the implementation in
:func:`~pydvl.value.shapley.montecarlo.permutation_montecarlo_shapley`.

Additionally, one can implement an early stopping strategy to
adapt computation time. This is done in
:func:`~pydvl.value.shapley.montecarlo.truncated_montecarlo_shapley`.

Finally, you can consider grouping your data points using
:class:`~pydvl.utils.dataset.GroupedDataset` and computing the values of the
groups instead.
"""

import logging
import math
from collections import OrderedDict
from enum import Enum
from time import sleep, time
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy as np

from ...reporting.scores import sort_values
from ...utils import Utility, maybe_progress
from ...utils.config import ParallelConfig
from ...utils.numeric import get_running_avg_variance, random_powerset
from ...utils.parallel import MapReduceJob, init_parallel_backend
from .actor import get_shapley_coordinator, get_shapley_worker

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MonteCarloResults(NamedTuple):
    values: "NDArray"
    stderr: "NDArray"
    n_samples: Optional[int] = None


logger = logging.getLogger(__name__)

__all__ = [
    "truncated_montecarlo_shapley",
    "permutation_montecarlo_shapley",
    "combinatorial_montecarlo_shapley",
    "owen_sampling_shapley",
]


def truncated_montecarlo_shapley(
    u: Utility,
    value_tolerance: Optional[float] = None,
    max_iterations: Optional[int] = None,
    *,
    n_jobs: Optional[int] = None,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
    coordinator_update_frequency: int = 10,
    worker_update_frequency: int = 5,
) -> Tuple["OrderedDict[str, float]", Dict[str, float]]:
    """Monte Carlo approximation to the Shapley value of data points.

    This implements the permutation-based method described in [1]. It is a Monte
    Carlo estimate of the sum over all possible permutations of the index set,
    with a double stopping criterion.

    .. warning::

       This function does not exactly reproduce the stopping criterion of [1]
       which uses a hardcoded time delay in the sequence of values. Instead, we
       use a moving average and the stopping criterion detailed in
       :meth:`~pydvl.value.shapley.actor.ShapleyCoordinator.check_done`.

    .. todo::
       Implement the original stopping criterion, maybe Robin-Gelman or some
       other more principled one.

    Instead of naively implementing the expectation, we sequentially add points
    to a dataset from a permutation. We keep sampling permutations and updating
    all shapley values until the std/value score in the moving average falls
    below a given threshold (value_tolerance) or when the number of iterations
    exceeds a certain number (max_iterations).

    :param u: Utility object with model, data, and scoring function
    :param value_tolerance: Terminate if the standard deviation of the
        average value for every sample has dropped below this value
    :param max_iterations: Terminate if the total number of permutations exceeds
        this number.
    :param n_jobs: number of jobs processing permutations. If None, it will be
        set to :func:`available_cpus`.
    :param config: Object configuring parallel computation, with cluster address,
        number of cpus, etc.
    :param progress: Whether to display progress bars for each job.
    :param coordinator_update_frequency: in seconds. Check status with the job
        coordinator every so often.
    :param worker_update_frequency: interval in seconds between different updates to
        and from the coordinator
    :return: Tuple with the first element being an :obj:`collections.OrderedDict`
        of approximate Shapley values for the indices, and the second being the
        estimated standard error of each value.

    .. rubric::References

    [1]: Ghorbani, Amirata, and James Zou. ‘Data Shapley: Equitable Valuation of
    Data for Machine Learning’. In International Conference on Machine Learning,
    2242–51. PMLR, 2019. http://proceedings.mlr.press/v97/ghorbani19c.html.
    """
    parallel_backend = init_parallel_backend(config)

    n_jobs = parallel_backend.effective_n_jobs(n_jobs)

    u_id = parallel_backend.put(u)

    coordinator = get_shapley_coordinator(  # type: ignore
        value_tolerance, max_iterations, progress, config=config
    )
    workers = [
        get_shapley_worker(  # type: ignore
            u=u_id,
            coordinator=coordinator,
            worker_id=worker_id,
            progress=progress,
            update_frequency=worker_update_frequency,
            config=config,
        )
        for worker_id in range(n_jobs)
    ]
    for worker_id in range(n_jobs):
        workers[worker_id].run(block=False)
    last_update_time = time()
    is_done = False
    while not is_done:
        sleep(0.01)
        if time() - last_update_time > coordinator_update_frequency:
            is_done = coordinator.check_done()
            last_update_time = time()
    dvl_values, dvl_std = coordinator.get_results()
    sorted_shapley_values = sort_values(
        {u.data.data_names[i]: v for i, v in enumerate(dvl_values)}
    )
    montecarlo_error = {u.data.data_names[i]: v for i, v in enumerate(dvl_std)}
    return sorted_shapley_values, montecarlo_error


def _permutation_montecarlo_marginals(
    u: Utility, max_permutations: int, progress: bool = False, job_id: int = 1
) -> "NDArray":
    """Helper function for :func:`permutation_montecarlo_shapley`.

    Computes marginal utilities of each training sample in
    :obj:`pydvl.utils.utility.Utility.data` by iterating through randomly
    sampled permutations.

    :param u: Utility object with model, data, and scoring function
    :param max_permutations: total number of permutations to use
    :param progress: Whether to display progress bars for each job.
    :param job_id: id to use for reporting progress (e.g. to place progres bars)
    :return: a matrix with each row being a different permutation and each
        column being the score of a different data point
    """
    n = len(u.data)
    values = np.zeros(shape=(max_permutations, n))
    pbar = maybe_progress(max_permutations, progress, position=job_id)
    for iter_idx in pbar:
        prev_score = 0.0
        permutation = np.random.permutation(u.data.indices)
        marginals = np.zeros(shape=n)
        for i, idx in enumerate(permutation):
            score = u(permutation[: i + 1])
            marginals[idx] = score - prev_score
            prev_score = score
        values[iter_idx] = marginals
    return values


def permutation_montecarlo_shapley(
    u: Utility,
    max_iterations: int,
    *,
    n_jobs: int,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> Tuple["OrderedDict[str, float]", Dict[str, float]]:
    """Computes an approximate Shapley value using independent index permutations.

    :param u: Utility object with model, data, and scoring function
    :param max_iterations: total number of iterations (permutations) to use
    :param n_jobs: number of jobs across which to distribute the computation.
    :param config: Object configuring parallel computation, with cluster address,
        number of cpus, etc.
    :param progress: Whether to display progress bars for each job.
    :return: Tuple with the first element being an ordered Dict of approximate
        Shapley values for the indices, the second being their standard error
    """
    parallel_backend = init_parallel_backend(config)

    u_id = parallel_backend.put(u)

    iterations_per_job = max(1, max_iterations // n_jobs)

    map_reduce_job: MapReduceJob["NDArray", "NDArray"] = MapReduceJob(
        map_func=_permutation_montecarlo_marginals,
        reduce_func=np.concatenate,  # type: ignore
        map_kwargs=dict(max_permutations=iterations_per_job, progress=progress),
        reduce_kwargs=dict(axis=0),
        config=config,
        chunkify_inputs=False,
        n_jobs=n_jobs,
    )
    full_results = map_reduce_job(u_id)[0]

    acc = np.mean(full_results, axis=0)
    acc_std = np.std(full_results, axis=0) / np.sqrt(full_results.shape[0])
    sorted_shapley_values = sort_values(
        {u.data.data_names[i]: v for i, v in enumerate(acc)}
    )
    montecarlo_error = {u.data.data_names[i]: v for i, v in enumerate(acc_std)}
    return sorted_shapley_values, montecarlo_error


def _combinatorial_montecarlo_shapley(
    indices: Sequence[int],
    u: Utility,
    max_iterations: int,
    *,
    progress: bool = False,
    job_id: int = 1,
) -> MonteCarloResults:
    """Helper function for :func:`combinatorial_montecarlo_shapley`.

    This is the code that is sent to workers to compute values using the
    combinatorial definition.

    :param u: Utility object with model, data, and scoring function
    :param max_iterations: total number of subsets to sample.
    :param progress: Whether to display progress bars for each job.
    :param job_id: id to use for reporting progress
    :return: A tuple of ndarrays with estimated values and standard errors
    """
    n = len(u.data)

    if len(np.unique(indices)) != len(indices):
        raise ValueError("Repeated indices passed")

    # Correction coming from Monte Carlo integration so that the mean of the
    # marginals converges to the value: the uniform distribution over the
    # powerset of a set with n-1 elements has mass 2^{n-1} over each subset. The
    # additional factor n corresponds to the one in the Shapley definition
    correction = 2 ** (n - 1) / n

    values = np.zeros(n)
    variances = np.zeros(n)
    counts = np.zeros(n)
    pbar = maybe_progress(indices, progress, position=job_id)
    for idx in pbar:
        # Randomly sample subsets of full dataset without idx
        subset = np.setxor1d(u.data.indices, [idx], assume_unique=True)
        power_set = random_powerset(subset, max_subsets=max_iterations)
        for s in maybe_progress(
            power_set,
            progress,
            desc=f"Index {idx}",
            total=max_iterations,
            position=job_id,
        ):
            marginal = (u({idx}.union(s)) - u(s)) / math.comb(n - 1, len(s))
            values[idx], variances[idx] = get_running_avg_variance(
                values[idx], variances[idx], marginal, counts[idx]
            )
            counts[idx] += 1

    return MonteCarloResults(
        values=correction * values,
        stderr=np.sqrt(correction**2 * variances / np.maximum(1, counts)),
    )


def combinatorial_montecarlo_shapley(
    u: Utility,
    max_iterations: int,
    *,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> Tuple["OrderedDict[str, float]", Dict[str, float]]:
    """Computes an approximate Shapley value using the combinatorial definition.

    This consists of randomly sampling subsets of the power set of the training
    indices in :attr:`~pydvl.utils.utility.Utility.data`, and computing their
    marginal utilities.

    :param u: Utility object with model, data, and scoring function
    :param max_iterations: Number of subsets to sample from the power set for
        every index
    :param n_jobs: number of parallel jobs across which to distribute the
        computation. Each worker receives a chunk of
        :attr:`~pydvl.utils.dataset.Dataset.indices`
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param progress: Whether to display progress bars for each job.
    :return: Tuple with the first element being an ordered Dict of approximate
        Shapley values for the indices, the second being their standard error
    """
    parallel_backend = init_parallel_backend(config)
    u_id = parallel_backend.put(u)

    def reducer(results_it: Iterable[MonteCarloResults]) -> MonteCarloResults:
        values = np.zeros(len(u.data))
        stderr = np.zeros_like(values)

        # non-zero indices in results are disjoint by construction, so it is ok
        # to add the results
        for val, std, _ in results_it:
            values += val
            stderr += std
        return MonteCarloResults(values=values, stderr=stderr)

    # FIXME? max_iterations has different semantics in permutation-based methods
    map_reduce_job: MapReduceJob["NDArray", MonteCarloResults] = MapReduceJob(
        map_func=_combinatorial_montecarlo_shapley,
        reduce_func=reducer,
        map_kwargs=dict(u=u_id, max_iterations=max_iterations, progress=progress),
        chunkify_inputs=True,
        n_jobs=n_jobs,
        config=config,
    )
    results = map_reduce_job(u.data.indices)[0]
    sorted_shapley_values = sort_values(
        {u.data.data_names[i]: v for i, v in enumerate(results.values)}
    )
    montecarlo_errors = {u.data.data_names[i]: v for i, v in enumerate(results.stderr)}

    return sorted_shapley_values, montecarlo_errors


class OwenAlgorithm(Enum):
    Full = "full"
    Halved = "halved"


def _owen_sampling_shapley(
    q_values: Sequence[float],
    u: Utility,
    method: OwenAlgorithm,
    max_iterations: int,
    *,
    progress: bool = False,
    job_id: int = 1,
) -> MonteCarloResults:
    r"""This is the algorithm as detailed in the paper: to compute the outer
    integral over q ∈ [0,1], use uniformly distributed points for evaluation
    of the integrand. For the integrand (the expected marginal utility over the
    power set), use Monte Carlo.

    .. todo::
        We might want to try better quadrature rules like Gauss or Rombert or
        use Monte Carlo for the double integral.

    :param q_values: Values for the sampling probability to evaluate
    :param u: Utility object with model, data, and scoring function
    :param method: Either :attr:`~OwenAlgorithm.Full` for $q \in [0,1]$ or
        :attr:`~OwenAlgorithm.Halved` for $q \in [0,0.5]$ and correlated samples
    :param max_iterations: Number of subsets to sample to estimate the integrand
    :param progress: Whether to display progress bars for each job
    :param job_id: For positioning of the progress bar
    :return: Values and standard errors
    """
    n = len(u.data)
    values = np.zeros(n)
    # variances = np.zeros(n)
    # counts = np.zeros(n)

    indices = set(u.data.indices)
    pbar = maybe_progress(q_values, progress, position=job_id)
    for q in pbar:
        power_set = random_powerset(u.data.indices, max_subsets=max_iterations, q=q)
        for s in maybe_progress(
            power_set,
            progress,
            desc=f"Iterating over subsets",  # FIXME: Useless
            total=max_iterations,
            position=job_id,
        ):
            # Instead of sampling from D\{i} for each i ∈ D, above we
            # sampled from all of D. Therefore {i}.union(s) == s for all
            # i ∈ s, and we need to skip those indices here
            # Also, because the expectation in the integrand is over D\{i},
            # we need to add a correction factor of 1/(1-q)
            # FIXME: for exact u, the factor has to be q, for linear u, 2(1-q)
            #  etc. => Something is very fishy
            correction = max(1 / 500, 1 - q)
            # q if method is OwenAlgorithm.Full else (2*(1 - q)))
            for i in u.data.indices:
                if i in s:
                    if method == OwenAlgorithm.Full:
                        continue
                    # OwenHalved:
                    # i not in s_complement => compute marginal for D\s
                    s_complement = indices.difference(s)
                    marginal = u({i}.union(s_complement)) - u(s_complement)
                else:
                    marginal = u({i}.union(s)) - u(s)
                values[i] += marginal / correction

                # values[i], variances[i] = get_running_avg_variance(
                #     values[i], variances[i], marginal, counts[i]
                # )
                # counts[i] += 1

    return MonteCarloResults(
        values=values,
        stderr=np.zeros_like(values),  # np.sqrt(variances / np.maximum(1, counts)),
        n_samples=len(q_values) * max_iterations,
    )


def owen_sampling_shapley(
    u: Utility,
    max_iterations: int,
    max_q: int,
    *,
    method: OwenAlgorithm = OwenAlgorithm.Full,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> Tuple["OrderedDict[str, float]", Dict[str, float]]:
    r"""Owen sampling of Shapley values.

    This function computes a Monte Carlo approximation to

    $$v_u(i) = \int_0^1 \mathbb{E}_{S \sim P_q(D_{\backslash \{ i \}})} [u(S \cup {i}) - u(S)]$$

    as described in [1], using one of two methods. The first one, selected with
    the argument `mode = OwenAlgorithm.Full`, approximates the integral with:

    $$\hat{v}_u(i) = \frac{1}{Q M} \sum_{j=0}^Q \sum_{m=1}^M [u(S^{(q_j)}_m \cup {i}) - u(S^{(q_j)}_m)],$$

    where $q_j = \frac{j}{Q} \in [0,1]$ and the sets $S^{(q_j)}$ are such that a
    sample $x \in S^{(q_j)}$ if a draw from a $Ber(q_j)$ distribution is 1. The
    second method, selected with the argument `mode = OwenAlgorithm.Halved`,
    uses correlated samples in the inner sum to reduce the variance:

    $$\hat{v}_u(i) = \frac{1}{Q M} \sum_{j=0}^Q \sum_{m=1}^M [u(S^{(q_j)}_m
    \cup {i}) - u(S^{(q_j)}_m) + u((S^{(q_j)}_m)^c \cup {i}) - u((S^{(
    q_j)}_m)^c)],$$

    where now $q_j = \frac{j}{2Q} \in [0,\frac{1}{2}]$, and $S^c$ is the
    complement of $S$.

    :param u: :class:`~pydvl.utils.utility.Utility` object holding data, model
        and scoring function.
    :param max_iterations: Numer of sets to sample for each value of q
    :param max_q: Number of subdivisions for q ∈ [0,1] (the element sampling
        probability) used to approximate the outer integral.
    :param method: Selects the algorithm to use, see the description. Either
        :attr:`~OwenAlgorithm.Full` for $q \in [0,1]$ or
        :attr:`~OwenAlgorithm.Halved` for $q \in [0,0.5]$ and correlated samples
    :param n_jobs: Number of parallel jobs to use. Each worker receives a chunk
        of the total of `max_q` values for q.
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param progress: Whether to display progress bars for each job.
    :return: Tuple with the first element being an ordered Dict of approximate
        Shapley values for the indices, the second being their standard error

    .. rubric:: References

    [1]: Okhrati, Ramin, and Aldo Lipani. ‘A Multilinear Sampling Algorithm
    to Estimate Shapley Values’. In 2020 25th International Conference on
    Pattern Recognition (ICPR), 7992–99. IEEE, 2021.
    https://doi.org/10.1109/ICPR48806.2021.9412511.

    .. versionadded:: 0.3.0

    """

    parallel_backend = init_parallel_backend(config)
    u_id = parallel_backend.put(u)

    def reducer(results_it: Iterable[MonteCarloResults]) -> MonteCarloResults:
        values = np.zeros(len(u.data))
        n_samples = 0
        for val, _, samples in results_it:
            values += val
            n_samples += cast(int, samples)  # We know samples is not None
        values /= n_samples
        # values /= max_q * max_iterations
        return MonteCarloResults(values=values, stderr=np.zeros_like(values))

    map_reduce_job: MapReduceJob["NDArray", MonteCarloResults] = MapReduceJob(
        map_func=_owen_sampling_shapley,
        reduce_func=reducer,
        map_kwargs=dict(
            u=u_id, method=method, max_iterations=max_iterations, progress=progress
        ),
        chunkify_inputs=True,
        n_jobs=n_jobs,
        config=config,
    )

    # TODO: try non-uniform steps: can one obtain e.g. Beta-Shapley?
    q_stop = {OwenAlgorithm.Full: 1.0, OwenAlgorithm.Halved: 0.5}
    q_steps = np.linspace(start=0, stop=q_stop[method], num=max_q)
    results = map_reduce_job(q_steps)[0]
    sorted_shapley_values = sort_values(
        {u.data.data_names[i]: v for i, v in enumerate(results.values)}
    )
    montecarlo_errors = {u.data.data_names[i]: v for i, v in enumerate(results.stderr)}

    return sorted_shapley_values, montecarlo_errors
