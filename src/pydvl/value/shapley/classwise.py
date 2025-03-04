r"""
Class-wise Shapley (Schoch et al., 2022)[^1] offers a Shapley framework tailored
for classification problems. Let $D$ be a dataset, $D_{y_i}$ be the subset of
$D$ with labels $y_i$, and $D_{-y_i}$ be the complement of $D_{y_i}$ in $D$. The
key idea is that a sample $(x_i, y_i)$, might enhance the overall performance on
$D$, while being detrimental for the performance on $D_{y_i}$. The Class-wise
value is defined as:

$$
v_u(i) = \frac{1}{2^{|D_{-y_i}|}} \sum_{S_{-y_i}} \frac{1}{|D_{y_i}|!}
\sum_{S_{y_i}} \binom{|D_{y_i}|-1}{|S_{y_i}|}^{-1}
[u( S_{y_i} \cup \{i\} | S_{-y_i} ) − u( S_{y_i} | S_{-y_i})],
$$

where $S_{y_i} \subseteq D_{y_i} \setminus \{i\}$ and $S_{-y_i} \subseteq
D_{-y_i}$.

!!! tip "Analysis of Class-wise Shapley"
    For a detailed analysis of the method, with comparison to other valuation
    techniques, please refer to the [main documentation][intro-to-cw-shapley].

In practice, the quantity above is estimated using Monte Carlo sampling of
the powerset and the set of index permutations. This results in the estimator

$$
v_u(i) = \frac{1}{K} \sum_k \frac{1}{L} \sum_l
[u(\sigma^{(l)}_{:i} \cup \{i\} | S^{(k)} ) − u( \sigma^{(l)}_{:i} | S^{(k)})],
$$

with $S^{(1)}, \dots, S^{(K)} \subseteq T_{-y_i},$ $\sigma^{(1)}, \dots,
\sigma^{(L)} \in \Pi(T_{y_i}\setminus\{i\}),$ and $\sigma^{(l)}_{:i}$ denoting
the set of indices in permutation $\sigma^{(l)}$ before the position where $i$
appears. The sets $T_{y_i}$ and $T_{-y_i}$ are the training sets for the labels
$y_i$ and $-y_i$, respectively.

??? info "Notes for derivation of test cases"
    The unit tests include the following manually constructed data:
    Let $D=\{(1,0),(2,0),(3,0),(4,1)\}$ be the test set and $T=\{(1,0),(2,0),(3,1),(4,1)\}$
    the train set. This specific dataset is chosen as it allows to solve the model

    $$y = \max(0, \min(1, \text{round}(\beta^T x)))$$

    in closed form $\beta = \frac{\text{dot}(x, y)}{\text{dot}(x, x)}$. From the closed-form
    solution, the tables for in-class accuracy $a_S(D_{y_i})$ and out-of-class accuracy
    $a_S(D_{-y_i})$ can be calculated. By using these tables and setting
    $\{S^{(1)}, \dots, S^{(K)}\} = 2^{T_{-y_i}}$ and
    $\{\sigma^{(1)}, \dots, \sigma^{(L)}\} = \Pi(T_{y_i}\setminus\{i\})$,
    the Monte Carlo estimator can be evaluated ($2^M$ is the powerset of $M$).
    The details of the derivation are left to the eager reader.

## References

[^1]: <a name="schoch_csshapley_2022"></a>Schoch, Stephanie, Haifeng Xu, and
    Yangfeng Ji. [CS-Shapley: Class-wise Shapley Values for Data Valuation in
    Classification](https://openreview.net/forum?id=KTOcrOR5mQ9). In Proc. of
    the Thirty-Sixth Conference on Neural Information Processing Systems
    (NeurIPS). New Orleans, Louisiana, USA, 2022.

"""

import logging
import numbers
from concurrent.futures import FIRST_COMPLETED, Future, wait
from copy import copy
from typing import Callable, Optional, Set, Tuple, Union, cast

import numpy as np
from deprecate import deprecated
from numpy.random import SeedSequence
from numpy.typing import NDArray
from tqdm import tqdm

from pydvl.parallel import ParallelBackend, ParallelConfig, _maybe_init_parallel_backend
from pydvl.utils import (
    Dataset,
    Scorer,
    ScorerCallable,
    Seed,
    SupervisedModel,
    Utility,
    ensure_seed_sequence,
    random_powerset_label_min,
)
from pydvl.value.result import ValuationResult
from pydvl.value.shapley.truncated import TruncationPolicy
from pydvl.value.stopping import MaxChecks, StoppingCriterion

logger = logging.getLogger(__name__)

__all__ = ["ClasswiseScorer", "compute_classwise_shapley_values"]


class ClasswiseScorer(Scorer):
    r"""A Scorer designed for evaluation in classification problems. Its value
    is computed from an in-class and an out-of-class "inner score" (Schoch et
    al., 2022) <sup><a href="#schoch_csshapley_2022">1</a></sup>. Let $S$ be the
    training set and $D$ be the valuation set. For each label $c$, $D$ is
    factorized into two disjoint sets: $D_c$ for in-class instances and $D_{-c}$
    for out-of-class instances. The score combines an in-class metric of
    performance, adjusted by a discounted out-of-class metric. These inner
    scores must be provided upon construction or default to accuracy. They are
    combined into:

    $$
    u(S_{y_i}) = f(a_S(D_{y_i}))\ g(a_S(D_{-y_i})),
    $$

    where $f$ and $g$ are continuous, monotonic functions. For a detailed
    explanation, refer to section four of (Schoch et al., 2022)<sup><a
    href="#schoch_csshapley_2022"> 1</a></sup>.

    !!! warning Multi-class support
        Metrics must support multiple class labels if you intend to apply them
        to a multi-class problem. For instance, the metric 'accuracy' supports
        multiple classes, but the metric `f1` does not. For a two-class
        classification problem, using `f1_weighted` is essentially equivalent to
        using `accuracy`.

    Args:
        scoring: Name of the scoring function or a callable that can be passed
            to [Scorer][pydvl.utils.score.Scorer].
        default: Score to use when a model fails to provide a number, e.g. when
            too little was used to train it, or errors arise.
        range: Numerical range of the score function. Some Monte Carlo methods
            can use this to estimate the number of samples required for a
            certain quality of approximation. If not provided, it can be read
            from the `scoring` object if it provides it, for instance if it was
            constructed with
            [compose_score][pydvl.utils.score.compose_score].
        in_class_discount_fn: Continuous, monotonic increasing function used to
            discount the in-class score.
        out_of_class_discount_fn: Continuous, monotonic increasing function used
            to discount the out-of-class score.
        initial_label: Set initial label (for the first iteration)
        name: Name of the scorer. If not provided, the name of the inner scoring
            function will be prefixed by `classwise `.

    !!! tip "New in version 0.7.1"
    """

    def __init__(
        self,
        scoring: Union[str, ScorerCallable] = "accuracy",
        default: float = 0.0,
        range: Tuple[float, float] = (0, 1),
        in_class_discount_fn: Callable[[float], float] = lambda x: x,
        out_of_class_discount_fn: Callable[[float], float] = np.exp,
        initial_label: Optional[int] = None,
        name: Optional[str] = None,
    ):
        disc_score_in_class = in_class_discount_fn(range[1])
        disc_score_out_of_class = out_of_class_discount_fn(range[1])
        transformed_range = (0, disc_score_in_class * disc_score_out_of_class)
        super().__init__(
            scoring=scoring,
            range=transformed_range,
            default=default,
            name=name or f"classwise {str(scoring)}",
        )
        self._in_class_discount_fn = in_class_discount_fn
        self._out_of_class_discount_fn = out_of_class_discount_fn
        self.label = initial_label

    def __str__(self):
        return self._name

    def __call__(
        self: "ClasswiseScorer",
        model: SupervisedModel,
        x_test: NDArray[np.float64],
        y_test: NDArray[np.int_],
    ) -> float:
        (
            in_class_score,
            out_of_class_score,
        ) = self.estimate_in_class_and_out_of_class_score(model, x_test, y_test)
        disc_score_in_class = self._in_class_discount_fn(in_class_score)
        disc_score_out_of_class = self._out_of_class_discount_fn(out_of_class_score)
        return disc_score_in_class * disc_score_out_of_class

    def estimate_in_class_and_out_of_class_score(
        self,
        model: SupervisedModel,
        x_test: NDArray[np.float64],
        y_test: NDArray[np.int_],
        rescale_scores: bool = True,
    ) -> Tuple[float, float]:
        r"""
        Computes in-class and out-of-class scores using the provided inner
        scoring function. The result is

        $$
        a_S(D=\{(x_1, y_1), \dots, (x_K, y_K)\}) = \frac{1}{N} \sum_k s(y(x_k), y_k).
        $$

        In this context, for label $c$ calculations are executed twice: once for $D_c$
        and once for $D_{-c}$ to determine the in-class and out-of-class scores,
        respectively. By default, the raw scores are multiplied by $\frac{|D_c|}{|D|}$
        and $\frac{|D_{-c}|}{|D|}$, respectively. This is done to ensure that both
        scores are of the same order of magnitude. This normalization is particularly
        useful when the inner score function $a_S$ is calculated by an estimator of the
        form $\frac{1}{N} \sum_i x_i$, e.g. the accuracy.

        Args:
            model: Model used for computing the score on the validation set.
            x_test: Array containing the features of the classification problem.
            y_test: Array containing the labels of the classification problem.
            rescale_scores: If set to True, the scores will be denormalized. This is
                particularly useful when the inner score function $a_S$ is calculated by
                an estimator of the form $\frac{1}{N} \sum_i x_i$.

        Returns:
            Tuple containing the in-class and out-of-class scores.
        """
        scorer = self._scorer
        label_set_match = y_test == self.label
        label_set = np.where(label_set_match)[0]
        num_classes = len(np.unique(y_test))

        if len(label_set) == 0:
            return 0, 1 / (num_classes - 1)

        complement_label_set = np.where(~label_set_match)[0]
        in_class_score = scorer(model, x_test[label_set], y_test[label_set])
        out_of_class_score = scorer(
            model, x_test[complement_label_set], y_test[complement_label_set]
        )

        if rescale_scores:
            n_in_class = np.count_nonzero(y_test == self.label)
            n_out_of_class = len(y_test) - n_in_class
            in_class_score *= n_in_class / (n_in_class + n_out_of_class)
            out_of_class_score *= n_out_of_class / (n_in_class + n_out_of_class)

        return in_class_score, out_of_class_score


@deprecated(
    target=True,
    args_mapping={"config": "config"},
    deprecated_in="0.9.0",
    remove_in="0.10.0",
)
def compute_classwise_shapley_values(
    u: Utility,
    *,
    done: StoppingCriterion,
    truncation: TruncationPolicy,
    done_sample_complements: Optional[StoppingCriterion] = None,
    normalize_values: bool = True,
    use_default_scorer_value: bool = True,
    min_elements_per_label: int = 1,
    n_jobs: int = 1,
    parallel_backend: Optional[ParallelBackend] = None,
    config: Optional[ParallelConfig] = None,
    progress: bool = False,
    seed: Optional[Seed] = None,
) -> ValuationResult:
    r"""
    Computes an approximate Class-wise Shapley value by sampling independent
    permutations of the index set for each label and index sets sampled from the
    powerset of the complement (with respect to the currently evaluated label),
    approximating the sum:

    $$
    v_u(i) = \frac{1}{K} \sum_k \frac{1}{L} \sum_l
    [u(\sigma^{(l)}_{:i} \cup \{i\} | S^{(k)} ) − u( \sigma^{(l)}_{:i} | S^{(k)})],
    $$

    where $\sigma_{:i}$ denotes the set of indices in permutation sigma before
    the position where $i$ appears and $S$ is a subset of the index set of all
    other labels (see [the main documentation][intro-to-cw-shapley] for
    details).

    Args:
        u: Utility object containing model, data, and scoring function. The
            scorer must be of type
            [ClasswiseScorer][pydvl.value.shapley.classwise.ClasswiseScorer].
        done: Function that checks whether the computation needs to stop.
        truncation: Callable function that decides whether to interrupt processing a
            permutation and set subsequent marginals to zero.
        done_sample_complements: Function checking whether computation needs to stop.
            Otherwise, it will resample conditional sets until the stopping criterion is
            met.
        normalize_values: Indicates whether to normalize the values by the variation
            in each class times their in-class accuracy.
        done_sample_complements: Number of times to resample the complement set
            for each permutation.
        use_default_scorer_value: The first set of indices is the sampled complement
            set. Unless not otherwise specified, the default scorer value is used for
            this. If it is set to false, the base score is calculated from the utility.
        min_elements_per_label: The minimum number of elements for each opposite
            label.
        n_jobs: Number of parallel jobs to run.
        parallel_backend: Parallel backend instance to use
            for parallelizing computations. If `None`,
            use [JoblibParallelBackend][pydvl.parallel.backends.JoblibParallelBackend] backend.
            See the [Parallel Backends][pydvl.parallel.backends] package
            for available options.
        config: (**DEPRECATED**) Object configuring parallel computation,
            with cluster address, number of cpus, etc.
        progress: Whether to display a progress bar.
        seed: Either an instance of a numpy random number generator or a seed for it.

    Returns:
        ValuationResult object containing computed data values.

    !!! tip "New in version 0.7.1"
    """
    dim_correct = u.data.y_train.ndim == 1 and u.data.y_test.ndim == 1
    is_integral = all(
        map(
            lambda v: isinstance(v, numbers.Integral), (*u.data.y_train, *u.data.y_test)
        )
    )
    if not dim_correct or not is_integral:
        raise ValueError(
            "The supplied dataset has to be a 1-dimensional classification dataset."
        )

    if not isinstance(u.scorer, ClasswiseScorer):
        raise ValueError(
            "Please set a subclass of ClasswiseScorer object as scorer object of the"
            " utility. See scoring argument of Utility."
        )

    parallel_backend = _maybe_init_parallel_backend(parallel_backend, config)
    u_ref = parallel_backend.put(u)
    n_jobs = parallel_backend.effective_n_jobs(n_jobs)
    n_submitted_jobs = 2 * n_jobs

    pbar = tqdm(disable=not progress, position=0, total=100, unit="%")
    algorithm = "classwise_shapley"
    accumulated_result = ValuationResult.zeros(
        algorithm=algorithm, indices=u.data.indices, data_names=u.data.data_names
    )
    terminate_exec = False
    seed_sequence = ensure_seed_sequence(seed)

    parallel_backend = _maybe_init_parallel_backend(parallel_backend, config)

    with parallel_backend.executor(max_workers=n_jobs) as executor:
        pending: Set[Future] = set()
        while True:
            completed_futures, pending = wait(
                pending, timeout=60, return_when=FIRST_COMPLETED
            )
            for future in completed_futures:
                accumulated_result += future.result()
                if done(accumulated_result):
                    terminate_exec = True
                    break

            pbar.n = 100 * done.completion()
            pbar.refresh()
            if terminate_exec:
                break

            n_remaining_slots = n_submitted_jobs - len(pending)
            seeds = seed_sequence.spawn(n_remaining_slots)
            for i in range(n_remaining_slots):
                future = executor.submit(
                    _permutation_montecarlo_classwise_shapley_one_step,
                    u_ref,
                    truncation=truncation,
                    done_sample_complements=done_sample_complements,
                    use_default_scorer_value=use_default_scorer_value,
                    min_elements_per_label=min_elements_per_label,
                    algorithm_name=algorithm,
                    seed=seeds[i],
                )
                pending.add(future)

    result = accumulated_result
    if normalize_values:
        result = _normalize_classwise_shapley_values(result, u)

    return result


def _permutation_montecarlo_classwise_shapley_one_step(
    u: Utility,
    *,
    done_sample_complements: Optional[StoppingCriterion] = None,
    truncation: TruncationPolicy,
    use_default_scorer_value: bool = True,
    min_elements_per_label: int = 1,
    algorithm_name: str = "classwise_shapley",
    seed: Optional[SeedSequence] = None,
) -> ValuationResult:
    """Helper function for [compute_classwise_shapley_values()]
    [pydvl.value.shapley.classwise.compute_classwise_shapley_values].

    Args:
        u: Utility object containing model, data, and scoring function. The
            scorer must be of type [ClasswiseScorer]
            [pydvl.value.shapley.classwise.ClasswiseScorer].
        done_sample_complements: Function checking whether computation needs to stop.
            Otherwise, it will resample conditional sets until the stopping criterion is
            met.
        truncation: Callable function that decides whether to interrupt processing a
            permutation and set subsequent marginals to zero.
        use_default_scorer_value: The first set of indices is the sampled complement
            set. Unless not otherwise specified, the default scorer value is used for
            this. If it is set to false, the base score is calculated from the utility.
        min_elements_per_label: The minimum number of elements for each opposite
            label.
        algorithm_name: For the results object.
        seed: Either an instance of a numpy random number generator or a seed for it.

    Returns:
        ValuationResult object containing computed data values.

    !!! tip "Changed in version 0.9.0"
        Deprecated `config` argument and added a `parallel_backend`
        argument to allow users to pass the Parallel Backend configuration.
    """
    if done_sample_complements is None:
        done_sample_complements = MaxChecks(1)

    result = ValuationResult.zeros(
        algorithm=algorithm_name, indices=u.data.indices, data_names=u.data.data_names
    )
    rng = np.random.default_rng(seed)
    x_train, y_train = u.data.get_training_data(u.data.indices)
    unique_labels = np.unique(y_train)
    scorer = cast(ClasswiseScorer, copy(u.scorer))
    u.scorer = scorer

    for label in unique_labels:
        u.scorer.label = label
        class_indices_set, class_complement_indices_set = _split_indices_by_label(
            u.data.indices, y_train, label
        )
        _, complement_y_train = u.data.get_training_data(class_complement_indices_set)
        indices_permutation = rng.permutation(class_indices_set)
        done_sample_complements.reset()

        for subset_idx, subset_complement in enumerate(
            random_powerset_label_min(
                class_complement_indices_set,
                complement_y_train,
                min_elements_per_label=min_elements_per_label,
                seed=rng,
            )
        ):
            result += _permutation_montecarlo_shapley_rollout(
                u,
                indices_permutation,
                additional_indices=subset_complement,
                truncation=truncation,
                algorithm_name=algorithm_name,
                use_default_scorer_value=use_default_scorer_value,
            )
            if done_sample_complements(result):
                break

    return result


def _normalize_classwise_shapley_values(
    result: ValuationResult, u: Utility
) -> ValuationResult:
    r"""
    Normalize a valuation result specific to classwise Shapley.

    Each value $v_i$ associated with the sample $(x_i, y_i)$ is normalized by
    multiplying it with $a_S(D_{y_i})$ and dividing by $\sum_{j \in D_{y_i}} v_j$. For
    more details, see (Schoch et al., 2022) <sup><a href="#schoch_csshapley_2022">1</a>
    </sup>.

    Args:
        result: ValuationResult object to be normalized.
        u: Utility object containing model, data, and scoring function. The
            scorer must be of type [ClasswiseScorer]
            [pydvl.value.shapley.classwise.ClasswiseScorer].

    Returns:
        Normalized ValuationResult object.
    """
    y_train = u.data.y_train
    unique_labels = np.unique(np.concatenate((y_train, u.data.y_test)))
    scorer = cast(ClasswiseScorer, u.scorer)

    for idx_label, label in enumerate(unique_labels):
        scorer.label = label
        active_elements = y_train == label
        indices_label_set = np.where(active_elements)[0]
        indices_label_set = u.data.indices[indices_label_set]

        u.model.fit(u.data.x_train, u.data.y_train)
        scorer.label = label
        in_class_acc, _ = scorer.estimate_in_class_and_out_of_class_score(
            u.model, u.data.x_test, u.data.y_test
        )

        sigma = np.sum(result.values[indices_label_set])
        if sigma != 0:
            result.scale(in_class_acc / sigma, indices=indices_label_set)

    return result


def _permutation_montecarlo_shapley_rollout(
    u: Utility,
    permutation: NDArray[np.int_],
    truncation: TruncationPolicy,
    algorithm_name: str,
    additional_indices: Optional[NDArray[np.int_]] = None,
    use_default_scorer_value: bool = True,
) -> ValuationResult:
    """
    Represents a truncated version of a permutation-based MC estimator. It iterates over
    all subsets starting from the empty set to the full set of indices as specified by
    `permutation`. For each subset, the marginal contribution is computed and added to
    the result. The computation is interrupted if the truncation policy returns `True`.

    !!! Todo
        Reuse in [permutation_montecarlo_shapley()]
        [pydvl.value.shapley.montecarlo.permutation_montecarlo_shapley]

    Args:
        u: Utility object containing model, data, and scoring function.
        permutation: Permutation of indices to be considered.
        truncation: Callable which decides whether to interrupt processing a
            permutation and set all subsequent marginals to zero.
        algorithm_name: For the results object. Used internally by different
            variants of Shapley using this subroutine
        additional_indices: Set of additional indices for data points which should be
            always considered.
        use_default_scorer_value: Use default scorer value even if additional_indices
            is not None.

    Returns:
         ValuationResult object containing computed data values.
    """
    if (
        additional_indices is not None
        and len(np.intersect1d(permutation, additional_indices)) > 0
    ):
        raise ValueError(
            "The class label set and the complement set have to be disjoint."
        )

    result = ValuationResult.zeros(
        algorithm=algorithm_name, indices=u.data.indices, data_names=u.data.data_names
    )

    prev_score = (
        u.default_score
        if (
            use_default_scorer_value
            or additional_indices is None
            or additional_indices is not None
            and len(additional_indices) == 0
        )
        else u(additional_indices)
    )

    truncation_u = u
    if additional_indices is not None:
        # hack to calculate the correct value in reset.
        truncation_indices = np.sort(np.concatenate((permutation, additional_indices)))
        truncation_u = Utility(
            u.model,
            Dataset(
                u.data.x_train[truncation_indices],
                u.data.y_train[truncation_indices],
                u.data.x_test,
                u.data.y_test,
            ),
            u.scorer,
        )
    truncation.reset(truncation_u)

    is_terminated = False
    for i, idx in enumerate(permutation):
        if is_terminated or (is_terminated := truncation(i, prev_score)):
            score = prev_score
        else:
            score = u(
                np.concatenate((permutation[: i + 1], additional_indices))
                if additional_indices is not None and len(additional_indices) > 0
                else permutation[: i + 1]
            )

        marginal = score - prev_score
        result.update(idx, marginal)
        prev_score = score

    return result


def _split_indices_by_label(
    indices: NDArray[np.int_], labels: NDArray[np.int_], label: int
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Splits the indices into two sets based on the value of `label`, e.g. those samples
    with and without that label.

    Args:
        indices: The indices to be used for referring to the data.
        labels: Corresponding labels for the indices.
        label: Label to be used for splitting.

    Returns:
        Tuple with two sets of indices.
    """
    active_elements = labels == label
    class_indices_set = np.where(active_elements)[0]
    class_complement_indices_set = np.where(~active_elements)[0]
    class_indices_set = indices[class_indices_set]
    class_complement_indices_set = indices[class_complement_indices_set]
    return class_indices_set, class_complement_indices_set
