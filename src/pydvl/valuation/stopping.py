"""
Stopping criteria for value computations.

This module provides a basic set of stopping criteria to be used with valuation
algorithms, in particular all
[semi-values][pydvl.valuation.methods.semivalue.SemivalueValuation]. Common examples are
[MinUpdates][pydvl.valuation.stopping.MinUpdates],
[MaxTime][pydvl.valuation.stopping.MaxTime], or
[HistoryDeviation][pydvl.valuation.stopping.HistoryDeviation].

Stopping criteria can behave in different ways depending on the context. For example,
[MaxUpdates][pydvl.valuation.stopping.MaxUpdates] limits the number of updates to
values, which depending on the algorithm may mean a different number of utility
evaluations or imply other computations like solving a linear or quadratic program.
In the case of [SemivalueValuation][pydvl.valuation.methods.semivalue.SemivalueValuation],
the criteria are evaluated once per batch, which might lead to different behavior
depending on the batch size (e.g. for certain batch_sizes it might happen that the
number of updates to values after convergence is not exactly was required, since
multiple updates might happen at once).

Stopping criteria are callables that are evaluated on a
[ValuationResult][pydvl.valuation.result.ValuationResult] and return a
[Status][pydvl.utils.status.Status] object. They can be combined using boolean
operators.

## Combining stopping criteria

Objects of type [StoppingCriterion][pydvl.valuation.stopping.StoppingCriterion] can
be combined with the binary operators `&` (*and*), and `|` (*or*), following the
truth tables of [Status][pydvl.utils.status.Status]. The unary operator `~`
(*not*) is also supported. See
[StoppingCriterion][pydvl.valuation.stopping.StoppingCriterion] for details on how
these operations affect the behavior of the stopping criteria.


## How convergence is determined

Most stopping criteria keep track of the convergence of each index separately
but make global decisions based on the overall convergence of some fraction of
all indices. For example, if we have a stopping criterion that checks whether
the standard error of 90% of values is below a threshold, then methods will keep
updating **all** indices until 90% of them have converged, irrespective of the
quality of the individual estimates, and *without freezing updates* for indices
along the way as values individually attain low standard error.

This has some practical implications, because some values do tend to converge
sooner than others. For example, assume we use the criterion
`AbsoluteStandardError(0.02) | MaxUpdates(1000)`. Then values close to 0 might
be marked as "converged" rather quickly because they fulfill the first
criterion, say after 20 iterations, despite being poor estimates (see the section on
pitfalls below for commentary on stopping criteria based on standard errors).
Because other indices take longer to reach a low standard error and the criterion is a
global check, the "converged" ones keep being updated and end up being good
estimates. In this case, this has been beneficial, but one might not wish for
converged values to be updated, if one is sure that the criterion is adequate
for individual values.

[Semi-value methods][pydvl.valuation.methods.semivalue.SemivalueValuation] include a
parameter `skip_converged` that allows to skip the computation of values that have
converged. The way to avoid doing this too early is to use a more stringent
check, e.g. `AbsoluteStandardError(1e-3) | MaxUpdates(1000)`. With
`skip_converged=True` this check can still take less time than the first one,
despite requiring more iterations for some indices.

??? Tip "Stoppig criterion for finite samplers"
    Using a finite sampler naturally defines when the valuation algorithm terminates.
    However, in order to properly report progress, we need to use a stopping criterion
    that keeps track of the number of iterations. In this case, one can use
    [NoStopping][pydvl.valuation.stopping.NoStopping] with the sampler as an argument.
    This quirk is due to progress reported depending on the
    [completion][pydvl.valuation.stopping.StoppingCriterion.completion] attribute of
    a criterion. Here's how it's done:

    ```python
    from pydvl.valuation import ShapleyValuation, NoStopping

    sampler = DeterministicUniformSampler()
    stopping = NoStopping(sampler)
    valuation = ShapleyValuation(
        utility=utility, sampler=sampler, is_done=stopping, progress=True
    )
    with parallel_config(n_jobs=4):
        valuation.fit(data)
    ```

## Choosing a stopping criterion

The choice of a stopping criterion greatly depends on the algorithm and the
context. A safe bet is to combine a [MaxUpdates][pydvl.valuation.stopping.MaxUpdates]
or a [MaxTime][pydvl.valuation.stopping.MaxTime] with a
[HistoryDeviation][pydvl.valuation.stopping.HistoryDeviation] or an
[RankCorrelation][pydvl.valuation.stopping.RankCorrelation]. The former
will ensure that the computation does not run for too long, while the latter
will try to achieve results that are stable enough. Note however that if the
threshold is too strict, one will always end up running until a maximum number
of iterations or time. Also keep in mind that different values converge at
different times, so you might want to use tight thresholds and `skip_converged`
as described above for semi-values.


??? Example
    ```python
    from pydvl.valuation import BanzhafValuation, MinUpdates, MSRSampler, RankCorrelation

    model = ... # Some sklearn-compatible model
    scorer = SupervisedScorer("accuracy", test_data, default=0.0)
    utility = ModelUtility(model, scorer)
    sampler = MSRSampler(seed=seed)
    stopping = RankCorrelation(rtol=1e-2, burn_in=32) | MinUpdates(1000)
    valuation = BanzhafValuation(utility=utility, sampler=sampler, is_done=stopping)
    with parallel_config(n_jobs=4):
        valuation.fit(training_data)
    result = valuation.values()
    ```
    This will compute the Banzhaf semivalues for `utility` until either the change in
    Spearman rank correlation between updates is below `1e-2` or `1000` updates have
    been performed. The `burn_in` parameter is used to discard the first `32` updates
    from the computation of the standard error.

!!! Warning
    Be careful not to reuse the same stopping criterion for different computations. The
    object has state, which is reset by `fit()` for some valuation methods, but this is
    **not guaranteed** for all methods. If you need to reuse the same criterion, it's
    safer to create a new instance.

## Interactions with sampling schemes and other pitfalls of stopping criteria

Different samplers define different "update strategies" for values. For example,
[MSRSampler][pydvl.valuation.samplers.msr.MSRSampler] updates the `counts` field
of a [ValuationResult][pydvl.valuation.result.ValuationResult] only for about half of
the utility evaluations, because it reuses samples. This means that a stopping criterion
like [MaxChecks][pydvl.valuation.stopping.MaxChecks] will not work as expected, because
it will count the number of calls to the criterion, not the number of updates to the
values. In this case, one should use [MaxUpdates][pydvl.valuation.stopping.MaxUpdates]
or, more likely, [MinUpdates][pydvl.valuation.stopping.MinUpdates] instead.

Another pitfall is the interaction with the `skip_converged` parameter of
[Semi-value methods][pydvl.valuation.methods.semivalue.SemivalueValuation]. If this is
set to `True`, the stopping criterion should probably be more stringent.

Finally, stopping criteria that rely on the standard error of the values, like
[AbsoluteStandardError][pydvl.valuation.stopping.AbsoluteStandardError], should be
used with care. The standard error is a measure of the uncertainty of the estimate,
but **it does not guarantee that the estimate is close to the true value**. For example,
if the utility function is very noisy, the standard error might be very low, but the
estimate might be far from the true value. In this case, one might want to use a
[RankCorrelation][pydvl.valuation.stopping.RankCorrelation] instead, which checks
whether the rank of the values is stable.


## Creating stopping criteria

In order to create a new stopping criterion, one can subclass
[StoppingCriterion][pydvl.valuation.stopping.StoppingCriterion] and implement the
`_check` method. This method should return a [Status][pydvl.utils.status.Status] and
update the `_converged` attribute, which is a boolean array indicating whether the
value for each index has converged. When this does not make sense for a particular
stopping criterion, [completion][pydvl.valuation.stopping.StoppingCriterion.completion]
should be overridden to provide an overall completion value, since its default
implementation attempts to compute the mean of `_converged`.


## References

[^1]: <a name="ghorbani_data_2019"></a>Ghorbani, A., Zou, J., 2019.
    [Data Shapley: Equitable Valuation of Data for Machine Learning](https://proceedings.mlr.press/v97/ghorbani19c.html).
    In: Proceedings of the 36th International Conference on Machine Learning, PMLR, pp. 2242–2251.
"""

from __future__ import annotations

import abc
import logging
from time import time
from typing import Callable, Generic, Type, TypeVar, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr
from typing_extensions import Self

from pydvl.utils.status import Status
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers.base import IndexSampler

__all__ = [
    "AbsoluteStandardError",
    "HistoryDeviation",
    "MaxChecks",
    "MaxUpdates",
    "MinUpdates",
    "MaxTime",
    "NoStopping",
    "RankCorrelation",
    "StoppingCriterion",
]

logger = logging.getLogger(__name__)


class StoppingCriterion(abc.ABC):
    """A composable callable object to determine whether a computation
    must stop.

    A `StoppingCriterion` is a callable taking a
    [ValuationResult][pydvl.valuation.result.ValuationResult] and returning a
    [Status][pydvl.valuation.result.Status]. It also keeps track of individual
    convergence of values with
    [converged][pydvl.valuation.stopping.StoppingCriterion.converged], and reports
    the overall completion of the computation with
    [completion][pydvl.valuation.stopping.StoppingCriterion.completion].

    Instances of `StoppingCriterion` can be composed with the binary operators
    `&` (*and*), and `|` (*or*), following the truth tables of
    [Status][pydvl.utils.status.Status]. The unary operator `~` (*not*) is
    also supported. These boolean operations act according to the following
    rules:

    - The results of `check()` are combined with the operator. See
      [Status][pydvl.utils.status.Status] for the truth tables.
    - The results of
      [converged][pydvl.valuation.stopping.StoppingCriterion.converged] are combined
      with the operator (returning another boolean array).
    - The [completion][pydvl.valuation.stopping.StoppingCriterion.completion]
      method returns the min, max, or the complement to 1 of the completions of
      the operands, for AND, OR and NOT respectively. This is required for cases
      where one of the criteria does not keep track of the convergence of single
      values, e.g. [MaxUpdates][pydvl.valuation.stopping.MaxUpdates], because
      [completion][pydvl.valuation.stopping.StoppingCriterion.completion] by
      default returns the mean of the boolean convergence array.

    # Subclassing

    Subclassing this class requires implementing a `check()` method that
    returns a [Status][pydvl.utils.status.Status] object based on a given
    [ValuationResult][pydvl.valuation.result.ValuationResult]. This method should
    update the attribute `_converged`, which is a boolean array indicating
    whether the value for each index has converged.
    When this does not make sense for a particular stopping criterion,
    [completion][pydvl.valuation.stopping.StoppingCriterion.completion] should be
    overridden to provide an overall completion value, since its default
    implementation attempts to compute the mean of `_converged`.

    Args:
        modify_result: If `True` the status of the input
            [ValuationResult][pydvl.valuation.result.ValuationResult] is modified in
            place after the call.
    """

    _converged: NDArray[
        np.bool_
    ]  #: A boolean array indicating whether the corresponding element has converged

    def __init__(self, modify_result: bool = True):
        self.modify_result = modify_result
        self._converged = np.full(0, False)
        self._count = 0

    @abc.abstractmethod
    def _check(self, result: ValuationResult) -> Status:
        """Check whether the computation should stop."""
        ...

    @property
    def count(self) -> int:
        """The number of times that the criterion has been checked."""
        return self._count

    def completion(self) -> float:
        """Returns a value between 0 and 1 indicating the completion of the
        computation."""
        if self.converged.size == 0:
            return 0.0
        return float(np.mean(self.converged).item())

    def reset(self) -> Self:
        self._converged = np.full(0, False)
        self._count = 0
        return self

    @property
    def converged(self) -> NDArray[np.bool_]:
        """Returns a boolean array indicating whether the values have converged
        for each data point.

        Inheriting classes must set the `_converged` attribute in their
        `check()`.

        Returns:
            A boolean array indicating whether the values have converged for
            each data point.
        """
        return self._converged

    def __str__(self) -> str:
        return type(self).__name__

    def __call__(self, result: ValuationResult) -> Status:
        """Calls `check()`, maybe updating the result."""
        if len(result) == 0:
            logger.warning(
                "At least one iteration finished but no results where generated. "
                "Please check that your scorer and utility return valid numbers."
            )
        self._count += 1
        if self._converged.size == 0:
            self._converged = np.full_like(result.indices, False, dtype=bool)
        status = self._check(result)
        if self.modify_result:  # FIXME: this is not nice
            result._status = status
        return status

    def __and__(self, other: StoppingCriterion) -> StoppingCriterion:
        def fun(result: ValuationResult) -> Status:
            return self._check(result) & other._check(result)

        return _make_criterion(
            check=fun,
            criteria=[self, other],
            converged=lambda: self.converged & other.converged,
            completion=lambda: min(self.completion(), other.completion()),
            name=f"{str(self)} AND {str(other)}",
        )(modify_result=self.modify_result or other.modify_result)

    def __or__(self, other: StoppingCriterion) -> StoppingCriterion:
        def fun(result: ValuationResult) -> Status:
            return self._check(result) | other._check(result)

        return _make_criterion(
            check=fun,
            criteria=[self, other],
            converged=lambda: self.converged | other.converged,
            completion=lambda: max(self.completion(), other.completion()),
            name=f"{str(self)} OR {str(other)}",
        )(modify_result=self.modify_result or other.modify_result)

    def __invert__(self) -> StoppingCriterion:
        def fun(result: ValuationResult) -> Status:
            return ~self._check(result)

        return _make_criterion(
            check=fun,
            criteria=[self],
            converged=lambda: ~self.converged,
            completion=lambda: 1 - self.completion(),
            name=f"NOT {str(self)}",
        )(modify_result=self.modify_result)


def _make_criterion(
    check: Callable[[ValuationResult], Status],
    criteria: list[StoppingCriterion],
    converged: Callable[[], NDArray[np.bool_]],
    completion: Callable[[], float],
    name: str,
) -> Type[StoppingCriterion]:
    """Create a new [StoppingCriterion][pydvl.valuation.stopping.StoppingCriterion] from
    several callables. Used to compose simpler criteria with bitwise operators

    Args:
        check: The callable to wrap.
        criteria: A list of criteria that are combined into a new criterion
        converged: A callable that returns a boolean array indicating what
            values have converged.
        completion: A callable that returns a value between 0 and 1 indicating
            the rate of completion of the computation. If not provided, the fraction
            of converged values is used.
        name: The name of the new criterion. If `None`, the `__name__` of
            the function is used.

    Returns:
        A new subclass of [StoppingCriterion][pydvl.valuation.stopping.StoppingCriterion].
    """

    class WrappedCriterion(StoppingCriterion):
        def __init__(self, modify_result: bool = True):
            super().__init__(modify_result=modify_result)
            self._name = name or cast(
                str, getattr(check, "__name__", "WrappedCriterion")
            )
            self._criteria = criteria if criteria is not None else []

        @property
        def criteria(self) -> list[StoppingCriterion]:
            return self._criteria

        def increase_criteria_count(self):
            for criterion in self._criteria:
                if hasattr(criterion, "_criteria"):
                    cast(WrappedCriterion, criterion).increase_criteria_count()
                else:
                    criterion._count += 1

        def init_criteria_converged(self, result: ValuationResult):
            for criterion in self._criteria:
                if hasattr(criterion, "_criteria"):
                    cast(WrappedCriterion, criterion).init_criteria_converged(result)
                else:
                    if criterion._converged.size == 0:
                        criterion._converged = np.full_like(
                            result.indices, False, dtype=bool
                        )

        def __call__(self, result: ValuationResult) -> Status:
            self.increase_criteria_count()
            self.init_criteria_converged(result)
            return super().__call__(result)

        def _check(self, result: ValuationResult) -> Status:
            status = check(result)
            self._converged = converged()
            return status

        def __str__(self) -> str:
            return self._name

        def completion(self) -> float:
            return completion()

    return WrappedCriterion


class AbsoluteStandardError(StoppingCriterion):
    r"""Determine convergence based on the standard error of the values.

    If $s_i$ is the standard error for datum $i$, then this criterion returns
    [Converged][pydvl.utils.status.Status] if $s_i < \epsilon$ for all $i$ and a
    threshold value $\epsilon \gt 0$.

    !!! Warning
        This criterion should be used with care. The standard error is a measure of the
        uncertainty of the estimate, but **it does not guarantee that the estimate is
        close to the true value**. For example, if the utility function is very noisy,
        the standard error might be very low, but the estimate might be far from the
        true value. In this case, one might want to use a
        [RankCorrelation][pydvl.valuation.stopping.RankCorrelation] instead, which
        checks whether the rank of the values is stable.

    Args:
        threshold: A value is considered to have converged if the standard
            error is below this threshold. A way of choosing it is to pick some
            percentage of the range of the values. For Shapley values this is
            the difference between the maximum and minimum of the utility
            function (to see this substitute the maximum and minimum values of
            the utility into the marginal contribution formula).
        fraction: The fraction of values that must have converged for the
            criterion to return [Converged][pydvl.utils.status.Status].
        burn_in: The number of iterations to ignore before checking for
            convergence. This is required because computations typically start
            with zero variance, as a result of using
            [zeros()][pydvl.valuation.result.ValuationResult.zeros]. The default is
            set to an arbitrary minimum which is usually enough but may need to
            be increased.
        modify_result: If `True` the status of the input
            [ValuationResult][pydvl.valuation.result.ValuationResult] is modified in
            place after the call.
    """

    def __init__(
        self,
        threshold: float,
        fraction: float = 1.0,
        burn_in: int = 4,
        modify_result: bool = True,
    ):
        super().__init__(modify_result=modify_result)
        self.threshold = threshold
        self.fraction = fraction
        self.burn_in = burn_in

    def _check(self, result: ValuationResult) -> Status:
        self._converged = (result.stderr < self.threshold) & (
            result.counts > self.burn_in
        )
        if np.mean(self._converged) >= self.fraction:
            return Status.Converged
        return Status.Pending

    def __str__(self) -> str:
        return f"AbsoluteStandardError(threshold={self.threshold}, fraction={self.fraction}, burn_in={self.burn_in})"


class MaxChecks(StoppingCriterion):
    """Terminate as soon as the number of checks exceeds the threshold.

    A "check" is one call to the criterion. Note that this might have different
    interpretations depending on the sampler. For example,
    [MSRSampler][pydvl.valuation.samplers.msr.MSRSampler] performs a single
    utility evaluation to update all indices, so that's `len(training_data)` checks for
    a single training of the model. But it also only changes the `counts` field of the
    [ValuationResult][pydvl.valuation.result.ValuationResult] for about half of the
    indices, which is what e.g. [MaxUpdates][pydvl.valuation.stopping.MaxUpdates] checks.


    Args:
        n_checks: Threshold: if `None`, no _check is performed,
            effectively creating a (never) stopping criterion that always returns
            `Pending`.
        modify_result: If `True` the status of the input
            [ValuationResult][pydvl.valuation.result.ValuationResult] is modified in
            place after the call.
    """

    def __init__(self, n_checks: int | None, modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        if n_checks is not None and n_checks < 1:
            raise ValueError("n_iterations must be at least 1 or None")
        self.n_checks = n_checks or np.inf

    def _check(self, result: ValuationResult) -> Status:
        if self._count >= self.n_checks:
            self._converged = np.full_like(result.indices, True, dtype=bool)
            return Status.Converged
        return Status.Pending

    def completion(self) -> float:
        if self.n_checks:
            return min(1.0, self._count / self.n_checks)
        return 0.0

    def __str__(self) -> str:
        return f"MaxChecks(n_checks={self.n_checks})"


class MaxUpdates(StoppingCriterion):
    """Terminate if any number of value updates exceeds or equals the given
    threshold.

    !!! Note
        If you want to ensure that **all** values have been updated, you
        probably want [MinUpdates][pydvl.valuation.stopping.MinUpdates] instead.

    This checks the `counts` field of a
    [ValuationResult][pydvl.valuation.result.ValuationResult], i.e. the number of
    times that each index has been updated. For powerset samplers, the maximum
    of this number coincides with the maximum number of subsets sampled. For
    permutation samplers, it coincides with the number of permutations sampled.

    Args:
        n_updates: Threshold: if `None`, no _check is performed,
            effectively creating a (never) stopping criterion that always returns
            `Pending`.
        modify_result: If `True` the status of the input
            [ValuationResult][pydvl.valuation.result.ValuationResult] is modified in
            place after the call.
    """

    def __init__(self, n_updates: int | None, modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        if n_updates is not None and n_updates < 1:
            raise ValueError("n_updates must be at least 1 or None")
        self.n_updates = n_updates
        self.last_max = 0

    def _check(self, result: ValuationResult) -> Status:
        if result.counts.size == 0:
            return Status.Pending
        if self.n_updates:
            self._converged = result.counts >= self.n_updates
            try:
                self.last_max = int(np.max(result.counts))
                if self.last_max >= self.n_updates:
                    return Status.Converged
            except ValueError:  # empty counts array. This should not happen
                pass
        return Status.Pending

    def completion(self) -> float:
        if self.n_updates:
            return self.last_max / self.n_updates
        return 0.0

    def reset(self) -> Self:
        self.last_max = 0
        return super().reset()

    def __str__(self) -> str:
        return f"MaxUpdates(n_updates={self.n_updates})"


class NoStopping(StoppingCriterion):
    """Keep running forever or until sampling stops.

    If a sampler instance is passed, and it is a finite sampler, its counter will be
    used to update completion status.

    Args:
        sampler: A sampler instance to use for completion status.
        modify_result: If `True` the status of the input
            [ValuationResult][pydvl.valuation.result.ValuationResult] is modified in
            place after the call
    """

    def __init__(self, sampler: IndexSampler | None = None, modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        self.sampler = sampler

    def _check(self, result: ValuationResult) -> Status:
        if self.sampler is not None:
            try:
                if self.sampler.n_samples >= len(self.sampler):
                    self._converged = np.full_like(result.indices, True, dtype=bool)
                    return Status.Converged
            except TypeError:  # Sampler has no len()
                pass
        return Status.Pending

    def completion(self) -> float:
        if self.sampler is None:
            return 0.0
        try:
            return self.sampler.n_samples / len(self.sampler)
        except TypeError:  # Sampler has no length
            return 0.0

    def __str__(self) -> str:
        return "NoStopping()"


class MinUpdates(StoppingCriterion):
    """Terminate as soon as all value updates exceed or equal the given threshold.

    This checks the `counts` field of a
    [ValuationResult][pydvl.valuation.result.ValuationResult], i.e. the number of times
    that each index has been updated. For powerset samplers, the minimum of this number
    is a lower bound for the number of subsets sampled. For permutation samplers, it
    lower-bounds the amount of permutations sampled.

    Args:
        n_updates: Threshold: if `None`, no _check is performed,
            effectively creating a (never) stopping criterion that always returns
            `Pending`.
        modify_result: If `True` the status of the input
            [ValuationResult][pydvl.valuation.result.ValuationResult] is modified in
            place after the call.
    """

    def __init__(self, n_updates: int | None, modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        self.n_updates = n_updates
        self.last_min = 0
        self._actual_completion = 0.0

    def _check(self, result: ValuationResult) -> Status:
        if result.counts.size == 0:
            return Status.Pending
        if self.n_updates is not None:
            self._converged = result.counts >= self.n_updates
            progress = np.clip(result.counts, 0, self.n_updates) / self.n_updates
            self._actual_completion = float(np.mean(progress))

            self.last_min = int(np.min(result.counts))
            if self.last_min >= self.n_updates:
                return Status.Converged
        return Status.Pending

    def completion(self) -> float:
        return self._actual_completion

    def reset(self) -> Self:
        self.last_min = 0
        self._actual_completion = 0.0
        return super().reset()

    def __str__(self) -> str:
        return f"MinUpdates(n_updates={self.n_updates})"


class MaxTime(StoppingCriterion):
    """Terminate if the computation time exceeds the given number of seconds.

    Checks the elapsed time *since construction*.

    Args:
        seconds: Threshold: The computation is terminated if the elapsed time
            between object construction and a _check exceeds this value. If `None`,
            no _check is performed, effectively creating a (never) stopping criterion
            that always returns `Pending`.
        modify_result: If `True` the status of the input
            [ValuationResult][pydvl.valuation.result.ValuationResult] is modified in
            place after the call.
    """

    def __init__(self, seconds: float | None, modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        self.max_seconds = seconds or np.inf
        if self.max_seconds <= 0:
            raise ValueError("Number of seconds for MaxTime must be positive or None")
        self.start = time()

    def _check(self, result: ValuationResult) -> Status:
        if time() > self.start + self.max_seconds:
            self._converged.fill(True)
            return Status.Converged
        return Status.Pending

    def completion(self) -> float:
        if self.max_seconds is None:
            return 0.0
        return float(np.clip((time() - self.start) / self.max_seconds, 0.0, 1.0))

    def reset(self) -> Self:
        self.start = time()
        return super().reset()

    def __str__(self) -> str:
        return f"MaxTime(seconds={self.max_seconds})"


DT = TypeVar("DT", bound=np.generic)


class RollingMemory(Generic[DT]):
    """A simple rolling memory for the last `n_steps` values of each index.

    Updating the memory results in new values are copied to the last column of the
    matrix and old ones removed from the first.

    Args:
        n_steps: The number of steps to remember.
        default: The default value to use when the memory is empty.
    """

    def __init__(
        self,
        n_steps: int,
        default: Union[DT, int, float],
        *,
        dtype: Type[DT] | None = None,
    ):
        if not isinstance(default, np.generic):  # convert to np scalar
            default = cast(DT, np.array(default, dtype=np.result_type(default))[()])
        if dtype is not None:  # user forced conversion
            default = dtype(default)
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        self.n_steps = n_steps
        self._count = 0
        self._default = cast(DT, default)
        self._data: NDArray[DT] = np.full(0, default, dtype=type(default))

    @property
    def count(self) -> int:
        return self._count

    @property
    def data(self) -> NDArray[DT]:
        view = self._data.view()
        view.setflags(write=False)
        return view.T

    def reset(self) -> Self:
        self._data = np.full(0, self._default, dtype=type(self._default))
        return self

    def update(self, values: NDArray[DT]) -> Self:
        """Update the memory with the values of the current result.

        The values are appended to the memory as its last column, and the oldest values
        (the first column) are removed
        """
        if len(self._data) == 0:
            self._data = np.full(
                (len(values), self.n_steps),
                self._default,
                dtype=type(self._default),
            )
        self._data = np.concatenate([self._data[:, 1:], values.reshape(-1, 1)], axis=1)
        self._count += 1
        return self

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self) -> int:
        return self._data.shape[1] if self._data.size > 0 else 0


class HistoryDeviation(StoppingCriterion):
    r"""A simple check for relative distance to a previous step in the computation.

    The method used by Ghorbani and Zou, (2019)<sup><a
    href="#ghorbani_data_2019">1</a></sup> computes the relative distances between the
    current values $v_i^t$ and the values at the previous checkpoint $v_i^{t-\tau}$. If
    the sum is below a given threshold, the computation is terminated.

    $$\sum_{i=1}^n \frac{\left| v_i^t - v_i^{t-\tau} \right|}{v_i^t} < \epsilon.$$

    When the denominator is zero, the summand is set to the value of $v_i^{ t-\tau}$.

    This implementation is slightly generalised to allow for different number of
    updates to individual indices, as happens with powerset samplers instead of
    permutations. Every subset of indices that is found to converge can be pinned to
    that state. Once all indices have converged the method has converged.

    !!! Warning
        This criterion is meant for the reproduction of the results in the paper,
        but we do not recommend using it in practice.

    Args:
        n_steps: Compare values after so many steps. A step is one evaluation of the
            criterion, which happens once per batch.
        rtol: Relative tolerance for convergence ($\epsilon$ in the formula).
        pin_converged: If `True`, once an index has converged, it is pinned
    """

    def __init__(
        self,
        n_steps: int,
        rtol: float,
        pin_converged: bool = True,
        modify_result: bool = True,
    ):
        super().__init__(modify_result=modify_result)
        if rtol <= 0 or rtol >= 1:
            raise ValueError("rtol must be in (0, 1)")

        self.memory = RollingMemory(n_steps + 1, default=np.inf, dtype=np.float64)
        self.rtol = rtol
        self.update_op = np.logical_or if pin_converged else np.logical_and

    def _check(self, r: ValuationResult) -> Status:
        if r.values.size == 0:
            return Status.Pending
        self.memory.update(r.values)
        if self.memory.count < self.memory.n_steps:  # Memory not full yet
            return Status.Pending

        # Look at indices that have been updated more than n_steps times
        ii = r.counts > self.memory.n_steps
        if sum(ii) > 0:
            curr = self.memory[-1]
            saved = self.memory[0]
            diffs = np.abs(curr[ii] - saved[ii])
            quots = np.divide(diffs, curr[ii], out=diffs, where=curr[ii] != 0)
            # if np.any(~np.isfinite(quots)):
            #     warnings.warn("HistoryDeviation memory contains non-finite entries")
            # quots holds the quotients when the denominator is non-zero, and
            # the absolute difference, which is just the memory, otherwise.
            if len(quots) > 0 and np.mean(quots) < self.rtol:
                self._converged = self.update_op(
                    self._converged, r.counts > self.memory.n_steps
                )  # type: ignore
                if np.all(self._converged):
                    return Status.Converged
        return Status.Pending

    def reset(self) -> Self:
        self.memory.reset()
        return super().reset()

    def __str__(self) -> str:
        return f"HistoryDeviation(n_steps={self.memory.n_steps}, rtol={self.rtol})"


class RankCorrelation(StoppingCriterion):
    r"""A check for stability of Spearman correlation between checks.

    Convergence is reached when the change in rank correlation between two successive
    iterations is below a given threshold.

    This criterion is used in (Wang et al.)<sup><a href="wang_data_2023">2</a></sup>.

    !!! Info "The meaning of _successive iterations_"
        Stopping criteria in pyDVL are typically evaluated after each batch of value
        updates is received. This can imply very different things, depending on the
        configuration of the samplers. For this reason, `RankCorrelation` keeps itself
        track of the number of updates that each index has seen, and only checks for
        correlation changes when a given fraction of all indices has been updated more
        than `burn_in` times **and** once since last time the criterion was checked.

    Args:
        rtol: Relative tolerance for convergence ($\epsilon$ in the formula)
        burn_in: The minimum number of updates an index must have seen before checking
            for convergence. This is required because the first correlation checks are
            usually meaningless.
        fraction: The fraction of values that must have been updated between two
            correlation checks. This is to avoid comparing two results where only one
            value has been updated, which would have almost perfect rank correlation.
        modify_result: If `True`, the status of the input
            [ValuationResult][pydvl.value.result.ValuationResult] is modified in
            place after the call.

    !!! tip "Added in 0.9.0"
    !!! tip "Changed in 0.10.0"
        The behaviour of the `burn_in` parameter was changed to look at value updates.
        The parameter `fraction` was added.
    """

    def __init__(
        self,
        rtol: float,
        burn_in: int,
        fraction: float = 1.0,
        modify_result: bool = True,
    ):
        super().__init__(modify_result=modify_result)
        if rtol < 0 or rtol > 1:
            raise ValueError("rtol must be in [0, 1]")
        if fraction <= 0 or fraction > 1:
            raise ValueError("fraction must be in (0, 1]")

        self.rtol = rtol
        self.burn_in = burn_in
        self.fraction = fraction
        self.memory = RollingMemory(n_steps=2, default=np.nan, dtype=np.float64)
        self.count_memory = RollingMemory(n_steps=2, default=0, dtype=np.int_)
        self._corr = np.nan
        self._completion = 0.0

    def _check(self, r: ValuationResult) -> Status:
        if r.values.size == 0:
            return Status.Pending
        # The first update is typically of constant values, so that the Spearman
        # correlation is undefined. We need to wait for the second update.
        if self.memory.count < 1:
            self.memory.update(r.values)
            self.count_memory.update(r.counts)
            self._converged = np.full_like(r.indices, False, dtype=bool)
            return Status.Pending

        self.count_memory.update(r.counts)
        self.memory.update(r.values)

        burnt = np.asarray(self.count_memory[-1] > self.burn_in)
        valid_updated = np.asarray(
            self.count_memory[-1][burnt] - self.count_memory[0][burnt] > 0
        )
        if valid_updated.sum() / len(r) >= self.fraction:
            corr = spearmanr(
                self.memory[0][valid_updated],  # type: ignore
                self.memory[-1][valid_updated],  # type: ignore
            )[0]
            self._update_completion(corr)
            if np.isclose(corr, self._corr, rtol=self.rtol):
                self._converged = np.full(len(r), True)
                logger.debug(
                    f"RankCorrelation has converged with {corr=} for "
                    f"{valid_updated.sum()} values in iteration {self._count}"
                )
                return Status.Converged

            self._corr = corr

        return Status.Pending

    def _update_completion(self, corr: float) -> None:
        if np.isnan(corr) or np.isnan(self._corr):
            self._completion = 0.0
        elif not np.isclose(corr, self._corr, rtol=self.rtol):
            if self._corr == 0.0:
                self._completion = 0.0
            else:
                self._completion = np.abs(corr - self._corr) / self._corr
        else:
            self._completion = 1.0

    def completion(self) -> float:
        return self._completion

    def reset(self) -> Self:
        self.memory.reset()
        self._corr = 0.0
        self._completion = 0.0
        return super().reset()

    def __str__(self):
        return f"RankCorrelation({self.rtol=}, {self.burn_in=}, {self.fraction=})"
