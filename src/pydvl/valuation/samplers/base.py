"""
Base classes for samplers and evaluation strategies.

Read [here][pydvl.valuation.samplers] for an architectural overview of the
samplers and their evaluation strategies.

For an explanation of the interactions between sampler weights, semi-value coefficients
and importance sampling, read [[semi-values-sampling]].
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Generic, Protocol, TypeVar

import numpy as np
from more_itertools import chunked

from pydvl.utils import log_running_moments
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.types import (
    BatchGenerator,
    IndexSetT,
    NullaryPredicate,
    SampleBatch,
    SampleGenerator,
    ValueUpdate,
    ValueUpdateT,
)
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["EvaluationStrategy", "IndexSampler", "ResultUpdater"]


# Sequence.register(np.ndarray)  # <- Doesn't seem to work

logger = logging.getLogger(__name__)


class ResultUpdater(Protocol[ValueUpdateT]):
    """Protocol for result updaters.

    A result updater is a strategy to update a valuation result with a value update.
    """

    def __init__(self, result: ValuationResult): ...

    def __call__(self, update: ValueUpdateT) -> ValuationResult: ...


class IndexSampler(ABC, Generic[ValueUpdateT]):
    r"""Samplers are custom iterables over batches of subsets of indices.

    Calling `from_indices(indexset)` on a sampler returns a generator over **batches**
    of `Samples`. A [Sample][pydvl.valuation.types.Sample] is a tuple of the form
    $(i, S)$, where $i$ is an index of interest, and $S \subset I \setminus \{i\}$ is a
    subset of the complement of $i$ in $I$.

    !!! Note
        Samplers are **not** iterators themselves, so that each call to
        `from_indices(data)` e.g. in a new for loop creates a new iterator.

    Derived samplers must implement
    [log_weight()][pydvl.valuation.samplers.base.IndexSampler.log_weight] and
    [generate()][pydvl.valuation.samplers.base.IndexSampler.generate]. See the
    module's documentation for more on these.

    ## Interrupting samplers

    Calling [interrupt()][pydvl.valuation.samplers.base.IndexSampler.interrupt] on a
    sampler will stop the batched generator after the current batch has been yielded.

    Args:
        batch_size: The number of samples to generate per batch. Batches are
            processed by
            [EvaluationStrategy][pydvl.valuation.samplers.base.EvaluationStrategy]
            so that individual valuations in batch are guaranteed to be received in the
            right sequence.

    ??? Example
        ``` pycon
        >>>from pydvl.valuation.samplers import DeterministicUniformSampler
        >>>import numpy as np
        >>>sampler = DeterministicUniformSampler()
        >>>for idx, s in sampler.generate_batches(np.arange(2)):
        >>>    print(s, end="")
        [][2,][][1,]
        ```
    """

    def __init__(self, batch_size: int = 1):
        """
        Args:
            batch_size: The number of samples to generate per batch. Batches are
                processed by the
                [EvaluationStrategy][pydvl.valuation.samplers.base.EvaluationStrategy]
        """
        self._batch_size = batch_size
        self._n_samples = 0
        self._interrupted = False
        self._skip_indices = np.empty(0, dtype=bool)
        self._len: int | None = None

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @n_samples.setter
    def n_samples(self, n: int) -> None:
        raise AttributeError("Cannot reset a sampler's number of samples")

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        if value < 1:
            raise ValueError("batch_size must be at least 1")
        self._batch_size = value

    @property
    def skip_indices(self) -> IndexSetT:
        """Indices being skipped in the sampler. The exact behaviour will be
        sampler-dependent, so that setting this property is disabled by default."""
        return self._skip_indices

    @skip_indices.setter
    def skip_indices(self, indices: IndexSetT):
        """Sets the indices to skip in the sampler. Since this requires different
        implementations and may  affect expected statistical properties of samplers, it
        is deactivated by default. Samplers must explicitly override the setter to
        signal that they support skipping indices.
        """
        warnings.warn(f"Cannot skip indices in {self.__class__.__name__}.")

    def interrupt(self):
        """Signals the sampler to stop generating samples after the current batch."""
        self._interrupted = True

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def __len__(self) -> int:
        """Returns the length of the current sample generation in generate_batches.

        Raises:
            `TypeError`: if the sampler is infinite or
                [generate_batches][pydvl.valuation.samplers.IndexSampler.generate_batches]
                has not been called yet.
        """
        if self._len is None:
            raise TypeError(f"This {self.__class__.__name__} has no length")
        return self._len

    def from_data(self, data: Dataset) -> BatchGenerator:
        return self.generate_batches(data.indices)

    def generate_batches(self, indices: IndexSetT) -> BatchGenerator:
        """Batches the samples and yields them."""
        self._len = self.sample_limit(indices)

        # Create an empty generator if the indices are empty: `return` acts like a
        # `break`, and produces an empty generator.
        if len(indices) == 0:
            return

        self._interrupted = False
        self._n_samples = 0
        for batch in chunked(self.generate(indices), self.batch_size):
            self._n_samples += len(batch)
            yield batch
            if self._interrupted:
                break

    @abstractmethod
    def sample_limit(self, indices: IndexSetT) -> int | None:
        """Number of samples that can be generated from the indices.

        Args:
            indices: The indices used in the sampler.

        Returns:
            The maximum number of samples that will be generated, or  `None` if the
                number of samples is infinite. This will depend, among other things,
                on the type of [IndexIteration][pydvl.valuation.samplers.IndexIteration].
        """
        ...

    @abstractmethod
    def generate(self, indices: IndexSetT) -> SampleGenerator:
        """Generates single samples.

        `IndexSampler.generate_batches()` will batch these samples according to the
        batch size set upon construction.

        Args:
            indices:

        Yields:
            A tuple (idx, subset) for each sample.
        """
        ...

    @abstractmethod
    def log_weight(self, n: int, subset_len: int) -> float:
        r"""Factor by which to multiply Monte Carlo samples, so that the mean converges
        to the desired expression.

        For details on weighting, importance sampling and usage with semi-values, see
        [[semi-values-sampling]].

        !!! Info "Log-space computation"
            Because the weight is a probability that can be arbitrarily small, we
            compute it in log-space for numerical stability.

        Args:
            n: The size of the index set. Note that the actual size of the set being
                sampled will often be n-1, as one index might be removed from the set.
                See [IndexIteration][pydvl.valuation.samplers.IndexIteration] for more.
            subset_len: The size of the subset being sampled

        Returns:
            The natural logarithm of the probability of sampling a set of the given
                size, when the index set has size `n`, under the
                [IndexIteration][pydvl.valuation.samplers.IndexIteration] given upon
                construction.
        """
        ...

    @abstractmethod
    def make_strategy(
        self,
        utility: UtilityBase,
        log_coefficient: Callable[[int, int], float] | None = None,
    ) -> EvaluationStrategy:
        """Returns the strategy for this sampler.

        Args:
            utility: The utility to use for the evaluation strategy.
            log_coefficient: An additional coefficient to multiply marginals with. This
                depends on the valuation method, hence the delayed setup. If `None`,
                the default is **to apply no correction**. This makes the sampling
                probabilities of the sampler the effective valuation coefficient in the
                limit.
        """
        ...  # return SomeLogEvaluationStrategy(self)

    def result_updater(self, result: ValuationResult) -> ResultUpdater[ValueUpdateT]:
        """Returns a callable that updates a valuation result with a value update.

        Because we use log-space computation for numerical stability, the default result
        updater keeps track of several quantities required to maintain accurate running
        1st and 2nd moments.

        Args:
            result: The result to update
        Returns:
            A callable object that updates the result with a value update
        """
        return LogResultUpdater(result)


class LogResultUpdater(ResultUpdater[ValueUpdateT]):
    """Updates a valuation result with a value update in log-space."""

    def __init__(self, result: ValuationResult):
        self.result = result
        self._log_sum_positive = np.full_like(result.values, -np.inf)
        self._log_sum_negative = np.full_like(result.values, -np.inf)
        self._log_sum2 = np.full_like(result.values, -np.inf)

    def __call__(self, update: ValueUpdate) -> ValuationResult:
        assert update.idx is not None

        try:
            loc = self.result.positions([update.idx]).item()
        except KeyError:
            raise IndexError(f"Index {update.idx} not found in ValuationResult")

        item = self.result[loc]

        new_val, new_var, log_sum_pos, log_sum_neg, log_sum2 = log_running_moments(
            self._log_sum_positive[loc],
            self._log_sum_negative[loc],
            self._log_sum2[loc],
            item.count or 0,
            update.log_update,
            new_sign=update.sign,
            unbiased=True,
        )
        self._log_sum_positive[loc] = log_sum_pos
        self._log_sum_negative[loc] = log_sum_neg
        self._log_sum2[loc] = log_sum2

        item.value = new_val
        item.variance = new_var
        item.count = item.count + 1 if item.count is not None else 1
        self.result[loc] = item
        return self.result


SamplerT = TypeVar("SamplerT", bound=IndexSampler)


class EvaluationStrategy(ABC, Generic[SamplerT, ValueUpdateT]):
    """An evaluation strategy for samplers.

    Implements the processing strategy for batches returned by an
    [IndexSampler][pydvl.valuation.samplers.IndexSampler].

    Different sampling schemes require different strategies for the evaluation of the
    utilities. For instance permutations generated by
    [PermutationSampler][pydvl.valuation.samplers.PermutationSampler] must be evaluated
    in sequence to save computation, see
    [PermutationEvaluationStrategy][pydvl.valuation.samplers.permutation.PermutationEvaluationStrategy].

    This class defines the common interface.

    ??? Example "Usage pattern in valuation methods"
        ```python
        def fit(self, data: Dataset):
            self.utility = self.utility.with_dataset(data)
            strategy = self.sampler.strategy(self.utility, self.log_coefficient)
            delayed_batches = Parallel()(
                delayed(strategy.process)(batch=list(batch), is_interrupted=flag)
                for batch in self.sampler
            )
            for batch in delayed_batches:
                for evaluation in batch:
                    self.result.update(evaluation.idx, evaluation.update)
                if self.is_done(self.result):
                    flag.set()
                    break
        ```

    Args:
        sampler: Required to set up some strategies.
        utility: Required to set up some strategies and to process the samples. Since
            this contains the training data, it is expensive to pickle and send to
            workers.
        log_coefficient: An additional coefficient to multiply marginals with. This
            depends on the valuation method, hence the delayed setup. If `None`, the
            default is **to apply no correction**. This makes the sampling probabilities
            of the sampler the effective valuation coefficient in the limit.
    """

    def __init__(
        self,
        sampler: SamplerT,
        utility: UtilityBase,
        log_coefficient: Callable[[int, int], float] | None = None,
    ):
        self.utility = utility
        # Used by the decorator suppress_warnings:
        self.show_warnings = getattr(utility, "show_warnings", False)
        self.n_indices = (
            len(utility.training_data) if utility.training_data is not None else 0
        )

        if log_coefficient is not None:

            def correction_fun(n: int, subset_len: int) -> float:
                return log_coefficient(n, subset_len) - sampler.log_weight(
                    n, subset_len
                )

            self.log_correction = correction_fun
        else:
            self.log_correction = lambda n, subset_len: 0.0

    @abstractmethod
    def process(
        self, batch: SampleBatch, is_interrupted: NullaryPredicate
    ) -> list[ValueUpdateT]:
        """Processes batches of samples using the evaluator, with the strategy
        required for the sampler.

        !!! Warning
            This method is intended to be used by the evaluator to process the samples
            in one batch, which means it might be sent to another process. Be careful
            with the objects you use here, as they will be pickled and sent over the
            wire.

        Args:
            batch: A batch of samples to process.
            is_interrupted: A predicate that returns True if the processing should be
                interrupted.

        Yields:
            Updates to values as tuples (idx, update)
        """
        ...
