"""
This module implements Maximum Sample Re-use (MSR) sampling for valuation, as described
in Wang and Jia (2023)[^1], where it was introduced specifically for
[Data Banhzaf][pydvl.valuation.methods.data_banzhaf].

!!! Warning
    MSR can be very unstable when used with valuation algorithms other than
    [Data Banzhaf][pydvl.valuation.methods.data_banzhaf]. This is because of the
    instabilities introduced by the correction coefficients. For more, see Appendix C.1
    of Wang and Jia (2023)[^1].

The idea behind MSR is to update all indices in the dataset with every evaluation of the
utility function on a sample. Updates are divided into positive, if the index is in the
sample, and negative, if it is not. The two running means are later combined into a
final result.

Note that this requires defining a special evaluation strategy and result updater, as
returned by the [make_strategy][pydvl.valuation.samplers.msr.MSRSampler.make_strategy]
and [result_updater][pydvl.valuation.samplers.msr.MSRSampler.result_updater] methods,
respectively.

For more on the general architecture of samplers see
[pydvl.valuation.samplers][pydvl.valuation.samplers].


## References

[^1]: <a name="wang_data_2023"></a>Wang, J.T. and Jia, R., 2023.
    [Data Banzhaf: A Robust Data Valuation Framework for Machine Learning](
    https://proceedings.mlr.press/v206/wang23e.html).
    In: Proceedings of The 26th International Conference on Artificial Intelligence and
    Statistics, pp. 6388-6421.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from pydvl.utils.functional import suppress_warnings
from pydvl.utils.numeric import random_subset
from pydvl.utils.types import Seed
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers.base import (
    EvaluationStrategy,
    IndexSampler,
    LogResultUpdater,
    ResultUpdater,
)
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.valuation.types import (
    IndexSetT,
    IndexT,
    NullaryPredicate,
    Sample,
    SampleBatch,
    SampleGenerator,
    ValueUpdate,
)
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["MSRSampler"]


@dataclass(frozen=True)
class MSRValueUpdate(ValueUpdate):
    """Update for Maximum Sample Re-use (MSR) valuation (in log space).

    Attributes:
        in_sample: Whether the index to be updated was in the sample.
    """

    in_sample: bool

    def __init__(self, idx: IndexT, log_update: float, sign: int, in_sample: bool):
        object.__setattr__(self, "idx", idx)
        object.__setattr__(self, "log_update", log_update)
        object.__setattr__(self, "sign", sign)
        object.__setattr__(self, "in_sample", in_sample)
        object.__setattr__(self, "update", np.exp(log_update) * sign)


class MSRResultUpdater(ResultUpdater[MSRValueUpdate]):
    """Update running means for MSR valuation (in log-space).

    This class is used to update two running means for positive and negative updates
    separately. The two running means are later combined into a final result.

    Since values computed with MSR are not a mean over marginals, both the variances of
    the marginals and the update counts are ill-defined. We use the following
    conventions:

    1. The counts are defined as the minimum of the two counts. This definition enables
    us to ensure a minimal number of updates for both running means via stopping
    criteria and correctly detects that no actual update has taken place if one of the
    counts is zero.

    2. We reverse engineer the variances so that they yield correct standard errors
    given our convention for the counts and the normal calculation of standard errors in
    the valuation result.

    Note that we cannot use the normal addition or subtraction defined by the
    [ValuationResult][pydvl.valuation.result.ValuationResult] because it is weighted
    with counts. If we were to simply subtract the negative result from the positive we
    would get wrong variance estimates, misleading update counts and even wrong values
    if no further precaution is taken.
    """

    def __init__(self, result: ValuationResult):
        self.result = result
        self.in_sample = ValuationResult.zeros(
            algorithm=result.algorithm, indices=result.indices, data_names=result.names
        )
        self.out_of_sample = ValuationResult.zeros(
            algorithm=result.algorithm, indices=result.indices, data_names=result.names
        )

        self.update_in_sample = LogResultUpdater[MSRValueUpdate](self.in_sample)
        self.update_out_of_sample = LogResultUpdater[MSRValueUpdate](self.out_of_sample)

    def __call__(self, update: MSRValueUpdate) -> ValuationResult:
        assert update.idx is not None
        if update.in_sample:
            self.update_in_sample(update)
        else:
            self.update_out_of_sample(update)
        return self.combine_results()

    def combine_results(self) -> ValuationResult:
        """Combine the positive and negative running means into a final result.
        Returns:
            The combined valuation result.

        TODO: Verify that the two running means are statistically independent (which is
            assumed in the aggregation of variances).
        """
        # define counts as minimum of the two counts (see docstring)
        counts = np.minimum(self.in_sample.counts, self.out_of_sample.counts)

        values = self.in_sample.values - self.out_of_sample.values
        values[counts == 0] = np.nan

        # define variances that yield correct standard errors (see docstring)
        pos_var = self.in_sample.variances / np.clip(self.in_sample.counts, 1, np.inf)
        neg_var = self.out_of_sample.variances / np.clip(
            self.out_of_sample.counts, 1, np.inf
        )
        variances = np.where(counts != 0, (pos_var + neg_var) * counts, np.inf)

        self.result = ValuationResult(
            values=values,
            variances=variances,
            counts=counts,
            indices=self.result.indices,
            data_names=self.result.names,
            algorithm=self.result.algorithm,
        )

        return self.result


class MSRSampler(StochasticSamplerMixin, IndexSampler[MSRValueUpdate]):
    """Sampler for unweighted Maximum Sample Re-use (MSR) valuation.

    The sampling is similar to a
    [UniformSampler][pydvl.valuation.samplers.powerset.UniformSampler] but without an outer
    index. However,the MSR sampler uses a special evaluation strategy and result updater,
    as returned by the [make_strategy()][pydvl.valuation.samplers.msr.MSRSampler.make_strategy]
    and [result_updater()][pydvl.valuation.samplers.msr.MSRSampler.result_updater] methods,
    respectively.

    Two running means are updated separately for positive and negative updates. The two
    running means are later combined into a final result.

    Args:
        batch_size: Number of samples to generate in each batch.
        seed: Seed for the random number generator.

    """

    def __init__(self, batch_size: int = 1, seed: Seed | None = None):
        super().__init__(batch_size=batch_size, seed=seed)

    def generate(self, indices: IndexSetT) -> SampleGenerator:
        while True:
            subset = random_subset(indices, seed=self._rng)
            yield Sample(None, subset)

    def log_weight(self, n: int, subset_len: int) -> float:
        r"""Probability of sampling a set of size k.

        In the **MSR scheme**, the sampling is done from the full power set $2^N$ (each
        set $S \subseteq N$ with probability $1 / 2^n$), and then for each data point
        $i$ one partitions the sample into:

            * $\mathcal{S}_{\ni i} = \{S \in \mathcal{S}: i \in S\},$ and
            * $\mathcal{S}_{\nni i} = \{S \in \mathcal{S}: i \nin S\}.$.

        When we condition on the event $i \in S$, the remaining part $S_{- i}$ is
        uniformly distributed over $2^{N_{- i}}$. In other words, the act of
        partitioning recovers the uniform distribution on $2^{N_{- i}}$ "for free"
        because

        $$P (S_{- i} = T \mid i \in S) = \frac{1}{2^{n - 1}},$$

        for each $T \subseteq N_{- i}$.

        Args:
            n: Size of the index set.
            subset_len: Size of the subset.

        Returns:
            The logarithm of the probability of having sampled a set of size
                `subset_len`.
        """
        return float(-(n - 1) * np.log(2)) if n > 0 else 0.0

    def make_strategy(
        self,
        utility: UtilityBase,
        coefficient: Callable[[int, int], float] | None = None,
    ) -> MSREvaluationStrategy:
        """Returns the strategy for this sampler.

        Args:
            utility: Utility function to evaluate.
            coefficient: Coefficient function for the utility function.
        """
        assert coefficient is not None
        return MSREvaluationStrategy(self, utility, coefficient)

    def result_updater(self, result: ValuationResult) -> ResultUpdater:
        """Returns a callable that updates a valuation result with an MSR value update.

        MSR updates two running means for positive and negative updates separately. The
        two running means are later combined into a final result.

        Args:
            result: The valuation result to update with each call of the returned
                callable.
        Returns:
            A callable object that updates the valuation result with very
                [MSRValueUpdate][pydvl.valuation.samplers.msr.MSRValueUpdate].
        """
        return MSRResultUpdater(result)


class MSREvaluationStrategy(EvaluationStrategy[MSRSampler, MSRValueUpdate]):
    """Evaluation strategy for Maximum Sample Re-use (MSR) valuation in log space.

    The MSR evaluation strategy makes one utility evaluation per sample but generates
    `n_indices` many updates from it. The updates will be used to update two running
    means that will later be combined into a final value. We use the field
    `ValueUpdate.in_sample` field to inform
    [MSRResultUpdater][pydvl.valuation.samplers.msr.MSRResultUpdater] of which of the two
    running means must be updated.
    """

    @suppress_warnings(categories=(RuntimeWarning,), flag="show_warnings")
    def process(
        self, batch: SampleBatch, is_interrupted: NullaryPredicate
    ) -> List[MSRValueUpdate]:
        updates = []
        for sample in batch:
            updates.extend(self._process_sample(sample))
            if is_interrupted():
                break
        return updates

    def _process_sample(self, sample: Sample) -> List[MSRValueUpdate]:
        u = self.utility(sample)
        sign = np.sign(u)
        mask = np.zeros(self.n_indices, dtype=bool)
        mask[sample.subset] = True

        updates = []
        for i, in_sample in enumerate(mask):  # type: int, bool
            k = len(sample.subset) - int(in_sample)
            update = -np.inf if u == 0 else np.log(u * sign)
            update += self.log_correction(self.n_indices, k)
            updates.append(MSRValueUpdate(np.int_(i), update, sign, in_sample))
        return updates
