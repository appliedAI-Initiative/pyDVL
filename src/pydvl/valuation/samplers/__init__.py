r"""
Samplers iterate over subsets of indices.

The classes in this module are used to iterate over indices, and subsets of their
complement in the whole set, as required for the computation of marginal utilities
for semi-values and other marginal-utility based methods.

These samplers are used by all game-theoretic valuation methods, as well as for LOO and
any other marginal-contribution-based method which iterates over subsets of the training
data, and because of intertwining of these algorithms with the sampling, there are
several strategies to choose when constructing them.

## Index iteration

Subclasses of [IndexSampler][pydvl.valuation.samplers.IndexSampler] are iterators
over **batches** of [Samples][pydvl.valuation.types.Sample]. These are typically of
the form $(i, S)$, where $i$ is an index of interest, and $S \subset I \setminus \{i\}$
is a subset of the complement of $i.$

This type of iteration over indices $i$ and their complements is configured upon
construction of the sampler with the classes
[SequentialIndexIteration][pydvl.valuation.samplers.powerset.SequentialIndexIteration],
[RandomIndexIteration][pydvl.valuation.samplers.powerset.RandomIndexIteration], or their finite
counterparts, when each index must be visited just once (albeit possibly generating many
samples per index).

However, some valuation schemes require iteration over subsets of the whole set (as
opposed to iterating over complements of individual indices). For this purpose, one can
use [NoIndexIteration][pydvl.valuation.samplers.powerset.NoIndexIteration] or its finite
counterpart.

## Sampler evaluation

Different samplers imply different strategies for processing samples, i.e. for
evaluating the utility of the subsets. For instance permutation samplers generate
increasing subsets of permutations, allowing semi-value calculations to benefit an
incremental evaluation of the utility that reuses the previous computation.

This behaviour is communicated to the valuation method through the
[EvaluationStrategy][pydvl.valuation.samplers.base.EvaluationStrategy] class. The basic
usage pattern inside a valuation method is the following (see below for info on the
`updater`):

```python
    def fit(self, data: Dataset):

        ...

        strategy = self.sampler.make_strategy(self.utility, self.log_coefficient)
        processor = delayed(strategy.process)
        updater = self.sampler.result_updater(self.result)

        delayed_batches = Parallel()(
            processor(batch=list(batch), is_interrupted=flag) for batch in self.sampler
        )
        for batch in delayed_batches:
            for evaluation in batch:
                self.result = updater(evaluation)
            ...
```

## Updating the result

Yet another behaviour that depends on the sampling scheme is the way that results are
updated. For instance, the [MSRSampler][pydvl.valuation.samplers.msr.MSRSampler]
requires tracking updates to two sequences of samples which are then merged in a
specific way. This strategy is declared by the sampler through the factory method
[result_updater()][pydvl.valuation.samplers.base.IndexSampler.result_updater],
which returns a callable that updates the result with a single evaluation.


## Creating custom samplers

To create a custom sampler, subclass either
[PowersetSampler][pydvl.valuation.samplers.PowersetSampler]
or [PermutationSamplerBase][pydvl.valuation.samplers.permutation.PermutationSamplerBase],
or implement the [IndexSampler][pydvl.valuation.samplers.IndexSampler] interface
directly.

There are three main methods to implement (and others that can be overridden):

* [generate()][pydvl.valuation.samplers.base.IndexSampler.generate], which yields
  samples of the form $(i, S)$. These will be batched together by `__iter__` for
  parallel processing. Note that, if the index set has size $N$, for
  [PermutationSampler][pydvl.valuation.samplers.permutation.PermutationSampler], a
  batch size of $B$ implies $O(B*N)$ evaluations of the utility in one process, since
  single permutations are always processed in one go.
* [log_weight()][pydvl.valuation.samplers.base.IndexSampler.log_weight] to provide a
  factor by which to multiply Monte Carlo samples in stochastic methods, so that the
  mean converges to the desired expression. This will be the logarithm of the
  probability of sampling a given subset. For an explanation of the interactions between
  sampler weights, semi-value coefficients and importance sampling, see
  [Sampling strategies for semi-values][semi-values-sampling].

    ??? tip "Disabling importance sampling"
        If you want to disable importance sampling, you can override the property
        [log_coefficient()][pydvl.valuation.methods.semivalue.SemivalueValuation.log_coefficient]
        and return `None`. This will make the evaluation strategy ignore the sampler
        weights and the Monte Carlo sums converge to the expectation of the marginal
        utilities wrt. the sampling distribution, with no change.

* [make_strategy()][pydvl.valuation.samplers.base.IndexSampler.make_strategy] to create
  an evaluation strategy that processes the samples. This is typically a subclass of
  [EvaluationStrategy][pydvl.valuation.samplers.base.EvaluationStrategy] that computes
  utilities and weights them with coefficients and sampler weights.
  One can also use any of the predefined strategies, like the successive marginal
  evaluations of
  [PowersetEvaluationStrategy][pydvl.valuation.samplers.powerset.PowersetEvaluationStrategy]
  or the successive evaluations of
  [PermutationEvaluationStrategy][pydvl.valuation.samplers.permutation.PermutationEvaluationStrategy].

Finally, if the sampler requires a dedicated result updater, you must override
[result_updater()][pydvl.valuation.samplers.base.IndexSampler.result_updater] to return
a callable that updates a [ValuationResult][pydvl.valuation.result.ValuationResult] with
one evaluation [ValueUpdate][pydvl.valuation.types.ValueUpdate]. This is used e.g. for
the [MSRSampler][pydvl.valuation.samplers.msr.MSRSampler] which uses two running means
for positive and negative updates.

!!! tip "Changed in version 0.10.0"
    All the samplers in this module have been changed to work with the new
    evaluation strategies.

## References

[^1]: <a name="mitchell_sampling_2022"></a>Mitchell, Rory, Joshua Cooper, Eibe
      Frank, and Geoffrey Holmes. [Sampling Permutations for Shapley Value
      Estimation](https://jmlr.org/papers/v23/21-0439.html). Journal of Machine
      Learning Research 23, no. 43 (2022): 1–46.
[^2]: <a name="watson_accelerated_2023"></a>Watson, Lauren, Zeno Kujawa, Rayna Andreeva,
      Hao-Tsung Yang, Tariq Elahi, and Rik Sarkar. [Accelerated Shapley Value
      Approximation for Data Evaluation](https://doi.org/10.48550/arXiv.2311.05346).
      arXiv, 9 November 2023.
"""

from typing import Union

from .base import *
from .classwise import *
from .msr import *
from .owen import *
from .permutation import *
from .powerset import *
from .stratified import *
from .truncation import *

# TODO Replace by Intersection[StochasticSamplerMixin, PowersetSampler[T]]
# See https://github.com/python/typing/issues/213
StochasticSampler = Union[
    UniformSampler,
    PermutationSampler,
    AntitheticSampler,
    StratifiedSampler,
    OwenSampler,
    AntitheticOwenSampler,
]
