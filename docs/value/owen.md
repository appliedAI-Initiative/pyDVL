---
title: Owen Shapley values
alias: 
  name: owen-shapley
---

# Owen sampling for Shapley values

Owen sampling schemes are a family of sampling schemes that are used to estimate
Shapley values. They are based on a multilinear extension technique from game theory,
and were introduced in [@okhrati_multilinear_2021]. The core idea is to use different
probabilities of including indices into samples.

By choosing these probabilities at uniform, the expected value of the marginal utility
over the sampling probability of $p$ is equal to the Shapley value:

$$v_{sh}(i) = \int_0^1 \mathbb{E}_{D^p_{-i}} \left[ u_(S_{+i}) - u(S) \right] dp,$$

where $D^p_{-i}$ is the distribution over the subsets of the training set, not
containing $i$, whose elements are included with probability $p$.

There is an outer loop that picks sampling probabilities between 0 and 1, and an inner
loop that samples indices from the dataset using that probability. Depending on the
properties of each choice the samplers can be finite or infinite. The original method
introduced in the paper sampled a fixed number of values for $p \in (0,1)$ and for each
one of those sampled just a few (2) sets of indices, where the probability of including
an index is $p$.

In order to compute values it is enough to use any of the Owen samplers together with a
[ShapleyValuation][pydvl.valuation.methods.ShapleyValuation] object.

## Finite Owen Sampler

[OwenSampler][pydvl.valuation.samplers.owen.OwenSampler] with a
[FiniteSequentialIndexIteration][pydvl.valuation.samplers.powerset.FiniteSequentialIndexIteration]
for the outer loop and a
[GridOwenStrategy][pydvl.valuation.samplers.owen.GridOwenStrategy] for the sampling
probabilities is the most basic Owen sampler. It uses a deterministic grid of
probability values between 0 and 1 for the inner sampling. It follows the idea of
the original paper and should be instantiated with
[NoStopping][pydvl.valuation.stopping.NoStopping] as stopping criterion.

??? Example
    ```python
    from pydvl.valuation.methods import ShapleyValuation
    from pydvl.valuation.samplers import OwenSampler
    from pydvl.valuation.stopping import RankCorrelation

    ...

    sampler = OwenSampler(
        outer_sampling_strategy=GridOwenStrategy(n_samples_outer=200),
        n_samples_inner=2,
        index_iteration=FiniteSequentialIndexIteration,
    )
    stopping = NoStopping(sampler)  # Pass the sampler for progress updates
    valuation = ShapleyValuation(utility, sampler, stopping, progress=True)
    with parallel_config(n_jobs=8):
        valuation.fit(train)
    result = valuation.values()
    ```

## Infinite (uniform) Owen Sampler

[OwenSampler][pydvl.valuation.samplers.owen.OwenSampler] follows the same principle
as [OwenSampler][pydvl.valuation.samplers.owen.OwenSampler], but samples
probability values between 0 and 1 at random indefinitely. It requires a stopping
criterion to be used with the valuation method, and thus follows more closely the
general pattern of the valuation methods. This makes it more adequate for actual use
since it is no longer required to estimate a number of outer samples required. Because
we sample uniformly the statistical properties of the method are conserved, in
particular the [correction coefficient][pydvl.valuation.samplers.owen.OwenSampler.log_weight]
for semi-value computation remains the same.

??? Example "Owen Sampler"
    ```python
    from pydvl.valuation.methods import ShapleyValuation
    from pydvl.valuation.samplers import OwenSampler, GridOwenStrategy
    from pydvl.valuation.stopping import RankCorrelation

    utility = ModelUtility(...)
    sampler = OwenSampler(outer_sampling_strategy=GridOwenStrategy(n_samples_outer=200))
    stopping = RankCorrelation(rtol=1e-3, burn_int=100)
    valuation = ShapleyValuation(utility, sampler, stopping,  progress=True)
    with parallel_config(n_jobs=-1)
        valuation.fit(dataset)
    result = valuation.values()
    ```

## Antithetic Owen Sampler

[AntitheticOwenSampler][pydvl.valuation.samplers.owen.AntitheticOwenSampler] is a
variant of the [OwenSampler][pydvl.valuation.samplers.owen.OwenSampler] that draws
probability values $q$ between 0 and 0.5 at random and then generates two samples
for each index, one using the probability $q$ for index draws, and another with
probability $1-q$.

!!! Example
    ```python
    from pydvl.valuation import AntitheticOwenSampler, ShapleyValuation, RankCorrelation
    ...

    sampler = AntitheticOwenSampler()
    valuation = ShapleyValuation(utility, sampler, RankCorrelation(rtol=1e-3))
    valuation.fit(dataset)
    ```
